"""Compact AnnData perturbation-score implementation with exact and approximate modes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from .parallel import normalize_n_jobs, run_parallel_tasks
from .stats import (
    extract_anndata_matrix,
    get_obs_column,
    resolve_perturbations,
    validate_fidelity,
    validate_layer,
    validate_perturbations,
    welch_t_scores,
)


RESULT_COLUMNS = [
    "row_id",
    "perturbation_label",
    "target_perturbation",
    "ps_score",
    "fidelity",
    "method",
    "selected_target_gene_count",
    "score_status",
    "scale_factor",
    "lambda_",
]


def run_ps_score_anndata(
    adata: Any,
    *,
    layer: str | None = None,
    perturbation_key: str,
    control_label: str,
    fidelity: str = "exact",
    perturbations: Sequence[str] | None = None,
    n_jobs: int | None = 1,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    target_gene_min: int = 1,
    target_gene_max: int = 50,
    scale_factor: float = 3.0,
    lambda_: float = 0.0,
    use_perturb_expression: bool = True,
    scale_score: bool = True,
) -> pd.DataFrame:
    """Score AnnData perturbations with a compact PS-score workflow."""

    if adata is None:
        raise ValueError("adata must not be None")
    if not isinstance(perturbation_key, str) or not perturbation_key:
        raise ValueError("perturbation_key must be a non-empty string")
    if not isinstance(control_label, str) or not control_label:
        raise ValueError("control_label must be a non-empty string")
    if not isinstance(use_perturb_expression, bool):
        raise TypeError("use_perturb_expression must be a boolean")
    if not isinstance(scale_score, bool):
        raise TypeError("scale_score must be a boolean")

    validate_layer(layer)
    fidelity = validate_fidelity(fidelity)
    validate_perturbations(perturbations)
    normalize_n_jobs(n_jobs)
    _validate_target_genes(target_genes)
    _validate_positive_int("target_gene_min", target_gene_min)
    _validate_positive_int("target_gene_max", target_gene_max)
    if target_gene_max < target_gene_min:
        raise ValueError("target_gene_max must be greater than or equal to target_gene_min")
    scale_factor = _validate_positive_float("scale_factor", scale_factor)
    lambda_ = _validate_non_negative_float("lambda_", lambda_)

    if not hasattr(adata, "obs") or not hasattr(adata, "obs_names") or not hasattr(adata, "var_names"):
        raise TypeError("adata must provide obs, obs_names, and var_names")

    labels = np.asarray(get_obs_column(adata.obs, perturbation_key), dtype=object)
    if labels.size == 0:
        raise ValueError("adata must contain at least one observation")
    if not np.any(labels == control_label):
        raise ValueError(f"control_label {control_label!r} was not found in adata.obs[{perturbation_key!r}]")

    selected = resolve_perturbations(
        labels,
        control_label=control_label,
        perturbations=perturbations,
    )
    if not selected:
        return _empty_result_frame()

    matrix = extract_anndata_matrix(adata, layer=layer)
    row_ids = np.asarray(adata.obs_names, dtype=object)
    var_names = np.asarray(adata.var_names, dtype=object)
    gene_lookup = {str(name): index for index, name in enumerate(var_names)}

    worker = (
        lambda perturbation: _run_exact_for_perturbation(
            perturbation=perturbation,
            labels=labels,
            row_ids=row_ids,
            var_names=var_names,
            gene_lookup=gene_lookup,
            matrix=matrix,
            control_label=control_label,
            target_genes=target_genes,
            target_gene_min=target_gene_min,
            target_gene_max=target_gene_max,
            scale_factor=scale_factor,
            lambda_=lambda_,
            use_perturb_expression=use_perturb_expression,
            scale_score=scale_score,
        )
        if fidelity == "exact"
        else _run_approx_for_perturbation(
            perturbation=perturbation,
            labels=labels,
            row_ids=row_ids,
            var_names=var_names,
            gene_lookup=gene_lookup,
            matrix=matrix,
            control_label=control_label,
            target_genes=target_genes,
            target_gene_min=target_gene_min,
            target_gene_max=target_gene_max,
            scale_factor=scale_factor,
            lambda_=lambda_,
            use_perturb_expression=use_perturb_expression,
            scale_score=scale_score,
        )
    )

    outputs = run_parallel_tasks(selected, worker, n_jobs=n_jobs)
    frames = [frame for frame, _ in outputs]
    metadata = {perturbation: meta for (_, meta), perturbation in zip(outputs, selected, strict=False)}
    result = pd.concat(frames, ignore_index=True) if frames else _empty_result_frame()
    result.attrs["ps_score"] = {
        "algorithm": "ps_score",
        "fidelity": fidelity,
        "layer": layer,
        "perturbation_key": perturbation_key,
        "control_label": control_label,
        "perturbations": list(selected),
        "metadata_by_perturbation": metadata,
    }
    return result


def run_ps_score_stream(
    batches: Any,
    *,
    obs: Any,
    var_names: Sequence[str],
    perturbation_key: str,
    control_label: str,
    fidelity: str = "exact",
    perturbations: Sequence[str] | None = None,
    n_jobs: int | None = 1,
) -> Any:
    """Validate the streamed perturbation score API contract for later phases."""

    if batches is None:
        raise ValueError("batches must not be None")
    if obs is None:
        raise ValueError("obs must not be None")
    if not isinstance(var_names, Sequence) or isinstance(var_names, str):
        raise TypeError("var_names must be a sequence of feature names")
    if not isinstance(perturbation_key, str) or not perturbation_key:
        raise ValueError("perturbation_key must be a non-empty string")
    if not isinstance(control_label, str) or not control_label:
        raise ValueError("control_label must be a non-empty string")
    validate_fidelity(fidelity)
    validate_perturbations(perturbations)
    normalize_n_jobs(n_jobs)
    raise NotImplementedError(
        "PS score streamed execution is not implemented yet; Phase 4 only adds the AnnData path."
    )


def _run_exact_for_perturbation(
    *,
    perturbation: Any,
    labels: np.ndarray,
    row_ids: np.ndarray,
    var_names: np.ndarray,
    gene_lookup: Mapping[str, int],
    matrix: Any,
    control_label: str,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    target_gene_min: int,
    target_gene_max: int,
    scale_factor: float,
    lambda_: float,
    use_perturb_expression: bool,
    scale_score: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    control_idx = np.flatnonzero(labels == control_label)
    target_idx = np.flatnonzero(labels == perturbation)
    _validate_cell_counts(control_idx, target_idx, perturbation=perturbation, control_label=control_label)

    gene_indices, gene_names, gene_source = _resolve_target_gene_indices(
        perturbation=perturbation,
        target_genes=target_genes,
        gene_lookup=gene_lookup,
        var_names=var_names,
        matrix=matrix,
        control_idx=control_idx,
        target_idx=target_idx,
        target_gene_min=target_gene_min,
        target_gene_max=target_gene_max,
        use_perturb_expression=use_perturb_expression,
    )

    combined_idx = np.flatnonzero((labels == control_label) | (labels == perturbation))
    combined_expr = _select_dense(matrix, combined_idx, gene_indices)
    control_expr = _select_dense(matrix, control_idx, gene_indices)
    target_expr = _select_dense(matrix, target_idx, gene_indices)
    beta, control_mean = _estimate_signature(control_expr, target_expr, perturbation=perturbation)
    centered_combined = combined_expr - control_mean
    centered_target = target_expr - control_mean

    target_scores, optimize_result = _optimize_target_scores(
        centered_target,
        beta,
        scale_factor=scale_factor,
        lambda_=lambda_,
    )

    target_pos = np.flatnonzero(labels[combined_idx] == perturbation)
    full_scores = np.zeros(combined_idx.shape[0], dtype=float)
    full_scores[target_pos] = target_scores
    full_scores, scaling_value, scaling_applied = _maybe_scale_scores(
        full_scores,
        target_pos=target_pos,
        scale_score=scale_score,
    )

    score_status = np.full(combined_idx.shape[0], "fixed-control", dtype=object)
    score_status[target_pos] = "optimized"
    frame = _build_result_frame(
        row_ids=row_ids[combined_idx],
        perturbation_labels=labels[combined_idx],
        target_perturbation=str(perturbation),
        scores=full_scores,
        fidelity="exact",
        method="bounded_least_squares",
        selected_target_gene_count=gene_indices.size,
        score_status=score_status,
        scale_factor=scale_factor,
        lambda_=lambda_,
    )
    metadata = {
        "target_genes": gene_names,
        "target_gene_source": gene_source,
        "method": "bounded_least_squares",
        "optimizer_success": bool(optimize_result.success),
        "optimizer_status": int(optimize_result.status),
        "optimizer_message": str(optimize_result.message),
        "optimizer_iterations": int(getattr(optimize_result, "nit", 0)),
        "signature_norm": float(np.linalg.norm(beta)),
        "scale_score": scale_score,
        "scale_score_applied": scaling_applied,
        "scale_score_max_before_scaling": scaling_value,
    }
    return frame, metadata


def _run_approx_for_perturbation(
    *,
    perturbation: Any,
    labels: np.ndarray,
    row_ids: np.ndarray,
    var_names: np.ndarray,
    gene_lookup: Mapping[str, int],
    matrix: Any,
    control_label: str,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    target_gene_min: int,
    target_gene_max: int,
    scale_factor: float,
    lambda_: float,
    use_perturb_expression: bool,
    scale_score: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    control_idx = np.flatnonzero(labels == control_label)
    target_idx = np.flatnonzero(labels == perturbation)
    _validate_cell_counts(control_idx, target_idx, perturbation=perturbation, control_label=control_label)

    gene_indices, gene_names, gene_source = _resolve_target_gene_indices(
        perturbation=perturbation,
        target_genes=target_genes,
        gene_lookup=gene_lookup,
        var_names=var_names,
        matrix=matrix,
        control_idx=control_idx,
        target_idx=target_idx,
        target_gene_min=target_gene_min,
        target_gene_max=target_gene_max,
        use_perturb_expression=use_perturb_expression,
    )

    combined_idx = np.flatnonzero((labels == control_label) | (labels == perturbation))
    combined_expr = _select_dense(matrix, combined_idx, gene_indices)
    control_expr = _select_dense(matrix, control_idx, gene_indices)
    target_expr = _select_dense(matrix, target_idx, gene_indices)
    beta, control_mean = _estimate_signature(control_expr, target_expr, perturbation=perturbation)
    centered_combined = combined_expr - control_mean

    scores = _analytic_projection_scores(
        centered_combined,
        beta,
        scale_factor=scale_factor,
        lambda_=lambda_,
    )
    target_pos = np.flatnonzero(labels[combined_idx] == perturbation)
    scores, scaling_value, scaling_applied = _maybe_scale_scores(
        scores,
        target_pos=target_pos,
        scale_score=scale_score,
    )

    frame = _build_result_frame(
        row_ids=row_ids[combined_idx],
        perturbation_labels=labels[combined_idx],
        target_perturbation=str(perturbation),
        scores=scores,
        fidelity="approx",
        method="analytic_projection",
        selected_target_gene_count=gene_indices.size,
        score_status=np.full(combined_idx.shape[0], "projected", dtype=object),
        scale_factor=scale_factor,
        lambda_=lambda_,
    )
    metadata = {
        "target_genes": gene_names,
        "target_gene_source": gene_source,
        "method": "analytic_projection",
        "signature_norm": float(np.linalg.norm(beta)),
        "scale_score": scale_score,
        "scale_score_applied": scaling_applied,
        "scale_score_max_before_scaling": scaling_value,
    }
    return frame, metadata


def _resolve_target_gene_indices(
    *,
    perturbation: Any,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    gene_lookup: Mapping[str, int],
    var_names: np.ndarray,
    matrix: Any,
    control_idx: np.ndarray,
    target_idx: np.ndarray,
    target_gene_min: int,
    target_gene_max: int,
    use_perturb_expression: bool,
) -> tuple[np.ndarray, list[str], str]:
    explicit = _get_explicit_target_genes(target_genes, perturbation=perturbation)
    if explicit is not None:
        gene_names = _normalize_gene_names(explicit)
        source = "provided"
        if not use_perturb_expression:
            gene_names = [gene for gene in gene_names if gene != str(perturbation)]
        if len(gene_names) > target_gene_max:
            gene_names = gene_names[:target_gene_max]
        if len(gene_names) < target_gene_min:
            raise ValueError(
                f"PS score needs at least {target_gene_min} target genes for perturbation {perturbation!r}"
            )
        missing = [gene for gene in gene_names if gene not in gene_lookup]
        if missing:
            raise ValueError(
                "Unknown target genes requested for perturbation "
                f"{perturbation!r}: {', '.join(sorted(missing))}"
            )
        indices = np.asarray([gene_lookup[gene] for gene in gene_names], dtype=np.int64)
        return indices, gene_names, source

    case_expr = _select_dense(matrix, target_idx, None)
    control_expr = _select_dense(matrix, control_idx, None)
    scores = welch_t_scores(case_expr, control_expr)
    mean_diff = case_expr.mean(axis=0) - control_expr.mean(axis=0)
    ranking = np.argsort(-np.abs(scores), kind="stable")
    informative = ranking[np.abs(mean_diff[ranking]) > 0]
    if informative.size == 0:
        informative = ranking

    if not use_perturb_expression:
        informative = informative[var_names[informative] != str(perturbation)]
    selected = informative[: min(target_gene_max, informative.size)]
    if selected.size < target_gene_min:
        raise ValueError(
            f"PS score found fewer than {target_gene_min} target genes for perturbation {perturbation!r}"
        )
    return selected.astype(np.int64, copy=False), var_names[selected].astype(str).tolist(), "de"


def _estimate_signature(
    control_expr: np.ndarray,
    target_expr: np.ndarray,
    *,
    perturbation: Any,
) -> tuple[np.ndarray, np.ndarray]:
    control_mean = control_expr.mean(axis=0)
    target_mean = target_expr.mean(axis=0)
    beta = target_mean - control_mean
    if not np.any(np.abs(beta) > 0):
        raise ValueError(f"PS score signature is zero for perturbation {perturbation!r}")
    return beta, control_mean


def _optimize_target_scores(
    centered_target_expr: np.ndarray,
    beta: np.ndarray,
    *,
    scale_factor: float,
    lambda_: float,
) -> tuple[np.ndarray, Any]:
    beta_norm_sq = float(np.dot(beta, beta))
    if beta_norm_sq <= 0:
        raise ValueError("PS score signature norm must be positive")

    initial = _analytic_projection_scores(
        centered_target_expr,
        beta,
        scale_factor=scale_factor,
        lambda_=lambda_,
    )

    def objective(scores: np.ndarray) -> float:
        residual = scores[:, None] * beta[None, :] - centered_target_expr
        return 0.5 * float(np.square(residual).sum()) + (lambda_ * float(scores.sum()))

    def gradient(scores: np.ndarray) -> np.ndarray:
        residual = scores[:, None] * beta[None, :] - centered_target_expr
        return residual @ beta + lambda_

    bounds = [(0.0, scale_factor)] * centered_target_expr.shape[0]
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds,
    )
    if not result.success:
        raise RuntimeError(f"PS score optimization failed: {result.message}")
    return np.asarray(result.x, dtype=float), result


def _analytic_projection_scores(
    centered_expr: np.ndarray,
    beta: np.ndarray,
    *,
    scale_factor: float,
    lambda_: float,
) -> np.ndarray:
    beta_norm_sq = float(np.dot(beta, beta))
    if beta_norm_sq <= 0:
        raise ValueError("PS score signature norm must be positive")
    raw = (centered_expr @ beta - lambda_) / beta_norm_sq
    return np.clip(raw, 0.0, scale_factor)


def _maybe_scale_scores(
    scores: np.ndarray,
    *,
    target_pos: np.ndarray,
    scale_score: bool,
) -> tuple[np.ndarray, float | None, bool]:
    if not scale_score:
        return scores, None, False
    if target_pos.size == 0:
        return scores, None, False
    max_target = float(np.max(scores[target_pos]))
    if max_target <= 0:
        return scores, max_target, False
    return scores / max_target, max_target, True


def _build_result_frame(
    *,
    row_ids: np.ndarray,
    perturbation_labels: np.ndarray,
    target_perturbation: str,
    scores: np.ndarray,
    fidelity: str,
    method: str,
    selected_target_gene_count: int,
    score_status: np.ndarray,
    scale_factor: float,
    lambda_: float,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_id": row_ids,
            "perturbation_label": perturbation_labels,
            "target_perturbation": target_perturbation,
            "ps_score": scores,
            "fidelity": fidelity,
            "method": method,
            "selected_target_gene_count": int(selected_target_gene_count),
            "score_status": score_status,
            "scale_factor": float(scale_factor),
            "lambda_": float(lambda_),
        }
    )
    return frame.loc[:, RESULT_COLUMNS]


def _select_dense(
    matrix: Any,
    row_indices: np.ndarray,
    col_indices: np.ndarray | None,
) -> np.ndarray:
    if sparse.issparse(matrix):
        subset = matrix[row_indices] if col_indices is None else matrix[row_indices][:, col_indices]
        return np.asarray(subset.toarray(), dtype=float)

    dense = np.asarray(matrix, dtype=float)
    if dense.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if col_indices is None:
        return np.asarray(dense[row_indices], dtype=float)
    return np.asarray(dense[np.ix_(row_indices, col_indices)], dtype=float)


def _get_explicit_target_genes(
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    *,
    perturbation: Any,
) -> Sequence[str] | None:
    if target_genes is None:
        return None
    if isinstance(target_genes, Mapping):
        if perturbation in target_genes:
            return target_genes[perturbation]
        return target_genes.get(str(perturbation))
    return target_genes


def _normalize_gene_names(genes: Sequence[str]) -> list[str]:
    if isinstance(genes, str):
        raise TypeError("target_genes must be a sequence of gene names, not a string")
    normalized: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        if not isinstance(gene, str) or not gene:
            raise TypeError("target_genes entries must be non-empty strings")
        if gene in seen:
            continue
        normalized.append(gene)
        seen.add(gene)
    return normalized


def _validate_target_genes(
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
) -> None:
    if target_genes is None:
        return
    if isinstance(target_genes, str):
        raise TypeError("target_genes must be a mapping or sequence of gene names, not a string")
    if isinstance(target_genes, Mapping):
        for key, value in target_genes.items():
            if not isinstance(key, str) or not key:
                raise TypeError("target_genes mapping keys must be non-empty strings")
            _normalize_gene_names(value)
        return
    _normalize_gene_names(target_genes)


def _validate_cell_counts(
    control_idx: np.ndarray,
    target_idx: np.ndarray,
    *,
    perturbation: Any,
    control_label: str,
) -> None:
    if control_idx.size < 2:
        raise ValueError(f"PS score requires at least 2 {control_label!r} control cells")
    if target_idx.size < 2:
        raise ValueError(f"PS score requires at least 2 cells for perturbation {perturbation!r}")


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_positive_float(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _validate_non_negative_float(name: str, value: float) -> float:
    if not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


def _empty_result_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)
