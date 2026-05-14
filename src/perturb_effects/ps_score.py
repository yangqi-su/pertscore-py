"""Compact perturbation-score implementation with exact and approximate modes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from .parallel import normalize_n_jobs, run_parallel_tasks
from .stats import (
    csr_batch_to_matrix,
    extract_anndata_matrix,
    get_obs_column,
    get_obs_row_ids,
    iter_csr_batches,
    require_reiterable_batches,
    resolve_perturbations,
    validate_fidelity,
    validate_layer,
    validate_perturbations,
    welch_t_scores,
    welch_t_scores_from_stats,
)
from .types import StreamFeatureStats


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


@dataclass(frozen=True)
class _StreamSignature:
    gene_indices: np.ndarray
    gene_names: list[str]
    beta: np.ndarray
    control_mean: np.ndarray
    source: str


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
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    target_gene_min: int = 1,
    target_gene_max: int = 50,
    scale_factor: float = 3.0,
    lambda_: float = 0.0,
    use_perturb_expression: bool = True,
    scale_score: bool = True,
    target_signatures: Mapping[str, Sequence[float]] | Sequence[float] | None = None,
    control_means: Mapping[str, Sequence[float]] | Sequence[float] | None = None,
) -> pd.DataFrame:
    """Score streamed CSR batches with exact or approximate PS-score semantics."""

    if batches is None:
        raise ValueError("batches must not be None")
    if obs is None:
        raise ValueError("obs must not be None")
    if isinstance(var_names, str):
        raise TypeError("var_names must be a sequence of feature names")
    if not isinstance(perturbation_key, str) or not perturbation_key:
        raise ValueError("perturbation_key must be a non-empty string")
    if not isinstance(control_label, str) or not control_label:
        raise ValueError("control_label must be a non-empty string")
    if not isinstance(use_perturb_expression, bool):
        raise TypeError("use_perturb_expression must be a boolean")
    if not isinstance(scale_score, bool):
        raise TypeError("scale_score must be a boolean")

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
    _validate_optional_numeric_vectors("target_signatures", target_signatures)
    _validate_optional_numeric_vectors("control_means", control_means)

    labels = np.asarray(get_obs_column(obs, perturbation_key), dtype=object)
    row_ids = np.asarray(get_obs_row_ids(obs), dtype=object)
    if labels.size == 0:
        raise ValueError("obs must contain at least one observation")
    if row_ids.shape[0] != labels.shape[0]:
        raise ValueError("obs row identifiers and perturbation labels must have the same length")
    if not np.any(labels == control_label):
        raise ValueError(f"control_label {control_label!r} was not found in obs[{perturbation_key!r}]")

    selected = resolve_perturbations(
        labels,
        control_label=control_label,
        perturbations=perturbations,
    )
    if not selected:
        return _annotate_stream_result(
            _empty_result_frame(),
            fidelity=fidelity,
            perturbation_key=perturbation_key,
            control_label=control_label,
            perturbations=[],
            metadata={},
            stream_mode="none",
        )

    var_names_array = np.asarray(var_names, dtype=object)
    if var_names_array.ndim != 1 or var_names_array.size == 0:
        raise ValueError("var_names must be a non-empty one-dimensional sequence")
    gene_lookup = {str(name): index for index, name in enumerate(var_names_array)}
    label_by_row_id = _build_label_lookup(row_ids=row_ids, labels=labels)

    if fidelity == "exact":
        batch_factory = require_reiterable_batches(
            batches,
            operation="run_ps_score_stream exact mode",
        )
        worker = lambda perturbation: _run_stream_exact_for_perturbation(
            batches=batch_factory,
            perturbation=perturbation,
            label_by_row_id=label_by_row_id,
            var_names=var_names_array,
            gene_lookup=gene_lookup,
            control_label=control_label,
            target_genes=target_genes,
            target_gene_min=target_gene_min,
            target_gene_max=target_gene_max,
            scale_factor=scale_factor,
            lambda_=lambda_,
            use_perturb_expression=use_perturb_expression,
            scale_score=scale_score,
            target_signatures=target_signatures,
            control_means=control_means,
        )
        outputs = run_parallel_tasks(selected, worker, n_jobs=n_jobs)
        stream_mode = "multi-pass"
    elif callable(batches):
        worker = lambda perturbation: _run_stream_approx_for_perturbation(
            batches=batches,
            perturbation=perturbation,
            label_by_row_id=label_by_row_id,
            var_names=var_names_array,
            gene_lookup=gene_lookup,
            control_label=control_label,
            target_genes=target_genes,
            target_gene_min=target_gene_min,
            target_gene_max=target_gene_max,
            scale_factor=scale_factor,
            lambda_=lambda_,
            use_perturb_expression=use_perturb_expression,
            scale_score=scale_score,
            target_signatures=target_signatures,
            control_means=control_means,
        )
        outputs = run_parallel_tasks(selected, worker, n_jobs=n_jobs)
        stream_mode = "multi-pass"
    else:
        outputs = _run_stream_approx_one_shot(
            batches=batches,
            selected=selected,
            label_by_row_id=label_by_row_id,
            var_names=var_names_array,
            gene_lookup=gene_lookup,
            control_label=control_label,
            target_genes=target_genes,
            target_gene_min=target_gene_min,
            target_gene_max=target_gene_max,
            scale_factor=scale_factor,
            lambda_=lambda_,
            use_perturb_expression=use_perturb_expression,
            scale_score=scale_score,
            target_signatures=target_signatures,
            control_means=control_means,
        )
        stream_mode = "one-pass-precomputed"

    frames = [frame for frame, _ in outputs]
    metadata = {perturbation: meta for (_, meta), perturbation in zip(outputs, selected, strict=False)}
    result = pd.concat(frames, ignore_index=True) if frames else _empty_result_frame()
    return _annotate_stream_result(
        result,
        fidelity=fidelity,
        perturbation_key=perturbation_key,
        control_label=control_label,
        perturbations=list(selected),
        metadata=metadata,
        stream_mode=stream_mode,
    )


def _run_stream_exact_for_perturbation(
    *,
    batches: Any,
    perturbation: Any,
    label_by_row_id: dict[Any, Any],
    var_names: np.ndarray,
    gene_lookup: Mapping[str, int],
    control_label: str,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    target_gene_min: int,
    target_gene_max: int,
    scale_factor: float,
    lambda_: float,
    use_perturb_expression: bool,
    scale_score: bool,
    target_signatures: Mapping[str, Sequence[float]] | Sequence[float] | None,
    control_means: Mapping[str, Sequence[float]] | Sequence[float] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    signature = _resolve_stream_signature(
        batches=batches,
        perturbation=perturbation,
        label_by_row_id=label_by_row_id,
        var_names=var_names,
        gene_lookup=gene_lookup,
        control_label=control_label,
        target_genes=target_genes,
        target_gene_min=target_gene_min,
        target_gene_max=target_gene_max,
        use_perturb_expression=use_perturb_expression,
        target_signatures=target_signatures,
        control_means=control_means,
        require_precomputed=False,
        precomputed_context="run_ps_score_stream",
    )

    frames: list[pd.DataFrame] = []
    raw_target_max: float | None = None
    total_iterations = 0
    optimizer_batches = 0
    optimized_cells = 0
    control_count = 0
    target_count = 0

    for batch in iter_csr_batches(batches):
        _validate_stream_feature_count(batch.shape[1], var_names.size)
        matrix = csr_batch_to_matrix(batch)
        labels = _batch_labels(batch.row_ids, label_by_row_id)
        row_ids = np.asarray(batch.row_ids, dtype=object)
        combined_mask = (labels == control_label) | (labels == perturbation)
        if not combined_mask.any():
            continue

        combined_labels = labels[combined_mask]
        combined_row_ids = row_ids[combined_mask]
        control_count += int(np.sum(combined_labels == control_label))
        target_count += int(np.sum(combined_labels == perturbation))
        selected_expr = np.asarray(
            matrix[combined_mask][:, signature.gene_indices].toarray(),
            dtype=float,
        )
        centered_expr = selected_expr - signature.control_mean
        target_mask = combined_labels == perturbation

        scores = np.zeros(combined_labels.shape[0], dtype=float)
        score_status = np.full(combined_labels.shape[0], "fixed-control", dtype=object)
        if target_mask.any():
            target_scores, optimize_result = _optimize_target_scores(
                centered_expr[target_mask],
                signature.beta,
                scale_factor=scale_factor,
                lambda_=lambda_,
            )
            scores[target_mask] = target_scores
            score_status[target_mask] = "optimized"
            optimizer_batches += 1
            optimized_cells += int(target_mask.sum())
            total_iterations += int(getattr(optimize_result, "nit", 0))
            batch_target_max = float(np.max(target_scores))
            raw_target_max = batch_target_max if raw_target_max is None else max(raw_target_max, batch_target_max)

        frames.append(
            _build_result_frame(
                row_ids=combined_row_ids,
                perturbation_labels=combined_labels,
                target_perturbation=str(perturbation),
                scores=scores,
                fidelity="exact",
                method="bounded_least_squares",
                selected_target_gene_count=signature.gene_indices.size,
                score_status=score_status,
                scale_factor=scale_factor,
                lambda_=lambda_,
            )
        )

    _validate_stream_cell_counts(
        control_count=control_count,
        target_count=target_count,
        perturbation=perturbation,
        control_label=control_label,
    )
    _apply_scaled_stream_scores(
        frames,
        scale_score=scale_score,
        raw_target_max=raw_target_max,
    )

    metadata = {
        "target_genes": signature.gene_names,
        "target_gene_source": signature.source,
        "method": "bounded_least_squares",
        "optimizer_success": True,
        "optimizer_status": 0,
        "optimizer_message": "success",
        "optimizer_iterations": int(total_iterations),
        "optimizer_batches": int(optimizer_batches),
        "optimized_cell_count": int(optimized_cells),
        "signature_norm": float(np.linalg.norm(signature.beta)),
        "scale_score": scale_score,
        "scale_score_applied": bool(scale_score and raw_target_max is not None and raw_target_max > 0),
        "scale_score_max_before_scaling": raw_target_max,
        "stream_semantics": "multi-pass",
    }
    frame = pd.concat(frames, ignore_index=True) if frames else _empty_result_frame()
    return frame, metadata


def _run_stream_approx_for_perturbation(
    *,
    batches: Any,
    perturbation: Any,
    label_by_row_id: dict[Any, Any],
    var_names: np.ndarray,
    gene_lookup: Mapping[str, int],
    control_label: str,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    target_gene_min: int,
    target_gene_max: int,
    scale_factor: float,
    lambda_: float,
    use_perturb_expression: bool,
    scale_score: bool,
    target_signatures: Mapping[str, Sequence[float]] | Sequence[float] | None,
    control_means: Mapping[str, Sequence[float]] | Sequence[float] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    signature = _resolve_stream_signature(
        batches=batches,
        perturbation=perturbation,
        label_by_row_id=label_by_row_id,
        var_names=var_names,
        gene_lookup=gene_lookup,
        control_label=control_label,
        target_genes=target_genes,
        target_gene_min=target_gene_min,
        target_gene_max=target_gene_max,
        use_perturb_expression=use_perturb_expression,
        target_signatures=target_signatures,
        control_means=control_means,
        require_precomputed=False,
        precomputed_context="run_ps_score_stream",
    )

    frames: list[pd.DataFrame] = []
    raw_target_max: float | None = None
    control_count = 0
    target_count = 0

    for batch in iter_csr_batches(batches):
        _validate_stream_feature_count(batch.shape[1], var_names.size)
        matrix = csr_batch_to_matrix(batch)
        labels = _batch_labels(batch.row_ids, label_by_row_id)
        row_ids = np.asarray(batch.row_ids, dtype=object)
        combined_mask = (labels == control_label) | (labels == perturbation)
        if not combined_mask.any():
            continue

        combined_labels = labels[combined_mask]
        combined_row_ids = row_ids[combined_mask]
        control_count += int(np.sum(combined_labels == control_label))
        target_count += int(np.sum(combined_labels == perturbation))
        selected_expr = np.asarray(
            matrix[combined_mask][:, signature.gene_indices].toarray(),
            dtype=float,
        )
        centered_expr = selected_expr - signature.control_mean
        scores = _analytic_projection_scores(
            centered_expr,
            signature.beta,
            scale_factor=scale_factor,
            lambda_=lambda_,
        )
        target_mask = combined_labels == perturbation
        if target_mask.any():
            batch_target_max = float(np.max(scores[target_mask]))
            raw_target_max = batch_target_max if raw_target_max is None else max(raw_target_max, batch_target_max)

        frames.append(
            _build_result_frame(
                row_ids=combined_row_ids,
                perturbation_labels=combined_labels,
                target_perturbation=str(perturbation),
                scores=scores,
                fidelity="approx",
                method="analytic_projection",
                selected_target_gene_count=signature.gene_indices.size,
                score_status=np.full(combined_labels.shape[0], "projected", dtype=object),
                scale_factor=scale_factor,
                lambda_=lambda_,
            )
        )

    _validate_stream_cell_counts(
        control_count=control_count,
        target_count=target_count,
        perturbation=perturbation,
        control_label=control_label,
    )
    _apply_scaled_stream_scores(
        frames,
        scale_score=scale_score,
        raw_target_max=raw_target_max,
    )

    metadata = {
        "target_genes": signature.gene_names,
        "target_gene_source": signature.source,
        "method": "analytic_projection",
        "signature_norm": float(np.linalg.norm(signature.beta)),
        "scale_score": scale_score,
        "scale_score_applied": bool(scale_score and raw_target_max is not None and raw_target_max > 0),
        "scale_score_max_before_scaling": raw_target_max,
        "stream_semantics": "multi-pass" if signature.source != "precomputed" else "precomputed-signature",
    }
    frame = pd.concat(frames, ignore_index=True) if frames else _empty_result_frame()
    return frame, metadata


def _run_stream_approx_one_shot(
    *,
    batches: Any,
    selected: Sequence[Any],
    label_by_row_id: dict[Any, Any],
    var_names: np.ndarray,
    gene_lookup: Mapping[str, int],
    control_label: str,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    target_gene_min: int,
    target_gene_max: int,
    scale_factor: float,
    lambda_: float,
    use_perturb_expression: bool,
    scale_score: bool,
    target_signatures: Mapping[str, Sequence[float]] | Sequence[float] | None,
    control_means: Mapping[str, Sequence[float]] | Sequence[float] | None,
) -> list[tuple[pd.DataFrame, dict[str, Any]]]:
    signatures = {
        perturbation: _resolve_stream_signature(
            batches=None,
            perturbation=perturbation,
            label_by_row_id=label_by_row_id,
            var_names=var_names,
            gene_lookup=gene_lookup,
            control_label=control_label,
            target_genes=target_genes,
            target_gene_min=target_gene_min,
            target_gene_max=target_gene_max,
            use_perturb_expression=use_perturb_expression,
            target_signatures=target_signatures,
            control_means=control_means,
            require_precomputed=True,
            precomputed_context=(
                "run_ps_score_stream approx mode with a one-shot iterator"
            ),
        )
        for perturbation in selected
    }

    frames_by_perturbation: dict[Any, list[pd.DataFrame]] = {perturbation: [] for perturbation in selected}
    raw_target_max: dict[Any, float | None] = {perturbation: None for perturbation in selected}
    control_counts: dict[Any, int] = {perturbation: 0 for perturbation in selected}
    target_counts: dict[Any, int] = {perturbation: 0 for perturbation in selected}

    for batch in iter_csr_batches(batches):
        _validate_stream_feature_count(batch.shape[1], var_names.size)
        matrix = csr_batch_to_matrix(batch)
        labels = _batch_labels(batch.row_ids, label_by_row_id)
        row_ids = np.asarray(batch.row_ids, dtype=object)

        for perturbation in selected:
            signature = signatures[perturbation]
            combined_mask = (labels == control_label) | (labels == perturbation)
            if not combined_mask.any():
                continue

            combined_labels = labels[combined_mask]
            combined_row_ids = row_ids[combined_mask]
            control_counts[perturbation] += int(np.sum(combined_labels == control_label))
            target_counts[perturbation] += int(np.sum(combined_labels == perturbation))
            selected_expr = np.asarray(
                matrix[combined_mask][:, signature.gene_indices].toarray(),
                dtype=float,
            )
            centered_expr = selected_expr - signature.control_mean
            scores = _analytic_projection_scores(
                centered_expr,
                signature.beta,
                scale_factor=scale_factor,
                lambda_=lambda_,
            )
            target_mask = combined_labels == perturbation
            if target_mask.any():
                batch_target_max = float(np.max(scores[target_mask]))
                previous_max = raw_target_max[perturbation]
                raw_target_max[perturbation] = (
                    batch_target_max if previous_max is None else max(previous_max, batch_target_max)
                )

            frames_by_perturbation[perturbation].append(
                _build_result_frame(
                    row_ids=combined_row_ids,
                    perturbation_labels=combined_labels,
                    target_perturbation=str(perturbation),
                    scores=scores,
                    fidelity="approx",
                    method="analytic_projection",
                    selected_target_gene_count=signature.gene_indices.size,
                    score_status=np.full(combined_labels.shape[0], "projected", dtype=object),
                    scale_factor=scale_factor,
                    lambda_=lambda_,
                )
            )

    outputs: list[tuple[pd.DataFrame, dict[str, Any]]] = []
    for perturbation in selected:
        _validate_stream_cell_counts(
            control_count=control_counts[perturbation],
            target_count=target_counts[perturbation],
            perturbation=perturbation,
            control_label=control_label,
        )
        frames = frames_by_perturbation[perturbation]
        _apply_scaled_stream_scores(
            frames,
            scale_score=scale_score,
            raw_target_max=raw_target_max[perturbation],
        )
        signature = signatures[perturbation]
        metadata = {
            "target_genes": signature.gene_names,
            "target_gene_source": signature.source,
            "method": "analytic_projection",
            "signature_norm": float(np.linalg.norm(signature.beta)),
            "scale_score": scale_score,
            "scale_score_applied": bool(
                scale_score
                and raw_target_max[perturbation] is not None
                and raw_target_max[perturbation] > 0
            ),
            "scale_score_max_before_scaling": raw_target_max[perturbation],
            "stream_semantics": "one-pass-precomputed",
        }
        frame = pd.concat(frames, ignore_index=True) if frames else _empty_result_frame()
        outputs.append((frame, metadata))
    return outputs


def _resolve_stream_signature(
    *,
    batches: Any,
    perturbation: Any,
    label_by_row_id: dict[Any, Any],
    var_names: np.ndarray,
    gene_lookup: Mapping[str, int],
    control_label: str,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    target_gene_min: int,
    target_gene_max: int,
    use_perturb_expression: bool,
    target_signatures: Mapping[str, Sequence[float]] | Sequence[float] | None,
    control_means: Mapping[str, Sequence[float]] | Sequence[float] | None,
    require_precomputed: bool,
    precomputed_context: str,
) -> _StreamSignature:
    precomputed = _maybe_resolve_stream_precomputed_signature(
        perturbation=perturbation,
        target_genes=target_genes,
        target_signatures=target_signatures,
        control_means=control_means,
        gene_lookup=gene_lookup,
        target_gene_min=target_gene_min,
        target_gene_max=target_gene_max,
        use_perturb_expression=use_perturb_expression,
        require_precomputed=require_precomputed,
        context=precomputed_context,
    )
    if precomputed is not None:
        return precomputed

    if batches is None:
        raise ValueError(
            f"{precomputed_context} requires explicit target_genes, target_signatures, and control_means"
        )

    control_stats, target_stats = _collect_stream_feature_stats(
        batches=batches,
        perturbation=perturbation,
        label_by_row_id=label_by_row_id,
        control_label=control_label,
        n_features=var_names.size,
    )
    _validate_stream_cell_counts(
        control_count=control_stats.count,
        target_count=target_stats.count,
        perturbation=perturbation,
        control_label=control_label,
    )
    gene_indices, gene_names, source = _resolve_stream_target_gene_indices(
        perturbation=perturbation,
        target_genes=target_genes,
        gene_lookup=gene_lookup,
        var_names=var_names,
        control_stats=control_stats,
        target_stats=target_stats,
        target_gene_min=target_gene_min,
        target_gene_max=target_gene_max,
        use_perturb_expression=use_perturb_expression,
    )
    control_mean = control_stats.means()[gene_indices]
    target_mean = target_stats.means()[gene_indices]
    beta = target_mean - control_mean
    if not np.any(np.abs(beta) > 0):
        raise ValueError(f"PS score signature is zero for perturbation {perturbation!r}")
    return _StreamSignature(
        gene_indices=gene_indices,
        gene_names=gene_names,
        beta=beta.astype(float, copy=False),
        control_mean=control_mean.astype(float, copy=False),
        source=source,
    )


def _maybe_resolve_stream_precomputed_signature(
    *,
    perturbation: Any,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    target_signatures: Mapping[str, Sequence[float]] | Sequence[float] | None,
    control_means: Mapping[str, Sequence[float]] | Sequence[float] | None,
    gene_lookup: Mapping[str, int],
    target_gene_min: int,
    target_gene_max: int,
    use_perturb_expression: bool,
    require_precomputed: bool,
    context: str,
) -> _StreamSignature | None:
    explicit = _get_explicit_target_genes(target_genes, perturbation=perturbation)
    if require_precomputed and (
        explicit is None or target_signatures is None or control_means is None
    ):
        raise ValueError(
            f"{context} requires explicit target_genes, target_signatures, and control_means"
        )
    if target_signatures is None and control_means is None:
        return None
    if target_signatures is None or control_means is None:
        raise ValueError(
            f"{context} requires both target_signatures and control_means when either is provided"
        )
    if explicit is None:
        raise ValueError(f"{context} requires explicit target_genes when using precomputed signatures")

    all_gene_names = _normalize_gene_names(explicit)
    beta_values = _get_explicit_numeric_values(target_signatures, perturbation=perturbation)
    control_values = _get_explicit_numeric_values(control_means, perturbation=perturbation)
    if beta_values is None or control_values is None:
        raise ValueError(
            f"{context} requires explicit target_genes, target_signatures, and control_means"
        )

    beta_all = _coerce_numeric_vector(beta_values, name="target_signatures")
    control_all = _coerce_numeric_vector(control_values, name="control_means")
    expected_full = len(all_gene_names)
    if beta_all.shape[0] != expected_full:
        raise ValueError(
            f"target_signatures for perturbation {perturbation!r} must have length {expected_full}"
        )
    if control_all.shape[0] != expected_full:
        raise ValueError(
            f"control_means for perturbation {perturbation!r} must have length {expected_full}"
        )

    selected_positions = [
        index
        for index, gene in enumerate(all_gene_names)
        if use_perturb_expression or gene != str(perturbation)
    ]
    selected_positions = selected_positions[:target_gene_max]
    gene_names = [all_gene_names[index] for index in selected_positions]
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

    position_array = np.asarray(selected_positions, dtype=np.int64)
    beta = beta_all[position_array]
    control_mean = control_all[position_array]
    if not np.any(np.abs(beta) > 0):
        raise ValueError(f"PS score signature is zero for perturbation {perturbation!r}")

    gene_indices = np.asarray([gene_lookup[gene] for gene in gene_names], dtype=np.int64)
    return _StreamSignature(
        gene_indices=gene_indices,
        gene_names=gene_names,
        beta=beta,
        control_mean=control_mean,
        source="precomputed",
    )


def _resolve_stream_target_gene_indices(
    *,
    perturbation: Any,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    gene_lookup: Mapping[str, int],
    var_names: np.ndarray,
    control_stats: StreamFeatureStats,
    target_stats: StreamFeatureStats,
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

    scores = welch_t_scores_from_stats(target_stats, control_stats)
    mean_diff = target_stats.means() - control_stats.means()
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


def _collect_stream_feature_stats(
    *,
    batches: Any,
    perturbation: Any,
    label_by_row_id: dict[Any, Any],
    control_label: str,
    n_features: int,
) -> tuple[StreamFeatureStats, StreamFeatureStats]:
    control_count = 0
    target_count = 0
    control_sums = np.zeros(n_features, dtype=float)
    target_sums = np.zeros(n_features, dtype=float)
    control_squared = np.zeros(n_features, dtype=float)
    target_squared = np.zeros(n_features, dtype=float)

    for batch in iter_csr_batches(batches):
        _validate_stream_feature_count(batch.shape[1], n_features)
        matrix = csr_batch_to_matrix(batch)
        labels = _batch_labels(batch.row_ids, label_by_row_id)
        control_mask = labels == control_label
        target_mask = labels == perturbation
        if control_mask.any():
            control_block = matrix[control_mask]
            control_count += int(control_block.shape[0])
            control_sums += np.asarray(control_block.sum(axis=0)).ravel()
            control_squared += np.asarray(control_block.power(2).sum(axis=0)).ravel()
        if target_mask.any():
            target_block = matrix[target_mask]
            target_count += int(target_block.shape[0])
            target_sums += np.asarray(target_block.sum(axis=0)).ravel()
            target_squared += np.asarray(target_block.power(2).sum(axis=0)).ravel()

    return (
        StreamFeatureStats(count=control_count, sums=control_sums, squared_sums=control_squared),
        StreamFeatureStats(count=target_count, sums=target_sums, squared_sums=target_squared),
    )


def _apply_scaled_stream_scores(
    frames: Sequence[pd.DataFrame],
    *,
    scale_score: bool,
    raw_target_max: float | None,
) -> None:
    if not scale_score or raw_target_max is None or raw_target_max <= 0:
        return
    for frame in frames:
        frame.loc[:, "ps_score"] = frame["ps_score"] / raw_target_max


def _annotate_stream_result(
    result: pd.DataFrame,
    *,
    fidelity: str,
    perturbation_key: str,
    control_label: str,
    perturbations: list[Any],
    metadata: dict[Any, Any],
    stream_mode: str,
) -> pd.DataFrame:
    result.attrs["ps_score"] = {
        "algorithm": "ps_score",
        "input_mode": "stream",
        "fidelity": fidelity,
        "perturbation_key": perturbation_key,
        "control_label": control_label,
        "perturbations": perturbations,
        "metadata_by_perturbation": metadata,
        "stream_mode": stream_mode,
    }
    return result


def _build_label_lookup(*, row_ids: np.ndarray, labels: np.ndarray) -> dict[Any, Any]:
    lookup: dict[Any, Any] = {}
    for row_id, label in zip(row_ids, labels, strict=False):
        if row_id in lookup:
            raise ValueError(f"obs contains duplicate row_id {row_id!r}")
        lookup[row_id] = label
    return lookup


def _batch_labels(batch_row_ids: Sequence[Any], label_by_row_id: dict[Any, Any]) -> np.ndarray:
    labels: list[Any] = []
    missing: list[str] = []
    for row_id in batch_row_ids:
        if row_id not in label_by_row_id:
            missing.append(str(row_id))
            continue
        labels.append(label_by_row_id[row_id])
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(f"batch row_ids were not found in obs: {preview}")
    return np.asarray(labels, dtype=object)


def _validate_stream_feature_count(batch_features: int, expected_features: int) -> None:
    if batch_features != expected_features:
        raise ValueError(
            f"batch feature count {batch_features} does not match len(var_names)={expected_features}"
        )


def _validate_stream_cell_counts(
    *,
    control_count: int,
    target_count: int,
    perturbation: Any,
    control_label: str,
) -> None:
    if control_count < 2:
        raise ValueError(f"PS score requires at least 2 {control_label!r} control cells")
    if target_count < 2:
        raise ValueError(f"PS score requires at least 2 cells for perturbation {perturbation!r}")


def _validate_optional_numeric_vectors(
    name: str,
    values: Mapping[str, Sequence[float]] | Sequence[float] | None,
) -> None:
    if values is None:
        return
    if isinstance(values, Mapping):
        for key, vector in values.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{name} mapping keys must be non-empty strings")
            _coerce_numeric_vector(vector, name=name)
        return
    _coerce_numeric_vector(values, name=name)


def _get_explicit_numeric_values(
    values: Mapping[str, Sequence[float]] | Sequence[float] | None,
    *,
    perturbation: Any,
) -> Sequence[float] | None:
    if values is None:
        return None
    if isinstance(values, Mapping):
        if perturbation in values:
            return values[perturbation]
        return values.get(str(perturbation))
    return values


def _coerce_numeric_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numeric values, not a string")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional numeric sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return array.astype(float, copy=False)


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
