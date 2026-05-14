"""Exact scMAGeCK-style PS scores for single-label AnnData inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, sparse

from .stats import extract_anndata_matrix, get_obs_column, resolve_perturbations, validate_layer


RESULT_COLUMNS = [
    "row_id",
    "perturbation_label",
    "target_perturbation",
    "ps_score",
    "method",
    "selected_target_gene_count",
    "score_status",
    "scale_factor",
    "score_lambda",
    "lr_lambda",
]

SUPPORTED_TARGET_GENE_SOURCES = ("provided", "scanpy_de", "hvg")


def run_ps_score_exact_anndata(
    adata: Any,
    *,
    perturb_column: str,
    ctrl_name: str,
    layer: str | None = None,
    counts_layer: str | None = "counts",
    perturbations: Sequence[str] | None = None,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    target_gene_source: str = "provided",
    hvg_key: str = "highly_variable",
    target_gene_min: int = 10,
    target_gene_max: int = 500,
    gene_filter_min_fraction: float = 0.01,
    apply_gene_filter: bool = True,
    clip_quantile: float = 0.95,
    apply_quantile_clip: bool = True,
    lr_lambda: float = 0.01,
    score_lambda: float = 0.0,
    scale_factor: float = 3.0,
    scale_score: bool = True,
    return_wide: bool = False,
) -> pd.DataFrame:
    """Run the exact single-label scMAGeCK-style PS-score workflow on AnnData."""

    _validate_adata_like(adata)
    if not isinstance(perturb_column, str) or not perturb_column:
        raise ValueError("perturb_column must be a non-empty string")
    if not isinstance(ctrl_name, str) or not ctrl_name:
        raise ValueError("ctrl_name must be a non-empty string")
    validate_layer(layer)
    counts_layer = _validate_optional_layer_name(counts_layer, name="counts_layer")
    _validate_target_genes(target_genes)
    target_gene_source = _validate_target_gene_source(target_gene_source)
    if not isinstance(hvg_key, str) or not hvg_key:
        raise ValueError("hvg_key must be a non-empty string")
    target_gene_min = _validate_positive_int("target_gene_min", target_gene_min)
    target_gene_max = _validate_positive_int("target_gene_max", target_gene_max)
    if target_gene_max < target_gene_min:
        raise ValueError("target_gene_max must be greater than or equal to target_gene_min")
    gene_filter_min_fraction = _validate_fraction(
        "gene_filter_min_fraction",
        gene_filter_min_fraction,
        allow_zero=True,
        allow_one=True,
    )
    clip_quantile = _validate_fraction(
        "clip_quantile",
        clip_quantile,
        allow_zero=False,
        allow_one=True,
    )
    lr_lambda = _validate_non_negative_float("lr_lambda", lr_lambda)
    score_lambda = _validate_non_negative_float("score_lambda", score_lambda)
    scale_factor = _validate_positive_float("scale_factor", scale_factor)
    _validate_bool("apply_gene_filter", apply_gene_filter)
    _validate_bool("apply_quantile_clip", apply_quantile_clip)
    _validate_bool("scale_score", scale_score)
    _validate_bool("return_wide", return_wide)

    if target_gene_source == "provided" and target_genes is None:
        raise ValueError("target_genes must be provided when target_gene_source='provided'")
    if target_gene_source != "provided" and target_genes is not None:
        raise ValueError(
            "target_genes must be None unless target_gene_source='provided'"
        )

    labels_all = np.asarray(get_obs_column(adata.obs, perturb_column), dtype=object)
    row_ids_all = np.asarray(adata.obs_names, dtype=object)
    if labels_all.ndim != 1 or labels_all.size == 0:
        raise ValueError("adata must contain at least one observation")
    if row_ids_all.shape[0] != labels_all.shape[0]:
        raise ValueError("adata.obs_names and the perturbation column must have the same length")
    if not np.any(labels_all == ctrl_name):
        raise ValueError(
            f"ctrl_name {ctrl_name!r} was not found in adata.obs[{perturb_column!r}]"
        )

    selected_perturbations = resolve_perturbations(
        labels_all,
        control_label=ctrl_name,
        perturbations=perturbations,
    )
    if not selected_perturbations:
        return _empty_result(return_wide=return_wide)

    cell_mask = (labels_all == ctrl_name) | np.isin(labels_all, selected_perturbations)
    cell_indices = np.flatnonzero(cell_mask)
    labels = labels_all[cell_mask]
    row_ids = row_ids_all[cell_mask]
    for perturbation in selected_perturbations:
        if not np.any(labels == perturbation):
            raise ValueError(f"No cells were found for perturbation {perturbation!r}")

    var_names = np.asarray(adata.var_names, dtype=object)
    if var_names.ndim != 1 or var_names.size == 0:
        raise ValueError("adata.var_names must be a non-empty one-dimensional sequence")
    gene_lookup = {str(gene): index for index, gene in enumerate(var_names.astype(str))}

    expression_matrix = extract_anndata_matrix(adata, layer=layer)
    filter_matrix, filter_source = _resolve_filter_matrix(
        adata,
        expression_matrix=expression_matrix,
        expression_source=_layer_name_or_default(layer),
        counts_layer=counts_layer,
        apply_gene_filter=apply_gene_filter,
    )

    genes_by_perturbation, source_metadata = _resolve_target_genes_by_perturbation(
        adata,
        labels_all=labels_all,
        perturb_column=perturb_column,
        ctrl_name=ctrl_name,
        selected_perturbations=selected_perturbations,
        target_genes=target_genes,
        target_gene_source=target_gene_source,
        hvg_key=hvg_key,
        target_gene_min=target_gene_min,
        target_gene_max=target_gene_max,
        layer=layer,
        cell_mask=cell_mask,
        gene_lookup=gene_lookup,
    )

    filter_metadata = _filter_target_genes(
        filter_matrix=filter_matrix,
        cell_indices=cell_indices,
        labels=labels,
        ctrl_name=ctrl_name,
        selected_perturbations=selected_perturbations,
        genes_by_perturbation=genes_by_perturbation,
        gene_lookup=gene_lookup,
        target_gene_min=target_gene_min,
        gene_filter_min_fraction=gene_filter_min_fraction,
        apply_gene_filter=apply_gene_filter,
    )
    filtered_genes_by_perturbation = filter_metadata["genes_by_perturbation"]

    union_genes = _ordered_union(
        filtered_genes_by_perturbation[perturbation]
        for perturbation in selected_perturbations
    )
    union_gene_indices = np.asarray([gene_lookup[gene] for gene in union_genes], dtype=np.int64)

    y_matrix = _select_dense(expression_matrix, cell_indices, union_gene_indices)
    clip_values: np.ndarray | None = None
    if apply_quantile_clip:
        y_matrix, clip_values = _clip_columns(y_matrix, clip_quantile)

    x_matrix = _build_design_matrix(labels, selected_perturbations)
    beta = _solve_ridge_beta(x_matrix, y_matrix, lr_lambda)
    score_matrix, score_metadata = _compute_scores(
        y_matrix=y_matrix,
        labels=labels,
        ctrl_name=ctrl_name,
        selected_perturbations=selected_perturbations,
        beta=beta,
        score_lambda=score_lambda,
        scale_factor=scale_factor,
        scale_score=scale_score,
    )

    metadata = {
        "algorithm": "ps_score_exact",
        "input_type": "anndata-single-label",
        "layer": layer,
        "counts_layer": counts_layer,
        "perturb_column": perturb_column,
        "ctrl_name": ctrl_name,
        "perturbations": list(selected_perturbations),
        "target_gene_source": target_gene_source,
        "target_gene_source_detail": source_metadata,
        "target_gene_min": int(target_gene_min),
        "target_gene_max": int(target_gene_max),
        "genes_by_perturbation": filtered_genes_by_perturbation,
        "union_target_genes": union_genes,
        "union_target_gene_count": len(union_genes),
        "apply_gene_filter": bool(apply_gene_filter),
        "gene_filter_min_fraction": float(gene_filter_min_fraction),
        "gene_filter_source": filter_source,
        "gene_filter_metadata": filter_metadata,
        "apply_quantile_clip": bool(apply_quantile_clip),
        "clip_quantile": float(clip_quantile),
        "clip_values": None if clip_values is None else clip_values.tolist(),
        "lr_lambda": float(lr_lambda),
        "score_lambda": float(score_lambda),
        "scale_factor": float(scale_factor),
        "scale_score": bool(scale_score),
        "x_shape": tuple(int(value) for value in x_matrix.shape),
        "y_shape": tuple(int(value) for value in y_matrix.shape),
        "beta_shape": tuple(int(value) for value in beta.shape),
        "score_metadata": score_metadata,
    }

    if return_wide:
        result = _build_wide_result(
            row_ids=row_ids,
            labels=labels,
            selected_perturbations=selected_perturbations,
            score_matrix=score_matrix,
        )
    else:
        result = _build_long_result(
            row_ids=row_ids,
            labels=labels,
            ctrl_name=ctrl_name,
            selected_perturbations=selected_perturbations,
            score_matrix=score_matrix,
            selected_target_gene_counts={
                perturbation: len(filtered_genes_by_perturbation[perturbation])
                for perturbation in selected_perturbations
            },
            scale_factor=scale_factor,
            score_lambda=score_lambda,
            lr_lambda=lr_lambda,
        )
    result.attrs["ps_score_exact"] = metadata
    return result


def _validate_adata_like(adata: Any) -> None:
    if adata is None:
        raise ValueError("adata must not be None")
    for attribute in ("X", "layers", "obs", "obs_names", "var", "var_names"):
        if not hasattr(adata, attribute):
            raise TypeError(f"adata must provide {attribute}")


def _validate_optional_layer_name(value: str | None, *, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    return value


def _validate_target_genes(
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
) -> None:
    if target_genes is None:
        return
    if isinstance(target_genes, str):
        raise TypeError("target_genes must be a mapping or sequence of gene names, not a string")
    if isinstance(target_genes, Mapping):
        for key, genes in target_genes.items():
            if not isinstance(key, str) or not key:
                raise TypeError("target_genes mapping keys must be non-empty strings")
            _normalize_gene_names(genes)
        return
    _normalize_gene_names(target_genes)


def _validate_target_gene_source(value: str) -> str:
    if value not in SUPPORTED_TARGET_GENE_SOURCES:
        allowed = ", ".join(SUPPORTED_TARGET_GENE_SOURCES)
        raise ValueError(f"Unsupported target_gene_source {value!r}; expected one of: {allowed}")
    return value


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


def _validate_fraction(
    name: str,
    value: float,
    *,
    allow_zero: bool,
    allow_one: bool,
) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a numeric value")
    lower_ok = value >= 0 if allow_zero else value > 0
    upper_ok = value <= 1 if allow_one else value < 1
    if not lower_ok or not upper_ok:
        if allow_zero and allow_one:
            raise ValueError(f"{name} must be between 0 and 1 inclusive")
        if allow_zero:
            raise ValueError(f"{name} must be in [0, 1)")
        if allow_one:
            raise ValueError(f"{name} must be in (0, 1]")
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _normalize_gene_names(genes: Sequence[str]) -> list[str]:
    if isinstance(genes, str):
        raise TypeError("target gene lists must be sequences of gene names, not strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        if not isinstance(gene, str) or not gene:
            raise TypeError("target gene names must be non-empty strings")
        if gene in seen:
            continue
        normalized.append(gene)
        seen.add(gene)
    return normalized


def _resolve_filter_matrix(
    adata: Any,
    *,
    expression_matrix: Any,
    expression_source: str,
    counts_layer: str | None,
    apply_gene_filter: bool,
) -> tuple[Any, str]:
    if not apply_gene_filter:
        return expression_matrix, "none"
    if counts_layer is not None and counts_layer in adata.layers:
        return adata.layers[counts_layer], counts_layer
    return expression_matrix, expression_source


def _layer_name_or_default(layer: str | None) -> str:
    return "adata.X" if layer is None else layer


def _resolve_target_genes_by_perturbation(
    adata: Any,
    *,
    labels_all: np.ndarray,
    perturb_column: str,
    ctrl_name: str,
    selected_perturbations: Sequence[str],
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    target_gene_source: str,
    hvg_key: str,
    target_gene_min: int,
    target_gene_max: int,
    layer: str | None,
    cell_mask: np.ndarray,
    gene_lookup: Mapping[str, int],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if target_gene_source == "provided":
        genes = _resolve_provided_target_genes(
            target_genes=target_genes,
            selected_perturbations=selected_perturbations,
            gene_lookup=gene_lookup,
            target_gene_min=target_gene_min,
            target_gene_max=target_gene_max,
        )
        return genes, {"mode": "provided"}
    if target_gene_source == "hvg":
        genes = _resolve_hvg_target_genes(
            adata,
            hvg_key=hvg_key,
            selected_perturbations=selected_perturbations,
            gene_lookup=gene_lookup,
            target_gene_min=target_gene_min,
            target_gene_max=target_gene_max,
        )
        return genes, {"mode": "hvg", "hvg_key": hvg_key}
    genes = _resolve_scanpy_target_genes(
        adata,
        labels_all=labels_all,
        perturb_column=perturb_column,
        ctrl_name=ctrl_name,
        selected_perturbations=selected_perturbations,
        layer=layer,
        cell_mask=cell_mask,
        gene_lookup=gene_lookup,
        target_gene_min=target_gene_min,
        target_gene_max=target_gene_max,
    )
    return genes, {"mode": "scanpy_de", "layer": layer}


def _resolve_provided_target_genes(
    *,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    selected_perturbations: Sequence[str],
    gene_lookup: Mapping[str, int],
    target_gene_min: int,
    target_gene_max: int,
) -> dict[str, list[str]]:
    genes_by_perturbation: dict[str, list[str]] = {}
    for perturbation in selected_perturbations:
        provided = _get_provided_genes(target_genes, perturbation=perturbation)
        if provided is None:
            raise ValueError(
                f"No target genes were provided for perturbation {perturbation!r}"
            )
        genes = _normalize_gene_names(provided)
        if len(genes) > target_gene_max:
            genes = genes[:target_gene_max]
        if len(genes) < target_gene_min:
            raise ValueError(
                f"Need at least {target_gene_min} target genes for perturbation {perturbation!r}"
            )
        missing = [gene for gene in genes if gene not in gene_lookup]
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(
                f"Unknown target genes requested for perturbation {perturbation!r}: {joined}"
            )
        genes_by_perturbation[str(perturbation)] = genes
    return genes_by_perturbation


def _resolve_hvg_target_genes(
    adata: Any,
    *,
    hvg_key: str,
    selected_perturbations: Sequence[str],
    gene_lookup: Mapping[str, int],
    target_gene_min: int,
    target_gene_max: int,
) -> dict[str, list[str]]:
    if hvg_key not in adata.var:
        raise ValueError(f"HVG key {hvg_key!r} was not found in adata.var")
    values = np.asarray(adata.var[hvg_key])
    if values.ndim != 1:
        raise ValueError(f"adata.var[{hvg_key!r}] must be one-dimensional")

    if np.issubdtype(values.dtype, np.bool_):
        selected_indices = np.flatnonzero(values)
    else:
        numeric = pd.to_numeric(pd.Series(values), errors="coerce")
        valid_mask = ~numeric.isna().to_numpy()
        if not valid_mask.any():
            raise ValueError(f"adata.var[{hvg_key!r}] does not contain any HVG entries")
        valid_indices = np.flatnonzero(valid_mask)
        order = np.argsort(numeric.to_numpy()[valid_indices], kind="stable")
        selected_indices = valid_indices[order]

    if selected_indices.size == 0:
        raise ValueError(f"adata.var[{hvg_key!r}] does not contain any HVGs")

    genes = [str(adata.var_names[index]) for index in selected_indices[:target_gene_max]]
    if len(genes) < target_gene_min:
        raise ValueError(f"Need at least {target_gene_min} HVGs from adata.var[{hvg_key!r}]")
    missing = [gene for gene in genes if gene not in gene_lookup]
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Resolved HVG genes were missing from adata.var_names: {joined}")
    return {str(perturbation): list(genes) for perturbation in selected_perturbations}


def _resolve_scanpy_target_genes(
    adata: Any,
    *,
    labels_all: np.ndarray,
    perturb_column: str,
    ctrl_name: str,
    selected_perturbations: Sequence[str],
    layer: str | None,
    cell_mask: np.ndarray,
    gene_lookup: Mapping[str, int],
    target_gene_min: int,
    target_gene_max: int,
) -> dict[str, list[str]]:
    try:
        import scanpy as sc
    except ImportError as error:  # pragma: no cover - exercised in Phase 3
        raise ImportError(
            "target_gene_source='scanpy_de' requires scanpy to be installed"
        ) from error

    subset = adata[cell_mask].copy()
    subset_labels = np.asarray(labels_all[cell_mask], dtype=object)
    if not np.any(subset_labels == ctrl_name):
        raise ValueError("Selected AnnData subset does not contain any control cells")
    sc.tl.rank_genes_groups(
        subset,
        groupby=perturb_column,
        groups=list(selected_perturbations),
        reference=ctrl_name,
        use_raw=False,
        layer=layer,
        n_genes=target_gene_max,
    )

    genes_by_perturbation: dict[str, list[str]] = {}
    for perturbation in selected_perturbations:
        de_frame = sc.get.rank_genes_groups_df(subset, group=perturbation)
        if "names" not in de_frame.columns:
            raise ValueError("scanpy rank_genes_groups output does not contain a 'names' column")
        genes = _normalize_gene_names(
            [gene for gene in de_frame["names"].tolist() if isinstance(gene, str) and gene]
        )
        genes = [gene for gene in genes if gene in gene_lookup][:target_gene_max]
        if len(genes) < target_gene_min:
            raise ValueError(
                f"Scanpy DE found fewer than {target_gene_min} target genes for perturbation {perturbation!r}"
            )
        genes_by_perturbation[str(perturbation)] = genes
    return genes_by_perturbation


def _get_provided_genes(
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None,
    *,
    perturbation: str,
) -> Sequence[str] | None:
    if target_genes is None:
        return None
    if isinstance(target_genes, Mapping):
        if perturbation in target_genes:
            return target_genes[perturbation]
        return target_genes.get(str(perturbation))
    return target_genes


def _filter_target_genes(
    *,
    filter_matrix: Any,
    cell_indices: np.ndarray,
    labels: np.ndarray,
    ctrl_name: str,
    selected_perturbations: Sequence[str],
    genes_by_perturbation: Mapping[str, Sequence[str]],
    gene_lookup: Mapping[str, int],
    target_gene_min: int,
    gene_filter_min_fraction: float,
    apply_gene_filter: bool,
) -> dict[str, Any]:
    filtered: dict[str, list[str]] = {}
    counts_before: dict[str, int] = {}
    counts_after: dict[str, int] = {}

    for perturbation in selected_perturbations:
        genes = list(genes_by_perturbation[str(perturbation)])
        counts_before[str(perturbation)] = len(genes)
        if not apply_gene_filter:
            filtered[str(perturbation)] = genes
            counts_after[str(perturbation)] = len(genes)
            continue

        relevant_rows = np.flatnonzero((labels == ctrl_name) | (labels == perturbation))
        gene_indices = np.asarray([gene_lookup[gene] for gene in genes], dtype=np.int64)
        candidate = _select_dense(filter_matrix, cell_indices[relevant_rows], gene_indices)
        keep_fraction = (candidate > 0).mean(axis=0)
        keep_mask = keep_fraction >= gene_filter_min_fraction
        kept = [gene for gene, keep in zip(genes, keep_mask, strict=False) if keep]
        if len(kept) < target_gene_min:
            raise ValueError(
                "Gene filtering left fewer than "
                f"{target_gene_min} target genes for perturbation {perturbation!r}"
            )
        filtered[str(perturbation)] = kept
        counts_after[str(perturbation)] = len(kept)

    return {
        "genes_by_perturbation": filtered,
        "target_gene_counts_before_filter": counts_before,
        "target_gene_counts_after_filter": counts_after,
    }


def _ordered_union(groups: Any) -> list[str]:
    union: list[str] = []
    seen: set[str] = set()
    for genes in groups:
        for gene in genes:
            if gene in seen:
                continue
            union.append(gene)
            seen.add(gene)
    return union


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
        raise ValueError("selected matrix must be two-dimensional")
    if col_indices is None:
        return np.asarray(dense[row_indices], dtype=float)
    return np.asarray(dense[np.ix_(row_indices, col_indices)], dtype=float)


def _clip_columns(matrix: np.ndarray, quantile: float) -> tuple[np.ndarray, np.ndarray]:
    clip_values = np.quantile(matrix, quantile, axis=0)
    clipped = np.minimum(matrix, clip_values)
    return clipped, np.asarray(clip_values, dtype=float)


def _build_design_matrix(labels: np.ndarray, selected_perturbations: Sequence[str]) -> np.ndarray:
    x_matrix = np.ones((labels.shape[0], len(selected_perturbations) + 1), dtype=float)
    for column_index, perturbation in enumerate(selected_perturbations, start=1):
        x_matrix[:, column_index] = (labels == perturbation).astype(float)
    return x_matrix


def _solve_ridge_beta(x_matrix: np.ndarray, y_matrix: np.ndarray, lr_lambda: float) -> np.ndarray:
    gram = x_matrix.T @ x_matrix
    ridge = gram + lr_lambda * np.eye(gram.shape[0], dtype=float)
    rhs = x_matrix.T @ y_matrix
    try:
        factor = linalg.cho_factor(ridge, lower=True, check_finite=True)
    except linalg.LinAlgError as error:
        raise ValueError(
            "Ridge system is not solvable under the requested lr_lambda"
        ) from error
    return np.asarray(linalg.cho_solve(factor, rhs, check_finite=True), dtype=float)


def _compute_scores(
    *,
    y_matrix: np.ndarray,
    labels: np.ndarray,
    ctrl_name: str,
    selected_perturbations: Sequence[str],
    beta: np.ndarray,
    score_lambda: float,
    scale_factor: float,
    scale_score: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    score_matrix = np.zeros((labels.shape[0], len(selected_perturbations)), dtype=float)
    baseline = beta[0]
    metadata: dict[str, Any] = {}

    for column_index, perturbation in enumerate(selected_perturbations):
        target_mask = labels == perturbation
        perturbation_beta = beta[column_index + 1]
        beta_norm_sq = float(np.dot(perturbation_beta, perturbation_beta))
        if beta_norm_sq <= 0:
            raise ValueError(f"Perturbation {perturbation!r} produced a zero beta vector")
        if target_mask.any():
            centered = y_matrix[target_mask] - baseline
            raw = (centered @ perturbation_beta - score_lambda) / beta_norm_sq
            bounded = np.clip(raw, 0.0, scale_factor)
            score_matrix[target_mask, column_index] = bounded / scale_factor
        max_before_scaling = float(score_matrix[:, column_index].max(initial=0.0))
        if scale_score and max_before_scaling > 0:
            score_matrix[:, column_index] = score_matrix[:, column_index] / max_before_scaling
        metadata[str(perturbation)] = {
            "beta_norm_sq": beta_norm_sq,
            "control_count": int(np.sum(labels == ctrl_name)),
            "target_count": int(np.sum(target_mask)),
            "max_score_before_column_scale": max_before_scaling,
            "column_scaled": bool(scale_score and max_before_scaling > 0),
        }

    return score_matrix, metadata


def _build_long_result(
    *,
    row_ids: np.ndarray,
    labels: np.ndarray,
    ctrl_name: str,
    selected_perturbations: Sequence[str],
    score_matrix: np.ndarray,
    selected_target_gene_counts: Mapping[str, int],
    scale_factor: float,
    score_lambda: float,
    lr_lambda: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for column_index, perturbation in enumerate(selected_perturbations):
        mask = (labels == ctrl_name) | (labels == perturbation)
        score_status = np.where(labels[mask] == ctrl_name, "control-zero", "optimized-active")
        frame = pd.DataFrame(
            {
                "row_id": row_ids[mask],
                "perturbation_label": labels[mask],
                "target_perturbation": perturbation,
                "ps_score": score_matrix[mask, column_index],
                "method": "ps_score_exact",
                "selected_target_gene_count": int(selected_target_gene_counts[str(perturbation)]),
                "score_status": score_status,
                "scale_factor": float(scale_factor),
                "score_lambda": float(score_lambda),
                "lr_lambda": float(lr_lambda),
            }
        )
        frames.append(frame.loc[:, RESULT_COLUMNS])
    if not frames:
        return _empty_result(return_wide=False)
    return pd.concat(frames, ignore_index=True)


def _build_wide_result(
    *,
    row_ids: np.ndarray,
    labels: np.ndarray,
    selected_perturbations: Sequence[str],
    score_matrix: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(score_matrix, index=pd.Index(row_ids, name="row_id"), columns=selected_perturbations)
    frame.insert(0, "perturbation_label", labels)
    return frame


def _empty_result(*, return_wide: bool) -> pd.DataFrame:
    if return_wide:
        return pd.DataFrame(columns=["perturbation_label"])
    return pd.DataFrame(columns=RESULT_COLUMNS)
