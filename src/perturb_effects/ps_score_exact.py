"""In-memory sparse scMAGeCK/R-like PS score reference path.

This implementation keeps the full selected AnnData matrix in memory, converts it
to CSR, solves ridge beta from sparse sufficient statistics, and optimizes PS
scores with grouped L-BFGS-B. It is intended as a simple R-like reference path;
large production runs should use ``ps_score_exact_fast``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

from .stats import solve_ridge_beta
from .stream import as_csr_matrix, clip_sparse_columns_by_quantile, extract_anndata_matrix
from .utils import (
    PERTURBATION_DELIMITER,
    background_cluster_codes as parse_background_cluster_codes,
    clean_obs_labels,
    group_rows_by_active_set,
    parse_perturbation_labels,
    ps_score_long_dataframe,
)


SUPPORTED_TARGET_GENE_SOURCES = ("provided", "scanpy_de", "hvg")
SCANPY_DE_METHOD = "wilcoxon"
SCANPY_DE_LOGFC_THRESHOLD = 0.1
SCANPY_DE_LOGFC_THRESHOLD_DECAY = 0.8
SCANPY_DE_MAX_LOGFC_ROUNDS = 3


def run_ps_score_exact_anndata(
    adata: Any,
    *,
    perturb_column: str,
    ctrl_name: str,
    layer: str | None = None,
    counts_layer: str | None = "counts",
    perturbations: Sequence[str] | None = None,
    target_genes: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    target_gene_source: str = "hvg",
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
    background_cluster_column: str | None = None,
) -> pd.DataFrame:
    """Run the sparse in-memory R-like exact PS-score workflow on AnnData."""

    if target_gene_source not in SUPPORTED_TARGET_GENE_SOURCES:
        raise ValueError(f"Unsupported target_gene_source {target_gene_source!r}")
    if target_gene_source == "provided" and target_genes is None:
        raise ValueError("target_genes must be provided when target_gene_source='provided'")
    if target_gene_source != "provided" and target_genes is not None:
        raise ValueError("target_genes must be None unless target_gene_source='provided'")

    labels_all = clean_obs_labels(adata, perturb_column)
    obs_index = np.asarray(adata.obs_names, dtype=object)
    if not np.any(labels_all == ctrl_name):
        raise ValueError(f"ctrl_name {ctrl_name!r} was not found in adata.obs[{perturb_column!r}]")

    parsed = parse_perturbation_labels(labels_all, mode="multilabel", ctrl_name=ctrl_name, perturbations=perturbations)
    model_indices = np.flatnonzero(parsed.model_mask)
    model_guides = parsed.guides[model_indices].tocsr()
    model_control_mask = parsed.control_mask[model_indices]

    var_names = np.asarray(adata.var_names, dtype=object)
    gene_lookup = {str(gene): index for index, gene in enumerate(var_names.astype(str))}
    expression_matrix_raw = extract_anndata_matrix(adata, layer=layer)
    expression_matrix_format = "sparse" if sparse.issparse(expression_matrix_raw) else "dense"
    expression_matrix = as_csr_matrix(expression_matrix_raw)
    if apply_gene_filter and counts_layer is not None and counts_layer in adata.layers:
        filter_matrix, filter_source = as_csr_matrix(adata.layers[counts_layer]), counts_layer
    else:
        filter_matrix = expression_matrix
        filter_source = "none" if not apply_gene_filter else ("adata.X" if layer is None else layer)
    cluster_codes, cluster_names = parse_background_cluster_codes(adata, background_cluster_column)

    stage_timings: dict[str, float] = {}

    stage_start = time.perf_counter()
    if target_gene_source == "provided":
        genes_by_perturbation = _resolve_provided_target_genes(
            target_genes=target_genes,
            selected_perturbations=parsed.perturbations,
            gene_lookup=gene_lookup,
            target_gene_min=target_gene_min,
            target_gene_max=target_gene_max,
        )
        source_metadata = {"mode": "provided"}
    elif target_gene_source == "hvg":
        genes_by_perturbation = _resolve_hvg_target_genes(
            adata,
            hvg_key=hvg_key,
            selected_perturbations=parsed.perturbations,
            gene_lookup=gene_lookup,
            target_gene_min=target_gene_min,
            target_gene_max=target_gene_max,
        )
        source_metadata = {"mode": "hvg", "hvg_key": hvg_key}
    else:
        genes_by_perturbation = _resolve_scanpy_target_genes(
            adata,
            parsed=parsed,
            ctrl_name=ctrl_name,
            layer=layer,
            gene_lookup=gene_lookup,
            target_gene_min=target_gene_min,
            target_gene_max=target_gene_max,
        )
        source_metadata = {
            "mode": "scanpy_de",
            "layer": layer,
            "method": SCANPY_DE_METHOD,
            "logfc_threshold": SCANPY_DE_LOGFC_THRESHOLD,
            "logfc_threshold_decay": SCANPY_DE_LOGFC_THRESHOLD_DECAY,
            "max_logfc_rounds": SCANPY_DE_MAX_LOGFC_ROUNDS,
            "direction": "both",
            "rank_by": "pvals",
        }
    filter_metadata = _filter_target_genes(
        filter_matrix=filter_matrix,
        control_mask=parsed.control_mask,
        guides=parsed.guides,
        selected_perturbations=parsed.perturbations,
        genes_by_perturbation=genes_by_perturbation,
        gene_lookup=gene_lookup,
        target_gene_min=target_gene_min,
        gene_filter_min_fraction=gene_filter_min_fraction,
        apply_gene_filter=apply_gene_filter,
        require_min_genes=target_gene_source != "scanpy_de",
    )
    stage_timings["target_gene_selection"] = float(time.perf_counter() - stage_start)
    filtered_genes_by_perturbation = filter_metadata["genes_by_perturbation"]
    union_genes: list[str] = []
    seen_genes: set[str] = set()
    for perturbation in parsed.perturbations:
        for gene in filtered_genes_by_perturbation[perturbation]:
            if gene not in seen_genes:
                union_genes.append(gene)
                seen_genes.add(gene)
    if not union_genes:
        raise ValueError("No target genes left after target gene selection and filtering")
    union_gene_indices = np.asarray([gene_lookup[gene] for gene in union_genes], dtype=np.int64)

    stage_start = time.perf_counter()
    clip_values: np.ndarray | None = None
    background_matrix: np.ndarray | None = None
    background_control_counts: np.ndarray | None = None

    y_matrix = expression_matrix[model_indices][:, union_gene_indices].tocsr().astype(np.float64, copy=False)
    if apply_quantile_clip:
        y_matrix, clip_values = clip_sparse_columns_by_quantile(y_matrix, clip_quantile)

    design = sparse.hstack(
        [sparse.csr_matrix(np.ones((model_guides.shape[0], 1), dtype=np.float64)), model_guides],
        format="csr",
        dtype=np.float64,
    )
    beta = solve_ridge_beta(design, y_matrix, lr_lambda)

    if cluster_codes is not None:
        background_matrix, background_control_counts = _cluster_background_matrix(
            y_matrix,
            control_mask=model_control_mask,
            cluster_codes=cluster_codes[model_indices],
            cluster_names=cluster_names,
        )
        beta = beta.copy()
        beta[0] = 0.0
    stage_timings["beta_solve"] = float(time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    score_values, cell_indices, perturbation_indices, valid_mask = _score_grouped_lbfgsb(
        y_matrix=y_matrix,
        model_indices=model_indices,
        model_guides=model_guides,
        beta=beta,
        perturbations=parsed.perturbations,
        obs_count=obs_index.shape[0],
        score_lambda=score_lambda,
        scale_factor=scale_factor,
        scale_score=scale_score,
        cluster_codes=None if cluster_codes is None else cluster_codes[model_indices],
        background_matrix=background_matrix,
    )
    result = ps_score_long_dataframe(
        obs_index=obs_index,
        control_mask=parsed.control_mask,
        valid_mask=valid_mask,
        scores=score_values,
        cell_indices=cell_indices,
        perturbation_indices=perturbation_indices,
        perturbations=parsed.perturbations,
        ctrl_name=ctrl_name,
        missing_perturbation="",
    )
    stage_timings["scoring"] = float(time.perf_counter() - stage_start)

    metadata = {
        "algorithm": "ps_score_exact",
        "input_type": "anndata-r-like-sparse",
        "layer": layer,
        "counts_layer": counts_layer,
        "perturb_column": perturb_column,
        "ctrl_name": ctrl_name,
        "perturbation_delimiter": PERTURBATION_DELIMITER,
        "perturbations": list(parsed.perturbations),
        "target_gene_source": target_gene_source,
        "target_gene_source_detail": source_metadata,
        "expression_matrix_format": expression_matrix_format,
        "computation_path": "in_memory_sparse_lbfgsb",
        "sparse_fallback_reason": None,
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
        "background_correction": bool(cluster_codes is not None),
        "background_correction_mode": "r_like_score_correction" if cluster_codes is not None else None,
        "background_cluster_column": background_cluster_column,
        "lr_lambda": float(lr_lambda),
        "score_lambda": float(score_lambda),
        "scale_factor": float(scale_factor),
        "scale_score": bool(scale_score),
        "x_shape": (int(model_guides.shape[0]), int(model_guides.shape[1] + 1)),
        "y_shape": tuple(int(value) for value in y_matrix.shape),
        "beta_shape": tuple(int(value) for value in beta.shape),
        "model_cell_count": int(model_indices.shape[0]),
        "score_output_format": "csv_long",
        "stage_timings": stage_timings,
    }
    if background_control_counts is not None:
        metadata["background_cluster_count"] = int(len(cluster_names))
        metadata["background_control_cell_counts"] = {
            name: int(background_control_counts[index]) for index, name in enumerate(cluster_names)
        }
    result.attrs["ps_score_exact"] = metadata
    return result


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
        provided = target_genes.get(perturbation, target_genes.get(str(perturbation))) if isinstance(target_genes, Mapping) else target_genes
        if provided is None:
            raise ValueError(f"No target genes were provided for perturbation {perturbation!r}")
        genes = _normalize_gene_names(provided)[:target_gene_max]
        if len(genes) < target_gene_min:
            raise ValueError(f"Need at least {target_gene_min} target genes for perturbation {perturbation!r}")
        missing = [gene for gene in genes if gene not in gene_lookup]
        if missing:
            raise ValueError(f"Unknown target genes requested for perturbation {perturbation!r}: " + ", ".join(sorted(missing)))
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
    if np.issubdtype(values.dtype, np.bool_):
        selected_indices = np.flatnonzero(values)
    else:
        numeric = pd.to_numeric(pd.Series(values), errors="coerce")
        valid_mask = ~numeric.isna().to_numpy()
        if not valid_mask.any():
            raise ValueError(f"adata.var[{hvg_key!r}] does not contain any HVG entries")
        valid_indices = np.flatnonzero(valid_mask)
        selected_indices = valid_indices[np.argsort(numeric.to_numpy()[valid_indices], kind="stable")]
    if selected_indices.size == 0:
        raise ValueError(f"adata.var[{hvg_key!r}] does not contain any HVGs")
    genes = [str(adata.var_names[index]) for index in selected_indices[:target_gene_max]]
    if len(genes) < target_gene_min:
        raise ValueError(f"Need at least {target_gene_min} HVGs from adata.var[{hvg_key!r}]")
    missing = [gene for gene in genes if gene not in gene_lookup]
    if missing:
        raise ValueError("Resolved HVG genes were missing from adata.var_names: " + ", ".join(sorted(missing)))
    return {str(perturbation): list(genes) for perturbation in selected_perturbations}


def _resolve_scanpy_target_genes(
    adata: Any,
    *,
    parsed: Any,
    ctrl_name: str,
    layer: str | None,
    gene_lookup: Mapping[str, int],
    target_gene_min: int,
    target_gene_max: int,
    de_method: str = SCANPY_DE_METHOD,
    logfc_threshold: float = SCANPY_DE_LOGFC_THRESHOLD,
    logfc_threshold_decay: float = SCANPY_DE_LOGFC_THRESHOLD_DECAY,
    max_logfc_rounds: int = SCANPY_DE_MAX_LOGFC_ROUNDS,
) -> dict[str, list[str]]:
    import scanpy as sc

    genes_by_perturbation: dict[str, list[str]] = {}
    thresholds = [
        float(logfc_threshold) * (float(logfc_threshold_decay) ** round_index)
        for round_index in range(max_logfc_rounds)
    ]
    for perturbation_index, perturbation in enumerate(parsed.perturbations):
        active = np.asarray(parsed.guides[:, perturbation_index].toarray()).ravel() > 0
        subset_mask = parsed.control_mask | active
        subset = adata[subset_mask].copy()
        subset.obs["_ps_score_exact_de_group"] = np.where(active[subset_mask], "target", ctrl_name)
        sc.tl.rank_genes_groups(
            subset,
            groupby="_ps_score_exact_de_group",
            groups=["target"],
            reference=ctrl_name,
            use_raw=False,
            layer=layer,
            n_genes=subset.n_vars,
            method=de_method,
        )
        de_frame = sc.get.rank_genes_groups_df(subset, group="target")
        known_gene = de_frame["names"].isin(gene_lookup)
        de_frame = de_frame[known_gene].copy()
        de_frame["_abs_logfc"] = pd.to_numeric(de_frame["logfoldchanges"], errors="coerce").abs()
        selected = de_frame.iloc[0:0]
        for threshold in thresholds:
            selected = de_frame[de_frame["_abs_logfc"] > threshold]
            if selected.shape[0] >= target_gene_min:
                break
        selected = selected.sort_values("pvals", kind="stable")
        genes = _normalize_gene_names(selected["names"].tolist())[:target_gene_max]
        genes_by_perturbation[str(perturbation)] = genes
    return genes_by_perturbation


def _filter_target_genes(
    *,
    filter_matrix: sparse.csr_matrix,
    control_mask: np.ndarray,
    guides: sparse.csr_matrix,
    selected_perturbations: Sequence[str],
    genes_by_perturbation: Mapping[str, Sequence[str]],
    gene_lookup: Mapping[str, int],
    target_gene_min: int,
    gene_filter_min_fraction: float,
    apply_gene_filter: bool,
    require_min_genes: bool = True,
) -> dict[str, Any]:
    filtered: dict[str, list[str]] = {}
    counts_before: dict[str, int] = {}
    counts_after: dict[str, int] = {}
    for perturbation_index, perturbation in enumerate(selected_perturbations):
        genes = list(genes_by_perturbation[str(perturbation)])
        counts_before[str(perturbation)] = len(genes)
        if not apply_gene_filter:
            filtered[str(perturbation)] = genes
            counts_after[str(perturbation)] = len(genes)
            continue
        active = np.asarray(guides[:, perturbation_index].toarray()).ravel() > 0
        relevant_rows = np.flatnonzero(control_mask | active)
        gene_indices = np.asarray([gene_lookup[gene] for gene in genes], dtype=np.int64)
        candidate = filter_matrix[relevant_rows][:, gene_indices]
        keep_fraction = np.asarray(candidate.getnnz(axis=0), dtype=float).ravel() / float(candidate.shape[0])
        kept = [gene for gene, keep in zip(genes, keep_fraction >= gene_filter_min_fraction, strict=False) if keep]
        if require_min_genes and len(kept) < target_gene_min:
            raise ValueError(f"Gene filtering left fewer than {target_gene_min} target genes for perturbation {perturbation!r}")
        filtered[str(perturbation)] = kept
        counts_after[str(perturbation)] = len(kept)
    return {
        "genes_by_perturbation": filtered,
        "target_gene_counts_before_filter": counts_before,
        "target_gene_counts_after_filter": counts_after,
    }


def _cluster_background_matrix(
    y_matrix: sparse.csr_matrix,
    *,
    control_mask: np.ndarray,
    cluster_codes: np.ndarray,
    cluster_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    needed_clusters = np.unique(cluster_codes)
    control_counts = np.zeros(len(cluster_names), dtype=np.int64)
    background = np.zeros((len(cluster_names), y_matrix.shape[1]), dtype=np.float64)
    for cluster in needed_clusters:
        cluster_control = control_mask & (cluster_codes == cluster)
        control_counts[int(cluster)] = int(np.count_nonzero(cluster_control))
        if control_counts[int(cluster)] == 0:
            raise ValueError("background correction requires control cells in each modeled cluster: " + cluster_names[int(cluster)])
        background[int(cluster)] = np.asarray(y_matrix[cluster_control].sum(axis=0), dtype=np.float64).ravel() / float(control_counts[int(cluster)])
    return background, control_counts


def _score_grouped_lbfgsb(
    *,
    y_matrix: sparse.csr_matrix,
    model_indices: np.ndarray,
    model_guides: sparse.csr_matrix,
    beta: np.ndarray,
    perturbations: Sequence[str],
    obs_count: int,
    score_lambda: float,
    scale_factor: float,
    scale_score: bool,
    cluster_codes: np.ndarray | None,
    background_matrix: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score_values: list[np.ndarray] = []
    cell_index_values: list[np.ndarray] = []
    perturbation_index_values: list[np.ndarray] = []
    max_score_by_perturbation = np.zeros(len(perturbations), dtype=np.float64)
    valid_mask = np.zeros(obs_count, dtype=bool)

    for active_set, local_rows in group_rows_by_active_set(model_guides).items():
        active_indices = np.asarray(active_set, dtype=np.int64)
        active_beta = beta[active_indices + 1]
        gram = active_beta @ active_beta.T
        rhs = np.asarray(y_matrix[local_rows] @ active_beta.T, dtype=np.float64)
        if background_matrix is None:
            rhs -= (beta[0] @ active_beta.T)[None, :]
        else:
            if cluster_codes is None:
                raise ValueError("background scoring requires cluster codes")
            rhs -= background_matrix[cluster_codes[local_rows]] @ active_beta.T

        bounded = _solve_bounded_quadratic_lbfgsb(
            gram=gram,
            rhs=rhs,
            linear_penalty=float(score_lambda),
            upper=float(scale_factor),
        )
        normalized = bounded / float(scale_factor)
        global_rows = model_indices[local_rows]
        valid_mask[global_rows] = True

        for offset, perturbation_index in enumerate(active_indices):
            values = normalized[:, offset].astype(np.float64, copy=False)
            score_values.append(values)
            cell_index_values.append(global_rows.copy())
            perturbation_index_values.append(np.full(values.shape[0], int(perturbation_index), dtype=np.int32))
            if values.size:
                max_score_by_perturbation[perturbation_index] = max(max_score_by_perturbation[perturbation_index], float(np.max(values)))

    if score_values:
        scores = np.concatenate(score_values).astype(np.float64, copy=False)
        cell_indices = np.concatenate(cell_index_values).astype(np.int64, copy=False)
        perturbation_indices = np.concatenate(perturbation_index_values).astype(np.int32, copy=False)
    else:
        scores = np.zeros(0, dtype=np.float64)
        cell_indices = np.zeros(0, dtype=np.int64)
        perturbation_indices = np.zeros(0, dtype=np.int32)

    if scale_score and scores.size:
        row_max = max_score_by_perturbation[perturbation_indices]
        nonzero = row_max > 0.0
        scores[nonzero] /= row_max[nonzero]
        scores[~nonzero] = 0.0

    return scores, cell_indices, perturbation_indices, valid_mask


def _solve_bounded_quadratic_lbfgsb(
    *,
    gram: np.ndarray,
    rhs: np.ndarray,
    linear_penalty: float,
    upper: float,
) -> np.ndarray:
    rhs = np.asarray(rhs, dtype=np.float64)
    if rhs.ndim == 1:
        rhs = rhs[:, None]
    row_count, variable_count = rhs.shape
    initial = np.ones((row_count, variable_count), dtype=np.float64)
    initial = np.clip(initial, 0.0, upper)

    def objective(flat_scores: np.ndarray) -> float:
        scores = flat_scores.reshape(row_count, variable_count)
        return float(
            0.5 * np.sum((scores @ gram) * scores)
            - np.sum(rhs * scores)
            + linear_penalty * np.sum(scores)
        )

    def gradient(flat_scores: np.ndarray) -> np.ndarray:
        scores = flat_scores.reshape(row_count, variable_count)
        return (scores @ gram - rhs + linear_penalty).ravel()

    result = minimize(
        objective,
        initial.ravel(),
        jac=gradient,
        bounds=[(0.0, upper)] * initial.size,
        method="L-BFGS-B",
    )
    scores = np.clip(result.x.reshape(row_count, variable_count), 0.0, upper)
    return scores


__all__ = ["run_ps_score_exact_anndata"]
