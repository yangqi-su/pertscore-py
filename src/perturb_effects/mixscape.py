"""Compact AnnData Mixscape implementation with exact and approximate modes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logsumexp
from scipy.spatial.distance import cdist

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
    "perturbation_score",
    "posterior_probability",
    "class_label",
    "global_class_label",
    "fidelity",
    "method",
    "reference_mode",
    "marker_gene_count",
    "iteration_count",
]


def run_mixscape_anndata(
    adata: Any,
    *,
    layer: str | None = None,
    perturbation_key: str,
    control_label: str,
    fidelity: str = "exact",
    perturbations: Sequence[str] | None = None,
    n_jobs: int | None = 1,
    de_layer: str | None = None,
    n_neighbors: int = 20,
    marker_top_k: int | None = None,
    min_de_genes: int = 1,
    iter_num: int = 10,
    scale: bool = True,
    control_sample_size: int | None = None,
    perturbation_sample_size: int | None = None,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Score AnnData perturbations with a small Mixscape-style workflow."""

    if adata is None:
        raise ValueError("adata must not be None")
    if not isinstance(perturbation_key, str) or not perturbation_key:
        raise ValueError("perturbation_key must be a non-empty string")
    if not isinstance(control_label, str) or not control_label:
        raise ValueError("control_label must be a non-empty string")

    validate_layer(layer)
    validate_layer(de_layer)
    fidelity = validate_fidelity(fidelity)
    validate_perturbations(perturbations)
    normalize_n_jobs(n_jobs)
    _validate_positive_int("n_neighbors", n_neighbors)
    _validate_positive_int("min_de_genes", min_de_genes)
    _validate_positive_int("iter_num", iter_num)
    _validate_optional_positive_int("marker_top_k", marker_top_k)
    _validate_optional_positive_int("control_sample_size", control_sample_size)
    _validate_optional_positive_int("perturbation_sample_size", perturbation_sample_size)

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
    de_matrix = extract_anndata_matrix(adata, layer=de_layer) if de_layer is not None else matrix
    row_ids = np.asarray(adata.obs_names, dtype=object)
    var_names = np.asarray(adata.var_names, dtype=object)
    base_seed = 0 if random_state is None else int(random_state)

    worker = (
        lambda perturbation: _run_exact_for_perturbation(
            perturbation=perturbation,
            labels=labels,
            row_ids=row_ids,
            var_names=var_names,
            matrix=matrix,
            de_matrix=de_matrix,
            control_label=control_label,
            n_neighbors=n_neighbors,
            marker_top_k=marker_top_k,
            min_de_genes=min_de_genes,
            iter_num=iter_num,
            scale=scale,
        )
        if fidelity == "exact"
        else _run_approx_for_perturbation(
            perturbation=perturbation,
            labels=labels,
            row_ids=row_ids,
            var_names=var_names,
            matrix=matrix,
            de_matrix=de_matrix,
            control_label=control_label,
            marker_top_k=marker_top_k,
            min_de_genes=min_de_genes,
            scale=scale,
            control_sample_size=control_sample_size,
            perturbation_sample_size=perturbation_sample_size,
            seed=_stable_seed(base_seed, str(perturbation)),
        )
    )

    outputs = run_parallel_tasks(selected, worker, n_jobs=n_jobs)
    frames = [frame for frame, _ in outputs]
    metadata = {perturbation: meta for (_, meta), perturbation in zip(outputs, selected, strict=False)}
    result = pd.concat(frames, ignore_index=True) if frames else _empty_result_frame()
    result.attrs["mixscape"] = {
        "algorithm": "mixscape",
        "fidelity": fidelity,
        "layer": layer,
        "de_layer": de_layer if de_layer is not None else layer,
        "perturbation_key": perturbation_key,
        "control_label": control_label,
        "perturbations": list(selected),
        "metadata_by_perturbation": metadata,
    }
    return result


def run_mixscape_stream(
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
    """Validate the streamed Mixscape API contract for later phases."""

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
        "Mixscape streamed execution is not implemented yet; Phase 3 only adds the AnnData path."
    )


def _run_exact_for_perturbation(
    *,
    perturbation: Any,
    labels: np.ndarray,
    row_ids: np.ndarray,
    var_names: np.ndarray,
    matrix: Any,
    de_matrix: Any,
    control_label: str,
    n_neighbors: int,
    marker_top_k: int | None,
    min_de_genes: int,
    iter_num: int,
    scale: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    control_idx = np.flatnonzero(labels == control_label)
    target_idx = np.flatnonzero(labels == perturbation)
    _validate_cell_counts(control_idx, target_idx, perturbation=perturbation, control_label=control_label)

    marker_indices = _select_marker_indices(
        de_matrix=de_matrix,
        target_idx=target_idx,
        control_idx=control_idx,
        min_de_genes=min_de_genes,
        marker_top_k=marker_top_k,
    )
    combined_idx = np.flatnonzero((labels == control_label) | (labels == perturbation))
    combined_expr = _select_dense(matrix, combined_idx, marker_indices)
    control_expr = _select_dense(matrix, control_idx, marker_indices)
    signature = _knn_signature(combined_expr, control_expr, n_neighbors=n_neighbors)
    if scale:
        signature = _scale_columns(signature)

    control_pos = np.flatnonzero(labels[combined_idx] == control_label)
    target_pos = np.flatnonzero(labels[combined_idx] == perturbation)
    projections, posterior, target_mask, iterations = _iterative_exact_classification(
        signature=signature,
        control_pos=control_pos,
        target_pos=target_pos,
        iter_num=iter_num,
    )

    frame = _build_result_frame(
        row_ids=row_ids[combined_idx],
        perturbation_labels=labels[combined_idx],
        target_perturbation=str(perturbation),
        control_label=control_label,
        fidelity="exact",
        method="knn_signature_gmm",
        reference_mode="knn-control",
        marker_gene_count=marker_indices.size,
        iteration_count=iterations,
        projections=projections,
        posterior=posterior,
        target_pos=target_pos,
        target_mask=target_mask,
    )
    metadata = {
        "marker_genes": var_names[marker_indices].tolist(),
        "n_neighbors": min(n_neighbors, control_idx.size),
        "method": "knn_signature_gmm",
        "reference_mode": "knn-control",
        "iteration_count": iterations,
    }
    return frame, metadata


def _run_approx_for_perturbation(
    *,
    perturbation: Any,
    labels: np.ndarray,
    row_ids: np.ndarray,
    var_names: np.ndarray,
    matrix: Any,
    de_matrix: Any,
    control_label: str,
    marker_top_k: int | None,
    min_de_genes: int,
    scale: bool,
    control_sample_size: int | None,
    perturbation_sample_size: int | None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    control_idx = np.flatnonzero(labels == control_label)
    target_idx = np.flatnonzero(labels == perturbation)
    _validate_cell_counts(control_idx, target_idx, perturbation=perturbation, control_label=control_label)

    sampled_control_idx = _sample_indices(control_idx, control_sample_size, rng)
    sampled_target_idx = _sample_indices(target_idx, perturbation_sample_size, rng)
    marker_indices = _select_marker_indices(
        de_matrix=de_matrix,
        target_idx=sampled_target_idx,
        control_idx=sampled_control_idx,
        min_de_genes=min_de_genes,
        marker_top_k=marker_top_k if marker_top_k is not None else 20,
    )

    combined_idx = np.flatnonzero((labels == control_label) | (labels == perturbation))
    combined_expr = _select_dense(matrix, combined_idx, marker_indices)
    reference_expr = _select_dense(matrix, sampled_control_idx, marker_indices).mean(axis=0, keepdims=True)
    signature = np.repeat(reference_expr, combined_expr.shape[0], axis=0) - combined_expr
    if scale:
        signature = _scale_columns(signature)

    control_pos = np.flatnonzero(labels[combined_idx] == control_label)
    target_pos = np.flatnonzero(labels[combined_idx] == perturbation)
    projections, posterior, target_mask = _approximate_classification(
        signature=signature,
        control_pos=control_pos,
        target_pos=target_pos,
        reference_target_pos=np.flatnonzero(np.isin(combined_idx, sampled_target_idx)),
    )

    frame = _build_result_frame(
        row_ids=row_ids[combined_idx],
        perturbation_labels=labels[combined_idx],
        target_perturbation=str(perturbation),
        control_label=control_label,
        fidelity="approx",
        method="centroid_projection",
        reference_mode="control-centroid",
        marker_gene_count=marker_indices.size,
        iteration_count=1,
        projections=projections,
        posterior=posterior,
        target_pos=target_pos,
        target_mask=target_mask,
    )
    metadata = {
        "marker_genes": var_names[marker_indices].tolist(),
        "method": "centroid_projection",
        "reference_mode": "control-centroid",
        "control_sample_size": int(sampled_control_idx.size),
        "perturbation_sample_size": int(sampled_target_idx.size),
        "iteration_count": 1,
    }
    return frame, metadata


def _validate_cell_counts(
    control_idx: np.ndarray,
    target_idx: np.ndarray,
    *,
    perturbation: Any,
    control_label: str,
) -> None:
    if control_idx.size < 2:
        raise ValueError(f"Mixscape requires at least 2 {control_label!r} control cells")
    if target_idx.size < 2:
        raise ValueError(f"Mixscape requires at least 2 cells for perturbation {perturbation!r}")


def _select_marker_indices(
    *,
    de_matrix: Any,
    target_idx: np.ndarray,
    control_idx: np.ndarray,
    min_de_genes: int,
    marker_top_k: int | None,
) -> np.ndarray:
    target_expr = _select_dense(de_matrix, target_idx, None)
    control_expr = _select_dense(de_matrix, control_idx, None)
    scores = welch_t_scores(target_expr, control_expr)
    mean_diff = target_expr.mean(axis=0) - control_expr.mean(axis=0)
    ranking = np.argsort(-np.abs(scores), kind="stable")
    positive = ranking[mean_diff[ranking] > 0]
    selected = positive if positive.size else ranking
    if marker_top_k is not None:
        selected = selected[:marker_top_k]
    required = min(max(1, min_de_genes), target_expr.shape[1])
    if selected.size < required:
        raise ValueError(f"Mixscape found fewer than {required} marker genes for this perturbation")
    return np.asarray(selected, dtype=np.int64)


def _knn_signature(query_expr: np.ndarray, control_expr: np.ndarray, *, n_neighbors: int) -> np.ndarray:
    neighbor_count = min(max(1, n_neighbors), control_expr.shape[0])
    distances = cdist(query_expr, control_expr, metric="sqeuclidean")
    neighbor_order = np.argpartition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
    reference = control_expr[neighbor_order].mean(axis=1)
    return reference - query_expr


def _iterative_exact_classification(
    *,
    signature: np.ndarray,
    control_pos: np.ndarray,
    target_pos: np.ndarray,
    iter_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    control_mean = signature[control_pos].mean(axis=0)
    current_mask = np.ones(target_pos.shape[0], dtype=bool)
    last_mask: np.ndarray | None = None
    projections = np.zeros(signature.shape[0], dtype=float)
    posterior = np.zeros(signature.shape[0], dtype=float)
    iterations = 0

    for iteration in range(iter_num):
        iterations = iteration + 1
        if not current_mask.any():
            break
        guide_rows = target_pos[current_mask]
        guide_mean = signature[guide_rows].mean(axis=0)
        vec = guide_mean - control_mean
        projections = _project_signature(signature, vec)
        posterior = _fit_fixed_reference_gmm(projections, control_pos=control_pos, target_pos=target_pos)
        next_mask = posterior[target_pos] > 0.5
        if last_mask is not None and np.array_equal(next_mask, last_mask):
            current_mask = next_mask
            break
        last_mask = current_mask
        current_mask = next_mask

    return projections, posterior, current_mask, iterations


def _approximate_classification(
    *,
    signature: np.ndarray,
    control_pos: np.ndarray,
    target_pos: np.ndarray,
    reference_target_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reference_target_pos.size == 0:
        reference_target_pos = target_pos
    control_mean = signature[control_pos].mean(axis=0)
    target_mean = signature[reference_target_pos].mean(axis=0)
    projections = _project_signature(signature, target_mean - control_mean)

    control_scores = projections[control_pos]
    target_scores = projections[target_pos]
    threshold = 0.5 * (control_scores.mean() + target_scores.mean())
    spread = max(
        float(np.std(control_scores, ddof=1)) if control_scores.size > 1 else 0.0,
        float(np.std(target_scores, ddof=1)) if target_scores.size > 1 else 0.0,
        1e-6,
    )
    posterior = expit((projections - threshold) / spread)
    return projections, posterior, posterior[target_pos] > 0.5


def _project_signature(signature: np.ndarray, vec: np.ndarray) -> np.ndarray:
    norm = float(np.dot(vec, vec))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.zeros(signature.shape[0], dtype=float)
    projections = (signature @ vec) / norm
    return projections.astype(float, copy=False)


def _fit_fixed_reference_gmm(
    scores: np.ndarray,
    *,
    control_pos: np.ndarray,
    target_pos: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(scores, dtype=float)
    if x[target_pos].mean() < x[control_pos].mean():
        x = -x

    control_scores = x[control_pos]
    target_scores = x[target_pos]
    mu0 = float(control_scores.mean())
    var0 = max(float(np.var(control_scores, ddof=1)) if control_scores.size > 1 else 0.0, 1e-6)
    mu1 = float(target_scores.mean())
    var1 = max(float(np.var(target_scores, ddof=1)) if target_scores.size > 1 else 0.0, var0, 1e-6)
    weight1 = min(max(target_scores.size / max(x.size, 1), 1e-3), 1.0 - 1e-3)

    for _ in range(max_iter):
        log_p0 = np.log1p(-weight1) + _log_normal_pdf(x, mu0, var0)
        log_p1 = np.log(weight1) + _log_normal_pdf(x, mu1, var1)
        log_norm = logsumexp(np.column_stack([log_p0, log_p1]), axis=1)
        resp1 = np.exp(log_p1 - log_norm)

        total1 = max(float(resp1.sum()), 1e-6)
        new_weight1 = min(max(total1 / x.size, 1e-3), 1.0 - 1e-3)
        new_mu1 = float(np.sum(resp1 * x) / total1)
        new_var1 = max(float(np.sum(resp1 * np.square(x - new_mu1)) / total1), 1e-6)

        if (
            abs(new_weight1 - weight1) < tol
            and abs(new_mu1 - mu1) < tol
            and abs(new_var1 - var1) < tol
        ):
            weight1 = new_weight1
            mu1 = new_mu1
            var1 = new_var1
            break

        weight1 = new_weight1
        mu1 = new_mu1
        var1 = new_var1

    log_p0 = np.log1p(-weight1) + _log_normal_pdf(x, mu0, var0)
    log_p1 = np.log(weight1) + _log_normal_pdf(x, mu1, var1)
    log_norm = logsumexp(np.column_stack([log_p0, log_p1]), axis=1)
    return np.exp(log_p1 - log_norm)


def _build_result_frame(
    *,
    row_ids: np.ndarray,
    perturbation_labels: np.ndarray,
    target_perturbation: str,
    control_label: str,
    fidelity: str,
    method: str,
    reference_mode: str,
    marker_gene_count: int,
    iteration_count: int,
    projections: np.ndarray,
    posterior: np.ndarray,
    target_pos: np.ndarray,
    target_mask: np.ndarray,
) -> pd.DataFrame:
    target_row_mask = np.zeros(row_ids.shape[0], dtype=bool)
    target_row_mask[target_pos] = True
    class_labels = np.full(row_ids.shape[0], control_label, dtype=object)
    global_labels = np.full(row_ids.shape[0], control_label, dtype=object)

    target_class_labels = np.where(
        target_mask,
        f"{target_perturbation} KO",
        f"{target_perturbation} NP",
    )
    class_labels[target_pos] = target_class_labels
    global_labels[target_pos] = np.where(target_mask, "KO", "NP")

    frame = pd.DataFrame(
        {
            "row_id": row_ids.astype(str),
            "perturbation_label": perturbation_labels.astype(str),
            "target_perturbation": target_perturbation,
            "perturbation_score": projections,
            "posterior_probability": posterior,
            "class_label": class_labels,
            "global_class_label": global_labels,
            "fidelity": fidelity,
            "method": method,
            "reference_mode": reference_mode,
            "marker_gene_count": int(marker_gene_count),
            "iteration_count": int(iteration_count),
        }
    )
    frame.loc[~target_row_mask, "posterior_probability"] = 0.0
    return frame[RESULT_COLUMNS]


def _select_dense(matrix: Any, row_idx: np.ndarray, col_idx: np.ndarray | None) -> np.ndarray:
    selected = matrix[row_idx] if col_idx is None else matrix[row_idx][:, col_idx]
    if sparse.issparse(selected):
        return selected.toarray().astype(float, copy=False)
    return np.asarray(selected, dtype=float)


def _scale_columns(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, ddof=1, keepdims=True)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return (values - means) / std


def _sample_indices(indices: np.ndarray, sample_size: int | None, rng: np.random.Generator) -> np.ndarray:
    if sample_size is None or sample_size >= indices.size:
        return indices
    selected = np.sort(rng.choice(indices, size=sample_size, replace=False))
    return selected.astype(np.int64, copy=False)


def _stable_seed(base_seed: int, label: str) -> int:
    return int(base_seed + sum((offset + 1) * ord(char) for offset, char in enumerate(label)))


def _log_normal_pdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return -0.5 * (np.log(2.0 * np.pi * variance) + np.square(x - mean) / variance)


def _empty_result_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=object) for column in RESULT_COLUMNS})


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_optional_positive_int(name: str, value: int | None) -> None:
    if value is None:
        return
    _validate_positive_int(name, value)
