"""Shared matrix, stream, and small statistics helpers."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np
from scipy import sparse

from .types import BatchFactory, BatchSource, CsrBatch, StreamFeatureStats


SUPPORTED_FIDELITIES = ("exact", "approx")


def validate_fidelity(fidelity: str) -> str:
    if fidelity not in SUPPORTED_FIDELITIES:
        allowed = ", ".join(SUPPORTED_FIDELITIES)
        raise ValueError(f"Unsupported fidelity {fidelity!r}; expected one of: {allowed}")
    return fidelity


def validate_layer(layer: str | None) -> str | None:
    if layer is not None and not isinstance(layer, str):
        raise TypeError("layer must be a string or None")
    return layer


def validate_perturbations(perturbations: Sequence[str] | None) -> Sequence[str] | None:
    if perturbations is None:
        return None
    if isinstance(perturbations, str):
        raise TypeError("perturbations must be a sequence of labels, not a single string")
    return perturbations


def extract_anndata_matrix(adata: Any, *, layer: str | None = None) -> Any:
    """Return ``adata.X`` or the requested ``adata.layers`` entry."""

    validate_layer(layer)
    if adata is None or not hasattr(adata, "X") or not hasattr(adata, "layers"):
        raise TypeError("adata must provide X and layers attributes")
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise ValueError(f"Layer {layer!r} was not found in adata.layers")
    return adata.layers[layer]


def validate_csr_batch(batch: CsrBatch) -> CsrBatch:
    """Validate a ``CsrBatch`` beyond basic shape-length checks."""

    if not isinstance(batch, CsrBatch):
        raise TypeError("batch must be a CsrBatch instance")

    n_rows, n_cols = batch.shape
    indptr = np.asarray(batch.indptr, dtype=np.int64)
    indices = np.asarray(batch.indices, dtype=np.int64)

    if indptr.shape != (n_rows + 1,):
        raise ValueError("indptr length must equal shape[0] + 1")
    if np.any(np.diff(indptr) < 0):
        raise ValueError("indptr must be non-decreasing")
    if indices.shape != (len(batch.data),):
        raise ValueError("indices and data must have the same length")
    if indices.size and (indices.min() < 0 or indices.max() >= n_cols):
        raise ValueError("indices must be within the declared column range")
    return batch


def csr_batch_to_matrix(batch: CsrBatch) -> sparse.csr_matrix:
    """Convert a validated ``CsrBatch`` into a SciPy CSR matrix."""

    validate_csr_batch(batch)
    return sparse.csr_matrix(
        (
            np.asarray(batch.data, dtype=float),
            np.asarray(batch.indices, dtype=np.int64),
            np.asarray(batch.indptr, dtype=np.int64),
        ),
        shape=batch.shape,
    )


def iter_csr_batches(batches: BatchSource) -> list[CsrBatch] | Any:
    """Yield validated CSR batches from either an iterable or factory."""

    source = batches() if callable(batches) else batches
    for batch in source:
        yield validate_csr_batch(batch)


def require_reiterable_batches(batches: BatchSource, *, operation: str) -> BatchFactory:
    """Require a callable batch factory for multi-pass streamed work."""

    if callable(batches):
        return batches
    raise ValueError(
        f"{operation} requires a callable batch factory for multi-pass stream input"
    )


def get_obs_column(obs: Any, key: str) -> list[Any]:
    """Extract an observation column from a small set of direct input shapes."""

    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")

    if hasattr(obs, "columns"):
        if key not in obs.columns:
            raise KeyError(f"obs does not contain column {key!r}")
        return list(obs[key])

    if isinstance(obs, Mapping):
        if key not in obs:
            raise KeyError(f"obs does not contain key {key!r}")
        return list(obs[key])

    if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes)):
        values: list[Any] = []
        for row in obs:
            if not isinstance(row, Mapping) or key not in row:
                raise KeyError(f"obs rows must be mappings containing {key!r}")
            values.append(row[key])
        return values

    raise TypeError("obs must be a dataframe-like object, mapping, or sequence of mappings")


def get_obs_row_ids(obs: Any) -> list[Any]:
    """Extract stable row identifiers from streamed observation metadata."""

    if hasattr(obs, "index") and not callable(obs.index):
        return list(obs.index)

    if isinstance(obs, Mapping):
        for candidate in ("row_id", "row_ids", "index", "obs_names"):
            if candidate in obs:
                return list(obs[candidate])
        raise KeyError("obs mapping must contain row_id, row_ids, index, or obs_names")

    if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes)):
        row_ids: list[Any] = []
        for row in obs:
            if not isinstance(row, Mapping):
                raise TypeError("obs rows must be mappings when obs is a sequence")
            if "row_id" in row:
                row_ids.append(row["row_id"])
                continue
            if "index" in row:
                row_ids.append(row["index"])
                continue
            raise KeyError("obs rows must contain row_id or index")
        return row_ids

    raise TypeError("obs must be a dataframe-like object, mapping, or sequence of mappings")


def resolve_perturbations(
    labels: Sequence[Any],
    *,
    control_label: str,
    perturbations: Sequence[str] | None = None,
) -> list[Any]:
    """Resolve the perturbations that should be scored in deterministic order."""

    if not isinstance(control_label, str) or not control_label:
        raise ValueError("control_label must be a non-empty string")

    available: list[Any] = []
    available_set: set[Any] = set()
    for label in labels:
        if _is_missing_label(label) or label == control_label or label in available_set:
            continue
        available.append(label)
        available_set.add(label)

    requested = validate_perturbations(perturbations)
    if requested is None:
        return available

    selected: list[Any] = []
    seen: set[Any] = set()
    missing: list[str] = []
    for perturbation in requested:
        if perturbation == control_label:
            raise ValueError("control_label cannot be included in perturbations")
        if perturbation in seen:
            continue
        if perturbation not in available_set:
            missing.append(str(perturbation))
            continue
        selected.append(perturbation)
        seen.add(perturbation)

    if missing:
        raise ValueError(
            "Unknown perturbation labels requested: " + ", ".join(sorted(missing))
        )
    return selected


def summarize_matrix_features(matrix: Any) -> StreamFeatureStats:
    """Compute feature-wise count, sums, and squared sums for a matrix."""

    array_like = _ensure_2d_matrix(matrix)
    count = int(array_like.shape[0])
    if sparse.issparse(array_like):
        sums = np.asarray(array_like.sum(axis=0)).ravel().astype(float, copy=False)
        squared_sums = np.asarray(array_like.power(2).sum(axis=0)).ravel().astype(
            float,
            copy=False,
        )
    else:
        dense = np.asarray(array_like, dtype=float)
        sums = dense.sum(axis=0)
        squared_sums = np.square(dense).sum(axis=0)
    return StreamFeatureStats(count=count, sums=sums, squared_sums=squared_sums)


def summarize_streamed_features(
    batches: BatchSource,
    *,
    selected_row_ids: Collection[Any] | None = None,
) -> StreamFeatureStats:
    """Compute feature moments over streamed CSR batches."""

    selected = None if selected_row_ids is None else set(selected_row_ids)
    count = 0
    sums: np.ndarray | None = None
    squared_sums: np.ndarray | None = None

    for batch in iter_csr_batches(batches):
        matrix = csr_batch_to_matrix(batch)
        if selected is not None:
            mask = np.asarray([row_id in selected for row_id in batch.row_ids], dtype=bool)
            if not mask.any():
                continue
            matrix = matrix[mask]

        summary = summarize_matrix_features(matrix)
        if sums is None:
            sums = np.zeros(summary.n_features, dtype=float)
            squared_sums = np.zeros(summary.n_features, dtype=float)
        sums += summary.sums
        squared_sums += summary.squared_sums
        count += summary.count

    if sums is None or squared_sums is None:
        raise ValueError("stream did not yield any rows for summarization")
    return StreamFeatureStats(count=count, sums=sums, squared_sums=squared_sums)


def welch_t_scores(case_matrix: Any, control_matrix: Any) -> np.ndarray:
    """Compute per-feature Welch t statistics for ranking markers."""

    case = summarize_matrix_features(case_matrix)
    control = summarize_matrix_features(control_matrix)
    return welch_t_scores_from_stats(case, control)


def welch_t_scores_from_stats(case: StreamFeatureStats, control: StreamFeatureStats) -> np.ndarray:
    """Compute per-feature Welch t statistics from pre-aggregated moments."""

    if case.count == 0 or control.count == 0:
        raise ValueError("Welch ranking requires at least one case row and one control row")

    case_mean = case.means()
    control_mean = control.means()
    case_var = case.variances(ddof=1)
    control_var = control.variances(ddof=1)
    denominator = np.sqrt((case_var / max(case.count, 1)) + (control_var / max(control.count, 1)))
    diff = case_mean - control_mean

    scores = np.zeros_like(diff)
    valid = denominator > 0
    scores[valid] = diff[valid] / denominator[valid]
    degenerate = ~valid & (diff != 0)
    scores[degenerate] = np.sign(diff[degenerate]) * np.inf
    return scores


def top_k_indices(scores: Sequence[float], k: int, *, absolute: bool = True) -> np.ndarray:
    """Return stable top-k feature indices ordered by score strength."""

    if k < 1:
        raise ValueError("k must be at least 1")

    values = np.asarray(scores, dtype=float)
    order_values = np.abs(values) if absolute else values
    order = np.argsort(-order_values, kind="stable")
    return order[: min(k, values.shape[0])]


def rank_features_by_welch_t(
    case_matrix: Any,
    control_matrix: Any,
    *,
    top_k: int | None = None,
    absolute: bool = True,
) -> np.ndarray:
    """Rank features by Welch t statistic and optionally truncate to top-k."""

    scores = welch_t_scores(case_matrix, control_matrix)
    if top_k is None:
        return top_k_indices(scores, scores.shape[0], absolute=absolute)
    return top_k_indices(scores, top_k, absolute=absolute)


def _ensure_2d_matrix(matrix: Any) -> Any:
    if sparse.issparse(matrix):
        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        return matrix

    dense = np.asarray(matrix)
    if dense.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    return dense


def _is_missing_label(value: Any) -> bool:
    if value is None:
        return True
    return bool(isinstance(value, float) and np.isnan(value))
