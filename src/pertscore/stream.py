"""Shared matrix streaming, normalization, and clipping helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from scipy import sparse
from tqdm import tqdm


def extract_anndata_matrix(adata: Any, *, layer: str | None = None) -> Any:
    return adata.X if layer is None else adata.layers[layer]


def as_csr_matrix(matrix: Any) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float64, copy=False)
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))


def iter_matrix_chunks(
    matrix: Any,
    *,
    n_obs: int,
    chunk_size: int,
    target_sum: float,
    gene_indices: np.ndarray | None = None,
    clip_values: np.ndarray | None = None,
    apply_log1p: bool = True,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> Iterator[tuple[int, int, Any]]:
    progress = tqdm(total=n_obs, desc=progress_desc, unit="cells") if show_progress else None
    try:
        for start in range(0, n_obs, chunk_size):
            stop = min(start + chunk_size, n_obs)
            chunk = matrix[start:stop]
            chunk = log_normalize_chunk(chunk, target_sum=target_sum) if apply_log1p else normalize_chunk(chunk, target_sum=target_sum)
            if gene_indices is not None:
                chunk = chunk[:, gene_indices]
            if clip_values is not None:
                chunk = clip_matrix_columns(chunk, clip_values)
            yield start, stop, chunk
            if progress is not None:
                progress.update(stop - start)
    finally:
        if progress is not None:
            progress.close()


def normalize_chunk(matrix: Any, *, target_sum: float) -> Any:
    if sparse.issparse(matrix):
        work = matrix.tocsr(copy=True).astype(np.float64)
        totals = np.asarray(work.sum(axis=1)).ravel()
        scales = np.zeros(work.shape[0], dtype=np.float64)
        nonzero = totals > 0
        scales[nonzero] = target_sum / totals[nonzero]
        return work.multiply(scales[:, None]).tocsr()

    dense = np.asarray(matrix, dtype=np.float64).copy()
    totals = dense.sum(axis=1, keepdims=True)
    nonzero = totals[:, 0] > 0
    dense[nonzero] *= target_sum / totals[nonzero]
    dense[~nonzero] = 0.0
    return dense


def log_normalize_chunk(matrix: Any, *, target_sum: float) -> Any:
    return apply_log1p_chunk(normalize_chunk(matrix, target_sum=target_sum))


def apply_log1p_chunk(matrix: Any) -> Any:
    if sparse.issparse(matrix):
        matrix.data = np.log1p(matrix.data)
        return matrix
    np.log1p(matrix, out=matrix)
    return matrix


def accumulate_nonzero_histogram(matrix: Any, *, hist: np.ndarray, nonzero_counts: np.ndarray, max_value: float) -> None:
    if sparse.issparse(matrix):
        coo = matrix.tocoo()
        positive = coo.data > 0.0
        if not np.any(positive):
            return
        columns = coo.col[positive]
        nonzero_counts += np.bincount(columns, minlength=hist.shape[0]).astype(np.int64, copy=False)
        np.add.at(hist, (columns, histogram_bin_indices(coo.data[positive], bins=hist.shape[1], max_value=max_value)), 1)
        return

    dense = np.asarray(matrix, dtype=np.float64)
    nonzero_rows, nonzero_cols = np.nonzero(dense > 0.0)
    if nonzero_cols.size:
        nonzero_counts += np.bincount(nonzero_cols, minlength=hist.shape[0]).astype(np.int64, copy=False)
        np.add.at(hist, (nonzero_cols, histogram_bin_indices(dense[nonzero_rows, nonzero_cols], bins=hist.shape[1], max_value=max_value)), 1)


def histogram_bin_indices(values: np.ndarray, *, bins: int, max_value: float) -> np.ndarray:
    return np.clip(np.floor((np.asarray(values, dtype=np.float64) / max_value) * bins).astype(np.int64, copy=False), 0, bins - 1)


def histogram_quantiles(hist: np.ndarray, *, zero_counts: np.ndarray, total_count: int, quantile: float, max_value: float) -> np.ndarray:
    edges = (np.arange(1, hist.shape[1] + 1, dtype=np.float64) * max_value) / float(hist.shape[1])
    position = float(total_count - 1) * quantile
    lower_rank = int(np.floor(position))
    upper_rank = int(np.ceil(position))
    fraction = position - float(lower_rank)
    clip_values = np.zeros(hist.shape[0], dtype=np.float64)
    for gene_index in range(hist.shape[0]):
        lower = histogram_value_at_rank(hist[gene_index], zero_count=int(zero_counts[gene_index]), rank=lower_rank, edges=edges)
        upper = histogram_value_at_rank(hist[gene_index], zero_count=int(zero_counts[gene_index]), rank=upper_rank, edges=edges)
        clip_values[gene_index] = lower + fraction * (upper - lower)
    return clip_values


def histogram_value_at_rank(hist_row: np.ndarray, *, zero_count: int, rank: int, edges: np.ndarray) -> float:
    if rank < zero_count:
        return 0.0
    cumulative = np.cumsum(hist_row, dtype=np.int64)
    if cumulative.size == 0 or cumulative[-1] == 0:
        return 0.0
    bin_index = int(np.searchsorted(cumulative, rank - zero_count + 1, side="left"))
    return float(edges[min(bin_index, edges.shape[0] - 1)])


def clip_values_from_histogram(
    *,
    hist: np.ndarray,
    nonzero_counts: np.ndarray,
    gene_indices: np.ndarray,
    model_cell_count: int,
    quantile: float,
    max_value: float,
) -> np.ndarray:
    zero_counts = np.full(gene_indices.shape[0], model_cell_count, dtype=np.int64) - nonzero_counts[gene_indices]
    return histogram_quantiles(
        hist[gene_indices],
        zero_counts=zero_counts,
        total_count=model_cell_count,
        quantile=quantile,
        max_value=max_value,
    )


def clip_matrix_columns(matrix: Any, clip_values: np.ndarray) -> Any:
    if sparse.issparse(matrix):
        work = matrix.tocsr(copy=True).astype(np.float64)
        if work.data.size:
            work.data = np.minimum(work.data, clip_values[work.indices])
            work.eliminate_zeros()
        return work
    return np.minimum(np.asarray(matrix, dtype=np.float64).copy(), clip_values[None, :])


def clip_sparse_columns_by_quantile(matrix: sparse.csr_matrix, quantile: float) -> tuple[sparse.csr_matrix, np.ndarray]:
    clip_values = sparse_column_quantiles(matrix, quantile)
    return clip_matrix_columns(matrix, clip_values), clip_values


def sparse_column_quantiles(matrix: sparse.csr_matrix, quantile: float) -> np.ndarray:
    csc = matrix.tocsc()
    values = np.zeros(csc.shape[1], dtype=np.float64)
    for column_index in range(csc.shape[1]):
        start, stop = csc.indptr[column_index], csc.indptr[column_index + 1]
        nonzero = np.asarray(csc.data[start:stop], dtype=np.float64)
        zero_count = csc.shape[0] - nonzero.shape[0]
        column = nonzero if zero_count == 0 else np.concatenate([np.zeros(zero_count, dtype=np.float64), nonzero])
        values[column_index] = float(np.quantile(column, quantile))
    return values
