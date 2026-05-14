"""Shared lightweight types for perturbation effect calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence, TypeAlias

import numpy as np


@dataclass(frozen=True)
class CsrBatch:
    """A single CSR batch with stable row identifiers."""

    row_ids: Sequence[Any]
    indptr: Sequence[int]
    indices: Sequence[int]
    data: Sequence[float]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.shape) != 2:
            raise ValueError("shape must be a 2-tuple")

        n_rows, n_cols = self.shape
        if n_rows < 0 or n_cols < 0:
            raise ValueError("shape entries must be non-negative")
        if len(self.row_ids) != n_rows:
            raise ValueError("row_ids length must match shape[0]")
        if len(self.indptr) != n_rows + 1:
            raise ValueError("indptr length must equal shape[0] + 1")
        if self.indptr[0] != 0:
            raise ValueError("indptr must start at 0")
        nnz = len(self.indices)
        if len(self.data) != nnz:
            raise ValueError("indices and data must have the same length")
        if self.indptr[-1] != nnz:
            raise ValueError("indptr[-1] must equal the number of nonzero entries")


BatchIterable: TypeAlias = Iterable[CsrBatch]
BatchFactory: TypeAlias = Callable[[], BatchIterable]
BatchSource: TypeAlias = BatchIterable | BatchFactory


@dataclass(frozen=True)
class StreamFeatureStats:
    """Streaming feature moments for later mean/variance calculations."""

    count: int
    sums: np.ndarray
    squared_sums: np.ndarray

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be non-negative")
        if self.sums.ndim != 1:
            raise ValueError("sums must be a one-dimensional array")
        if self.squared_sums.ndim != 1:
            raise ValueError("squared_sums must be a one-dimensional array")
        if self.sums.shape != self.squared_sums.shape:
            raise ValueError("sums and squared_sums must have the same shape")

    @property
    def n_features(self) -> int:
        return int(self.sums.shape[0])

    def means(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros(self.n_features, dtype=float)
        return self.sums / float(self.count)

    def variances(self, *, ddof: int = 1) -> np.ndarray:
        if ddof < 0:
            raise ValueError("ddof must be non-negative")
        if self.count <= ddof:
            return np.zeros(self.n_features, dtype=float)

        means = self.means()
        raw = (self.squared_sums / float(self.count)) - np.square(means)
        raw = np.maximum(raw, 0.0)
        if ddof == 0:
            return raw
        scale = self.count / float(self.count - ddof)
        return raw * scale
