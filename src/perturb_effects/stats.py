"""Small statistical helpers for feature summaries and ranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg, sparse

from .types import StreamFeatureStats


@dataclass
class FeatureMoments:
    count: int
    sums: np.ndarray
    squared_sums: np.ndarray

    @classmethod
    def zeros(cls, n_features: int) -> FeatureMoments:
        return cls(
            count=0,
            sums=np.zeros(n_features, dtype=np.float64),
            squared_sums=np.zeros(n_features, dtype=np.float64),
        )

    def add_matrix(self, matrix: Any) -> None:
        summary = summarize_matrix_features(matrix)
        self.count += summary.count
        self.sums += summary.sums
        self.squared_sums += summary.squared_sums

    def freeze(self) -> StreamFeatureStats:
        return StreamFeatureStats(
            count=self.count,
            sums=self.sums.copy(),
            squared_sums=self.squared_sums.copy(),
        )


def summarize_matrix_features(matrix: Any) -> StreamFeatureStats:
    count = int(matrix.shape[0])
    if sparse.issparse(matrix):
        sums = np.asarray(matrix.sum(axis=0)).ravel().astype(float, copy=False)
        squared_sums = np.asarray(matrix.power(2).sum(axis=0)).ravel().astype(float, copy=False)
    else:
        dense = np.asarray(matrix, dtype=float)
        sums = dense.sum(axis=0)
        squared_sums = np.square(dense).sum(axis=0)
    return StreamFeatureStats(count=count, sums=sums, squared_sums=squared_sums)


def welch_t_scores(case_matrix: Any, control_matrix: Any) -> np.ndarray:
    case = summarize_matrix_features(case_matrix)
    control = summarize_matrix_features(control_matrix)
    return welch_t_scores_from_stats(case, control)


def welch_t_scores_from_stats(case: StreamFeatureStats, control: StreamFeatureStats) -> np.ndarray:
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
    if k < 1:
        raise ValueError("k must be at least 1")
    values = np.asarray(scores, dtype=float)
    order_values = np.abs(values) if absolute else values
    order = np.argsort(-order_values, kind="stable")
    return order[: min(k, values.shape[0])]


def column_sums(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64, copy=False)
    return np.asarray(matrix, dtype=np.float64).sum(axis=0)


def log2_fold_change(case_mean: np.ndarray, control_mean: np.ndarray, *, pseudocount: float = 1.0) -> np.ndarray:
    return np.log2((np.asarray(case_mean, dtype=np.float64) + pseudocount) / (np.asarray(control_mean, dtype=np.float64) + pseudocount))


def solve_ridge_beta(x_matrix: Any, y_matrix: Any, lr_lambda: float) -> np.ndarray:
    gram = x_matrix.T @ x_matrix
    rhs = x_matrix.T @ y_matrix
    if sparse.issparse(gram):
        gram = gram.toarray()
    if sparse.issparse(rhs):
        rhs = rhs.toarray()
    return solve_ridge_from_gram_rhs(
        gram=np.asarray(gram, dtype=np.float64),
        rhs=np.asarray(rhs, dtype=np.float64),
        lr_lambda=lr_lambda,
    )


def solve_ridge_from_gram_rhs(*, gram: np.ndarray, rhs: np.ndarray, lr_lambda: float) -> np.ndarray:
    ridge = gram + lr_lambda * np.eye(gram.shape[0], dtype=np.float64)
    try:
        factor = linalg.cho_factor(ridge, lower=True, check_finite=True)
    except linalg.LinAlgError:
        return np.asarray(linalg.solve(ridge, rhs, assume_a="sym", check_finite=True), dtype=np.float64)
    return np.asarray(linalg.cho_solve(factor, rhs, check_finite=True), dtype=np.float64)
