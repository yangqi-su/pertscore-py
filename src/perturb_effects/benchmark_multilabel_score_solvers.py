"""Benchmark small multi-label PS score solvers on synthetic data."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize

from .ps_score_exact_fast import _bounded_quadratic_objective, _solve_bounded_quadratic_scores


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    args = build_parser().parse_args(argv)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for k in args.k_values:
        for cell_count in args.cell_counts:
            active_beta, centered, rhs, gram = _make_problem(
                rng,
                k=k,
                cell_count=cell_count,
                gene_count=args.genes,
                upper=args.scale_factor,
                noise_sd=args.noise_sd,
            )
            active_seconds, active_scores = _time_call(
                lambda: _solve_bounded_quadratic_scores(
                    gram=gram,
                    rhs=rhs,
                    linear_penalty=args.score_lambda,
                    upper=args.scale_factor,
                ),
                repeats=args.repeats,
            )
            rows.append(_summarize_result("active_set", k, cell_count, args.genes, active_seconds, active_scores, active_scores, gram, rhs, args.score_lambda))

            if "lbfgsb" in args.methods:
                seconds, scores = _time_call(
                    lambda: _solve_lbfgsb(
                        gram=gram,
                        rhs=rhs,
                        linear_penalty=args.score_lambda,
                        upper=args.scale_factor,
                        maxiter=args.maxiter,
                    ),
                    repeats=args.repeats,
                )
                rows.append(_summarize_result("lbfgsb", k, cell_count, args.genes, seconds, scores, active_scores, gram, rhs, args.score_lambda))

            if "lsq_linear" in args.methods:
                seconds, scores = _time_call(
                    lambda: _solve_lsq_linear(
                        active_beta=active_beta,
                        centered=centered,
                        gram=gram,
                        linear_penalty=args.score_lambda,
                        upper=args.scale_factor,
                        tol=args.lsq_tol,
                    ),
                    repeats=args.repeats,
                )
                rows.append(_summarize_result("lsq_linear", k, cell_count, args.genes, seconds, scores, active_scores, gram, rhs, args.score_lambda))

            print(pd.DataFrame(rows).tail(1 + int("lbfgsb" in args.methods) + int("lsq_linear" in args.methods)).to_string(index=False))
            print()

    table = pd.DataFrame(rows)
    if args.output_csv is not None:
        table.to_csv(args.output_csv, index=False)
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--cell-counts", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--genes", type=int, default=5000)
    parser.add_argument("--methods", choices=["lbfgsb", "lsq_linear"], nargs="+", default=["lbfgsb", "lsq_linear"])
    parser.add_argument("--score-lambda", type=float, default=0.0)
    parser.add_argument("--scale-factor", type=float, default=3.0)
    parser.add_argument("--noise-sd", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--lsq-tol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-csv", type=Path)
    return parser


def _make_problem(
    rng: np.random.Generator,
    *,
    k: int,
    cell_count: int,
    gene_count: int,
    upper: float,
    noise_sd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active_beta = rng.normal(size=(k, gene_count)).astype(np.float64)
    active_beta /= np.linalg.norm(active_beta, axis=1, keepdims=True)
    true_scores = rng.uniform(0.0, upper, size=(cell_count, k))
    centered = true_scores @ active_beta + rng.normal(scale=noise_sd, size=(cell_count, gene_count))
    gram = active_beta @ active_beta.T
    rhs = centered @ active_beta.T
    return active_beta, centered, rhs, gram


def _time_call(function: Callable[[], np.ndarray], *, repeats: int) -> tuple[float, np.ndarray]:
    best_seconds = np.inf
    best_result: np.ndarray | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        seconds = time.perf_counter() - start
        if seconds < best_seconds:
            best_seconds = seconds
            best_result = result
    assert best_result is not None
    return float(best_seconds), best_result


def _solve_lbfgsb(
    *,
    gram: np.ndarray,
    rhs: np.ndarray,
    linear_penalty: float,
    upper: float,
    maxiter: int,
) -> np.ndarray:
    scores = np.zeros_like(rhs, dtype=np.float64)
    bounds = [(0.0, upper)] * rhs.shape[1]
    for row_index, row_rhs in enumerate(rhs):
        def objective(value: np.ndarray) -> float:
            return float(0.5 * value @ gram @ value - row_rhs @ value + linear_penalty * np.sum(value))

        def gradient(value: np.ndarray) -> np.ndarray:
            return gram @ value - row_rhs + linear_penalty

        result = minimize(
            objective,
            np.zeros(rhs.shape[1], dtype=np.float64),
            jac=gradient,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": maxiter},
        )
        scores[row_index] = np.clip(result.x, 0.0, upper)
    return scores


def _solve_lsq_linear(
    *,
    active_beta: np.ndarray,
    centered: np.ndarray,
    gram: np.ndarray,
    linear_penalty: float,
    upper: float,
    tol: float,
) -> np.ndarray:
    design = active_beta.T
    target = centered
    if linear_penalty != 0.0:
        shift = design @ np.linalg.solve(gram, np.full(active_beta.shape[0], linear_penalty, dtype=np.float64))
        target = centered - shift[None, :]

    scores = np.zeros((centered.shape[0], active_beta.shape[0]), dtype=np.float64)
    for row_index, row in enumerate(target):
        scores[row_index] = lsq_linear(design, row, bounds=(0.0, upper), tol=tol).x
    return scores


def _summarize_result(
    method: str,
    k: int,
    cell_count: int,
    gene_count: int,
    seconds: float,
    scores: np.ndarray,
    reference_scores: np.ndarray,
    gram: np.ndarray,
    rhs: np.ndarray,
    linear_penalty: float,
) -> dict[str, Any]:
    diff = np.abs(scores - reference_scores)
    objective = _bounded_quadratic_objective(scores, gram=gram, rhs=rhs, linear_penalty=linear_penalty)
    return {
        "method": method,
        "k": int(k),
        "cells": int(cell_count),
        "genes": int(gene_count),
        "seconds": float(seconds),
        "cells_per_second": float(cell_count / seconds) if seconds > 0.0 else np.inf,
        "mean_objective": float(np.mean(objective)),
        "max_abs_diff_vs_active_set": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_diff_vs_active_set": float(np.mean(diff)) if diff.size else 0.0,
    }


if __name__ == "__main__":
    main()
