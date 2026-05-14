"""Small parallelism helpers for perturbation-scoped execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar


T = TypeVar("T")
U = TypeVar("U")


def normalize_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        return 1
    if not isinstance(n_jobs, int):
        raise TypeError("n_jobs must be an integer or None")
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    return n_jobs


def partition_perturbations(
    items: Sequence[T],
    partition_index: int,
    partition_count: int,
) -> list[T]:
    """Return a deterministic round-robin partition of perturbations."""

    if partition_count < 1:
        raise ValueError("partition_count must be at least 1")
    if partition_index < 0 or partition_index >= partition_count:
        raise ValueError("partition_index must be in [0, partition_count)")
    return [item for offset, item in enumerate(items) if offset % partition_count == partition_index]


def run_parallel_tasks(
    items: Sequence[T],
    worker: Callable[[T], U],
    *,
    n_jobs: int | None = 1,
) -> list[U]:
    """Run perturbation-scoped tasks while preserving input order."""

    normalized_jobs = min(normalize_n_jobs(n_jobs), max(len(items), 1))
    if normalized_jobs == 1 or len(items) <= 1:
        return [worker(item) for item in items]

    with ThreadPoolExecutor(max_workers=normalized_jobs) as executor:
        return list(executor.map(worker, items))
