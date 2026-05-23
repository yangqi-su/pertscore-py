"""Small generic helpers shared across perturb_effects modules."""

from __future__ import annotations

import resource
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def is_missing_label(value: Any) -> bool:
    return value is None or bool(isinstance(value, (float, np.floating)) and np.isnan(value))


def ordered_unique(values: Sequence[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def ordered_union_indices(groups: Any) -> np.ndarray:
    union: list[int] = []
    seen: set[int] = set()
    for group in groups:
        for index in group:
            key = int(index)
            if key in seen:
                continue
            union.append(key)
            seen.add(key)
    return np.asarray(union, dtype=np.int64)


def resolve_perturbations(
    labels: Sequence[Any],
    *,
    control_label: str,
    perturbations: Sequence[str] | None = None,
    null_labels: set[str] | None = None,
) -> list[Any]:
    null_set = set() if null_labels is None else null_labels
    available: list[Any] = []
    available_set: set[Any] = set()
    for label in labels:
        if is_missing_label(label) or label == control_label or str(label) in null_set or label in available_set:
            continue
        available.append(label)
        available_set.add(label)

    if perturbations is None:
        return available

    selected: list[Any] = []
    missing: list[str] = []
    seen: set[Any] = set()
    for perturbation in perturbations:
        if perturbation in seen:
            continue
        if perturbation not in available_set:
            missing.append(str(perturbation))
            continue
        selected.append(perturbation)
        seen.add(perturbation)
    if missing:
        raise ValueError("Unknown perturbation labels requested: " + ", ".join(sorted(missing)))
    return selected


def max_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
