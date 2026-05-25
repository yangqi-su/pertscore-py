"""Small generic helpers shared across pertscore modules."""

from __future__ import annotations

import resource
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


PERTURBATION_DELIMITER = "+"
PS_SCORE_COLUMNS = ["obs_index", "ps_score", "perturbation"]


@dataclass(frozen=True)
class ParsedPerturbations:
    perturbations: list[str]
    guides: sparse.csr_matrix
    control_mask: np.ndarray
    model_mask: np.ndarray
    active_counts: np.ndarray


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


def clean_obs_labels(adata: Any, perturb_column: str) -> np.ndarray:
    raw = np.asarray(adata.obs[perturb_column], dtype=object)
    labels: list[str] = []
    for row_index, value in enumerate(raw):
        if pd.isna(value):
            raise ValueError(f"Missing perturbation label at obs row {row_index}")
        label = str(value)
        if not label:
            raise ValueError(f"Empty perturbation label at obs row {row_index}")
        labels.append(label)
    return np.asarray(labels, dtype=object)


def background_cluster_codes(adata: Any, cluster_column: str | None) -> tuple[np.ndarray | None, list[str]]:
    if cluster_column is None:
        return None, []
    raw = np.asarray(adata.obs[cluster_column], dtype=object)
    labels: list[str] = []
    for row_index, value in enumerate(raw):
        if pd.isna(value):
            raise ValueError(f"Missing background cluster label at obs row {row_index}")
        label = str(value)
        if not label:
            raise ValueError(f"Empty background cluster label at obs row {row_index}")
        labels.append(label)
    cluster_names = ordered_unique(labels)
    lookup = {cluster: index for index, cluster in enumerate(cluster_names)}
    return np.asarray([lookup[label] for label in labels], dtype=np.int32), cluster_names


def parse_perturbation_labels(
    labels: np.ndarray,
    *,
    mode: str,
    ctrl_name: str,
    perturbations: Sequence[str] | None,
) -> ParsedPerturbations:
    tokenized: list[list[str]] = []
    known: list[str] = []
    known_set: set[str] = set()
    control_mask = labels == ctrl_name
    for row_index, label in enumerate(labels):
        if label == ctrl_name:
            tokenized.append([])
            continue
        tokens = [str(label)] if mode == "single" else [token.strip() for token in str(label).split(PERTURBATION_DELIMITER)]
        if any(not token for token in tokens):
            raise ValueError(f"Malformed perturbation value at obs row {row_index}: {label!r}")
        if ctrl_name in tokens:
            raise ValueError(f"Control label cannot appear inside a perturbation combination at obs row {row_index}")
        active = ordered_unique(tokens)
        tokenized.append(active)
        for token in active:
            if token not in known_set:
                known.append(token)
                known_set.add(token)

    selected = select_perturbations(known, perturbations)
    selected_set = set(selected)
    selected_lookup = {perturbation: index for index, perturbation in enumerate(selected)}
    rows: list[int] = []
    columns: list[int] = []
    model_mask = control_mask.copy()
    for row_index, active in enumerate(tokenized):
        if not active or not set(active).issubset(selected_set):
            continue
        model_mask[row_index] = True
        for perturbation in active:
            rows.append(row_index)
            columns.append(selected_lookup[perturbation])

    guides = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, columns)),
        shape=(labels.shape[0], len(selected)),
        dtype=np.float64,
    )
    guides.sort_indices()
    active_counts = np.asarray(guides.getnnz(axis=1)).ravel().astype(np.int64, copy=False)
    return ParsedPerturbations(selected, guides, control_mask, model_mask, active_counts)


def select_perturbations(known: Sequence[str], perturbations: Sequence[str] | None) -> list[str]:
    if not known:
        raise ValueError("No perturbation labels were found outside the control label")
    if perturbations is None:
        return list(known)
    selected = [str(perturbation) for perturbation in perturbations]
    if len(set(selected)) != len(selected):
        raise ValueError("Requested perturbations must be unique")
    missing = sorted(set(selected) - set(known))
    if missing:
        raise ValueError("Requested perturbations are not present in the perturbation column: " + ", ".join(missing))
    return selected


def group_rows_by_active_set(guides: sparse.csr_matrix) -> dict[tuple[int, ...], np.ndarray]:
    groups: dict[tuple[int, ...], list[int]] = {}
    indptr = guides.indptr
    indices = guides.indices
    for row_index in range(guides.shape[0]):
        active = tuple(int(index) for index in indices[indptr[row_index] : indptr[row_index + 1]])
        if active:
            groups.setdefault(active, []).append(row_index)
    return {key: np.asarray(rows, dtype=np.int64) for key, rows in groups.items()}


def ps_score_long_dataframe(
    *,
    obs_index: np.ndarray,
    control_mask: np.ndarray,
    valid_mask: np.ndarray,
    scores: np.ndarray,
    cell_indices: np.ndarray,
    perturbation_indices: np.ndarray,
    perturbations: Sequence[str],
    ctrl_name: str,
    missing_perturbation: Any = None,
) -> pd.DataFrame:
    perturbation_names = np.asarray(perturbations, dtype=object)
    control_rows = np.flatnonzero(control_mask)
    missing_rows = np.flatnonzero(~control_mask & ~valid_mask)
    table = pd.concat(
        [
            pd.DataFrame(
                {
                    "_row_order": cell_indices,
                    "_perturbation_order": perturbation_indices,
                    "obs_index": obs_index[cell_indices],
                    "ps_score": scores,
                    "perturbation": perturbation_names[perturbation_indices],
                }
            ),
            pd.DataFrame(
                {
                    "_row_order": control_rows,
                    "_perturbation_order": np.full(control_rows.shape[0], -1, dtype=np.int32),
                    "obs_index": obs_index[control_rows],
                    "ps_score": np.zeros(control_rows.shape[0], dtype=np.float64),
                    "perturbation": np.full(control_rows.shape[0], ctrl_name, dtype=object),
                }
            ),
            pd.DataFrame(
                {
                    "_row_order": missing_rows,
                    "_perturbation_order": np.full(missing_rows.shape[0], -1, dtype=np.int32),
                    "obs_index": obs_index[missing_rows],
                    "ps_score": np.full(missing_rows.shape[0], np.nan, dtype=np.float64),
                    "perturbation": np.full(missing_rows.shape[0], missing_perturbation, dtype=object),
                }
            ),
        ],
        ignore_index=True,
    )
    table.sort_values(["_row_order", "_perturbation_order"], kind="stable", inplace=True)
    table.reset_index(drop=True, inplace=True)
    return table[PS_SCORE_COLUMNS]


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
