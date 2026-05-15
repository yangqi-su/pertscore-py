"""Fast approximate perturbation scores from streamed group statistics."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse

from .stats import extract_anndata_matrix, get_obs_column, top_k_indices, validate_layer, welch_t_scores_from_stats
from .types import StreamFeatureStats


DEFAULT_TARGET_SUM = 1e4


@dataclass(frozen=True)
class FastApproxPsResult:
    scores: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any]


@dataclass
class _MutableFeatureStats:
    count: int
    sums: np.ndarray
    squared_sums: np.ndarray

    @classmethod
    def zeros(cls, n_features: int) -> _MutableFeatureStats:
        return cls(
            count=0,
            sums=np.zeros(n_features, dtype=np.float64),
            squared_sums=np.zeros(n_features, dtype=np.float64),
        )

    def add_matrix(self, matrix: Any) -> None:
        if sparse.issparse(matrix):
            self.count += int(matrix.shape[0])
            self.sums += np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64, copy=False)
            self.squared_sums += np.asarray(matrix.power(2).sum(axis=0)).ravel().astype(
                np.float64,
                copy=False,
            )
            return

        dense = np.asarray(matrix, dtype=np.float64)
        self.count += int(dense.shape[0])
        self.sums += dense.sum(axis=0)
        self.squared_sums += np.square(dense).sum(axis=0)

    def freeze(self) -> StreamFeatureStats:
        return StreamFeatureStats(
            count=self.count,
            sums=self.sums.copy(),
            squared_sums=self.squared_sums.copy(),
        )


@dataclass(frozen=True)
class _Signature:
    gene_indices: np.ndarray
    gene_names: list[str]
    beta: np.ndarray
    control_mean: np.ndarray
    beta_norm: float
    cell_count: int


def run_ps_score_fast_approx_anndata(
    adata: Any,
    *,
    perturb_column: str,
    ctrl_name: str,
    layer: str | None = None,
    perturbations: Sequence[str] | None = None,
    null_labels: Sequence[str] | None = None,
    top_n: int = 100,
    chunk_size: int = 8192,
    scale_factor: float = 3.0,
    target_sum: float = DEFAULT_TARGET_SUM,
    min_cells_per_perturbation: int = 2,
) -> FastApproxPsResult:
    """Score each cell only against its observed perturbation label."""

    if adata is None:
        raise ValueError("adata must not be None")
    if not isinstance(perturb_column, str) or not perturb_column:
        raise ValueError("perturb_column must be a non-empty string")
    if not isinstance(ctrl_name, str) or not ctrl_name:
        raise ValueError("ctrl_name must be a non-empty string")
    validate_layer(layer)
    top_n = _validate_positive_int("top_n", top_n)
    chunk_size = _validate_positive_int("chunk_size", chunk_size)
    min_cells_per_perturbation = _validate_positive_int(
        "min_cells_per_perturbation",
        min_cells_per_perturbation,
    )
    scale_factor = _validate_positive_float("scale_factor", scale_factor)
    target_sum = _validate_positive_float("target_sum", target_sum)

    labels = np.asarray(get_obs_column(adata.obs, perturb_column), dtype=object)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("adata must contain at least one observation")
    if not np.any(labels == ctrl_name):
        raise ValueError(f"ctrl_name {ctrl_name!r} was not found in adata.obs[{perturb_column!r}]")

    var_names = np.asarray(adata.var_names, dtype=object)
    if var_names.ndim != 1 or var_names.size == 0:
        raise ValueError("adata.var_names must be a non-empty one-dimensional sequence")

    matrix = extract_anndata_matrix(adata, layer=layer)
    null_label_set = _normalize_label_set(null_labels)
    selected = _resolve_selected_perturbations(
        labels,
        control_label=ctrl_name,
        perturbations=perturbations,
        null_labels=null_label_set,
    )
    selected_set = set(selected)

    stage_start = time.perf_counter()
    stats_by_label: dict[str, _MutableFeatureStats] = {
        ctrl_name: _MutableFeatureStats.zeros(var_names.shape[0])
    }
    for perturbation in selected:
        stats_by_label[perturbation] = _MutableFeatureStats.zeros(var_names.shape[0])

    pass1_start = time.perf_counter()
    for start in range(0, labels.shape[0], chunk_size):
        stop = min(start + chunk_size, labels.shape[0])
        chunk = _log_normalize_chunk(matrix[start:stop], target_sum=target_sum)
        chunk_labels = labels[start:stop]
        for label in np.unique(chunk_labels):
            if label == ctrl_name:
                mask = chunk_labels == ctrl_name
            elif label in selected_set:
                mask = chunk_labels == label
            else:
                continue
            if mask.any():
                stats_by_label[str(label)].add_matrix(chunk[mask])
    pass1_seconds = time.perf_counter() - pass1_start

    control_stats = stats_by_label[ctrl_name].freeze()
    control_mean_full = control_stats.means()
    if control_stats.count == 0:
        raise ValueError("control cells are required for fast approximate scoring")

    signature_start = time.perf_counter()
    signatures: dict[str, _Signature] = {}
    skipped: dict[str, dict[str, Any]] = {}
    signature_metadata: dict[str, dict[str, Any]] = {}
    for perturbation in selected:
        perturb_stats = stats_by_label[perturbation].freeze()
        if perturb_stats.count < min_cells_per_perturbation:
            skipped[perturbation] = {
                "reason": "too-few-cells",
                "cell_count": int(perturb_stats.count),
            }
            continue

        t_scores = welch_t_scores_from_stats(perturb_stats, control_stats)
        gene_indices = top_k_indices(t_scores, min(top_n, t_scores.shape[0]), absolute=True)
        perturb_mean = perturb_stats.means()
        beta = perturb_mean[gene_indices] - control_mean_full[gene_indices]
        nonzero = np.abs(beta) > 0
        if not nonzero.any():
            skipped[perturbation] = {
                "reason": "zero-beta-norm",
                "cell_count": int(perturb_stats.count),
            }
            continue

        gene_indices = gene_indices[nonzero]
        beta = beta[nonzero]
        beta_norm = float(np.linalg.norm(beta))
        if beta_norm == 0.0:
            skipped[perturbation] = {
                "reason": "zero-beta-norm",
                "cell_count": int(perturb_stats.count),
            }
            continue

        signature = _Signature(
            gene_indices=gene_indices.astype(np.int64, copy=False),
            gene_names=[str(var_names[index]) for index in gene_indices],
            beta=beta.astype(np.float64, copy=False),
            control_mean=control_mean_full[gene_indices].astype(np.float64, copy=False),
            beta_norm=beta_norm,
            cell_count=int(perturb_stats.count),
        )
        signatures[perturbation] = signature
        signature_metadata[perturbation] = {
            "cell_count": int(perturb_stats.count),
            "selected_gene_count": int(signature.gene_indices.shape[0]),
            "selected_genes": signature.gene_names,
            "beta_norm": beta_norm,
        }
    signature_seconds = time.perf_counter() - signature_start

    raw_scores = np.zeros(labels.shape[0], dtype=np.float32)
    valid_mask = np.zeros(labels.shape[0], dtype=bool)
    max_raw_by_perturbation = {perturbation: 0.0 for perturbation in signatures}

    pass2_start = time.perf_counter()
    for start in range(0, labels.shape[0], chunk_size):
        stop = min(start + chunk_size, labels.shape[0])
        chunk = _log_normalize_chunk(matrix[start:stop], target_sum=target_sum)
        chunk_labels = labels[start:stop]
        row_indices = np.arange(start, stop, dtype=np.int64)
        for perturbation in np.unique(chunk_labels):
            if perturbation not in signatures:
                continue
            mask = chunk_labels == perturbation
            if not mask.any():
                continue
            signature = signatures[str(perturbation)]
            projected = _project_signature(chunk[mask], signature)
            clipped = np.clip(projected, 0.0, scale_factor)
            selected_rows = row_indices[mask]
            raw_scores[selected_rows] = clipped.astype(np.float32, copy=False)
            valid_mask[selected_rows] = True
            if clipped.size:
                max_raw_by_perturbation[str(perturbation)] = max(
                    max_raw_by_perturbation[str(perturbation)],
                    float(np.max(clipped)),
                )
    pass2_seconds = time.perf_counter() - pass2_start

    scores = np.zeros(labels.shape[0], dtype=np.float32)
    for perturbation, signature in signatures.items():
        perturbation_mask = labels == perturbation
        max_raw = max_raw_by_perturbation[perturbation]
        if max_raw > 0.0:
            scores[perturbation_mask] = raw_scores[perturbation_mask] / np.float32(max_raw)
        else:
            scores[perturbation_mask] = 0.0
        signature_metadata[perturbation]["max_raw_score"] = float(max_raw)
        signature_metadata[perturbation]["valid_cell_count"] = int(np.count_nonzero(perturbation_mask))

    unknown_count = int(
        sum(
            1
            for label in labels
            if not _is_missing_label(label)
            and label != ctrl_name
            and label not in null_label_set
            and label not in selected_set
        )
    )
    invalid_count = int(sum(1 for label in labels if _is_missing_label(label) or label in null_label_set))
    skipped_cell_count = int(sum(signature_data["cell_count"] for signature_data in skipped.values()))
    metadata = {
        "algorithm": "ps_score_fast_approx",
        "layer": layer,
        "perturb_column": perturb_column,
        "control_label": ctrl_name,
        "top_gene_count": int(top_n),
        "chunk_size": int(chunk_size),
        "target_sum": float(target_sum),
        "scale_factor": float(scale_factor),
        "score_vector_shape": (int(labels.shape[0]), 1),
        "control_cell_count": int(np.count_nonzero(labels == ctrl_name)),
        "invalid_label_count": invalid_count,
        "unknown_label_count": unknown_count,
        "skipped_cell_count": skipped_cell_count,
        "valid_scored_cell_count": int(np.count_nonzero(valid_mask)),
        "selected_perturbations": list(selected),
        "valid_perturbation_count": int(len(signatures)),
        "skipped_perturbation_count": int(len(skipped)),
        "signature_metadata": signature_metadata,
        "skipped_perturbations": skipped,
        "timings": {
            "pass1_seconds": float(pass1_seconds),
            "signature_seconds": float(signature_seconds),
            "pass2_seconds": float(pass2_seconds),
            "total_seconds": float(time.perf_counter() - stage_start),
        },
        "max_rss_kb": _max_rss_kb(),
    }
    return FastApproxPsResult(scores=scores.reshape(-1, 1), valid_mask=valid_mask, metadata=metadata)


def run_ps_score_fast_approx_dataset(
    dataset_path: str | Path,
    *,
    output_dir: str | Path,
    perturb_column: str,
    ctrl_name: str,
    layer: str | None = None,
    perturbations: Sequence[str] | None = None,
    null_labels: Sequence[str] | None = None,
    top_n: int = 100,
    chunk_size: int = 8192,
    scale_factor: float = 3.0,
    target_sum: float = DEFAULT_TARGET_SUM,
    min_cells_per_perturbation: int = 2,
) -> dict[str, Any]:
    adata = ad.read_h5ad(Path(dataset_path), backed="r")
    try:
        result = run_ps_score_fast_approx_anndata(
            adata,
            perturb_column=perturb_column,
            ctrl_name=ctrl_name,
            layer=layer,
            perturbations=perturbations,
            null_labels=null_labels,
            top_n=top_n,
            chunk_size=chunk_size,
            scale_factor=scale_factor,
            target_sum=target_sum,
            min_cells_per_perturbation=min_cells_per_perturbation,
        )
    finally:
        _close_adata(adata)
    return write_ps_score_fast_approx_output(
        result,
        output_dir=output_dir,
        dataset_path=dataset_path,
    )


def write_ps_score_fast_approx_output(
    result: FastApproxPsResult,
    *,
    output_dir: str | Path,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    score_path = output_path / "ps-score-fast-approx.npy"
    valid_mask_path = output_path / "ps-score-fast-approx-valid-mask.npy"
    signature_path = output_path / "ps-score-fast-approx-signatures.json"
    manifest_path = output_path / "ps-score-fast-approx-manifest.json"

    np.save(score_path, result.scores)
    np.save(valid_mask_path, result.valid_mask)
    with signature_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _to_jsonable(
                {
                    "signature_metadata": result.metadata["signature_metadata"],
                    "skipped_perturbations": result.metadata["skipped_perturbations"],
                }
            ),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    manifest = {
        "algorithm": result.metadata["algorithm"],
        "dataset_path": None if dataset_path is None else str(dataset_path),
        "perturbation_column": result.metadata["perturb_column"],
        "control_label": result.metadata["control_label"],
        "top_gene_count": result.metadata["top_gene_count"],
        "chunk_size": result.metadata["chunk_size"],
        "target_sum": result.metadata["target_sum"],
        "valid_perturbation_count": result.metadata["valid_perturbation_count"],
        "skipped_perturbation_count": result.metadata["skipped_perturbation_count"],
        "score_vector_shape": result.metadata["score_vector_shape"],
        "score_output_paths": {
            "normalized_scores": str(score_path),
            "valid_mask": str(valid_mask_path),
            "signature_metadata": str(signature_path),
        },
        "timings": result.metadata["timings"],
        "max_rss_kb": result.metadata["max_rss_kb"],
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--perturb-column", required=True)
    parser.add_argument("--ctrl-name", required=True)
    parser.add_argument("--layer")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--scale-factor", type=float, default=3.0)
    parser.add_argument("--target-sum", type=float, default=DEFAULT_TARGET_SUM)
    parser.add_argument("--min-cells-per-perturbation", type=int, default=2)
    parser.add_argument("--perturbation", action="append", dest="perturbations")
    parser.add_argument("--null-label", action="append", dest="null_labels")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run_ps_score_fast_approx_dataset(
        args.dataset_path,
        output_dir=args.output_dir,
        perturb_column=args.perturb_column,
        ctrl_name=args.ctrl_name,
        layer=args.layer,
        perturbations=args.perturbations,
        null_labels=args.null_labels,
        top_n=args.top_n,
        chunk_size=args.chunk_size,
        scale_factor=args.scale_factor,
        target_sum=args.target_sum,
        min_cells_per_perturbation=args.min_cells_per_perturbation,
    )


def _resolve_selected_perturbations(
    labels: Sequence[Any],
    *,
    control_label: str,
    perturbations: Sequence[str] | None,
    null_labels: set[str],
) -> list[str]:
    available: list[str] = []
    available_set: set[str] = set()
    for label in labels:
        if _is_missing_label(label) or label == control_label or label in null_labels:
            continue
        key = str(label)
        if key in available_set:
            continue
        available.append(key)
        available_set.add(key)

    if perturbations is None:
        return available

    selected: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for perturbation in perturbations:
        if perturbation == control_label:
            raise ValueError("control label cannot be included in perturbations")
        if perturbation in seen:
            continue
        if perturbation not in available_set:
            missing.append(str(perturbation))
            continue
        selected.append(str(perturbation))
        seen.add(str(perturbation))

    if missing:
        raise ValueError("Unknown perturbation labels requested: " + ", ".join(sorted(missing)))
    return selected


def _log_normalize_chunk(matrix: Any, *, target_sum: float) -> Any:
    if sparse.issparse(matrix):
        work = matrix.tocsr(copy=True).astype(np.float64)
        totals = np.asarray(work.sum(axis=1)).ravel()
        scales = np.zeros(work.shape[0], dtype=np.float64)
        nonzero = totals > 0
        scales[nonzero] = target_sum / totals[nonzero]
        work = work.multiply(scales[:, None]).tocsr()
        work.data = np.log1p(work.data)
        return work

    dense = np.asarray(matrix, dtype=np.float64).copy()
    if dense.ndim != 2:
        raise ValueError("matrix chunks must be two-dimensional")
    totals = dense.sum(axis=1, keepdims=True)
    nonzero = totals[:, 0] > 0
    dense[nonzero] *= target_sum / totals[nonzero]
    dense[~nonzero] = 0.0
    return np.log1p(dense)


def _project_signature(matrix: Any, signature: _Signature) -> np.ndarray:
    selected = matrix[:, signature.gene_indices]
    if sparse.issparse(selected):
        selected = selected.toarray()
    dense = np.asarray(selected, dtype=np.float64)
    centered = dense - signature.control_mean
    denominator = signature.beta_norm * signature.beta_norm
    if denominator == 0.0:
        return np.zeros(dense.shape[0], dtype=np.float64)
    return centered @ signature.beta / denominator


def _normalize_label_set(labels: Sequence[str] | None) -> set[str]:
    if labels is None:
        return set()
    return {str(label) for label in labels}


def _is_missing_label(label: Any) -> bool:
    if label is None:
        return True
    if isinstance(label, (float, np.floating)):
        return bool(np.isnan(label))
    return False


def _validate_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_positive_float(name: str, value: Any) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)) or float(value) <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _max_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _close_adata(adata: Any) -> None:
    if hasattr(adata, "file") and hasattr(adata.file, "close"):
        adata.file.close()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "FastApproxPsResult",
    "build_parser",
    "main",
    "run_ps_score_fast_approx_anndata",
    "run_ps_score_fast_approx_dataset",
    "write_ps_score_fast_approx_output",
]
