"""Fast approximate perturbation scores from streamed group statistics."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse

from .stats import FeatureMoments, top_k_indices, welch_t_scores_from_stats
from .stream import (
    accumulate_nonzero_histogram,
    clip_values_from_histogram,
    extract_anndata_matrix,
    iter_matrix_chunks,
)
from .utils import (
    is_missing_label,
    max_rss_kb,
    ordered_union_indices,
    ordered_unique,
    ps_score_long_dataframe,
    resolve_perturbations,
    to_jsonable,
)


DEFAULT_TARGET_SUM = 1e4
DEFAULT_CLIP_BINS = 2048


@dataclass(frozen=True)
class FastApproxPsResult:
    scores: np.ndarray
    valid_mask: np.ndarray
    obs_index: np.ndarray
    labels: np.ndarray
    control_mask: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _Signature:
    gene_indices: np.ndarray
    beta: np.ndarray
    control_mean: np.ndarray
    beta_norm: float


def run_ps_score_fast_approx(
    data: Any,
    *,
    output_dir: str | Path | None = None,
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
    clip_quantile: float | None = None,
    clip_bins: int = DEFAULT_CLIP_BINS,
    target_basis: str = "per_perturbation",
) -> FastApproxPsResult | dict[str, Any]:
    """Score each cell only against its observed perturbation label."""

    if target_basis not in {"per_perturbation", "union"}:
        raise ValueError("target_basis must be 'per_perturbation' or 'union'")
    if clip_quantile is not None and not (0.0 < clip_quantile <= 1.0):
        raise ValueError("clip_quantile must be in (0, 1]")
    if clip_bins < 2:
        raise ValueError("clip_bins must be >= 2")

    dataset_path = Path(data) if isinstance(data, (str, Path)) else None
    adata = ad.read_h5ad(dataset_path, backed="r") if dataset_path is not None else data

    raw_labels = np.asarray(adata.obs[perturb_column], dtype=object)
    labels = np.asarray(
        [label if is_missing_label(label) else str(label) for label in raw_labels],
        dtype=object,
    )
    obs_index = np.asarray(adata.obs_names, dtype=object)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("adata must contain at least one observation")
    if not np.any(labels == ctrl_name):
        raise ValueError(f"ctrl_name {ctrl_name!r} was not found in adata.obs[{perturb_column!r}]")

    var_names = np.asarray(adata.var_names, dtype=object)
    if var_names.ndim != 1 or var_names.size == 0:
        raise ValueError("adata.var_names must be a non-empty one-dimensional sequence")

    matrix = extract_anndata_matrix(adata, layer=layer)
    null_label_set = set() if null_labels is None else {str(label) for label in null_labels}
    if perturbations is not None and any(str(perturbation) == ctrl_name for perturbation in perturbations):
        raise ValueError("control label cannot be included in perturbations")
    requested = None if perturbations is None else [str(perturbation) for perturbation in perturbations]
    selected = [
        str(perturbation)
        for perturbation in resolve_perturbations(
            labels,
            control_label=ctrl_name,
            perturbations=requested,
            null_labels=null_label_set,
        )
    ]
    selected_set = set(selected)
    model_label_set = selected_set | {ctrl_name}

    stage_start = time.perf_counter()
    stats_by_label: dict[str, FeatureMoments] = {
        ctrl_name: FeatureMoments.zeros(var_names.shape[0])
    }
    for perturbation in selected:
        stats_by_label[perturbation] = FeatureMoments.zeros(var_names.shape[0])

    max_value = float(np.log1p(target_sum))
    all_gene_hist: np.ndarray | None = None
    all_gene_nonzero_counts: np.ndarray | None = None
    if clip_quantile is not None:
        all_gene_hist = np.zeros((var_names.shape[0], clip_bins), dtype=np.uint32)
        all_gene_nonzero_counts = np.zeros(var_names.shape[0], dtype=np.int64)

    pass1_start = time.perf_counter()
    for start, stop, chunk in iter_matrix_chunks(
        matrix,
        n_obs=labels.shape[0],
        chunk_size=chunk_size,
        target_sum=target_sum,
    ):
        chunk_labels = labels[start:stop]
        if all_gene_hist is not None and all_gene_nonzero_counts is not None:
            model_mask = np.asarray(
                [not is_missing_label(label) and str(label) in model_label_set for label in chunk_labels],
                dtype=bool,
            )
            if np.any(model_mask):
                accumulate_nonzero_histogram(
                    chunk[model_mask],
                    hist=all_gene_hist,
                    nonzero_counts=all_gene_nonzero_counts,
                    max_value=max_value,
                )
        label_keys = ordered_unique([label for label in chunk_labels if not is_missing_label(label)])
        for label_key in label_keys:
            if label_key == ctrl_name:
                mask = chunk_labels == ctrl_name
            elif label_key in selected_set:
                mask = chunk_labels == label_key
            else:
                continue
            if mask.any():
                stats_by_label[label_key].add_matrix(chunk[mask])
    pass1_seconds = time.perf_counter() - pass1_start

    control_stats = stats_by_label[ctrl_name].freeze()
    control_mean_full = control_stats.means()
    if control_stats.count == 0:
        raise ValueError("control cells are required for fast approximate scoring")

    signature_start = time.perf_counter()
    signatures: dict[str, _Signature] = {}
    skipped: dict[str, dict[str, Any]] = {}
    target_gene_indices_by_perturbation: dict[str, np.ndarray] = {}
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
        target_gene_indices_by_perturbation[perturbation] = gene_indices.astype(np.int64, copy=False)

    union_target_gene_indices = ordered_union_indices(target_gene_indices_by_perturbation.values())
    union_target_genes = [str(var_names[index]) for index in union_target_gene_indices]
    clip_values: np.ndarray | None = None
    clip_threshold_seconds = 0.0
    clipped_stats_seconds = 0.0

    if clip_quantile is None and target_basis == "per_perturbation":
        for perturbation in selected:
            if perturbation not in target_gene_indices_by_perturbation:
                continue
            perturb_stats = stats_by_label[perturbation].freeze()
            signature = _build_signature(
                projection_gene_indices=target_gene_indices_by_perturbation[perturbation],
                perturb_mean=perturb_stats.means()[target_gene_indices_by_perturbation[perturbation]],
                control_mean=control_mean_full[target_gene_indices_by_perturbation[perturbation]],
                drop_zero_genes=True,
            )
            if signature is None:
                skipped[perturbation] = {
                    "reason": "zero-beta-norm",
                    "cell_count": int(perturb_stats.count),
                }
                continue
            signatures[perturbation] = signature
    else:
        union_index_lookup = {int(index): position for position, index in enumerate(union_target_gene_indices)}
        clipped_union_stats: dict[str, FeatureMoments] | None = None
        clipped_control_mean = None
        if clip_quantile is not None and union_target_gene_indices.size:
            clip_start = time.perf_counter()
            clip_values = clip_values_from_histogram(
                hist=all_gene_hist,
                nonzero_counts=all_gene_nonzero_counts,
                gene_indices=union_target_gene_indices,
                model_cell_count=control_stats.count + sum(stats_by_label[label].count for label in selected),
                quantile=clip_quantile,
                max_value=max_value,
            )
            clip_threshold_seconds = time.perf_counter() - clip_start

            clipped_stats_start = time.perf_counter()
            clipped_union_stats = {ctrl_name: FeatureMoments.zeros(union_target_gene_indices.shape[0])}
            for perturbation in target_gene_indices_by_perturbation:
                clipped_union_stats[perturbation] = FeatureMoments.zeros(union_target_gene_indices.shape[0])
            for start, stop, chunk in iter_matrix_chunks(
                matrix,
                n_obs=labels.shape[0],
                chunk_size=chunk_size,
                target_sum=target_sum,
                gene_indices=union_target_gene_indices,
                clip_values=clip_values,
            ):
                chunk_labels = labels[start:stop]
                label_keys = ordered_unique([label for label in chunk_labels if not is_missing_label(label)])
                for label_key in label_keys:
                    if label_key not in clipped_union_stats:
                        continue
                    mask = chunk_labels == label_key
                    if mask.any():
                        clipped_union_stats[label_key].add_matrix(chunk[mask])
            clipped_stats_seconds = time.perf_counter() - clipped_stats_start
            clipped_control_mean = clipped_union_stats[ctrl_name].freeze().means()

        for perturbation in selected:
            target_gene_indices = target_gene_indices_by_perturbation.get(perturbation)
            if target_gene_indices is None:
                continue
            perturb_stats = stats_by_label[perturbation].freeze()
            if target_basis == "union":
                projection_gene_indices = np.arange(union_target_gene_indices.shape[0], dtype=np.int64)
            else:
                projection_gene_indices = np.asarray(
                    [union_index_lookup[int(index)] for index in target_gene_indices],
                    dtype=np.int64,
                )

            if clipped_union_stats is None:
                perturb_mean_union = perturb_stats.means()[union_target_gene_indices]
                control_mean_union = control_mean_full[union_target_gene_indices]
            else:
                perturb_mean_union = clipped_union_stats[perturbation].freeze().means()
                control_mean_union = clipped_control_mean

            if target_basis == "union":
                perturb_mean = perturb_mean_union
                control_mean = control_mean_union
            else:
                perturb_mean = perturb_mean_union[projection_gene_indices]
                control_mean = control_mean_union[projection_gene_indices]

            signature = _build_signature(
                projection_gene_indices=projection_gene_indices,
                perturb_mean=perturb_mean,
                control_mean=control_mean,
                drop_zero_genes=target_basis == "per_perturbation",
            )
            if signature is None:
                skipped[perturbation] = {
                    "reason": "zero-beta-norm",
                    "cell_count": int(perturb_stats.count),
                }
                continue
            signatures[perturbation] = signature
    signature_seconds = time.perf_counter() - signature_start

    raw_scores = np.zeros(labels.shape[0], dtype=np.float32)
    valid_mask = np.zeros(labels.shape[0], dtype=bool)
    max_raw_by_perturbation = {perturbation: 0.0 for perturbation in signatures}

    pass2_start = time.perf_counter()
    score_gene_indices = union_target_gene_indices if clip_quantile is not None or target_basis == "union" else None
    for start, stop, chunk in iter_matrix_chunks(
        matrix,
        n_obs=labels.shape[0],
        chunk_size=chunk_size,
        target_sum=target_sum,
        gene_indices=score_gene_indices,
        clip_values=clip_values,
    ):
        chunk_labels = labels[start:stop]
        row_indices = np.arange(start, stop, dtype=np.int64)
        label_keys = ordered_unique([label for label in chunk_labels if not is_missing_label(label)])
        for perturbation in label_keys:
            if perturbation not in signatures:
                continue
            mask = chunk_labels == perturbation
            if not mask.any():
                continue
            signature = signatures[perturbation]
            projected = _project_signature(chunk[mask], signature)
            clipped = np.clip(projected, 0.0, scale_factor)
            selected_rows = row_indices[mask]
            raw_scores[selected_rows] = clipped.astype(np.float32, copy=False)
            valid_mask[selected_rows] = True
            if clipped.size:
                max_raw_by_perturbation[perturbation] = max(
                    max_raw_by_perturbation[perturbation],
                    float(np.max(clipped)),
                )
    pass2_seconds = time.perf_counter() - pass2_start

    scores = np.zeros(labels.shape[0], dtype=np.float32)
    for perturbation in signatures:
        perturbation_mask = labels == perturbation
        max_raw = max_raw_by_perturbation[perturbation]
        if max_raw > 0.0:
            scores[perturbation_mask] = raw_scores[perturbation_mask] / np.float32(max_raw)
        else:
            scores[perturbation_mask] = 0.0

    unknown_count = int(
        sum(
            1
            for label in labels
            if not is_missing_label(label)
            and label != ctrl_name
            and label not in null_label_set
            and label not in selected_set
        )
    )
    invalid_count = int(sum(1 for label in labels if is_missing_label(label) or label in null_label_set))
    skipped_cell_count = int(sum(signature_data["cell_count"] for signature_data in skipped.values()))
    metadata = {
        "algorithm": "ps_score_fast_approx",
        "layer": layer,
        "perturb_column": perturb_column,
        "control_label": ctrl_name,
        "top_gene_count": int(top_n),
        "target_basis": target_basis,
        "quantile_clip": clip_quantile is not None,
        "clip_quantile": None if clip_quantile is None else float(clip_quantile),
        "clip_method": None if clip_quantile is None else "streaming_histogram",
        "clip_bins": None if clip_quantile is None else int(clip_bins),
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
        "union_target_gene_count": int(union_target_gene_indices.shape[0]),
        "union_target_genes": union_target_genes,
        "skipped_perturbations": skipped,
        "timings": {
            "pass1_seconds": float(pass1_seconds),
            "clip_threshold_seconds": float(clip_threshold_seconds),
            "clipped_stats_seconds": float(clipped_stats_seconds),
            "signature_seconds": float(signature_seconds),
            "pass2_seconds": float(pass2_seconds),
            "total_seconds": float(time.perf_counter() - stage_start),
        },
        "max_rss_kb": max_rss_kb(),
    }
    result = FastApproxPsResult(
        scores=scores.reshape(-1, 1),
        valid_mask=valid_mask,
        obs_index=obs_index,
        labels=labels,
        control_mask=labels == ctrl_name,
        metadata=metadata,
    )
    if output_dir is None:
        return result
    return write_ps_score_fast_approx_output(result, output_dir=output_dir, dataset_path=dataset_path)


def write_ps_score_fast_approx_output(
    result: FastApproxPsResult,
    *,
    output_dir: str | Path,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    score_path = output_path / "ps-score-fast-approx.csv"
    manifest_path = output_path / "ps-score-fast-approx-manifest.json"

    table = _score_result_dataframe(result)
    table.to_csv(score_path, index=False)

    manifest = dict(result.metadata)
    manifest.update(
        {
            "dataset_path": None if dataset_path is None else str(dataset_path),
            "score_output_format": "csv_long",
            "score_count": int(table.shape[0]),
            "score_output_paths": {"scores": str(score_path)},
        }
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def _score_result_dataframe(result: FastApproxPsResult) -> Any:
    scored_rows = np.flatnonzero(result.valid_mask)
    perturbations = result.metadata["selected_perturbations"]
    lookup = {perturbation: index for index, perturbation in enumerate(perturbations)}
    perturbation_indices = np.asarray([lookup[str(label)] for label in result.labels[scored_rows]], dtype=np.int32)
    return ps_score_long_dataframe(
        obs_index=result.obs_index,
        control_mask=result.control_mask,
        valid_mask=result.valid_mask,
        scores=result.scores[scored_rows, 0].astype(np.float64, copy=False),
        cell_indices=scored_rows,
        perturbation_indices=perturbation_indices,
        perturbations=perturbations,
        ctrl_name=result.metadata["control_label"],
    )


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
    parser.add_argument("--clip-quantile", type=float)
    parser.add_argument("--clip-bins", type=int, default=DEFAULT_CLIP_BINS)
    parser.add_argument("--min-cells-per-perturbation", type=int, default=2)
    parser.add_argument("--target-basis", choices=["per_perturbation", "union"], default="per_perturbation")
    parser.add_argument("--perturbation", action="append", dest="perturbations")
    parser.add_argument("--null-label", action="append", dest="null_labels")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run_ps_score_fast_approx(
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
        clip_quantile=args.clip_quantile,
        clip_bins=args.clip_bins,
        target_basis=args.target_basis,
    )


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


def _build_signature(
    *,
    projection_gene_indices: np.ndarray,
    perturb_mean: np.ndarray,
    control_mean: np.ndarray,
    drop_zero_genes: bool,
) -> _Signature | None:
    gene_indices = np.asarray(projection_gene_indices, dtype=np.int64)
    control = np.asarray(control_mean, dtype=np.float64)
    beta = np.asarray(perturb_mean, dtype=np.float64) - control
    if drop_zero_genes:
        nonzero = np.abs(beta) > 0.0
        gene_indices = gene_indices[nonzero]
        control = control[nonzero]
        beta = beta[nonzero]
    beta_norm = float(np.linalg.norm(beta)) if beta.size else 0.0
    if beta_norm == 0.0:
        return None
    return _Signature(
        gene_indices=gene_indices,
        beta=beta.astype(np.float64, copy=False),
        control_mean=control.astype(np.float64, copy=False),
        beta_norm=beta_norm,
    )


__all__ = [
    "FastApproxPsResult",
    "build_parser",
    "main",
    "run_ps_score_fast_approx",
    "write_ps_score_fast_approx_output",
]
