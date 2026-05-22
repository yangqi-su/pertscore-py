"""Fast exact PS scores from backed AnnData streams.

Perturbations always come from one obs column. In single mode each non-control
label is one perturbation; in multi-label mode labels like `pertA+pertB` are
parsed as active perturbation sets. Target genes are either union DEGs selected
from streamed Welch statistics, or an existing `adata.var['highly_variable']`
set.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve

from .stats import extract_anndata_matrix, get_obs_column, top_k_indices, validate_layer, welch_t_scores_from_stats
from .types import StreamFeatureStats


DEFAULT_TARGET_SUM = 1e4
DEFAULT_CLIP_BINS = 2048


@dataclass(frozen=True)
class ExactFastPsResult:
    scores: np.ndarray
    valid_mask: np.ndarray
    obs_index: np.ndarray
    labels: np.ndarray
    control_mask: np.ndarray
    beta: np.ndarray
    union_gene_indices: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExactFastMultiLabelPsResult:
    scores: np.ndarray
    cell_indices: np.ndarray
    perturbation_indices: np.ndarray
    perturbations: list[str]
    valid_mask: np.ndarray
    obs_index: np.ndarray
    control_mask: np.ndarray
    beta: np.ndarray
    union_gene_indices: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _ParsedPerturbations:
    perturbations: list[str]
    guides: sparse.csr_matrix
    control_mask: np.ndarray
    model_mask: np.ndarray
    active_counts: np.ndarray


def run_ps_score_exact_fast(
    data: Any,
    *,
    mode: Literal["single", "multilabel"] = "single",
    perturb_column: str,
    ctrl_name: str,
    output_dir: str | Path | None = None,
    layer: str | None = None,
    perturbations: Sequence[str] | None = None,
    target_mode: Literal["union_deg", "hvg"] = "union_deg",
    target_gene_max: int = 500,
    chunk_size: int = 8192,
    lr_lambda: float = 0.01,
    score_lambda: float = 0.0,
    scale_factor: float = 3.0,
    target_sum: float = DEFAULT_TARGET_SUM,
    rank_by_abs_t: bool = True,
    scale_score: bool = True,
    clip_quantile: float | None = None,
    clip_bins: int = DEFAULT_CLIP_BINS,
) -> ExactFastPsResult | ExactFastMultiLabelPsResult | dict[str, Any]:
    """Run exact-fast PS scoring from an AnnData object or backed h5ad path."""

    validate_layer(layer)
    assert clip_quantile is None or 0.0 < clip_quantile <= 1.0
    assert clip_bins >= 2 and type(clip_bins) == int
    if target_mode not in {"union_deg", "hvg"}:
        raise ValueError("target_mode must be 'union_deg' or 'hvg'")
    if mode not in {"single", "multilabel"}:
        raise ValueError("mode must be 'single' or 'multilabel'")

    dataset_path = Path(data) if isinstance(data, (str, Path)) else None
    adata = ad.read_h5ad(dataset_path, backed="r") if dataset_path is not None else data
    labels = _clean_obs_labels(adata, perturb_column)
    obs_index = np.asarray(adata.obs_names, dtype=object)
    matrix = extract_anndata_matrix(adata, layer=layer)
    parsed = _parse_perturbations(labels, mode=mode, ctrl_name=ctrl_name, perturbations=perturbations)
    var_names = np.asarray(adata.var_names, dtype=object)
    stage_start = time.perf_counter()

    full_stats_seconds = 0.0
    full_stats: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if target_mode == "union_deg":
        stats_start = time.perf_counter()
        full_stats = _collect_group_stats(
            matrix,
            guides=parsed.guides,
            control_mask=parsed.control_mask,
            chunk_size=chunk_size,
            target_sum=target_sum,
        )
        full_stats_seconds = time.perf_counter() - stats_start
        counts = full_stats[2]
    else:
        counts = _group_counts(parsed.guides, parsed.control_mask)
    _check_group_counts(counts, parsed.perturbations)

    targets, target_metadata, target_source = _select_target_genes(
        adata=adata,
        target_mode=target_mode,
        selected_perturbations=parsed.perturbations,
        var_names=var_names,
        counts=counts,
        full_stats=full_stats,
        target_gene_max=target_gene_max,
        rank_by_abs_t=rank_by_abs_t,
    )
    union_gene_indices = _ordered_union_indices(targets.values())

    clip_values = None
    clip_threshold_seconds = 0.0
    if clip_quantile is not None:
        clip_start = time.perf_counter()
        clip_values = _estimate_histogram_clip_values(
            matrix,
            model_rows=parsed.model_mask,
            union_gene_indices=union_gene_indices,
            model_cell_count=int(np.count_nonzero(parsed.model_mask)),
            chunk_size=chunk_size,
            target_sum=target_sum,
            quantile=clip_quantile,
            bins=clip_bins,
        )
        clip_threshold_seconds = time.perf_counter() - clip_start

    clipped_stats_seconds = 0.0
    ridge_stats_seconds = 0.0
    if mode == "single":
        if clip_values is None and full_stats is not None:
            beta_sums = full_stats[0][:, union_gene_indices]
        else:
            clipped_stats_start = time.perf_counter()
            beta_sums, _, counts = _collect_group_stats(
                matrix,
                guides=parsed.guides,
                control_mask=parsed.control_mask,
                gene_indices=union_gene_indices,
                clip_values=clip_values,
                chunk_size=chunk_size,
                target_sum=target_sum,
            )
            clipped_stats_seconds = time.perf_counter() - clipped_stats_start
        ridge_start = time.perf_counter()
        beta = _solve_single_label_ridge(
            total_rhs=beta_sums.sum(axis=0),
            perturbation_rhs=beta_sums[1:],
            perturbation_counts=counts[1:].astype(np.float64, copy=False),
            model_cell_count=float(counts.sum()),
            lr_lambda=float(lr_lambda),
        )
        ridge_seconds = time.perf_counter() - ridge_start
        scores, valid_mask, max_score_by_perturbation, scoring_seconds = _score_single_label(
            matrix,
            guides=parsed.guides,
            beta=beta,
            union_gene_indices=union_gene_indices,
            clip_values=clip_values,
            chunk_size=chunk_size,
            target_sum=target_sum,
            score_lambda=score_lambda,
            scale_factor=scale_factor,
            scale_score=scale_score,
        )
    else:
        ridge_stats_start = time.perf_counter()
        xtx, xty = _collect_multilabel_ridge_stats(
            matrix,
            guides=parsed.guides,
            control_mask=parsed.control_mask,
            union_gene_indices=union_gene_indices,
            clip_values=clip_values,
            chunk_size=chunk_size,
            target_sum=target_sum,
        )
        ridge_stats_seconds = time.perf_counter() - ridge_stats_start
        ridge_start = time.perf_counter()
        beta = np.asarray(spsolve(xtx + sparse.eye(xtx.shape[0], format="csc") * float(lr_lambda), xty), dtype=np.float64)
        if beta.ndim == 1:
            beta = beta[:, None]
        ridge_seconds = time.perf_counter() - ridge_start
        (
            scores,
            cell_indices,
            perturbation_indices,
            valid_mask,
            max_score_by_perturbation,
            scoring_seconds,
        ) = _score_multilabel(
            matrix,
            guides=parsed.guides,
            beta=beta,
            union_gene_indices=union_gene_indices,
            clip_values=clip_values,
            chunk_size=chunk_size,
            target_sum=target_sum,
            score_lambda=score_lambda,
            scale_factor=scale_factor,
            scale_score=scale_score,
        )

    _add_score_metadata(
        target_metadata,
        parsed.perturbations,
        beta=beta,
        max_score_by_perturbation=max_score_by_perturbation,
        scale_score=scale_score,
    )
    metadata = {
        "algorithm": "ps_score_exact_fast" if mode == "single" else "ps_score_exact_fast_multilabel",
        "mode": mode,
        "input_type": f"anndata-{mode}-backed-stream",
        "layer": layer,
        "perturb_column": perturb_column,
        "control_label": ctrl_name,
        "target_mode": target_mode,
        "target_gene_source": target_source["mode"],
        "target_gene_source_detail": target_source,
        "target_gene_max": int(target_gene_max),
        "rank_by_abs_t": bool(rank_by_abs_t),
        "quantile_clip": clip_quantile is not None,
        "clip_quantile": None if clip_quantile is None else float(clip_quantile),
        "clip_method": None if clip_quantile is None else "streaming_histogram",
        "clip_bins": None if clip_quantile is None else int(clip_bins),
        "chunk_size": int(chunk_size),
        "target_sum": float(target_sum),
        "lr_lambda": float(lr_lambda),
        "score_lambda": float(score_lambda),
        "scale_factor": float(scale_factor),
        "scale_score": bool(scale_score),
        "selected_perturbations": list(parsed.perturbations),
        "control_cell_count": int(counts[0]),
        "perturbation_cell_counts": {perturbation: int(counts[index + 1]) for index, perturbation in enumerate(parsed.perturbations)},
        "union_target_gene_count": int(union_gene_indices.shape[0]),
        "union_target_genes": [str(var_names[index]) for index in union_gene_indices],
        "target_gene_metadata": target_metadata,
        "beta_shape": tuple(int(value) for value in beta.shape),
        "valid_scored_cell_count": int(np.count_nonzero(valid_mask)),
        "timings": {
            "target_stats_seconds": float(full_stats_seconds),
            "clip_threshold_seconds": float(clip_threshold_seconds),
            "clipped_sufficient_stats_seconds": float(clipped_stats_seconds),
            "ridge_sufficient_stats_seconds": float(ridge_stats_seconds),
            "ridge_solve_seconds": float(ridge_seconds),
            "scoring_seconds": float(scoring_seconds),
            "total_seconds": float(time.perf_counter() - stage_start),
        },
        "max_rss_kb": _max_rss_kb(),
    }
    if mode == "single":
        metadata["score_vector_shape"] = (int(labels.shape[0]), 1)
        result = ExactFastPsResult(
            scores=scores,
            valid_mask=valid_mask,
            obs_index=obs_index,
            labels=labels,
            control_mask=parsed.control_mask,
            beta=beta,
            union_gene_indices=union_gene_indices,
            metadata=metadata,
        )
    else:
        metadata.update(
            {
                "guide_multiplicity": _summarize_guide_multiplicity(parsed.active_counts),
                "scored_pair_count": int(scores.shape[0]),
            }
        )
        result = ExactFastMultiLabelPsResult(
            scores=scores,
            cell_indices=cell_indices,
            perturbation_indices=perturbation_indices,
            perturbations=list(parsed.perturbations),
            valid_mask=valid_mask,
            obs_index=obs_index,
            control_mask=parsed.control_mask,
            beta=beta,
            union_gene_indices=union_gene_indices,
            metadata=metadata,
        )

    if output_dir is None:
        return result
    return write_ps_score_exact_fast_output(result, output_dir=output_dir, dataset_path=dataset_path)


def write_ps_score_exact_fast_output(
    result: ExactFastPsResult | ExactFastMultiLabelPsResult,
    *,
    output_dir: str | Path,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    score_path = output_path / "ps-score-exact-fast.csv"
    manifest_path = output_path / "ps-score-exact-fast-manifest.json"

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
    _write_json(manifest, manifest_path)
    return manifest


def _score_result_dataframe(result: ExactFastPsResult | ExactFastMultiLabelPsResult) -> pd.DataFrame:
    if isinstance(result, ExactFastMultiLabelPsResult):
        perturbation_names = np.asarray(result.perturbations, dtype=object)
        control_rows = np.flatnonzero(result.control_mask)
        missing_rows = np.flatnonzero(~result.control_mask & ~result.valid_mask)
        table = pd.concat(
            [
                pd.DataFrame(
                    {
                        "_row_order": result.cell_indices,
                        "_perturbation_order": result.perturbation_indices,
                        "obs_index": result.obs_index[result.cell_indices],
                        "ps_score": result.scores.astype(np.float64, copy=False),
                        "perturbation": perturbation_names[result.perturbation_indices],
                    }
                ),
                pd.DataFrame(
                    {
                        "_row_order": control_rows,
                        "_perturbation_order": np.full(control_rows.shape[0], -1, dtype=np.int32),
                        "obs_index": result.obs_index[control_rows],
                        "ps_score": np.zeros(control_rows.shape[0], dtype=np.float64),
                        "perturbation": np.full(control_rows.shape[0], result.metadata["control_label"], dtype=object),
                    }
                ),
                pd.DataFrame(
                    {
                        "_row_order": missing_rows,
                        "_perturbation_order": np.full(missing_rows.shape[0], -1, dtype=np.int32),
                        "obs_index": result.obs_index[missing_rows],
                        "ps_score": np.full(missing_rows.shape[0], np.nan, dtype=np.float64),
                        "perturbation": np.full(missing_rows.shape[0], None, dtype=object),
                    }
                ),
            ],
            ignore_index=True,
        )
        table.sort_values(["_row_order", "_perturbation_order"], kind="stable", inplace=True)
        return table[["obs_index", "ps_score", "perturbation"]]

    scores = np.full(result.obs_index.shape[0], np.nan, dtype=np.float64)
    perturbations = np.full(result.obs_index.shape[0], None, dtype=object)
    scores[result.control_mask] = 0.0
    perturbations[result.control_mask] = result.metadata["control_label"]
    scores[result.valid_mask] = result.scores[result.valid_mask, 0].astype(np.float64, copy=False)
    perturbations[result.valid_mask] = result.labels[result.valid_mask]
    return pd.DataFrame({"obs_index": result.obs_index, "ps_score": scores, "perturbation": perturbations})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["single", "multilabel"], default="single")
    parser.add_argument("--perturb-column", required=True)
    parser.add_argument("--ctrl-name", required=True)
    parser.add_argument("--layer")
    parser.add_argument("--target-mode", choices=["union_deg", "hvg"], default="union_deg")
    parser.add_argument("--target-gene-max", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--lr-lambda", type=float, default=0.01)
    parser.add_argument("--score-lambda", type=float, default=0.0)
    parser.add_argument("--scale-factor", type=float, default=3.0)
    parser.add_argument("--target-sum", type=float, default=DEFAULT_TARGET_SUM)
    parser.add_argument("--clip-quantile", type=float)
    parser.add_argument("--clip-bins", type=int, default=DEFAULT_CLIP_BINS)
    parser.add_argument("--perturbation", action="append", dest="perturbations")
    parser.add_argument("--rank-by-signed-t", action="store_true")
    parser.add_argument("--no-scale-score", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run_ps_score_exact_fast(
        args.dataset_path,
        mode=args.mode,
        output_dir=args.output_dir,
        perturb_column=args.perturb_column,
        ctrl_name=args.ctrl_name,
        layer=args.layer,
        perturbations=args.perturbations,
        target_mode=args.target_mode,
        target_gene_max=args.target_gene_max,
        chunk_size=args.chunk_size,
        lr_lambda=args.lr_lambda,
        score_lambda=args.score_lambda,
        scale_factor=args.scale_factor,
        target_sum=args.target_sum,
        rank_by_abs_t=not args.rank_by_signed_t,
        scale_score=not args.no_scale_score,
        clip_quantile=args.clip_quantile,
        clip_bins=args.clip_bins,
    )


def _clean_obs_labels(adata: Any, perturb_column: str) -> np.ndarray:
    raw = np.asarray(get_obs_column(adata.obs, perturb_column), dtype=object)
    labels: list[str] = []
    for row_index, value in enumerate(raw):
        if pd.isna(value):
            raise ValueError(f"Missing perturbation label at obs row {row_index}")
        label = str(value)
        if not label:
            raise ValueError(f"Empty perturbation label at obs row {row_index}")
        labels.append(label)
    return np.asarray(labels, dtype=object)


def _parse_perturbations(
    labels: np.ndarray,
    *,
    mode: str,
    ctrl_name: str,
    perturbations: Sequence[str] | None,
) -> _ParsedPerturbations:
    tokenized: list[list[str]] = []
    known: list[str] = []
    known_set: set[str] = set()
    control_mask = labels == ctrl_name
    for row_index, label in enumerate(labels):
        if label == ctrl_name:
            tokenized.append([])
            continue
        tokens = [str(label)] if mode == "single" else [token.strip() for token in str(label).split("+")]
        if any(not token for token in tokens):
            raise ValueError(f"Malformed perturbation value at obs row {row_index}: {label!r}")
        if ctrl_name in tokens:
            raise ValueError(f"Control label cannot appear inside a perturbation combination at obs row {row_index}")
        active = _ordered_unique(tokens)
        tokenized.append(active)
        for token in active:
            if token not in known_set:
                known.append(token)
                known_set.add(token)

    selected = _select_perturbations(known, perturbations)
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

    values = np.ones(len(rows), dtype=np.float64)
    guides = sparse.csr_matrix((values, (rows, columns)), shape=(labels.shape[0], len(selected)), dtype=np.float64)
    guides.sort_indices()
    active_counts = np.asarray(guides.getnnz(axis=1)).ravel().astype(np.int64, copy=False)
    return _ParsedPerturbations(perturbations=selected, guides=guides, control_mask=control_mask, model_mask=model_mask, active_counts=active_counts)


def _select_perturbations(known: Sequence[str], perturbations: Sequence[str] | None) -> list[str]:
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


def _ordered_unique(values: Any) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _select_target_genes(
    *,
    adata: Any,
    target_mode: str,
    selected_perturbations: Sequence[str],
    var_names: np.ndarray,
    counts: np.ndarray,
    full_stats: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    target_gene_max: int,
    rank_by_abs_t: bool,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]], dict[str, Any]]:
    if target_mode == "hvg":
        hvg = _hvg_indices(adata)
        targets = {perturbation: hvg for perturbation in selected_perturbations}
        source = {"mode": "hvg", "var_column": "highly_variable"}
    else:
        if full_stats is None:
            raise ValueError("union_deg target mode requires streamed full-gene group statistics")
        sums, squared_sums, _ = full_stats
        control_stats = StreamFeatureStats(count=int(counts[0]), sums=sums[0], squared_sums=squared_sums[0])
        targets = {}
        for perturbation_index, perturbation in enumerate(selected_perturbations, start=1):
            perturb_stats = StreamFeatureStats(
                count=int(counts[perturbation_index]),
                sums=sums[perturbation_index],
                squared_sums=squared_sums[perturbation_index],
            )
            t_scores = welch_t_scores_from_stats(perturb_stats, control_stats)
            targets[perturbation] = top_k_indices(
                t_scores,
                min(target_gene_max, t_scores.shape[0]),
                absolute=rank_by_abs_t,
            ).astype(np.int64, copy=False)
        source = {"mode": "union_deg", "rank_by_abs_t": bool(rank_by_abs_t)}

    metadata = {
        perturbation: {
            "cell_count": int(counts[index + 1]),
            "selected_gene_count": int(targets[perturbation].shape[0]),
            "selected_genes": [str(var_names[gene_index]) for gene_index in targets[perturbation]],
        }
        for index, perturbation in enumerate(selected_perturbations)
    }
    return targets, metadata, source


def _hvg_indices(adata: Any) -> np.ndarray:
    if "highly_variable" not in adata.var:
        raise ValueError("target_mode='hvg' requires adata.var['highly_variable']")
    mask = np.asarray(adata.var["highly_variable"], dtype=bool)
    indices = np.flatnonzero(mask).astype(np.int64, copy=False)
    if indices.size == 0:
        raise ValueError("adata.var['highly_variable'] does not contain any selected genes")
    return indices


def _check_group_counts(counts: np.ndarray, perturbations: Sequence[str]) -> None:
    if counts[0] == 0:
        raise ValueError("At least one control cell is required")
    missing = [perturbation for index, perturbation in enumerate(perturbations) if counts[index + 1] == 0]
    if missing:
        raise ValueError("Selected perturbations have no modeled cells: " + ", ".join(missing))


def _group_counts(guides: sparse.csr_matrix, control_mask: np.ndarray) -> np.ndarray:
    counts = np.zeros(guides.shape[1] + 1, dtype=np.int64)
    counts[0] = int(np.count_nonzero(control_mask))
    counts[1:] = np.asarray(guides.sum(axis=0)).ravel().astype(np.int64, copy=False)
    return counts


def _iter_chunks(
    matrix: Any,
    *,
    n_obs: int,
    chunk_size: int,
    target_sum: float,
    gene_indices: np.ndarray | None = None,
    clip_values: np.ndarray | None = None,
) -> Any:
    for start in range(0, n_obs, chunk_size):
        stop = min(start + chunk_size, n_obs)
        chunk = _log_normalize_chunk(matrix[start:stop], target_sum=target_sum)
        if gene_indices is not None:
            chunk = chunk[:, gene_indices]
        if clip_values is not None:
            chunk = _clip_matrix_columns(chunk, clip_values)
        yield start, stop, chunk


def _collect_group_stats(
    matrix: Any,
    *,
    guides: sparse.csr_matrix,
    control_mask: np.ndarray,
    chunk_size: int,
    target_sum: float,
    gene_indices: np.ndarray | None = None,
    clip_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_count = int(matrix.shape[1] if gene_indices is None else gene_indices.shape[0])
    group_count = guides.shape[1] + 1
    sums = np.zeros((group_count, feature_count), dtype=np.float64)
    squared_sums = np.zeros_like(sums)
    counts = np.zeros(group_count, dtype=np.int64)
    for start, stop, chunk in _iter_chunks(
        matrix,
        n_obs=guides.shape[0],
        chunk_size=chunk_size,
        target_sum=target_sum,
        gene_indices=gene_indices,
        clip_values=clip_values,
    ):
        chunk_control = control_mask[start:stop]
        if np.any(chunk_control):
            _add_group_stats(chunk[chunk_control], row=0, sums=sums, squared_sums=squared_sums, counts=counts)
        _add_multilabel_group_stats(chunk, guides[start:stop], sums=sums, squared_sums=squared_sums, counts=counts)
    return sums, squared_sums, counts


def _score_single_label(
    matrix: Any,
    *,
    guides: sparse.csr_matrix,
    beta: np.ndarray,
    union_gene_indices: np.ndarray,
    clip_values: np.ndarray | None,
    chunk_size: int,
    target_sum: float,
    score_lambda: float,
    scale_factor: float,
    scale_score: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    beta_norm_sq = np.einsum("ij,ij->i", beta[1:], beta[1:])
    baseline_projection = beta[1:] @ beta[0]
    codes = _single_codes_from_guides(guides)
    scores = np.zeros((guides.shape[0], 1), dtype=np.float32)
    valid_mask = np.zeros(guides.shape[0], dtype=bool)
    max_score_by_perturbation = np.zeros(beta.shape[0] - 1, dtype=np.float64)
    start_time = time.perf_counter()

    for start, stop, chunk in _iter_chunks(
        matrix,
        n_obs=guides.shape[0],
        chunk_size=chunk_size,
        target_sum=target_sum,
        gene_indices=union_gene_indices,
        clip_values=clip_values,
    ):
        chunk_codes = codes[start:stop]
        row_indices = np.arange(start, stop, dtype=np.int64)
        for code in np.unique(chunk_codes):
            if code < 0:
                continue
            denominator = beta_norm_sq[int(code)]
            if denominator <= 0.0:
                continue
            mask = chunk_codes == code
            projected = np.asarray(chunk[mask] @ beta[int(code) + 1], dtype=np.float64).ravel()
            raw = (projected - baseline_projection[int(code)] - score_lambda) / denominator
            clipped = np.clip(raw, 0.0, scale_factor) / scale_factor
            selected_rows = row_indices[mask]
            scores[selected_rows, 0] = clipped.astype(np.float32, copy=False)
            valid_mask[selected_rows] = True
            if clipped.size:
                max_score_by_perturbation[int(code)] = max(max_score_by_perturbation[int(code)], float(np.max(clipped)))

    if scale_score:
        valid_indices = np.flatnonzero(valid_mask & (codes >= 0))
        row_max = max_score_by_perturbation[codes[valid_indices]]
        nonzero = row_max > 0.0
        scores[valid_indices[nonzero], 0] /= row_max[nonzero].astype(np.float32, copy=False)
        scores[valid_indices[~nonzero], 0] = 0.0
    return scores, valid_mask, max_score_by_perturbation, time.perf_counter() - start_time


def _single_codes_from_guides(guides: sparse.csr_matrix) -> np.ndarray:
    if np.max(np.asarray(guides.getnnz(axis=1)).ravel(), initial=0) > 1:
        raise ValueError("single mode received cells with multiple perturbations")
    codes = np.full(guides.shape[0], -1, dtype=np.int32)
    coo = guides.tocoo()
    if coo.nnz:
        codes[coo.row] = coo.col.astype(np.int32, copy=False)
    return codes


def _score_multilabel(
    matrix: Any,
    *,
    guides: sparse.csr_matrix,
    beta: np.ndarray,
    union_gene_indices: np.ndarray,
    clip_values: np.ndarray | None,
    chunk_size: int,
    target_sum: float,
    score_lambda: float,
    scale_factor: float,
    scale_score: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    score_values: list[np.ndarray] = []
    cell_index_values: list[np.ndarray] = []
    perturbation_index_values: list[np.ndarray] = []
    max_score_by_perturbation = np.zeros(guides.shape[1], dtype=np.float64)
    valid_mask = np.zeros(guides.shape[0], dtype=bool)
    start_time = time.perf_counter()

    for start, stop, chunk in _iter_chunks(
        matrix,
        n_obs=guides.shape[0],
        chunk_size=chunk_size,
        target_sum=target_sum,
        gene_indices=union_gene_indices,
        clip_values=clip_values,
    ):
        row_indices = np.arange(start, stop, dtype=np.int64)
        for active_set, local_rows in _group_rows_by_active_set(guides[start:stop]).items():
            active_indices = np.asarray(active_set, dtype=np.int64)
            active_beta = beta[active_indices + 1]
            gram = active_beta @ active_beta.T
            rhs = np.asarray(chunk[local_rows] @ active_beta.T, dtype=np.float64) - (active_beta @ beta[0])[None, :]
            bounded = _solve_bounded_quadratic_scores(
                gram=gram,
                rhs=rhs,
                linear_penalty=float(score_lambda),
                upper=float(scale_factor),
            )
            normalized = bounded / float(scale_factor)
            global_rows = row_indices[local_rows]
            valid_mask[global_rows] = True
            for offset, perturbation_index in enumerate(active_indices):
                values = normalized[:, offset].astype(np.float32, copy=False)
                score_values.append(values)
                cell_index_values.append(global_rows.copy())
                perturbation_index_values.append(np.full(values.shape[0], int(perturbation_index), dtype=np.int32))
                if values.size:
                    max_score_by_perturbation[perturbation_index] = max(max_score_by_perturbation[perturbation_index], float(np.max(values)))

    if score_values:
        scores = np.concatenate(score_values).astype(np.float32, copy=False)
        cell_indices = np.concatenate(cell_index_values).astype(np.int64, copy=False)
        perturbation_indices = np.concatenate(perturbation_index_values).astype(np.int32, copy=False)
    else:
        scores = np.zeros(0, dtype=np.float32)
        cell_indices = np.zeros(0, dtype=np.int64)
        perturbation_indices = np.zeros(0, dtype=np.int32)

    if scale_score and scores.size:
        row_max = max_score_by_perturbation[perturbation_indices]
        nonzero = row_max > 0.0
        scores[nonzero] /= row_max[nonzero].astype(np.float32, copy=False)
        scores[~nonzero] = 0.0
    return scores, cell_indices, perturbation_indices, valid_mask, max_score_by_perturbation, time.perf_counter() - start_time


def _collect_multilabel_ridge_stats(
    matrix: Any,
    *,
    guides: sparse.csr_matrix,
    control_mask: np.ndarray,
    union_gene_indices: np.ndarray,
    clip_values: np.ndarray | None,
    chunk_size: int,
    target_sum: float,
) -> tuple[sparse.csc_matrix, np.ndarray]:
    perturbation_count = guides.shape[1]
    xty = np.zeros((perturbation_count + 1, union_gene_indices.shape[0]), dtype=np.float64)
    intercept_count = 0.0
    perturbation_counts = np.zeros(perturbation_count, dtype=np.float64)
    cooccurrence = sparse.csr_matrix((perturbation_count, perturbation_count), dtype=np.float64)

    for start, stop, chunk in _iter_chunks(
        matrix,
        n_obs=guides.shape[0],
        chunk_size=chunk_size,
        target_sum=target_sum,
        gene_indices=union_gene_indices,
        clip_values=clip_values,
    ):
        chunk_guides = guides[start:stop]
        active = np.asarray(chunk_guides.getnnz(axis=1)).ravel() > 0
        model_mask = control_mask[start:stop] | active
        if not np.any(model_mask):
            continue
        chunk = chunk[model_mask]
        chunk_guides = chunk_guides[model_mask]
        intercept_count += float(chunk.shape[0])
        xty[0] += _column_sums(chunk)
        perturbation_counts += np.asarray(chunk_guides.sum(axis=0)).ravel().astype(np.float64, copy=False)
        cooccurrence = cooccurrence + (chunk_guides.T @ chunk_guides).tocsr()
        _add_multilabel_xty(chunk, chunk_guides, xty=xty)

    top = sparse.hstack(
        [sparse.csr_matrix([[intercept_count]], dtype=np.float64), sparse.csr_matrix(perturbation_counts[None, :])],
        format="csr",
    )
    bottom = sparse.hstack([sparse.csr_matrix(perturbation_counts[:, None]), cooccurrence], format="csr")
    return sparse.vstack([top, bottom], format="csc"), xty


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
    totals = dense.sum(axis=1, keepdims=True)
    nonzero = totals[:, 0] > 0
    dense[nonzero] *= target_sum / totals[nonzero]
    dense[~nonzero] = 0.0
    return np.log1p(dense)


def _estimate_histogram_clip_values(
    matrix: Any,
    *,
    model_rows: np.ndarray,
    union_gene_indices: np.ndarray,
    model_cell_count: int,
    chunk_size: int,
    target_sum: float,
    quantile: float,
    bins: int,
) -> np.ndarray:
    if model_cell_count <= 0:
        raise ValueError("Cannot estimate clip values without model cells")
    max_value = float(np.log1p(target_sum))
    hist = np.zeros((union_gene_indices.shape[0], bins), dtype=np.uint32)
    nonzero_counts = np.zeros(union_gene_indices.shape[0], dtype=np.int64)
    for start, stop, chunk in _iter_chunks(
        matrix,
        n_obs=model_rows.shape[0],
        chunk_size=chunk_size,
        target_sum=target_sum,
        gene_indices=union_gene_indices,
    ):
        model_mask = model_rows[start:stop]
        if np.any(model_mask):
            _accumulate_nonzero_histogram(chunk[model_mask], hist=hist, nonzero_counts=nonzero_counts, max_value=max_value)
    zero_counts = np.full(union_gene_indices.shape[0], model_cell_count, dtype=np.int64) - nonzero_counts
    return _histogram_quantiles(hist, zero_counts=zero_counts, total_count=model_cell_count, quantile=quantile, max_value=max_value)


def _accumulate_nonzero_histogram(matrix: Any, *, hist: np.ndarray, nonzero_counts: np.ndarray, max_value: float) -> None:
    if sparse.issparse(matrix):
        coo = matrix.tocoo()
        positive = coo.data > 0.0
        if not np.any(positive):
            return
        columns = coo.col[positive]
        nonzero_counts += np.bincount(columns, minlength=hist.shape[0]).astype(np.int64, copy=False)
        np.add.at(hist, (columns, _histogram_bin_indices(coo.data[positive], bins=hist.shape[1], max_value=max_value)), 1)
        return
    dense = np.asarray(matrix, dtype=np.float64)
    nonzero_rows, nonzero_cols = np.nonzero(dense > 0.0)
    if nonzero_cols.size:
        nonzero_counts += np.bincount(nonzero_cols, minlength=hist.shape[0]).astype(np.int64, copy=False)
        np.add.at(hist, (nonzero_cols, _histogram_bin_indices(dense[nonzero_rows, nonzero_cols], bins=hist.shape[1], max_value=max_value)), 1)


def _histogram_bin_indices(values: np.ndarray, *, bins: int, max_value: float) -> np.ndarray:
    return np.clip(np.floor((np.asarray(values, dtype=np.float64) / max_value) * bins).astype(np.int64, copy=False), 0, bins - 1)


def _histogram_quantiles(hist: np.ndarray, *, zero_counts: np.ndarray, total_count: int, quantile: float, max_value: float) -> np.ndarray:
    edges = (np.arange(1, hist.shape[1] + 1, dtype=np.float64) * max_value) / float(hist.shape[1])
    position = float(total_count - 1) * quantile
    lower_rank = int(np.floor(position))
    upper_rank = int(np.ceil(position))
    fraction = position - float(lower_rank)
    clip_values = np.zeros(hist.shape[0], dtype=np.float64)
    for gene_index in range(hist.shape[0]):
        lower = _histogram_value_at_rank(hist[gene_index], zero_count=int(zero_counts[gene_index]), rank=lower_rank, edges=edges)
        upper = _histogram_value_at_rank(hist[gene_index], zero_count=int(zero_counts[gene_index]), rank=upper_rank, edges=edges)
        clip_values[gene_index] = lower + fraction * (upper - lower)
    return clip_values


def _histogram_value_at_rank(hist_row: np.ndarray, *, zero_count: int, rank: int, edges: np.ndarray) -> float:
    if rank < zero_count:
        return 0.0
    cumulative = np.cumsum(hist_row, dtype=np.int64)
    if cumulative.size == 0 or cumulative[-1] == 0:
        return 0.0
    bin_index = int(np.searchsorted(cumulative, rank - zero_count + 1, side="left"))
    return float(edges[min(bin_index, edges.shape[0] - 1)])


def _clip_matrix_columns(matrix: Any, clip_values: np.ndarray) -> Any:
    if sparse.issparse(matrix):
        work = matrix.tocsr(copy=True).astype(np.float64)
        if work.data.size:
            work.data = np.minimum(work.data, clip_values[work.indices])
            work.eliminate_zeros()
        return work
    return np.minimum(np.asarray(matrix, dtype=np.float64).copy(), clip_values[None, :])


def _add_group_stats(matrix: Any, *, row: int, sums: np.ndarray, squared_sums: np.ndarray, counts: np.ndarray) -> None:
    counts[row] += int(matrix.shape[0])
    if sparse.issparse(matrix):
        sums[row] += np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64, copy=False)
        squared_sums[row] += np.asarray(matrix.power(2).sum(axis=0)).ravel().astype(np.float64, copy=False)
        return
    dense = np.asarray(matrix, dtype=np.float64)
    sums[row] += dense.sum(axis=0)
    squared_sums[row] += np.square(dense).sum(axis=0)


def _add_multilabel_group_stats(matrix: Any, guides: sparse.csr_matrix, *, sums: np.ndarray, squared_sums: np.ndarray, counts: np.ndarray) -> None:
    coo = guides.tocoo()
    if coo.nnz == 0:
        return
    columns = coo.col
    rows = coo.row
    for column in np.unique(columns):
        _add_group_stats(matrix[rows[columns == column]], row=int(column) + 1, sums=sums, squared_sums=squared_sums, counts=counts)


def _add_multilabel_xty(matrix: Any, guides: sparse.csr_matrix, *, xty: np.ndarray) -> None:
    coo = guides.tocoo()
    if coo.nnz == 0:
        return
    columns = coo.col
    rows = coo.row
    for column in np.unique(columns):
        xty[int(column) + 1] += _column_sums(matrix[rows[columns == column]])


def _column_sums(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64, copy=False)
    return np.asarray(matrix, dtype=np.float64).sum(axis=0)


def _solve_single_label_ridge(
    *,
    total_rhs: np.ndarray,
    perturbation_rhs: np.ndarray,
    perturbation_counts: np.ndarray,
    model_cell_count: float,
    lr_lambda: float,
) -> np.ndarray:
    perturbation_denominator = perturbation_counts + lr_lambda
    weighted_rhs = ((perturbation_counts / perturbation_denominator)[:, None] * perturbation_rhs).sum(axis=0)
    intercept_denominator = (model_cell_count + lr_lambda) - np.sum(perturbation_counts * perturbation_counts / perturbation_denominator)
    beta0 = (total_rhs - weighted_rhs) / intercept_denominator
    perturbation_beta = (perturbation_rhs - perturbation_counts[:, None] * beta0[None, :]) / perturbation_denominator[:, None]
    return np.vstack([beta0[None, :], perturbation_beta])


def _group_rows_by_active_set(guides: sparse.csr_matrix) -> dict[tuple[int, ...], np.ndarray]:
    groups: dict[tuple[int, ...], list[int]] = {}
    indptr = guides.indptr
    indices = guides.indices
    for row_index in range(guides.shape[0]):
        active = tuple(int(index) for index in indices[indptr[row_index] : indptr[row_index + 1]])
        if active:
            groups.setdefault(active, []).append(row_index)
    return {key: np.asarray(rows, dtype=np.int64) for key, rows in groups.items()}


def _solve_bounded_quadratic_scores(*, gram: np.ndarray, rhs: np.ndarray, linear_penalty: float, upper: float) -> np.ndarray:
    rhs = np.asarray(rhs, dtype=np.float64)
    if rhs.ndim == 1:
        rhs = rhs[:, None]
    if gram.shape[0] == 1:
        denominator = float(gram[0, 0])
        if denominator <= 0.0:
            return np.zeros((rhs.shape[0], 1), dtype=np.float64)
        return np.clip((rhs[:, [0]] - linear_penalty) / denominator, 0.0, upper)
    if gram.shape[0] <= 4:
        return _solve_bounded_quadratic_scores_active_set(gram=gram, rhs=rhs, linear_penalty=linear_penalty, upper=upper)
    return _solve_bounded_quadratic_scores_lbfgsb(gram=gram, rhs=rhs, linear_penalty=linear_penalty, upper=upper)


def _solve_bounded_quadratic_scores_active_set(*, gram: np.ndarray, rhs: np.ndarray, linear_penalty: float, upper: float) -> np.ndarray:
    cell_count, variable_count = rhs.shape
    best = np.zeros((cell_count, variable_count), dtype=np.float64)
    best_objective = np.full(cell_count, np.inf, dtype=np.float64)
    for states in product((0, 1, 2), repeat=variable_count):
        states_array = np.asarray(states, dtype=np.int8)
        free = states_array == 0
        upper_fixed = states_array == 2
        fixed = ~free
        candidate = np.zeros((cell_count, variable_count), dtype=np.float64)
        if np.any(upper_fixed):
            candidate[:, upper_fixed] = upper
        if np.any(free):
            free_rhs = rhs[:, free] - linear_penalty
            if np.any(fixed):
                free_rhs -= candidate[:, fixed] @ gram[np.ix_(fixed, free)]
            gram_free = gram[np.ix_(free, free)]
            try:
                candidate[:, free] = np.linalg.solve(gram_free, free_rhs.T).T
            except np.linalg.LinAlgError:
                candidate[:, free] = np.linalg.lstsq(gram_free, free_rhs.T, rcond=None)[0].T
        feasible = np.all(candidate >= -1e-9, axis=1) & np.all(candidate <= upper + 1e-9, axis=1)
        if not np.any(feasible):
            continue
        candidate = np.clip(candidate, 0.0, upper)
        objective = _bounded_quadratic_objective(candidate, gram=gram, rhs=rhs, linear_penalty=linear_penalty)
        update = feasible & (objective < best_objective)
        if np.any(update):
            best[update] = candidate[update]
            best_objective[update] = objective[update]
    return best


def _solve_bounded_quadratic_scores_lbfgsb(*, gram: np.ndarray, rhs: np.ndarray, linear_penalty: float, upper: float) -> np.ndarray:
    scores = np.zeros_like(rhs, dtype=np.float64)
    bounds = [(0.0, upper)] * rhs.shape[1]
    for row_index, row_rhs in enumerate(rhs):
        def objective(value: np.ndarray) -> float:
            return float(0.5 * value @ gram @ value - row_rhs @ value + linear_penalty * np.sum(value))

        def gradient(value: np.ndarray) -> np.ndarray:
            return gram @ value - row_rhs + linear_penalty

        result = minimize(objective, np.zeros(rhs.shape[1], dtype=np.float64), jac=gradient, bounds=bounds, method="L-BFGS-B")
        scores[row_index] = np.clip(result.x, 0.0, upper)
    return scores


def _bounded_quadratic_objective(scores: np.ndarray, *, gram: np.ndarray, rhs: np.ndarray, linear_penalty: float) -> np.ndarray:
    return 0.5 * np.sum((scores @ gram) * scores, axis=1) - np.sum(rhs * scores, axis=1) + linear_penalty * np.sum(scores, axis=1)


def _add_score_metadata(
    metadata: dict[str, dict[str, Any]],
    perturbations: Sequence[str],
    *,
    beta: np.ndarray,
    max_score_by_perturbation: np.ndarray,
    scale_score: bool,
) -> None:
    beta_norm_sq = np.einsum("ij,ij->i", beta[1:], beta[1:])
    for index, perturbation in enumerate(perturbations):
        metadata[perturbation]["beta_norm_sq"] = float(beta_norm_sq[index])
        metadata[perturbation]["max_score_before_column_scale"] = float(max_score_by_perturbation[index])
        metadata[perturbation]["column_scaled"] = bool(scale_score and max_score_by_perturbation[index] > 0.0)


def _summarize_guide_multiplicity(active_counts: np.ndarray) -> dict[str, Any]:
    return {
        "min": int(np.min(active_counts)) if active_counts.size else 0,
        "max": int(np.max(active_counts)) if active_counts.size else 0,
        "mean": float(np.mean(active_counts)) if active_counts.size else 0.0,
        "zero_count": int(np.count_nonzero(active_counts == 0)),
        "single_count": int(np.count_nonzero(active_counts == 1)),
        "multi_count": int(np.count_nonzero(active_counts >= 2)),
    }


def _ordered_union_indices(groups: Any) -> np.ndarray:
    union: list[int] = []
    seen: set[int] = set()
    for group in groups:
        for index in group:
            key = int(index)
            if key not in seen:
                union.append(key)
                seen.add(key)
    return np.asarray(union, dtype=np.int64)


def _write_json(value: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _max_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


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
    "ExactFastMultiLabelPsResult",
    "ExactFastPsResult",
    "build_parser",
    "main",
    "run_ps_score_exact_fast",
    "write_ps_score_exact_fast_output",
]


if __name__ == "__main__":
    print(json.dumps(_to_jsonable(main()), indent=2, sort_keys=True))
