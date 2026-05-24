"""Benchmark harness for exact PS-score AnnData runs."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse

from .ps_score_exact import run_ps_score_exact_anndata


RUN_MODES = ("probe", "pilot", "full")
TARGET_GENE_SOURCES = ("provided", "scanpy_de", "hvg")


@dataclass(frozen=True)
class ExactPsBenchmarkConfig:
    output_dir: Path
    perturb_column: str | None = None
    ctrl_name: str | None = None
    dataset_label: str | None = None
    mode: str = "pilot"
    layer: str | None = None
    counts_layer: str | None = "counts"
    target_gene_source: str = "scanpy_de"
    hvg_key: str = "highly_variable"
    target_gene_min: int = 10
    target_gene_max: int = 50
    apply_gene_filter: bool = True
    gene_filter_min_fraction: float = 0.01
    apply_quantile_clip: bool = False
    clip_quantile: float = 0.95
    lr_lambda: float = 0.01
    score_lambda: float = 0.0
    scale_factor: float = 3.0
    scale_score: bool = True
    pilot_perturbation_count: int = 50
    min_cells_per_perturbation: int = 2
    random_seed: int = 0
    perturbations: tuple[str, ...] | None = None
    ensure_log1p: bool = False
    lognorm_layer: str | None = None
    write_scores: bool = False
    metrics_only: bool = False
    memory_cap_gb: float = 1024.0

    def validate(self) -> None:
        if self.mode not in RUN_MODES:
            raise ValueError(f"Unsupported mode {self.mode!r}; expected one of {RUN_MODES}")
        if self.target_gene_source not in TARGET_GENE_SOURCES:
            raise ValueError(
                f"Unsupported target_gene_source {self.target_gene_source!r}; expected one of {TARGET_GENE_SOURCES}"
            )
        if self.mode != "probe":
            if not self.perturb_column:
                raise ValueError("perturb_column is required for pilot/full benchmark runs")
            if not self.ctrl_name:
                raise ValueError("ctrl_name is required for pilot/full benchmark runs")
        if self.pilot_perturbation_count < 1:
            raise ValueError("pilot_perturbation_count must be positive")
        if self.min_cells_per_perturbation < 1:
            raise ValueError("min_cells_per_perturbation must be positive")
        if self.target_gene_min < 1 or self.target_gene_max < self.target_gene_min:
            raise ValueError("target_gene_min/target_gene_max must define a valid positive range")
        if self.memory_cap_gb <= 0:
            raise ValueError("memory_cap_gb must be positive")

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "perturb_column": self.perturb_column,
            "ctrl_name": self.ctrl_name,
            "dataset_label": self.dataset_label,
            "mode": self.mode,
            "layer": self.layer,
            "counts_layer": self.counts_layer,
            "target_gene_source": self.target_gene_source,
            "hvg_key": self.hvg_key,
            "target_gene_min": self.target_gene_min,
            "target_gene_max": self.target_gene_max,
            "apply_gene_filter": self.apply_gene_filter,
            "gene_filter_min_fraction": self.gene_filter_min_fraction,
            "apply_quantile_clip": self.apply_quantile_clip,
            "clip_quantile": self.clip_quantile,
            "lr_lambda": self.lr_lambda,
            "score_lambda": self.score_lambda,
            "scale_factor": self.scale_factor,
            "scale_score": self.scale_score,
            "pilot_perturbation_count": self.pilot_perturbation_count,
            "min_cells_per_perturbation": self.min_cells_per_perturbation,
            "random_seed": self.random_seed,
            "perturbations": list(self.perturbations) if self.perturbations else None,
            "ensure_log1p": self.ensure_log1p,
            "lognorm_layer": self.lognorm_layer,
            "write_scores": self.write_scores,
            "metrics_only": self.metrics_only,
            "memory_cap_gb": self.memory_cap_gb,
        }


class JsonlProgressLogger:
    def __init__(self, path: Path) -> None:
        self.path = path

    def emit(self, event: str, **payload: Any) -> None:
        record = {"timestamp": _utc_timestamp(), "event": event, **_to_jsonable(payload)}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_exact_ps_benchmark(
    dataset_path: str | Path,
    config: ExactPsBenchmarkConfig,
) -> dict[str, Any]:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = JsonlProgressLogger(output_dir / "benchmark-progress.jsonl")
    progress.emit(
        "run_start",
        dataset_path=str(dataset_path),
        mode=config.mode,
        dataset_label=config.dataset_label,
    )

    dataset_path = Path(dataset_path)
    if config.mode == "probe":
        report = _run_probe_only(dataset_path=dataset_path, config=config, progress=progress)
    else:
        load_stage: dict[str, Any] = {}
        adata = _run_stage(
            "load",
            load_stage,
            progress,
            lambda: ad.read_h5ad(dataset_path),
        )
        try:
            report = _run_benchmark_core(
                adata=adata,
                config=config,
                progress=progress,
                dataset_path=dataset_path,
                stage_reports={"load": load_stage},
            )
        finally:
            _close_adata(adata)

    progress.emit("run_complete", summary_path=str(output_dir / "benchmark-summary.json"))
    return report


def run_exact_ps_benchmark_from_adata(
    adata: Any,
    config: ExactPsBenchmarkConfig,
) -> dict[str, Any]:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = JsonlProgressLogger(output_dir / "benchmark-progress.jsonl")
    progress.emit(
        "run_start",
        dataset_path=None,
        mode=config.mode,
        dataset_label=config.dataset_label,
    )
    report = _run_benchmark_core(
        adata=adata,
        config=config,
        progress=progress,
        dataset_path=None,
        stage_reports={
            "load": {
                "seconds": 0.0,
                "status": "in-memory",
                "max_rss_kb": _max_rss_kb(),
            }
        },
    )
    progress.emit("run_complete", summary_path=str(output_dir / "benchmark-summary.json"))
    return report


def probe_h5ad_metadata(
    dataset_path: str | Path,
    *,
    dataset_label: str | None = None,
    perturb_column: str | None = None,
    ctrl_name: str | None = None,
    layer: str | None = None,
    counts_layer: str | None = "counts",
    min_cells_per_perturbation: int = 2,
) -> dict[str, Any]:
    adata = ad.read_h5ad(Path(dataset_path), backed="r")
    try:
        return probe_anndata_metadata(
            adata,
            dataset_label=dataset_label,
            perturb_column=perturb_column,
            ctrl_name=ctrl_name,
            layer=layer,
            counts_layer=counts_layer,
            min_cells_per_perturbation=min_cells_per_perturbation,
        )
    finally:
        _close_adata(adata)


def probe_anndata_metadata(
    adata: Any,
    *,
    dataset_label: str | None = None,
    perturb_column: str | None = None,
    ctrl_name: str | None = None,
    layer: str | None = None,
    counts_layer: str | None = "counts",
    min_cells_per_perturbation: int = 2,
) -> dict[str, Any]:
    matrix = _resolve_matrix(adata, layer=layer)
    expression_nnz = _safe_nnz(matrix)
    shape = tuple(int(value) for value in matrix.shape)
    total_values = shape[0] * shape[1]
    density = None if expression_nnz is None or total_values == 0 else expression_nnz / total_values
    value_profile = _sample_matrix_value_profile(matrix)

    obs_columns = [str(column) for column in getattr(adata.obs, "columns", [])]
    var_columns = [str(column) for column in getattr(adata.var, "columns", [])]
    perturbation_counts: dict[str, int] = {}
    eligible_cell_counts: dict[str, int] = {}
    control_count = None
    control_candidates: list[str] = []
    if perturb_column and perturb_column in getattr(adata.obs, "columns", []):
        labels = np.asarray(adata.obs[perturb_column], dtype=object)
        values, counts = np.unique(labels, return_counts=True)
        perturbation_counts = {str(value): int(count) for value, count in zip(values, counts, strict=True)}
        eligible_cell_counts = {
            label: count
            for label, count in perturbation_counts.items()
            if label != ctrl_name and count >= min_cells_per_perturbation
        }
        if ctrl_name is not None:
            control_count = int(perturbation_counts.get(ctrl_name, 0))
        control_candidates = [
            label
            for label in perturbation_counts
            if any(token in label.lower() for token in ("ctrl", "control", "nt", "neg"))
        ]

    return {
        "dataset_label": dataset_label,
        "shape": shape,
        "n_obs": shape[0],
        "n_vars": shape[1],
        "expression_layer": layer,
        "counts_layer": counts_layer,
        "available_layers": sorted(str(key) for key in getattr(adata.layers, "keys", lambda: [])()),
        "expression_matrix_format": _matrix_format(matrix),
        "expression_nnz": expression_nnz,
        "expression_density": density,
        "value_profile": value_profile,
        "obs_columns": obs_columns,
        "var_columns": var_columns,
        "perturbation_column": perturb_column,
        "perturbation_column_candidates": [
            column
            for column in obs_columns
            if any(token in column.lower() for token in ("pert", "guide", "sg", "target"))
        ],
        "gene_identifier_candidates": [
            column
            for column in ["var_names", *var_columns]
            if any(token in column.lower() for token in ("gene", "symbol", "id"))
        ],
        "control_label": ctrl_name,
        "control_label_candidates": control_candidates,
        "control_count": control_count,
        "perturbation_counts": perturbation_counts,
        "eligible_perturbation_count": len(eligible_cell_counts),
        "eligible_cell_counts": eligible_cell_counts,
        "counts_layer_present": bool(counts_layer is not None and counts_layer in adata.layers),
    }


def sample_benchmark_perturbations(
    labels: Sequence[object],
    *,
    ctrl_name: str,
    max_perturbations: int,
    min_cells_per_perturbation: int,
    random_seed: int,
    requested_perturbations: Sequence[str] | None = None,
) -> dict[str, Any]:
    label_values = np.asarray(labels, dtype=object)
    unique, counts = np.unique(label_values, return_counts=True)
    count_lookup = {str(value): int(count) for value, count in zip(unique, counts, strict=True)}
    eligible = [
        label
        for label, count in sorted(count_lookup.items())
        if label != ctrl_name and count >= min_cells_per_perturbation
    ]
    if requested_perturbations is not None:
        selected = [str(label) for label in requested_perturbations]
        missing = [label for label in selected if label not in eligible]
        if missing:
            raise ValueError(f"Requested perturbations are not eligible: {missing}")
    elif len(eligible) <= max_perturbations:
        selected = eligible
    else:
        rng = np.random.default_rng(random_seed)
        permutation = rng.permutation(np.asarray(eligible, dtype=object))
        selected = [str(value) for value in permutation[:max_perturbations].tolist()]

    return {
        "selected_perturbations": selected,
        "selected_cell_counts": {label: count_lookup[label] for label in selected},
        "eligible_perturbations": eligible,
        "eligible_cell_counts": {label: count_lookup[label] for label in eligible},
        "control_count": int(count_lookup.get(ctrl_name, 0)),
        "random_seed": random_seed,
        "min_cells_per_perturbation": min_cells_per_perturbation,
    }


def estimate_full_run_feasibility(
    *,
    metadata_probe: Mapping[str, Any],
    sampling: Mapping[str, Any] | None,
    process_max_rss_kb: int,
    observed_runtime_seconds: float,
    memory_cap_gb: float,
) -> dict[str, Any]:
    eligible_cell_counts = {
        str(label): int(count)
        for label, count in dict(metadata_probe.get("eligible_cell_counts", {})).items()
    }
    selected_perturbations = [] if sampling is None else list(sampling.get("selected_perturbations", []))
    sampled_cell_counts = {
        str(label): int(eligible_cell_counts[label])
        for label in selected_perturbations
        if label in eligible_cell_counts
    }
    control_count = int(metadata_probe.get("control_count") or 0)
    full_perturbation_count = len(eligible_cell_counts)
    sampled_perturbation_count = len(sampled_cell_counts)
    pilot_scope_cell_count = control_count + sum(sampled_cell_counts.values())
    full_scope_cell_count = control_count + sum(eligible_cell_counts.values())
    pilot_output_rows = control_count * sampled_perturbation_count + sum(sampled_cell_counts.values())
    full_output_rows = control_count * full_perturbation_count + sum(eligible_cell_counts.values())
    cell_scope_scale = 1.0
    if pilot_scope_cell_count > 0:
        cell_scope_scale = full_scope_cell_count / pilot_scope_cell_count
    perturbation_scale = 1.0
    if sampled_perturbation_count > 0:
        perturbation_scale = full_perturbation_count / sampled_perturbation_count

    estimated_runtime_seconds = observed_runtime_seconds * perturbation_scale
    estimated_peak_rss_kb = int(process_max_rss_kb * max(1.0, cell_scope_scale))
    memory_cap_kb = int(memory_cap_gb * 1024 * 1024)

    return {
        "eligible_perturbation_count": full_perturbation_count,
        "sampled_perturbation_count": sampled_perturbation_count,
        "pilot_scope_cell_count": pilot_scope_cell_count,
        "full_scope_cell_count": full_scope_cell_count,
        "pilot_output_row_estimate": pilot_output_rows,
        "full_output_row_estimate": full_output_rows,
        "cell_scope_scale": cell_scope_scale,
        "perturbation_scale": perturbation_scale,
        "estimated_runtime_seconds": estimated_runtime_seconds,
        "estimated_peak_rss_kb": estimated_peak_rss_kb,
        "memory_cap_gb": memory_cap_gb,
        "feasible_under_memory_cap": estimated_peak_rss_kb <= memory_cap_kb,
        "estimation_basis": "Linear perturbation and cell-scope extrapolation from current benchmark summary.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-label")
    parser.add_argument("--mode", choices=RUN_MODES, default="pilot")
    parser.add_argument("--perturb-column")
    parser.add_argument("--ctrl-name")
    parser.add_argument("--layer")
    parser.add_argument("--counts-layer", default="counts")
    parser.add_argument("--target-gene-source", choices=TARGET_GENE_SOURCES, default="scanpy_de")
    parser.add_argument("--hvg-key", default="highly_variable")
    parser.add_argument("--target-gene-min", type=int, default=10)
    parser.add_argument("--target-gene-max", type=int, default=50)
    parser.add_argument("--pilot-perturbation-count", type=int, default=50)
    parser.add_argument("--min-cells-per-perturbation", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--perturbations")
    parser.add_argument("--ensure-log1p", action="store_true")
    parser.add_argument("--lognorm-layer")
    parser.add_argument("--write-scores", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--apply-quantile-clip", action="store_true")
    parser.add_argument("--memory-cap-gb", type=float, default=1024.0)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    perturbations = None
    if args.perturbations:
        perturbations = tuple(part.strip() for part in args.perturbations.split(",") if part.strip())
    config = ExactPsBenchmarkConfig(
        output_dir=Path(args.output_dir),
        perturb_column=args.perturb_column,
        ctrl_name=args.ctrl_name,
        dataset_label=args.dataset_label,
        mode=args.mode,
        layer=args.layer,
        counts_layer=args.counts_layer,
        target_gene_source=args.target_gene_source,
        hvg_key=args.hvg_key,
        target_gene_min=args.target_gene_min,
        target_gene_max=args.target_gene_max,
        apply_quantile_clip=args.apply_quantile_clip,
        pilot_perturbation_count=args.pilot_perturbation_count,
        min_cells_per_perturbation=args.min_cells_per_perturbation,
        random_seed=args.random_seed,
        perturbations=perturbations,
        ensure_log1p=args.ensure_log1p,
        lognorm_layer=args.lognorm_layer,
        write_scores=args.write_scores,
        metrics_only=args.metrics_only,
        memory_cap_gb=args.memory_cap_gb,
    )
    return run_exact_ps_benchmark(dataset_path=args.dataset_path, config=config)


def _run_probe_only(
    *,
    dataset_path: Path,
    config: ExactPsBenchmarkConfig,
    progress: JsonlProgressLogger,
) -> dict[str, Any]:
    stage_reports: dict[str, dict[str, Any]] = {}

    metadata_probe = _run_stage(
        "schema_resolution",
        stage_reports.setdefault("schema_resolution", {}),
        progress,
        lambda: probe_h5ad_metadata(
            dataset_path,
            dataset_label=config.dataset_label or dataset_path.stem,
            perturb_column=config.perturb_column,
            ctrl_name=config.ctrl_name,
            layer=config.layer,
            counts_layer=config.counts_layer,
            min_cells_per_perturbation=config.min_cells_per_perturbation,
        ),
    )
    stage_reports["load"] = {
        "seconds": 0.0,
        "status": "backed-probe",
        "max_rss_kb": _max_rss_kb(),
    }
    stage_reports["log_normalization"] = {
        "seconds": 0.0,
        "status": "skipped",
        "max_rss_kb": _max_rss_kb(),
    }
    stage_reports["output_report_write"] = {}
    report = {
        "dataset_path": str(dataset_path),
        "dataset_label": config.dataset_label or dataset_path.stem,
        "mode": config.mode,
        "config": config.to_report_dict(),
        "stage_timings": stage_reports,
        "metadata_probe": metadata_probe,
        "sampling": None,
        "result_summary": None,
        "slurm": _slurm_context(),
    }
    report["feasibility"] = estimate_full_run_feasibility(
        metadata_probe=metadata_probe,
        sampling=None,
        process_max_rss_kb=_max_rss_kb(),
        observed_runtime_seconds=stage_reports["schema_resolution"]["seconds"],
        memory_cap_gb=config.memory_cap_gb,
    )
    _write_report(report=report, output_dir=config.output_dir, progress=progress, stage_report=stage_reports["output_report_write"])
    return report


def _run_benchmark_core(
    *,
    adata: Any,
    config: ExactPsBenchmarkConfig,
    progress: JsonlProgressLogger,
    dataset_path: Path | None,
    stage_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    dataset_label = config.dataset_label or (dataset_path.stem if dataset_path is not None else "in-memory")
    metadata_probe = _run_stage(
        "schema_resolution",
        stage_reports.setdefault("schema_resolution", {}),
        progress,
        lambda: probe_anndata_metadata(
            adata,
            dataset_label=dataset_label,
            perturb_column=config.perturb_column,
            ctrl_name=config.ctrl_name,
            layer=config.layer,
            counts_layer=config.counts_layer,
            min_cells_per_perturbation=config.min_cells_per_perturbation,
        ),
    )

    labels = np.asarray(adata.obs[config.perturb_column], dtype=object)
    if config.mode == "pilot":
        sampling = sample_benchmark_perturbations(
            labels,
            ctrl_name=config.ctrl_name,
            max_perturbations=config.pilot_perturbation_count,
            min_cells_per_perturbation=config.min_cells_per_perturbation,
            random_seed=config.random_seed,
            requested_perturbations=config.perturbations,
        )
    else:
        sampling = sample_benchmark_perturbations(
            labels,
            ctrl_name=config.ctrl_name,
            max_perturbations=max(1, len(set(labels.tolist()))),
            min_cells_per_perturbation=config.min_cells_per_perturbation,
            random_seed=config.random_seed,
            requested_perturbations=config.perturbations,
        )
    progress.emit("sampling_complete", sampling=sampling)

    stage_reports["log_normalization"] = _prepare_expression_layer(
        adata=adata,
        config=config,
        progress=progress,
    )
    layer_to_use = config.lognorm_layer if config.ensure_log1p and config.lognorm_layer else config.layer

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column=config.perturb_column,
        ctrl_name=config.ctrl_name,
        layer=layer_to_use,
        counts_layer=config.counts_layer,
        perturbations=sampling["selected_perturbations"],
        target_gene_source=config.target_gene_source,
        hvg_key=config.hvg_key,
        target_gene_min=config.target_gene_min,
        target_gene_max=config.target_gene_max,
        apply_gene_filter=config.apply_gene_filter,
        gene_filter_min_fraction=config.gene_filter_min_fraction,
        apply_quantile_clip=config.apply_quantile_clip,
        clip_quantile=config.clip_quantile,
        lr_lambda=config.lr_lambda,
        score_lambda=config.score_lambda,
        scale_factor=config.scale_factor,
        scale_score=config.scale_score,
    )
    exact_metadata = dict(result.attrs.get("ps_score_exact", {}))
    for stage_name, seconds in exact_metadata.get("stage_timings", {}).items():
        stage_reports[stage_name] = {
            "seconds": float(seconds),
            "status": "success",
            "max_rss_kb": _max_rss_kb(),
        }

    result_summary = {
        "row_count": int(len(result)),
        "column_count": int(result.shape[1]),
        "target_perturbations": list(exact_metadata.get("perturbations", [])),
        "union_target_gene_count": int(exact_metadata.get("union_target_gene_count", 0)),
        "stage_timings": exact_metadata.get("stage_timings", {}),
        "score_metadata": exact_metadata.get("score_metadata", {}),
        "computation_path": exact_metadata.get("computation_path"),
        "sparse_fallback_reason": exact_metadata.get("sparse_fallback_reason"),
    }

    report = {
        "dataset_path": None if dataset_path is None else str(dataset_path),
        "dataset_label": dataset_label,
        "mode": config.mode,
        "config": config.to_report_dict(),
        "stage_timings": stage_reports,
        "metadata_probe": metadata_probe,
        "sampling": sampling,
        "result_summary": result_summary,
        "process_max_rss_kb": _max_rss_kb(),
        "slurm": _slurm_context(),
    }
    observed_runtime_seconds = float(
        stage_reports.get("target_gene_selection", {}).get("seconds", 0.0)
        + stage_reports.get("beta_solve", {}).get("seconds", 0.0)
        + stage_reports.get("scoring", {}).get("seconds", 0.0)
    )
    report["feasibility"] = estimate_full_run_feasibility(
        metadata_probe=metadata_probe,
        sampling=sampling,
        process_max_rss_kb=report["process_max_rss_kb"],
        observed_runtime_seconds=observed_runtime_seconds,
        memory_cap_gb=config.memory_cap_gb,
    )
    _write_report(report=report, output_dir=config.output_dir, progress=progress, stage_report=stage_reports.setdefault("output_report_write", {}), result=result if config.write_scores and not config.metrics_only else None)
    return report


def _prepare_expression_layer(
    *,
    adata: Any,
    config: ExactPsBenchmarkConfig,
    progress: JsonlProgressLogger,
) -> dict[str, Any]:
    if not config.ensure_log1p:
        return {"seconds": 0.0, "status": "skipped", "max_rss_kb": _max_rss_kb()}
    if not config.counts_layer:
        raise ValueError("counts_layer is required when ensure_log1p=True")
    if config.counts_layer not in adata.layers:
        raise ValueError(f"counts_layer {config.counts_layer!r} was not found in adata.layers")
    target_layer = config.lognorm_layer or config.layer
    if target_layer and target_layer in adata.layers:
        return {"seconds": 0.0, "status": "reused-existing-layer", "max_rss_kb": _max_rss_kb()}

    report: dict[str, Any] = {}

    def _lognorm() -> str | None:
        try:
            import scanpy as sc
        except ImportError as error:
            raise RuntimeError("scanpy is required when ensure_log1p=True") from error

        counts = adata.layers[config.counts_layer]
        work = ad.AnnData(X=counts.copy(), obs=adata.obs.copy(), var=adata.var.copy())
        sc.pp.normalize_total(work)
        sc.pp.log1p(work)
        if target_layer:
            adata.layers[target_layer] = work.X.copy()
        else:
            adata.X = work.X.copy()
        return target_layer

    produced_layer = _run_stage("log_normalization", report, progress, _lognorm)
    report["produced_layer"] = produced_layer
    return report


def _write_report(
    *,
    report: Mapping[str, Any],
    output_dir: Path,
    progress: JsonlProgressLogger,
    stage_report: dict[str, Any],
    result: Any | None = None,
) -> None:
    def _writer() -> None:
        summary_path = output_dir / "benchmark-summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(report), handle, indent=2, sort_keys=True)
            handle.write("\n")
        if result is not None:
            result.to_csv(output_dir / "benchmark-scores.csv", index=False)
        progress.emit("report_written", summary_path=str(summary_path), wrote_scores=result is not None)

    _run_stage("output_report_write", stage_report, progress, _writer)


def _run_stage(
    name: str,
    report: dict[str, Any],
    progress: JsonlProgressLogger,
    func: Any,
) -> Any:
    progress.emit("stage_start", stage=name, max_rss_kb=_max_rss_kb())
    start = time.perf_counter()
    try:
        value = func()
    except Exception as error:
        report.update(
            {
                "seconds": float(time.perf_counter() - start),
                "status": "failed",
                "max_rss_kb": _max_rss_kb(),
                "error": str(error),
            }
        )
        progress.emit("stage_error", stage=name, error=str(error), max_rss_kb=_max_rss_kb())
        raise
    report.update(
        {
            "seconds": float(time.perf_counter() - start),
            "status": "success",
            "max_rss_kb": _max_rss_kb(),
        }
    )
    progress.emit("stage_end", stage=name, seconds=report["seconds"], max_rss_kb=report["max_rss_kb"])
    return value


def _resolve_matrix(adata: Any, *, layer: str | None) -> Any:
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise ValueError(f"layer {layer!r} was not found in adata.layers")
    return adata.layers[layer]


def _sample_matrix_value_profile(
    matrix: Any,
    *,
    max_rows: int = 128,
    max_cols: int = 128,
    max_values: int = 4096,
) -> dict[str, Any]:
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return {"sample_size": 0, "state": "empty"}
    block = matrix[: min(matrix.shape[0], max_rows), : min(matrix.shape[1], max_cols)]
    if sparse.issparse(block):
        values = np.asarray(block.data, dtype=float)
    else:
        values = np.asarray(block, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"sample_size": 0, "state": "all-zero"}
    if values.size > max_values:
        values = values[:max_values]
    integer_like_fraction = float(np.mean(np.isclose(values, np.round(values), atol=1e-6)))
    state = "unknown"
    if np.all(values >= 0) and integer_like_fraction >= 0.98:
        state = "counts_like"
    elif np.all(values >= 0) and float(values.max(initial=0.0)) <= 20.0:
        state = "lognorm_like"
    return {
        "sample_size": int(values.size),
        "min": float(values.min(initial=0.0)),
        "max": float(values.max(initial=0.0)),
        "mean": float(values.mean()),
        "integer_like_fraction": integer_like_fraction,
        "state": state,
    }


def _matrix_format(matrix: Any) -> str:
    if sparse.issparse(matrix):
        return "sparse"
    name = type(matrix).__name__.lower()
    if any(token in name for token in ("sparse", "csr", "csc")):
        return "sparse"
    return "dense"


def _safe_nnz(matrix: Any) -> int | None:
    nnz = getattr(matrix, "nnz", None)
    if isinstance(nnz, (int, np.integer)):
        return int(nnz)
    return None


def _slurm_context() -> dict[str, Any]:
    return {
        "job_id": os.getenv("SLURM_JOB_ID"),
        "array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "node_name": os.getenv("SLURMD_NODENAME"),
    }


def _close_adata(adata: Any) -> None:
    file_handle = getattr(adata, "file", None)
    if file_handle is None:
        return
    close = getattr(file_handle, "close", None)
    if callable(close):
        close()


def _max_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
