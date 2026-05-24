from __future__ import annotations

import json

import numpy as np
from anndata import AnnData
from scipy import sparse

from perturb_effects.benchmark_exact_ps import (
    ExactPsBenchmarkConfig,
    main,
    probe_anndata_metadata,
    run_exact_ps_benchmark_from_adata,
    sample_benchmark_perturbations,
)


def _make_benchmark_adata() -> AnnData:
    counts = np.array(
        [
            [2.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
            [0.0, 4.0, 0.0, 1.0],
            [1.0, 0.0, 4.0, 0.0],
            [1.0, 0.0, 5.0, 0.0],
            [0.0, 1.0, 0.0, 4.0],
            [0.0, 1.0, 0.0, 5.0],
        ],
        dtype=float,
    )
    adata = AnnData(X=sparse.csr_matrix(counts))
    adata.layers["expr"] = sparse.csr_matrix(counts)
    adata.layers["counts"] = sparse.csr_matrix(counts)
    adata.obs["perturbation"] = [
        "control",
        "control",
        "pertA",
        "pertA",
        "pertB",
        "pertB",
        "pertC",
        "pertC",
    ]
    adata.obs_names = [
        "ctrl-1",
        "ctrl-2",
        "pert-a-1",
        "pert-a-2",
        "pert-b-1",
        "pert-b-2",
        "pert-c-1",
        "pert-c-2",
    ]
    adata.var_names = ["g1", "g2", "g3", "g4"]
    adata.var["highly_variable"] = [True, True, False, True]
    return adata


def test_sample_benchmark_perturbations_is_deterministic() -> None:
    adata = _make_benchmark_adata()

    first = sample_benchmark_perturbations(
        adata.obs["perturbation"],
        ctrl_name="control",
        max_perturbations=2,
        min_cells_per_perturbation=2,
        random_seed=11,
    )
    second = sample_benchmark_perturbations(
        adata.obs["perturbation"],
        ctrl_name="control",
        max_perturbations=2,
        min_cells_per_perturbation=2,
        random_seed=11,
    )

    assert first == second
    assert "control" not in first["selected_perturbations"]
    assert len(first["selected_perturbations"]) == 2


def test_probe_anndata_metadata_reports_sparse_counts_and_candidates() -> None:
    metadata = probe_anndata_metadata(
        _make_benchmark_adata(),
        dataset_label="toy",
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        counts_layer="counts",
        min_cells_per_perturbation=2,
    )

    assert metadata["dataset_label"] == "toy"
    assert metadata["expression_matrix_format"] == "sparse"
    assert metadata["eligible_perturbation_count"] == 3
    assert metadata["control_count"] == 2
    assert metadata["counts_layer_present"] is True
    assert metadata["value_profile"]["state"] == "counts_like"
    assert "perturbation" in metadata["perturbation_column_candidates"]


def test_benchmark_runner_writes_summary_and_progress_logs(tmp_path) -> None:
    dataset_path = tmp_path / "toy.h5ad"
    output_dir = tmp_path / "benchmark"
    _make_benchmark_adata().write_h5ad(dataset_path)

    report = main(
        [
            "--dataset-path",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--dataset-label",
            "toy",
            "--mode",
            "pilot",
            "--perturb-column",
            "perturbation",
            "--ctrl-name",
            "control",
            "--layer",
            "expr",
            "--target-gene-source",
            "hvg",
            "--hvg-key",
            "highly_variable",
            "--target-gene-min",
            "1",
            "--target-gene-max",
            "2",
            "--pilot-perturbation-count",
            "2",
            "--min-cells-per-perturbation",
            "2",
            "--random-seed",
            "7",
        ]
    )

    summary_path = output_dir / "benchmark-summary.json"
    progress_path = output_dir / "benchmark-progress.jsonl"

    assert summary_path.exists()
    assert progress_path.exists()
    saved = json.loads(summary_path.read_text(encoding="utf-8"))

    assert report["dataset_label"] == "toy"
    assert saved["sampling"]["selected_perturbations"] == report["sampling"]["selected_perturbations"]
    assert set(saved["stage_timings"]) >= {
        "load",
        "schema_resolution",
        "log_normalization",
        "target_gene_selection",
        "beta_solve",
        "scoring",
        "output_report_write",
    }
    assert saved["result_summary"]["computation_path"] == "in_memory_sparse_lbfgsb"
    assert saved["feasibility"]["sampled_perturbation_count"] == 2
    assert "exact_stage_event" in progress_path.read_text(encoding="utf-8")


def test_benchmark_runner_supports_in_memory_hvg_mode(tmp_path) -> None:
    report = run_exact_ps_benchmark_from_adata(
        _make_benchmark_adata(),
        ExactPsBenchmarkConfig(
            output_dir=tmp_path / "in-memory",
            dataset_label="toy-in-memory",
            mode="pilot",
            perturb_column="perturbation",
            ctrl_name="control",
            layer="expr",
            target_gene_source="hvg",
            hvg_key="highly_variable",
            target_gene_min=1,
            target_gene_max=2,
            pilot_perturbation_count=2,
            min_cells_per_perturbation=2,
            random_seed=3,
        ),
    )

    assert report["result_summary"]["union_target_gene_count"] >= 1
    assert report["stage_timings"]["log_normalization"]["status"] == "skipped"
