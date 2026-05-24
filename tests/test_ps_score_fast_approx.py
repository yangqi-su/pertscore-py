from __future__ import annotations

import json

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from perturb_effects.ps_score_fast_approx import main, run_ps_score_fast_approx


def _make_fast_approx_adata() -> AnnData:
    counts = np.array(
        [
            [5.0, 5.0, 0.0],
            [5.0, 5.0, 0.0],
            [9.0, 1.0, 0.0],
            [8.0, 2.0, 0.0],
            [1.0, 9.0, 0.0],
            [2.0, 8.0, 0.0],
            [4.0, 4.0, 0.0],
            [0.0, 0.0, 5.0],
        ],
        dtype=float,
    )
    adata = AnnData(X=sparse.csr_matrix(counts))
    adata.obs["perturbation"] = [
        "control",
        "control",
        "pertA",
        "pertA",
        "pertB",
        "pertB",
        "unassigned",
        "pertSolo",
    ]
    adata.obs_names = [
        "ctrl-1",
        "ctrl-2",
        "pert-a-1",
        "pert-a-2",
        "pert-b-1",
        "pert-b-2",
        "null-1",
        "solo-1",
    ]
    adata.var_names = ["g1", "g2", "g3"]
    return adata


def _make_observed_only_adata() -> AnnData:
    counts = np.array(
        [
            [5.0, 5.0],
            [5.0, 5.0],
            [9.0, 1.0],
            [8.0, 2.0],
            [1.0, 9.0],
            [1.0, 9.0],
            [2.0, 8.0],
            [9.0, 1.0],
        ],
        dtype=float,
    )
    adata = AnnData(X=sparse.csr_matrix(counts))
    adata.obs["perturbation"] = [
        "control",
        "control",
        "pertA",
        "pertA",
        "pertA",
        "pertB",
        "pertB",
        "pertB",
    ]
    adata.obs_names = [
        "ctrl-1",
        "ctrl-2",
        "pert-a-1",
        "pert-a-2",
        "pert-a-cross",
        "pert-b-1",
        "pert-b-2",
        "pert-b-cross",
    ]
    adata.var_names = ["g1", "g2"]
    return adata


def _make_mixed_label_adata() -> AnnData:
    counts = np.array(
        [
            [5.0, 5.0],
            [0.0, 0.0],
            [9.0, 1.0],
            [8.0, 2.0],
            [1.0, 9.0],
            [2.0, 8.0],
        ],
        dtype=float,
    )
    adata = AnnData(X=sparse.csr_matrix(counts))
    adata.obs["perturbation"] = ["control", np.nan, "pertA", "pertA", "pertB", "pertB"]
    adata.var_names = ["g1", "g2"]
    return adata


def _make_clip_demo_adata() -> AnnData:
    counts = np.array(
        [
            [5.0, 5.0],
            [5.0, 5.0],
            [10.0, 0.0],
            [7.0, 3.0],
            [0.0, 10.0],
            [3.0, 7.0],
        ],
        dtype=float,
    )
    adata = AnnData(X=sparse.csr_matrix(counts))
    adata.obs["perturbation"] = ["control", "control", "pertA", "pertA", "pertB", "pertB"]
    adata.var_names = ["g1", "g2"]
    return adata


def test_fast_approx_scores_have_expected_shape_and_zero_invalid_rows() -> None:
    result = run_ps_score_fast_approx(
        _make_fast_approx_adata(),
        perturb_column="perturbation",
        ctrl_name="control",
        null_labels=["unassigned"],
        top_n=1,
        chunk_size=3,
        min_cells_per_perturbation=2,
    )

    scores = result.scores[:, 0]

    assert result.scores.shape == (8, 1)
    assert result.scores.dtype == np.float32
    assert np.allclose(scores[:2], 0.0)
    assert np.isclose(scores[6], 0.0)
    assert np.isclose(scores[7], 0.0)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert result.metadata["valid_perturbation_count"] == 2
    assert result.metadata["skipped_perturbations"]["pertSolo"]["reason"] == "too-few-cells"
    assert result.valid_mask.tolist() == [False, False, True, True, True, True, False, False]


def test_fast_approx_scores_use_only_the_observed_perturbation_signature() -> None:
    result = run_ps_score_fast_approx(
        _make_observed_only_adata(),
        perturb_column="perturbation",
        ctrl_name="control",
        top_n=1,
        chunk_size=4,
        min_cells_per_perturbation=2,
    )

    assert np.isclose(result.scores[4, 0], 0.0)
    assert np.isclose(result.scores[7, 0], 0.0)
    assert result.scores[2, 0] > 0.0
    assert result.scores[5, 0] > 0.0
    assert result.metadata["valid_perturbation_count"] == 2
    assert "signature_metadata" not in result.metadata


def test_fast_approx_handles_mixed_string_and_nan_labels() -> None:
    result = run_ps_score_fast_approx(
        _make_mixed_label_adata(),
        perturb_column="perturbation",
        ctrl_name="control",
        top_n=1,
        chunk_size=3,
        min_cells_per_perturbation=2,
    )

    assert result.scores.shape == (6, 1)
    assert np.isclose(result.scores[1, 0], 0.0)
    assert result.metadata["invalid_label_count"] == 1
    assert result.valid_mask.tolist() == [False, False, True, True, True, True]


def test_fast_approx_default_options_match_explicit_defaults() -> None:
    implicit = run_ps_score_fast_approx(
        _make_observed_only_adata(),
        perturb_column="perturbation",
        ctrl_name="control",
        top_n=1,
        chunk_size=4,
        min_cells_per_perturbation=2,
    )
    explicit = run_ps_score_fast_approx(
        _make_observed_only_adata(),
        perturb_column="perturbation",
        ctrl_name="control",
        top_n=1,
        chunk_size=4,
        min_cells_per_perturbation=2,
        target_basis="per_perturbation",
        clip_quantile=None,
    )

    assert np.allclose(implicit.scores, explicit.scores)
    assert implicit.valid_mask.tolist() == explicit.valid_mask.tolist()
    assert implicit.metadata["union_target_genes"] == explicit.metadata["union_target_genes"]


def test_fast_approx_union_basis_uses_ordered_shared_union() -> None:
    result = run_ps_score_fast_approx(
        _make_observed_only_adata(),
        perturb_column="perturbation",
        ctrl_name="control",
        top_n=1,
        chunk_size=4,
        min_cells_per_perturbation=2,
        target_basis="union",
    )

    metadata = result.metadata
    assert metadata["target_basis"] == "union"
    assert metadata["union_target_gene_count"] == 2
    assert set(metadata["union_target_genes"]) == {"g1", "g2"}
    assert "signature_metadata" not in metadata


def test_fast_approx_histclip_changes_scores() -> None:
    unclipped = run_ps_score_fast_approx(
        _make_clip_demo_adata(),
        perturb_column="perturbation",
        ctrl_name="control",
        top_n=1,
        chunk_size=3,
        min_cells_per_perturbation=2,
    )
    clipped = run_ps_score_fast_approx(
        _make_clip_demo_adata(),
        perturb_column="perturbation",
        ctrl_name="control",
        top_n=1,
        chunk_size=3,
        min_cells_per_perturbation=2,
        clip_quantile=0.5,
        clip_bins=16,
    )

    assert clipped.metadata["quantile_clip"] is True
    assert clipped.metadata["clip_quantile"] == 0.5
    assert clipped.metadata["clip_bins"] == 16
    assert clipped.metadata["clip_method"] == "streaming_histogram"
    assert "signature_metadata" not in clipped.metadata
    assert not np.allclose(clipped.scores, unclipped.scores)


def test_fast_approx_main_writes_manifest_and_outputs(tmp_path) -> None:
    dataset_path = tmp_path / "toy.h5ad"
    output_dir = tmp_path / "fast-approx"
    _make_fast_approx_adata().write_h5ad(dataset_path)

    report = main(
        [
            "--dataset-path",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--perturb-column",
            "perturbation",
            "--ctrl-name",
            "control",
            "--null-label",
            "unassigned",
            "--top-n",
            "1",
            "--chunk-size",
            "3",
            "--target-basis",
            "union",
            "--clip-quantile",
            "0.5",
            "--clip-bins",
            "16",
        ]
    )

    manifest_path = output_dir / "ps-score-fast-approx-manifest.json"
    score_path = output_dir / "ps-score-fast-approx.csv"

    assert manifest_path.exists()
    assert score_path.exists()
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["dataset_path"] == str(dataset_path)
    assert tuple(saved["score_vector_shape"]) == (8, 1)
    assert saved["score_output_format"] == "csv_long"
    assert saved["score_count"] == 8
    assert saved["target_basis"] == "union"
    assert saved["clip_quantile"] == 0.5
    assert saved["clip_bins"] == 16
    assert saved["clip_method"] == "streaming_histogram"
    assert saved["union_target_gene_count"] == 2
    assert report["score_output_paths"]["scores"] == str(score_path)
    scores = pd.read_csv(score_path)
    assert list(scores.columns) == ["obs_index", "ps_score", "perturbation"]
    assert scores.shape == (8, 3)
    by_obs = scores.set_index("obs_index")
    assert by_obs.loc["ctrl-1", "ps_score"] == 0.0
    assert by_obs.loc["ctrl-1", "perturbation"] == "control"
    assert pd.isna(by_obs.loc["solo-1", "ps_score"])
    assert pd.isna(by_obs.loc["solo-1", "perturbation"])
    assert pd.isna(by_obs.loc["null-1", "ps_score"])
    assert pd.isna(by_obs.loc["null-1", "perturbation"])
