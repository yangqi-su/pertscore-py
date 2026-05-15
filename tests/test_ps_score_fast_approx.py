from __future__ import annotations

import json

import numpy as np
from anndata import AnnData
from scipy import sparse

from perturb_effects.ps_score_fast_approx import main, run_ps_score_fast_approx_anndata


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


def test_fast_approx_scores_have_expected_shape_and_zero_invalid_rows() -> None:
    result = run_ps_score_fast_approx_anndata(
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
    result = run_ps_score_fast_approx_anndata(
        _make_observed_only_adata(),
        perturb_column="perturbation",
        ctrl_name="control",
        top_n=1,
        chunk_size=4,
        min_cells_per_perturbation=2,
    )

    scores = result.metadata["signature_metadata"]

    assert np.isclose(result.scores[4, 0], 0.0)
    assert np.isclose(result.scores[7, 0], 0.0)
    assert result.scores[2, 0] > 0.0
    assert result.scores[5, 0] > 0.0
    assert scores["pertA"]["max_raw_score"] > 0.0
    assert scores["pertB"]["max_raw_score"] > 0.0
    assert scores["pertA"]["selected_gene_count"] == 1
    assert scores["pertB"]["selected_gene_count"] == 1


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
        ]
    )

    manifest_path = output_dir / "ps-score-fast-approx-manifest.json"
    score_path = output_dir / "ps-score-fast-approx.npy"
    valid_mask_path = output_dir / "ps-score-fast-approx-valid-mask.npy"

    assert manifest_path.exists()
    assert score_path.exists()
    assert valid_mask_path.exists()
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["dataset_path"] == str(dataset_path)
    assert tuple(saved["score_vector_shape"]) == (8, 1)
    assert report["score_output_paths"]["normalized_scores"] == str(score_path)
    assert np.load(score_path).shape == (8, 1)
