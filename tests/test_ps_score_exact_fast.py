from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from scipy.optimize import minimize

from perturb_effects.ps_score_exact import run_ps_score_exact_anndata
from perturb_effects.ps_score_exact_fast import main, run_ps_score_exact_fast


def test_exact_fast_histogram_clip_matches_dense_exact_clip() -> None:
    counts = np.asarray(
        [
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [20.0, 1.0, 0.0],
            [24.0, 2.0, 0.0],
            [28.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["control", "control", "control", "pertA", "pertA", "pertA"], dtype=object)
    obs_names = [f"cell_{index}" for index in range(counts.shape[0])]
    library_sizes = counts.sum(axis=1, keepdims=True)
    lognorm = np.log1p((counts / library_sizes) * 1e4)
    adata = AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame({"perturbation": labels}, index=obs_names),
        var=pd.DataFrame({"highly_variable": [True, True, False]}, index=["g1", "g2", "g3"]),
    )
    adata.layers["lognorm"] = lognorm

    exact = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="lognorm",
        counts_layer=None,
        perturbations=["pertA"],
        target_genes={"pertA": ["g1", "g2"]},
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=2,
        apply_gene_filter=False,
        apply_quantile_clip=True,
        clip_quantile=0.5,
        lr_lambda=0.01,
        score_lambda=0.0,
        scale_factor=3.0,
        scale_score=True,
        return_wide=True,
    )
    fast = run_ps_score_exact_fast(
        adata,
        mode="single",
        perturb_column="perturbation",
        ctrl_name="control",
        perturbations=["pertA"],
        target_mode="hvg",
        chunk_size=2,
        lr_lambda=0.01,
        score_lambda=0.0,
        scale_factor=3.0,
        target_sum=1e4,
        scale_score=True,
        clip_quantile=0.5,
        clip_bins=200000,
    )

    perturbed = labels == "pertA"
    exact_scores = exact.loc[np.asarray(obs_names, dtype=object)[perturbed], "pertA"].to_numpy(dtype=float)
    fast_scores = fast.scores[perturbed, 0].astype(float)
    assert fast.metadata["target_mode"] == "hvg"
    assert fast.metadata["quantile_clip"] is True
    assert np.allclose(fast_scores, exact_scores, atol=1e-3)


def test_multilabel_matches_single_label_when_one_perturbation_per_cell() -> None:
    counts = np.asarray(
        [
            [4.0, 1.0, 0.0],
            [5.0, 1.0, 0.0],
            [1.0, 5.0, 0.0],
            [1.0, 6.0, 0.0],
            [0.0, 1.0, 5.0],
            [0.0, 1.0, 6.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["control", "control", "pertA", "pertA", "pertB", "pertB"], dtype=object)
    adata = AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame({"perturbation": labels}, index=[f"cell_{index}" for index in range(counts.shape[0])]),
        var=pd.DataFrame({"highly_variable": [True, True, True]}, index=["g1", "g2", "g3"]),
    )
    kwargs = dict(
        perturb_column="perturbation",
        ctrl_name="control",
        perturbations=["pertA", "pertB"],
        target_mode="hvg",
        chunk_size=2,
        lr_lambda=0.01,
        score_lambda=0.0,
        scale_factor=3.0,
        target_sum=1e4,
        scale_score=False,
    )

    single = run_ps_score_exact_fast(adata, mode="single", **kwargs)
    multi = run_ps_score_exact_fast(adata, mode="multilabel", **kwargs)
    long_scores = {
        (int(cell), multi.perturbations[int(perturbation)]): float(score)
        for cell, perturbation, score in zip(multi.cell_indices, multi.perturbation_indices, multi.scores, strict=False)
    }
    for cell_index, label in enumerate(labels):
        if label != "control":
            assert np.isclose(long_scores[(cell_index, str(label))], float(single.scores[cell_index, 0]))


def test_multilabel_two_guides_matches_masked_lbfgsb() -> None:
    counts = np.asarray(
        [
            [8.0, 1.0, 1.0],
            [7.0, 1.0, 1.0],
            [1.0, 8.0, 1.0],
            [1.0, 7.0, 1.0],
            [1.0, 1.0, 8.0],
            [1.0, 1.0, 7.0],
            [1.0, 7.0, 7.0],
        ],
        dtype=np.float64,
    )
    labels = ["control", "control", "pertA", "pertA", "pertB", "pertB", "pertA+pertB"]
    adata = AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame({"perturbation": labels}, index=[f"cell_{index}" for index in range(counts.shape[0])]),
        var=pd.DataFrame({"highly_variable": [True, True, True]}, index=["g1", "g2", "g3"]),
    )
    result = run_ps_score_exact_fast(
        adata,
        mode="multilabel",
        perturb_column="perturbation",
        ctrl_name="control",
        target_mode="hvg",
        chunk_size=3,
        lr_lambda=0.01,
        score_lambda=0.2,
        scale_factor=10.0,
        target_sum=1e4,
        scale_score=False,
    )

    multi_cell = counts.shape[0] - 1
    observed = {
        result.perturbations[int(perturbation)]: float(score) * 10.0
        for cell, perturbation, score in zip(result.cell_indices, result.perturbation_indices, result.scores, strict=False)
        if int(cell) == multi_cell
    }
    y = np.log1p((counts[multi_cell] / counts[multi_cell].sum()) * 1e4)[result.union_gene_indices]
    centered = y - result.beta[0]
    active_beta = result.beta[[1, 2]]

    def objective(value: np.ndarray) -> float:
        residual = value @ active_beta - centered
        return float(0.5 * residual @ residual + 0.2 * np.sum(value))

    def gradient(value: np.ndarray) -> np.ndarray:
        residual = value @ active_beta - centered
        return residual @ active_beta.T + 0.2

    expected = minimize(
        objective,
        np.zeros(2, dtype=np.float64),
        jac=gradient,
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        method="L-BFGS-B",
    ).x
    assert set(observed) == {"pertA", "pertB"}
    assert np.allclose([observed["pertA"], observed["pertB"]], expected, atol=1e-6)
    assert result.metadata["guide_multiplicity"]["multi_count"] == 1


def test_selected_perturbations_fail_loudly_when_missing() -> None:
    adata = AnnData(
        X=sparse.csr_matrix(np.ones((3, 2), dtype=np.float64)),
        obs=pd.DataFrame({"perturbation": ["control", "pertA", "pertA"]}),
        var=pd.DataFrame({"highly_variable": [True, True]}, index=["g1", "g2"]),
    )
    with pytest.raises(ValueError, match="not present"):
        run_ps_score_exact_fast(
            adata,
            mode="single",
            perturb_column="perturbation",
            ctrl_name="control",
            perturbations=["pertB"],
            target_mode="hvg",
        )


def test_single_output_csv_marks_controls_and_unselected_cells(tmp_path) -> None:
    counts = np.asarray(
        [
            [5.0, 1.0],
            [4.0, 1.0],
            [1.0, 5.0],
            [3.0, 3.0],
        ],
        dtype=np.float64,
    )
    adata = AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame(
            {"perturbation": ["control", "control", "pertA", "pertB"]},
            index=[f"cell_{index}" for index in range(counts.shape[0])],
        ),
        var=pd.DataFrame({"highly_variable": [True, True]}, index=["g1", "g2"]),
    )
    output_dir = tmp_path / "out"

    manifest = run_ps_score_exact_fast(
        adata,
        mode="single",
        output_dir=output_dir,
        perturb_column="perturbation",
        ctrl_name="control",
        perturbations=["pertA"],
        target_mode="hvg",
        chunk_size=2,
        scale_score=False,
    )

    score_path = output_dir / "ps-score-exact-fast.csv"
    table = pd.read_csv(score_path)
    assert manifest["score_output_format"] == "csv_long"
    assert manifest["score_count"] == counts.shape[0]
    assert manifest["score_output_paths"] == {"scores": str(score_path)}
    assert list(table.columns) == ["obs_index", "ps_score", "perturbation"]
    assert np.allclose(table.loc[table["perturbation"] == "control", "ps_score"], 0.0)
    assert table.loc[table["obs_index"] == "cell_2", "perturbation"].item() == "pertA"
    assert pd.isna(table.loc[table["obs_index"] == "cell_3", "ps_score"].item())
    assert pd.isna(table.loc[table["obs_index"] == "cell_3", "perturbation"].item())


def test_multilabel_cli_writes_long_csv_outputs(tmp_path) -> None:
    counts = np.asarray(
        [
            [4.0, 1.0, 0.0],
            [5.0, 1.0, 0.0],
            [1.0, 5.0, 0.0],
            [1.0, 1.0, 5.0],
            [1.0, 5.0, 5.0],
        ],
        dtype=np.float64,
    )
    adata = AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame(
            {"perturbation": ["control", "control", "pertA", "pertB", "pertA+pertB"]},
            index=[f"cell_{index}" for index in range(counts.shape[0])],
        ),
        var=pd.DataFrame({"highly_variable": [True, True, True]}, index=["g1", "g2", "g3"]),
    )
    dataset_path = tmp_path / "tiny.h5ad"
    output_dir = tmp_path / "out"
    adata.write_h5ad(dataset_path)

    manifest = main(
        [
            "--mode",
            "multilabel",
            "--dataset-path",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--perturb-column",
            "perturbation",
            "--ctrl-name",
            "control",
            "--target-mode",
            "hvg",
            "--chunk-size",
            "2",
            "--no-scale-score",
        ]
    )

    score_path = output_dir / "ps-score-exact-fast.csv"
    assert manifest["score_output_format"] == "csv_long"
    assert manifest["score_count"] == 6
    assert manifest["scored_pair_count"] == 4
    assert manifest["score_output_paths"] == {"scores": str(score_path)}
    assert score_path.exists()
    table = pd.read_csv(score_path)
    assert list(table.columns) == ["obs_index", "ps_score", "perturbation"]
    assert table.shape[0] == 6
    assert np.allclose(table.loc[table["perturbation"] == "control", "ps_score"], 0.0)
    assert set(table.loc[table["obs_index"] == "cell_4", "perturbation"]) == {"pertA", "pertB"}
