from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from scipy.optimize import minimize

from perturb_effects.ps_score_exact import run_ps_score_exact_anndata
from perturb_effects.ps_score_exact_fast import _select_target_genes, main, run_ps_score_exact_fast


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
    adata = AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame({"perturbation": labels}, index=obs_names),
        var=pd.DataFrame({"highly_variable": [True, True, False]}, index=["g1", "g2", "g3"]),
    )

    exact = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer=None,
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
    exact_scores = exact.set_index("obs_index").loc[np.asarray(obs_names, dtype=object)[perturbed], "ps_score"].to_numpy(dtype=float)
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


def test_union_deg_logfc_threshold_filters_low_logfc_genes() -> None:
    counts = np.array([20, 20], dtype=np.int64)
    control_log = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.05, 0.05, 0.05],
            [0.00, 0.00, 0.00],
            [0.05, 0.05, 0.05],
        ],
        dtype=np.float64,
    )
    pert_log = np.array(
        [
            [0.40, 0.08, 0.25],
            [0.45, 0.09, 0.30],
            [0.40, 0.08, 0.25],
            [0.45, 0.09, 0.30],
        ],
        dtype=np.float64,
    )
    full_stats = (
        np.vstack([control_log.sum(axis=0), pert_log.sum(axis=0)]),
        np.vstack([np.square(control_log).sum(axis=0), np.square(pert_log).sum(axis=0)]),
        counts,
        None,
    )
    linear_sums = np.array(
        [
            [20.0, 20.0, 20.0],
            [40.0, 21.0, 32.0],
        ],
        dtype=np.float64,
    )

    targets, metadata, source = _select_target_genes(
        adata=None,
        target_mode="union_deg",
        selected_perturbations=["pertA"],
        var_names=np.asarray(["g1", "g2", "g3"], dtype=object),
        counts=counts,
        full_stats=full_stats,
        target_gene_max=3,
        rank_by_abs_t=True,
        linear_sums=linear_sums,
        logfc_threshold=0.1,
    )

    assert targets["pertA"].tolist() == [0, 2]
    assert metadata["pertA"]["logfc_filtered_gene_count"] == 2
    assert metadata["pertA"]["selected_genes"] == ["g1", "g3"]
    assert source["logfc_threshold"] == 0.1


def test_single_background_correction_uses_cluster_control_baseline() -> None:
    counts = np.asarray(
        [
            [9.0, 1.0],
            [8.0, 2.0],
            [2.0, 8.0],
            [1.0, 9.0],
            [9.0, 1.0],
            [4.0, 6.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["control", "control", "control", "control", "pertA", "pertA"], dtype=object)
    clusters = np.asarray(["c1", "c1", "c2", "c2", "c1", "c2"], dtype=object)
    adata = AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame({"perturbation": labels, "cluster": clusters}, index=[f"cell_{index}" for index in range(counts.shape[0])]),
        var=pd.DataFrame({"highly_variable": [True, True]}, index=["g1", "g2"]),
    )

    result = run_ps_score_exact_fast(
        adata,
        mode="single",
        perturb_column="perturbation",
        ctrl_name="control",
        perturbations=["pertA"],
        target_mode="hvg",
        background_cluster_column="cluster",
        chunk_size=2,
        lr_lambda=0.0,
        score_lambda=0.0,
        scale_factor=10.0,
        target_sum=100.0,
        scale_score=False,
    )

    lognorm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 100.0)
    cluster_codes = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
    background = np.vstack([lognorm[(labels == "control") & (clusters == "c1")].mean(axis=0), lognorm[(labels == "control") & (clusters == "c2")].mean(axis=0)])
    perturb_rows = labels == "pertA"
    corrected = lognorm[perturb_rows] - background[cluster_codes[perturb_rows]]
    expected_beta = corrected.mean(axis=0)
    expected_raw = corrected @ expected_beta / float(expected_beta @ expected_beta)

    assert result.metadata["background_correction"] is True
    assert result.metadata["background_control_cell_counts"] == {"c1": 2, "c2": 2}
    assert np.allclose(result.beta[0], 0.0)
    assert np.allclose(result.beta[1], expected_beta)
    assert np.allclose(result.scores[perturb_rows, 0] * 10.0, np.clip(expected_raw, 0.0, 10.0))


def test_multilabel_background_correction_matches_corrected_quadratic() -> None:
    counts = np.asarray(
        [
            [9.0, 1.0, 1.0],
            [8.0, 2.0, 1.0],
            [1.0, 8.0, 1.0],
            [2.0, 7.0, 1.0],
            [12.0, 1.0, 1.0],
            [1.0, 12.0, 1.0],
            [4.0, 10.0, 1.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(["control", "control", "control", "control", "pertA", "pertB", "pertA+pertB"], dtype=object)
    clusters = np.asarray(["c1", "c1", "c2", "c2", "c1", "c2", "c2"], dtype=object)
    adata = AnnData(
        X=sparse.csr_matrix(counts),
        obs=pd.DataFrame({"perturbation": labels, "cluster": clusters}, index=[f"cell_{index}" for index in range(counts.shape[0])]),
        var=pd.DataFrame({"highly_variable": [True, True, True]}, index=["g1", "g2", "g3"]),
    )

    result = run_ps_score_exact_fast(
        adata,
        mode="multilabel",
        perturb_column="perturbation",
        ctrl_name="control",
        target_mode="hvg",
        background_cluster_column="cluster",
        chunk_size=3,
        lr_lambda=0.0,
        score_lambda=0.0,
        scale_factor=10.0,
        target_sum=100.0,
        scale_score=False,
    )

    lognorm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 100.0)
    background = np.vstack([lognorm[(labels == "control") & (clusters == "c1")].mean(axis=0), lognorm[(labels == "control") & (clusters == "c2")].mean(axis=0)])
    active_rows = np.asarray([4, 5, 6], dtype=np.int64)
    active_clusters = np.asarray([0, 1, 1], dtype=np.int64)
    design = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    corrected = lognorm[active_rows] - background[active_clusters]
    expected_beta = np.linalg.solve(design.T @ design, design.T @ corrected)

    assert np.allclose(result.beta[0], 0.0)
    assert np.allclose(result.beta[1:], expected_beta)

    combo_row = 6
    observed = {
        result.perturbations[int(perturbation)]: float(score) * 10.0
        for cell, perturbation, score in zip(result.cell_indices, result.perturbation_indices, result.scores, strict=False)
        if int(cell) == combo_row
    }
    z = lognorm[combo_row] - background[1]
    gram = expected_beta @ expected_beta.T
    rhs = z @ expected_beta.T

    def objective(value: np.ndarray) -> float:
        return float(0.5 * value @ gram @ value - rhs @ value)

    def gradient(value: np.ndarray) -> np.ndarray:
        return gram @ value - rhs

    expected = minimize(
        objective,
        np.zeros(2, dtype=np.float64),
        jac=gradient,
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        method="L-BFGS-B",
    ).x
    assert set(observed) == {"pertA", "pertB"}
    assert np.allclose([observed["pertA"], observed["pertB"]], expected, atol=1e-6)


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
