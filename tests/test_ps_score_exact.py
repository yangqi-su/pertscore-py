from __future__ import annotations

import numpy as np
from anndata import AnnData

from perturb_effects.ps_score_exact import (
    _build_design_matrix,
    _solve_ridge_beta,
    run_ps_score_exact_anndata,
)


def _make_layer_selection_adata() -> AnnData:
    expression = np.array(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ],
        dtype=float,
    )
    adata = AnnData(X=np.zeros_like(expression))
    adata.layers["expr"] = expression
    adata.obs["perturbation"] = ["control", "pertA", "pertB", "control"]
    adata.obs_names = ["ctrl-1", "pert-a-1", "pert-b-1", "ctrl-2"]
    adata.var_names = ["g1", "g2", "g3"]
    return adata


def _make_single_perturbation_adata() -> AnnData:
    expression = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [6.0, 1.0],
        ],
        dtype=float,
    )
    adata = AnnData(X=np.zeros_like(expression))
    adata.layers["expr"] = expression
    adata.obs["perturbation"] = ["control", "control", "pertA", "pertA"]
    adata.obs_names = ["ctrl-1", "ctrl-2", "pert-a-1", "pert-a-2"]
    adata.var_names = ["g1", "g2"]
    return adata


def test_build_design_matrix_has_negctrl_and_single_active_columns() -> None:
    labels = np.array(["control", "pertA", "pertB", "control", "pertA"], dtype=object)

    matrix = _build_design_matrix(labels, ["pertA", "pertB"])

    assert np.array_equal(
        matrix,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
    )
    assert np.all(matrix[:, 0] == 1.0)
    assert np.all(matrix[labels == "control", 1:] == 0.0)


def test_run_ps_score_exact_uses_selected_layer_union_genes_and_clipping() -> None:
    adata = _make_layer_selection_adata()

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes={"pertA": ["g2", "g1"], "pertB": ["g3", "g2"]},
        target_gene_min=1,
        target_gene_max=5,
        apply_gene_filter=False,
        apply_quantile_clip=True,
        clip_quantile=0.5,
        lr_lambda=0.0,
        scale_score=False,
    )

    metadata = result.attrs["ps_score_exact"]

    assert metadata["layer"] == "expr"
    assert metadata["union_target_genes"] == ["g2", "g1", "g3"]
    assert metadata["y_shape"] == (4, 3)
    assert np.allclose(metadata["clip_values"], np.array([25.0, 2.5, 250.0]))


def test_solve_ridge_beta_matches_direct_formula() -> None:
    x_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    y_matrix = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    lr_lambda = 0.5

    beta = _solve_ridge_beta(x_matrix, y_matrix, lr_lambda)
    expected = np.linalg.solve(
        x_matrix.T @ x_matrix + lr_lambda * np.eye(x_matrix.shape[1]),
        x_matrix.T @ y_matrix,
    )

    assert np.allclose(beta, expected)


def test_exact_scores_match_closed_form_with_control_zero_and_scale_factor() -> None:
    adata = _make_single_perturbation_adata()

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes=["g1", "g2"],
        target_gene_min=1,
        target_gene_max=5,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        lr_lambda=0.0,
        score_lambda=0.0,
        scale_factor=1.5,
        scale_score=False,
    )

    scores = result.set_index("row_id")["ps_score"]
    labels = np.array(["control", "control", "pertA", "pertA"], dtype=object)
    x_matrix = _build_design_matrix(labels, ["pertA"])
    y_matrix = adata.layers["expr"]
    beta = np.linalg.solve(x_matrix.T @ x_matrix, x_matrix.T @ y_matrix)
    baseline = beta[0]
    perturbation_beta = beta[1]
    raw = ((y_matrix[labels == "pertA"] - baseline) @ perturbation_beta) / np.dot(
        perturbation_beta,
        perturbation_beta,
    )
    expected_perturbed = np.clip(raw, 0.0, 1.5) / 1.5

    assert np.allclose(scores.loc[["ctrl-1", "ctrl-2"]].to_numpy(), np.zeros(2))
    assert np.allclose(scores.loc[["pert-a-1", "pert-a-2"]].to_numpy(), expected_perturbed)
    assert np.allclose(scores.to_numpy(), np.array([0.0, 0.0, 2.0 / 9.0, 1.0]))
    assert np.all((scores.to_numpy() >= 0.0) & (scores.to_numpy() <= 1.0))
    assert set(result["score_status"]) == {"control-zero", "optimized-active"}


def test_scale_score_normalizes_by_column_max_after_scale_factor_division() -> None:
    adata = _make_single_perturbation_adata()

    unscaled = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes=["g1", "g2"],
        target_gene_min=1,
        target_gene_max=5,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        lr_lambda=0.0,
        scale_factor=3.0,
        scale_score=False,
    )
    scaled = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes=["g1", "g2"],
        target_gene_min=1,
        target_gene_max=5,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        lr_lambda=0.0,
        scale_factor=3.0,
        scale_score=True,
    )

    unscaled_scores = unscaled.set_index("row_id")["ps_score"]
    scaled_scores = scaled.set_index("row_id")["ps_score"]
    expected_scaled = unscaled_scores / unscaled_scores.max()

    assert np.allclose(scaled_scores.to_numpy(), expected_scaled.to_numpy())
    assert np.isclose(scaled_scores.loc["pert-a-1"], 0.2)
    assert np.isclose(scaled_scores.loc["pert-a-2"], 1.0)
