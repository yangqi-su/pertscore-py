from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import perturb_effects.stats as stats_module
from perturb_effects.ps_score_exact import run_ps_score_exact_anndata
from perturb_effects.stats import solve_ridge_beta
from perturb_effects.utils import parse_perturbation_labels


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


def _make_target_strategy_adata(*, include_counts: bool = True) -> AnnData:
    expression = np.array(
        [
            [1.0, 1.0, 0.0, 2.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 3.0, 0.0],
            [1.0, 1.0, 3.0, 0.0],
            [0.0, 2.0, 1.0, 4.0],
            [0.0, 2.0, 1.0, 4.0],
        ],
        dtype=float,
    )
    adata = AnnData(X=np.zeros_like(expression))
    adata.layers["expr"] = expression
    if include_counts:
        adata.layers["counts"] = np.array(
            [
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 2.0, 0.0],
                [1.0, 0.0, 2.0, 0.0],
                [0.0, 2.0, 1.0, 4.0],
                [0.0, 2.0, 1.0, 4.0],
            ],
            dtype=float,
        )
    adata.obs["perturbation"] = ["control", "control", "pertA", "pertA", "pertB", "pertB"]
    adata.obs_names = ["ctrl-1", "ctrl-2", "pert-a-1", "pert-a-2", "pert-b-1", "pert-b-2"]
    adata.var_names = ["g1", "g2", "g3", "g4"]
    adata.var["highly_variable"] = [True, False, True, False]
    adata.var["custom_hvg"] = [True, True, False, False]
    return adata


def _make_sparse_adata(adata: AnnData) -> AnnData:
    sparse_adata = adata.copy()
    sparse_adata.X = sparse.csr_matrix(np.asarray(adata.X, dtype=float))
    for layer_name in list(adata.layers.keys()):
        sparse_adata.layers[layer_name] = sparse.csr_matrix(np.asarray(adata.layers[layer_name], dtype=float))
    return sparse_adata


def test_parse_perturbations_uses_fixed_plus_delimiter_and_design_has_negctrl_column() -> None:
    labels = np.array(["control", "pertA", "pertA+pertB", "pertB+pertA"], dtype=object)

    parsed = parse_perturbation_labels(labels, mode="multilabel", ctrl_name="control", perturbations=None)
    design = sparse.hstack(
        [sparse.csr_matrix(np.ones((parsed.guides.shape[0], 1))), parsed.guides],
        format="csr",
    )

    assert parsed.perturbations == ["pertA", "pertB"]
    assert np.array_equal(
        design.toarray(),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ),
    )


def test_run_ps_score_exact_uses_selected_layer_union_genes_clipping_and_csv_like_output() -> None:
    adata = _make_layer_selection_adata()

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes={"pertA": ["g2", "g1"], "pertB": ["g3", "g2"]},
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=5,
        apply_gene_filter=False,
        apply_quantile_clip=True,
        clip_quantile=0.5,
        lr_lambda=0.1,
        scale_score=False,
    )

    metadata = result.attrs["ps_score_exact"]

    assert list(result.columns) == ["obs_index", "ps_score", "perturbation"]
    assert metadata["layer"] == "expr"
    assert metadata["computation_path"] == "in_memory_sparse_lbfgsb"
    assert metadata["normalization"] == "normalize_total_log1p"
    assert metadata["target_sum"] == 1e4
    assert metadata["union_target_genes"] == ["g2", "g1", "g3"]
    assert metadata["y_shape"] == (4, 3)
    assert np.allclose(metadata["clip_values"], np.array([6.804504648063392, 4.511849017779065, 9.106091350491896]))


def test_solve_ridge_beta_matches_direct_formula() -> None:
    x_matrix = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
    )
    y_matrix = sparse.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))
    lr_lambda = 0.5

    beta = solve_ridge_beta(x_matrix, y_matrix, lr_lambda)
    dense_x = x_matrix.toarray()
    dense_y = y_matrix.toarray()
    expected = np.linalg.solve(dense_x.T @ dense_x + lr_lambda * np.eye(dense_x.shape[1]), dense_x.T @ dense_y)

    assert np.allclose(beta, expected)


def test_solve_ridge_beta_falls_back_to_direct_solve_when_zero_lambda_cholesky_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_matrix = sparse.eye(2, format="csr")
    y_matrix = sparse.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    solve_called = False
    original_solve = stats_module.linalg.solve

    def fake_cho_factor(*args: object, **kwargs: object):
        raise stats_module.linalg.LinAlgError("forced cholesky failure")

    def wrapped_solve(*args: object, **kwargs: object):
        nonlocal solve_called
        solve_called = True
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(stats_module.linalg, "cho_factor", fake_cho_factor)
    monkeypatch.setattr(stats_module.linalg, "solve", wrapped_solve)

    beta = solve_ridge_beta(x_matrix, y_matrix, lr_lambda=0.0)

    assert solve_called
    assert np.allclose(beta, y_matrix.toarray())


def test_exact_scores_match_closed_form_with_control_zero_and_scale_factor() -> None:
    adata = _make_single_perturbation_adata()

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes=["g1", "g2"],
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=5,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        lr_lambda=0.0,
        score_lambda=0.0,
        scale_factor=1.5,
        scale_score=False,
    )

    scores = result.set_index("obs_index")["ps_score"]
    assert np.allclose(scores.loc[["ctrl-1", "ctrl-2"]].to_numpy(), np.zeros(2))
    assert np.allclose(scores.to_numpy(), np.array([0.0, 0.0, 0.35352564, 0.97980769]), atol=1e-6)
    assert np.all((scores.to_numpy() >= 0.0) & (scores.to_numpy() <= 1.0))


def test_scale_score_normalizes_by_column_max_after_scale_factor_division() -> None:
    adata = _make_single_perturbation_adata()

    unscaled = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes=["g1", "g2"],
        target_gene_source="provided",
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
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=5,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        lr_lambda=0.0,
        scale_factor=3.0,
        scale_score=True,
    )

    unscaled_scores = unscaled.set_index("obs_index")["ps_score"]
    scaled_scores = scaled.set_index("obs_index")["ps_score"]
    expected_scaled = unscaled_scores / unscaled_scores.max()

    assert np.allclose(scaled_scores.to_numpy(), expected_scaled.to_numpy(), atol=1e-6)
    assert np.isclose(scaled_scores.loc["pert-a-1"], 0.3608112543358848, atol=1e-6)
    assert np.isclose(scaled_scores.loc["pert-a-2"], 1.0, atol=1e-6)


def test_provided_target_gene_mapping_deduplicates_and_truncates_by_max() -> None:
    adata = _make_target_strategy_adata()

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes={"pertA": ["g1", "g1", "g2", "g3"], "pertB": ["g4", "g3", "g4"]},
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=2,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        scale_score=False,
    )

    metadata = result.attrs["ps_score_exact"]

    assert metadata["genes_by_perturbation"] == {"pertA": ["g1", "g2"], "pertB": ["g4", "g3"]}
    assert metadata["union_target_genes"] == ["g1", "g2", "g4", "g3"]


@pytest.mark.parametrize(
    ("target_genes", "target_gene_min", "message"),
    [
        ({"pertA": ["g1", "missing"]}, 1, "Unknown target genes requested"),
        ({"pertA": ["g1", "g1"]}, 2, "Need at least 2 target genes"),
    ],
)
def test_provided_target_genes_raise_for_missing_or_too_few_genes(
    target_genes: dict[str, list[str]],
    target_gene_min: int,
    message: str,
) -> None:
    adata = _make_target_strategy_adata()

    with pytest.raises(ValueError, match=message):
        run_ps_score_exact_anndata(
            adata,
            perturb_column="perturbation",
            ctrl_name="control",
            layer="expr",
            perturbations=["pertA"],
            target_genes=target_genes,
            target_gene_source="provided",
            target_gene_min=target_gene_min,
            target_gene_max=4,
            apply_gene_filter=False,
            apply_quantile_clip=False,
            scale_score=False,
        )


def test_hvg_target_gene_mode_reuses_requested_hvg_key_for_each_perturbation() -> None:
    adata = _make_target_strategy_adata()

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_gene_source="hvg",
        hvg_key="custom_hvg",
        target_gene_min=1,
        target_gene_max=3,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        scale_score=False,
    )

    metadata = result.attrs["ps_score_exact"]

    assert metadata["target_gene_source_detail"] == {"mode": "hvg", "hvg_key": "custom_hvg"}
    assert metadata["genes_by_perturbation"] == {"pertA": ["g1", "g2"], "pertB": ["g1", "g2"]}


def test_hvg_target_gene_mode_errors_when_no_hvgs_exist() -> None:
    adata = _make_target_strategy_adata()
    adata.var["empty_hvg"] = [False, False, False, False]

    with pytest.raises(ValueError, match="does not contain any HVGs"):
        run_ps_score_exact_anndata(
            adata,
            perturb_column="perturbation",
            ctrl_name="control",
            layer="expr",
            target_gene_source="hvg",
            hvg_key="empty_hvg",
            target_gene_min=1,
            target_gene_max=3,
            apply_gene_filter=False,
            apply_quantile_clip=False,
        )


def test_scanpy_target_gene_mode_selects_non_empty_genes_when_available() -> None:
    pytest.importorskip("scanpy")
    adata = _make_target_strategy_adata(include_counts=False)

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        perturbations=["pertA"],
        target_gene_source="scanpy_de",
        target_gene_min=1,
        target_gene_max=2,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        scale_score=False,
    )

    metadata = result.attrs["ps_score_exact"]

    assert metadata["target_gene_source"] == "scanpy_de"
    assert metadata["target_gene_source_detail"] == {
        "mode": "scanpy_de",
        "layer": "expr",
        "method": "wilcoxon",
        "logfc_threshold": 0.1,
        "logfc_threshold_decay": 0.8,
        "max_logfc_rounds": 3,
        "direction": "both",
        "rank_by": "pvals",
    }
    assert metadata["genes_by_perturbation"]["pertA"]
    assert set(metadata["genes_by_perturbation"]["pertA"]).issubset(set(adata.var_names))


def test_gene_filter_uses_counts_layer_when_available() -> None:
    adata = _make_target_strategy_adata(include_counts=True)

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        perturbations=["pertA"],
        target_genes=["g1", "g2"],
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=4,
        apply_gene_filter=True,
        gene_filter_min_fraction=0.75,
        apply_quantile_clip=False,
        scale_score=False,
    )

    metadata = result.attrs["ps_score_exact"]

    assert metadata["gene_filter_source"] == "counts"
    assert metadata["genes_by_perturbation"]["pertA"] == ["g1"]
    assert metadata["gene_filter_metadata"]["target_gene_counts_before_filter"] == {"pertA": 2}
    assert metadata["gene_filter_metadata"]["target_gene_counts_after_filter"] == {"pertA": 1}


def test_dense_and_sparse_inputs_use_same_sparse_reference_path() -> None:
    dense_adata = _make_target_strategy_adata(include_counts=True)
    sparse_adata = _make_sparse_adata(dense_adata)
    kwargs = dict(
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes={"pertA": ["g1", "g2", "g3"], "pertB": ["g4", "g3", "g2"]},
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=3,
        apply_gene_filter=True,
        gene_filter_min_fraction=0.5,
        apply_quantile_clip=False,
        lr_lambda=0.1,
        score_lambda=0.05,
        scale_factor=2.0,
        scale_score=True,
    )

    dense_result = run_ps_score_exact_anndata(dense_adata, **kwargs)
    sparse_result = run_ps_score_exact_anndata(sparse_adata, **kwargs)

    pd.testing.assert_frame_equal(sparse_result, dense_result)
    assert dense_result.attrs["ps_score_exact"]["computation_path"] == "in_memory_sparse_lbfgsb"
    assert sparse_result.attrs["ps_score_exact"]["expression_matrix_format"] == "sparse"


def test_sparse_quantile_clip_stays_on_sparse_reference_path() -> None:
    sparse_adata = _make_sparse_adata(_make_layer_selection_adata())

    result = run_ps_score_exact_anndata(
        sparse_adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes={"pertA": ["g2", "g1"], "pertB": ["g3", "g2"]},
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=5,
        apply_gene_filter=False,
        apply_quantile_clip=True,
        clip_quantile=0.5,
        lr_lambda=0.1,
        scale_score=False,
    )

    metadata = result.attrs["ps_score_exact"]
    assert metadata["computation_path"] == "in_memory_sparse_lbfgsb"
    assert metadata["sparse_fallback_reason"] is None
    assert metadata["normalization"] == "normalize_total_log1p"
    assert np.allclose(metadata["clip_values"], [6.804504648063392, 4.511849017779065, 9.106091350491896])


def test_multilabel_output_has_one_row_per_active_selected_perturbation() -> None:
    expression = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [4.0, 2.0, 3.0],
            [5.0, 2.0, 4.0],
        ],
        dtype=float,
    )
    adata = AnnData(X=sparse.csr_matrix(expression))
    adata.obs["perturbation"] = ["control", "control", "pertA+pertB", "pertA+pertB"]
    adata.obs_names = ["ctrl-1", "ctrl-2", "combo-1", "combo-2"]
    adata.var_names = ["g1", "g2", "g3"]

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        target_genes={"pertA": ["g1", "g2"], "pertB": ["g2", "g3"]},
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=2,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        lr_lambda=0.1,
        scale_score=False,
    )

    combo = result[result["obs_index"] == "combo-1"]
    assert combo["perturbation"].tolist() == ["pertA", "pertB"]
    assert result.attrs["ps_score_exact"]["perturbation_delimiter"] == "+"


def test_background_correction_uses_cluster_control_baseline_during_scoring() -> None:
    expression = np.array([[1.0], [5.0], [4.0], [8.0]], dtype=float)
    adata = AnnData(X=sparse.csr_matrix(expression))
    adata.obs["perturbation"] = ["control", "control", "pertA", "pertA"]
    adata.obs["cluster"] = ["c1", "c2", "c1", "c2"]
    adata.obs_names = ["ctrl-c1", "ctrl-c2", "pert-c1", "pert-c2"]
    adata.var_names = ["g1"]

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        target_genes=["g1"],
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=1,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        lr_lambda=0.0,
        scale_factor=3.0,
        scale_score=False,
        background_cluster_column="cluster",
    )

    scores = result.set_index("obs_index")["ps_score"]
    assert np.allclose(scores.loc[["pert-c1", "pert-c2"]].to_numpy(), [1.0 / 3.0, 1.0 / 3.0], atol=1e-6)
    assert result.attrs["ps_score_exact"]["background_correction"] is True
    assert result.attrs["ps_score_exact"]["background_control_cell_counts"] == {"c1": 1, "c2": 1}


def test_exact_ps_records_stage_timings() -> None:
    adata = _make_target_strategy_adata()

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        perturbations=["pertA"],
        target_genes={"pertA": ["g1", "g4"]},
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=2,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        scale_score=False,
    )

    stage_timings = result.attrs["ps_score_exact"]["stage_timings"]

    assert set(stage_timings) == {"target_gene_selection", "beta_solve", "scoring"}
    assert all(stage_timings[name] >= 0.0 for name in stage_timings)
