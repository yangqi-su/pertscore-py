from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import perturb_effects.ps_score_exact as ps_score_exact_module
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
        sparse_adata.layers[layer_name] = sparse.csr_matrix(
            np.asarray(adata.layers[layer_name], dtype=float)
        )
    return sparse_adata


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
        target_gene_source="provided",
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


def test_solve_ridge_beta_falls_back_to_direct_solve_when_zero_lambda_cholesky_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_matrix = np.eye(2)
    y_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    solve_called = False
    original_solve = ps_score_exact_module.linalg.solve

    def fake_cho_factor(*args: object, **kwargs: object):
        raise ps_score_exact_module.linalg.LinAlgError("forced cholesky failure")

    def wrapped_solve(*args: object, **kwargs: object):
        nonlocal solve_called
        solve_called = True
        return original_solve(*args, **kwargs)

    monkeypatch.setattr(ps_score_exact_module.linalg, "cho_factor", fake_cho_factor)
    monkeypatch.setattr(ps_score_exact_module.linalg, "solve", wrapped_solve)

    beta = _solve_ridge_beta(x_matrix, y_matrix, lr_lambda=0.0)

    assert solve_called
    assert np.allclose(beta, y_matrix)


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

    unscaled_scores = unscaled.set_index("row_id")["ps_score"]
    scaled_scores = scaled.set_index("row_id")["ps_score"]
    expected_scaled = unscaled_scores / unscaled_scores.max()

    assert np.allclose(scaled_scores.to_numpy(), expected_scaled.to_numpy())
    assert np.isclose(scaled_scores.loc["pert-a-1"], 0.2)
    assert np.isclose(scaled_scores.loc["pert-a-2"], 1.0)


def test_lsq_linear_score_solver_matches_closed_form_scores() -> None:
    adata = _make_single_perturbation_adata()

    closed_form = run_ps_score_exact_anndata(
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
        score_lambda=0.1,
        scale_factor=1.5,
        scale_score=False,
        score_solver="closed_form",
    )
    lsq = run_ps_score_exact_anndata(
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
        score_lambda=0.1,
        scale_factor=1.5,
        scale_score=False,
        score_solver="lsq_linear",
    )

    assert np.allclose(closed_form["ps_score"], lsq["ps_score"])
    assert lsq.attrs["ps_score_exact"]["score_solver"] == "lsq_linear"
    assert lsq.attrs["ps_score_exact"]["score_metadata"]["pertA"]["score_solver"] == "lsq_linear"


def test_provided_target_gene_mapping_deduplicates_and_truncates_by_max() -> None:
    adata = _make_target_strategy_adata()

    result = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes={
            "pertA": ["g1", "g1", "g2", "g3"],
            "pertB": ["g4", "g3", "g4"],
        },
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=2,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        scale_score=False,
    )

    metadata = result.attrs["ps_score_exact"]

    assert metadata["genes_by_perturbation"] == {
        "pertA": ["g1", "g2"],
        "pertB": ["g4", "g3"],
    }
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
    assert metadata["genes_by_perturbation"] == {
        "pertA": ["g1", "g2"],
        "pertB": ["g1", "g2"],
    }


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
            scale_score=False,
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
    assert metadata["target_gene_source_detail"] == {"mode": "scanpy_de", "layer": "expr"}
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


def test_gene_filter_falls_back_to_selected_expression_layer_without_counts() -> None:
    adata = _make_target_strategy_adata(include_counts=False)

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

    assert metadata["gene_filter_source"] == "expr"
    assert metadata["genes_by_perturbation"]["pertA"] == ["g1", "g2"]
    assert metadata["gene_filter_metadata"]["target_gene_counts_before_filter"] == {"pertA": 2}
    assert metadata["gene_filter_metadata"]["target_gene_counts_after_filter"] == {"pertA": 2}


def test_sparse_closed_form_matches_dense_outputs_and_selects_sparse_path() -> None:
    dense_adata = _make_target_strategy_adata(include_counts=True)
    sparse_adata = _make_sparse_adata(dense_adata)

    kwargs = dict(
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes={
            "pertA": ["g1", "g2", "g3"],
            "pertB": ["g4", "g3", "g2"],
        },
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

    dense_metadata = dense_result.attrs["ps_score_exact"]
    sparse_metadata = sparse_result.attrs["ps_score_exact"]
    assert sparse_metadata["computation_path"] == "sparse_closed_form"
    assert sparse_metadata["expression_matrix_format"] == "sparse"
    assert sparse_metadata["sparse_fallback_reason"] is None
    assert sparse_metadata["genes_by_perturbation"] == dense_metadata["genes_by_perturbation"]
    assert sparse_metadata["score_metadata"].keys() == dense_metadata["score_metadata"].keys()
    for perturbation in sparse_metadata["score_metadata"]:
        assert sparse_metadata["score_metadata"][perturbation]["score_solver"] == "closed_form"
        assert sparse_metadata["score_metadata"][perturbation]["control_count"] == dense_metadata[
            "score_metadata"
        ][perturbation]["control_count"]
        assert sparse_metadata["score_metadata"][perturbation]["target_count"] == dense_metadata[
            "score_metadata"
        ][perturbation]["target_count"]
        assert sparse_metadata["score_metadata"][perturbation]["column_scaled"] == dense_metadata[
            "score_metadata"
        ][perturbation]["column_scaled"]
        assert sparse_metadata["score_metadata"][perturbation]["beta_norm_sq"] == pytest.approx(
            dense_metadata["score_metadata"][perturbation]["beta_norm_sq"]
        )
        assert sparse_metadata["score_metadata"][perturbation][
            "max_score_before_column_scale"
        ] == pytest.approx(dense_metadata["score_metadata"][perturbation]["max_score_before_column_scale"])


def test_sparse_quantile_clip_falls_back_to_dense_path() -> None:
    dense_adata = _make_layer_selection_adata()
    sparse_adata = _make_sparse_adata(dense_adata)

    kwargs = dict(
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
        lr_lambda=0.0,
        scale_score=False,
    )

    dense_result = run_ps_score_exact_anndata(dense_adata, **kwargs)
    sparse_result = run_ps_score_exact_anndata(sparse_adata, **kwargs)

    pd.testing.assert_frame_equal(sparse_result, dense_result)
    sparse_metadata = sparse_result.attrs["ps_score_exact"]
    assert sparse_metadata["computation_path"] == "dense_fallback"
    assert sparse_metadata["sparse_fallback_reason"] == "apply_quantile_clip=True"


def test_quantile_clipping_is_optional_at_the_requested_095_quantile() -> None:
    adata = _make_layer_selection_adata()

    clipped = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes={"pertA": ["g3"], "pertB": ["g3"]},
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=2,
        apply_gene_filter=False,
        apply_quantile_clip=True,
        clip_quantile=0.95,
        scale_score=False,
    )
    unclipped = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="expr",
        target_genes={"pertA": ["g3"], "pertB": ["g3"]},
        target_gene_source="provided",
        target_gene_min=1,
        target_gene_max=2,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        scale_score=False,
    )

    assert np.allclose(clipped.attrs["ps_score_exact"]["clip_values"], [385.0])
    assert clipped.attrs["ps_score_exact"]["clip_quantile"] == 0.95
    assert unclipped.attrs["ps_score_exact"]["clip_values"] is None


def test_exact_ps_records_stage_timings_and_observer_events() -> None:
    adata = _make_target_strategy_adata()
    events: list[tuple[str, str, dict[str, object]]] = []

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
        stage_observer=lambda stage, event, details: events.append((stage, event, dict(details))),
    )

    stage_timings = result.attrs["ps_score_exact"]["stage_timings"]

    assert set(stage_timings) == {"target_gene_selection", "beta_solve", "scoring"}
    assert all(stage_timings[name] >= 0.0 for name in stage_timings)
    assert [stage for stage, event, _ in events if event == "start"] == [
        "target_gene_selection",
        "beta_solve",
        "scoring",
    ]
    assert [stage for stage, event, _ in events if event == "end"] == [
        "target_gene_selection",
        "beta_solve",
        "scoring",
    ]
