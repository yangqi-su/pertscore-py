from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from perturb_effects import run_ps_score_anndata


def test_ps_score_exact_returns_bounded_scaled_scores() -> None:
    adata = _make_ps_adata()

    result = run_ps_score_anndata(
        adata,
        layer="counts",
        perturbation_key="perturbation",
        control_label="control",
        fidelity="exact",
        perturbations=["pert-a"],
        target_gene_min=1,
        target_gene_max=2,
        scale_factor=3.0,
        lambda_=0.05,
        scale_score=True,
    )

    pert_rows = result[result["target_perturbation"] == "pert-a"]
    target = pert_rows[pert_rows["perturbation_label"] == "pert-a"]
    control = pert_rows[pert_rows["perturbation_label"] == "control"]

    assert set(result["fidelity"]) == {"exact"}
    assert set(pert_rows["score_status"]) == {"fixed-control", "optimized"}
    assert np.isfinite(pert_rows["ps_score"]).all()
    assert ((pert_rows["ps_score"] >= 0.0) & (pert_rows["ps_score"] <= 1.0)).all()
    assert target["ps_score"].max() == pytest.approx(1.0)
    assert target["ps_score"].median() > control["ps_score"].median()


def test_ps_score_approx_matches_directionality_and_subset_behavior() -> None:
    adata = _make_ps_adata()

    exact = run_ps_score_anndata(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="exact",
        perturbations=["pert-b"],
        target_gene_min=1,
        target_gene_max=2,
    )
    approx = run_ps_score_anndata(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
        perturbations=["pert-b"],
        target_gene_min=1,
        target_gene_max=2,
        scale_factor=2.0,
        scale_score=False,
    )

    exact_target = exact[exact["perturbation_label"] == "pert-b"]["ps_score"].to_numpy()
    approx_target = approx[approx["perturbation_label"] == "pert-b"]["ps_score"].to_numpy()
    approx_control = approx[approx["perturbation_label"] == "control"]["ps_score"]

    assert approx["target_perturbation"].unique().tolist() == ["pert-b"]
    assert {
        "row_id",
        "perturbation_label",
        "target_perturbation",
        "ps_score",
        "fidelity",
        "method",
        "selected_target_gene_count",
        "score_status",
    }.issubset(approx.columns)
    assert set(approx["score_status"]) == {"projected"}
    assert np.isfinite(approx["ps_score"]).all()
    assert ((approx["ps_score"] >= 0.0) & (approx["ps_score"] <= 2.0)).all()
    assert approx_target.mean() > approx_control.mean()
    assert np.corrcoef(exact_target, approx_target)[0, 1] > 0.95


def test_ps_score_explicit_target_genes_are_supported() -> None:
    adata = _make_ps_adata()

    result = run_ps_score_anndata(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
        perturbations=["pert-a"],
        target_genes={"pert-a": ["g0", "g1"]},
        target_gene_min=1,
        target_gene_max=2,
    )

    assert set(result["target_perturbation"]) == {"pert-a"}
    assert set(result["selected_target_gene_count"]) == {2}


def test_ps_score_missing_control_or_too_few_cells_fails_clearly() -> None:
    adata = _make_ps_adata()

    with pytest.raises(ValueError, match="control_label"):
        run_ps_score_anndata(
            adata,
            perturbation_key="perturbation",
            control_label="missing",
            fidelity="exact",
        )

    small = AnnData(
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [5.0, 5.0, 1.0],
            ],
            dtype=float,
        ),
        obs=pd.DataFrame(
            {"perturbation": ["control", "control", "pert-a"]},
            index=["c0", "c1", "p0"],
        ),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )

    with pytest.raises(ValueError, match="at least 2 cells"):
        run_ps_score_anndata(
            small,
            perturbation_key="perturbation",
            control_label="control",
            fidelity="exact",
        )


def _make_ps_adata() -> AnnData:
    matrix = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.1, 0.9, 1.1, 0.9],
            [0.9, 1.0, 0.8, 1.1],
            [1.0, 1.2, 1.0, 1.0],
            [0.8, 1.1, 1.2, 0.9],
            [1.2, 0.8, 1.0, 1.1],
            [5.0, 4.7, 1.0, 1.0],
            [5.2, 4.9, 1.1, 0.9],
            [4.8, 5.0, 0.9, 1.1],
            [5.1, 4.8, 1.0, 1.0],
            [1.0, 1.0, 4.9, 4.6],
            [1.1, 0.9, 5.0, 4.8],
            [0.9, 1.1, 4.8, 4.7],
            [1.0, 1.0, 5.1, 4.9],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {"perturbation": ["control"] * 6 + ["pert-a"] * 4 + ["pert-b"] * 4},
        index=[f"cell-{index}" for index in range(matrix.shape[0])],
    )
    var = pd.DataFrame(index=["g0", "g1", "g2", "g3"])
    adata = AnnData(matrix, obs=obs, var=var)
    adata.layers["counts"] = matrix.copy()
    return adata
