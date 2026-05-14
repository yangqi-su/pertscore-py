from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from anndata import AnnData

from perturb_effects import run_mixscape_anndata


def test_mixscape_exact_separates_obvious_perturbation_from_control() -> None:
    adata = _make_mixscape_adata()

    result = run_mixscape_anndata(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="exact",
        marker_top_k=2,
        n_neighbors=3,
        iter_num=5,
    )

    pert_rows = result[result["target_perturbation"] == "pert-a"]
    target = pert_rows[pert_rows["perturbation_label"] == "pert-a"]
    control = pert_rows[pert_rows["perturbation_label"] == "control"]

    assert set(result["fidelity"]) == {"exact"}
    assert (target["class_label"] == "pert-a KO").sum() >= 5
    assert target["perturbation_score"].median() > control["perturbation_score"].median()
    assert set(control["global_class_label"]) == {"control"}


def test_mixscape_approx_returns_finite_schema() -> None:
    adata = _make_mixscape_adata()

    result = run_mixscape_anndata(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
        marker_top_k=2,
        control_sample_size=4,
        perturbation_sample_size=4,
        random_state=11,
    )

    assert {
        "row_id",
        "perturbation_label",
        "target_perturbation",
        "perturbation_score",
        "posterior_probability",
        "class_label",
        "global_class_label",
        "fidelity",
        "method",
        "reference_mode",
    }.issubset(result.columns)
    assert np.isfinite(result["perturbation_score"]).all()
    assert np.isfinite(result["posterior_probability"]).all()
    assert set(result["fidelity"]) == {"approx"}


def test_mixscape_perturbation_subset_limits_output() -> None:
    adata = _make_mixscape_adata()

    result = run_mixscape_anndata(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="exact",
        perturbations=["pert-b"],
        marker_top_k=2,
        n_neighbors=3,
    )

    assert result["target_perturbation"].unique().tolist() == ["pert-b"]
    assert set(result["perturbation_label"]) == {"control", "pert-b"}


def test_mixscape_n_jobs_is_deterministic_for_fixed_inputs() -> None:
    adata = _make_mixscape_adata()

    single = run_mixscape_anndata(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="exact",
        marker_top_k=2,
        n_neighbors=3,
        iter_num=5,
        n_jobs=1,
    )
    parallel = run_mixscape_anndata(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="exact",
        marker_top_k=2,
        n_neighbors=3,
        iter_num=5,
        n_jobs=2,
    )

    sort_columns = ["target_perturbation", "row_id"]
    pdt.assert_frame_equal(
        single.sort_values(sort_columns).reset_index(drop=True),
        parallel.sort_values(sort_columns).reset_index(drop=True),
    )


def test_mixscape_missing_control_or_too_few_cells_fails_clearly() -> None:
    adata = _make_mixscape_adata()
    adata.obs["perturbation"] = pd.Categorical(adata.obs["perturbation"])

    with pytest.raises(ValueError, match="control_label"):
        run_mixscape_anndata(
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
        run_mixscape_anndata(
            small,
            perturbation_key="perturbation",
            control_label="control",
            fidelity="exact",
        )


def _make_mixscape_adata() -> AnnData:
    matrix = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.1, 0.9, 1.0],
            [0.9, 1.1, 1.1],
            [1.0, 1.0, 0.9],
            [1.2, 0.8, 1.0],
            [0.8, 1.2, 1.0],
            [5.0, 4.8, 1.0],
            [5.2, 5.0, 1.1],
            [4.9, 5.1, 0.9],
            [5.1, 4.9, 1.0],
            [5.3, 5.2, 1.0],
            [4.8, 4.7, 1.2],
            [1.0, 1.0, 4.5],
            [1.2, 1.0, 4.8],
            [0.9, 1.1, 4.7],
            [1.1, 0.9, 4.6],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "perturbation": ["control"] * 6 + ["pert-a"] * 6 + ["pert-b"] * 4,
        },
        index=[f"cell-{index}" for index in range(matrix.shape[0])],
    )
    var = pd.DataFrame(index=["g0", "g1", "g2"])
    adata = AnnData(matrix, obs=obs, var=var)
    adata.layers["counts"] = matrix.copy()
    return adata
