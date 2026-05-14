from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from perturb_effects import CsrBatch, run_ps_score_anndata, run_ps_score_stream


def test_ps_score_stream_exact_matches_anndata_directionality() -> None:
    adata = _make_ps_adata()

    expected = run_ps_score_anndata(
        adata,
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
    result = run_ps_score_stream(
        _batch_factory(adata.layers["counts"], adata.obs.index.to_numpy()),
        obs=adata.obs,
        var_names=adata.var_names,
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

    expected = expected.sort_values(["target_perturbation", "row_id"]).reset_index(drop=True)
    result = result.sort_values(["target_perturbation", "row_id"]).reset_index(drop=True)
    target = result[result["perturbation_label"] == "pert-a"]["ps_score"]
    control = result[result["perturbation_label"] == "control"]["ps_score"]

    assert list(result.columns) == list(expected.columns)
    assert np.allclose(result["ps_score"], expected["ps_score"], atol=1e-6)
    assert target.median() > control.median()
    assert result.attrs["ps_score"]["stream_mode"] == "multi-pass"


def test_ps_score_stream_approx_matches_anndata_directionality() -> None:
    adata = _make_ps_adata()

    expected = run_ps_score_anndata(
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
    result = run_ps_score_stream(
        _batch_factory(adata.layers["counts"], adata.obs.index.to_numpy()),
        obs=adata.obs,
        var_names=adata.var_names,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
        perturbations=["pert-b"],
        target_gene_min=1,
        target_gene_max=2,
        scale_factor=2.0,
        scale_score=False,
    )

    expected = expected.sort_values(["target_perturbation", "row_id"]).reset_index(drop=True)
    result = result.sort_values(["target_perturbation", "row_id"]).reset_index(drop=True)
    target = result[result["perturbation_label"] == "pert-b"]["ps_score"]
    control = result[result["perturbation_label"] == "control"]["ps_score"]

    assert np.allclose(result["ps_score"], expected["ps_score"], atol=1e-6)
    assert np.isfinite(result["ps_score"]).all()
    assert ((result["ps_score"] >= 0.0) & (result["ps_score"] <= 2.0)).all()
    assert target.mean() > control.mean()
    assert result.attrs["ps_score"]["stream_mode"] == "multi-pass"


def test_ps_score_stream_one_shot_approx_requires_precomputed_signature_and_supports_one_pass() -> None:
    adata = _make_ps_adata()

    with pytest.raises(
        ValueError,
        match="explicit target_genes, target_signatures, and control_means",
    ):
        run_ps_score_stream(
            iter(_make_csr_batches(adata.layers["counts"], adata.obs.index.to_numpy())),
            obs=adata.obs,
            var_names=adata.var_names,
            perturbation_key="perturbation",
            control_label="control",
            fidelity="approx",
            perturbations=["pert-a"],
        )

    gene_names = ["g0", "g1"]
    control_mean = adata.layers["counts"][:6, :2].mean(axis=0)
    target_mean = adata.layers["counts"][6:10, :2].mean(axis=0)
    beta = target_mean - control_mean

    expected = run_ps_score_anndata(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
        perturbations=["pert-a"],
        target_genes={"pert-a": gene_names},
        target_gene_min=1,
        target_gene_max=2,
        scale_score=False,
    )
    result = run_ps_score_stream(
        iter(_make_csr_batches(adata.layers["counts"], adata.obs.index.to_numpy())),
        obs=adata.obs,
        var_names=adata.var_names,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
        perturbations=["pert-a"],
        target_genes={"pert-a": gene_names},
        target_signatures={"pert-a": beta.tolist()},
        control_means={"pert-a": control_mean.tolist()},
        target_gene_min=1,
        target_gene_max=2,
        scale_score=False,
    )

    expected = expected.sort_values(["target_perturbation", "row_id"]).reset_index(drop=True)
    result = result.sort_values(["target_perturbation", "row_id"]).reset_index(drop=True)

    assert np.allclose(result["ps_score"], expected["ps_score"], atol=1e-6)
    assert result.attrs["ps_score"]["stream_mode"] == "one-pass-precomputed"


def test_ps_score_stream_exact_requires_batch_factory() -> None:
    adata = _make_ps_adata()

    with pytest.raises(ValueError, match="callable batch factory"):
        run_ps_score_stream(
            iter(_make_csr_batches(adata.layers["counts"], adata.obs.index.to_numpy())),
            obs=adata.obs,
            var_names=adata.var_names,
            perturbation_key="perturbation",
            control_label="control",
            fidelity="exact",
        )


def test_ps_score_stream_subset_and_n_jobs_limit_output() -> None:
    adata = _make_ps_adata()

    result = run_ps_score_stream(
        _batch_factory(adata.layers["counts"], adata.obs.index.to_numpy()),
        obs=adata.obs,
        var_names=adata.var_names,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
        perturbations=["pert-b", "pert-a"],
        target_gene_min=1,
        target_gene_max=2,
        scale_score=False,
        n_jobs=2,
    )

    assert result["target_perturbation"].drop_duplicates().tolist() == ["pert-b", "pert-a"]
    pert_b_rows = result[result["target_perturbation"] == "pert-b"]
    assert set(pert_b_rows["perturbation_label"]) == {"control", "pert-b"}


def _batch_factory(matrix: np.ndarray, row_ids: np.ndarray):
    def factory() -> list[CsrBatch]:
        return _make_csr_batches(matrix, row_ids)

    return factory


def _make_csr_batches(matrix: np.ndarray, row_ids: np.ndarray) -> list[CsrBatch]:
    batches: list[CsrBatch] = []
    for start in (0, 5, 10):
        stop = min(start + 5, matrix.shape[0])
        block = sparse.csr_matrix(matrix[start:stop])
        batches.append(
            CsrBatch(
                row_ids=row_ids[start:stop].tolist(),
                indptr=block.indptr.tolist(),
                indices=block.indices.tolist(),
                data=block.data.tolist(),
                shape=block.shape,
            )
        )
    return batches


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
