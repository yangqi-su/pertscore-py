from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from perturb_effects import (
    CsrBatch,
    run_mixscape_anndata,
    run_mixscape_stream,
    run_ps_score_anndata,
    run_ps_score_exact_anndata,
    run_ps_score_stream,
)


def test_phase7_smoke_mixscape_all_modes_share_expected_schema() -> None:
    adata = _make_mixscape_adata()

    anndata_exact = run_mixscape_anndata(
        adata,
        layer="counts",
        de_layer="counts",
        perturbation_key="perturbation",
        control_label="control",
        fidelity="exact",
        marker_top_k=2,
        n_neighbors=3,
        iter_num=5,
    )
    anndata_approx = run_mixscape_anndata(
        adata,
        layer="counts",
        de_layer="counts",
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
        perturbations=["pert-a"],
        marker_top_k=2,
        control_sample_size=4,
        perturbation_sample_size=4,
        random_state=11,
    )
    stream_exact = run_mixscape_stream(
        _batch_factory(adata.layers["counts"], adata.obs.index.to_numpy()),
        obs=adata.obs,
        var_names=adata.var_names,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="exact",
        marker_top_k=2,
        n_neighbors=3,
        iter_num=5,
    )
    stream_approx = run_mixscape_stream(
        _batch_factory(adata.layers["counts"], adata.obs.index.to_numpy()),
        obs=adata.obs,
        var_names=adata.var_names,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
        perturbations=["pert-a"],
        marker_top_k=2,
        control_sample_size=4,
        perturbation_sample_size=4,
        random_state=11,
    )

    assert not anndata_exact.empty
    assert not anndata_approx.empty
    assert not stream_exact.empty
    assert not stream_approx.empty
    assert list(stream_exact.columns) == list(anndata_exact.columns)
    assert list(stream_approx.columns) == list(anndata_approx.columns)
    assert anndata_exact.attrs["mixscape"]["layer"] == "counts"
    assert anndata_exact.attrs["mixscape"]["de_layer"] == "counts"
    assert stream_exact.attrs["mixscape"]["stream_mode"] == "multi-pass"
    assert stream_approx.attrs["mixscape"]["stream_mode"] == "multi-pass"
    target_exact = stream_exact[stream_exact["perturbation_label"] == "pert-a"]["perturbation_score"]
    control_exact = stream_exact[stream_exact["perturbation_label"] == "control"]["perturbation_score"]
    assert target_exact.median() > control_exact.median()


def test_phase7_smoke_ps_score_all_modes_share_expected_schema() -> None:
    adata = _make_ps_adata()

    anndata_exact = run_ps_score_anndata(
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
    anndata_approx = run_ps_score_anndata(
        adata,
        layer="counts",
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
        perturbations=["pert-b"],
        target_gene_min=1,
        target_gene_max=2,
        scale_factor=2.0,
        scale_score=False,
    )
    stream_exact = run_ps_score_stream(
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
    stream_approx = run_ps_score_stream(
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

    assert not anndata_exact.empty
    assert not anndata_approx.empty
    assert not stream_exact.empty
    assert not stream_approx.empty
    assert list(stream_exact.columns) == list(anndata_exact.columns)
    assert list(stream_approx.columns) == list(anndata_approx.columns)
    assert anndata_exact.attrs["ps_score"]["layer"] == "counts"
    assert stream_exact.attrs["ps_score"]["stream_mode"] == "multi-pass"
    assert stream_approx.attrs["ps_score"]["stream_mode"] == "multi-pass"
    target_exact = stream_exact[stream_exact["perturbation_label"] == "pert-a"]["ps_score"]
    control_exact = stream_exact[stream_exact["perturbation_label"] == "control"]["ps_score"]
    assert target_exact.median() > control_exact.median()


def test_phase7_smoke_public_exact_ps_api_supports_provided_and_hvg_targets() -> None:
    adata = _make_ps_adata()
    adata.var["exact_hvg"] = [False, False, True, True]

    provided = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="counts",
        perturbations=["pert-a"],
        target_genes={"pert-a": ["g0", "g1"]},
        target_gene_min=1,
        target_gene_max=2,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        scale_score=False,
    )
    hvg = run_ps_score_exact_anndata(
        adata,
        perturb_column="perturbation",
        ctrl_name="control",
        layer="counts",
        perturbations=["pert-b"],
        target_gene_source="hvg",
        hvg_key="exact_hvg",
        target_gene_min=1,
        target_gene_max=2,
        apply_gene_filter=False,
        apply_quantile_clip=False,
        scale_score=False,
    )

    assert not provided.empty
    assert not hvg.empty
    assert set(provided["method"]) == {"ps_score_exact"}
    assert set(provided["target_perturbation"]) == {"pert-a"}
    assert np.allclose(
        provided.loc[provided["perturbation_label"] == "control", "ps_score"],
        0.0,
    )
    assert provided.attrs["ps_score_exact"]["target_gene_source"] == "provided"
    assert hvg.attrs["ps_score_exact"]["target_gene_source_detail"] == {
        "mode": "hvg",
        "hvg_key": "exact_hvg",
    }
    assert hvg.attrs["ps_score_exact"]["genes_by_perturbation"] == {"pert-b": ["g2", "g3"]}


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
        {"perturbation": ["control"] * 6 + ["pert-a"] * 6 + ["pert-b"] * 4},
        index=[f"cell-{index}" for index in range(matrix.shape[0])],
    )
    var = pd.DataFrame(index=["g0", "g1", "g2"])
    adata = AnnData(matrix, obs=obs, var=var)
    adata.layers["counts"] = matrix.copy()
    return adata


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
