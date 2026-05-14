from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from perturb_effects import CsrBatch, run_mixscape_anndata, run_mixscape_stream


def test_mixscape_stream_exact_matches_schema_and_separates_target() -> None:
    adata = _make_mixscape_adata()

    anndata_result = run_mixscape_anndata(
        adata,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="exact",
        marker_top_k=2,
        n_neighbors=3,
        iter_num=5,
    )
    stream_result = run_mixscape_stream(
        _batch_factory(adata.X, adata.obs.index.to_numpy()),
        obs=adata.obs,
        var_names=adata.var_names,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="exact",
        marker_top_k=2,
        n_neighbors=3,
        iter_num=5,
    )

    assert list(stream_result.columns) == list(anndata_result.columns)
    pert_rows = stream_result[stream_result["target_perturbation"] == "pert-a"]
    target = pert_rows[pert_rows["perturbation_label"] == "pert-a"]
    control = pert_rows[pert_rows["perturbation_label"] == "control"]
    assert target["perturbation_score"].median() > control["perturbation_score"].median()
    assert stream_result.attrs["mixscape"]["stream_mode"] == "multi-pass"


def test_mixscape_stream_approx_supports_one_shot_iterator() -> None:
    adata = _make_mixscape_adata()

    result = run_mixscape_stream(
        iter(_make_csr_batches(adata.X, adata.obs.index.to_numpy())),
        obs=adata.obs,
        var_names=adata.var_names,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
        perturbations=["pert-a"],
        marker_top_k=2,
        control_sample_size=4,
        perturbation_sample_size=4,
        random_state=7,
    )

    assert set(result["fidelity"]) == {"approx"}
    assert np.isfinite(result["perturbation_score"]).all()
    assert np.isfinite(result["posterior_probability"]).all()
    assert result.attrs["mixscape"]["stream_mode"] == "buffered-one-shot"


def test_mixscape_stream_exact_requires_batch_factory_for_multi_pass() -> None:
    adata = _make_mixscape_adata()

    with pytest.raises(ValueError, match="callable batch factory"):
        run_mixscape_stream(
            iter(_make_csr_batches(adata.X, adata.obs.index.to_numpy())),
            obs=adata.obs,
            var_names=adata.var_names,
            perturbation_key="perturbation",
            control_label="control",
            fidelity="exact",
        )


def test_mixscape_stream_perturbation_subset_limits_output() -> None:
    adata = _make_mixscape_adata()

    result = run_mixscape_stream(
        _batch_factory(adata.X, adata.obs.index.to_numpy()),
        obs=adata.obs,
        var_names=adata.var_names,
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
        perturbations=["pert-b"],
        marker_top_k=2,
        random_state=5,
    )

    assert result["target_perturbation"].unique().tolist() == ["pert-b"]
    assert set(result["perturbation_label"]) == {"control", "pert-b"}


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
    return AnnData(matrix, obs=obs, var=var)
