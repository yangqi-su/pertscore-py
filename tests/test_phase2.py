from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from perturb_effects.parallel import partition_perturbations, run_parallel_tasks
from perturb_effects.stats import (
    csr_batch_to_matrix,
    extract_anndata_matrix,
    rank_features_by_welch_t,
    require_reiterable_batches,
    resolve_perturbations,
    summarize_streamed_features,
    top_k_indices,
    validate_csr_batch,
)
from perturb_effects.types import CsrBatch


def test_extract_anndata_matrix_respects_layer_selection() -> None:
    adata = AnnData(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    adata.layers["counts"] = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)

    assert np.array_equal(np.asarray(extract_anndata_matrix(adata)), adata.X)
    assert np.array_equal(
        np.asarray(extract_anndata_matrix(adata, layer="counts")),
        adata.layers["counts"],
    )


def test_validate_csr_batch_and_convert_to_matrix() -> None:
    batch = CsrBatch(
        row_ids=["cell-1", "cell-2"],
        indptr=[0, 2, 3],
        indices=[0, 2, 1],
        data=[1.0, 2.0, 3.0],
        shape=(2, 4),
    )

    assert validate_csr_batch(batch) is batch
    matrix = csr_batch_to_matrix(batch)
    assert np.array_equal(
        matrix.toarray(),
        np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0]], dtype=float),
    )


def test_validate_csr_batch_rejects_out_of_bounds_indices() -> None:
    batch = CsrBatch(
        row_ids=["cell-1"],
        indptr=[0, 1],
        indices=[2],
        data=[1.0],
        shape=(1, 2),
    )

    with pytest.raises(ValueError, match="column range"):
        validate_csr_batch(batch)


def test_resolve_perturbations_defaults_to_all_non_controls() -> None:
    labels = ["control", "pert-a", "pert-b", "pert-a", None]

    assert resolve_perturbations(labels, control_label="control") == ["pert-a", "pert-b"]
    assert resolve_perturbations(
        labels,
        control_label="control",
        perturbations=["pert-b"],
    ) == ["pert-b"]


def test_partition_perturbations_is_deterministic() -> None:
    items = ["a", "b", "c", "d", "e"]

    assert partition_perturbations(items, 0, 3) == ["a", "d"]
    assert partition_perturbations(items, 1, 3) == ["b", "e"]
    assert partition_perturbations(items, 2, 3) == ["c"]


def test_require_reiterable_batches_rejects_one_shot_iterators() -> None:
    one_shot = iter(
        [
            CsrBatch(
                row_ids=["cell-1"],
                indptr=[0, 1],
                indices=[0],
                data=[1.0],
                shape=(1, 1),
            )
        ]
    )

    with pytest.raises(ValueError, match="multi-pass"):
        require_reiterable_batches(one_shot, operation="exact stream scoring")


def test_streamed_summary_matches_expected_moments() -> None:
    batch = CsrBatch(
        row_ids=["cell-1", "cell-2", "cell-3"],
        indptr=[0, 2, 3, 5],
        indices=[0, 1, 1, 0, 2],
        data=[1.0, 2.0, 3.0, 4.0, 5.0],
        shape=(3, 3),
    )

    summary = summarize_streamed_features([batch], selected_row_ids={"cell-1", "cell-3"})

    assert summary.count == 2
    assert np.allclose(summary.means(), np.array([2.5, 1.0, 2.5]))
    assert np.allclose(summary.variances(ddof=1), np.array([4.5, 2.0, 12.5]))


def test_welch_ranking_and_top_k_find_shifted_marker() -> None:
    control = np.array(
        [
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    perturbed = np.array(
        [
            [5.0, 1.0, 1.0],
            [6.0, 1.0, 1.0],
            [7.0, 1.0, 1.0],
        ]
    )

    ranking = rank_features_by_welch_t(perturbed, control, top_k=2)
    assert ranking.tolist()[0] == 0
    assert top_k_indices([0.1, -5.0, 1.0], 2).tolist() == [1, 2]


def test_run_parallel_tasks_preserves_input_order() -> None:
    items = ["pert-a", "pert-b", "pert-c"]

    results = run_parallel_tasks(items, lambda item: item.upper(), n_jobs=2)

    assert results == ["PERT-A", "PERT-B", "PERT-C"]
