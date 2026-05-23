from __future__ import annotations

import numpy as np
from anndata import AnnData

from perturb_effects.parallel import partition_perturbations, run_parallel_tasks
from perturb_effects.stats import top_k_indices, welch_t_scores
from perturb_effects.stream import extract_anndata_matrix
from perturb_effects.utils import resolve_perturbations


def test_extract_anndata_matrix_respects_layer_selection() -> None:
    adata = AnnData(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float))
    adata.layers["counts"] = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=float)

    assert np.array_equal(np.asarray(extract_anndata_matrix(adata)), adata.X)
    assert np.array_equal(
        np.asarray(extract_anndata_matrix(adata, layer="counts")),
        adata.layers["counts"],
    )


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

    ranking = top_k_indices(welch_t_scores(perturbed, control), 2)
    assert ranking.tolist()[0] == 0
    assert top_k_indices([0.1, -5.0, 1.0], 2).tolist() == [1, 2]


def test_run_parallel_tasks_preserves_input_order() -> None:
    items = ["pert-a", "pert-b", "pert-c"]

    results = run_parallel_tasks(items, lambda item: item.upper(), n_jobs=2)

    assert results == ["PERT-A", "PERT-B", "PERT-C"]
