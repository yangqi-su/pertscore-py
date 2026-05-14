from __future__ import annotations

import pytest

import perturb_effects
from perturb_effects import CsrBatch


def test_public_api_exports_exist() -> None:
    assert perturb_effects.run_mixscape_anndata is not None
    assert perturb_effects.run_mixscape_stream is not None
    assert perturb_effects.run_ps_score_anndata is not None
    assert perturb_effects.run_ps_score_exact_anndata is not None
    assert perturb_effects.run_ps_score_stream is not None


def test_csr_batch_construction() -> None:
    batch = CsrBatch(
        row_ids=["cell-1", "cell-2"],
        indptr=[0, 2, 3],
        indices=[0, 3, 1],
        data=[1.0, 2.0, 3.0],
        shape=(2, 4),
    )

    assert batch.shape == (2, 4)
    assert list(batch.row_ids) == ["cell-1", "cell-2"]


def test_invalid_fidelity_is_rejected_before_execution() -> None:
    with pytest.raises(ValueError, match="Unsupported fidelity"):
        perturb_effects.run_mixscape_anndata(
            object(),
            layer=None,
            perturbation_key="perturbation",
            control_label="control",
            fidelity="unsupported",
        )


def test_stream_ps_api_returns_empty_frame_when_only_control_rows_are_present() -> None:
    result = perturb_effects.run_ps_score_stream(
        [
            CsrBatch(
                row_ids=["cell-1"],
                indptr=[0, 1],
                indices=[0],
                data=[1.0],
                shape=(1, 1),
            )
        ],
        obs=[{"row_id": "cell-1", "perturbation": "control"}],
        var_names=["gene-1"],
        perturbation_key="perturbation",
        control_label="control",
        fidelity="approx",
    )

    assert result.empty
