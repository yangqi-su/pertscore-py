"""Public perturbation score API contract for Phase 1 implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .parallel import normalize_n_jobs
from .stats import validate_fidelity, validate_layer, validate_perturbations


def run_ps_score_anndata(
    adata: Any,
    *,
    layer: str | None = None,
    perturbation_key: str,
    control_label: str,
    fidelity: str = "exact",
    perturbations: Sequence[str] | None = None,
    n_jobs: int | None = 1,
) -> Any:
    """Validate the AnnData perturbation score API contract for later phases."""

    if adata is None:
        raise ValueError("adata must not be None")
    if not isinstance(perturbation_key, str) or not perturbation_key:
        raise ValueError("perturbation_key must be a non-empty string")
    if not isinstance(control_label, str) or not control_label:
        raise ValueError("control_label must be a non-empty string")
    validate_layer(layer)
    validate_fidelity(fidelity)
    validate_perturbations(perturbations)
    normalize_n_jobs(n_jobs)
    raise NotImplementedError(
        "PS score AnnData execution is not implemented yet; Phase 1 defines the API contract only."
    )


def run_ps_score_stream(
    batches: Any,
    *,
    obs: Any,
    var_names: Sequence[str],
    perturbation_key: str,
    control_label: str,
    fidelity: str = "exact",
    perturbations: Sequence[str] | None = None,
    n_jobs: int | None = 1,
) -> Any:
    """Validate the streamed perturbation score API contract for later phases."""

    if batches is None:
        raise ValueError("batches must not be None")
    if obs is None:
        raise ValueError("obs must not be None")
    if not isinstance(var_names, Sequence) or isinstance(var_names, str):
        raise TypeError("var_names must be a sequence of feature names")
    if not isinstance(perturbation_key, str) or not perturbation_key:
        raise ValueError("perturbation_key must be a non-empty string")
    if not isinstance(control_label, str) or not control_label:
        raise ValueError("control_label must be a non-empty string")
    validate_fidelity(fidelity)
    validate_perturbations(perturbations)
    normalize_n_jobs(n_jobs)
    raise NotImplementedError(
        "PS score streamed execution is not implemented yet; Phase 1 defines the API contract only."
    )
