"""Public API for PS score calculation from h5ad data."""

__all__ = [
    "run_ps_score_exact_fast",
]


def __getattr__(name: str):
    if name == "run_ps_score_exact_fast":
        from .ps_score_exact_fast import run_ps_score_exact_fast

        return run_ps_score_exact_fast
    raise AttributeError(name)
