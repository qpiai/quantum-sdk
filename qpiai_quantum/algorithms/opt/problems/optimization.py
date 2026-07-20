import warnings
from qpiai_quantum.applications.optimization.problems import (
    MaxCutProblem as _NewMaxCutProblem,
    PortfolioOptimizationProblem as _NewPortfolioProblem,
)

__all__ = ["MaxCutProblem", "PortfolioOptimizationProblem"]  # noqa: F822


def __getattr__(name: str):
    if name in ("MaxCutProblem", "PortfolioOptimizationProblem"):
        warnings.warn(
            f"Importing {name} from `qpiai_quantum.algorithms.opt.problems` is deprecated. "
            f"Please import from `qpiai_quantum.applications.optimization` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name == "MaxCutProblem":
            return _NewMaxCutProblem
        return _NewPortfolioProblem

    raise AttributeError(f"module {__name__} has no attribute {name}")
