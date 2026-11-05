"""Progress retained when an optimizer exits before returning its result."""

from dataclasses import dataclass


@dataclass
class OptimizationProgress:
    """Keep completed steps separate from backend callback notifications.

    ``completed_iterations`` stays unknown for interrupted SciPy methods
    unless a callback provides the native count (as Differential Evolution
    does for generations). Other callback counts are reported separately.
    """

    completed_iterations: int | None = None
    callback_reports: int = 0

    def complete_iteration(self) -> None:
        """Record a fully completed step in an SDK-owned optimization loop."""
        if self.completed_iterations is None:
            self.completed_iterations = 0
        self.completed_iterations += 1
