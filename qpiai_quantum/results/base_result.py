"""
Base Result Class

Provides an abstract base class for all quantum execution result types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseQuantumResult(ABC):
    """
    Abstract base class for quantum execution results.

    This class defines a common interface that all result types must implement,
    ensuring consistency across different execution methods (Executor, JobManager, etc.).

    All result classes should inherit from this base to provide:
    - Consistent access to measurement counts
    - Standardized statevector/state access
    - Uniform timing and metadata retrieval
    - Common plotting and analysis methods

    Subclasses must implement the abstract properties for core result data.
    """

    @property
    @abstractmethod
    def counts(self) -> dict[str, int] | None:
        """
        Get measurement counts.

        Returns a dictionary mapping measurement outcomes (bitstrings) to
        the number of times each outcome was observed.

        Returns:
            Dict[str, int]: Measurement counts, e.g. {'00': 512, '11': 512}
                           Returns None if no measurements were performed.

        Example:
            >>> result = circuit.run(shots=1024)
            >>> result.counts
            {'00': 256, '01': 256, '10': 256, '11': 256}
        """
        pass

    @property
    @abstractmethod
    def statevector(self) -> list | None:
        """
        Get the quantum statevector (if available).

        Returns the full quantum state as a statevector. This is only available
        for statevector simulation methods.

        Returns:
            List[complex]: The statevector as a list of complex amplitudes
                          Returns None if statevector not available (e.g., for hardware runs)

        Example:
            >>> result = circuit.run(backend=Backend.STATEVECTOR_SIMULATOR_CPU)
            >>> result.statevector
            [[0.707+0j], [0+0j], [0+0j], [0.707+0j]]
        """
        pass

    @property
    @abstractmethod
    def execution_time(self) -> float:
        """
        Get the execution time in seconds.

        Returns:
            float: Execution time in seconds

        Example:
            >>> result = circuit.run(shots=1024)
            >>> result.execution_time
            0.523
        """
        pass

    @property
    @abstractmethod
    def shots(self) -> int | None:
        """
        Get the number of shots used.

        Returns:
            int: Number of measurement shots

        Example:
            >>> result = circuit.run(shots=2048)
            >>> result.shots
            2048
        """
        pass

    @property
    def probabilities(self) -> dict[str, float] | None:
        """
        Get measurement probabilities (derived from counts).

        Calculates the probability of each measurement outcome by dividing
        counts by total shots.

        Returns:
            Dict[str, float]: Probabilities for each outcome
                             Returns None if counts not available

        Example:
            >>> result = circuit.run(shots=1024)
            >>> result.probabilities
            {'00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25}
        """
        if self.counts is None or not self.shots:
            return None

        total_shots = self.shots
        assert total_shots is not None
        return {outcome: count / total_shots for outcome, count in self.counts.items()}

    def get_counts(self) -> dict[str, int] | None:
        """
        Get measurement counts (method version of property).

        Returns:
            Dict[str, int]: Measurement counts

        Example:
            >>> counts = result.get_counts()
        """
        return self.counts

    def get_statevector(self) -> list | None:
        """
        Get statevector (method version of property).

        Returns:
            List[complex]: Statevector amplitudes

        Example:
            >>> sv = result.get_statevector()
        """
        return self.statevector

    def get_probabilities(self) -> dict[str, float] | None:
        """
        Get measurement probabilities (method version of property).

        Returns:
            Dict[str, float]: Outcome probabilities

        Example:
            >>> probs = result.get_probabilities()
        """
        return self.probabilities

    @property
    def job_id(self) -> str | None:
        """
        Get the job ID associated with the execution.

        Returns:
            Optional[str]: The job ID if available, None otherwise.
        """
        return None

    @property
    def job_status(self) -> str | None:
        """
        Get the status of the execution job.

        Returns:
            Optional[str]: The job status if available, None otherwise.
        """
        return None

    def get_job_id(self) -> str | None:
        """
        Get the job ID associated with the execution (method version of property).

        Returns:
            Optional[str]: The job ID if available.
        """
        return self.job_id

    def get_job_status(self) -> str | None:
        """
        Get the status of the execution job (method version of property).

        Returns:
            Optional[str]: The job status if available.
        """
        return self.job_status

    @abstractmethod
    def get(self, *params) -> dict[str, Any]:
        """
        Get result data as a dictionary.

        If no parameters provided, returns all available data.
        If parameters provided, returns only requested fields.

        Args:
            *params: Optional field names to retrieve

        Returns:
            Dict[str, Any]: Dictionary containing requested result data

        Example:
            >>> # Get all data
            >>> data = result.get()

            >>> # Get specific fields
            >>> data = result.get("counts", "execution_time")
        """
        pass

    def __repr__(self) -> str:
        """
        String representation of the result.

        Returns:
            str: Human-readable result summary
        """
        class_name = self.__class__.__name__
        has_counts = self.counts is not None
        has_sv = self.statevector is not None

        parts = [f"{class_name}("]
        if self.shots:
            parts.append(f"shots={self.shots}")
        if has_counts:
            parts.append(f"outcomes={len(self.counts or {})}")
        if has_sv:
            parts.append("statevector=available")
        if self.execution_time:
            parts.append(f"time={self.execution_time:.3f}s")

        return ", ".join(parts) + ")"

    def plot(self, *other_results, **kwargs):
        """
        Plot measurement results.

        Creates a bar plot comparing measurement probabilities across
        multiple results (if provided).

        Args:
            *other_results: Additional results to compare
            **kwargs: Plotting options including:
                - labels (list): Custom labels for each result
                - title, xlabel, ylabel, figsize, rotation, legend_loc

        Example:
            >>> result1 = circuit1.run(shots=1024)
            >>> result2 = circuit2.run(shots=1024)
            >>> result1.plot(result2, labels=["Hardware", "Simulator"])
            >>> result1.plot(result2, title="Comparison")
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install matplotlib"
            )

        all_probabilities = []

        # Collect probabilities from self and other results
        if self.probabilities:
            all_probabilities.append(self.probabilities)

        for other in other_results:
            if hasattr(other, "probabilities") and other.probabilities:
                all_probabilities.append(other.probabilities)
            elif isinstance(other, dict):
                all_probabilities.append(other)

        if not all_probabilities:
            raise ValueError("No probability data available to plot")

        # Handle labels
        custom_labels = kwargs.get("labels", None)
        if custom_labels:
            if len(custom_labels) != len(all_probabilities):
                raise ValueError(
                    f"Number of labels ({len(custom_labels)}) must match "
                    f"number of results ({len(all_probabilities)})"
                )
            labels = custom_labels
        else:
            # Auto-generate labels: "Result 1", "Result 2", etc.
            labels = [f"Result {i + 1}" for i in range(len(all_probabilities))]

        all_outcomes = sorted(set().union(*[p.keys() for p in all_probabilities]))

        x_pos = np.arange(len(all_outcomes))
        width = 0.8 / len(all_probabilities)

        plt.figure(figsize=kwargs.get("figsize", (10, 6)))

        for idx, (probs, label) in enumerate(zip(all_probabilities, labels)):
            values = [probs.get(outcome, 0) for outcome in all_outcomes]
            offset = width * (idx - (len(all_probabilities) - 1) / 2)
            plt.bar(x_pos + offset, values, width, label=label)

        plt.xlabel(kwargs.get("xlabel", "Measurement Outcome"))
        plt.ylabel(kwargs.get("ylabel", "Probability"))
        plt.title(kwargs.get("title", "Measurement Results"))
        plt.xticks(x_pos, all_outcomes, rotation=kwargs.get("rotation", 45))
        plt.legend(loc=kwargs.get("legend_loc", "upper right"))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
