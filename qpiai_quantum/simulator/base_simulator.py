"""
Abstract Base Simulator
=======================
Defines the standard interface for all local quantum simulators
in the QpiAI Quantum SDK.
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

from qpiai_quantum.results.base_result import BaseQuantumResult

if TYPE_CHECKING:
    from qpiai_quantum.circuit import Circuit


class BaseSimulator(ABC):
    """
    Abstract base interface for all local simulators.

    Any new local simulator (e.g., density matrix, tensor network,
    stabilizer) should implement this interface to ensure consistent
    behaviour across the SDK.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the simulator (e.g., 'QpiAI-QSV-Local')."""
        pass

    @abstractmethod
    def run(
        self,
        circuit: "Circuit",
        shots: int = 1024,
        seed: int | None = None,
        name: str | None = None,
    ) -> BaseQuantumResult:
        """
        Execute the given circuit and return a result object.

        Args:
            circuit: The quantum circuit to execute.
            shots: Number of measurement shots to perform.
            seed: Optional RNG seed for reproducibility.
            name: Optional name for the result object.

        Returns:
            A result object adhering to BaseQuantumResult.
        """
        pass
