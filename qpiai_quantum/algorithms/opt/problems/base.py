from abc import ABC, abstractmethod
from typing import Any, Optional


class OptimizationProblem(ABC):
    """Abstract base class for optimization problems."""

    def __init__(self):
        """Initialize optimization problem."""
        self.n_qubits = 0

    @abstractmethod
    def _construct_hamiltonian(self) -> Any:
        """
        Construct problem Hamiltonian in library-specific format.

        Returns:
            Hamiltonian representation
        """
        pass

    @abstractmethod
    def _get_hamiltonian_terms(self) -> list[tuple[list[tuple[int, str]], float]]:
        """
        Get Hamiltonian terms in library-agnostic format.

        Returns:
            list of (operator_list, coefficient) tuples where:
            - operator_list: [(qubit_idx, 'X'/'Y'/'Z'), ...]
            - coefficient: float weight for this term
        """
        pass

    def get_hamiltonian_terms(self) -> list[tuple[list[tuple[int, str]], float]]:
        """
        Public method to get Hamiltonian terms for VQE and other solvers.

        Returns:
            list of (operator_list, coefficient) tuples
        """
        return self._get_hamiltonian_terms()

    @abstractmethod
    def decode_solution(self, bitstring: str) -> dict[str, Any]:
        """
        Decode quantum measurement bitstring into problem solution.

        Args:
            bitstring: Binary measurement outcome

        Returns:
            Problem-specific solution dictionary
        """
        pass

    @abstractmethod
    def validate_solution(self, solution: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate if solution satisfies problem constraints.

        Args:
            solution: Problem solution to validate

        Returns:
            tuple of (is_valid, message)
        """
        pass

    @abstractmethod
    def compute_solution_quality(self, solution: dict[str, Any]) -> dict[str, float]:
        """
        Compute quality metrics for a solution.

        Args:
            solution: Problem solution

        Returns:
            Dictionary of quality metrics
        """
        pass
