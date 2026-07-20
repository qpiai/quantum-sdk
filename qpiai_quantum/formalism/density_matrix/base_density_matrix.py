"""
Base Density Matrix Class

Provides common functionality shared between different DensityMatrix implementations:
- qpiai_quantum.quantum_info.DensityMatrix (user-facing, circuit-based)
- qpiai_quantum.formalism.DensityMatrix (advanced formalism operations)

This base class contains core operations that all density matrix implementations share.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class BaseDensityMatrix(ABC):
    """
    Abstract base class for density matrix implementations.

    A density matrix is a mathematical representation of a quantum state that can describe
    both pure and mixed states. This base class provides common operations that all
    density matrix implementations should support.

    Subclasses must implement:
    - _get_data(): Return the underlying numpy array representation
    - _get_num_qubits(): Return the number of qubits
    """

    @abstractmethod
    def _get_data(self) -> np.ndarray:
        """
        Get the underlying density matrix data as a numpy array.

        Returns:
            np.ndarray: The density matrix as a 2D complex numpy array
        """
        pass

    @abstractmethod
    def _get_num_qubits(self) -> int:
        """
        Get the number of qubits represented by this density matrix.

        Returns:
            int: Number of qubits
        """
        pass

    def purity(self) -> float:
        """
        Calculate the purity of the quantum state.

        Purity is defined as Tr(ρ²) where ρ is the density matrix.

        Properties:
        - Pure state: purity = 1
        - Maximally mixed state: purity = 1/d (where d is the dimension)
        - Range: 1/d ≤ purity ≤ 1

        Returns:
            float: The purity value

        Example:
            >>> dm = DensityMatrix.from_label("0")
            >>> dm.purity()
            1.0  # Pure state

            >>> # Maximally mixed state
            >>> dm_mixed = DensityMatrix(np.eye(2) / 2)
            >>> dm_mixed.purity()
            0.5
        """
        rho = self._get_data()
        rho_squared = rho @ rho
        return float(np.real(np.trace(rho_squared)))

    def von_neumann_entropy(self) -> float:
        """
        Calculate the von Neumann entropy of the quantum state.

        The von Neumann entropy is defined as:
        S(ρ) = -Tr(ρ log₂ ρ) = -Σᵢ λᵢ log₂(λᵢ)

        where λᵢ are the eigenvalues of the density matrix.

        Properties:
        - Pure state: entropy = 0
        - Maximally mixed state: entropy = log₂(d) where d is the dimension
        - Range: 0 ≤ entropy ≤ log₂(d)

        Returns:
            float: The von Neumann entropy in bits (base 2)

        Example:
            >>> dm = DensityMatrix.from_label("0")
            >>> dm.von_neumann_entropy()
            0.0  # Pure state has zero entropy

            >>> # Maximally mixed single-qubit state
            >>> dm_mixed = DensityMatrix(np.eye(2) / 2)
            >>> dm_mixed.von_neumann_entropy()
            1.0  # log₂(2) = 1
        """
        rho = self._get_data()
        eigenvalues = np.linalg.eigvalsh(rho)

        # Filter out zero and negative eigenvalues (due to numerical errors)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]

        # Compute -Σ λᵢ log₂(λᵢ)
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

        return float(np.real(entropy))

    def trace(self) -> complex:
        """
        Calculate the trace of the density matrix.

        For a valid density matrix, Tr(ρ) should equal 1.

        Returns:
            complex: The trace of the density matrix

        Example:
            >>> dm = DensityMatrix.from_label("0")
            >>> dm.trace()
            (1+0j)
        """
        rho = self._get_data()
        return np.trace(rho)

    def is_valid(self, atol: float = 1e-8) -> bool:
        """
        Check if this is a valid density matrix.

        A valid density matrix must satisfy three conditions:
        1. Hermitian: ρ = ρ† (equal to its conjugate transpose)
        2. Positive semidefinite: all eigenvalues ≥ 0
        3. Unit trace: Tr(ρ) = 1

        Args:
            atol: Absolute tolerance for comparisons (default: 1e-8)

        Returns:
            bool: True if valid density matrix, False otherwise

        Example:
            >>> dm = DensityMatrix.from_label("0")
            >>> dm.is_valid()
            True
        """
        rho = self._get_data()

        # Check 1: Hermitian (ρ = ρ†)
        if not np.allclose(rho, np.conj(rho.T), atol=atol):
            return False

        # Check 2: Positive semidefinite (all eigenvalues ≥ 0)
        eigenvalues = np.linalg.eigvalsh(rho)
        if np.any(eigenvalues < -atol):
            return False

        # Check 3: Unit trace (Tr(ρ) = 1)
        trace = np.trace(rho)
        if not np.abs(trace - 1.0) < atol:
            return False

        return True

    def is_pure(self, atol: float = 1e-8) -> bool:
        """
        Check if the state is pure.

        A pure state has purity = 1, or equivalently Tr(ρ²) = 1.

        Args:
            atol: Absolute tolerance (default: 1e-8)

        Returns:
            bool: True if pure state, False if mixed

        Example:
            >>> dm = DensityMatrix.from_label("0")
            >>> dm.is_pure()
            True

            >>> dm_mixed = DensityMatrix(np.eye(2) / 2)
            >>> dm_mixed.is_pure()
            False
        """
        return abs(self.purity() - 1.0) < atol

    def fidelity(self, other: "BaseDensityMatrix", validate: bool = True) -> float:
        """
        Calculate the fidelity between this density matrix and another.

        Fidelity is a measure of similarity between quantum states:
        F(ρ, σ) = Tr(√(√ρ σ √ρ))²

        Properties:
        - Range: 0 ≤ F ≤ 1
        - F = 1 for identical states
        - F = 0 for orthogonal states

        Args:
            other: Another density matrix
            validate: Whether to validate that both are valid density matrices

        Returns:
            float: The fidelity value between 0 and 1

        Raises:
            ValueError: If dimensions don't match or states are invalid

        Example:
            >>> dm1 = DensityMatrix.from_label("0")
            >>> dm2 = DensityMatrix.from_label("0")
            >>> dm1.fidelity(dm2)
            1.0  # Identical states

            >>> dm3 = DensityMatrix.from_label("1")
            >>> dm1.fidelity(dm3)
            0.0  # Orthogonal states
        """
        rho = self._get_data()
        sigma = other._get_data()

        # Check dimensions match
        if rho.shape != sigma.shape:
            raise ValueError(
                f"Density matrices must have same dimensions. "
                f"Got {rho.shape} and {sigma.shape}"
            )

        # Validate if requested
        if validate:
            if not self.is_valid():
                raise ValueError("First density matrix is not valid")
            if not other.is_valid():
                raise ValueError("Second density matrix is not valid")

        # Compute fidelity: F = Tr(√(√ρ σ √ρ))²
        # For numerical stability, use eigendecomposition

        # Compute √ρ
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
        sqrt_rho = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T

        # Compute √ρ σ √ρ
        M = sqrt_rho @ sigma @ sqrt_rho

        # Compute √M
        eigenvalues_M = np.linalg.eigvalsh(M)
        eigenvalues_M = np.maximum(eigenvalues_M, 0)

        # Fidelity = (Tr(√M))²
        fidelity = np.sum(np.sqrt(eigenvalues_M)) ** 2

        return float(np.real(fidelity))

    def partial_trace(
        self, qubits_to_keep: list, dims: list | None = None
    ) -> np.ndarray:
        """
        Compute the partial trace over specified qubits.

        The partial trace is an operation that "traces out" or discards some subsystems,
        leaving a reduced density matrix describing only the remaining subsystems.

        Args:
            qubits_to_keep: list of qubit indices to keep (0-indexed)
            dims: list of dimensions for each subsystem. If None, assumes all qubits are 2D

        Returns:
            np.ndarray: The reduced density matrix

        Example:
            >>> # 2-qubit Bell state |00⟩ + |11⟩
            >>> circuit = Circuit(2)
            >>> circuit.h(0)
            >>> circuit.cx(0, 1)
            >>> dm = DensityMatrix(circuit)
            >>> # Trace out qubit 1, keep qubit 0
            >>> reduced = dm.partial_trace([0])
            >>> # Result is maximally mixed: [[0.5, 0], [0, 0.5]]
        """
        rho = self._get_data()
        num_qubits = self._get_num_qubits()

        # Set default dimensions (all qubits are 2D)
        if dims is None:
            dims = [2] * num_qubits

        # Validate input
        if len(dims) != num_qubits:
            raise ValueError(
                f"dims length {len(dims)} must match num_qubits {num_qubits}"
            )

        # Reshape to tensor form
        shape = dims + dims
        rho_tensor = rho.reshape(shape)

        # Determine which qubits to trace out
        all_qubits = set(range(num_qubits))
        qubits_to_trace = sorted(list(all_qubits - set(qubits_to_keep)))

        # Trace out qubits one by one (in reverse order to maintain indices)
        for qubit in reversed(qubits_to_trace):
            # Sum over the diagonal of this qubit
            rho_tensor = np.trace(rho_tensor, axis1=qubit, axis2=qubit + num_qubits)
            # Adjust num_qubits for next iteration
            num_qubits -= 1

        # Reshape back to matrix form
        remaining_dim = int(np.sqrt(rho_tensor.size))
        reduced_rho = rho_tensor.reshape(remaining_dim, remaining_dim)

        return reduced_rho

    def __str__(self) -> str:
        """String representation of the density matrix."""
        return f"DensityMatrix({self._get_num_qubits()} qubits, purity={self.purity():.4f})"

    def __repr__(self) -> str:
        """Detailed representation of the density matrix."""
        rho = self._get_data()
        return f"DensityMatrix(shape={rho.shape}, purity={self.purity():.4f}, valid={self.is_valid()})"
