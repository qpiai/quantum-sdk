"""
Quantum State Interface
=======================
Abstract base class defining the standard interface for quantum state
representations (StateVector, DensityMatrix, Stabilizer, etc.) in the
QpiAI Quantum SDK.
"""

from abc import ABC, abstractmethod
import numpy as np


class QuantumState(ABC):
    """Abstract interface for quantum state representations."""

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Return the number of qubits in the quantum state."""
        pass

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """Return the underlying state array (e.g. 1D vector or 2D matrix)."""
        pass

    # ----------------------------------------------------------------------
    # Single-Qubit Gates
    # ----------------------------------------------------------------------

    @abstractmethod
    def apply_h(self, target: int) -> None:
        """Apply Hadamard gate to target qubit."""
        pass

    @abstractmethod
    def apply_x(self, target: int) -> None:
        """Apply Pauli-X gate to target qubit."""
        pass

    @abstractmethod
    def apply_y(self, target: int) -> None:
        """Apply Pauli-Y gate to target qubit."""
        pass

    @abstractmethod
    def apply_z(self, target: int) -> None:
        """Apply Pauli-Z gate to target qubit."""
        pass

    @abstractmethod
    def apply_s(self, target: int) -> None:
        """Apply Phase (S) gate to target qubit."""
        pass

    @abstractmethod
    def apply_sdg(self, target: int) -> None:
        """Apply S† gate to target qubit."""
        pass

    @abstractmethod
    def apply_t(self, target: int) -> None:
        """Apply T (π/8) gate to target qubit."""
        pass

    @abstractmethod
    def apply_tdg(self, target: int) -> None:
        """Apply T† gate to target qubit."""
        pass

    @abstractmethod
    def apply_rx(self, target: int, theta: float) -> None:
        """Apply RX rotation gate by theta radians to target qubit."""
        pass

    @abstractmethod
    def apply_ry(self, target: int, theta: float) -> None:
        """Apply RY rotation gate by theta radians to target qubit."""
        pass

    @abstractmethod
    def apply_rz(self, target: int, theta: float) -> None:
        """Apply RZ rotation gate by theta radians to target qubit."""
        pass

    # ----------------------------------------------------------------------
    # Multi-Qubit Gates
    # ----------------------------------------------------------------------

    @abstractmethod
    def apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT (CX) gate between control and target qubits."""
        pass

    @abstractmethod
    def apply_cz(self, control: int, target: int) -> None:
        """Apply Controlled-Z (CZ) gate between control and target qubits."""
        pass

    @abstractmethod
    def apply_swap(self, target1: int, target2: int) -> None:
        """Apply SWAP gate between target1 and target2 qubits."""
        pass

    @abstractmethod
    def apply_cswap(self, control: int, target1: int, target2: int) -> None:
        """Apply Fredkin (CSWAP) gate."""
        pass

    # ----------------------------------------------------------------------
    # Generic Unitary & Channel Evolution
    # ----------------------------------------------------------------------

    @abstractmethod
    def apply_unitary(self, matrix: np.ndarray, qubits: list[int]) -> None:
        """Apply an arbitrary k-qubit unitary matrix to the specified qubits."""
        pass

    # ----------------------------------------------------------------------
    # Diagnostics & Measurement
    # ----------------------------------------------------------------------

    @abstractmethod
    def get_probabilities(self) -> np.ndarray:
        """Return measurement probabilities for all 2^n computational basis states."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the quantum state back to |0...0>."""
        pass

