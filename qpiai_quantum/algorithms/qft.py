"""
Quantum Fourier Transform (QFT) Algorithm Implementation
"""

import math
from typing import Optional
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager.job_result import JobResult
from .base import QuantumAlgorithm


class QFT(QuantumAlgorithm):
    """
    Quantum Fourier Transform algorithm.

    The QFT transforms a quantum state from the computational basis to the Fourier basis.
    For an n-qubit state |x⟩, the QFT produces:

    QFT|x⟩ = (1/√2^n) Σ_k e^(2πixk/2^n) |k⟩

    """

    def __init__(self, num_qubits: int, inverse: bool = False):
        super().__init__(
            num_qubits=num_qubits, name="QFT" if not inverse else "Inverse QFT"
        )
        self.inverse = inverse
        self.description = (
            "Quantum Fourier Transform - quantum analog of the discrete Fourier transform"
            if not inverse
            else "Inverse Quantum Fourier Transform"
        )

    def build_circuit(
        self, initialize_superposition: bool = False, measure: bool = True
    ) -> Circuit:
        """
        Build the QFT circuit.

        Args:
            initialize_superposition (bool): If True, initialize all qubits to |+⟩ state
            measure (bool): If True, add measurement operations

        Returns:
            Circuit: The QFT circuit
        """
        if measure:
            self.circuit = Circuit(self.num_qubits, self.num_qubits)
        else:
            from qpiai_quantum.circuit.quantumregister import QuantumRegister

            self.circuit = Circuit(QuantumRegister(self.num_qubits, "q"))

        # Optional: Initialize to superposition
        if initialize_superposition:
            for i in range(self.num_qubits):
                self.circuit.h(i)

        # Build QFT or inverse QFT
        if self.inverse:
            self._build_inverse_qft()
        else:
            self._build_qft()

        # Optional: Add measurements
        if measure:
            for i in range(self.num_qubits):
                self.circuit.measure(i, i)

        return self.circuit

    def _build_qft(self):
        """Build the forward QFT circuit."""
        # NOTE: This SDK uses little-endian qubit ordering (qubit 0 is MSB)
        # Standard QFT assumes big-endian (qubit 0 is LSB) and needs swaps
        # For little-endian, we need to iterate in REVERSE order to avoid swaps

        for i in range(self.num_qubits - 1, -1, -1):
            # Apply Hadamard gate
            self.circuit.h(i)  # type: ignore

            # Apply controlled phase rotations
            for j in range(i - 1, -1, -1):
                # Controlled phase rotation by 2π/2^(i-j+1)
                theta = 2 * math.pi / (2 ** (i - j + 1))
                self.circuit.cp(j, i, theta)  # type: ignore

    def _build_inverse_qft(self):
        """Build the inverse QFT circuit."""
        # NOTE: Inverse QFT for little-endian qubit ordering (no swaps needed)
        # Just reverse the forward QFT with negated phases

        # Apply gates in reverse order of forward QFT
        for i in range(self.num_qubits):
            # Apply controlled phase rotations in reverse (negated phases)
            for j in range(i - 1, -1, -1):
                theta = -2 * math.pi / (2 ** (i - j + 1))
                self.circuit.cp(j, i, theta)  # type: ignore

            # Apply Hadamard gate
            self.circuit.h(i)  # type: ignore

    @staticmethod
    def apply_qft_to_circuit(circuit: Circuit, start: int, n: int):
        """
        Apply QFT to a subset of qubits in an existing circuit.

        This is useful for embedding QFT as a subroutine in larger algorithms.
        NOTE: Adapted for little-endian qubit ordering (qubit 0 is MSB).

        Args:
            circuit (Circuit): The circuit to modify
            start (int): Starting qubit index
            n (int): Number of qubits to apply QFT to
        """
        # Iterate in reverse for little-endian ordering
        for i in range(start + n - 1, start - 1, -1):
            circuit.h(i)
            for j in range(i - 1, start - 1, -1):
                theta = 2 * math.pi / (2 ** (i - j + 1))
                circuit.cp(j, i, theta)

    @staticmethod
    def apply_inverse_qft_to_circuit(circuit: Circuit, start: int, n: int):
        """
        Apply inverse QFT to a subset of qubits in an existing circuit.

        NOTE: Adapted for little-endian qubit ordering (qubit 0 is MSB).

        Args:
            circuit (Circuit): The circuit to modify
            start (int): Starting qubit index
            n (int): Number of qubits to apply inverse QFT to
        """

        # Apply inverse transformations in reverse order of forward QFT
        for i in range(start, start + n):
            # Controlled phases first (reversed order with negated angles)
            for j in range(i - 1, start - 1, -1):
                theta = -2 * math.pi / (2 ** (i - j + 1))
                circuit.cp(j, i, theta)
            # H gate last
            circuit.h(i)
