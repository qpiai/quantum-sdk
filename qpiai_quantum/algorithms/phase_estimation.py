"""
Quantum Phase Estimation Algorithm Implementation

Quantum Phase Estimation is a key subroutine in many quantum algorithms,
including Shor's algorithm and quantum chemistry simulations.
"""

import math
from typing import Optional
from collections.abc import Callable
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager.job_result import JobResult
from .base import QuantumAlgorithm
from .qft import QFT


class QuantumPhaseEstimation(QuantumAlgorithm):
    """
    Quantum Phase Estimation Algorithm.

    Estimates the phase θ in U|ψ⟩ = e^(2πiθ)|ψ⟩ where U is a unitary operator
    and |ψ⟩ is an eigenvector of U.

    Example:
        >>> from qpiai_quantum.algorithms import QuantumPhaseEstimation
        >>> # Estimate phase for T gate (phase = 1/8)
        >>> qpe = QuantumPhaseEstimation(precision_qubits=4, eigenstate_qubits=1)
        >>> circuit = qpe.build_circuit(unitary="T")
        >>> result = qpe.run(shots=1024)
        >>> print(result.get()["counts"])
    """

    def __init__(self, precision_qubits: int, eigenstate_qubits: int):
        """
        Initialize Quantum Phase Estimation.

        Args:
            precision_qubits (int): Number of qubits for phase precision
            eigenstate_qubits (int): Number of qubits in the eigenstate
        """
        total_qubits = precision_qubits + eigenstate_qubits
        super().__init__(num_qubits=total_qubits, name="Quantum Phase Estimation")

        self.precision_qubits = precision_qubits
        self.eigenstate_qubits = eigenstate_qubits
        self.description = (
            f"Quantum Phase Estimation with {precision_qubits}-bit precision"
        )

    def build_circuit(
        self, unitary: str = "T", eigenstate_preparation: Callable | None = None
    ) -> Circuit:
        """
        Build the Quantum Phase Estimation circuit.

        Args:
            unitary (str): Name of the unitary operator ('T', 'S', 'Z', 'custom')
            eigenstate_preparation (Callable, optional): Function to prepare eigenstate

        Returns:
            Circuit: The QPE circuit
        """
        self.circuit = Circuit(self.num_qubits, self.precision_qubits)

        # Step 1: Initialize precision register to superposition
        for i in range(self.precision_qubits):
            self.circuit.h(i)

        # Step 2: Prepare eigenstate (default: |1⟩ for single-qubit gates)
        if eigenstate_preparation is not None:
            eigenstate_preparation(self.circuit, self.precision_qubits)
        else:
            # Default: prepare |1⟩ state (eigenstate of Z, S, T gates)
            self.circuit.x(self.precision_qubits)

        # Step 3: Apply controlled-U^(2^j) operations
        self._apply_controlled_unitaries(unitary)

        # Step 4: Apply inverse QFT to precision register
        QFT.apply_inverse_qft_to_circuit(self.circuit, 0, self.precision_qubits)

        # Step 5: Measure precision register
        for i in range(self.precision_qubits):
            self.circuit.measure(i, i)

        return self.circuit

    def _apply_controlled_unitaries(self, unitary: str):
        """
        Apply controlled unitary operations U^(2^j).

        Args:
            unitary (str): Name of the unitary operator
        """
        target_qubit = self.precision_qubits  # First eigenstate qubit

        for j in range(self.precision_qubits):
            control_qubit = j
            repetitions = 2 ** (self.precision_qubits - 1 - j)

            self._apply_controlled_unitary(
                control_qubit, target_qubit, unitary, repetitions
            )

    def _apply_controlled_unitary(
        self, control: int, target: int, unitary: str, repetitions: int = 1
    ):
        """
        Apply a controlled unitary operation exponentiated by repetitions.

        Args:
            control (int): Control qubit index
            target (int): Target qubit index
            unitary (str): Unitary operator name
            repetitions (int): The power to raise the unitary to
        """
        if unitary == "T":
            # T gate: phase = π/4
            self.circuit.cp(control, target, repetitions * math.pi / 4)  # type: ignore

        elif unitary == "S":
            # S gate: phase = π/2
            self.circuit.cp(control, target, repetitions * math.pi / 2)  # type: ignore

        elif unitary == "Z":
            # Z gate: phase = π. CZ^2 = I
            if repetitions % 2 != 0:
                self.circuit.cz(control, target)  # type: ignore

        elif unitary == "X":
            # X gate (pauli-X). CX^2 = I
            if repetitions % 2 != 0:
                self.circuit.cx(control, target)  # type: ignore

        elif unitary == "H":
            # Controlled-Hadamard. CH^2 = I
            if repetitions % 2 != 0:
                # Decompose into rotations
                self.circuit.cp(control, target, math.pi / 2)  # type: ignore
                self.circuit.cx(control, target)  # type: ignore
                self.circuit.cp(control, target, -math.pi / 2)  # type: ignore

        else:
            # For custom unitaries, repeat the base gate natively
            for _ in range(repetitions):
                pass

    def estimate_phase(
        self, unitary: str = "T", eigenstate_preparation: Callable | None = None
    ) -> float:
        """
        Estimate the phase θ by running the algorithm.

        Args:
            unitary (str): Unitary operator name
            eigenstate_preparation (Callable, optional): Eigenstate preparation

        Returns:
            float: Estimated phase θ (between 0 and 1)
        """
        self.build_circuit(
            unitary=unitary, eigenstate_preparation=eigenstate_preparation
        )
        result = self.run(shots=1)

        # Get measurement result
        counts = result.get()["counts"]
        measured_bitstring = list(counts.keys())[0]
        measured_value = int(measured_bitstring, 2)

        # Convert to phase
        phase = measured_value / (2**self.precision_qubits)

        return phase

    def get_theoretical_phase(self, unitary: str) -> float:
        """
        Get the theoretical phase for standard unitaries.

        Args:
            unitary (str): Unitary operator name

        Returns:
            float: Theoretical phase θ
        """
        phases = {
            "T": 1 / 8,  # e^(iπ/4) = e^(2πi * 1/8)
            "S": 1 / 4,  # e^(iπ/2) = e^(2πi * 1/4)
            "Z": 1 / 2,  # e^(iπ) = e^(2πi * 1/2)
        }

        return phases.get(unitary, 0.0)
