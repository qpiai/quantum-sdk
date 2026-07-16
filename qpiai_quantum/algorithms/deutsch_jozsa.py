"""
Deutsch-Jozsa Algorithm Implementation

The Deutsch-Jozsa algorithm determines whether a given boolean function
f:{0,1}^n -> {0,1} is constant or balanced using a single quantum query.

A constant function returns the same value for all inputs.
A balanced function returns 0 for exactly half of inputs and 1 for the other half.
"""

from typing import Optional, Dict, Any
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager.job_result import JobResult
from .base import QuantumAlgorithm


class DeutschJozsa(QuantumAlgorithm):
    """
    Deutsch-Jozsa Algorithm.

    Determines whether a boolean function f:{0,1}^n -> {0,1} is constant or
    balanced in a single quantum query, achieving exponential speedup over
    deterministic classical algorithms.

    Args:
        num_qubits (int): Number of input qubits (n). The circuit uses
            num_qubits + 1 physical qubits internally (n input + 1 ancilla).
        oracle_type (str, optional): Type of oracle to build. One of:
            - ``"constant_zero"``: f(x) = 0 for all x (no gates on ancilla)
            - ``"constant_one"``:  f(x) = 1 for all x (X gate on ancilla)
            - ``"balanced"``:      f(x) = x_0 ⊕ x_1 ⊕ ... ⊕ x_{n-1}
              (CNOT from each input qubit to the ancilla)
            If None, must be provided when calling build_circuit().

    Example:
        >>> from qpiai_quantum.algorithms import DeutschJozsa
        >>> dj = DeutschJozsa(num_qubits=3, oracle_type="balanced")
        >>> circuit = dj.build_circuit()
        >>> result = dj.run(shots=1024)
        >>> print(DeutschJozsa.interpret_result(result))
    """

    VALID_ORACLE_TYPES = ("constant_zero", "constant_one", "balanced")

    def __init__(
        self,
        num_qubits: int,
        oracle_type: str | None = None,
    ):
        if num_qubits < 1:
            raise ValueError("Deutsch-Jozsa requires at least 1 input qubit.")

        super().__init__(num_qubits=num_qubits, name="Deutsch-Jozsa")
        self.oracle_type = oracle_type
        self.description = (
            "Deutsch-Jozsa Algorithm - determines if a boolean function "
            "is constant or balanced in a single query"
        )

    def build_circuit(
        self,
        oracle_type: str | None = None,
    ) -> Circuit:
        """
        Build the Deutsch-Jozsa circuit.

        The circuit consists of five stages:
          1. Ancilla preparation: X then H on the ancilla qubit → |−⟩
          2. Input superposition: H on all input qubits → uniform superposition
          3. Oracle: applies Uf depending on oracle_type
          4. Decoding: H on all input qubits
          5. Measurement: Measure the input register

        Args:
            oracle_type (str, optional): Oracle type to use. Overrides the
                value set in __init__ if provided.

        Returns:
            Circuit: The constructed Deutsch-Jozsa circuit

        Raises:
            ValueError: If no oracle type is provided or if it is invalid.
        """
        if oracle_type is not None:
            self.oracle_type = oracle_type

        # Validate oracle type
        if self.oracle_type is None:
            raise ValueError(
                "oracle_type must be specified either in __init__ "
                "or in build_circuit(). "
                f"Valid options: {self.VALID_ORACLE_TYPES}"
            )

        if self.oracle_type not in self.VALID_ORACLE_TYPES:
            raise ValueError(
                f"Invalid oracle_type '{self.oracle_type}'. "
                f"Valid options: {self.VALID_ORACLE_TYPES}"
            )

        # n input qubits + 1 ancilla qubit, n classical bits (only input measured)
        total_qubits = self.num_qubits + 1
        self.circuit = Circuit(total_qubits, self.num_qubits)

        ancilla = self.num_qubits  # Last qubit is the ancilla

        # ── Stage 1: Prepare ancilla in |−⟩ = H|1⟩ ──
        # The ancilla must be in |−⟩ for phase kickback to work.
        self.circuit.x(ancilla)
        self.circuit.h(ancilla)

        # ── Stage 2: Create uniform superposition on input register ──
        for i in range(self.num_qubits):
            self.circuit.h(i)

        # ── Stage 3: Apply oracle ──
        self._apply_oracle()

        # ── Stage 4: Decode by applying Hadamard to input register ──
        for i in range(self.num_qubits):
            self.circuit.h(i)

        # ── Stage 5: Measure input register ──
        for i in range(self.num_qubits):
            self.circuit.measure(i, i)

        return self.circuit

    def _apply_oracle(self):
        """
        Apply the oracle Uf based on the selected oracle_type.

        - ``constant_zero``:  f(x) = 0 → identity (no gates).
        - ``constant_one``:   f(x) = 1 → X on ancilla (flip it regardless of input).
        - ``balanced``:       f(x) = x_0 ⊕ x_1 ⊕ … ⊕ x_{n-1}
          → CNOT from every input qubit to the ancilla.
        """
        ancilla = self.num_qubits

        if self.oracle_type == "constant_zero":
            # f(x) = 0 for all x → do nothing (identity oracle)
            pass

        elif self.oracle_type == "constant_one":
            # f(x) = 1 for all x → flip the ancilla unconditionally
            self.circuit.x(ancilla)

        elif self.oracle_type == "balanced":
            # f(x) = x_0 ⊕ x_1 ⊕ … ⊕ x_{n-1}
            # CNOT from each input qubit to the ancilla
            for i in range(self.num_qubits):
                self.circuit.cx(i, ancilla)

    def determine_function_type(self, shots: int = 1024) -> str:
        """
        Run the algorithm and determine if the function is constant or balanced.

        Since the Deutsch-Jozsa algorithm produces a deterministic outcome
        (all-zeros for constant, non-zero for balanced) in the ideal case,
        this method runs the circuit and interprets the result.

        Args:
            shots (int): Number of measurement shots. Default: 1024.

        Returns:
            str: ``"constant"`` or ``"balanced"``

        Raises:
            ValueError: If the circuit has not been built (no oracle_type set).
        """
        if self.circuit is None:
            self.build_circuit()

        result = self.run(shots=shots, experiment_name="Default Experiment")
        return self.interpret_result(result)

    @staticmethod
    def interpret_result(result) -> str:
        """
        Interpret the result of the Deutsch-Jozsa algorithm.

        If the measured bitstring is all zeros => constant
        Else => balanced
        """
        if not result.counts:
            raise ValueError("No measurement results found.")

        # Take the most frequent outcome
        most_likely = max(result.counts, key=result.counts.get)

        # If all zeros -> constant
        if set(most_likely) == {"0"}:
            return "constant"
        else:
            return "balanced"

    def get_theoretical_result(self) -> dict[str, Any]:
        """
        Get the theoretically expected measurement outcome.

        Returns:
            dict: Contains 'oracle_type', 'expected_result',
                  'expected_counts' (for 1024 shots), and
                  'success_probability'.

        Raises:
            ValueError: If no oracle_type has been set.
        """
        if self.oracle_type is None:
            raise ValueError(
                "oracle_type not set. Provide it in __init__ or build_circuit() "
                "before calling get_theoretical_result()."
            )

        if self.oracle_type in ("constant_zero", "constant_one"):
            expected_result = "constant"
            expected_bitstring = "0" * self.num_qubits
        else:
            expected_result = "balanced"
            expected_bitstring = "1" * self.num_qubits

        return {
            "oracle_type": self.oracle_type,
            "expected_result": expected_result,
            "expected_counts": {expected_bitstring: 1024},
            "success_probability": 1.0,
            "num_oracle_queries": 1,
            "classical_queries_needed": f"{2 ** (self.num_qubits - 1) + 1} (worst case)",
            "speedup": "exponential (1 query vs 2^(n-1)+1)",
        }
