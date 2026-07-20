"""
Bernstein-Vazirani Algorithm Implementation

The Bernstein-Vazirani algorithm determines a hidden bitstring s ∈ {0,1}^n
in a single query to the oracle f(x) = s·x mod 2 (bitwise inner product).

"""

from typing import Optional, Any
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager.job_result import JobResult
from .base import QuantumAlgorithm


class BernsteinVazirani(QuantumAlgorithm):
    """
    Bernstein-Vazirani Algorithm.

    Finds a hidden bitstring s given oracle access to f(x) = s·x mod 2.
    Bernstein-Vazirani finds the hidden string in a single circuit execution with deterministic output.

    Args:
        num_qubits (int): Size of the hidden bitstring (number of input qubits).
            The circuit will use num_qubits + 1 physical qubits internally
            (n input + 1 ancilla).
        hidden_string (str, optional): The hidden bitstring s to encode in the
            oracle. Must be a binary string of length num_qubits. If None,
            must be provided when calling build_circuit().

    Example:
        >>> from qpiai_quantum.algorithms import BernsteinVazirani
        >>> bv = BernsteinVazirani(num_qubits=4, hidden_string="1011")
        >>> circuit = bv.build_circuit()
        >>> result = bv.run(shots=1024)
        >>> print(result.get()["counts"])
    """

    def __init__(self, num_qubits: int, hidden_string: str | None = None):
        super().__init__(num_qubits=num_qubits, name="Bernstein-Vazirani")
        self.hidden_string = hidden_string
        self.description = (
            "Bernstein-Vazirani Algorithm - determine hidden bitstring "
            "in a single query via phase kickback"
        )

    def build_circuit(self, hidden_string: str | None = None) -> Circuit:
        """
        Build the Bernstein-Vazirani circuit.

        The circuit consists of five stages:
          1. Ancilla preparation: X then H on the ancilla qubit → |−⟩
          2. Input superposition: H on all input qubits → uniform superposition
          3. Oracle: CX(i, ancilla) for each i where s[i] = '1' → phase kickback
          4. Decoding: H on all input qubits → collapses to |s⟩
          5. Measurement: Measure the input register

        Args:
            hidden_string (str, optional): Hidden bitstring to encode. Overrides
                the value set in __init__ if provided.

        Returns:
            Circuit: The constructed BV circuit

        Raises:
            ValueError: If no hidden string is provided or if its length
                does not match num_qubits.
        """
        if hidden_string is not None:
            self.hidden_string = hidden_string

        # Validate hidden string
        if self.hidden_string is None:
            raise ValueError(
                "Hidden bitstring must be specified either in __init__ "
                "or in build_circuit()."
            )

        if len(self.hidden_string) != self.num_qubits:
            raise ValueError(
                f"Hidden string length ({len(self.hidden_string)}) must match "
                f"num_qubits ({self.num_qubits})"
            )

        if not all(bit in ("0", "1") for bit in self.hidden_string):
            raise ValueError(
                f"Hidden string must contain only '0' and '1' characters, "
                f"got: '{self.hidden_string}'"
            )

        # n input qubits + 1 ancilla qubit, n classical bits (only input measured)
        total_qubits = self.num_qubits + 1
        self.circuit = Circuit(total_qubits, self.num_qubits)

        ancilla = self.num_qubits  # Last qubit is the ancilla

        # ── Stage 1: Prepare ancilla in |−⟩ = H|1⟩ ──
        # The ancilla must be in |−⟩ for phase kickback to work.
        # CX with target |−⟩ flips the phase of the control qubit:
        #   |x⟩|−⟩ → (−1)^x |x⟩|−⟩
        self.circuit.x(ancilla)
        self.circuit.h(ancilla)

        # ── Stage 2: Create uniform superposition on input register ──
        # |0⟩^n  →  (1/√2^n) Σ_x |x⟩
        for i in range(self.num_qubits):
            self.circuit.h(i)

        # ── Stage 3: Apply the inner-product oracle ──
        # After this stage: (1/√2^n) Σ_x (−1)^(s·x) |x⟩|−⟩
        self._apply_oracle()

        # ── Stage 4: Decode by applying Hadamard to input register ──
        # H^⊗n applied to (−1)^(s·x) |x⟩ collapses to |s⟩
        # This is because H^⊗n (−1)^(s·x) H^⊗n = |s⟩⟨s|
        for i in range(self.num_qubits):
            self.circuit.h(i)

        # ── Stage 5: Measure input register ──
        # The result is deterministically the hidden string s
        for i in range(self.num_qubits):
            self.circuit.measure(i, i)

        return self.circuit

    def _apply_oracle(self):
        """
        Apply the inner-product oracle via phase kickback.

        For each bit position i where s[i] = '1', apply a CNOT gate with
        input qubit i as control and the ancilla as target.

        The ancilla is in the |−⟩ state, so each CX gate produces:
            |x_i⟩|−⟩  →  (−1)^(x_i) |x_i⟩|−⟩

        The combined effect across all CX gates is:
            |x⟩|−⟩  →  (−1)^(s·x) |x⟩|−⟩

        where s·x = Σ_i s_i * x_i  (mod 2).

        Note: If the hidden string is "000...0" (all zeros), no CX gates are
        applied. The oracle is simply the identity, and the measurement will
        correctly return "000...0".
        """
        ancilla = self.num_qubits

        # Reverse iteration: hidden_string[0] → qubit (n-1) so that
        # the MSB-first measurement output matches the input string order.
        reversed_s = self.hidden_string[::-1]
        for i, bit in enumerate(reversed_s):
            if bit == "1":
                self.circuit.cx(i, ancilla)

    def find_hidden_string(self, shots: int = 1024) -> str:
        """
        Run the algorithm and extract the hidden bitstring from results.

        Since the BV algorithm produces a deterministic outcome (the hidden
        string s appears with probability 1 in the ideal case), this method
        runs the circuit and returns the most frequently measured bitstring.

        On a noisy backend, the most frequent measurement will still be
        the correct hidden string (as long as noise is below ~50%).

        Args:
            shots (int): Number of measurement shots. More shots improve
                confidence on noisy backends. Default: 1024.

        Returns:
            str: The recovered hidden bitstring.

        Raises:
            ValueError: If the circuit has not been built (no hidden string set).
        """
        if self.circuit is None:
            self.build_circuit()

        result = self.run(shots=shots, experiment_name="Default Experiment")
        counts = result.get()["counts"]

        # Return the most frequently measured bitstring
        # In the ideal case, this is the only outcome with all shots
        hidden_string = max(counts, key=counts.get)

        return hidden_string

    def get_theoretical_result(self) -> dict[str, Any]:
        """
        Get the theoretically expected measurement outcome.

        For the Bernstein-Vazirani algorithm, the output is deterministic:
        measuring the input register always yields the hidden string s
        with probability 1 (in the absence of noise).

        Returns:
            dict: Contains 'hidden_string', 'expected_counts' (for 1024 shots),
                  and 'success_probability' (always 1.0 for ideal execution).

        Raises:
            ValueError: If no hidden string has been set.
        """
        if self.hidden_string is None:
            raise ValueError(
                "Hidden string not set. Provide it in __init__ or build_circuit() "
                "before calling get_theoretical_result()."
            )

        return {
            "hidden_string": self.hidden_string,
            "expected_counts": {self.hidden_string: 1024},
            "success_probability": 1.0,
            "num_oracle_queries": 1,
            "classical_queries_needed": self.num_qubits,
            "speedup": f"{self.num_qubits}x (linear to constant)",
        }
