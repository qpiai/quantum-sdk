"""
Simon's Algorithm Implementation

Simon's algorithm finds hidden bitstring patterns with exponential speedup
over classical algorithms.
"""

from typing import Optional, List
from collections.abc import Callable
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager.job_result import JobResult
from .base import QuantumAlgorithm
import random


class SimonAlgorithm(QuantumAlgorithm):
    def __init__(
        self,
        num_qubits: int,
        hidden_string: str | None = None,
        oracle_function: Callable | None = None,
    ):
        """
        Initialize Simon's algorithm.

        Args:
            num_qubits (int): Number of qubits (input size)
            hidden_string (str, optional): Hidden bitstring s
            oracle_function (Callable, optional): Custom oracle function
        """
        super().__init__(num_qubits=num_qubits, name="Simon's Algorithm")
        self.hidden_string = hidden_string
        self.oracle_function = oracle_function
        self.description = (
            "Simon's Algorithm - find hidden bitstring with exponential speedup"
        )

    def build_circuit(self, hidden_string: str | None = None) -> Circuit:
        """
        Build Simon's algorithm circuit.

        Args:
            hidden_string (str, optional): Hidden bitstring for the oracle

        Returns:
            Circuit: The Simon's algorithm circuit
        """
        if hidden_string is not None:
            self.hidden_string = hidden_string

        if self.hidden_string is None:
            # NOTE: Generate random hidden string
            self.hidden_string = "".join(
                random.choice(["0", "1"]) for _ in range(self.num_qubits)
            )

        if len(self.hidden_string) != self.num_qubits:
            raise ValueError(
                f"Hidden string length ({len(self.hidden_string)}) must match "
                f"num_qubits ({self.num_qubits})"
            )

        # We need 2n qubits: n for input, n for output
        total_qubits = 2 * self.num_qubits
        self.circuit = Circuit(total_qubits, self.num_qubits)

        # Step 1: Apply Hadamard to input register
        for i in range(self.num_qubits):
            self.circuit.h(i)

        # Step 2: Apply oracle
        self._apply_oracle()

        # Step 3: Apply Hadamard to input register again
        for i in range(self.num_qubits):
            self.circuit.h(i)

        # Step 4: Measure input register
        for i in range(self.num_qubits):
            self.circuit.measure(i, i)

        return self.circuit

    def _apply_oracle(self):
        """
        Apply the oracle circuit for Simon's problem.

        The oracle implements f(x) where f(x) = f(x ⊕ s) for hidden string s.

        Construction:
          1. Copy the input register to the output register, EXCEPT for the
             first qubit position j where s[j] = '1'. By not copying qubit j,
             output[j] remains |0⟩ — trivially invariant under x → x ⊕ s.
          2. For every OTHER index k where s[k] = '1', apply CX(j, n+k).
             This sets output[k] = x[k] ⊕ x[j], which is also invariant:
             when both x[k] and x[j] flip (as s[k]=s[j]=1), the XOR stays.

        If s = 0^n, first_one is None, so only step 1 runs and f is
        one-to-one (every qubit is copied). This is correct.

        Note: The hidden string is reversed before mapping to qubits so that
        hidden_string[0] corresponds to qubit (n-1). This matches the SDK's
        big-endian measurement convention (qubit 0 → rightmost bit of the
        output string), consistent with BernsteinVazirani._apply_oracle().
        """
        n = self.num_qubits

        # Reverse the hidden string: hidden_string[0] → qubit (n-1)
        # so the MSB-first measurement output matches the string order.
        reversed_s = self.hidden_string[::-1]

        # Find the first qubit position where reversed_s has a '1'
        first_one = None
        for j in range(n):
            if reversed_s[j] == "1":
                first_one = j
                break

        # Step 1: Copy input register → output register, skipping first_one
        for i in range(n):
            if i != first_one:
                self.circuit.cx(i, n + i)

        # Step 2: For each other '1' bit in reversed_s, XOR that output with input[first_one]
        if first_one is not None:
            for k in range(n):
                if k != first_one and reversed_s[k] == "1":
                    self.circuit.cx(first_one, n + k)

    def find_hidden_string(self, max_attempts: int | None = None) -> str:
        """
        Find the hidden bitstring by running the algorithm multiple times.

        Simon's algorithm needs n-1 linearly independent measurements
        to solve for the hidden string. This method keeps running until
        enough independent equations are collected, or max_attempts is
        reached.

        Args:
            max_attempts (int, optional): Maximum number of algorithm runs
                (defaults to 10*n to ensure reliability)

        Returns:
            str: The recovered hidden bitstring
        """
        if self.circuit is None:
            self.build_circuit()

        n = self.num_qubits
        if max_attempts is None:
            max_attempts = 10 * n

        measurements: list[str] = []
        zero_str = "0" * n

        for _ in range(max_attempts):
            result = self.run(shots=1, experiment_name="Default Experiment")
            counts = result.get()["counts"]
            measured = list(counts.keys())[0]

            # Skip all-zero measurements (no information)
            if measured == zero_str:
                continue

            # Skip duplicate measurements (no new information)
            if measured in measurements:
                continue

            measurements.append(measured)

            # We need n-1 linearly independent equations
            if len(measurements) >= n - 1:
                break

        # Solve the system of linear equations (in GF(2))
        hidden_string = self._solve_for_hidden_string(measurements)

        return hidden_string

    def _solve_for_hidden_string(self, measurements: list[str]) -> str:
        """
        Solve for the hidden string given measurement results.

        Uses Gaussian elimination in GF(2) to find the null space of the
        matrix formed by the measurement bitstrings. Each measurement y
        satisfies s · y = 0 (mod 2), so s lies in the null space.

        Args:
            measurements (List[str]): List of measurement bitstrings

        Returns:
            str: The recovered hidden string
        """
        n = self.num_qubits

        if not measurements:
            return "0" * n

        # Build matrix from measurements (each row is a bitstring)
        matrix = []
        for m in measurements:
            matrix.append([int(b) for b in m])

        # Gaussian elimination in GF(2) — row echelon form
        num_rows = len(matrix)
        pivot_cols = []  # tracks which columns have pivots

        row = 0
        for col in range(n):
            # Find a row with a 1 in this column at or below current row
            found = None
            for r in range(row, num_rows):
                if matrix[r][col] == 1:
                    found = r
                    break

            if found is None:
                continue  # this column is a free variable

            # Swap found row into position
            matrix[row], matrix[found] = matrix[found], matrix[row]
            pivot_cols.append(col)

            # Eliminate all other rows in this column
            for r in range(num_rows):
                if r != row and matrix[r][col] == 1:
                    matrix[r] = [matrix[r][j] ^ matrix[row][j] for j in range(n)]

            row += 1

        # Find a free variable (column without a pivot) — s has a 1 there
        free_cols = [c for c in range(n) if c not in pivot_cols]

        if not free_cols:
            # All columns are pivots → s = 0^n (one-to-one function)
            return "0" * n

        # Build s by setting the first free variable to 1
        s = [0] * n
        free_col = free_cols[0]
        s[free_col] = 1

        # Back-substitute: for each pivot row, s[pivot_col] = matrix[row][free_col]
        for i, pc in enumerate(pivot_cols):
            s[pc] = matrix[i][free_col]

        return "".join(str(b) for b in s)
