from typing import Any, Dict, List, Optional
from ..algorithms.base import QuantumAlgorithm
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager.job_result import JobResult
from qpiai_quantum.results.base_result import BaseQuantumResult
from qpiai_quantum.jobmanager import Backend


class GHZStateGenerator(QuantumAlgorithm):
    """
    The GHZ state is a maximally entangled quantum state of n qubits:

    |GHZ_n⟩ = (|0...0⟩ + |1...1⟩)/√2
    """

    def __init__(self, num_qubits: int = 3):
        """
        Initialize the GHZ State Generator.

        Args:
            num_qubits (int): Number of qubits for the GHZ state.
                             Must be >= 3. Default is 3.

        Raises:
            ValueError: If num_qubits < 3
        """
        if num_qubits < 3:
            raise ValueError(
                f"GHZ state requires at least 3 qubits, got {num_qubits}. "
                f"For 2 qubits, use BellStateGenerator instead."
            )

        super().__init__(num_qubits=num_qubits, name=f"GHZ State ({num_qubits} qubits)")

        self.description = (
            f"GHZ state with {num_qubits} qubits: "
            f"|GHZ_{num_qubits}⟩ = (|{'0' * num_qubits}⟩ + |{'1' * num_qubits}⟩)/√2"
        )

    def build_circuit(self, measure: bool = True, **kwargs) -> Circuit:

        num_classical = self.num_qubits if measure else 0
        self.circuit = Circuit(self.num_qubits, num_classical)

        self.circuit.h(0)

        for i in range(self.num_qubits - 1):
            self.circuit.cx(i, i + 1)

        if measure:
            for i in range(self.num_qubits):
                self.circuit.measure(i, i)

        return self.circuit

    def get_expected_outcomes(self) -> Dict[str, float]:
        """
        Get the theoretical probability distribution for the GHZ state.
        """
        all_zeros = "0" * self.num_qubits
        all_ones = "1" * self.num_qubits

        return {all_zeros: 0.5, all_ones: 0.5}

    def verify_entanglement(
        self, result: BaseQuantumResult, threshold: float = 0.4
    ) -> bool:
        """
        Verify that the measurement results show GHZ entanglement.
        """
        counts = result.get()["counts"]
        total_shots = sum(counts.values())

        expected = self.get_expected_outcomes()

        for state, expected_prob in expected.items():
            measured_count = counts.get(state, 0)
            measured_prob = measured_count / total_shots

            if measured_prob < threshold:
                return False

        expected_count = sum(counts.get(state, 0) for state in expected.keys())
        unexpected_prob = 1.0 - (expected_count / total_shots)

        if unexpected_prob > (1 - 2 * threshold):
            return False

        return True

    def calculate_entanglement_depth(self) -> int:
        """
        Calculate the entanglement depth of the GHZ state.
        """
        return self.num_qubits

    def get_circuit_depth(self) -> int:
        """
        Get the depth of the GHZ circuit.

        The depth is the number of time steps required to execute the circuit
        (assuming all gates that don't share qubits can be executed in parallel).
        """
        return self.num_qubits

    def compare_with_product_state(self, result: BaseQuantumResult) -> Dict[str, float]:
        """
        Compare GHZ state measurements with what a product state would give.

        A product state would show all 2^n possible outcomes with equal probability.
        GHZ state only shows 2 outcomes with equal probability.
        """
        counts = result.get()["counts"]
        total_shots = sum(counts.values())

        num_observed_states = len(counts)
        expected_ghz_states = 2
        expected_product_states = 2**self.num_qubits

        import math

        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total_shots
                entropy -= p * math.log2(p)

        max_entropy = math.log2(expected_product_states)

        return {
            "observed_states": num_observed_states,
            "expected_ghz_states": expected_ghz_states,
            "expected_product_states": expected_product_states,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "entropy_ratio": entropy / max_entropy,
            "is_ghz_like": num_observed_states <= 3,  # NOTE Allowing small noise
        }

    @staticmethod
    def create_multiple_sizes(
        min_qubits: int = 3, max_qubits: int = 6
    ) -> Dict[int, "GHZStateGenerator"]:
        """
        Create GHZ state generators for multiple qubit counts.

        Args:
            min_qubits (int): Minimum number of qubits. Default is 3.
            max_qubits (int): Maximum number of qubits. Default is 6.

        Returns:
            Dict[int, GHZStateGenerator]: Dictionary mapping qubit counts to generators
        """
        return {
            n: GHZStateGenerator(num_qubits=n)
            for n in range(min_qubits, max_qubits + 1)
        }

    @staticmethod
    def compare_different_sizes(
        qubit_range: List[int] = [3, 4, 5],
        shots: int = 1024,
        backend: Optional[Backend] = None,
    ) -> Dict[int, BaseQuantumResult]:
        """
        Compare GHZ states of different sizes.

        This is useful for educational purposes to see how GHZ states
        scale with the number of qubits.

        Args:
            qubit_range (List[int]): List of qubit counts to test
            shots (int): Number of shots for each state
            backend (Backend): Backend to use for execution

        Returns:
            Dict[int, BaseQuantumResult]: Results for each qubit count
        """
        results = {}

        from qpiai_quantum.jobmanager import Backend

        if backend is None:
            backend = Backend.STATEVECTOR_SIMULATOR_CPU

        for n in qubit_range:
            print(f"\n{'─' * 70}")
            print(f"GHZ State with {n} qubits")
            print(f"{'─' * 70}")

            ghz = GHZStateGenerator(num_qubits=n)
            result = ghz.run(shots=shots, backend=backend)
            results[n] = result

            counts = result.get()["counts"]
            print(f"Measurement counts: {counts}")

            is_entangled = ghz.verify_entanglement(result)
            print(f"Entanglement verified: {'✓' if is_entangled else '✗'}")

            expected = ghz.get_expected_outcomes()
            print(f"Expected outcomes: {list(expected.keys())}")

        return results


def create_ghz_state(num_qubits: int = 3) -> GHZStateGenerator:
    """
    Convenience function to create a GHZ state generator.

    Args:
        num_qubits (int): Number of qubits for the GHZ state

    Returns:
        GHZStateGenerator: The GHZ state generator instance
    """
    return GHZStateGenerator(num_qubits=num_qubits)


def get_ghz_circuit(num_qubits: int = 3, measure: bool = True) -> Circuit:
    """
    Get a GHZ state circuit directly.

    Args:
        num_qubits (int): Number of qubits
        measure (bool): Include measurements

    Returns:
        Circuit: The GHZ state circuit
    """
    ghz = GHZStateGenerator(num_qubits=num_qubits)
    return ghz.build_circuit(measure=measure)
