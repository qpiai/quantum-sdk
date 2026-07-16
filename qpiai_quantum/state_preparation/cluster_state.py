from typing import Any, Dict, List, Optional
import math
from ..algorithms.base import QuantumAlgorithm
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager.job_result import JobResult
from qpiai_quantum.results.base_result import BaseQuantumResult
from qpiai_quantum.jobmanager import Backend


class ClusterStateGenerator(QuantumAlgorithm):
    """
    Generator for Cluster states (also known as Graph states).

    A 1D (linear) cluster state of n qubits is prepared by:
    1. Preparing all qubits in the |+⟩ state.
    2. Applying Controlled-Z (CZ) gates between all adjacent qubits.

    The cluster state is a highly entangled state used in measurement-based
    quantum computation (MBQC).
    """

    def __init__(self, num_qubits: int = 3):
        """
        Initialize the Cluster State Generator.

        Args:
            num_qubits (int): Number of qubits for the cluster state.
                             Must be >= 2. Default is 3.

        Raises:
            ValueError: If num_qubits < 2
        """
        if num_qubits < 2:
            raise ValueError(
                f"Cluster state requires at least 2 qubits, got {num_qubits}."
            )

        super().__init__(
            num_qubits=num_qubits, name=f"Cluster State ({num_qubits} qubits)"
        )

        self.description = (
            f"1D Linear Cluster state with {num_qubits} qubits. "
            f"Prepared using H gates followed by nearest-neighbor CZ gates."
        )

    def build_circuit(self, measure: bool = True, **kwargs) -> Circuit:
        """
        Build the circuit for the cluster state.

        Args:
            measure (bool): If True, add measurement gates at the end.
            **kwargs: Additional arguments for circuit construction.

        Returns:
            Circuit: The constructed quantum circuit.
        """
        self.circuit = Circuit(self.num_qubits, self.num_qubits)

        # Step 1: Prepare all qubits in |+> state
        for i in range(self.num_qubits):
            self.circuit.h(i)

        # Step 2: Apply CZ gates between adjacent qubits
        for i in range(self.num_qubits - 1):
            self.circuit.cz(i, i + 1)

        if measure:
            for i in range(self.num_qubits):
                self.circuit.measure(i, i)

        return self.circuit

    def get_expected_outcomes(self) -> dict[str, float]:
        """
        Get the theoretical probability distribution for the cluster state
        in the computational (Z) basis.

        For a cluster state, all 2^n possible bitstrings are equally likely
        with probability 1/2^n.
        """
        num_states = 2**self.num_qubits
        prob = 1.0 / num_states

        expected = {}
        for i in range(num_states):
            state = bin(i)[2:].zfill(self.num_qubits)
            expected[state] = prob

        return expected

    def verify_entanglement(
        self, result: BaseQuantumResult, threshold: float = 0.5
    ) -> bool:
        """
        Verify that the measurement results are consistent with a cluster state.

        In the Z-basis, we expect a uniform distribution. A simple check
        is to ensure we see a variety of states and no single state dominates.

        Note: True verification of cluster state entanglement requires
        measurements in multiple bases (stabilizer measurements).
        """
        counts = result.get()["counts"]
        total_shots = sum(counts.values())

        # Check if the distribution is roughly uniform
        num_observed_states = len(counts)
        num_expected_states = 2**self.num_qubits

        # If we see significantly fewer states than expected, it's likely not a cluster state
        # (Threshold depends on shots, but for a small number of qubits and reasonable shots,
        # we should see most states).
        coverage = num_observed_states / num_expected_states

        if coverage < threshold:
            return False

        # Check that no single state is way too frequent
        max_prob = max(counts.values()) / total_shots
        if max_prob > (2.0 / num_expected_states) + 0.1:  # Allow some variance
            return False

        return True

    def get_circuit_depth(self) -> int:
        """
        Get the depth of the cluster state circuit.
        """
        # H layer + CZ layers (CZs can be partially parallelized but linear is O(n))
        # 1 (H) + (num_qubits - 1) (CZ)
        return self.num_qubits

    @staticmethod
    def compare_different_sizes(
        qubit_range: list[int] = [2, 3, 4],
        shots: int = 1024,
        backend: Backend | None = None,
    ) -> dict[int, BaseQuantumResult]:
        """
        Compare cluster states of different sizes.
        """
        results = {}

        if backend is None:
            backend = Backend.STATEVECTOR_SIMULATOR_CPU

        for n in qubit_range:
            cluster = ClusterStateGenerator(num_qubits=n)
            result = cluster.run(shots=shots, backend=backend)
            results[n] = result

            print(
                f"Cluster State ({n} qubits) - Observed states: {len(result.get()['counts'])}"
            )

        return results


def create_cluster_state(num_qubits: int = 3) -> ClusterStateGenerator:
    """
    Convenience function to create a cluster state generator.
    """
    return ClusterStateGenerator(num_qubits=num_qubits)


def get_cluster_state_circuit(num_qubits: int = 3, measure: bool = True) -> Circuit:
    """
    Get a cluster state circuit directly.
    """
    cluster = ClusterStateGenerator(num_qubits=num_qubits)
    return cluster.build_circuit(measure=measure)
