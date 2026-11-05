from typing import Any, Optional
import math
from ..algorithms.base import QuantumAlgorithm
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager.job_result import JobResult
from qpiai_quantum.results.base_result import BaseQuantumResult


class WStateGenerator(QuantumAlgorithm):
    def __init__(self, num_qubits: int = 3):
        if num_qubits < 3:
            raise ValueError(
                f"W state requires at least 3 qubits, got {num_qubits}. For 2 qubits, use BellStateGenerator instead."
            )
        super().__init__(num_qubits=num_qubits, name=f"W State ({num_qubits} qubits)")
        self.description = f"W state with {num_qubits} qubits: symmetric superposition with exactly one excitation"

    def build_circuit(self, measure: bool = True, **kwargs) -> Circuit:
        num_classical = self.num_qubits if measure else 0
        self.circuit = Circuit(self.num_qubits, num_classical)
        self._build_w_state_recursive()
        if measure:
            for i in range(self.num_qubits):
                self.circuit.measure(i, i)
        return self.circuit

    def _build_w_state_recursive(self):
        n = self.num_qubits
        self.circuit.x(0)
        for k in range(n - 1):
            theta = math.acos(math.sqrt(1.0 / (n - k)))
            self.circuit.ry(k + 1, -theta)
            self.circuit.cz(k, k + 1)
            self.circuit.ry(k + 1, theta)
        for k in range(n - 1):
            self.circuit.cx(k + 1, k)

    def get_expected_outcomes(self) -> dict[str, float]:
        expected = {}
        prob = 1.0 / self.num_qubits
        for i in range(self.num_qubits):
            state = ["0"] * self.num_qubits
            state[i] = "1"
            state_str = "".join(state)
            expected[state_str] = prob
        return expected

    def verify_entanglement(
        self, result: BaseQuantumResult, threshold: float = 0.2
    ) -> bool:
        counts = result.get()["counts"]
        total_shots = sum(counts.values())
        expected = self.get_expected_outcomes()
        expected_prob = 1.0 / self.num_qubits
        min_expected = expected_prob * 0.5
        found_count = 0
        for state in expected.keys():
            if state in counts:
                measured_prob = counts[state] / total_shots
                if measured_prob >= min_expected:
                    found_count += 1
        if found_count < len(expected) * 0.5:
            return False
        for state in counts.keys():
            if state.count("1") != 1:
                measured_prob = counts[state] / total_shots
                if measured_prob > 0.1:
                    return False
        return True

    def count_excitations(self, result: BaseQuantumResult) -> dict[int, int]:
        counts = result.get()["counts"]
        excitation_counts: dict[int, int] = {}
        for state, count in counts.items():
            num_ones = state.count("1")
            excitation_counts[num_ones] = excitation_counts.get(num_ones, 0) + count
        return excitation_counts

    def calculate_robustness(self, particle_loss_count: int = 1) -> float:
        if particle_loss_count >= self.num_qubits:
            return 0.0
        remaining_qubits = self.num_qubits - particle_loss_count
        if remaining_qubits < 2:
            return 0.0
        prob_survive = ((self.num_qubits - 1) / self.num_qubits) ** particle_loss_count
        return prob_survive

    def compare_with_ghz_robustness(
        self, particle_loss_count: int = 1
    ) -> dict[str, float]:
        w_robustness = self.calculate_robustness(particle_loss_count)
        ghz_robustness = 0.0
        return {
            "w_robustness": w_robustness,
            "ghz_robustness": ghz_robustness,
            "advantage_ratio": float("inf")
            if ghz_robustness == 0
            else w_robustness / ghz_robustness,
            "num_qubits": self.num_qubits,
            "particles_lost": particle_loss_count,
        }


def create_w_state(num_qubits: int = 3) -> WStateGenerator:
    return WStateGenerator(num_qubits=num_qubits)


def get_w_circuit(num_qubits: int = 3, measure: bool = True) -> Circuit:
    w = WStateGenerator(num_qubits=num_qubits)
    return w.build_circuit(measure=measure)
