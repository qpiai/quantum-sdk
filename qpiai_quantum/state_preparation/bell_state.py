from typing import Any, Dict, Optional
from ..algorithms.base import QuantumAlgorithm
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager.job_result import JobResult
from qpiai_quantum.results.base_result import BaseQuantumResult
from qpiai_quantum.jobmanager import Backend


class BellStateGenerator(QuantumAlgorithm):
    """
    Generator for Bell states (maximally entangled 2-qubit states).

    The Four Bell States:
        |Ψ+⟩ = (|00⟩ + |11⟩)/√2  - Both qubits always same outcome
        |Ψ-⟩ = (|00⟩ - |11⟩)/√2  - Same outcome with phase difference
        |Φ+⟩ = (|01⟩ + |10⟩)/√2  - Qubits always opposite outcomes
        |Φ-⟩ = (|01⟩ - |10⟩)/√2  - Opposite outcomes with phase difference

    """

    VALID_STATES = ["|Ψ+>", "|Ψ->", "|Φ+>", "|Φ->"]

    def __init__(self, state_type: str = "|Ψ+>"):
        if state_type not in self.VALID_STATES:
            raise ValueError(
                f"Invalid state_type '{state_type}'. Choose from: {self.VALID_STATES}"
            )

        super().__init__(num_qubits=2, name=f"Bell State {state_type}")

        self.state_type = state_type
        self.description = self._get_state_description()

    def _get_state_description(self) -> str:
        """Get description for the current Bell state."""
        descriptions = {
            "|Ψ+>": "Bell state |Ψ+⟩ = (|00⟩ + |11⟩)/√2 - Both qubits always same",
            "|Ψ->": "Bell state |Ψ-⟩ = (|00⟩ - |11⟩)/√2 - Same with phase flip",
            "|Φ+>": "Bell state |Φ+⟩ = (|01⟩ + |10⟩)/√2 - Qubits always opposite",
            "|Φ->": "Bell state |Φ-⟩ = (|01⟩ - |10⟩)/√2 - Opposite with phase flip",
        }
        return descriptions[self.state_type]

    def build_circuit(self, measure: bool = True, **kwargs) -> Circuit:
        num_classical = 2 if measure else 0
        self.circuit = Circuit(self.num_qubits, num_classical)

        if self.state_type == "|Ψ+>":
            self._build_psi_plus()
        elif self.state_type == "|Ψ->":
            self._build_psi_minus()
        elif self.state_type == "|Φ+>":
            self._build_phi_plus()
        elif self.state_type == "|Φ->":
            self._build_phi_minus()

        if measure:
            self.circuit.measure(0, 0)
            self.circuit.measure(1, 1)

        return self.circuit

    def _build_psi_plus(self):
        self.circuit.h(0)  # type: ignore
        self.circuit.cx(0, 1)  # type: ignore

    def _build_psi_minus(self):
        self.circuit.x(0)  # type: ignore
        self.circuit.h(0)  # type: ignore
        self.circuit.cx(0, 1)  # type: ignore

    def _build_phi_plus(self):
        self.circuit.x(1)  # type: ignore
        self.circuit.h(0)  # type: ignore
        self.circuit.cx(0, 1)  # type: ignore

    def _build_phi_minus(self):
        self.circuit.x(1)  # type: ignore
        self.circuit.x(0)  # type: ignore
        self.circuit.h(0)  # type: ignore
        self.circuit.cx(0, 1)  # type: ignore

    def get_expected_outcomes(self) -> dict[str, float]:
        outcomes = {
            "|Ψ+>": {"00": 0.5, "11": 0.5},
            "|Ψ->": {"00": 0.5, "11": 0.5},
            "|Φ+>": {"01": 0.5, "10": 0.5},
            "|Φ->": {"01": 0.5, "10": 0.5},
        }
        return outcomes[self.state_type]

    def verify_entanglement(
        self, result: BaseQuantumResult, threshold: float = 0.4
    ) -> bool:
        counts = result.get_counts() or {}
        total_shots = sum(counts.values())

        expected = self.get_expected_outcomes()

        for state, expected_prob in expected.items():
            measured_count = counts.get(state, 0)
            measured_prob = measured_count / total_shots

            if measured_prob < threshold:
                return False

        for state, count in counts.items():
            if state not in expected:
                unexpected_prob = count / total_shots
                if unexpected_prob > (1 - 2 * threshold):
                    return False

        return True

    @staticmethod
    def get_all_bell_states() -> dict[str, "BellStateGenerator"]:
        return {
            state: BellStateGenerator(state)
            for state in BellStateGenerator.VALID_STATES
        }

    @staticmethod
    def compare_all_bell_states(shots: int = 1024, backend: Backend | None = None):
        from qpiai_quantum.jobmanager import Backend

        if backend is None:
            backend = Backend.STATEVECTOR_SIMULATOR_CPU

        results = {}

        for state_name in BellStateGenerator.VALID_STATES:
            bell = BellStateGenerator(state_name)
            result = bell.run(shots=shots, backend=backend)
            results[state_name] = result

            print(f"\n{state_name}:")
            print(f"  Counts: {result.get()['counts']}")
            print(f"  Entangled: {bell.verify_entanglement(result)}")

        return results


def create_bell_state(state_type: str = "|Ψ+>") -> BellStateGenerator:
    return BellStateGenerator(state_type)


def get_bell_state_circuit(state_type: str = "|Ψ+>", measure: bool = True) -> Circuit:
    bell = BellStateGenerator(state_type)
    return bell.build_circuit(measure=measure)


def get_all_bell_state_circuits(measure: bool = True) -> dict[str, Circuit]:
    circuits = {}
    for state_name in BellStateGenerator.VALID_STATES:
        bell = BellStateGenerator(state_name)
        circuits[state_name] = bell.build_circuit(measure=measure)
    return circuits
