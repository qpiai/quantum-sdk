import time
import sys
import types
import unittest
from pathlib import Path

import numpy as np

# Avoid executing qpiai_quantum.__init__, which imports optional algorithm
# dependencies that are unrelated to this simulator equivalence test.
if "qpiai_quantum" not in sys.modules:
    qpiai_quantum = types.ModuleType("qpiai_quantum")
    qpiai_quantum.__path__ = [str(Path(__file__).resolve().parents[1] / "qpiai_quantum")]
    sys.modules["qpiai_quantum"] = qpiai_quantum

from qpiai_quantum.circuit.circuit import Circuit
from qpiai_quantum.icr.circuitoperation import OperationType
from qpiai_quantum.simulator.gates import DECOMPOSED_GATES, decompose, gate_spec
from qpiai_quantum.simulator.result import QasmSimulatorResult
from qpiai_quantum.simulator.statevector import StatevectorSimulator


OPTIMIZED_GATES = ("h", "x", "y", "z", "s", "rx", "ry", "rz")
PARAMETRIC_GATES = {"rx", "ry", "rz"}


class OriginalStatevectorSimulator(StatevectorSimulator):
    def run(
        self,
        circuit: Circuit,
        shots: int = 1024,
        seed: int | None = None,
        name: str | None = None,
    ) -> QasmSimulatorResult:
        start_time = time.perf_counter()
        n_qubits = circuit.num_qubits
        n_cbits = circuit.num_clbits

        if n_qubits == 0:
            raise ValueError("Cannot simulate a circuit with 0 qubits.")

        dim = 2**n_qubits
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        measure_map: dict[int, int] = {}

        def _apply_gate(gate_name: str, params: list[float], qubits: list[int]) -> None:
            nonlocal state
            gate_name_lower = gate_name.lower()

            if gate_name_lower in DECOMPOSED_GATES:
                for sub_name, sub_params, sub_qubits in decompose(
                    gate_name_lower, qubits
                ):
                    _apply_gate(sub_name, sub_params, sub_qubits)
                return

            _, U = gate_spec(gate_name_lower, params, num_qubits=len(qubits))
            state = self._apply_unitary(state, n_qubits, qubits, U)

        def _apply_op(op) -> None:
            nonlocal state
            if op.operation_type == OperationType.BARRIER:
                return

            if op.operation_type == OperationType.MEASURE:
                if op.qubits and op.clbits:
                    for q, c in zip(op.qubits, op.clbits):
                        measure_map[q] = c
                return

            if op.operation_type in (
                OperationType.N_QUBIT_NON_PARAMETRIC,
                OperationType.N_QUBIT_PARAMETRIC,
                OperationType.SWAP,
            ):
                _apply_gate(op.gate_name, op.params or [], op.qubits)
                return

            if op.operation_type == OperationType.OPERATION:
                if hasattr(op, "order") and op.order is not None:
                    for sub_op in op.order:
                        _apply_op(sub_op)
                else:
                    _apply_gate(op.gate_name, op.params or [], op.qubits)
                return

            if op.gate_name.lower() == "reset" and op.qubits:
                for q in op.qubits:
                    state = self._apply_reset(state, n_qubits, q)
            else:
                raise ValueError(f"Unsupported operation type: {op.operation_type}")

        for op in circuit.icr.evolve:
            _apply_op(op)

        counts = (
            self._sample_counts(state, n_qubits, n_cbits, measure_map, shots, seed)
            if n_cbits > 0 and measure_map
            else {}
        )

        return QasmSimulatorResult(
            name=name or circuit.name,
            counts=counts,
            statevector=state.tolist(),
            shots=shots,
            executionTime=time.perf_counter() - start_time,
            method="statevector-original",
            job_status="completed",
            n_qubits=n_qubits,
            n_cbits=n_cbits,
        )


def apply_spec(circuit: Circuit, spec: tuple[str, tuple[int, ...], tuple[float, ...]]):
    gate, qubits, params = spec
    if gate in PARAMETRIC_GATES:
        getattr(circuit, gate)(qubits[0], params[0])
    elif gate == "cx":
        circuit.cx(qubits[0], qubits[1])
    else:
        getattr(circuit, gate)(qubits[0])


def build_circuit(n_qubits: int, specs) -> Circuit:
    circuit = Circuit(n_qubits)
    for spec in specs:
        apply_spec(circuit, spec)
    return circuit


def statevector(simulator, circuit: Circuit) -> np.ndarray:
    return np.asarray(simulator.run(circuit, shots=1).statevector, dtype=complex)


def describe_specs(specs) -> str:
    lines = []
    for index, (gate, qubits, params) in enumerate(specs, start=1):
        params_text = f", params={list(params)}" if params else ""
        lines.append(f"{index}. {gate.upper()} qubits={list(qubits)}{params_text}")
    return "\n".join(lines)


def first_divergence(n_qubits: int, specs) -> str:
    original = OriginalStatevectorSimulator()
    optimized = StatevectorSimulator()
    for index in range(1, len(specs) + 1):
        prefix = specs[:index]
        original_state = statevector(original, build_circuit(n_qubits, prefix))
        optimized_state = statevector(optimized, build_circuit(n_qubits, prefix))
        if not np.allclose(original_state, optimized_state):
            gate, qubits, params = specs[index - 1]
            return (
                f"first divergence at gate {index}: "
                f"{gate.upper()} qubits={list(qubits)} params={list(params)}"
            )
    return "no prefix divergence found"


def assert_equivalent(n_qubits: int, specs, expected: np.ndarray | None = None):
    circuit = build_circuit(n_qubits, specs)
    original_state = statevector(OriginalStatevectorSimulator(), circuit)
    optimized_state = statevector(StatevectorSimulator(), circuit)

    if expected is not None:
        assert np.allclose(original_state, expected), (
            "Original simulator did not match expected state.\n"
            f"Circuit:\n{describe_specs(specs)}\n"
            f"Original statevector:\n{original_state}\n"
            f"Expected statevector:\n{expected}"
        )
        assert np.allclose(optimized_state, expected), (
            "Optimized simulator did not match expected state.\n"
            f"Circuit:\n{describe_specs(specs)}\n"
            f"Optimized statevector:\n{optimized_state}\n"
            f"Expected statevector:\n{expected}"
        )

    assert np.allclose(original_state, optimized_state), (
        "Original and optimized statevectors differ.\n"
        f"Circuit:\n{describe_specs(specs)}\n"
        f"Original statevector:\n{original_state}\n"
        f"Optimized statevector:\n{optimized_state}\n"
        f"{first_divergence(n_qubits, specs)}"
    )


class TestStatevectorSingleQubitOptimization(unittest.TestCase):
    def test_individual_non_parametric_gates(self):
        for gate in ("h", "x", "y", "z", "s"):
            for n_qubits in (1, 2, 5, 10):
                with self.subTest(gate=gate, n_qubits=n_qubits):
                    target = min(n_qubits - 1, max(0, n_qubits // 2))
                    assert_equivalent(n_qubits, [(gate, (target,), ())])

    def test_individual_rotation_gates_multiple_angles(self):
        angles = np.random.default_rng(20260710).uniform(
            -4 * np.pi, 4 * np.pi, size=12
        )
        for gate in ("rx", "ry", "rz"):
            for n_qubits in (1, 2, 5, 10):
                for theta in angles:
                    with self.subTest(gate=gate, n_qubits=n_qubits, theta=theta):
                        target = min(n_qubits - 1, max(0, n_qubits // 2))
                        assert_equivalent(n_qubits, [(gate, (target,), (float(theta),))])

    def test_multiple_sequential_single_qubit_gates(self):
        specs = [
            ("h", (0,), ()),
            ("rx", (0,), (0.125,)),
            ("y", (0,), ()),
            ("ry", (0,), (-1.75,)),
            ("s", (0,), ()),
            ("z", (0,), ()),
            ("rz", (0,), (2.25,)),
            ("x", (0,), ()),
        ]
        assert_equivalent(1, specs)

    def test_single_qubit_gates_on_different_targets(self):
        for n_qubits in (2, 5, 10):
            with self.subTest(n_qubits=n_qubits):
                specs = [
                    ("h", (0,), ()),
                    ("x", (n_qubits - 1,), ()),
                    ("rx", (1,), (0.37,)),
                    ("ry", (n_qubits // 2,), (-0.91,)),
                    ("rz", (n_qubits - 1,), (1.23,)),
                    ("s", (0,), ()),
                    ("y", (n_qubits // 2,), ()),
                    ("z", (1,), ()),
                ]
                assert_equivalent(n_qubits, specs)

    def test_random_circuits_containing_only_optimized_gates(self):
        for n_qubits, depth in ((1, 40), (2, 60), (5, 100), (10, 120)):
            with self.subTest(n_qubits=n_qubits, depth=depth):
                rng = np.random.default_rng(12345 + n_qubits)
                specs = []
                for _ in range(depth):
                    gate = str(rng.choice(OPTIMIZED_GATES))
                    target = int(rng.integers(0, n_qubits))
                    params = (
                        (float(rng.uniform(-2 * np.pi, 2 * np.pi)),)
                        if gate in PARAMETRIC_GATES
                        else ()
                    )
                    specs.append((gate, (target,), params))
                assert_equivalent(n_qubits, specs)

    def test_bell_state_matches_original_and_expected_state(self):
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[3] = 1 / np.sqrt(2)
        assert_equivalent(2, [("h", (0,), ()), ("cx", (0, 1), ())], expected)

    def test_bell_state_variation_matches_original(self):
        specs = [("x", (1,), ()), ("h", (0,), ()), ("cx", (0, 1), ())]
        assert_equivalent(2, specs)


if __name__ == "__main__":
    unittest.main()
