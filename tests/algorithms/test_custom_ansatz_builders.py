"""Custom builder construction and binding below authenticated execution."""

import numpy as np
import pytest

from qpiai_quantum.algorithms.opt.solvers.qaoa import QAOASolver
from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.algorithms.opt import parameterize


def test_qaoa_builder_receives_width_and_layers_and_binds_angles() -> None:
    calls: list[tuple[int, int]] = []

    def builder(n_qubits: int, layers: int) -> Circuit:
        calls.append((n_qubits, layers))
        circuit = Circuit(n_qubits)
        for layer in range(layers):
            circuit.ry(0, 0.0)
            parameterize(circuit, layer)
        circuit.cx(0, 1)
        return circuit

    class Problem:
        n_qubits = 2

    solver = QAOASolver(problem=Problem(), layers=2, ansatz=builder, verbose=False)
    template = solver._get_ansatz()
    solver.num_qubits = 2
    bound = solver._build_circuit(template, np.array([0.2, -0.4]))
    rotations = [op for op in bound.icr.evolve if op.gate_name == "RY"]
    assert calls == [(2, 2)]
    assert bound.num_qubits == 2
    assert [op.params[0] for op in rotations] == pytest.approx([0.2, -0.4])
    assert any(op.gate_name == "CX" for op in bound.icr.evolve)
    assert [op.params[0] for op in template.icr.evolve if op.gate_name == "RY"] == [
        0.0,
        0.0,
    ]


def test_vqe_builder_receives_width_and_binds_angles() -> None:
    calls: list[int] = []

    def builder(n_qubits: int) -> Circuit:
        calls.append(n_qubits)
        circuit = Circuit(n_qubits)
        circuit.ry(0, 0.0)
        parameterize(circuit, 0)
        circuit.rz(1, 0.0)
        parameterize(circuit, 1)
        return circuit

    solver = VQESolver(n_qubits=2, ansatz=builder, verbose=False)
    template = solver._get_ansatz()
    bound = solver._build_circuit(template, np.array([0.2, -0.4]), measure=False)
    assert calls == [2]
    assert bound.num_qubits == 2
    assert [op.params[0] for op in bound.icr.evolve] == pytest.approx([0.2, -0.4])
    assert [op.params[0] for op in template.icr.evolve] == [0.0, 0.0]


def test_vqe_custom_builder_preserves_shared_angle_and_scale() -> None:
    def builder(n_qubits: int) -> Circuit:
        circuit = Circuit(n_qubits)
        circuit.ry(0, 0.0)
        circuit.rz(1, 0.0)
        for operation, scale in zip(circuit.icr.evolve, [1.0, 2.0]):
            operation.parameter_index = 0
            operation.parameter_scale = scale
        return circuit

    solver = VQESolver(n_qubits=2, ansatz=builder, verbose=False)
    assert solver.num_parameters == 1
    bound = solver._build_circuit(solver._get_ansatz(), np.array([0.3]), measure=False)
    assert [op.params[0] for op in bound.icr.evolve] == pytest.approx([0.3, 0.6])
