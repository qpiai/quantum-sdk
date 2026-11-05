"""Regression coverage for shared variational parameters and public dispatch."""

import numpy as np
import pytest
from qpiai_quantum.algorithms.opt import parameterize

from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.circuit.circuit import _map_device_name_to_method
from qpiai_quantum.icr.circuitoperation import CircuitOperation, OperationType


def test_shared_scaled_parameters_and_explicit_rotations() -> None:
    ansatz = Circuit(1)
    for scale in (1.0, -0.5):
        ansatz.add_operation(
            CircuitOperation(
                OperationType.N_QUBIT_PARAMETRIC,
                "rz",
                [0],
                [0.0],
                parameter_index=0,
                parameter_scale=scale,
            )
        )
    ansatz.ry(0, 0.0)
    ansatz.rx(0, 0.2)
    parameterize(ansatz, 1, operation_index=2)
    solver = VQESolver(n_qubits=1, ansatz=lambda _: ansatz)

    assert solver.num_parameters == 2
    assert solver.circuit is None
    bound = solver.build_circuit(parameters=np.array([0.8, 0.3]))
    angles = [op.params[0] for op in bound.icr.evolve if op.params]
    assert angles == pytest.approx([0.8, -0.4, 0.3, 0.2])
    original_operations = list(ansatz.icr.evolve)
    assert original_operations[0].params == [0.0]

    serialized = original_operations[1].to_json()
    assert serialized["parameter_index"] == 0
    assert serialized["parameter_scale"] == -0.5
    assert "parameter_index" not in original_operations[3].to_json()


def test_local_method_mapping_preserves_public_device_policy() -> None:
    assert _map_device_name_to_method("QpiAI-QSV-Local") == "statevector"
    assert _map_device_name_to_method("QpiAI-Indus-1") == "qpu"
    assert _map_device_name_to_method("QpiAI-QSV-Lite") == "statevector_sampling"
