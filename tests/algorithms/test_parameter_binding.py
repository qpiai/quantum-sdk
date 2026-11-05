"""Independent mathematical checks of the common QAOA/VQE binding contract."""

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qpiai_quantum.algorithms.opt import parameterize
from qpiai_quantum.algorithms.opt.parameters import bind_parameters, count_parameters
from qpiai_quantum.algorithms.opt.ansatz import (
    hardware_efficient_ansatz,
    standard_vqe_ansatz,
    two_local_ansatz,
)
from qpiai_quantum.algorithms.opt.solvers import QAOASolver, VQESolver
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.icr.circuitoperation import CircuitOperation, OperationType, RYGate
from qpiai_quantum.icr.operationmodifiers import CustomControlOperation
from qpiai_quantum.simulator.statevector import StatevectorSimulator


def bound(mode: str, template: Circuit, values: Any) -> Circuit:
    if mode == "vqe":
        solver = VQESolver(
            n_qubits=template.num_qubits, ansatz=lambda _: template, verbose=False
        )
        return solver._build_circuit(solver._get_ansatz(), values, measure=False)
    problem = SimpleNamespace(n_qubits=template.num_qubits)
    qaoa = QAOASolver(problem=problem, ansatz=lambda n, p: template, verbose=False)
    qaoa.num_qubits = template.num_qubits
    return qaoa._build_circuit(qaoa._get_ansatz(), values)


def state(circuit: Circuit) -> np.ndarray:
    return np.asarray(
        StatevectorSimulator().run(circuit, shots=16, seed=7).statevector
    ).reshape(-1)


@pytest.mark.parametrize("mode", ["qaoa", "vqe"])
@pytest.mark.parametrize("theta", [-0.41, 0.0, 0.73])
def test_fixed_and_shared_affine_angles_match_independent_state(
    mode: str, theta: float
) -> None:
    template = Circuit(1)
    template.ry(0, 0.7)
    template.rz(0, 0.0)
    parameterize(template, 0)
    template.rx(0, 0.0)
    parameterize(template, 0, scale=-2.0, offset=0.1)
    before = copy.deepcopy([op.to_json() for op in template.icr.evolve])
    circuit = bound(mode, template, np.array([theta]))
    phi = -2 * theta + 0.1
    expected = np.array([np.cos(0.35), np.sin(0.35)], dtype=complex)
    expected = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)]) @ expected
    expected = (
        np.array(
            [
                [np.cos(phi / 2), -1j * np.sin(phi / 2)],
                [-1j * np.sin(phi / 2), np.cos(phi / 2)],
            ]
        )
        @ expected
    )
    np.testing.assert_allclose(state(circuit), expected, atol=1e-12)
    assert count_parameters(template) == 1
    assert before == [op.to_json() for op in template.icr.evolve]
    assert count_parameters(circuit) == 0
    # A second binding must still use the unchanged original template.
    second = bound(mode, template, [theta + 0.2])
    assert list(second.icr.evolve)[1].params == pytest.approx([theta + 0.2])


@pytest.mark.parametrize("mode", ["qaoa", "vqe"])
def test_multi_angle_gate_keeps_fixed_argument_and_shares_others(mode: str) -> None:
    template = Circuit(1)
    template.h(0)
    template.u(0, 0.0, 0.4, 0.0)
    parameterize(template, 0, argument_index=0)
    parameterize(template, 0, argument_index=2, scale=2, offset=-0.1)
    theta, phi, lam = 0.6, 0.4, 1.1
    unitary = np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2),
            ],
        ]
    )
    circuit = bound(mode, template, [theta])
    assert list(circuit.icr.evolve)[1].params == pytest.approx([theta, phi, lam])
    np.testing.assert_allclose(
        state(circuit), unitary @ (np.ones(2) / np.sqrt(2)), atol=1e-12
    )


@pytest.mark.parametrize("mode", ["qaoa", "vqe"])
def test_rzz_cached_decomposition_uses_bound_angle(mode: str) -> None:
    template = Circuit(2)
    template.h(0)
    template.h(1)
    template.rzz(0, 1, 0.0)
    parameterize(template, 0, scale=2.0)
    theta = 0.37
    expected = np.exp(-1j * theta * np.array([1, -1, -1, 1])) / 2
    np.testing.assert_allclose(
        state(bound(mode, template, [theta])), expected, atol=1e-12
    )
    assert list(template.icr.evolve)[-1].order[1].params == [0.0]


@pytest.mark.parametrize("mode", ["qaoa", "vqe"])
def test_controlled_base_gate_is_bound_without_losing_controls(mode: str) -> None:
    template = Circuit(2)
    template.h(0)
    template.add_operation(CustomControlOperation(0, operation=RYGate(1, 0.0)))
    parameterize(template, 0)
    theta = 0.81
    expected = np.array([1, np.cos(theta / 2), 0, np.sin(theta / 2)]) / np.sqrt(2)
    circuit = bound(mode, template, [theta])
    np.testing.assert_allclose(state(circuit), expected, atol=1e-12)
    assert list(circuit.icr.evolve)[1].control_qubits == [0]


@pytest.mark.parametrize("mode", ["qaoa", "vqe"])
def test_composite_body_annotations_are_counted_and_bound(mode: str) -> None:
    template = Circuit(1)
    body = CircuitOperation(OperationType.OPERATION, "body", [0])
    body.order = [RYGate(0, 0.7), RYGate(0, 0.0)]
    body.order[1].parameter_index = 0
    body.order[1].parameter_scale = 2
    template.add_operation(body)
    assert count_parameters(template) == 1
    expected = np.array([np.cos(0.65), np.sin(0.65)])
    np.testing.assert_allclose(
        state(bound(mode, template, [0.3])), expected, atol=1e-12
    )


@pytest.mark.parametrize("mode", ["qaoa", "vqe"])
def test_all_numeric_angles_stay_fixed_and_empty_vector_is_valid(mode: str) -> None:
    template = Circuit(1)
    template.ry(0, 0.7)
    assert count_parameters(template) == 0
    np.testing.assert_allclose(
        state(bound(mode, template, [])), [np.cos(0.35), np.sin(0.35)], atol=1e-12
    )
    with pytest.raises(ValueError, match="parameterize"):
        bound(mode, template, [0.2])


@pytest.mark.parametrize("mode", ["qaoa", "vqe"])
@pytest.mark.parametrize(
    "values", [[], [0.1, 0.2], [[0.1]], [float("nan")], [float("inf")], 0.2, ["bad"]]
)
def test_invalid_vectors_fail_before_execution(mode: str, values: Any) -> None:
    template = Circuit(1)
    template.ry(0, 0.0)
    parameterize(template, 0)
    with pytest.raises(ValueError):
        bound(mode, template, values)


@pytest.mark.parametrize(
    "options",
    [
        {"parameter_index": -1},
        {"parameter_index": True},
        {"parameter_index": 0.5},
        {"argument_index": -1},
        {"argument_index": 1},
        {"operation_index": 4},
        {"scale": float("nan")},
        {"offset": float("inf")},
    ],
)
def test_invalid_binding_declarations_are_rejected(options: dict[str, Any]) -> None:
    template = Circuit(1)
    template.ry(0, 0.0)
    arguments = {"parameter_index": 0, **options}
    with pytest.raises(ValueError):
        parameterize(template, **arguments)
    assert count_parameters(template) == 0


def test_duplicate_or_nonparametric_bindings_are_rejected() -> None:
    template = Circuit(1)
    with pytest.raises(ValueError):
        parameterize(template, 0)
    template.h(0)
    with pytest.raises(ValueError):
        parameterize(template, 0)
    template.ry(0, 0.0)
    parameterize(template, 0)
    with pytest.raises(ValueError, match="already"):
        parameterize(template, 1)


@pytest.mark.parametrize(
    "builder", [standard_vqe_ansatz, hardware_efficient_ansatz, two_local_ansatz]
)
def test_builtin_builders_declare_all_adjustable_angles(builder: Any) -> None:
    template = builder(2)
    rotations = [op for op in template.icr.evolve if op.params]
    assert count_parameters(template) == len(rotations) > 0
    values = np.linspace(0.1, 0.9, len(rotations))
    circuit = bind_parameters(template, values)
    assert [op.params[0] for op in circuit.icr.evolve if op.params] == pytest.approx(
        values
    )


@pytest.mark.parametrize("mode", ["qaoa", "vqe"])
def test_fixed_circuit_evaluates_once_without_optimizer(mode: str) -> None:
    solver: VQESolver | QAOASolver = (
        VQESolver(n_qubits=1, verbose=False)
        if mode == "vqe"
        else QAOASolver(verbose=False)
    )
    calls: list[np.ndarray] = []

    def objective(values: np.ndarray) -> float:
        calls.append(values.copy())
        return -0.7

    point, energy, info = solver._optimize_parameters(objective, np.array([]), 0)
    assert point.size == 0
    assert energy == -0.7
    assert len(calls) == 1
    assert info["nit"] == 0


@pytest.mark.parametrize("mode", ["qaoa", "vqe"])
def test_shared_scaled_logical_gradient_matches_analytic_derivative(mode: str) -> None:
    template = Circuit(1)
    template.ry(0, 0.7)
    template.ry(0, 0.0)
    parameterize(template, 0)
    template.ry(0, 0.0)
    parameterize(template, 0, scale=2.0)
    if mode == "vqe":
        solver: Any = VQESolver(n_qubits=1, gradient_step=1e-5, verbose=False)
    else:
        solver = QAOASolver(gradient_step=1e-5, verbose=False)

    def objective(values: np.ndarray) -> float:
        amplitudes = state(bind_parameters(template, values))
        return float(abs(amplitudes[0]) ** 2 - abs(amplitudes[1]) ** 2)

    theta = np.array([0.31])
    actual = solver._compute_gradient(theta, objective)
    np.testing.assert_allclose(actual, [-3 * np.sin(0.7 + 3 * theta[0])], atol=1e-8)


@pytest.mark.parametrize("step", [0, -1, float("inf"), float("nan"), True])
def test_vqe_rejects_invalid_gradient_step(step: float) -> None:
    with pytest.raises(ValueError, match="gradient_step"):
        VQESolver(n_qubits=1, gradient_step=step)


@pytest.mark.parametrize("mode", ["qaoa", "vqe"])
def test_controlled_four_angle_gate_keeps_all_arguments(mode: str) -> None:
    template = Circuit(2)
    template.h(0)
    template.h(1)
    template.cu(0, 1, 0.0, 0.0, 0.3, 0.0)
    parameterize(template, 0, argument_index=0)
    parameterize(template, 1, argument_index=1)
    parameterize(template, 0, argument_index=3, scale=-0.5)
    theta, phi, lam, gamma = 0.6, -0.2, 0.3, -0.3
    u = np.exp(1j * gamma) * np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2),
            ],
        ]
    )
    expected = np.ones(4, dtype=complex) / 2
    expected[[1, 3]] = u @ expected[[1, 3]]
    np.testing.assert_allclose(
        state(bound(mode, template, [theta, phi])), expected, atol=1e-12
    )


def test_matter_and_chemistry_builders_keep_their_logical_angles() -> None:
    from qpiai_quantum.applications.matter import ChainLattice, heisenberg_hva_ansatz
    from qpiai_quantum.applications.chemistry.uccsd import uccsd_ansatz

    hva = heisenberg_hva_ansatz(ChainLattice(n_sites=3), layers=1)
    assert count_parameters(hva) == 6
    chemistry = uccsd_ansatz(n_qubits=4, n_electrons=2)
    assert count_parameters(chemistry) == 3
    values = np.array([0.1, 0.2, -0.3])
    numeric = bind_parameters(chemistry, values)
    for original, actual in zip(chemistry.icr.evolve, numeric.icr.evolve):
        if original.parameter_index is not None:
            assert actual.params == pytest.approx(
                [original.parameter_scale * values[original.parameter_index]]
            )
        else:
            assert actual.to_json() == original.to_json()


def test_none_binding_returns_independent_template_and_preserves_metadata() -> None:
    template = Circuit(1, metadata={"example": {"value": 2}})
    template.ry(0, 0.7)
    template.rz(0, 0.0)
    parameterize(template, 0, operation_index=1)
    copied = bind_parameters(template, None)
    assert count_parameters(copied) == 1
    list(copied.icr.evolve)[0].params[0] = 0.1
    copied.icr.metadata["example"]["value"] = 3
    assert list(template.icr.evolve)[0].params == [0.7]
    assert template.icr.metadata["example"]["value"] == 2
