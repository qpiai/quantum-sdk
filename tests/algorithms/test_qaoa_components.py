"""Independent matrix references for composable QAOA and differentiation."""

from types import SimpleNamespace
from typing import Any
from collections.abc import Mapping
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm

from qpiai_quantum.algorithms.opt import BasisState, CVaR, WarmStart, XMixer, XYMixer
from qpiai_quantum.algorithms.opt.qaoa_components import Aggregation
from qpiai_quantum.algorithms.opt.differentiation import CircuitObjective
from qpiai_quantum.algorithms.opt.solvers.qaoa import QAOASolver
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.simulator.statevector import StatevectorSimulator


def problem() -> SimpleNamespace:
    """A three-wire diagonal problem with shared, unequal cost coefficients."""
    return SimpleNamespace(
        n_qubits=3,
        get_hamiltonian_terms=lambda: [
            ([], 0.35),
            ([(0, "Z")], 1.3),
            ([(2, "Z")], -0.7),
            ([(0, "Z"), (1, "Z")], 2.4),
            ([(0, "Z"), (1, "Z"), (2, "Z")], -0.6),
        ],
    )


def energies() -> np.ndarray:
    """Compute original diagonal energies without using the SDK mapper."""
    values = []
    for basis in range(8):
        z0, z1, z2 = (1 - 2 * ((basis >> index) & 1) for index in range(3))
        values.append(0.35 + 1.3 * z0 - 0.7 * z2 + 2.4 * z0 * z1 - 0.6 * z0 * z1 * z2)
    return np.asarray(values)


def pauli(index: int, name: str) -> np.ndarray:
    """Build a full little-endian single-wire Pauli matrix independently."""
    matrices = {
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    result = np.ones((1, 1), dtype=complex)
    for wire in reversed(range(3)):
        result = np.kron(result, matrices[name] if wire == index else np.eye(2))
    return result


def state(circuit: Circuit) -> np.ndarray:
    """Use the direct mathematical simulator below the authentication boundary."""
    return np.asarray(
        StatevectorSimulator().run(circuit, shots=16, seed=7).statevector
    ).reshape(-1)


@pytest.mark.parametrize("mode", ["x", "xy", "warm"])
@pytest.mark.parametrize("depth", [1, 2])
def test_composed_state_matches_independent_exponentials(mode: str, depth: int) -> None:
    angles = np.linspace(-0.37, 0.41, 2 * depth)
    warm = WarmStart([0.12, 0.68, 0.91])
    options: dict[str, Any] = {}
    if mode == "xy":
        options = {
            "initial_state": BasisState("001"),
            "mixer": XYMixer(((0, 2), (1, 2))),
        }
        reference = np.eye(8, dtype=complex)[:, 1]
    elif mode == "warm":
        options = {"warm_start": warm}
        reference = np.array(
            [
                np.prod(
                    [
                        np.sqrt(p if (basis >> index) & 1 else 1 - p)
                        for index, p in enumerate(warm.probabilities)
                    ]
                )
                for basis in range(8)
            ],
            dtype=complex,
        )
    else:
        reference = np.ones(8, dtype=complex) / np.sqrt(8)
    for gamma, beta in angles.reshape(-1, 2):
        reference = np.exp(-1j * gamma * energies()) * reference
        if mode == "xy":
            for left, right in ((0, 2), (1, 2)):
                h = (
                    pauli(left, "X") @ pauli(right, "X")
                    + pauli(left, "Y") @ pauli(right, "Y")
                ) / 2
                reference = expm(-1j * beta * h) @ reference
        else:
            h = np.zeros((8, 8), dtype=complex)
            for index in range(3):
                if mode == "warm":
                    theta = 2 * np.arcsin(np.sqrt(warm.probabilities[index]))
                    h += np.sin(theta) * pauli(index, "X") + np.cos(theta) * pauli(
                        index, "Z"
                    )
                else:
                    h += pauli(index, "X")
            reference = expm(-1j * beta * h) @ reference
    actual = state(
        QAOASolver(problem(), layers=depth, verbose=False, **options).build_circuit(
            angles
        )
    )
    phase = np.vdot(reference, actual)
    np.testing.assert_allclose(actual, reference * phase / abs(phase), atol=1e-12)
    if mode == "xy":
        assert (
            sum(abs(actual[basis]) ** 2 for basis in range(8) if basis.bit_count() != 1)
            < 1e-24
        )


def test_warm_start_clipping_and_standard_limit() -> None:
    clipped = state(WarmStart([0, 1, 0.5], epsilon=0.1).initial_state(3))
    for index, expected in enumerate([0.1, 0.9, 0.5]):
        assert sum(
            abs(clipped[basis]) ** 2 for basis in range(8) if basis >> index & 1
        ) == pytest.approx(expected)
    theta = np.array([0.3, -0.6, 0.5, 0.12])
    ordinary = state(QAOASolver(problem(), layers=2).build_circuit(theta))
    warm = state(
        QAOASolver(
            problem(), layers=2, warm_start=WarmStart([0, 1, 0.5], epsilon=0.5)
        ).build_circuit(theta)
    )
    np.testing.assert_allclose(ordinary, warm, atol=1e-12)


def test_custom_components_are_independent_and_not_mutated() -> None:
    preparation = Circuit(3)
    preparation.x(2)

    def mixer(width: int, beta: float) -> Circuit:
        result = Circuit(width)
        result.ry(0, 3 * beta)
        return result

    solver = QAOASolver(problem(), initial_state=preparation, mixer=mixer)
    first = solver.build_circuit(np.array([0.1, 0.2]))
    first.icr.evolve.head.operation.qubits[0] = 1
    second = solver.build_circuit(np.array([0.1, 0.2]))
    assert preparation.icr.evolve.head.operation.qubits == [2]
    assert second.icr.evolve.head.operation.qubits == [2]
    solver.with_mixer(XYMixer()).with_ansatz("standard")
    with pytest.raises(ValueError, match="standard"):
        solver.with_ansatz("hardware_efficient")


@pytest.mark.parametrize("mode", ["x", "xy", "warm"])
@pytest.mark.parametrize("depth", [1, 2])
def test_gate_shift_shared_angles_match_independent_derivative(
    mode: str, depth: int
) -> None:
    options: dict[str, Any] = {}
    if mode == "xy":
        options = {"mixer": "xy", "initial_state": BasisState("001")}
    if mode == "warm":
        options = {"warm_start": WarmStart([0.2, 0.8, 0.4])}
    solver = QAOASolver(
        problem(), layers=depth, gradient_method="parameter_shift", **options
    )
    params = np.linspace(0.23, 0.67, 2 * depth)
    probes = []

    def evaluate(theta: np.ndarray, circuit: Circuit | None, probe: bool) -> float:
        probes.append(probe)
        vector = state(circuit if circuit is not None else solver.build_circuit(theta))
        return float(abs(vector) ** 2 @ energies())

    objective = CircuitObjective(evaluate)
    actual = solver._compute_gradient(params, objective)
    expected = []
    for index in range(len(params)):
        delta = np.zeros_like(params)
        delta[index] = 1e-6
        expected.append((objective(params + delta) - objective(params - delta)) / 2e-6)
    np.testing.assert_allclose(actual, expected, atol=2e-8)
    assert any(probes)


def test_cvar_fractional_boundary_and_mean_limit() -> None:
    probabilities = {"00": 0.1, "01": 0.3, "10": 0.6}
    values = {"00": -10.0, "01": -2.0, "10": 4.0}
    assert CVaR(0.25)(probabilities, values) == pytest.approx((-1 - 0.15 * 2) / 0.25)
    assert CVaR(1)(probabilities, values) == pytest.approx(0.8)
    assert CVaR(0.05)(probabilities, values) == -10
    assert CVaR(1e-20)({"0": 0.0, "1": 1.0}, {"0": -100.0, "1": -2.0}) == pytest.approx(
        -2.0
    )


def test_custom_aggregation_receives_probabilities_and_preserves_expectation() -> None:
    seen = []

    def aggregate(
        probabilities: Mapping[str, float], values: Mapping[str, float]
    ) -> float:
        seen.append((dict(probabilities), dict(values)))
        return min(values.values())

    solver = QAOASolver(problem(), aggregation=aggregate)
    result: Any = SimpleNamespace(counts={"000": 1, "111": 3})
    mean = solver._compute_expectation(result)
    score = solver._aggregate_objective(result, mean)
    assert score == min(energies()[0], energies()[7])
    assert mean == pytest.approx((energies()[0] + 3 * energies()[7]) / 4)
    assert seen[0][0] == {"000": 0.25, "111": 0.75}


@pytest.mark.parametrize("method", ["finite_difference", "spsa", "auto"])
def test_gradient_strategies_use_budgeted_callable_and_seed(method: str) -> None:
    left = QAOASolver(
        gradient_method=method, aggregation=CVaR(), seed=42, gradient_step=1e-5
    )
    right = QAOASolver(
        gradient_method=method, aggregation=CVaR(), seed=42, gradient_step=1e-5
    )
    theta = np.array([0.2, 0.3])
    calls = []

    def objective(values: np.ndarray) -> float:
        calls.append(values.copy())
        return float(values @ values)

    actual = left._compute_gradient(theta, objective)
    np.testing.assert_allclose(actual, right._compute_gradient(theta, objective))
    assert len(calls) == (4 if method == "spsa" else 8)
    if method != "spsa":
        np.testing.assert_allclose(actual, 2 * theta, atol=1e-10)
    np.testing.assert_array_equal(theta, [0.2, 0.3])


@pytest.mark.parametrize("budget", [1, 2, 3, 12, 30])
@pytest.mark.parametrize(
    "method", ["parameter_shift", "finite_difference", "spsa", "auto"]
)
@pytest.mark.parametrize("optimizer", ["ADAM", "GRADIENT_DESCENT", "L_BFGS_B"])
def test_new_gradients_respect_execution_budgets_and_probe_semantics(
    method: str, budget: int, optimizer: str
) -> None:
    solver = QAOASolver(
        problem(),
        optimizer=optimizer,
        max_iterations=2,
        num_starts=2,
        max_evaluations=budget,
        max_shots=(budget + 1) * 64,
        gradient_method=method,
        seed=17,
        verbose=False,
    )
    simulator = StatevectorSimulator()
    calls = []

    def execute(circuit: Circuit, **kwargs: object) -> object:
        calls.append(circuit)
        return simulator.run(circuit, shots=64, seed=91)

    with patch.object(solver, "_execute_circuit", side_effect=execute):
        result = solver.run(shots=64)
    assert len(calls) == len(result.evaluations) + 1 <= budget + 1
    assert result.metadata["total_shots"] == len(calls) * 64
    assert result.metadata["gradient_probe_evaluations"] == sum(
        e.kind == "gradient_probe" for e in result.evaluations
    )
    assert any(
        e.kind == "candidate"
        and e.parameters == result.optimal_parameters
        and e.energy == result.optimal_objective
        for e in result.evaluations
    )


@pytest.mark.parametrize(
    "aggregate",
    [
        CVaR(0.25),
        lambda probabilities, values: sum(
            probabilities[k] * values[k] ** 2 for k in probabilities
        ),
    ],
)
def test_aggregated_workflow_distinguishes_optimized_score_and_expectation(
    aggregate: Aggregation,
) -> None:
    solver = QAOASolver(
        problem(), aggregation=aggregate, max_evaluations=7, seed=21, verbose=False
    )
    simulator = StatevectorSimulator()
    with patch.object(
        solver,
        "_execute_circuit",
        side_effect=lambda circuit, **kwargs: simulator.run(
            circuit, shots=128, seed=10
        ),
    ):
        result = solver.run(shots=128)
    selected = [
        e for e in result.evaluations if e.parameters == result.optimal_parameters
    ]
    assert any(
        e.expectation == result.estimated_expectation
        and e.energy == result.optimal_objective
        for e in selected
    )
    assert solver.to_dict(result)["optimal_objective"] == result.optimal_objective
    assert result.metadata["energy_semantics"] == "estimated_expectation"


def test_unsafe_configuration_and_outputs_are_rejected() -> None:
    for alpha in [0, -0.1, 1.1, float("nan"), True]:
        with pytest.raises(ValueError):
            CVaR(alpha)
    for values in [[-0.1], [1.1], [float("nan")], []]:
        with pytest.raises(ValueError):
            WarmStart(values)
    with pytest.raises(ValueError, match="ordinary expectation"):
        QAOASolver(aggregation=CVaR(), gradient_method="parameter_shift")
    with pytest.raises(ValueError, match="supplies both"):
        QAOASolver(warm_start=WarmStart([0.5]), initial_state=BasisState("0"))
    with pytest.raises(ValueError, match="standard"):
        QAOASolver(ansatz="hardware_efficient", initial_state=BasisState("0"))
    with pytest.raises(ValueError, match="gradient"):
        QAOASolver(
            gradient_method=lambda theta, objective: np.array([np.nan])
        )._compute_gradient(np.zeros(2), lambda theta: 0)
    measured = Circuit(3)
    measured.measure_all()
    with pytest.raises(ValueError, match="measurement"):
        QAOASolver(problem(), initial_state=measured).build_circuit()
    with pytest.raises(ValueError, match="width"):
        QAOASolver(problem(), initial_state=Circuit(2)).build_circuit()
    with pytest.raises(ValueError, match="unique"):
        XYMixer(((0, 1), (1, 0)))(3, 0.1)
    with pytest.raises(ValueError, match="finite scalar"):
        solver = QAOASolver(
            problem(), aggregation=lambda probabilities, values: float("nan")
        )
        result: Any = SimpleNamespace(counts={"000": 1})
        solver._aggregate_objective(result, 0)


@pytest.mark.parametrize("method", ["parameter_shift", "auto", "spsa"])
def test_gradient_backend_failure_propagates_without_final_sampling(
    method: str,
) -> None:
    solver = QAOASolver(
        problem(), optimizer="ADAM", gradient_method=method, verbose=False
    )
    simulator = StatevectorSimulator()
    calls = []

    def execute(circuit: Circuit, **kwargs: Any) -> Any:
        calls.append(circuit)
        if len(calls) == 3:
            raise RuntimeError("test backend failure")
        return simulator.run(circuit, shots=32, seed=13)

    with patch.object(solver, "_execute_circuit", side_effect=execute):
        with pytest.raises(RuntimeError, match="test backend failure"):
            solver.run(shots=32)
    assert len(calls) == 3


def test_custom_gradient_uses_guarded_objective_and_preserves_inputs() -> None:
    def gradient(theta: np.ndarray, objective: Any) -> np.ndarray:
        objective(theta)
        theta[:] = 99
        return np.zeros(2)

    params = np.array([0.1, 0.2])
    solver = QAOASolver(gradient_method=gradient)
    calls = []

    def objective(theta: np.ndarray) -> float:
        calls.append(theta.copy())
        return 1.0

    solver._compute_gradient(params, objective)
    np.testing.assert_array_equal(params, [0.1, 0.2])
    np.testing.assert_array_equal(calls[0], params)
    solver = QAOASolver(
        problem(),
        optimizer="ADAM",
        gradient_method=gradient,
        max_evaluations=1,
        verbose=False,
    )
    simulator = StatevectorSimulator()
    with patch.object(
        solver,
        "_execute_circuit",
        side_effect=lambda circuit, **kwargs: simulator.run(circuit, shots=16, seed=3),
    ) as execute:
        result = solver.run(shots=16)
    assert execute.call_count == 2
    assert result.metadata["budget_exhausted"]


def test_gate_shift_rejects_custom_mixer_before_probing() -> None:
    solver = QAOASolver(
        problem(),
        mixer=lambda width, beta: XMixer()(width, beta),
        gradient_method="parameter_shift",
    )
    calls = []

    def evaluate(theta: np.ndarray, circuit: Circuit | None, probe: bool) -> float:
        calls.append(circuit)
        return 0.0

    objective = CircuitObjective(evaluate)
    with pytest.raises(ValueError, match="built-in mixer"):
        solver._compute_gradient(np.zeros(2), objective)
    assert calls == []


@pytest.mark.parametrize(
    "probabilities, values",
    [
        ({}, {}),
        ({"0": 1.0}, {"1": 0.0}),
        ({"0": 0.5}, {"0": 0.0}),
        ({"0": -1.0}, {"0": 0.0}),
        ({"0": float("nan")}, {"0": 0.0}),
        ({"0": 1.0}, {"0": float("inf")}),
    ],
)
def test_cvar_validates_distribution(
    probabilities: dict[str, float], values: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        CVaR()(probabilities, values)
