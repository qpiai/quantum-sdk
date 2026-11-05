"""Independent QAOA mathematics and complete offline orchestration checks."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm

from qpiai_quantum.algorithms.opt.solvers.qaoa import QAOASolver
from qpiai_quantum.simulator.statevector import StatevectorSimulator
from qpiai_quantum.circuit import Circuit


OPTIMIZERS = (
    "COBYLA",
    "SPSA",
    "DIFFERENTIAL_EVOLUTION",
    "GENETIC_ALGORITHM",
    "PARTICLE_SWARM",
    "SLSQP",
    "NELDER_MEAD",
    "POWELL",
    "ADAM",
    "GRADIENT_DESCENT",
    "L_BFGS_B",
)


class DiagonalTestProblem:
    """Small test-only Hamiltonian specified by an exhaustive energy table."""

    def __init__(self, energies: np.ndarray):
        self.energies = np.asarray(energies)
        self.n_qubits = len(energies).bit_length() - 1

    def get_hamiltonian_terms(self):
        terms = []
        for mask in range(len(self.energies)):
            signs = [
                (-1) ** ((basis & mask).bit_count())
                for basis in range(len(self.energies))
            ]
            coefficient = float(np.dot(self.energies, signs) / len(signs))
            terms.append(
                (
                    tuple((q, "Z") for q in range(self.n_qubits) if (mask >> q) & 1),
                    coefficient,
                )
            )
        return terms

    def decode_solution(self, bitstring):
        return bitstring

    def validate_solution(self, solution):
        return True, "Valid"

    def compute_solution_quality(self, solution):
        return {"energy": float(self.energies[int(solution, 2)])}


def weighted_problem(n_qubits: int = 3) -> tuple[DiagonalTestProblem, np.ndarray]:
    """Enumerate a binary polynomial independently of SDK problem builders."""
    linear = [1.3, -2.1, 0.7][:n_qubits]
    quadratic = {(0, 1): 2.4, (1, 2): -1.7, (0, 2): 0.6}
    quadratic = {
        pair: value for pair, value in quadratic.items() if max(pair) < n_qubits
    }
    energies = []
    for basis in range(2**n_qubits):
        bits = [(basis >> index) & 1 for index in range(n_qubits)]
        energies.append(
            0.35
            + sum(c * x for c, x in zip(linear, bits))
            + sum(c * bits[i] * bits[j] for (i, j), c in quadratic.items())
        )
    return DiagonalTestProblem(np.asarray(energies)), np.asarray(energies)


def reference_state(parameters: np.ndarray, energies: np.ndarray) -> np.ndarray:
    """Evolve full cost and mixer matrices without SDK gate construction."""
    size = len(energies)
    mixer = np.zeros((size, size))
    for basis in range(size):
        for qubit in range(size.bit_length() - 1):
            mixer[basis ^ (1 << qubit), basis] += 1.0
    state = np.ones(size, dtype=complex) / np.sqrt(size)
    for gamma, beta in parameters.reshape(-1, 2):
        state = np.exp(-1j * gamma * energies) * state
        state = expm(-1j * beta * mixer) @ state
    return state


@pytest.mark.parametrize("n_qubits", [2, 3])
@pytest.mark.parametrize("layers", [1, 2, 3])
def test_weighted_qubo_state_and_energy(n_qubits: int, layers: int) -> None:
    problem, energies = weighted_problem(n_qubits)
    params = np.random.default_rng(21 + layers).uniform(-1, 1, 2 * layers)
    solver = QAOASolver(problem=problem, layers=layers, verbose=False)
    result = StatevectorSimulator().run(solver.build_circuit(params), shots=100, seed=1)
    actual = np.asarray(result.statevector)
    expected = reference_state(params, energies)
    # Constants omitted from the circuit contribute only a global phase.
    overlap = np.vdot(expected, actual)
    np.testing.assert_allclose(actual, expected * overlap / abs(overlap), atol=1e-12)
    assert float(np.abs(actual) ** 2 @ energies) == pytest.approx(
        float(np.abs(expected) ** 2 @ energies), abs=1e-12
    )
    for basis, energy in enumerate(energies):
        assert solver._bitstring_energy(
            format(basis, f"0{n_qubits}b")
        ) == pytest.approx(energy)


@pytest.mark.parametrize("layers", [1, 2])
def test_shared_weighted_gradient_matches_numerical_derivative(layers: int) -> None:
    problem, energies = weighted_problem()
    params = np.random.default_rng(123).uniform(-0.8, 0.8, 2 * layers)
    solver = QAOASolver(
        problem=problem, layers=layers, gradient_step=1e-5, verbose=False
    )

    def objective(theta: np.ndarray) -> float:
        return float(np.abs(reference_state(theta, energies)) ** 2 @ energies)

    expected = []
    # Five-point stencil is independent of the solver's differentiation rule.
    step = 1e-4
    for index in range(len(params)):
        delta = np.eye(len(params))[index] * step
        expected.append(
            (
                objective(params - 2 * delta)
                - 8 * objective(params - delta)
                + 8 * objective(params + delta)
                - objective(params + 2 * delta)
            )
            / (12 * step)
        )
    np.testing.assert_allclose(
        solver._compute_gradient(params, objective), expected, atol=1e-6
    )


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("budget", [1, 7, 24])
@pytest.mark.parametrize("budget_kind", ["evaluations", "shots", "both"])
def test_all_optimizer_workflows_respect_budgets(
    optimizer: str, budget: int, budget_kind: str
) -> None:
    problem, energies = weighted_problem(2)
    shots = 4096
    solver = QAOASolver(
        problem=problem,
        optimizer=optimizer,
        num_starts=3,
        seed=19,
        max_iterations=4,
        max_evaluations=budget if budget_kind != "shots" else None,
        max_shots=(budget + 1) * shots if budget_kind != "evaluations" else None,
        initial_params=np.array([0.31, -0.27]),
        verbose=False,
    )
    simulator = StatevectorSimulator()
    calls = []

    def execute(circuit: Circuit, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return simulator.run(circuit, shots=int(kwargs["shots"]), seed=73)

    with patch.object(solver, "_execute_circuit", side_effect=execute):
        options: dict[str, Any] = {
            "GENETIC_ALGORITHM": {"population_size": 4},
            "PARTICLE_SWARM": {"n_particles": 4},
            "DIFFERENTIAL_EVOLUTION": {"popsize": 3},
        }.get(optimizer, {})
        result = solver.run(shots=shots, **options)
    assert 1 <= len(result.evaluations) <= budget
    assert len(calls) == len(result.evaluations) + 1
    assert result.metadata["total_shots"] == len(calls) * shots
    assert result.metadata["total_shots"] <= (budget + 1) * shots
    assert len(result.start_results) == min(3, budget)
    assert sum(start.evaluations for start in result.start_results) == len(
        result.evaluations
    )
    assert result.estimated_expectation == pytest.approx(
        min(start.optimal_energy for start in result.start_results)
    )
    for evaluation in result.evaluations:
        reference = float(
            np.abs(reference_state(np.array(evaluation.parameters), energies)) ** 2
            @ energies
        )
        assert evaluation.energy == pytest.approx(reference, abs=0.16)
    sampled = {
        bits: float(energies[int(bits, 2)])
        for bits, count in result.counts.items()
        if count > 0
    }
    assert result.best_sampled_energy == pytest.approx(min(sampled.values()))
    assert result.best_sampled_bitstring in sampled


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("fail_at", [1, 2, 4])
def test_backend_failures_propagate_without_extra_executions(
    optimizer: str, fail_at: int
) -> None:
    problem, _ = weighted_problem(2)
    solver = QAOASolver(
        problem=problem,
        optimizer=optimizer,
        max_evaluations=3,
        max_iterations=4,
        seed=1,
        verbose=False,
    )
    calls = 0

    def execute(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError("offline backend unavailable")
        return SimpleNamespace(counts={"00": 9, "01": 1})

    with patch.object(solver, "_execute_circuit", side_effect=execute):
        with pytest.raises(RuntimeError, match="offline backend unavailable"):
            solver.run(shots=10)
    assert calls == fail_at


@pytest.mark.parametrize(
    "optimizer",
    [
        "ADAM",
        "GRADIENT_DESCENT",
        "L_BFGS_B",
        "SLSQP",
        "NELDER_MEAD",
        "POWELL",
        "COBYLA",
        "SPSA",
        "DIFFERENTIAL_EVOLUTION",
    ],
)
def test_constant_objective_can_converge_early(optimizer: str) -> None:
    problem = DiagonalTestProblem(np.asarray([2.75, 2.75]))
    solver = QAOASolver(
        problem=problem,
        optimizer=optimizer,
        max_iterations=100,
        max_evaluations=200,
        num_starts=2,
        seed=4,
        verbose=False,
    )
    with patch.object(
        solver,
        "_execute_circuit",
        return_value=SimpleNamespace(counts={"0": 5, "1": 5}),
    ):
        result = solver.run(shots=10, popsize=3)
    assert result.estimated_expectation == pytest.approx(2.75)
    assert all(start.success for start in result.start_results)
    assert all(
        start.termination_reason == "converged" for start in result.start_results
    )
    assert all(
        start.iterations is None or start.iterations < 100
        for start in result.start_results
    )


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_all_optimizers_complete_unbudgeted_multiple_starts(optimizer: str) -> None:
    problem, energies = weighted_problem(2)
    solver = QAOASolver(
        problem=problem,
        optimizer=optimizer,
        max_iterations=3,
        num_starts=2,
        seed=41,
        verbose=False,
    )
    simulator = StatevectorSimulator()
    calls = 0

    def execute(circuit: Circuit, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return simulator.run(circuit, shots=kwargs["shots"], seed=91)

    options: dict[str, Any] = {
        "GENETIC_ALGORITHM": {"population_size": 4},
        "PARTICLE_SWARM": {"n_particles": 4},
        "DIFFERENTIAL_EVOLUTION": {"popsize": 3},
    }.get(optimizer, {})
    with patch.object(solver, "_execute_circuit", side_effect=execute):
        result = solver.run(shots=4096, **options)
    assert len(result.start_results) == 2
    assert calls == len(result.evaluations) + 1
    assert not result.metadata["budget_exhausted"]
    assert result.metadata["effective_objective_evaluation_limit"] is None
    assert result.estimated_expectation == pytest.approx(
        min(s.optimal_energy for s in result.start_results)
    )
    for start in result.start_results:
        exact = float(
            np.abs(reference_state(np.array(start.optimal_parameters), energies)) ** 2
            @ energies
        )
        assert start.optimal_energy == pytest.approx(exact, abs=0.16)
    assert result.best_sampled_energy == pytest.approx(
        min(
            energies[int(bits, 2)] for bits, count in result.counts.items() if count > 0
        )
    )


@pytest.mark.parametrize(
    "step", [0, -0.1, float("inf"), float("nan"), True, "small", None]
)
def test_gradient_step_rejects_invalid_values(step: Any) -> None:
    with pytest.raises(ValueError, match="gradient_step must be a positive finite"):
        QAOASolver(gradient_step=step)


def test_gradient_step_controls_probes_and_does_not_mutate_parameters() -> None:
    solver = QAOASolver(gradient_step=0.02, verbose=False)
    params = np.array([0.31, 0.27])
    probes = []

    def objective(theta: np.ndarray) -> float:
        probes.append(theta.copy())
        return float(np.sin(2 * theta[0]) * np.sin(2 * theta[1]))

    actual = solver._compute_gradient(params, objective)
    expected = np.array(
        [2 * np.cos(0.62) * np.sin(0.54), 2 * np.sin(0.62) * np.cos(0.54)]
    )
    np.testing.assert_allclose(actual, expected, atol=3e-4)
    np.testing.assert_array_equal(params, [0.31, 0.27])
    np.testing.assert_allclose(
        probes, [[0.33, 0.27], [0.29, 0.27], [0.31, 0.29], [0.31, 0.25]]
    )


@pytest.mark.parametrize("ansatz", ["hardware_efficient", "custom"])
def test_gradient_matches_custom_and_hardware_ansatz_derivatives(ansatz: str) -> None:
    problem, _ = weighted_problem(2)

    def custom(n_qubits: int, layers: int) -> Circuit:
        from qpiai_quantum.algorithms.opt import parameterize

        circuit = Circuit(n_qubits)
        circuit.ry(0, 0.0)
        parameterize(circuit, 0)
        circuit.rx(1, 0.0)
        parameterize(circuit, 1)
        circuit.cx(0, 1)
        return circuit

    solver = QAOASolver(
        problem=problem,
        ansatz=custom if ansatz == "custom" else ansatz,
        gradient_step=1e-5,
        verbose=False,
    )
    params = np.linspace(0.2, 0.6, solver._count_parameters())
    simulator = StatevectorSimulator()
    energies = np.array([solver._bitstring_energy(format(i, "02b")) for i in range(4)])

    def objective(theta: np.ndarray) -> float:
        result = simulator.run(solver.build_circuit(theta), shots=10, seed=1)
        return float(np.abs(np.asarray(result.statevector)) ** 2 @ energies)

    expected = []
    for index in range(len(params)):
        delta = np.eye(len(params))[index] * 1e-4
        expected.append(
            (
                objective(params - 2 * delta)
                - 8 * objective(params - delta)
                + 8 * objective(params + delta)
                - objective(params + 2 * delta)
            )
            / 0.0012
        )
    np.testing.assert_allclose(
        solver._compute_gradient(params, objective), expected, atol=1e-7
    )


@pytest.mark.parametrize("optimizer", ["ADAM", "GRADIENT_DESCENT", "L_BFGS_B"])
def test_gradient_optimizers_improve_exact_qaoa_energy(optimizer: str) -> None:
    problem = DiagonalTestProblem(np.asarray([1.0, -1.0]))
    solver = QAOASolver(
        problem=problem,
        optimizer=optimizer,
        max_iterations=80,
        initial_params=np.array([0.31, -0.27]),
        seed=7,
        verbose=False,
    )
    simulator = StatevectorSimulator()

    def execute(circuit: Circuit, **kwargs: Any) -> Any:
        return simulator.run(circuit, shots=kwargs["shots"], seed=92)

    def exact_energy(result: Any) -> float:
        probabilities = np.abs(np.asarray(result.statevector)) ** 2
        return float(probabilities[0] - probabilities[1])

    options: dict[str, Any] = {"learning_rate": 0.1} if optimizer != "L_BFGS_B" else {}
    with (
        patch.object(solver, "_execute_circuit", side_effect=execute),
        patch.object(solver, "_compute_expectation", side_effect=exact_energy),
    ):
        result = solver.run(shots=100, **options)
    initial_energy = float(np.sin(0.62) * np.sin(-0.54))
    assert result.estimated_expectation < initial_energy - 0.5
    assert result.estimated_expectation < -0.99
    assert result.best_sampled_bitstring == "1"
