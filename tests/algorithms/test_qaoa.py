import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import numpy as np
import pytest
from dotenv import load_dotenv

load_dotenv("qcloud.env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.opt.solvers.qaoa import (
    QAOAEvaluation,
    QAOASolver,
    QAOAResult,
)
from qpiai_quantum.algorithms.opt.ansatz.standard import standard_qaoa_ansatz
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.icr.circuitoperation import CircuitOperation, OperationType
from qpiai_quantum.simulator.statevector import StatevectorSimulator


class DummyProblem:
    def __init__(self):
        self.n_qubits = 2

    def get_hamiltonian_terms(self):
        # H = 1.0 * Z0 - 0.5 * Z1 + 0.25 * Z0*Z1
        return [
            (((0, "Z"),), 1.0),
            (((1, "Z"),), -0.5),
            (((0, "Z"), (1, "Z")), 0.25),
        ]

    def decode_solution(self, bitstring):
        return bitstring

    def validate_solution(self, solution):
        return True, "Valid"

    def compute_solution_quality(self, solution):
        return {"cost": 0.5}


class TestQAOAInitializationAndAnsatz(unittest.TestCase):
    def test_init(self):
        qaoa = QAOASolver(layers=2, optimizer="cobyla", max_iterations=10)
        self.assertEqual(qaoa.layers, 2)
        self.assertEqual(qaoa.optimizer, "COBYLA")
        self.assertEqual(qaoa.max_iterations, 10)

    def test_ansatz_types(self):
        problem = DummyProblem()
        for ansatz_name in ["standard", "hardware_efficient"]:
            qaoa = QAOASolver(layers=1, ansatz=ansatz_name)
            qaoa.problem = problem
            circuit = qaoa.build_circuit()
            self.assertIsNotNone(circuit)
            self.assertEqual(circuit.num_qubits, 2)

    def test_rejects_invalid_solver_configuration(self):
        with self.assertRaisesRegex(ValueError, "layers must be a positive integer"):
            QAOASolver(layers=0)
        with self.assertRaisesRegex(
            ValueError, "max_iterations must be a positive integer"
        ):
            QAOASolver(max_iterations=0)
        with self.assertRaisesRegex(ValueError, "mixer must"):
            QAOASolver(mixer="unknown")
        with self.assertRaisesRegex(
            ValueError, "num_starts must be a positive integer"
        ):
            QAOASolver(num_starts=0)
        with self.assertRaisesRegex(ValueError, "max_evaluations"):
            QAOASolver(max_evaluations=0)

    def test_seed_reproduces_generated_start_points(self):
        first = QAOASolver(num_starts=3, seed=42)
        second = QAOASolver(num_starts=3, seed=42)

        first_starts = first._make_start_points(2, np.random.default_rng(first.seed))
        second_starts = second._make_start_points(2, np.random.default_rng(second.seed))

        for first_start, second_start in zip(first_starts, second_starts):
            np.testing.assert_allclose(first_start, second_start)


class TestQAOAHelpers(unittest.TestCase):
    def test_count_parameters(self):
        qaoa = QAOASolver(layers=2)
        qaoa.problem = DummyProblem()
        n_params = qaoa._count_parameters()
        self.assertEqual(n_params, 4)

    def test_shared_angles_and_hamiltonian_coefficients_are_applied(self):
        gamma = 0.3
        beta = 0.4
        circuit = standard_qaoa_ansatz(
            n_qubits=2,
            hamiltonian=DummyProblem(),
            layers=1,
            parameters=[gamma, beta],
        )

        rotations = [
            (operation.gate_name, operation.qubits, operation.params)
            for operation in circuit.icr.evolve
            if isinstance(operation, CircuitOperation)
            and operation.operation_type == OperationType.N_QUBIT_PARAMETRIC
        ]

        self.assertEqual(
            rotations,
            [
                ("RZ", [0], [2.0 * gamma * 1.0]),
                ("RZ", [1], [2.0 * gamma * -0.5]),
                ("RZZ", [0, 1], [2.0 * gamma * 0.25]),
                ("RX", [0], [2.0 * beta]),
                ("RX", [1], [2.0 * beta]),
            ],
        )

    def test_one_qubit_qaoa_matches_analytic_expectation(self):
        problem = DummyProblem()
        problem.n_qubits = 1
        problem.get_hamiltonian_terms = MagicMock(return_value=[(((0, "Z"),), 1.0)])
        gamma = np.pi / 4.0
        beta = np.pi / 8.0
        circuit = standard_qaoa_ansatz(
            n_qubits=1,
            hamiltonian=problem,
            parameters=[gamma, beta],
        )

        result = StatevectorSimulator().run(circuit, shots=100, seed=7)
        probabilities = np.abs(np.asarray(result.statevector)) ** 2
        measured_expectation = probabilities[0] - probabilities[1]
        analytic_expectation = np.sin(2.0 * gamma) * np.sin(2.0 * beta)

        self.assertAlmostEqual(measured_expectation, analytic_expectation)

    def test_standard_ansatz_validates_parameter_contract(self):
        with self.assertRaisesRegex(ValueError, "requires 4 parameters"):
            standard_qaoa_ansatz(
                n_qubits=2,
                hamiltonian=DummyProblem(),
                layers=2,
                parameters=[0.1, 0.2],
            )

    def test_standard_ansatz_rejects_non_diagonal_cost_hamiltonian(self):
        problem = DummyProblem()
        problem.get_hamiltonian_terms = MagicMock(return_value=[(((0, "X"),), 1.0)])

        with self.assertRaisesRegex(ValueError, "only I and Z"):
            standard_qaoa_ansatz(n_qubits=2, hamiltonian=problem)

    def test_best_sample_is_not_assumed_to_be_the_mode(self):
        qaoa = QAOASolver(problem=DummyProblem(), verbose=False)

        best, most_probable, energies = qaoa._select_best_sample({"00": 60, "11": 40})

        self.assertEqual(most_probable, "00")
        self.assertEqual(best, "11")
        self.assertAlmostEqual(energies["00"], 0.75)
        self.assertAlmostEqual(energies["11"], -0.25)

    def test_sample_energy_uses_rightmost_bit_for_qubit_zero(self):
        problem = DummyProblem()
        problem.get_hamiltonian_terms = MagicMock(
            return_value=[((), 2.0), (((0, "Z"),), 1.0)]
        )
        qaoa = QAOASolver(problem=problem, verbose=False)

        self.assertAlmostEqual(qaoa._bitstring_energy("00"), 3.0)
        self.assertAlmostEqual(qaoa._bitstring_energy("01"), 1.0)

    def test_empty_counts_are_an_execution_failure(self):
        qaoa = QAOASolver(problem=DummyProblem(), verbose=False)

        with self.assertRaisesRegex(ValueError, "no measurement counts"):
            qaoa._select_best_sample({})

    def test_qubo_and_ising_energies_agree_for_every_assignment(self):
        class PolynomialProblem(DummyProblem):
            def get_hamiltonian_terms(self):
                # 1.25 + 2*x - 3*y + 4*x*y, with x=(1-Z0)/2.
                return [
                    ((), 1.75),
                    (((0, "Z"),), -2.0),
                    (((1, "Z"),), 0.5),
                    (((0, "Z"), (1, "Z")), 1.0),
                ]

        problem = PolynomialProblem()
        qaoa = QAOASolver(problem=problem, verbose=False)

        for x in (0, 1):
            for y in (0, 1):
                bitstring = f"{y}{x}"
                expected = 1.25 + 2.0 * x - 3.0 * y + 4.0 * x * y
                self.assertAlmostEqual(qaoa._bitstring_energy(bitstring), expected)

    def test_result_distinguishes_expectation_and_sample_energy(self):
        qaoa = QAOASolver(problem=DummyProblem(), verbose=False)
        qaoa._optimize_parameters = MagicMock(
            return_value=(
                np.array([0.3, 0.4]),
                -0.1,
                {
                    "nit": 1,
                    "nfev": 1,
                    "success": True,
                    "history": [-0.1],
                    "param_history": [np.array([0.3, 0.4])],
                    "message": "test",
                },
            )
        )
        qaoa._execute_circuit = MagicMock(
            return_value=MagicMock(counts={"00": 60, "11": 40})
        )

        result = qaoa.run(
            problem=qaoa.problem,
            shots=100,
            device_name="QpiAI-QSV-Local",
        )

        self.assertEqual(result.bitstring, "11")
        self.assertEqual(result.best_sampled_bitstring, "11")
        self.assertEqual(result.most_probable_bitstring, "00")
        self.assertAlmostEqual(result.best_sampled_energy, -0.25)
        self.assertAlmostEqual(result.estimated_expectation, -0.1)
        self.assertEqual(result.metadata["energy_semantics"], "estimated_expectation")
        self.assertNotIn("problem", qaoa._optimize_parameters.call_args.kwargs)

    def test_run_rejects_non_positive_shots_before_execution(self):
        qaoa = QAOASolver(problem=DummyProblem(), verbose=False)

        with self.assertRaisesRegex(ValueError, "shots must be a positive integer"):
            qaoa.run(shots=0)

    def test_multi_start_returns_best_run_and_summaries(self):
        qaoa = QAOASolver(problem=DummyProblem(), num_starts=2, seed=7, verbose=False)
        qaoa._optimize_parameters = MagicMock(
            side_effect=[
                (
                    np.array([0.1, 0.2]),
                    0.5,
                    {"nit": 2, "nfev": 3, "success": True, "message": "first"},
                ),
                (
                    np.array([0.3, 0.4]),
                    -0.5,
                    {"nit": 3, "nfev": 4, "success": True, "message": "second"},
                ),
            ]
        )
        qaoa._execute_circuit = MagicMock(
            return_value=MagicMock(counts={"00": 1, "11": 1})
        )

        result = qaoa.run(shots=10)

        self.assertEqual(result.optimal_parameters, [0.3, 0.4])
        self.assertEqual(result.estimated_expectation, -0.5)
        self.assertEqual(len(result.start_results), 2)
        self.assertEqual(result.metadata["best_start_index"], 1)
        self.assertEqual(result.metadata["completed_starts"], 2)

    def test_evaluation_and_shot_budgets_include_final_sampling(self):
        qaoa = QAOASolver(
            problem=DummyProblem(),
            max_evaluations=2,
            max_shots=300,
            verbose=False,
        )
        qaoa._execute_circuit = MagicMock(
            return_value=MagicMock(counts={"00": 1, "11": 1})
        )

        def consume_budget(objective, initial_point, **_kwargs):
            objective(initial_point)
            objective(initial_point + 0.1)
            objective(initial_point + 0.2)
            raise AssertionError("budget guard should stop before this line")

        qaoa._optimize_parameters = consume_budget
        observed: list[QAOAEvaluation] = []

        result = qaoa.run(shots=100, evaluation_callback=observed.append)

        self.assertEqual(len(result.evaluations), 2)
        self.assertEqual(observed, result.evaluations)
        self.assertEqual(qaoa._execute_circuit.call_count, 3)
        self.assertEqual(result.metadata["objective_evaluations"], 2)
        self.assertEqual(result.metadata["circuit_evaluations"], 3)
        self.assertEqual(result.metadata["total_shots"], 300)

    def test_shot_budget_must_reserve_optimization_and_final_sampling(self):
        qaoa = QAOASolver(problem=DummyProblem(), max_shots=100, verbose=False)

        with self.assertRaisesRegex(ValueError, "one objective evaluation"):
            qaoa.run(shots=100)

    def test_budget_stop_preserves_completed_custom_iterations(self):
        cases = [
            ("SPSA", 5, {}),
            ("GRADIENT_DESCENT", 7, {"tol": 0.0}),
            ("ADAM", 7, {"tol": 0.0}),
            ("GENETIC_ALGORITHM", 6, {"population_size": 4}),
            ("PARTICLE_SWARM", 7, {"n_particles": 3}),
        ]
        for optimizer, budget, options in cases:
            with self.subTest(optimizer=optimizer):
                solver = QAOASolver(
                    problem=DummyProblem(),
                    optimizer=optimizer,
                    max_iterations=20,
                    max_evaluations=budget,
                    seed=42,
                    verbose=False,
                )
                solver._execute_circuit = MagicMock(
                    return_value=MagicMock(counts={"00": 5, "11": 5})
                )
                result = solver.run(shots=10, **options)

                self.assertEqual(result.start_results[0].iterations, 1)
                self.assertEqual(result.metadata["actual_iterations"], 1)
                self.assertEqual(result.metadata["objective_evaluations"], budget)
                self.assertEqual(
                    result.metadata["termination_reason"], "evaluation_budget"
                )
                self.assertFalse(result.metadata["converged_early"])
                self.assertTrue(result.metadata["budget_exhausted"])

    def test_interrupted_scipy_callbacks_are_separate_from_iterations(self):
        solver = QAOASolver(
            problem=DummyProblem(),
            optimizer="NELDER_MEAD",
            max_evaluations=2,
            verbose=False,
        )
        solver._execute_circuit = MagicMock(
            return_value=MagicMock(counts={"00": 5, "11": 5})
        )

        def interrupted_minimize(objective, initial_point, callback, **_kwargs):
            objective(initial_point)
            callback(initial_point)
            objective(initial_point)
            callback(initial_point)
            objective(initial_point)

        with patch(
            "qpiai_quantum.algorithms.utils.optimizers.gradient_free.minimize",
            side_effect=interrupted_minimize,
        ):
            result = solver.run(shots=10)

        self.assertIsNone(result.start_results[0].iterations)
        self.assertIsNone(result.metadata["actual_iterations"])
        self.assertEqual(result.start_results[0].callback_reports, 2)
        self.assertFalse(result.metadata["converged_early"])

    def test_budget_stop_during_initialization_has_zero_completed_iterations(self):
        solver = QAOASolver(
            problem=DummyProblem(),
            optimizer="SPSA",
            max_evaluations=1,
            verbose=False,
        )
        solver._execute_circuit = MagicMock(
            return_value=MagicMock(counts={"00": 5, "11": 5})
        )

        result = solver.run(shots=10)

        self.assertEqual(result.start_results[0].iterations, 0)
        self.assertEqual(result.metadata["termination_reason"], "evaluation_budget")

    def test_interrupted_differential_evolution_preserves_native_generation_count(self):
        solver = QAOASolver(
            problem=DummyProblem(),
            optimizer="DIFFERENTIAL_EVOLUTION",
            max_evaluations=2,
            verbose=False,
        )
        solver._execute_circuit = MagicMock(
            return_value=MagicMock(counts={"00": 5, "11": 5})
        )

        def interrupted_evolution(objective, bounds, callback, **_kwargs):
            from scipy.optimize import OptimizeResult

            params = np.zeros(len(bounds))
            objective(params)
            callback(
                intermediate_result=OptimizeResult(nit=1, x=params, convergence=0.1)
            )
            objective(params)
            objective(params)

        with patch(
            "qpiai_quantum.algorithms.utils.optimizers.gradient_free.differential_evolution",
            side_effect=interrupted_evolution,
        ):
            result = solver.run(shots=10)

        self.assertEqual(result.start_results[0].iterations, 1)
        self.assertEqual(result.start_results[0].callback_reports, 1)
        self.assertEqual(result.metadata["termination_reason"], "evaluation_budget")

    def test_iteration_limit_is_not_convergence(self):
        solver = QAOASolver(
            problem=DummyProblem(),
            optimizer="PARTICLE_SWARM",
            max_iterations=2,
            verbose=False,
        )
        solver._execute_circuit = MagicMock(
            return_value=MagicMock(counts={"00": 5, "11": 5})
        )

        result = solver.run(shots=10, n_particles=3)

        self.assertEqual(result.start_results[0].iterations, 2)
        self.assertEqual(result.metadata["termination_reason"], "iteration_limit")
        self.assertFalse(result.metadata["converged_early"])

    def test_cobyla_budget_is_forwarded_without_duplicate_arguments(self):
        solver = QAOASolver(
            problem=DummyProblem(),
            optimizer="COBYLA",
            max_evaluations=10,
            verbose=False,
        )
        solver._execute_circuit = MagicMock(
            return_value=MagicMock(counts={"00": 5, "11": 5})
        )

        result = solver.run(shots=10, maxfun=10, rhobeg=0.5)

        self.assertGreater(result.metadata["objective_evaluations"], 0)
        self.assertLessEqual(result.metadata["objective_evaluations"], 10)
        self.assertLessEqual(solver._execute_circuit.call_count, 11)

    def test_total_evaluation_budget_is_shared_across_starts(self):
        qaoa = QAOASolver(
            problem=DummyProblem(),
            num_starts=2,
            max_evaluations=4,
            seed=9,
            verbose=False,
        )
        qaoa._execute_circuit = MagicMock(
            return_value=MagicMock(counts={"00": 1, "11": 1})
        )

        def consume_start_allowance(objective, initial_point, **_kwargs):
            while True:
                objective(initial_point)

        qaoa._optimize_parameters = consume_start_allowance

        result = qaoa.run(shots=10)

        self.assertEqual([item.evaluations for item in result.start_results], [2, 2])
        self.assertEqual(result.metadata["objective_evaluations"], 4)
        self.assertEqual(result.metadata["completed_starts"], 2)

    def test_evaluation_callback_exceptions_are_not_hidden(self):
        qaoa = QAOASolver(problem=DummyProblem(), verbose=False)
        qaoa._execute_circuit = MagicMock(
            return_value=MagicMock(counts={"00": 1, "11": 1})
        )

        def evaluate_once(objective, initial_point, **_kwargs):
            objective(initial_point)
            raise AssertionError("callback should stop before this line")

        def fail_callback(_evaluation):
            raise RuntimeError("stop QAOA")

        qaoa._optimize_parameters = evaluate_once

        with self.assertRaisesRegex(RuntimeError, "stop QAOA"):
            qaoa.run(shots=10, evaluation_callback=fail_callback)

    def test_population_optimizer_accepts_explicit_bounds_once(self):
        qaoa = QAOASolver(
            problem=DummyProblem(), optimizer="GENETIC_ALGORITHM", verbose=False
        )
        expected = (np.zeros(4), 0.0, {"success": True})

        with patch(
            "qpiai_quantum.algorithms.opt.solvers.qaoa.genetic_algorithm_optimize",
            return_value=expected,
        ) as optimize:
            result = qaoa._optimize_parameters(
                objective=lambda _params: 0.0,
                initial_point=np.zeros(4),
                n_params=4,
                bounds=[(-1.0, 1.0)] * 4,
            )

        self.assertEqual(result, expected)
        self.assertEqual(optimize.call_args.kwargs["bounds"], [(-1.0, 1.0)] * 4)

    def test_full_local_hybrid_loop_reaches_one_qubit_minimum(self):
        problem = DummyProblem()
        problem.n_qubits = 1
        problem.get_hamiltonian_terms = MagicMock(return_value=[(((0, "Z"),), 1.0)])
        qaoa = QAOASolver(
            problem=problem,
            layers=1,
            optimizer="COBYLA",
            max_iterations=40,
            initial_params=np.array([0.3, -0.3]),
            verbose=False,
        )
        simulator = StatevectorSimulator()

        def execute_locally(circuit, shots=None, **_kwargs):
            return simulator.run(circuit, shots=shots or 4096, seed=17)

        qaoa._execute_circuit = execute_locally

        result = qaoa.run(shots=4096)

        self.assertLess(result.estimated_expectation, -0.95)
        self.assertEqual(result.best_sampled_bitstring, "1")
        self.assertAlmostEqual(result.best_sampled_energy, -1.0)
        self.assertEqual(result.metadata["n_parameters"], 2)


@pytest.mark.usefixtures("authenticated_sdk_user")
@unittest.skipUnless(
    os.environ.get("RUN_ALGO_CORRECTNESS") == "1" and bool(os.getenv("API_KEY")),
    "Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 and API_KEY in environment to run.",
)
class TestQAOACorrectness(unittest.TestCase):
    def test_live_qaoa_optimization(self):
        # Run QAOA on DummyProblem (2 qubits) with COBYLA and 2 iterations
        # (extremely fast / minimal time consuming)
        qaoa = QAOASolver(
            layers=1,
            optimizer="cobyla",
            max_iterations=2,
        )
        problem = DummyProblem()
        res = qaoa.run(
            problem=problem,
            shots=100,
            device_name="QpiAI-QSV-Local",
        )
        self.assertIsInstance(res, QAOAResult)
        self.assertIsNotNone(res.optimal_energy)
        self.assertIsNotNone(res.optimal_parameters)
        self.assertIsNotNone(res.bitstring)


if __name__ == "__main__":
    unittest.main()
