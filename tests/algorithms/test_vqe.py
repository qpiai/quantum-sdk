import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import numpy as np
import pytest
from dotenv import load_dotenv

load_dotenv("qcloud.env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver, VQEResult
from qpiai_quantum.circuit import Circuit


class DummyHamiltonian:
    def get_hamiltonian_terms(self):
        # H = 1.0 * Z0 - 0.5 * Z1 + 0.25 * Z0*Z1
        return [
            (((0, "Z"),), 1.0),
            (((1, "Z"),), -0.5),
            (((0, "Z"), (1, "Z")), 0.25),
        ]


class TestVQEInitializationAndAnsatz(unittest.TestCase):
    def test_init(self):
        vqe = VQESolver(
            n_qubits=2, ansatz="standard", optimizer="cobyla", max_iterations=10
        )
        self.assertEqual(vqe.num_qubits, 2)
        self.assertEqual(vqe.ansatz, "standard")
        self.assertEqual(vqe.optimizer, "cobyla")
        self.assertEqual(vqe.max_iterations, 10)

    def test_ansatz_types(self):
        for ansatz_name in ["standard", "hardware_efficient", "two_local"]:
            vqe = VQESolver(n_qubits=2, ansatz=ansatz_name)
            circuit = vqe.build_circuit()
            self.assertIsNotNone(circuit)
            self.assertEqual(circuit.num_qubits, 2)

    def test_unknown_ansatz(self):
        vqe = VQESolver(n_qubits=2, ansatz="unknown_ansatz")
        with self.assertRaises(ValueError):
            vqe.build_circuit()


class TestVQEHelpers(unittest.TestCase):
    def test_unknown_optimizer_iterations_are_preserved_without_comparison_errors(self):
        solver = VQESolver(
            n_qubits=2,
            ansatz=lambda _: Circuit(2),
            optimizer="cobyla",
            verbose=True,
        )
        for success in (False, True):
            with self.subTest(success=success):
                info = {"nit": None, "nfev": 1, "success": success}
                with (
                    patch.object(
                        solver,
                        "_optimize_parameters",
                        return_value=(np.array([]), 0.75, info),
                    ),
                    patch.object(
                        solver,
                        "_execute_circuit",
                        return_value=MagicMock(counts={"00": 10}),
                    ),
                    patch("builtins.print"),
                ):
                    result = solver.run(
                        hamiltonian=DummyHamiltonian(),
                        device_name="QpiAI-QSV-Local",
                        method="statevector_sampling",
                        shots=10,
                    )

                self.assertIsNone(result.metadata["actual_iterations"])
                self.assertFalse(result.metadata["converged_early"])

    def test_count_parameters(self):
        vqe = VQESolver(n_qubits=2, ansatz="standard")
        n_params = vqe.num_parameters
        self.assertGreater(n_params, 0)
        self.assertIsNone(vqe.circuit)

    def test_compute_expectation(self):
        vqe = VQESolver(n_qubits=2)
        vqe.hamiltonian = DummyHamiltonian()

        # mock execution results
        mock_result = MagicMock()
        mock_result.counts = {
            "00": 1000
        }  # Z0=1, Z1=1 -> expectation = 1.0*1 - 0.5*1 + 0.25*1 = 0.75
        exp = vqe._compute_expectation(mock_result)
        self.assertAlmostEqual(exp, 0.75)

    def test_counts_energy_statistics_preserve_group_covariance(self):
        vqe = VQESolver(n_qubits=2)
        terms = [([(0, "Z")], 1.0), ([(1, "Z")], 1.0)]

        energy, standard_error = vqe._compute_counts_energy_statistics(
            {"00": 50, "11": 50}, terms
        )

        self.assertAlmostEqual(energy, 0.0)
        self.assertAlmostEqual(standard_error, np.sqrt(4.0 / 99.0))

    def test_sampled_run_reports_fresh_energy_and_uncertainty(self):
        class OneQubitHamiltonian:
            def get_hamiltonian_terms(self):
                return [([(0, "Z")], 1.0)]

        ansatz = Circuit(1)
        vqe = VQESolver(
            n_qubits=1,
            ansatz=lambda _: ansatz,
            optimizer="cobyla",
            verbose=False,
        )
        optimization_info = {
            "nfev": 1,
            "nit": 1,
            "history": [-9.0],
            "param_history": [np.array([])],
            "success": True,
            "message": "test",
        }
        energy_result = MagicMock()
        energy_result.counts = {"0": 75, "1": 25}
        bitstring_result = MagicMock()
        bitstring_result.counts = {"0": 100}

        with (
            patch.object(
                vqe,
                "_optimize_parameters",
                return_value=(np.array([]), -9.0, optimization_info),
            ),
            patch.object(
                vqe,
                "_execute_circuit",
                side_effect=[energy_result, bitstring_result],
            ),
        ):
            result = vqe.run(
                hamiltonian=OneQubitHamiltonian(),
                method="qasm",
                device_name="QpiAI-QSV-Local",
                shots=100,
            )

        self.assertAlmostEqual(result.optimal_energy, 0.5)
        self.assertAlmostEqual(result.energy_standard_error, np.sqrt(0.75 / 99.0))
        self.assertEqual(result.metadata["optimizer_energy"], -9.0)
        self.assertEqual(
            vqe.to_dict(result)["energy_standard_error"],
            result.energy_standard_error,
        )

    def test_statevector_run_keeps_exact_optimizer_energy(self):
        class OneQubitHamiltonian:
            def get_hamiltonian_terms(self):
                return [([(0, "Z")], 1.0)]

        ansatz = Circuit(1)
        vqe = VQESolver(
            n_qubits=1,
            ansatz=lambda _: ansatz,
            optimizer="cobyla",
            verbose=False,
        )
        optimization_info = {
            "nfev": 1,
            "nit": 1,
            "history": [-0.75],
            "param_history": [np.array([])],
            "success": True,
            "message": "test",
        }
        bitstring_result = MagicMock()
        bitstring_result.counts = {"0": 100}

        with (
            patch.object(
                vqe,
                "_optimize_parameters",
                return_value=(np.array([]), -0.75, optimization_info),
            ),
            patch.object(vqe, "_execute_circuit", return_value=bitstring_result),
        ):
            result = vqe.run(
                hamiltonian=OneQubitHamiltonian(),
                method="statevector",
                device_name="QpiAI-QSV-Local",
                shots=100,
            )

        self.assertEqual(result.optimal_energy, -0.75)
        self.assertIsNone(result.energy_standard_error)
        self.assertNotIn("optimizer_energy", result.metadata)


@pytest.mark.usefixtures("authenticated_sdk_user")
@unittest.skipUnless(
    os.environ.get("RUN_ALGO_CORRECTNESS") == "1" and bool(os.getenv("API_KEY")),
    "Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 and API_KEY in environment to run.",
)
class TestVQECorrectness(unittest.TestCase):
    def test_live_vqe_optimization(self):
        # Run VQE on DummyHamiltonian (2 qubits) with COBYLA and 2 iterations
        # (extremely fast / minimal time consuming)
        vqe = VQESolver(
            n_qubits=2,
            ansatz="standard",
            optimizer="cobyla",
            max_iterations=2,
        )
        hamiltonian = DummyHamiltonian()
        res = vqe.run(
            hamiltonian=hamiltonian,
            shots=100,
            device_name="QpiAI-QSV-Local",
        )
        self.assertIsInstance(res, VQEResult)
        self.assertIsNotNone(res.optimal_energy)
        self.assertIsNotNone(res.optimal_parameters)


if __name__ == "__main__":
    unittest.main()
