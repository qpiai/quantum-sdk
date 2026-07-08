import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()

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
    def test_count_parameters(self):
        vqe = VQESolver(n_qubits=2, ansatz="standard")
        n_params = vqe._count_parameters()
        self.assertGreater(n_params, 0)

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


@unittest.skipUnless(
    os.environ.get("RUN_ALGO_CORRECTNESS") == "1",
    "Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 to run.",
)
class TestVQECorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("API_KEY")
        if api_key:
            from qpiai_quantum.authentication.auth import QpiAIQuantumAuth

            try:
                QpiAIQuantumAuth.login(api_key)
            except Exception:
                pass

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
