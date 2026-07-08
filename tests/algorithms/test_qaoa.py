import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.opt.solvers.qaoa import QAOASolver, QAOAResult
from qpiai_quantum.circuit import Circuit


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


class TestQAOAHelpers(unittest.TestCase):
    def test_count_parameters(self):
        qaoa = QAOASolver(layers=1)
        qaoa.problem = DummyProblem()
        n_params = qaoa._count_parameters()
        self.assertEqual(
            n_params, 5
        )  # Standard QAOA has 3 cost terms + 2 mixer terms = 5 parameters


@unittest.skipUnless(
    os.environ.get("RUN_ALGO_CORRECTNESS") == "1",
    "Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 to run.",
)
class TestQAOACorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("API_KEY")
        if api_key:
            from qpiai_quantum.authentication.auth import QpiAIQuantumAuth

            try:
                QpiAIQuantumAuth.login(api_key)
            except Exception:
                pass

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
