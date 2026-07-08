import unittest
from unittest.mock import MagicMock, patch
import os
import sys
from dotenv import load_dotenv

load_dotenv("qcloud.env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.simon import SimonAlgorithm
from qpiai_quantum.circuit import Circuit


class TestSimonInitializationAndCircuit(unittest.TestCase):
    def test_init(self):
        simon = SimonAlgorithm(num_qubits=3, hidden_string="110")
        self.assertEqual(simon.num_qubits, 3)
        self.assertEqual(simon.hidden_string, "110")

    def test_build_circuit(self):
        simon = SimonAlgorithm(num_qubits=3, hidden_string="110")
        circuit = simon.build_circuit()
        self.assertIsNotNone(circuit)
        # We need 2 * n qubits = 6 qubits
        self.assertEqual(circuit.num_qubits, 6)

    def test_invalid_hidden_string_length(self):
        simon = SimonAlgorithm(num_qubits=3, hidden_string="11")
        with self.assertRaises(ValueError):
            simon.build_circuit()


class TestSimonSolver(unittest.TestCase):
    def test_linear_equations_solver(self):
        simon = SimonAlgorithm(num_qubits=3, hidden_string="110")
        # For hidden string "110", we need measurements y such that s . y = 0
        # Examples of y: "001", "110"
        # y = "001": 1*0 + 1*0 + 0*1 = 0
        # y = "110": 1*1 + 1*1 + 0*0 = 0 (mod 2)
        measurements = ["001", "110"]
        s = simon._solve_for_hidden_string(measurements)
        self.assertEqual(s, "110")

    def test_solve_zero_string(self):
        simon = SimonAlgorithm(num_qubits=3, hidden_string="000")
        # All columns pivots -> s = 000
        measurements = ["100", "010", "001"]
        s = simon._solve_for_hidden_string(measurements)
        self.assertEqual(s, "000")


class TestSimonExecutionWithMock(unittest.TestCase):
    def _make_mock_result(self, counts: dict):
        mock_result = MagicMock()
        mock_result.get.return_value = {"counts": counts}
        return mock_result

    @patch("qpiai_quantum.algorithms.simon.SimonAlgorithm.run")
    def test_find_hidden_string_mock(self, mock_run):
        # We need to mock successive runs returning different bitstrings
        # Let's say s = "10"
        # Equations y can be "01" (since 1*0 + 0*1 = 0)
        # So we mock run to return "01"
        mock_run.return_value = self._make_mock_result({"01": 1})
        simon = SimonAlgorithm(num_qubits=2, hidden_string="10")
        res = simon.find_hidden_string(max_attempts=5)
        self.assertEqual(res, "10")


@unittest.skipUnless(
    os.environ.get("RUN_ALGO_CORRECTNESS") == "1",
    "Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 to run.",
)
class TestSimonCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("API_KEY")
        if api_key:
            from qpiai_quantum.authentication.auth import QpiAIQuantumAuth

            try:
                QpiAIQuantumAuth.login(api_key)
            except Exception:
                pass

    def test_live_find_hidden_string_2_qubits(self):
        # Simplest Simon instance: 2 qubits, hidden string "11"
        simon = SimonAlgorithm(num_qubits=2, hidden_string="11")
        res = simon.find_hidden_string(max_attempts=10)
        self.assertEqual(res, "11")


if __name__ == "__main__":
    unittest.main()
