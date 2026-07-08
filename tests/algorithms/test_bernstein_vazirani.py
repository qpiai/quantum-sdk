import unittest
from unittest.mock import MagicMock, patch
import os
import sys
from dotenv import load_dotenv

load_dotenv("qcloud.env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.bernstein_vazirani import BernsteinVazirani
from qpiai_quantum.circuit import Circuit


class TestBVInitializationAndCircuit(unittest.TestCase):
    def test_init(self):
        bv = BernsteinVazirani(num_qubits=3, hidden_string="101")
        self.assertEqual(bv.num_qubits, 3)
        self.assertEqual(bv.hidden_string, "101")

    def test_build_circuit(self):
        bv = BernsteinVazirani(num_qubits=3, hidden_string="101")
        circuit = bv.build_circuit()
        self.assertIsNotNone(circuit)
        # We need n+1 qubits = 4 qubits
        self.assertEqual(circuit.num_qubits, 4)
        self.assertEqual(circuit.num_clbits, 3)

    def test_invalid_hidden_string_length(self):
        bv = BernsteinVazirani(num_qubits=3, hidden_string="11")
        with self.assertRaises(ValueError):
            bv.build_circuit()

    def test_invalid_hidden_string_chars(self):
        bv = BernsteinVazirani(num_qubits=3, hidden_string="10a")
        with self.assertRaises(ValueError):
            bv.build_circuit()


class TestBVTheoretical(unittest.TestCase):
    def test_get_theoretical_result(self):
        bv = BernsteinVazirani(num_qubits=3, hidden_string="101")
        res = bv.get_theoretical_result()
        self.assertEqual(res["hidden_string"], "101")
        self.assertEqual(res["success_probability"], 1.0)
        self.assertEqual(res["num_oracle_queries"], 1)


class TestBVExecutionWithMock(unittest.TestCase):
    def _make_mock_result(self, counts: dict):
        mock_result = MagicMock()
        mock_result.get.return_value = {"counts": counts}
        return mock_result

    @patch("qpiai_quantum.algorithms.bernstein_vazirani.BernsteinVazirani.run")
    def test_find_hidden_string_mock(self, mock_run):
        mock_run.return_value = self._make_mock_result({"101": 1})
        bv = BernsteinVazirani(num_qubits=3, hidden_string="101")
        res = bv.find_hidden_string()
        self.assertEqual(res, "101")


@unittest.skipUnless(
    os.environ.get("RUN_ALGO_CORRECTNESS") == "1",
    "Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 to run.",
)
class TestBVCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("API_KEY")
        if api_key:
            from qpiai_quantum.authentication.auth import QpiAIQuantumAuth

            try:
                QpiAIQuantumAuth.login(api_key)
            except Exception:
                pass

    def test_live_find_hidden_string_101(self):
        import uuid

        # Simplest BV instance: 3 qubits, hidden string "101"
        bv = BernsteinVazirani(num_qubits=3, hidden_string="101")
        bv.build_circuit()
        bv.circuit.name = f"bv_{uuid.uuid4().hex[:8]}"
        res = bv.find_hidden_string(shots=100)
        self.assertEqual(res, "101")


if __name__ == "__main__":
    unittest.main()
