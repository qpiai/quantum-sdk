import unittest
from unittest.mock import MagicMock, patch
import os
import sys
from dotenv import load_dotenv

load_dotenv("qcloud.env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.deutsch_jozsa import DeutschJozsa
from qpiai_quantum.circuit import Circuit


class TestDJInitializationAndCircuit(unittest.TestCase):
    def test_init(self):
        dj = DeutschJozsa(num_qubits=2, oracle_type="balanced")
        self.assertEqual(dj.num_qubits, 2)
        self.assertEqual(dj.oracle_type, "balanced")

    def test_build_circuit(self):
        dj = DeutschJozsa(num_qubits=2, oracle_type="balanced")
        circuit = dj.build_circuit()
        self.assertIsNotNone(circuit)
        # We need n+1 qubits = 3 qubits
        self.assertEqual(circuit.num_qubits, 3)
        self.assertEqual(circuit.num_clbits, 2)

    def test_invalid_oracle_type(self):
        dj = DeutschJozsa(num_qubits=2, oracle_type="invalid_type")
        with self.assertRaises(ValueError):
            dj.build_circuit()

    def test_invalid_qubits(self):
        with self.assertRaises(ValueError):
            DeutschJozsa(num_qubits=0)


class TestDJTheoreticalAndInterpretation(unittest.TestCase):
    def test_get_theoretical_result(self):
        dj = DeutschJozsa(num_qubits=2, oracle_type="constant_zero")
        res = dj.get_theoretical_result()
        self.assertEqual(res["expected_result"], "constant")
        self.assertEqual(res["success_probability"], 1.0)

        dj2 = DeutschJozsa(num_qubits=2, oracle_type="balanced")
        res2 = dj2.get_theoretical_result()
        self.assertEqual(res2["expected_result"], "balanced")

    def test_interpret_result(self):
        # mock job result
        mock_res = MagicMock()
        mock_res.counts = {"00": 100}
        self.assertEqual(DeutschJozsa.interpret_result(mock_res), "constant")

        mock_res2 = MagicMock()
        mock_res2.counts = {"11": 100}
        self.assertEqual(DeutschJozsa.interpret_result(mock_res2), "balanced")


class TestDJExecutionWithMock(unittest.TestCase):
    def _make_mock_result(self, counts: dict):
        mock_result = MagicMock()
        mock_result.get.return_value = {"counts": counts}
        mock_result.counts = counts
        return mock_result

    @patch("qpiai_quantum.algorithms.deutsch_jozsa.DeutschJozsa.run")
    def test_determine_function_type_mock(self, mock_run):
        mock_run.return_value = self._make_mock_result({"00": 1})
        dj = DeutschJozsa(num_qubits=2, oracle_type="constant_zero")
        res = dj.determine_function_type()
        self.assertEqual(res, "constant")


@unittest.skipUnless(
    os.environ.get("RUN_ALGO_CORRECTNESS") == "1",
    "Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 to run.",
)
class TestDJCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("API_KEY")
        if api_key:
            from qpiai_quantum.authentication.auth import QpiAIQuantumAuth

            try:
                QpiAIQuantumAuth.login(api_key)
            except Exception:
                pass

    def test_live_determine_function_type(self):
        import uuid

        # Test balanced oracle on 2 qubits
        dj_bal = DeutschJozsa(num_qubits=2, oracle_type="balanced")
        dj_bal.build_circuit()
        dj_bal.circuit.name = f"dj_bal_{uuid.uuid4().hex[:8]}"
        res_bal = dj_bal.determine_function_type(shots=100)
        self.assertEqual(res_bal, "balanced")

        # Test constant oracle on 2 qubits
        dj_const = DeutschJozsa(num_qubits=2, oracle_type="constant_one")
        dj_const.build_circuit()
        dj_const.circuit.name = f"dj_const_{uuid.uuid4().hex[:8]}"
        res_const = dj_const.determine_function_type(shots=100)
        self.assertEqual(res_const, "constant")


if __name__ == "__main__":
    unittest.main()
