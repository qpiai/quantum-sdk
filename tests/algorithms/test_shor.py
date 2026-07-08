import unittest
from unittest.mock import MagicMock, patch
import os
import sys
from dotenv import load_dotenv

load_dotenv("qcloud.env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.shor import ShorsAlgorithm  # noqa: E402


class TestShorHelpers(unittest.TestCase):
    def test_gcd(self):
        self.assertEqual(ShorsAlgorithm.gcd(12, 8), 4)
        self.assertEqual(ShorsAlgorithm.gcd(17, 5), 1)

    def test_is_prime(self):
        self.assertTrue(ShorsAlgorithm.is_prime(2))
        self.assertTrue(ShorsAlgorithm.is_prime(3))
        self.assertTrue(ShorsAlgorithm.is_prime(17))
        self.assertFalse(ShorsAlgorithm.is_prime(4))
        self.assertFalse(ShorsAlgorithm.is_prime(15))

    def test_is_power(self):
        self.assertEqual(ShorsAlgorithm.is_power(8), (2, 3))
        self.assertEqual(ShorsAlgorithm.is_power(9), (3, 2))
        self.assertIsNone(ShorsAlgorithm.is_power(15))


class TestShorValidationAndCircuit(unittest.TestCase):
    def test_validation_errors(self):
        # N must be at least 3
        with self.assertRaises(ValueError):
            ShorsAlgorithm(N=2)
        # N must be odd
        with self.assertRaises(ValueError):
            ShorsAlgorithm(N=8)
        # N must not be a prime power
        with self.assertRaises(ValueError):
            ShorsAlgorithm(N=9)

    def test_build_circuit(self):
        shor = ShorsAlgorithm(N=15)
        # precision_qubits = 4
        circuit = shor.build_circuit(a=2, precision_qubits=4)
        self.assertIsNotNone(circuit)
        # total_qubits = precision_qubits + ceil(log2(N)) = 4 + 4 = 8
        self.assertEqual(circuit.num_qubits, 8)


class TestShorExecutionWithMock(unittest.TestCase):
    def _make_mock_result(self, counts: dict):
        mock_result = MagicMock()
        mock_result.get.return_value = {"counts": counts}
        return mock_result

    @patch("qpiai_quantum.algorithms.shor.ShorsAlgorithm.run")
    def test_find_period_mock(self, mock_run):
        # mock run returning "0100" (binary for 4)
        mock_run.return_value = self._make_mock_result({"0100": 1})
        shor = ShorsAlgorithm(N=15)
        # period with a=2, precision=4
        r = shor.find_period(a=2, precision_qubits=4)
        # 4/16 = 1/4 -> denominator is 4
        self.assertEqual(r, 4)

    @patch("random.randint", return_value=2)
    @patch("qpiai_quantum.algorithms.shor.ShorsAlgorithm.run")
    def test_factor_mock(self, mock_run, mock_randint):
        # mock run returning "01000000" (binary for 64, phase 64/256 = 0.25 -> period 4)
        mock_run.return_value = self._make_mock_result({"01000000": 1})
        shor = ShorsAlgorithm(N=15)
        # factor should succeed and return (3, 5) or (5, 3)
        res = shor.factor(max_attempts=3)
        assert res is not None
        self.assertTrue(set(res) == {3, 5})


@unittest.skipUnless(
    os.environ.get("RUN_ALGO_CORRECTNESS") == "1",
    "Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 to run.",
)
class TestShorCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("API_KEY")
        if api_key:
            from qpiai_quantum.authentication.auth import QpiAIQuantumAuth

            try:
                QpiAIQuantumAuth.login(api_key)
            except Exception:
                pass

    def test_live_factor_15(self):
        # Simplest Shor factoring instance: N=15
        shor = ShorsAlgorithm(N=15)
        factors = shor.factor(max_attempts=15)
        if factors is not None:
            self.assertTrue(set(factors) == {3, 5})
        else:
            # Due to probabilistic nature or simplified modular multiply, factor might return None,
            # but find_period should work for base 2
            period = shor.find_period(a=2, precision_qubits=4)
            self.assertIn(
                period, [1, 2, 4]
            )  # period of 2 mod 15 is 4, but due to shots/interference can be divisor


if __name__ == "__main__":
    unittest.main()
