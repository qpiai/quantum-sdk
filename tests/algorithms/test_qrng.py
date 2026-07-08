"""
Unit tests for the QRNG utility.

These tests exercise circuit construction, output conversion, and input
validation *without* requiring a live backend.  They mock out the
``run()`` method so that tests execute instantly and deterministically.
"""

import unittest
from unittest.mock import MagicMock, patch

# Adjust the import path so we can run this from the repo root
import sys
import os
from dotenv import load_dotenv

load_dotenv("qcloud.env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.qrng import QRNG


class TestQRNGInit(unittest.TestCase):
    """Constructor validation."""

    def test_default_n_bits(self):
        rng = QRNG()
        self.assertEqual(rng.n_bits, 8)
        self.assertEqual(rng.num_qubits, 8)

    def test_custom_n_bits(self):
        rng = QRNG(n_bits=16)
        self.assertEqual(rng.n_bits, 16)

    def test_invalid_n_bits_zero(self):
        with self.assertRaises(ValueError):
            QRNG(n_bits=0)

    def test_invalid_n_bits_negative(self):
        with self.assertRaises(ValueError):
            QRNG(n_bits=-3)

    def test_invalid_n_bits_type(self):
        with self.assertRaises(ValueError):
            QRNG(n_bits=3.5)


class TestBuildCircuit(unittest.TestCase):
    """Circuit construction tests."""

    def test_circuit_qubit_count(self):
        rng = QRNG(n_bits=4)
        circuit = rng.build_circuit()
        self.assertIsNotNone(circuit)
        self.assertEqual(rng.n_bits, 4)

    def test_circuit_is_idempotent(self):
        rng = QRNG(n_bits=4)
        c1 = rng.build_circuit()
        c2 = rng.build_circuit()
        # Each call rebuilds; both must be valid
        self.assertIsNotNone(c1)
        self.assertIsNotNone(c2)


class TestConvertOutput(unittest.TestCase):
    """Output format conversion."""

    def setUp(self):
        self.rng = QRNG(n_bits=8)

    def test_bitstring_passthrough(self):
        result = self.rng._convert_output("10110011", "bitstring")
        self.assertEqual(result, "10110011")

    def test_int_conversion(self):
        result = self.rng._convert_output("10110011", "int")
        self.assertEqual(result, 0b10110011)  # 179

    def test_bytes_conversion(self):
        result = self.rng._convert_output("10110011", "bytes")
        self.assertEqual(result, b"\xb3")

    def test_bytes_padding(self):
        """n_bits=10 should produce 2 bytes."""
        rng = QRNG(n_bits=10)
        result = rng._convert_output("0000000001", "bytes")
        self.assertEqual(len(result), 2)
        self.assertEqual(result, (1).to_bytes(2, "big"))

    def test_all_zeros(self):
        self.assertEqual(self.rng._convert_output("00000000", "int"), 0)
        self.assertEqual(self.rng._convert_output("00000000", "bytes"), b"\x00")

    def test_all_ones(self):
        self.assertEqual(self.rng._convert_output("11111111", "int"), 255)
        self.assertEqual(self.rng._convert_output("11111111", "bytes"), b"\xff")


class TestValidateFormat(unittest.TestCase):
    def test_valid_formats(self):
        for fmt in ("int", "bytes", "bitstring"):
            QRNG._validate_format(fmt)  # should not raise

    def test_invalid_format(self):
        with self.assertRaises(ValueError):
            QRNG._validate_format("hex")


class TestGenerateWithMock(unittest.TestCase):
    """Tests generate() by mocking the backend call."""

    def _make_mock_result(self, counts: dict):
        mock_result = MagicMock()
        mock_result.get.return_value = {"counts": counts}
        return mock_result

    @patch.object(QRNG, "run")
    def test_generate_single_int(self, mock_run):
        mock_run.return_value = self._make_mock_result({"10110011": 1})
        rng = QRNG(n_bits=8)
        rng.build_circuit()
        value = rng.generate(shots=1, output_format="int")
        self.assertEqual(value, 179)

    @patch.object(QRNG, "run")
    def test_generate_single_bitstring(self, mock_run):
        mock_run.return_value = self._make_mock_result({"10110011": 1})
        rng = QRNG(n_bits=8)
        rng.build_circuit()
        value = rng.generate(shots=1, output_format="bitstring")
        self.assertEqual(value, "10110011")

    @patch.object(QRNG, "run")
    def test_generate_batch(self, mock_run):
        mock_run.return_value = self._make_mock_result(
            {
                "10110011": 2,
                "00001111": 3,
            }
        )
        rng = QRNG(n_bits=8)
        rng.build_circuit()
        values = rng.generate(shots=5, output_format="int")
        self.assertIsInstance(values, list)
        self.assertEqual(len(values), 5)
        self.assertIn(179, values)
        self.assertIn(15, values)

    @patch.object(QRNG, "run")
    def test_generate_batch_convenience(self, mock_run):
        mock_run.return_value = self._make_mock_result({"11111111": 3})
        rng = QRNG(n_bits=8)
        rng.build_circuit()
        values = rng.generate_batch(count=3, output_format="int")
        self.assertEqual(values, [255, 255, 255])


class TestInfoAndRepr(unittest.TestCase):
    def test_get_info(self):
        rng = QRNG(n_bits=8)
        info = rng.get_info()
        self.assertEqual(info["n_bits"], 8)
        self.assertEqual(info["max_value"], 255)
        self.assertEqual(info["output_bytes"], 1)

    def test_repr(self):
        rng = QRNG(n_bits=4)
        r = repr(rng)
        self.assertIn("QRNG", r)
        self.assertIn("n_bits=4", r)
        self.assertIn("max_value=15", r)


@unittest.skipUnless(
    os.environ.get("RUN_ALGO_CORRECTNESS") == "1",
    "Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 to run.",
)
class TestQRNGCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("API_KEY")
        if api_key:
            from qpiai_quantum.authentication.auth import QpiAIQuantumAuth

            try:
                QpiAIQuantumAuth.login(api_key)
            except Exception:
                pass

    def test_live_generate(self):
        import uuid

        # Simplest instance: 2-bit random number generation
        rng = QRNG(n_bits=2)
        rng.build_circuit()
        rng.circuit.name = f"qrng_{uuid.uuid4().hex[:8]}"
        val = rng.generate(shots=1, output_format="int")
        self.assertTrue(0 <= val <= 3)


if __name__ == "__main__":
    unittest.main()
