import unittest
from unittest.mock import MagicMock, patch
import os
import sys
from dotenv import load_dotenv

load_dotenv("qcloud.env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.phase_estimation import QuantumPhaseEstimation
from qpiai_quantum.circuit import Circuit


class TestQPEInitializationAndCircuit(unittest.TestCase):
    def test_init(self):
        qpe = QuantumPhaseEstimation(precision_qubits=3, eigenstate_qubits=1)
        self.assertEqual(qpe.precision_qubits, 3)
        self.assertEqual(qpe.eigenstate_qubits, 1)
        self.assertEqual(qpe.num_qubits, 4)

    def test_build_circuit(self):
        qpe = QuantumPhaseEstimation(precision_qubits=3, eigenstate_qubits=1)
        circuit = qpe.build_circuit(unitary="T")
        self.assertIsNotNone(circuit)
        self.assertEqual(circuit.num_qubits, 4)
        self.assertEqual(circuit.num_clbits, 3)


class TestQPEMathematicalHelpers(unittest.TestCase):
    def test_get_theoretical_phase(self):
        qpe = QuantumPhaseEstimation(precision_qubits=3, eigenstate_qubits=1)
        self.assertEqual(qpe.get_theoretical_phase("T"), 0.125)
        self.assertEqual(qpe.get_theoretical_phase("S"), 0.25)
        self.assertEqual(qpe.get_theoretical_phase("Z"), 0.5)
        self.assertEqual(qpe.get_theoretical_phase("Unknown"), 0.0)


class TestQPEExecutionWithMock(unittest.TestCase):
    def _make_mock_result(self, counts: dict):
        mock_result = MagicMock()
        mock_result.get.return_value = {"counts": counts}
        return mock_result

    @patch("qpiai_quantum.algorithms.phase_estimation.QuantumPhaseEstimation.run")
    def test_estimate_phase_mock(self, mock_run):
        # mock run returning "010" (binary 2, phase 2/8 = 0.25)
        mock_run.return_value = self._make_mock_result({"010": 1})
        qpe = QuantumPhaseEstimation(precision_qubits=3, eigenstate_qubits=1)
        phase = qpe.estimate_phase(unitary="S")
        self.assertEqual(phase, 0.25)


@unittest.skipUnless(
    os.environ.get("RUN_ALGO_CORRECTNESS") == "1",
    "Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 to run.",
)
class TestQPECorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("API_KEY")
        if api_key:
            from qpiai_quantum.authentication.auth import QpiAIQuantumAuth

            try:
                QpiAIQuantumAuth.login(api_key)
            except Exception:
                pass

    def test_live_estimate_phase_t_gate(self):
        import uuid

        # Simplest QPE instance: estimate T-gate phase with 3 precision qubits
        # e^(i * pi / 4) -> phase is 1/8 = 0.125
        qpe = QuantumPhaseEstimation(precision_qubits=3, eigenstate_qubits=1)
        qpe.build_circuit(unitary="T")
        qpe.circuit.name = f"qpe_{uuid.uuid4().hex[:8]}"

        # Monkey patch build_circuit so estimate_phase doesn't recreate/overwrite named circuit
        original_build = qpe.build_circuit
        qpe.build_circuit = lambda *args, **kwargs: qpe.circuit
        try:
            phase = qpe.estimate_phase(unitary="T")
        finally:
            qpe.build_circuit = original_build
        self.assertEqual(phase, 0.125)


if __name__ == "__main__":
    unittest.main()
