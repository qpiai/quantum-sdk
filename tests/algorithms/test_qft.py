import unittest
import os
import sys
from dotenv import load_dotenv

load_dotenv("qcloud.env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.qft import QFT
from qpiai_quantum.circuit import Circuit


class TestQFTInitializationAndCircuit(unittest.TestCase):
    def test_init(self):
        qft = QFT(num_qubits=3)
        self.assertEqual(qft.num_qubits, 3)
        self.assertFalse(qft.inverse)

        iqft = QFT(num_qubits=3, inverse=True)
        self.assertTrue(iqft.inverse)

    def test_build_circuit(self):
        qft = QFT(num_qubits=3)
        circuit = qft.build_circuit(measure=False)
        self.assertIsNotNone(circuit)
        self.assertEqual(circuit.num_qubits, 3)
        self.assertEqual(circuit.num_clbits, 0)

        # QFT on 3 qubits has:
        # q3: H, CP(1, 2), CP(0, 2)
        # q2: H, CP(0, 1)
        # q1: H
        # Total gates: 3 Hadamards + 3 controlled phase rotations = 6 gates
        stats = circuit.list_gates()
        self.assertEqual(stats["total_gates"], 6)

    def test_apply_qft_to_circuit(self):
        circ = Circuit(4)
        QFT.apply_qft_to_circuit(circ, start=1, n=3)
        # Applying QFT to qubits 1, 2, 3 in a 4-qubit circuit
        # Total gates should be 3 Hadamards + 3 CP rotations = 6 gates
        stats = circ.list_gates()
        self.assertEqual(stats["total_gates"], 6)


@unittest.skipUnless(
    os.environ.get("RUN_ALGO_CORRECTNESS") == "1",
    "Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 to run.",
)
class TestQFTCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api_key = os.getenv("API_KEY")
        if api_key:
            from qpiai_quantum.authentication.auth import QpiAIQuantumAuth

            try:
                QpiAIQuantumAuth.login(api_key)
            except Exception:
                pass

    def test_live_qft_inverse_identity(self):
        # QFT followed by inverse QFT should equal identity.
        # Initialize to |00⟩, apply QFT, apply Inverse QFT, measure.
        # Result should be "00" with probability 1.0.
        circ = Circuit(2, 2)
        QFT.apply_qft_to_circuit(circ, start=0, n=2)
        QFT.apply_inverse_qft_to_circuit(circ, start=0, n=2)
        circ.measure(0, 0)
        circ.measure(1, 1)

        import uuid

        circ.name = f"qft_{uuid.uuid4().hex[:8]}"
        result = circ.run(shots=100)
        counts = result.get()["counts"]
        self.assertEqual(len(counts), 1)
        self.assertIn("00", counts)
        self.assertEqual(counts["00"], 100)


if __name__ == "__main__":
    unittest.main()
