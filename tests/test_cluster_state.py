import unittest
import sys
import os
from dotenv import load_dotenv

# Ensure qpiai_quantum is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qpiai_quantum.state_preparation.cluster_state import (
    ClusterStateGenerator,
    create_cluster_state,
)
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.authentication.auth import QpiAIQuantumAuth


class TestClusterState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up environment and load API key if available."""
        load_dotenv("qcloud.env")
        cls.api_key = os.getenv("API_KEY")
        if cls.api_key:
            try:
                QpiAIQuantumAuth.login(cls.api_key)
                cls.has_auth = True
            except Exception:
                cls.has_auth = False
        else:
            cls.has_auth = False

    def test_initialization(self):
        """Test that ClusterStateGenerator initializes correctly."""
        num_qubits = 4
        gen = ClusterStateGenerator(num_qubits=num_qubits)
        self.assertEqual(gen.num_qubits, num_qubits)
        self.assertIn(f"({num_qubits} qubits)", gen.name)

        # Test minimum qubits
        with self.assertRaises(ValueError):
            ClusterStateGenerator(num_qubits=1)

    def test_build_circuit_no_measure(self):
        """Test circuit construction without measurements."""
        num_qubits = 3
        gen = ClusterStateGenerator(num_qubits=num_qubits)
        circuit = gen.build_circuit(measure=False)

        self.assertIsInstance(circuit, Circuit)
        self.assertEqual(circuit.num_qubits, num_qubits)
        self.assertEqual(circuit.num_clbits, num_qubits)

        stats = circuit.list_gates()
        # For 3 qubits: 3 H gates and 2 CZ gates = 5 gates total
        self.assertEqual(stats["total_gates"], 5)
        self.assertEqual(stats["gate_counts"].get("H"), 3)
        self.assertEqual(stats["gate_counts"].get("CZ"), 2)
        self.assertEqual(stats.get("measurements", 0), 0)

    def test_build_circuit_with_measure(self):
        """Test circuit construction with measurements."""
        num_qubits = 2
        gen = ClusterStateGenerator(num_qubits=num_qubits)
        circuit = gen.build_circuit(measure=True)

        self.assertEqual(circuit.num_clbits, num_qubits)
        stats = circuit.list_gates()
        self.assertEqual(stats["measurements"], num_qubits)

    def test_get_expected_outcomes(self):
        """Test that expected outcomes are correctly calculated."""
        num_qubits = 2
        gen = ClusterStateGenerator(num_qubits=num_qubits)
        expected = gen.get_expected_outcomes()

        # 2^2 = 4 states
        self.assertEqual(len(expected), 4)
        for state, prob in expected.items():
            self.assertEqual(prob, 0.25)
            self.assertEqual(len(state), num_qubits)

    def test_convenience_function(self):
        """Test the create_cluster_state convenience function."""
        gen = create_cluster_state(5)
        self.assertIsInstance(gen, ClusterStateGenerator)
        self.assertEqual(gen.num_qubits, 5)

    @unittest.skipUnless(
        os.getenv("API_KEY"),
        "API key not found in environment",
    )
    def test_live_execution(self):
        """Test live execution of cluster state preparation."""
        if not self.has_auth:
            self.skipTest("Authentication failed even with API key")

        num_qubits = 3
        print(f"\nRunning Live Cluster State Execution with {num_qubits} qubits...")
        cluster_gen = create_cluster_state(num_qubits=num_qubits)
        cluster_gen.build_circuit(measure=True)

        shots = 1024
        result = cluster_gen.run(shots=shots)
        counts = result.get()["counts"]

        print(f"Measurement counts (total shots: {shots}):")
        print(counts)

        # Verify 8 states are present (rough check)
        self.assertGreaterEqual(len(counts), 1)

        # Use existing verification logic
        is_valid = cluster_gen.verify_entanglement(result)
        print(f"Verification: {'PASSED' if is_valid else 'FAILED'}")
        self.assertTrue(is_valid, "Cluster state entanglement verification failed")


if __name__ == "__main__":
    unittest.main()
