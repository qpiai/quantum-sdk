import unittest
import os
import sys
from dotenv import load_dotenv

load_dotenv("qcloud.env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.grover import GroverSearch
from qpiai_quantum.circuit import Circuit


class TestGroverInitializationAndCircuit(unittest.TestCase):
    def test_init(self):
        grover = GroverSearch(num_qubits=3, target="101")
        self.assertEqual(grover.num_qubits, 3)
        self.assertEqual(grover.target, "101")

    def test_invalid_target_length(self):
        grover = GroverSearch(num_qubits=3, target="11")
        with self.assertRaises(ValueError):
            grover.build_circuit()

    def test_target_missing(self):
        grover = GroverSearch(num_qubits=3)
        with self.assertRaises(ValueError):
            grover.build_circuit()

    def test_build_circuit(self):
        grover = GroverSearch(num_qubits=2, target="10")
        circuit = grover.build_circuit(iterations=1)
        self.assertIsNotNone(circuit)
        self.assertEqual(circuit.num_qubits, 2)
        self.assertEqual(circuit.num_clbits, 2)


class TestGroverMath(unittest.TestCase):
    def test_success_probability(self):
        # For N=4 (2 qubits), theta = arcsin(1/2) = pi/6
        # With 1 iteration: sin^2(3 * pi/6) = sin^2(pi/2) = 1.0
        grover = GroverSearch(num_qubits=2, target="10")
        prob = grover.get_success_probability(iterations=1)
        self.assertAlmostEqual(prob, 1.0, places=5)


class TestGroverCorrectness(unittest.TestCase):
    def test_local_find_target_11(self):
        grover = GroverSearch(num_qubits=2, target="11")
        circuit = grover.build_circuit(iterations=1)
        result = circuit.run(device_name="QpiAI-QSV-Local", shots=100)
        counts = result.get()["counts"]
        self.assertIn("11", counts)
        self.assertEqual(counts["11"], 100)

    def test_local_find_target_111(self):
        grover = GroverSearch(num_qubits=3, target="111")
        circuit = grover.build_circuit(iterations=2)
        result = circuit.run(device_name="QpiAI-QSV-Local", shots=100)
        counts = result.get()["counts"]
        self.assertIn("111", counts)
        self.assertGreater(counts["111"], 50)

    def test_local_find_target_1111(self):
        grover = GroverSearch(num_qubits=4, target="1111")
        circuit = grover.build_circuit(iterations=3)
        result = circuit.run(device_name="QpiAI-QSV-Local", shots=100)
        counts = result.get()["counts"]
        self.assertIn("1111", counts)
        self.assertGreater(counts["1111"], 50)

    def test_local_find_target_11111(self):
        grover = GroverSearch(num_qubits=5, target="11111")
        circuit = grover.build_circuit(iterations=4)
        result = circuit.run(device_name="QpiAI-QSV-Local", shots=100)
        counts = result.get()["counts"]
        self.assertIn("11111", counts)
        self.assertGreater(counts["11111"], 50)


if __name__ == "__main__":
    unittest.main()
