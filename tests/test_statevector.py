import unittest
import sys
import os
import numpy as np

# Ensure qpiai_quantum is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qpiai_quantum.quantum_info.statevector import Statevector
from qpiai_quantum.quantum_info.density_matrix import DensityMatrix
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.simulator.statevector import StatevectorSimulator


class TestStatevector(unittest.TestCase):
    def test_init_from_array(self):
        """Test Statevector initialization from array/list and automatic normalization."""
        # Non-normalized input
        data = [1.0, 1.0j]
        sv = Statevector(data)

        # Verify length and attributes
        self.assertEqual(sv.num_qubits, 1)
        self.assertEqual(len(sv.data), 2)

        # Verify normalization (norm must be 1.0)
        norm = np.linalg.norm(sv.data)
        self.assertAlmostEqual(norm, 1.0)
        self.assertAlmostEqual(sv.data[0], 1.0 / np.sqrt(2))
        self.assertAlmostEqual(sv.data[1], 1.0j / np.sqrt(2))

    def test_init_copy(self):
        """Test copy constructor of Statevector."""
        sv1 = Statevector([1.0, 0.0])
        sv2 = Statevector(sv1)

        self.assertEqual(sv1.num_qubits, sv2.num_qubits)
        np.testing.assert_array_almost_equal(sv1.data, sv2.data)

        # Ensure it is a copy, not a reference
        sv2.data[0] = 0.0
        self.assertEqual(sv1.data[0], 1.0)

    def test_init_invalid_length(self):
        """Test that non-power-of-2 statevector length raises ValueError."""
        with self.assertRaises(ValueError):
            Statevector([1.0, 0.0, 0.0])

    def test_to_density_matrix(self):
        """Test conversion to DensityMatrix."""
        sv = Statevector([1.0, 0.0])
        dm = sv.to_density_matrix()

        self.assertIsInstance(dm, DensityMatrix)
        expected = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        np.testing.assert_array_almost_equal(dm.data, expected)

    def test_evolve_unitary(self):
        """Test evolution of Statevector by a unitary matrix."""
        # Start in state |0>
        sv = Statevector([1.0, 0.0])

        # Pauli-X unitary
        X = np.array([[0, 1], [1, 0]], dtype=complex)

        # Evolve state
        sv_new = sv.evolve(X)

        # Result should be |1>
        np.testing.assert_array_almost_equal(sv_new.data, [0.0, 1.0])

    def test_evolve_circuit(self):
        """Test evolution of Statevector by a Circuit."""
        # Start in state |0>
        sv = Statevector([1.0, 0.0])

        # Apply Hadamard circuit
        qc = Circuit(1)
        qc.h(0)

        sv_plus = sv.evolve(qc)

        # Result should be |+> = [1/sqrt(2), 1/sqrt(2)]
        expected = [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
        np.testing.assert_array_almost_equal(sv_plus.data, expected)

        # Evolve further with an X gate circuit
        qc2 = Circuit(1)
        qc2.x(0)

        sv_final = sv_plus.evolve(qc2)

        # |+> is symmetric under X: X|+> = |+>
        np.testing.assert_array_almost_equal(sv_final.data, expected)

    def test_evolve_multi_qubit_circuit(self):
        """Test evolution of a multi-qubit statevector by a Circuit."""
        # Start in state |01> = [0, 1, 0, 0]
        sv = Statevector([0.0, 1.0, 0.0, 0.0])

        # Apply CNOT where control is qubit 1 and target is qubit 0.
        # Note: state order is qubit 0 (least significant bit) and qubit 1 (most significant).
        # |01> (qubit 0 is 1, qubit 1 is 0).
        qc = Circuit(2)
        qc.cx(1, 0)  # Control qubit 1, Target qubit 0

        sv_new = sv.evolve(qc)

        # Since control (qubit 1) is 0, target (qubit 0) remains unchanged.
        np.testing.assert_array_almost_equal(sv_new.data, [0.0, 1.0, 0.0, 0.0])

        # Apply CNOT with control qubit 0, target qubit 1
        qc2 = Circuit(2)
        qc2.cx(0, 1)  # Control qubit 0 (is 1), Target qubit 1 (flips from 0 to 1)

        sv_final = sv_new.evolve(qc2)

        # Result should be |11> = [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(sv_final.data, [0.0, 0.0, 0.0, 1.0])

    def test_evolve_mismatched_dimensions(self):
        """Test that evolving by a circuit of mismatched size raises ValueError."""
        # 1-qubit statevector
        sv = Statevector([1.0, 0.0])

        # 2-qubit circuit
        qc = Circuit(2)
        qc.h(0)

        with self.assertRaises(ValueError):
            sv.evolve(qc)

    def test_simulator_unnormalized_initial_state(self):
        """Test that simulator raises ValueError if initial_state is not normalized."""
        sim = StatevectorSimulator()
        qc = Circuit(1)
        qc.h(0)

        # Unnormalized statevector
        with self.assertRaises(ValueError):
            sim.run(qc, initial_state=np.array([1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
