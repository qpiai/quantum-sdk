import unittest
import numpy as np

from qpiai_quantum.circuit import Circuit
from qpiai_quantum.simulator.statevector import StatevectorSimulator, NUMBA_AVAILABLE


class TestNumbaStatevectorSimulator(unittest.TestCase):
    def test_numba_available_flag(self):
        """Verify that Numba is available in the test environment."""
        self.assertTrue(NUMBA_AVAILABLE, "Numba should be installed and enabled.")

    def test_bell_state_numba(self):
        """Test Bell state creation using Numba-accelerated StatevectorSimulator."""
        qc = Circuit(2)
        qc.h(0)
        qc.cx(0, 1)

        sim = StatevectorSimulator()
        res = sim.run(qc, shots=1000)

        # Expected statevector: 1/sqrt(2) (|00> + |11>)
        expected_sv = np.array(
            [1.0 / np.sqrt(2), 0.0, 0.0, 1.0 / np.sqrt(2)], dtype=complex
        )
        np.testing.assert_array_almost_equal(res.statevector, expected_sv)

    def test_dynamic_ram_check(self):
        """Test that simulating an excessively large qubit count raises MemoryError."""
        sim = StatevectorSimulator()
        qc = Circuit(60)
        with self.assertRaises(MemoryError) as ctx:
            sim.run(qc)
        self.assertIn("Insufficient available RAM", str(ctx.exception))

    def test_numba_vs_numpy_equivalence(self):
        """Compare Numba multi-core simulation against NumPy fallback baseline across random circuits."""
        for n_qubits in [2, 4, 6, 8]:
            qc = Circuit(n_qubits)
            for i in range(n_qubits):
                qc.h(i)
                qc.rz(i, 0.45 * (i + 1))
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
                qc.rx(i + 1, 0.2 * (i + 1))

            sim = StatevectorSimulator()

            # Execute via Numba path (default)
            res_numba = sim.run(qc, shots=100)

            # Temporarily force NumPy path
            import qpiai_quantum.simulator.statevector as sv_mod

            original_flag = sv_mod.NUMBA_AVAILABLE
            try:
                sv_mod.NUMBA_AVAILABLE = False
                res_numpy = sim.run(qc, shots=100)
            finally:
                sv_mod.NUMBA_AVAILABLE = original_flag

            np.testing.assert_array_almost_equal(
                res_numba.statevector,
                res_numpy.statevector,
                decimal=12,
                err_msg=f"Numba statevector mismatch at n_qubits={n_qubits}",
            )


if __name__ == "__main__":
    unittest.main()
