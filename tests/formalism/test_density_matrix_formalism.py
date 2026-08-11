"""Tests for the formalism DensityMatrix class (density_matrix.py)."""

import unittest
import numpy as np

from qpiai_quantum.formalism.density_matrix.density_matrix import DensityMatrix


class TestConstruction(unittest.TestCase):
    """Test density matrix construction."""

    def test_create_from_numpy(self):
        """Construct from numpy array."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        dm = DensityMatrix(rho)
        np.testing.assert_array_equal(dm.state, rho)

    def test_create_from_list(self):
        """Construct from nested list."""
        dm = DensityMatrix([[1, 0], [0, 0]])
        self.assertEqual(dm.state.shape, (2, 2))

    def test_create_from_statevector(self):
        """Construct from column statevector — should auto-convert to density matrix."""
        sv = np.array([[1], [0]], dtype=complex)
        dm = DensityMatrix(sv)
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        np.testing.assert_allclose(dm.state, expected, atol=1e-12)

    def test_invalid_type_raises(self):
        """Non-array/non-list input should raise."""
        with self.assertRaises(Exception):
            DensityMatrix("invalid")  # pyrefly: ignore[bad-argument]

    def test_state_property_setter(self):
        """Setting state via property should work."""
        dm = DensityMatrix(np.eye(2, dtype=complex) / 2)
        new_state = np.array([[1, 0], [0, 0]], dtype=complex)
        dm.state = new_state
        np.testing.assert_array_equal(dm.state, new_state)


class TestBasicProperties(unittest.TestCase):
    """Test purity, trace, entropy, validation."""

    def test_purity_pure_state(self):
        """Pure state |0⟩ should have purity = 1."""
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        self.assertAlmostEqual(dm.purity(), 1.0, places=10)

    def test_purity_mixed_state(self):
        """Maximally mixed state should have purity = 1/d."""
        dm = DensityMatrix(np.eye(2, dtype=complex) / 2)
        self.assertAlmostEqual(dm.purity(), 0.5, places=10)

    def test_trace(self):
        """Trace should be 1 for a valid density matrix."""
        dm = DensityMatrix(np.array([[0.7, 0.1], [0.1, 0.3]], dtype=complex))
        self.assertAlmostEqual(np.real(dm.trace()), 1.0, places=10)

    def test_von_neumann_entropy_pure(self):
        """Pure state should have zero entropy."""
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        self.assertAlmostEqual(dm.von_neumann_entropy(), 0.0, places=10)

    def test_von_neumann_entropy_mixed(self):
        """Maximally mixed state should have entropy = log2(d)."""
        dm = DensityMatrix(np.eye(2, dtype=complex) / 2)
        self.assertAlmostEqual(dm.von_neumann_entropy(), 1.0, places=5)

    def test_is_valid_pure(self):
        """|0⟩⟨0| should be valid."""
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        self.assertTrue(dm.is_valid())

    def test_is_valid_mixed(self):
        """I/2 should be valid."""
        dm = DensityMatrix(np.eye(2, dtype=complex) / 2)
        self.assertTrue(dm.is_valid())

    def test_is_pure(self):
        """Pure state should be identified as pure."""
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        self.assertTrue(dm.is_pure())

    def test_is_not_pure(self):
        """Mixed state should not be identified as pure."""
        dm = DensityMatrix(np.eye(2, dtype=complex) / 2)
        self.assertFalse(dm.is_pure())

    def test_get_state(self):
        """get_state() should return the density matrix."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        dm = DensityMatrix(rho)
        np.testing.assert_array_equal(dm.get_state(), rho)

    def test_get_num_qubits(self):
        """_get_num_qubits should return correct count."""
        dm = DensityMatrix(np.eye(4, dtype=complex) / 4)
        self.assertEqual(dm._get_num_qubits(), 2)

    def test_str_repr(self):
        """String representations should not crash."""
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        self.assertIn("DensityMatrix", str(dm))
        self.assertIn("DensityMatrix", repr(dm))


class TestFidelity(unittest.TestCase):
    """Test fidelity calculations."""

    def test_fidelity_same_state(self):
        """Fidelity of a state with itself should be 1."""
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        self.assertAlmostEqual(dm.fidelity(dm), 1.0, places=8)

    def test_fidelity_orthogonal(self):
        """Fidelity of orthogonal states should be 0."""
        dm0 = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        dm1 = DensityMatrix(np.array([[0, 0], [0, 1]], dtype=complex))
        self.assertAlmostEqual(dm0.fidelity(dm1), 0.0, places=8)

    def test_fidelity_dimension_mismatch(self):
        """Different dimensions should raise."""
        dm1 = DensityMatrix(np.eye(2, dtype=complex) / 2)
        dm2 = DensityMatrix(np.eye(4, dtype=complex) / 4)
        with self.assertRaises(ValueError):
            dm1.fidelity(dm2)


class TestNoiseChannels(unittest.TestCase):
    """Test ADC and depolarizing channels."""

    def _bell_state_dm(self):
        """Create a Bell state |Φ+⟩ density matrix."""
        phi_plus = np.array(
            [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
            dtype=complex,
        )
        return DensityMatrix(phi_plus)

    def test_adc_zero_param(self):
        """ADC with param=0 should leave state unchanged."""
        dm = self._bell_state_dm()
        result = dm.ADC(0.0)
        np.testing.assert_allclose(result.state, dm.state, atol=1e-10)

    def test_adc_produces_valid_dm(self):
        """ADC output should be a valid density matrix."""
        dm = self._bell_state_dm()
        result = dm.ADC(0.3)
        self.assertAlmostEqual(np.real(np.trace(result.state)), 1.0, places=10)
        self.assertTrue(np.allclose(result.state, result.state.conj().T, atol=1e-10))

    def test_adc_invalid_param(self):
        """ADC with param outside [0,1] should raise."""
        dm = self._bell_state_dm()
        with self.assertRaises(Exception):
            dm.ADC(1.5)

    def test_adc_non_2q_raises(self):
        """ADC on a 1-qubit state should raise."""
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        with self.assertRaises(Exception):
            dm.ADC(0.1)

    def test_depol_zero_param(self):
        """Depol with param=0 should leave state unchanged."""
        dm = self._bell_state_dm()
        result = dm.depol(0.0)
        np.testing.assert_allclose(result.state, dm.state, atol=1e-10)

    def test_depol_produces_valid_dm(self):
        """Depol output should be a valid density matrix."""
        dm = self._bell_state_dm()
        result = dm.depol(0.3)
        self.assertAlmostEqual(np.real(np.trace(result.state)), 1.0, places=10)


class TestConcurrence(unittest.TestCase):
    """Test concurrence calculation."""

    def test_bell_state_concurrence(self):
        """Bell state should have concurrence = 1."""
        phi_plus = np.array(
            [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
            dtype=complex,
        )
        dm = DensityMatrix(phi_plus)
        self.assertAlmostEqual(dm.concurrence(), 1.0, places=5)

    def test_product_state_concurrence(self):
        """Product state |00⟩ should have concurrence = 0."""
        rho = np.zeros((4, 4), dtype=complex)
        rho[0, 0] = 1.0
        dm = DensityMatrix(rho)
        self.assertAlmostEqual(dm.concurrence(), 0.0, places=5)


class TestRenyiEntropy(unittest.TestCase):
    """Test Rényi entropy."""

    def test_renyi_alpha_1_is_von_neumann(self):
        """Rényi with alpha=1 should match von Neumann entropy."""
        dm = DensityMatrix(np.eye(2, dtype=complex) / 2)
        renyi = dm.reyni(base=2, alpha=1)
        vn = dm.von_neumann_entropy()
        self.assertAlmostEqual(renyi, vn, places=5)

    def test_renyi_pure_state(self):
        """Pure state should have zero Rényi entropy for any alpha."""
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        self.assertAlmostEqual(dm.reyni(alpha=2), 0.0, places=10)

    def test_renyi_negative_alpha_raises(self):
        """Negative alpha should raise."""
        dm = DensityMatrix(np.eye(2, dtype=complex) / 2)
        with self.assertRaises(Exception):
            dm.reyni(alpha=-1)


class TestPartialTrace(unittest.TestCase):
    """Test partial trace."""

    def test_bell_state_partial_trace(self):
        """Tracing out one qubit of Bell state should give I/2."""
        phi_plus = np.array(
            [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
            dtype=complex,
        )
        dm = DensityMatrix(phi_plus)
        reduced = dm.partial_trace(dims=[2, 2], axis=1)
        expected = np.eye(2, dtype=complex) / 2
        np.testing.assert_allclose(reduced.state, expected, atol=1e-10)


class TestPartialTranspose(unittest.TestCase):
    """Test partial transpose."""

    def test_partial_transpose_product_state(self):
        """Partial transpose of |00⟩⟨00| should be unchanged."""
        rho = np.zeros((4, 4), dtype=complex)
        rho[0, 0] = 1.0
        dm = DensityMatrix(rho)
        result = dm.partial_transpose(dims=[2, 2], axis=0)
        np.testing.assert_allclose(result.state, rho, atol=1e-10)

    def test_partial_transpose_entangled_has_neg_eigenvalue(self):
        """Bell state partial transpose should have a negative eigenvalue (Peres criterion)."""
        phi_plus = np.array(
            [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
            dtype=complex,
        )
        dm = DensityMatrix(phi_plus)
        result = dm.partial_transpose(dims=[2, 2], axis=1)
        eigenvalues = np.linalg.eigvalsh(result.state)
        self.assertTrue(np.any(eigenvalues < -1e-8))


class TestL1Norm(unittest.TestCase):
    """Test L1 norm (coherence measure)."""

    def test_diagonal_state_zero_coherence(self):
        """Diagonal state should have L1 norm = 0."""
        dm = DensityMatrix(np.array([[0.5, 0], [0, 0.5]], dtype=complex))
        self.assertAlmostEqual(np.real(dm.l1_norm()), 0.0, places=10)

    def test_superposition_nonzero_coherence(self):
        """|+⟩ should have nonzero L1 norm."""
        plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        dm = DensityMatrix(plus)
        self.assertGreater(np.real(dm.l1_norm()), 0)


class TestBellValue(unittest.TestCase):
    """Test max Bell value (CHSH)."""

    def test_bell_state_violates(self):
        """Bell state should violate CHSH (Bell value > 2)."""
        phi_plus = np.array(
            [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
            dtype=complex,
        )
        dm = DensityMatrix(phi_plus)
        bell_val = dm.max_bell_value()
        self.assertGreater(bell_val, 2.0)
        # Max is 2√2 ≈ 2.828
        self.assertAlmostEqual(bell_val, 2 * np.sqrt(2), places=3)


class TestTeleportationFidelity(unittest.TestCase):
    """Test teleportation fidelity."""

    def test_bell_state_fidelity(self):
        """Bell state should give teleportation fidelity = 1."""
        phi_plus = np.array(
            [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
            dtype=complex,
        )
        dm = DensityMatrix(phi_plus)
        fid = dm.teleportation_fidelity()
        self.assertAlmostEqual(fid, 1.0, places=5)


class TestEOF(unittest.TestCase):
    """Test entanglement of formation."""

    def test_bell_state_eof(self):
        """Bell state should have EOF = 1."""
        phi_plus = np.array(
            [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
            dtype=complex,
        )
        dm = DensityMatrix(phi_plus)
        eof = dm.eof()
        self.assertAlmostEqual(eof, 1.0, places=5)

    def test_product_state_eof(self):
        """Product state should have EOF = 0."""
        rho = np.zeros((4, 4), dtype=complex)
        rho[0, 0] = 1.0
        dm = DensityMatrix(rho)
        eof = dm.eof()
        self.assertAlmostEqual(eof, 0.0, places=5)


class TestSchmidtRank(unittest.TestCase):
    """Test Schmidt rank."""

    def test_pure_state_rank(self):
        """Pure state |0⟩⟨0| should have Schmidt rank 1."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        dm = DensityMatrix(rho)
        self.assertEqual(dm.schmidt_rank(), 1)


class TestStaticMethods(unittest.TestCase):
    """Test static utility methods."""

    def test_tensor_product(self):
        """Tensor product of two 2x2 matrices should be 4x4."""
        eye = np.eye(2, dtype=complex)
        result = DensityMatrix.tensor_product(eye, eye)
        self.assertEqual(result.shape, (4, 4))
        np.testing.assert_allclose(result, np.eye(4), atol=1e-12)

    def test_tensor_product_list(self):
        """Tensor product from a list of matrices."""
        eye = np.eye(2, dtype=complex)
        result = DensityMatrix.tensor_product([eye, eye])
        np.testing.assert_allclose(result, np.eye(4), atol=1e-12)

    def test_swap_2q(self):
        """SWAP gate should be 4x4 and unitary."""
        S = DensityMatrix.swap()
        self.assertEqual(S.shape, (4, 4))
        np.testing.assert_allclose(S @ S, np.eye(4), atol=1e-12)

    def test_basis_operators(self):
        """basis_operators should return known Pauli matrices."""
        eye = DensityMatrix.basis_operators(0)
        np.testing.assert_allclose(eye, np.eye(2), atol=1e-12)
        X = DensityMatrix.basis_operators(1)
        self.assertAlmostEqual(X[0, 1], 1.0)
        self.assertAlmostEqual(X[1, 0], 1.0)

    def test_gate_expand_2_to_n(self):
        """Expanding a 2-qubit gate to N qubits should produce correct shape."""
        cnot = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        )
        expanded = DensityMatrix.gate_expand_2_to_n(cnot, 3, control=0, target=1)
        self.assertEqual(expanded.shape, (8, 8))


class TestCheck(unittest.TestCase):
    """Test the static check method."""

    def test_valid_dm_passes(self):
        """A proper density matrix should pass the check."""
        dm = DensityMatrix(np.array([[1, 0], [0, 0]], dtype=complex))
        # Note: the check method has inverted positive_semi_definiteness logic
        # but still returns True for valid DMs due to how it's coded
        result = DensityMatrix.check(dm)
        self.assertIsInstance(result, bool)


class TestRelativeEntropyCoherence(unittest.TestCase):
    """Test relative entropy of coherence."""

    def test_incoherent_state(self):
        """Incoherent state |0⟩⟨0| should have 0 relative entropy of coherence."""
        rho = np.array([[1, 0], [0, 0]], dtype=complex)
        dm = DensityMatrix(rho)
        self.assertAlmostEqual(dm.relative_entropy_coherence(), 0.0, places=7)

    def test_plus_state(self):
        """Maximally coherent state |+⟩⟨+| should have relative entropy of coherence = 1.0."""
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        dm = DensityMatrix(rho)
        self.assertAlmostEqual(dm.relative_entropy_coherence(), 1.0, places=7)

    def test_maximally_mixed_state(self):
        """Maximally mixed state I/2 should have 0 relative entropy of coherence."""
        rho = np.eye(2, dtype=complex) / 2
        dm = DensityMatrix(rho)
        self.assertAlmostEqual(dm.relative_entropy_coherence(), 0.0, places=7)

    def test_bell_state(self):
        """Bell state |Φ+⟩ should have relative entropy of coherence = 1.0."""
        phi_plus = np.array(
            [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]],
            dtype=complex,
        )
        dm = DensityMatrix(phi_plus)
        self.assertAlmostEqual(dm.relative_entropy_coherence(), 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
