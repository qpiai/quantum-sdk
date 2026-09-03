"""
Unit tests for DensityMatrixState and DensityMatrixSimulator.
"""

import math
import numpy as np

from qpiai_quantum.circuit import Circuit
from qpiai_quantum.simulator.density_matrix import (
    DensityMatrixSimulator,
    DensityMatrixState,
)


class TestDensityMatrixState:
    """Tests for the DensityMatrixState representation."""

    def test_initial_zero_state(self):
        dm = DensityMatrixState(num_qubits=2)
        assert dm.num_qubits == 2
        assert dm.data.shape == (4, 4)
        expected = np.zeros((4, 4), dtype=complex)
        expected[0, 0] = 1.0
        np.testing.assert_allclose(dm.data, expected)
        assert math.isclose(dm.purity(), 1.0, abs_tol=1e-7)
        assert math.isclose(dm.von_neumann_entropy(), 0.0, abs_tol=1e-7)

    def test_maximally_mixed_state(self):
        dm = DensityMatrixState.maximally_mixed(num_qubits=2)
        expected = 0.25 * np.eye(4, dtype=complex)
        np.testing.assert_allclose(dm.data, expected)
        assert math.isclose(dm.purity(), 0.25, abs_tol=1e-7)
        assert math.isclose(dm.von_neumann_entropy(), 2.0, abs_tol=1e-7)

    def test_pure_state_constructor(self):
        # |+> state
        plus_sv = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
        dm = DensityMatrixState.pure_state(plus_sv)
        expected = 0.5 * np.ones((2, 2), dtype=complex)
        np.testing.assert_allclose(dm.data, expected)
        assert math.isclose(dm.purity(), 1.0, abs_tol=1e-7)

    def test_single_qubit_gates(self):
        # Pauli X on |0> -> |1><1|
        dm = DensityMatrixState(num_qubits=1)
        dm.apply_x(0)
        np.testing.assert_allclose(dm.data, [[0, 0], [0, 1]])

        # Hadamard on |0> -> |+><+|
        dm.reset()
        dm.apply_h(0)
        np.testing.assert_allclose(dm.data, [[0.5, 0.5], [0.5, 0.5]])

        # Phase (S) on |+>
        dm.apply_s(0)
        np.testing.assert_allclose(dm.data, [[0.5, -0.5j], [0.5j, 0.5]])

        # S† undoes S
        dm.apply_sdg(0)
        np.testing.assert_allclose(dm.data, [[0.5, 0.5], [0.5, 0.5]])

        # Rotations
        dm.reset()
        dm.apply_rx(0, np.pi)  # RX(pi) ~ -i X, so |0><0| -> |1><1|
        np.testing.assert_allclose(dm.data, [[0, 0], [0, 1]], atol=1e-12)

    def test_two_qubit_bell_state_and_partial_trace(self):
        dm = DensityMatrixState(num_qubits=2)
        # Create Bell state (|00> + |11>) / sqrt(2)
        dm.apply_h(0)
        dm.apply_cnot(0, 1)

        # Expected: 0.5 * (|00><00| + |00><11| + |11><00| + |11><11|)
        expected = np.zeros((4, 4), dtype=complex)
        expected[0, 0] = 0.5
        expected[0, 3] = 0.5
        expected[3, 0] = 0.5
        expected[3, 3] = 0.5
        np.testing.assert_allclose(dm.data, expected, atol=1e-12)
        assert math.isclose(dm.purity(), 1.0, abs_tol=1e-7)

        # Tracing out qubit 1 yields maximally mixed state on qubit 0
        reduced_0 = dm.partial_trace([1])
        assert reduced_0.num_qubits == 1
        np.testing.assert_allclose(reduced_0.data, 0.5 * np.eye(2), atol=1e-12)
        assert math.isclose(reduced_0.purity(), 0.5, abs_tol=1e-7)
        assert math.isclose(reduced_0.von_neumann_entropy(), 1.0, abs_tol=1e-7)

    def test_swap_and_cz_gates(self):
        # Prepare |10> = X on qubit 1
        dm = DensityMatrixState(num_qubits=2)
        dm.apply_x(1)  # state is |10> (basis index 2 in little-endian)
        assert math.isclose(dm.get_probabilities()[2], 1.0, abs_tol=1e-7)

        # SWAP(0, 1) -> |01> (basis index 1 in little-endian)
        dm.apply_swap(0, 1)
        assert math.isclose(dm.get_probabilities()[1], 1.0, abs_tol=1e-7)

        # Prepare |11>, apply CZ -> |11> with phase -1 (in DM, (-1)*(-1) = +1)
        dm.apply_x(1)  # now |11>
        dm.apply_cz(0, 1)
        assert math.isclose(dm.get_probabilities()[3], 1.0, abs_tol=1e-7)

    def test_kraus_channel(self):
        # Amplitude damping on |1><1|
        dm = DensityMatrixState(num_qubits=1)
        dm.apply_x(0)  # |1><1|

        gamma = 0.3
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)

        dm.apply_kraus([K0, K1], [0])
        np.testing.assert_allclose(
            dm.data, [[gamma, 0], [0, 1 - gamma]], atol=1e-12
        )
        assert math.isclose(np.trace(dm.data).real, 1.0, abs_tol=1e-7)


class TestDensityMatrixSimulator:
    """Tests for the DensityMatrixSimulator backend."""

    def test_simulator_bell_state_run(self):
        qc = Circuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, 0)
        qc.measure(1, 1)

        sim = DensityMatrixSimulator()
        result = sim.run(qc, shots=500, seed=42)

        assert result.method == "density_matrix"
        assert result.job_status == "completed"
        assert result.density_matrix is not None

        dm_arr = np.array(result.density_matrix)
        assert dm_arr.shape == (4, 4)

        # Measurement counts should only contain '00' and '11'
        assert set(result.counts.keys()).issubset({"00", "11"})
        assert sum(result.counts.values()) == 500

    def test_simulator_ghz_3qubit(self):
        qc = Circuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()

        sim = DensityMatrixSimulator()
        result = sim.run(qc, shots=1000, seed=123)

        assert set(result.counts.keys()).issubset({"000", "111"})
        assert result.counts.get("000", 0) > 400
        assert result.counts.get("111", 0) > 400

        dm_arr = np.array(result.density_matrix)
        assert dm_arr.shape == (8, 8)
        assert math.isclose(dm_arr[0, 0].real, 0.5, abs_tol=1e-7)
        assert math.isclose(dm_arr[7, 7].real, 0.5, abs_tol=1e-7)
        assert math.isclose(dm_arr[0, 7].real, 0.5, abs_tol=1e-7)
        assert math.isclose(dm_arr[7, 0].real, 0.5, abs_tol=1e-7)

    def test_simulator_result_interoperability_with_quantum_info_density_matrix(self):
        from qpiai_quantum.quantum_info.density_matrix import DensityMatrix

        qc = Circuit(2)
        qc.h(0)
        qc.cx(0, 1)

        sim = DensityMatrixSimulator()
        result = sim.run(qc)

        # Interoperability: loading nested list into quantum_info.DensityMatrix
        dm = DensityMatrix(result.density_matrix)
        assert dm.num_qubits == 2
        assert dm.is_pure() is True
        assert math.isclose(dm.purity(), 1.0, abs_tol=1e-7)
        probs = dm.probabilities_dict()
        assert math.isclose(probs.get("00", 0.0), 0.5, abs_tol=1e-7)
        assert math.isclose(probs.get("11", 0.0), 0.5, abs_tol=1e-7)

