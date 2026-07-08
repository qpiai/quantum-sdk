"""
Unit tests to verify quantum algorithm standardization.

This test file ensures that all algorithm classes in the SDK:
- Set self.circuit on calling build_circuit().
- Auto-build if self.circuit is None when calling execution helpers.
- Run without AttributeError.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Adjust the import path to make sure the local qpiai_quantum package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.shor import ShorsAlgorithm
from qpiai_quantum.algorithms.simon import SimonAlgorithm
from qpiai_quantum.algorithms.phase_estimation import QuantumPhaseEstimation
from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver
from qpiai_quantum.algorithms.opt.solvers.qaoa import QAOASolver

from qpiai_quantum.algorithms.qft import QFT
from qpiai_quantum.algorithms.grover import GroverSearch
from qpiai_quantum.algorithms.bernstein_vazirani import BernsteinVazirani
from qpiai_quantum.algorithms.deutsch_jozsa import DeutschJozsa
from qpiai_quantum.algorithms.qrng import QRNG
from qpiai_quantum.algorithms.amplitude_estimation import (
    EstimationProblem,
    IterativeAmplitudeEstimation,
)
from qpiai_quantum.circuit import Circuit


class TestAlgorithmStandardization(unittest.TestCase):
    def _make_mock_result(self, counts: dict):
        mock_result = MagicMock()
        mock_result.get.return_value = {"counts": counts}
        mock_result.counts = counts
        return mock_result

    @patch("qpiai_quantum.algorithms.shor.ShorsAlgorithm.run")
    def test_shor_standardization(self, mock_run):
        # Mock run to return counts that denote period 4
        # a=2, N=15. period is 4.
        # we mock the measurement result corresponding to 4/16 = 0.25 -> 4
        mock_run.return_value = self._make_mock_result({"0100": 1})

        shor = ShorsAlgorithm(N=15)
        self.assertIsNone(shor.circuit)

        # find_period should call run() instead of execute() and succeed
        period = shor.find_period(a=2, precision_qubits=4)
        self.assertIsNotNone(shor.circuit)
        mock_run.assert_called_once()
        self.assertEqual(period, 4)

    @patch("qpiai_quantum.algorithms.simon.SimonAlgorithm.run")
    def test_simon_standardization(self, mock_run):
        # Mock run to return the equation outcomes
        mock_run.return_value = self._make_mock_result({"110": 1})

        simon = SimonAlgorithm(num_qubits=3, hidden_string="110")
        self.assertIsNone(simon.circuit)

        # find_hidden_string should auto-build circuit and complete successfully
        res = simon.find_hidden_string(max_attempts=3)
        self.assertIsNotNone(simon.circuit)
        self.assertEqual(res, "110")

    @patch("qpiai_quantum.algorithms.phase_estimation.QuantumPhaseEstimation.run")
    def test_qpe_standardization(self, mock_run):
        # Mock run to return phase measurement corresponding to 2/8 = 0.25
        mock_run.return_value = self._make_mock_result({"010": 1})

        qpe = QuantumPhaseEstimation(precision_qubits=3, eigenstate_qubits=1)
        self.assertIsNone(qpe.circuit)

        phase = qpe.estimate_phase(unitary="T")
        self.assertIsNotNone(qpe.circuit)
        self.assertEqual(phase, 0.25)
        # Ensure that build parameters like 'unitary' were NOT passed to run() as kwargs
        mock_run.assert_called_once_with(shots=1)

    def test_vqe_standardization(self):
        vqe = VQESolver(n_qubits=3, ansatz="standard")
        self.assertIsNone(vqe.circuit)

        # build_circuit should set self.circuit
        circ = vqe.build_circuit()
        self.assertIsNotNone(vqe.circuit)
        self.assertIs(vqe.circuit, circ)

    def test_qaoa_standardization(self):
        # QAOA needs a problem to build the circuit
        mock_problem = MagicMock()
        mock_problem.n_qubits = 3
        mock_problem.get_hamiltonian_terms.return_value = []

        qaoa = QAOASolver(layers=1)
        qaoa.problem = mock_problem
        self.assertIsNone(qaoa.circuit)

        # build_circuit should set self.circuit
        circ = qaoa.build_circuit()
        self.assertIsNotNone(qaoa.circuit)
        self.assertIs(qaoa.circuit, circ)

    def test_qft_standardization(self):
        qft = QFT(num_qubits=3)
        self.assertIsNone(qft.circuit)
        circ = qft.build_circuit()
        self.assertIsNotNone(qft.circuit)
        self.assertIs(qft.circuit, circ)

    def test_grover_standardization(self):
        grover = GroverSearch(num_qubits=3, target="101")
        self.assertIsNone(grover.circuit)
        circ = grover.build_circuit()
        self.assertIsNotNone(grover.circuit)
        self.assertIs(grover.circuit, circ)

    @patch("qpiai_quantum.algorithms.bernstein_vazirani.BernsteinVazirani.run")
    def test_bernstein_vazirani_standardization(self, mock_run):
        mock_run.return_value = self._make_mock_result({"101": 1})
        bv = BernsteinVazirani(num_qubits=3, hidden_string="101")
        self.assertIsNone(bv.circuit)
        res = bv.find_hidden_string()
        self.assertIsNotNone(bv.circuit)
        self.assertEqual(res, "101")

    @patch("qpiai_quantum.algorithms.deutsch_jozsa.DeutschJozsa.run")
    def test_deutsch_jozsa_standardization(self, mock_run):
        mock_run.return_value = self._make_mock_result({"111": 1})
        dj = DeutschJozsa(num_qubits=3, oracle_type="balanced")
        self.assertIsNone(dj.circuit)
        res = dj.determine_function_type()
        self.assertIsNotNone(dj.circuit)
        self.assertEqual(res, "balanced")

    @patch("qpiai_quantum.algorithms.qrng.QRNG.run")
    def test_qrng_standardization(self, mock_run):
        mock_run.return_value = self._make_mock_result({"1010": 1})
        qrng = QRNG(n_bits=4)
        self.assertIsNone(qrng.circuit)
        res = qrng.generate()
        self.assertIsNotNone(qrng.circuit)
        self.assertEqual(res, 10)

    @patch("qpiai_quantum.circuit.Circuit.run")
    def test_amplitude_estimation_standardization(self, mock_run):
        mock_run.return_value = self._make_mock_result({"11": 500, "00": 500})
        iae = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05)
        self.assertIsNone(iae.circuit)

        # Prepare state preparation circuit
        prep = Circuit(2)
        prep.h(0)
        prep.cx(0, 1)

        problem = EstimationProblem(state_preparation=prep, objective_qubits=[1])
        circ = iae.build_circuit(problem, k=2)
        self.assertIsNotNone(iae.circuit)
        self.assertIs(iae.circuit, circ)

        # Test estimate logic
        amp = iae.estimate(problem, shots=1000)
        self.assertIsInstance(amp, float)


if __name__ == "__main__":
    unittest.main()
