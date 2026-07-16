import unittest
import numpy as np
from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver, VQEResult
from qpiai_quantum.circuit import Circuit


class XYHamiltonian:
    def __init__(self, terms):
        self.terms = terms

    def get_hamiltonian_terms(self):
        return self.terms


class TestVQEXYTerms(unittest.TestCase):
    def test_statevector_expectation_single_x(self):
        # H = 1.0 * X0
        ham = XYHamiltonian([([(0, "X")], 1.0)])
        vqe = VQESolver(n_qubits=1)
        vqe.hamiltonian = ham

        # Test |0> state, which has expectation <0|X|0> = 0.0
        state_0 = np.array([1.0, 0.0], dtype=complex)
        exp_0 = vqe._compute_statevector_expectation(
            state_0, ham.get_hamiltonian_terms()
        )
        self.assertAlmostEqual(exp_0, 0.0)

        # Test |+> state = [1/sqrt(2), 1/sqrt(2)], expectation <+|X|+> = 1.0
        state_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        exp_plus = vqe._compute_statevector_expectation(
            state_plus, ham.get_hamiltonian_terms()
        )
        self.assertAlmostEqual(exp_plus, 1.0)

        # Test |-> state = [1/sqrt(2), -1/sqrt(2)], expectation <-|X|-> = -1.0
        state_minus = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)
        exp_minus = vqe._compute_statevector_expectation(
            state_minus, ham.get_hamiltonian_terms()
        )
        self.assertAlmostEqual(exp_minus, -1.0)

    def test_statevector_expectation_mixed(self):
        # H = 0.5 * X0 Y1 + 0.25 * Z0 X1
        terms = [([(0, "X"), (1, "Y")], 0.5), ([(0, "Z"), (1, "X")], 0.25)]
        ham = XYHamiltonian(terms)
        vqe = VQESolver(n_qubits=2)
        vqe.hamiltonian = ham

        # |0>_0 |+>_1 = [1/sqrt(2), 0, 1/sqrt(2), 0]
        # Expectation of Z0 X1 under |0>_0 |+>_1 is 1.0
        # expectation of X0 Y1 is 0.
        # Expectation = 0.5 * 0 + 0.25 * 1.0 = 0.25
        state = np.array([1.0, 0.0, 1.0, 0.0], dtype=complex) / np.sqrt(2)
        exp = vqe._compute_statevector_expectation(state, terms)
        self.assertAlmostEqual(exp, 0.25)

    def test_grouping_terms(self):
        terms = [
            ([(0, "X"), (1, "Y")], 1.0),
            ([(0, "X"), (1, "Z")], 2.0),
            ([(0, "Y"), (1, "Z")], 3.0),
            ([(1, "X")], 4.0),
        ]
        vqe = VQESolver(n_qubits=2)
        groups = vqe._group_hamiltonian_terms(terms)

        # Verify that all terms are in groups and grouping is correct
        self.assertGreaterEqual(len(groups), 1)

    def test_counts_expectation(self):
        # H = 1.5 * X0
        # We measured in X basis (applied H to qubit 0).
        # State was |+>, which rotated to |0>.
        # Counts should be 100% "0".
        # Expectation should be 1.5
        vqe = VQESolver(n_qubits=1)
        terms = [([(0, "X")], 1.5)]
        counts = {"0": 1000}
        exp = vqe._compute_counts_expectation(counts, terms)
        self.assertAlmostEqual(exp, 1.5)

        # State was |->, which rotated to |1>.
        # Counts should be 100% "1".
        # Expectation should be -1.5
        counts = {"1": 1000}
        exp = vqe._compute_counts_expectation(counts, terms)
        self.assertAlmostEqual(exp, -1.5)

    def test_vqe_optimization_tfim_statevector(self):
        # Transverse-Field Ising model: H = -1.0 * Z0 Z1 - 0.5 * X0 - 0.5 * X1
        terms = [
            ([(0, "Z"), (1, "Z")], -1.0),
            ([(0, "X")], -0.5),
            ([(1, "X")], -0.5),
        ]
        ham = XYHamiltonian(terms)
        vqe = VQESolver(
            n_qubits=2,
            ansatz="standard",
            optimizer="cobyla",
            max_iterations=40,
        )
        res = vqe.run(
            hamiltonian=ham,
            method="statevector",
            device_name="QpiAI-QSV-Local",
        )
        self.assertIsInstance(res, VQEResult)
        # Ground state energy is around -1.25. The optimizer should find an energy <= -1.0.
        self.assertLess(res.optimal_energy, -0.9)

    def test_vqe_optimization_tfim_shots(self):
        # Shot-based optimization with grouping
        terms = [
            ([(0, "Z"), (1, "Z")], -1.0),
            ([(0, "X")], -0.5),
            ([(1, "X")], -0.5),
        ]
        ham = XYHamiltonian(terms)
        vqe = VQESolver(
            n_qubits=2,
            ansatz="standard",
            optimizer="cobyla",
            max_iterations=5,
            verbose=False,
        )
        res = vqe.run(
            hamiltonian=ham,
            method="qasm",
            device_name="QpiAI-QSV-Local",
            shots=500,
        )
        self.assertIsInstance(res, VQEResult)
        self.assertIsNotNone(res.optimal_energy)


if __name__ == "__main__":
    unittest.main()
