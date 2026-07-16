import unittest
import numpy as np
from qpiai_quantum.applications.chemistry import (
    FermionOperator,
    jordan_wigner,
    bravyi_kitaev,
    MolecularDriver,
    uccsd_ansatz,
)
from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver, VQEResult


class TestChemistryModule(unittest.TestCase):
    def test_fermion_operator_arithmetic(self):
        # H = 1.5 * a0^dagger + 0.5 * a1
        op1 = FermionOperator({((0, 1),): 1.5})
        op2 = FermionOperator({((1, 0),): 0.5})

        # Addition
        op_sum = op1 + op2
        self.assertIn(((0, 1),), op_sum.terms)
        self.assertIn(((1, 0),), op_sum.terms)
        self.assertAlmostEqual(op_sum.terms[((0, 1),)], 1.5)
        self.assertAlmostEqual(op_sum.terms[((1, 0),)], 0.5)

        # Multiplication: (1.5 * a0^dagger) * (0.5 * a1) = 0.75 * a0^dagger a1
        op_mul = op1 * op2
        self.assertIn(((0, 1), (1, 0)), op_mul.terms)
        self.assertAlmostEqual(op_mul.terms[((0, 1), (1, 0))], 0.75)

    def test_jordan_wigner_mapping(self):
        # Number operator n_0 = a0^dagger a0 -> 0.5 * (I - Z0)
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        pauli_terms = jordan_wigner(op)

        # We expect: 0.5 * I - 0.5 * Z0
        # Let's verify terms
        terms_dict = {tuple(p): val for p, val in pauli_terms}
        self.assertIn((), terms_dict)
        self.assertAlmostEqual(terms_dict[()], 0.5)
        self.assertIn(((0, "Z"),), terms_dict)
        self.assertAlmostEqual(terms_dict[((0, "Z"),)], -0.5)

    def test_bravyi_kitaev_mapping(self):
        # Test number operator n_0 under Bravyi-Kitaev (which is also 0.5 * I - 0.5 * Z0)
        op = FermionOperator({((0, 1), (0, 0)): 1.0})
        pauli_terms = bravyi_kitaev(op, n_qubits=2)

        terms_dict = {tuple(p): val for p, val in pauli_terms}
        self.assertIn((), terms_dict)
        self.assertAlmostEqual(terms_dict[()], 0.5)
        self.assertIn(((0, "Z"),), terms_dict)
        self.assertAlmostEqual(terms_dict[((0, "Z"),)], -0.5)

    def test_molecular_driver_fallback(self):
        # Run driver for H2 molecule (will fall back to database on Windows)
        driver = MolecularDriver(geometry="H2", basis="sto-3g")
        properties = driver.run()

        self.assertEqual(properties.n_electrons, 2)
        self.assertEqual(properties.n_orbitals, 4)
        self.assertAlmostEqual(properties.nuclear_repulsion_energy, 0.71996899)
        self.assertAlmostEqual(properties.hf_energy, -1.11699900)
        self.assertEqual(properties.hf_state, [1, 1, 0, 0])

        # Test core Hamiltonian integrals shape
        self.assertEqual(properties.one_body_integrals.shape, (4, 4))
        self.assertEqual(properties.two_body_integrals.shape, (4, 4, 4, 4))

    def test_uccsd_ansatz_compilation(self):
        # Generate UCCSD ansatz for H2 (4 orbitals, 2 electrons)
        ansatz = uccsd_ansatz(n_qubits=4, n_electrons=2, mapping="jordan_wigner")
        self.assertEqual(ansatz.num_qubits, 4)

        # Check that it has parametric gates
        vqe = VQESolver(n_qubits=4, ansatz=lambda n: ansatz)
        n_params = vqe._count_parameters()
        self.assertGreater(n_params, 0)

    def test_vqe_chemistry_optimization(self):
        # Load pre-computed molecular properties of H2
        driver = MolecularDriver(geometry="H2", basis="sto-3g")
        properties = driver.run()

        # Get Fermionic Hamiltonian
        ferm_ham = properties.get_fermionic_hamiltonian()

        # Map to qubit Hamiltonian
        qubit_ham_terms = jordan_wigner(ferm_ham)

        # Create a mock wrapper for Hamiltonian to pass to VQESolver
        class ChemistryHamiltonian:
            def get_hamiltonian_terms(self):
                return qubit_ham_terms

        # Generate UCCSD ansatz
        ansatz = uccsd_ansatz(n_qubits=4, n_electrons=2, mapping="jordan_wigner")

        # Determine number of parameters first
        vqe_temp = VQESolver(n_qubits=4, ansatz=lambda n: ansatz)
        n_params = vqe_temp._count_parameters()
        initial_point = np.zeros(n_params)

        # Instantiate VQESolver with the UCCSD ansatz
        # (Using standard COBYLA optimizer with 15 iterations for fast test runs)
        vqe = VQESolver(
            n_qubits=4,
            ansatz=lambda n: ansatz,
            optimizer="cobyla",
            max_iterations=15,
            initial_point=initial_point,
            verbose=False,
        )

        # Run statevector simulation
        res = vqe.run(
            hamiltonian=ChemistryHamiltonian(),
            method="statevector",
            device_name="QpiAI-QSV-Local",
        )

        self.assertIsInstance(res, VQEResult)
        # Hartree-Fock energy of H2 is -1.116. Variational ground state energy
        # should be lower than HF energy (exact ground state is ~-1.137)
        self.assertLess(res.optimal_energy, -1.11)


if __name__ == "__main__":
    unittest.main()
