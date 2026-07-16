import unittest
import numpy as np
from qpiai_quantum.applications.matter import (
    ChainLattice,
    SquareLattice,
    transverse_field_ising,
    heisenberg_xyz,
    fermi_hubbard,
    heisenberg_hva_ansatz,
    QITESolver,
)
from qpiai_quantum.algorithms.opt.solvers import VQESolver


class TestMatterModule(unittest.TestCase):
    """
    Unit tests for the Materials & Matter module (lattices, models, HVA, QITE).
    """

    def test_chain_lattice_edges(self):
        # 1. Open Boundary Conditions (OBC)
        chain_obc = ChainLattice(n_sites=4)
        self.assertEqual(chain_obc.n_sites, 4)
        edges_obc = chain_obc.get_edges(pbc=False)
        self.assertEqual(edges_obc, [(0, 1), (1, 2), (2, 3)])

        # 2. Periodic Boundary Conditions (PBC)
        chain_pbc = ChainLattice(n_sites=4)
        edges_pbc = chain_pbc.get_edges(pbc=True)
        self.assertEqual(edges_pbc, [(0, 1), (1, 2), (2, 3), (3, 0)])

        # 3. Invalid sites
        with self.assertRaises(ValueError):
            ChainLattice(n_sites=1)

    def test_square_lattice_edges(self):
        # 1. 2x2 grid, OBC
        grid_obc = SquareLattice(width=2, height=2)
        self.assertEqual(grid_obc.n_sites, 4)
        edges_obc = grid_obc.get_edges(pbc=False)
        # Horizontal: (0,1), (2,3)
        # Vertical: (0,2), (1,3)
        self.assertEqual(len(edges_obc), 4)
        self.assertIn((0, 1), edges_obc)
        self.assertIn((2, 3), edges_obc)
        self.assertIn((0, 2), edges_obc)
        self.assertIn((1, 3), edges_obc)

        # 2. 2x2 grid, PBC
        grid_pbc = SquareLattice(width=2, height=2)
        edges_pbc = grid_pbc.get_edges(pbc=True)
        # Horizontal: (0,1), (1,0), (2,3), (3,2) -> since width=2, pbc adds loopbacks
        # Vertical: (0,2), (2,0), (1,3), (3,1)
        self.assertEqual(len(edges_pbc), 8)

    def test_heisenberg_xyz_and_hva(self):
        lattice = ChainLattice(n_sites=3)
        ham = heisenberg_xyz(lattice, Jx=1.0, Jy=1.0, Jz=1.0, h=0.5, pbc=False)

        # 2 edges * 3 Pauli types (X, Y, Z) + 3 sites * 1 transverse field Z = 9 terms
        self.assertEqual(len(ham.get_hamiltonian_terms()), 9)

        # Compile 1-layer HVA circuit
        circuit = heisenberg_hva_ansatz(lattice, layers=1)
        self.assertEqual(circuit.num_qubits, 3)

        # Verify circuit compiles and contains parametric operations
        self.assertTrue(len(circuit.icr.evolve) > 0)

    def test_fermi_hubbard_mapping(self):
        lattice = ChainLattice(n_sites=2)
        # 2 sites = 4 spin-orbitals (qubits)
        ham_jw = fermi_hubbard(
            lattice, t=1.0, U=2.0, pbc=False, mapping="jordan_wigner"
        )

        # Verify Hamiltonian is a list of Pauli string terms and not empty
        terms = ham_jw.get_hamiltonian_terms()
        self.assertTrue(len(terms) > 0)
        for term, coeff in terms:
            self.assertIsInstance(term, list)
            self.assertIsInstance(coeff, float)

    def test_qite_ground_state(self):
        # 2-site TFIM: H = -1.0 * Z0 Z1 - 0.5 * (X0 + X1)
        lattice = ChainLattice(n_sites=2)
        ham = transverse_field_ising(lattice, J=1.0, g=0.5, pbc=False)

        # Solve using QITE
        qite = QITESolver(ham, n_qubits=2)
        energy_qite, state_qite = qite.compute_ground_state(steps=60, delta_tau=0.1)

        # Solve exactly using NumPy eigenvalue diagonalization of the Hamiltonian matrix
        eigenvalues, _ = np.linalg.eigh(qite.h_mat)
        exact_ground_energy = eigenvalues[0]

        # Verify QITE ground energy is close to the exact eigenvalue
        self.assertAlmostEqual(energy_qite, exact_ground_energy, places=4)

    def test_vqe_hva_optimization(self):
        # 3-site Heisenberg XYZ: H = -1.0 * (XX + YY + ZZ) on edges (0,1) and (1,2)
        lattice = ChainLattice(n_sites=3)
        ham = heisenberg_xyz(lattice, Jx=1.0, Jy=1.0, Jz=1.0, h=0.0, pbc=False)

        # Exact ground state energy via matrix representation
        qite = QITESolver(ham, n_qubits=3)
        exact_energy = np.linalg.eigh(qite.h_mat)[0][0]

        # Optimize using VQE and Heisenberg HVA ansatz
        hva = heisenberg_hva_ansatz(lattice, layers=1)

        vqe = VQESolver(
            n_qubits=3,
            hamiltonian=ham,
            ansatz=lambda n: hva,
            optimizer="cobyla",
            max_iterations=80,
            initial_point=np.zeros(
                6
            ),  # 6 parameters for 2 edges (XX, YY, ZZ) * 1 layer
        )

        result = vqe.run(method="statevector")

        # The HVA ansatz should be expressive enough to get close to the ground state
        self.assertTrue(result.optimal_energy < -1.0)
        self.assertAlmostEqual(result.optimal_energy, exact_energy, delta=0.3)
