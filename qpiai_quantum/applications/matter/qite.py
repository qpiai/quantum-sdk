import numpy as np
from scipy.linalg import expm
from typing import Optional, Union
from .models import QubitHamiltonian

# Standard Pauli matrices
_I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

_PAULI_MAP = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _term_to_matrix(term: list[tuple[int, str]], n_qubits: int) -> np.ndarray:
    """Convert a single Pauli term to its 2^n x 2^n matrix representation."""
    op_list = [_I] * n_qubits
    for qubit, pauli in term:
        op_list[qubit] = _PAULI_MAP[pauli]

    mat = op_list[0]
    for next_op in op_list[1:]:
        mat = np.kron(mat, next_op)
    return mat


def hamiltonian_to_matrix(
    hamiltonian: list[tuple[list[tuple[int, str]], float]] | QubitHamiltonian,
    n_qubits: int,
) -> np.ndarray:
    """
    Convert a list of Hamiltonian terms or a QubitHamiltonian to a full dense matrix representation.
    """
    if hasattr(hamiltonian, "get_hamiltonian_terms"):
        terms = hamiltonian.get_hamiltonian_terms()
    else:
        terms = hamiltonian

    dim = 2**n_qubits
    h_mat = np.zeros((dim, dim), dtype=complex)
    for term, coeff in terms:
        h_mat += coeff * _term_to_matrix(term, n_qubits)
    return h_mat


class QITESolver:
    """
    Quantum Imaginary Time Evolution (QITE) statevector solver.
    """

    def __init__(
        self,
        terms: list[tuple[list[tuple[int, str]], float]] | QubitHamiltonian,
        n_qubits: int,
    ):
        """
        Args:
            terms: Hamiltonian terms in the SDK format, or a QubitHamiltonian object
            n_qubits: Total number of qubits
        """
        self.terms = terms
        self.n_qubits = n_qubits
        self.h_mat = hamiltonian_to_matrix(terms, n_qubits)

    def compute_ground_state(
        self,
        steps: int = 50,
        delta_tau: float = 0.1,
        initial_state: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """
        Compute the ground state energy and statevector using imaginary time evolution.

        References:
            - Motta, M., et al. "Determining eigenstates and thermal states on a
              quantum computer using quantum imaginary time evolution."
              Nature Physics 16.2 (2020).

        Args:
            steps: Number of imaginary time steps
            delta_tau: Imaginary time step size (delta_tau)
            initial_state: Optional starting statevector (defaults to uniform superposition)

        Returns:
            tuple of (ground_state_energy, ground_state_vector)
        """
        dim = 2**self.n_qubits
        if initial_state is None:
            # Uniform superposition initial state
            psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
        else:
            psi = np.array(initial_state, dtype=complex)
            psi = psi / np.linalg.norm(psi)

        # Precompute the non-unitary time-evolution operator: e^(-delta_tau * H)
        u_imag = expm(-delta_tau * self.h_mat)

        for _ in range(steps):
            psi = u_imag @ psi
            psi = psi / np.linalg.norm(psi)

        # Compute ground state energy: E = <psi|H|psi>
        energy = float(np.real(np.dot(psi.conj().T, self.h_mat @ psi)))
        return energy, psi
