from typing import Any
from .lattices import Lattice
from ..chemistry.mappers import FermionOperator, jordan_wigner, bravyi_kitaev


class QubitHamiltonian:
    """
    A wrapper class representing a qubit Hamiltonian for VQESolver.
    """

    def __init__(self, terms: list[tuple[list[tuple[int, str]], float]]):
        self.terms = terms

    def get_hamiltonian_terms(self) -> list[tuple[list[tuple[int, str]], float]]:
        return self.terms


def transverse_field_ising(
    lattice: Lattice, J: float, g: float, pbc: bool = False
) -> QubitHamiltonian:
    """
    Generate the Hamiltonian terms for the Transverse-Field Ising Model (TFIM).

    H = -J * sum_<i,j> Z_i Z_j - g * sum_i X_i

    Returns:
        QubitHamiltonian representing the Ising model.
    """
    edges = lattice.get_edges(pbc=pbc)
    n_sites = lattice.n_sites
    terms = []

    # ZZ interaction terms
    for u, v in edges:
        terms.append(([(u, "Z"), (v, "Z")], -float(J)))

    # X transverse field terms
    for i in range(n_sites):
        terms.append(([(i, "X")], -float(g)))

    return QubitHamiltonian(terms)


def heisenberg_xyz(
    lattice: Lattice, Jx: float, Jy: float, Jz: float, h: float = 0.0, pbc: bool = False
) -> QubitHamiltonian:
    """
    Generate the Hamiltonian terms for the Heisenberg XYZ spin model.

    H = -sum_<i,j> (Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j) - h * sum_i Z_i

    Returns:
        QubitHamiltonian representing the Heisenberg model.
    """
    edges = lattice.get_edges(pbc=pbc)
    n_sites = lattice.n_sites
    terms = []

    for u, v in edges:
        if abs(Jx) > 1e-12:
            terms.append(([(u, "X"), (v, "X")], -float(Jx)))
        if abs(Jy) > 1e-12:
            terms.append(([(u, "Y"), (v, "Y")], -float(Jy)))
        if abs(Jz) > 1e-12:
            terms.append(([(u, "Z"), (v, "Z")], -float(Jz)))

    if abs(h) > 1e-12:
        for i in range(n_sites):
            terms.append(([(i, "Z")], -float(h)))

    return QubitHamiltonian(terms)


def fermi_hubbard(
    lattice: Lattice,
    t: float,
    U: float,
    pbc: bool = False,
    mapping: str = "jordan_wigner",
) -> QubitHamiltonian:
    """
    Generate the Hamiltonian terms for the Fermi-Hubbard model.

    H = -t * sum_<i,j>,s (a_i,s^dagger a_j,s + H.c.) + U * sum_i n_i,up n_i,down

    Qubit Mapping:
        Site i has:
          - spin-up: spin-orbital index 2*i
          - spin-down: spin-orbital index 2*i + 1

    Returns:
        QubitHamiltonian representing the Fermi-Hubbard model.
    """
    edges = lattice.get_edges(pbc=pbc)
    n_sites = lattice.n_sites
    ferm_op = FermionOperator()

    # 1. Kinetic hopping terms: -t * (a_i^dagger a_j + a_j^dagger a_i)
    for u, v in edges:
        for spin in [0, 1]:  # 0 for spin-up, 1 for spin-down
            orb1 = 2 * u + spin
            orb2 = 2 * v + spin
            term1 = ((orb1, 1), (orb2, 0))
            term2 = ((orb2, 1), (orb1, 0))
            ferm_op += -float(t) * FermionOperator({term1: 1.0})
            ferm_op += -float(t) * FermionOperator({term2: 1.0})

    # 2. Hubbard interaction terms: U * n_up * n_down
    for i in range(n_sites):
        orb_up = 2 * i
        orb_down = 2 * i + 1
        # n = a^dagger a
        term = ((orb_up, 1), (orb_up, 0), (orb_down, 1), (orb_down, 0))
        ferm_op += float(U) * FermionOperator({term: 1.0})

    # Map to qubit operators
    if mapping.lower() == "bravyi_kitaev":
        qubit_terms = bravyi_kitaev(ferm_op, 2 * n_sites)
    else:
        qubit_terms = jordan_wigner(ferm_op)

    return QubitHamiltonian(qubit_terms)
