import numpy as np
from typing import Dict, List, Tuple, Union, Optional


class FermionOperator:
    """
    FermionOperator represents a sum of products of creation (+) and annihilation (-) operators.

    Each term is represented as a tuple of tuples: `((qubit_idx, action), ...)`
    where action = 1 for creation (dagger/+) and 0 for annihilation (-).

    Example:
        >>> op = FermionOperator({((0, 1), (1, 0)): 1.0})  # 1.0 * a_0^dagger a_1
    """

    def __init__(
        self, terms: dict[tuple[tuple[int, int], ...], float | complex] | None = None
    ):
        if terms is None:
            self.terms = {}
        else:
            self.terms = {
                self._clean_key(k): complex(v)
                for k, v in terms.items()
                if abs(v) > 1e-12
            }

    def _clean_key(
        self, key: tuple[tuple[int, int], ...]
    ) -> tuple[tuple[int, int], ...]:
        return tuple((int(q), int(a)) for q, a in key)

    def __add__(
        self, other: Union["FermionOperator", float, complex]
    ) -> "FermionOperator":
        if isinstance(other, (float, complex, int)):
            other = FermionOperator({(): other})
        res_terms = self.terms.copy()
        for k, v in other.terms.items():
            val = res_terms.get(k, 0.0) + v
            if abs(val) > 1e-12:
                res_terms[k] = val
            elif k in res_terms:
                del res_terms[k]
        return FermionOperator(res_terms)

    def __radd__(self, other: float | complex) -> "FermionOperator":
        return self.__add__(other)

    def __sub__(
        self, other: Union["FermionOperator", float, complex]
    ) -> "FermionOperator":
        return self.__add__(-1.0 * other)

    def __rsub__(self, other: float | complex) -> "FermionOperator":
        return FermionOperator({(): other}) + (-1.0 * self)

    def __mul__(
        self, other: Union["FermionOperator", float, complex]
    ) -> "FermionOperator":
        if isinstance(other, (float, complex, int)):
            return FermionOperator({k: v * other for k, v in self.terms.items()})

        res_terms: dict[tuple[tuple[int, int], ...], complex] = {}
        for k1, v1 in self.terms.items():
            for k2, v2 in other.terms.items():
                new_key = k1 + k2
                val = res_terms.get(new_key, 0.0) + v1 * v2
                if abs(val) > 1e-12:
                    res_terms[new_key] = val
                elif new_key in res_terms:
                    del res_terms[new_key]
        return FermionOperator(res_terms)

    def __rmul__(self, other: float | complex) -> "FermionOperator":
        return self.__mul__(other)

    def __repr__(self) -> str:
        if not self.terms:
            return "0.0"
        parts = []
        for term, coeff in self.terms.items():
            coeff_str = (
                f"{coeff.real:+.4f}"
                if abs(coeff.imag) < 1e-9
                else f"+({coeff.real:+.4f}{coeff.imag:+.4f}j)"
            )
            if term == ():
                parts.append(coeff_str)
            else:
                term_str = " ".join(f"{q}{'+' if a == 1 else '-'}" for q, a in term)
                parts.append(f"{coeff_str} [{term_str}]")
        return " ".join(parts)


def multiply_paulis(
    pauli1: dict[int, str], coeff1: complex, pauli2: dict[int, str], coeff2: complex
) -> tuple[dict[int, str], complex]:
    """Multiply two Pauli strings and return the resulting Pauli string and coefficient."""
    res_pauli = {}
    res_coeff = coeff1 * coeff2
    all_qubits = set(pauli1.keys()).union(pauli2.keys())
    for q in all_qubits:
        op1 = pauli1.get(q, "I")
        op2 = pauli2.get(q, "I")
        if op1 == "I":
            if op2 != "I":
                res_pauli[q] = op2
        elif op2 == "I":
            res_pauli[q] = op1
        elif op1 == op2:
            pass
        elif op1 == "X" and op2 == "Y":
            res_coeff *= 1j
            res_pauli[q] = "Z"
        elif op1 == "Y" and op2 == "X":
            res_coeff *= -1j
            res_pauli[q] = "Z"
        elif op1 == "Y" and op2 == "Z":
            res_coeff *= 1j
            res_pauli[q] = "X"
        elif op1 == "Z" and op2 == "Y":
            res_coeff *= -1j
            res_pauli[q] = "X"
        elif op1 == "Z" and op2 == "X":
            res_coeff *= 1j
            res_pauli[q] = "Y"
        elif op1 == "X" and op2 == "Z":
            res_coeff *= -1j
            res_pauli[q] = "Y"
    return res_pauli, res_coeff


def _combine_pauli_terms(
    pauli_terms: list[tuple[dict[int, str], complex]],
) -> list[tuple[list[tuple[int, str]], float]]:
    """Combine duplicate Pauli terms and simplify the representation."""
    combined: dict[tuple[tuple[int, str], ...], complex] = {}
    for p, c in pauli_terms:
        # Sort by qubit index to ensure unique keys
        sorted_p = tuple(sorted(p.items()))
        combined[sorted_p] = combined.get(sorted_p, 0.0) + c

    simplified = []
    for p_tuple, c in combined.items():
        if abs(c) > 1e-10:
            val = c.real if abs(c.real) > abs(c.imag) else c.imag
            simplified.append((list(p_tuple), float(val)))
    return simplified


def jordan_wigner(
    fermion_op: FermionOperator,
) -> list[tuple[list[tuple[int, str]], float]]:
    """
    Map a FermionOperator to a Pauli representation using Jordan-Wigner transformation.

    Args:
        fermion_op: FermionOperator to map

    Returns:
        List of Pauli strings in the SDK Hamiltonian format: List[Tuple[List[Tuple[int, str]], float]]
    """
    all_pauli_terms = []

    for term, coeff in fermion_op.terms.items():
        current_terms = [({}, complex(coeff))]

        for qubit, action in term:
            # a_k^dagger = 0.5 * (X_k - i Y_k) Z_0...Z_k-1
            # a_k        = 0.5 * (X_k + i Y_k) Z_0...Z_k-1
            sign = -1.0 if action == 1 else 1.0

            term_x = {qubit: "X"}
            term_y = {qubit: "Y"}
            for q in range(qubit):
                term_x[q] = "Z"
                term_y[q] = "Z"

            op_terms = [
                (term_x, 0.5),
                (term_y, sign * 0.5j),
            ]

            next_terms = []
            for p1, c1 in current_terms:
                for p2, c2 in op_terms:
                    p_res, c_res = multiply_paulis(p1, c1, p2, c2)
                    next_terms.append((p_res, c_res))
            current_terms = next_terms

        all_pauli_terms.extend(current_terms)

    return _combine_pauli_terms(all_pauli_terms)


def _get_bk_matrix(n: int) -> np.ndarray:
    """Generate the Bravyi-Kitaev transformation matrix of size n x n recursively."""
    power_of_2 = 1
    while power_of_2 < n:
        power_of_2 *= 2

    def construct_bk(sz: int) -> np.ndarray:
        if sz == 1:
            return np.array([[1]], dtype=int)
        sub = construct_bk(sz // 2)
        top = np.hstack([sub, np.zeros((sz // 2, sz // 2), dtype=int)])
        A = np.zeros((sz // 2, sz // 2), dtype=int)
        A[-1, :] = 1
        bottom = np.hstack([A, sub])
        return np.vstack([top, bottom])

    B_full = construct_bk(power_of_2)
    return B_full[:n, :n]


def bravyi_kitaev(
    fermion_op: FermionOperator, n_qubits: int
) -> list[tuple[list[tuple[int, str]], float]]:
    """
    Map a FermionOperator to a Pauli representation using Bravyi-Kitaev transformation.

    References:
        - Seeley, J. T., Richard, M. J., & Love, P. J. "The Bravyi-Kitaev transformation
          for quantum computation of chemical wavefunctions." J. Chem. Phys. 137.22 (2012).

    Args:
        fermion_op: FermionOperator to map
        n_qubits: Number of qubits in the active space (pads to next power of 2)

    Returns:
        List of Pauli strings in the SDK Hamiltonian format: List[Tuple[List[Tuple[int, str]], float]]
    """
    power_of_2 = 1
    while power_of_2 < n_qubits:
        power_of_2 *= 2

    B = _get_bk_matrix(power_of_2)

    # Compute Update and Parity sets for each mode j
    update_sets: dict[int, list[int]] = {}
    parity_sets: dict[int, list[int]] = {}

    for j in range(power_of_2):
        # Update set: column j of B (excluding index j itself)
        update_sets[j] = [k for k in range(j + 1, power_of_2) if B[k, j] == 1]
        # Parity set: row sum of inverse B (which is B mod 2) for elements < j
        # parity_sets[j] consists of columns m where sum_{i < j} B_{i, m} == 1 (mod 2)
        parity_sets[j] = [
            m for m in range(power_of_2) if sum(B[i, m] for i in range(j)) % 2 == 1
        ]

    all_pauli_terms = []

    for term, coeff in fermion_op.terms.items():
        current_terms = [({}, complex(coeff))]

        for qubit, action in term:
            # a_j^dagger = 0.5 * (X_j * X_U - i Y_j * X_U) * Z_P
            # a_j        = 0.5 * (X_j * X_U + i Y_j * X_U) * Z_P
            sign = -1.0 if action == 1 else 1.0

            # First term: X_j * X_U * Z_P
            term1_pauli = {qubit: "X"}
            for q in update_sets[qubit]:
                term1_pauli[q] = "X"
            for q in parity_sets[qubit]:
                term1_pauli[q] = "Z"

            # Second term: Y_j * X_U * Z_P
            term2_pauli = {qubit: "Y"}
            for q in update_sets[qubit]:
                term2_pauli[q] = "X"
            for q in parity_sets[qubit]:
                term2_pauli[q] = "Z"

            op_terms = [
                (term1_pauli, 0.5),
                (term2_pauli, sign * 0.5j),
            ]

            next_terms = []
            for p1, c1 in current_terms:
                for p2, c2 in op_terms:
                    p_res, c_res = multiply_paulis(p1, c1, p2, c2)
                    next_terms.append((p_res, c_res))
            current_terms = next_terms

        all_pauli_terms.extend(current_terms)

    return _combine_pauli_terms(all_pauli_terms)
