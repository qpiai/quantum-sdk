"""
Gate Matrices and Specifications
=================================
Pure-math module containing single/multi-qubit gate matrices,
the gate-spec lookup, and decomposition rules for compound gates.

Conventions:
  - Qubit q[0] is the least-significant qubit in the statevector basis
    index (i.e. amplitude index i has bit b for qubit b given by
    (i >> b) & 1). This matches the common little-endian convention.
  - rz(theta), u1/p(theta) etc. follow the standard modern definitions:
        rz(theta) = diag(exp(-i*theta/2), exp(i*theta/2))
        u1(lambda) = p(lambda) = diag(1, exp(i*lambda))
"""

import numpy as np

# --------------------------------------------------------------------------
# Single-qubit gate matrix factories
# --------------------------------------------------------------------------


def u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """General single-qubit unitary U3(θ, φ, λ)."""
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2),
            ],
        ],
        dtype=complex,
    )


def rx_matrix(theta: float) -> np.ndarray:
    """Rotation around X-axis: RX(θ)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry_matrix(theta: float) -> np.ndarray:
    """Rotation around Y-axis: RY(θ)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz_matrix(theta: float) -> np.ndarray:
    """Rotation around Z-axis: RZ(θ) = diag(e^{-iθ/2}, e^{iθ/2})."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


def rxx_matrix(theta: float) -> np.ndarray:
    """Two-qubit XX rotation: RXX(θ) = exp(-i θ/2 X⊗X)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [
            [c, 0, 0, -1j * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [-1j * s, 0, 0, c],
        ],
        dtype=complex,
    )


def ryy_matrix(theta: float) -> np.ndarray:
    """Two-qubit YY rotation: RYY(θ) = exp(-i θ/2 Y⊗Y)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [
            [c, 0, 0, 1j * s],
            [0, c, -1j * s, 0],
            [0, -1j * s, c, 0],
            [1j * s, 0, 0, c],
        ],
        dtype=complex,
    )


def rzz_matrix(theta: float) -> np.ndarray:
    """Two-qubit ZZ rotation: RZZ(θ) = exp(-i θ/2 Z⊗Z)."""
    return np.diag(
        [
            np.exp(-1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(1j * theta / 2),
            np.exp(-1j * theta / 2),
        ]
    ).astype(complex)


# --------------------------------------------------------------------------
# Fixed single-qubit gate matrices
# --------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = S.conj().T
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG = T.conj().T
SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)
SXDG = SX.conj().T

# --------------------------------------------------------------------------
# Fixed multi-qubit gate matrices
# --------------------------------------------------------------------------

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)

ISWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
)
ISWAPDG = ISWAP.conj().T


ECR = (1 / np.sqrt(2)) * np.array(
    [
        [0, 0, 1, 1j],
        [0, 0, 1j, 1],
        [1, -1j, 0, 0],
        [-1j, 1, 0, 0],
    ],
    dtype=complex,
)


# --------------------------------------------------------------------------
# Controlled gate builder
# --------------------------------------------------------------------------


def controlled(U: np.ndarray, n_ctrl: int = 1) -> np.ndarray:
    """Build a controlled-U matrix.

    Constructs a (2^n_ctrl * k) x (2^n_ctrl * k) matrix that applies U on the
    last log2(k) qubits only when ALL n_ctrl control qubits (placed as the
    most-significant qubits, i.e. first args) are |1⟩.

    Args:
        U: The target unitary matrix (k x k).
        n_ctrl: Number of control qubits.

    Returns:
        The controlled-U matrix.
    """
    k = U.shape[0]
    dim = k * (2**n_ctrl)
    M = np.eye(dim, dtype=complex)
    M[dim - k :, dim - k :] = U
    return M


# --------------------------------------------------------------------------
# Gate specification lookup
# --------------------------------------------------------------------------

#: Set of all gate names recognised by :func:`gate_spec` or :func:`decompose`.
ALL_KNOWN_GATES: set[str] = {
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "sxdg",
    "rx",
    "ry",
    "rz",
    "u1",
    "u2",
    "u3",
    "u",
    "p",
    "cx",
    "cy",
    "cz",
    "ch",
    "cs",
    "swap",
    "iswap",
    "iswapdg",
    "dcx",
    "ecr",
    "rxx",
    "ryy",
    "rzz",
    "crx",
    "cry",
    "crz",
    "cu1",
    "cu3",
    "cu",
    "cp",
    "ccx",
    "cswap",
    "rccx",
    "mcx",
}

#: Gates handled via decomposition rather than a single matrix.
DECOMPOSED_GATES: set[str] = {"dcx", "rccx"}


def gate_spec(
    name: str, params: list[float], num_qubits: int | None = None
) -> tuple[int, np.ndarray]:
    """Return ``(num_qubits, matrix)`` for a given gate name and numeric params.

    Raises:
        ValueError: If the gate name is not recognised.
    """
    p = params

    # -- Single-qubit non-parametric --
    if name == "id":
        return 1, I2
    if name == "x":
        return 1, X
    if name == "y":
        return 1, Y
    if name == "z":
        return 1, Z
    if name == "h":
        return 1, H
    if name == "s":
        return 1, S
    if name == "sdg":
        return 1, SDG
    if name == "t":
        return 1, T
    if name == "tdg":
        return 1, TDG
    if name == "sx":
        return 1, SX
    if name == "sxdg":
        return 1, SXDG

    # -- Single-qubit parametric --
    if name == "rx":
        return 1, rx_matrix(p[0])
    if name == "ry":
        return 1, ry_matrix(p[0])
    if name == "rz":
        return 1, rz_matrix(p[0])
    if name == "u1" or name == "p":
        return 1, u3_matrix(0.0, 0.0, p[0])
    if name == "u2":
        return 1, u3_matrix(np.pi / 2, p[0], p[1])
    if name in ("u3", "u"):
        return 1, u3_matrix(p[0], p[1], p[2])

    # -- Two-qubit gates --
    if name == "cx":
        return 2, controlled(X, 1)
    if name == "cy":
        return 2, controlled(Y, 1)
    if name == "cz":
        return 2, controlled(Z, 1)
    if name == "ch":
        return 2, controlled(H, 1)
    if name == "cs":
        return 2, controlled(S, 1)
    if name == "swap":
        return 2, SWAP
    if name == "iswap":
        return 2, ISWAP
    if name == "iswapdg":
        return 2, ISWAPDG
    if name == "ecr":
        return 2, ECR
    if name == "rxx":
        return 2, rxx_matrix(p[0])
    if name == "ryy":
        return 2, ryy_matrix(p[0])
    if name == "rzz":
        return 2, rzz_matrix(p[0])
    if name == "crx":
        return 2, controlled(rx_matrix(p[0]), 1)
    if name == "cry":
        return 2, controlled(ry_matrix(p[0]), 1)
    if name == "crz":
        return 2, controlled(rz_matrix(p[0]), 1)
    if name == "cu1" or name == "cp":
        return 2, controlled(u3_matrix(0.0, 0.0, p[0]), 1)
    if name == "cu3":
        return 2, controlled(u3_matrix(p[0], p[1], p[2]), 1)
    if name == "cu":
        theta, phi, lam, gamma = p[0], p[1], p[2], p[3]
        return 2, controlled(np.exp(1j * gamma) * u3_matrix(theta, phi, lam), 1)

    # -- Three-qubit gates --
    if name == "ccx":
        return 3, controlled(X, 2)
    if name == "cswap":
        return 3, controlled(SWAP, 1)
    if name == "mcx":
        n_qubits = num_qubits if num_qubits is not None else 2
        return n_qubits, controlled(X, n_qubits - 1)

    raise ValueError(f"Unsupported gate: {name}")


# --------------------------------------------------------------------------
# Gate decomposition
# --------------------------------------------------------------------------


def decompose(name: str, qubits: list[int]) -> list[tuple[str, list[float], list[int]]]:
    """Return a list of ``(gate_name, params, qubits)`` sub-instructions.

    Used for gates that are defined as sequences of simpler gates
    (consistent with qelib1.inc definitions).

    Raises:
        ValueError: If no decomposition is known for *name*.
    """
    if name == "dcx":
        a, b = qubits
        return [("cx", [], [a, b]), ("cx", [], [b, a])]

    if name == "rccx":
        # Exact qelib1.inc definition (simplified / relative-phase Toffoli)
        a, b, c = qubits
        return [
            ("u2", [0.0, np.pi], [c]),
            ("u1", [np.pi / 4], [c]),
            ("cx", [], [b, c]),
            ("u1", [-np.pi / 4], [c]),
            ("cx", [], [a, c]),
            ("u1", [np.pi / 4], [c]),
            ("cx", [], [b, c]),
            ("u1", [-np.pi / 4], [c]),
        ]

    raise ValueError(f"No decomposition known for {name}")
