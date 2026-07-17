from __future__ import annotations
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.icr.circuitoperation import CircuitOperation, OperationType
from .lattices import Lattice


def _add_pauli_xx_rotation(circuit: Circuit, u: int, v: int):
    """Apply e^(-i * theta * X_u X_v) using parameterized Rz."""
    circuit.h(u)
    circuit.h(v)
    circuit.cx(u, v)

    # Parametric RZ
    rz_op = CircuitOperation(
        operation_type=OperationType.N_QUBIT_PARAMETRIC,
        gate_name="rz",
        qubits=[v],
        params=[0.0],
        clbits=[],
    )
    circuit.add_operation(rz_op)

    circuit.cx(u, v)
    circuit.h(u)
    circuit.h(v)


def _add_pauli_yy_rotation(circuit: Circuit, u: int, v: int):
    """Apply e^(-i * theta * Y_u Y_v) using parameterized Rz."""
    circuit.sdg(u)
    circuit.sdg(v)
    circuit.h(u)
    circuit.h(v)
    circuit.cx(u, v)

    rz_op = CircuitOperation(
        operation_type=OperationType.N_QUBIT_PARAMETRIC,
        gate_name="rz",
        qubits=[v],
        params=[0.0],
        clbits=[],
    )
    circuit.add_operation(rz_op)

    circuit.cx(u, v)
    circuit.h(u)
    circuit.h(v)
    circuit.s(u)
    circuit.s(v)


def _add_pauli_zz_rotation(circuit: Circuit, u: int, v: int):
    """Apply e^(-i * theta * Z_u Z_v) using parameterized Rz."""
    circuit.cx(u, v)

    rz_op = CircuitOperation(
        operation_type=OperationType.N_QUBIT_PARAMETRIC,
        gate_name="rz",
        qubits=[v],
        params=[0.0],
        clbits=[],
    )
    circuit.add_operation(rz_op)

    circuit.cx(u, v)


def heisenberg_hva_ansatz(lattice: Lattice, layers: int = 1) -> Circuit:
    """
    Generate a Hamiltonian Variational Ansatz (HVA) circuit for the Heisenberg XYZ model.

    References:
        - Cade, C., et al. "Strategies for solving the Fermi-Hubbard model on near-term
          quantum computers." Phys. Rev. B 102.23 (2020).

    Args:
        lattice: The lattice structure defining system connections
        layers: Number of Trotter-like variational layers (default: 1)

    Returns:
        Circuit: Parameterized quantum circuit template
    """
    n_qubits = lattice.n_sites
    circuit = Circuit(n_qubits)
    edges = lattice.get_edges(pbc=False)

    # Prepare a simple initial superposition state (e.g. Hadamard on all qubits)
    for i in range(n_qubits):
        circuit.h(i)

    for _ in range(layers):
        # 1. XX rotations on all edges
        for u, v in edges:
            _add_pauli_xx_rotation(circuit, u, v)

        # 2. YY rotations on all edges
        for u, v in edges:
            _add_pauli_yy_rotation(circuit, u, v)

        # 3. ZZ rotations on all edges
        for u, v in edges:
            _add_pauli_zz_rotation(circuit, u, v)

    return circuit
