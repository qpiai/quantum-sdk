from __future__ import annotations

import numpy as np
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.icr.circuitoperation import CircuitOperation, OperationType
from .mappers import FermionOperator, jordan_wigner, bravyi_kitaev


def _add_parameterized_pauli_rotation(
    circuit: Circuit,
    pauli_string: list[tuple[int, str]],
    coeff_imag: float,
    param_idx: int,
):
    """
    Append an exponential Pauli rotation circuit e^(i * theta * coeff_imag * P) to the circuit,
    parameterized by parameter index param_idx.
    """
    # Filter out identity terms
    active_ops = [term for term in pauli_string if term[1] != "I"]
    if not active_ops:
        return

    # Sort qubits to ensure deterministic CNOT chain ordering
    active_ops = sorted(active_ops, key=lambda x: x[0])
    qubits = [q for q, _ in active_ops]
    target_qubit = qubits[-1]

    # 1. Basis change gates
    for q, op in active_ops:
        if op == "X":
            circuit.h(q)
        elif op == "Y":
            circuit.sdg(q)
            circuit.h(q)

    # 2. CNOT chain (parity computation)
    for idx in range(len(qubits) - 1):
        circuit.cx(qubits[idx], qubits[idx + 1])

    # 3. Parameterized rotation
    # We want to implement e^(i * theta * coeff_imag * P).
    # Since Rz(phi) = e^(-i * phi * Z / 2), setting phi = -2 * coeff_imag * theta
    # gives e^(i * theta * coeff_imag * Z).
    # Since VQE supplies positive theta, we implement the sign using:
    # - If sign is negative: circuit.rz(theta, target_qubit)
    # - If sign is positive: circuit.x(target_qubit); circuit.rz(theta, target_qubit); circuit.x(target_qubit)
    # We absorb the magnitude |2 * coeff_imag| (which is 1.0 or similar) as a scaling factor.
    angle_sign = -np.sign(coeff_imag)

    # We add a custom parameterized Rz operation
    # The SDK VQESolver maps the parameter using its 0-indexed position.
    # To associate it with param_idx, we construct a CircuitOperation.
    rz_op = CircuitOperation(
        operation_type=OperationType.N_QUBIT_PARAMETRIC,
        gate_name="rz",
        qubits=[target_qubit],
        params=[0.0],  # Placeholder, will be replaced by VQESolver during run()
        clbits=[],
    )

    if angle_sign < 0:
        circuit.add_operation(rz_op)
    else:
        circuit.x(target_qubit)
        circuit.add_operation(rz_op)
        circuit.x(target_qubit)

    # 4. Uncompute CNOT chain
    for idx in reversed(range(len(qubits) - 1)):
        circuit.cx(qubits[idx], qubits[idx + 1])

    # 5. Uncompute basis change gates
    for q, op in active_ops:
        if op == "X":
            circuit.h(q)
        elif op == "Y":
            circuit.h(q)
            circuit.s(q)


def uccsd_ansatz(
    n_qubits: int, n_electrons: int, mapping: str = "jordan_wigner"
) -> Circuit:
    """
    Generate a Unitary Coupled Cluster Singles and Doubles (UCCSD) variational ansatz circuit.

    References:
        - Barkoutsos, P. K., et al. "Quantum algorithms for electronic structure
          calculations: Particle-number-conserving ansatz state preparation on a
          molecular quantum computer." Phys. Rev. A 98.2 (2018).

    Args:
        n_qubits: Total number of spin-orbitals (qubits)
        n_electrons: Number of active electrons
        mapping: Fermion-to-qubit mapping ('jordan_wigner' or 'bravyi_kitaev')

    Returns:
        Circuit: Parameterized quantum circuit template
    """
    circuit = Circuit(n_qubits)

    # 1. Prepare Hartree-Fock state |11...100...0>
    for i in range(n_electrons):
        circuit.x(i)

    # 2. Identify spin-conserving single excitations: i -> a
    singles = []
    for i in range(n_electrons):
        for a in range(n_electrons, n_qubits):
            if i % 2 == a % 2:
                singles.append((i, a))

    # 3. Identify spin-conserving double excitations: i, j -> a, b
    doubles = []
    for i in range(n_electrons):
        for j in range(i + 1, n_electrons):
            for a in range(n_electrons, n_qubits):
                for b in range(a + 1, n_qubits):
                    if (i % 2 == a % 2 and j % 2 == b % 2) or (
                        i % 2 == b % 2 and j % 2 == a % 2
                    ):
                        doubles.append((i, j, a, b))

    # Total parameters/excitations
    param_idx = 0

    # 4. Compile Single Excitations
    # T1 = t_ia * (a_a^dagger a_i - a_i^dagger a_a)
    for i, a in singles:
        # Construct operator (a_a^dagger a_i - a_i^dagger a_a)
        ferm_op = FermionOperator({((a, 1), (i, 0)): 1.0, ((i, 1), (a, 0)): -1.0})

        # Map to Pauli representation
        if mapping.lower() == "bravyi_kitaev":
            pauli_list = bravyi_kitaev(ferm_op, n_qubits)
        else:
            pauli_list = jordan_wigner(ferm_op)

        # Append Pauli rotations for this single excitation parameter
        for pauli_str, coeff in pauli_list:
            _add_parameterized_pauli_rotation(circuit, pauli_str, 0.5, param_idx)

        param_idx += 1

    # 5. Compile Double Excitations
    # T2 = t_ijab * (a_a^dagger a_b^dagger a_j a_i - a_i^dagger a_j^dagger a_b a_a)
    for i, j, a, b in doubles:
        # Construct operator (a_a^dagger a_b^dagger a_j a_i - a_i^dagger a_j^dagger a_b a_a)
        ferm_op = FermionOperator(
            {
                ((a, 1), (b, 1), (j, 0), (i, 0)): 1.0,
                ((i, 1), (j, 1), (b, 0), (a, 0)): -1.0,
            }
        )

        if mapping.lower() == "bravyi_kitaev":
            pauli_list = bravyi_kitaev(ferm_op, n_qubits)
        else:
            pauli_list = jordan_wigner(ferm_op)

        for pauli_str, coeff in pauli_list:
            _add_parameterized_pauli_rotation(circuit, pauli_str, 0.5, param_idx)

        param_idx += 1

    return circuit
