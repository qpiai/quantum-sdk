from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ....circuit import Circuit
from ....circuit.quantumregister import QuantumRegister


def standard_vqe_ansatz(
    n_qubits: int, layers: int = 2, entanglement: str = "linear"
) -> Circuit:
    """
    Create a standard VQE ansatz using hardware-efficient structure.

    This ansatz uses RY and RZ rotations with entangling layers,
    compatible with the QpiAI Quantum SDK's Circuit class.

    Args:
        n_qubits: Number of qubits
        layers: Number of ansatz layers
        entanglement: Entanglement pattern ('linear', 'circular', 'full')

    Returns:
        Parameterized quantum circuit compatible with SDK
    """
    qr = QuantumRegister(n_qubits, "q")
    circuit = Circuit(qr)

    # Initial layer
    for i in range(n_qubits):
        circuit.ry(i, 0.0)  # Parameterized by optimizer
        circuit.rz(i, 0.0)

    # Repeated layers
    for layer in range(layers):
        # Entangling layer
        if entanglement == "linear":
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)
        elif entanglement == "circular":
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)
            if n_qubits > 2:
                circuit.cx(n_qubits - 1, 0)
        elif entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    circuit.cx(i, j)

        # Rotation layer (except last layer - optional)
        if layer < layers - 1 or True:  # Always add rotations
            for i in range(n_qubits):
                circuit.ry(i, 0.0)
                circuit.rz(i, 0.0)

    return circuit


def standard_qaoa_ansatz(
    n_qubits: int, hamiltonian: Any, layers: int = 1, mixer: str = "x"
) -> Circuit:
    """
    Create a standard QAOA ansatz using SDK's Circuit class.

    This implements the QAOA circuit with:
    - Initial superposition state (H on all qubits)
    - Alternating cost and mixer layers
    - Parameterized evolution angles (gamma, beta)

    Args:
        n_qubits: Number of qubits
        hamiltonian: Problem Hamiltonian (must have get_hamiltonian_terms method)
        layers: Number of QAOA layers (p parameter)
        mixer: Mixer Hamiltonian type ('x' for X-mixer, 'xy' for XY-mixer)

    Returns:
        Parameterized QAOA circuit compatible with SDK

    Note:
        This function creates 2*layers parameters (gamma_i, beta_i for each layer).
        The parameters should be provided in order: [gamma_0, beta_0, gamma_1, beta_1, ...]
    """
    qr = QuantumRegister(n_qubits, "q")
    circuit = Circuit(qr)

    # Initial state: uniform superposition
    for i in range(n_qubits):
        circuit.h(i)

    # Get Hamiltonian terms
    if not hasattr(hamiltonian, "get_hamiltonian_terms"):
        raise ValueError("Hamiltonian must have get_hamiltonian_terms() method")

    terms = hamiltonian.get_hamiltonian_terms()

    # QAOA layers
    for layer in range(layers):
        # Cost Hamiltonian layer (problem-specific)
        # Each term in the Hamiltonian gets exp(-i * gamma * term)
        for ops, coeff in terms:
            if not ops:  # Constant term, skip
                continue

            # Apply cost layer evolution
            _apply_cost_evolution(circuit, ops, coeff, parameter_value=0.0)

        # Mixer Hamiltonian layer
        if mixer == "x":
            # X-mixer: exp(-i * beta * sum(X_i))
            for i in range(n_qubits):
                circuit.rx(i, 0.0)  # Parameterized by optimizer (beta)
        elif mixer == "xy":
            # XY-mixer: more complex mixing
            for i in range(n_qubits - 1):
                circuit.rx(i, 0.0)
                circuit.ry(i, 0.0)
            circuit.rx(n_qubits - 1, 0.0)
            circuit.ry(n_qubits - 1, 0.0)
        else:
            raise ValueError(f"Unknown mixer type: {mixer}")

    return circuit


def _apply_cost_evolution(
    circuit: Circuit,
    ops: List[Tuple[int, str]],
    coeff: float,
    parameter_value: float = 0.0,
):
    """
    Apply cost evolution for a single Hamiltonian term.

    For a term like Z_i Z_j, this applies exp(-i * gamma * coeff * Z_i Z_j).

    Args:
        circuit: Circuit to modify
        ops: List of (qubit_index, pauli_op) tuples
        coeff: Coefficient of this term
        parameter_value: Value of gamma parameter (0.0 means parameterized)
    """
    if not ops:
        return

    # Extract qubits and operators
    qubits = [qi for qi, _ in ops]
    operators = [op for _, op in ops]

    # Handle different Pauli string types
    if all(op == "Z" for op in operators):
        # Pure Z terms: use RZZ or RZ gates
        if len(qubits) == 1:
            # Single Z term: RZ gate
            circuit.rz(qubits[0], parameter_value)  # Will be parameterized
        elif len(qubits) == 2:
            # ZZ term: RZZ gate
            circuit.rzz(qubits[0], qubits[1], parameter_value)
        else:
            # Multi-qubit Z term: decompose into CNOTs and RZ
            # Z_i Z_j Z_k = CNOT(j,i) CNOT(k,i) RZ(i) CNOT(k,i) CNOT(j,i)
            for i in range(1, len(qubits)):
                circuit.cx(qubits[i], qubits[0])
            circuit.rz(qubits[0], parameter_value)
            for i in range(len(qubits) - 1, 0, -1):
                circuit.cx(qubits[i], qubits[0])

    elif any(op in ["X", "Y"] for op in operators):
        # Mixed Pauli terms: need basis rotations
        # Apply basis change
        for qi, op in ops:
            if op == "X":
                circuit.h(qi)
            elif op == "Y":
                circuit.sdg(qi)
                circuit.h(qi)

        # Apply Z evolution
        if len(qubits) == 1:
            circuit.rz(qubits[0], parameter_value)
        elif len(qubits) == 2:
            circuit.rzz(qubits[0], qubits[1], parameter_value)
        else:
            for i in range(1, len(qubits)):
                circuit.cx(qubits[i], qubits[0])
            circuit.rz(qubits[0], parameter_value)
            for i in range(len(qubits) - 1, 0, -1):
                circuit.cx(qubits[i], qubits[0])

        # Undo basis change
        for qi, op in reversed(ops):
            if op == "X":
                circuit.h(qi)
            elif op == "Y":
                circuit.h(qi)
                circuit.s(qi)


def qaoa_ansatz_from_problem(problem: Any, layers: int = 1) -> Circuit:
    """
    Convenience function to create QAOA ansatz directly from a problem.

    Args:
        problem: Problem instance with n_qubits and get_hamiltonian_terms()
        layers: Number of QAOA layers

    Returns:
        QAOA circuit
    """
    if not hasattr(problem, "n_qubits"):
        raise ValueError("Problem must have n_qubits attribute")

    return standard_qaoa_ansatz(
        n_qubits=problem.n_qubits, hamiltonian=problem, layers=layers
    )


def count_parameters(circuit: Circuit) -> int:
    """
    Count the number of parameterized gates in a circuit.

    Args:
        circuit: Quantum circuit

    Returns:
        Number of parameters
    """
    from ....icr.circuitoperation import OperationType, CircuitOperation

    if not hasattr(circuit, "icr") or not circuit.icr:
        return 0

    if not hasattr(circuit.icr, "evolve"):
        return 0

    return sum(
        1
        for op in circuit.icr.evolve
        if isinstance(op, CircuitOperation)
        and op.operation_type == OperationType.N_QUBIT_PARAMETRIC
    )
