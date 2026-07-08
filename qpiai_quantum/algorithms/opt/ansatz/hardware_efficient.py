from typing import List
from ....circuit import Circuit
from ....circuit.quantumregister import QuantumRegister


def hardware_efficient_ansatz(
    n_qubits: int, depth: int = 2, entanglement: str = "linear"
) -> Circuit:
    """
    Create a hardware-efficient ansatz.

    Args:
        n_qubits: Number of qubits
        depth: Number of layers
        entanglement: 'linear', 'circular', or 'full'

    Returns:
        Parameterized quantum circuit
    """
    qr = QuantumRegister(n_qubits, "q")
    circuit = Circuit(qr)

    param_count = 0

    # NOTE : Initial layer of single-qubit rotations
    for i in range(n_qubits):
        circuit.ry(i, 0.0)  # NOTE: To be parameterized by optimizer
        param_count += 1

    # NOTE: Alternating layers of entanglement and rotations
    for layer in range(depth):
        if entanglement == "linear":
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)
        elif entanglement == "circular":
            for i in range(n_qubits):
                circuit.cx(i, (i + 1) % n_qubits)
        elif entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    circuit.cz(i, j)

        for i in range(n_qubits):
            circuit.ry(i, 0.0)  # NOTE: To be parameterized by optimizer
            param_count += 1
            circuit.rz(i, 0.0)  # NOTE: To be parameterized by optimizer
            param_count += 1

    return circuit


def two_local_ansatz(
    n_qubits: int,
    rotation_blocks: str = "ry",
    entanglement_blocks: str = "cz",
    reps: int = 3,
) -> Circuit:
    """
    Create a two-local ansatz with custom rotation and entanglement blocks.

    Args:
        n_qubits: Number of qubits
        rotation_blocks: Single-qubit gates ('ry', 'rx', 'rz')
        entanglement_blocks: Two-qubit gates ('cx', 'cy', 'cz')
        reps: Number of repetitions

    Returns:
        Parameterized quantum circuit
    """
    qr = QuantumRegister(n_qubits, "q")
    circuit = Circuit(qr)
    param_count = 0

    for i in range(n_qubits):
        if rotation_blocks == "ry":
            circuit.ry(i, 0.0)  # NOTE: To be parameterized by optimizer
        elif rotation_blocks == "rx":
            circuit.rx(i, 0.0)  # NOTE: To  be parameterized by optimizer
        elif rotation_blocks == "rz":
            circuit.rz(i, 0.0)  # NOTE: To be parameterized by optimizer
        param_count += 1

    for _ in range(reps):
        for i in range(n_qubits - 1):
            if entanglement_blocks == "cz":
                circuit.cz(i, i + 1)
            elif entanglement_blocks == "cx":
                circuit.cx(i, i + 1)
            elif entanglement_blocks == "cy":
                circuit.cy(i, i + 1)

        for i in range(n_qubits):
            if rotation_blocks == "ry":
                circuit.ry(i, 0.0)  # NOTE: To be parameterized by optimizer
            elif rotation_blocks == "rx":
                circuit.rx(i, 0.0)  # NOTE: To be parameterized by optimizer
            elif rotation_blocks == "rz":
                circuit.rz(i, 0.0)  # NOTE: To be parameterized by optimizer
            param_count += 1

    return circuit
