"""Angle encoding for quantum data."""

from typing import Optional
import numpy as np
from ....circuit import Circuit


def angle_encode(
    data: np.ndarray,
    circuit: Circuit | None = None,
    qubits: list[int] | None = None,
) -> tuple[Circuit, list[int]]:
    """
    Encode classical data into quantum states using angle encoding.

    In angle encoding, each classical value is encoded as a rotation angle
    of a qubit using Ry gates.

    Args:
        data: Classical data to encode, should be normalized to [-pi, pi]
        circuit: Optional existing circuit to add encoding to
        qubits: Optional list of qubits to use for encoding

    Returns:
        tuple of (circuit, qubits used)
    """
    # Validate and reshape data
    data = np.asarray(data).ravel()
    n_qubits = len(data)

    # Create or validate circuit
    if circuit is None:
        circuit = Circuit(n_qubits)
    else:
        if circuit.num_qubits < n_qubits:
            raise ValueError(
                f"Circuit has {circuit.num_qubits} qubits but {n_qubits} needed"
            )

    # Use specified qubits or first n_qubits
    if qubits is None:
        qubits = list(range(n_qubits))
    elif len(qubits) < n_qubits:
        raise ValueError(f"Need {n_qubits} qubits but only {len(qubits)} provided")

    # Apply encoding
    for i, (qubit, value) in enumerate(zip(qubits, data)):
        circuit.ry(qubit, value)

    return circuit, qubits
