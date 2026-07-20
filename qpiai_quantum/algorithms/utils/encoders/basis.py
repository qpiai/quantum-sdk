"""Basis encoding for quantum data."""

from typing import Optional
import numpy as np
from ....circuit import Circuit


def basis_encode(
    data: np.ndarray,
    circuit: Circuit | None = None,
    qubits: list[int] | None = None,
) -> tuple[Circuit, list[int]]:
    """
    Encode classical data into quantum states using basis encoding.

    In basis encoding, classical data is encoded into computational basis states.
    Each qubit represents a binary feature.

    Args:
        data: Binary data to encode (should be 0s and 1s)
        circuit: Optional existing circuit to add encoding to
        qubits: Optional list of qubits to use for encoding

    Returns:
        tuple of (circuit, qubits used)
    """
    # Validate data
    data = np.asarray(data).ravel()
    if not np.all(np.logical_or(data == 0, data == 1)):
        raise ValueError("Data must be binary (0s and 1s)")

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

    # Apply X gates for 1s
    for i, (qubit, value) in enumerate(zip(qubits, data)):
        if value == 1:
            circuit.x(qubit)

    return circuit, qubits
