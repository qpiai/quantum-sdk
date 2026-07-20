"""Amplitude encoding for quantum data."""

from typing import Optional
import numpy as np
from ....circuit import Circuit


def amplitude_encode(
    data: np.ndarray,
    circuit: Circuit | None = None,
    qubits: list[int] | None = None,
) -> tuple[Circuit, list[int]]:
    """
    Encode classical data into quantum states using amplitude encoding.

    In amplitude encoding, classical data is encoded into the amplitudes
    of the quantum state directly.

    Args:
        data: Classical data to encode, should be normalized
        circuit: Optional existing circuit to add encoding to
        qubits: Optional list of qubits to use for encoding

    Returns:
        tuple of (circuit, qubits used)
    """
    # Validate and normalize data
    data = np.asarray(data).ravel()
    norm = np.linalg.norm(data)
    if norm > 0:
        data = data / norm

    n_features = len(data)
    n_qubits = int(np.ceil(np.log2(n_features)))

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

    # Pad data to power of 2
    n_padded = 1 << n_qubits
    if len(data) < n_padded:
        data = np.pad(data, (0, n_padded - len(data)))

    # For now, just use simple basis encoding with probabilities
    # In the future, implement proper amplitude encoding with isometries
    for i, amp in enumerate(data):
        if abs(amp) > 0:
            # Convert i to bit string and apply X gates
            bits = format(i, f"0{n_qubits}b")
            for j, bit in enumerate(bits):
                if bit == "1":
                    circuit.x(qubits[j])
            # Apply controlled amplitude
            if i < len(data) - 1:
                circuit.ry(qubits[-1], 2 * np.arccos(abs(amp)))

    return circuit, qubits
