"""
Density Matrix State & Local Simulator
=======================================
Implements the DensityMatrixState representation and the DensityMatrixSimulator
backend for executing quantum circuits using density matrix formalism.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
import numpy as np

from qpiai_quantum.icr.circuitoperation import OperationType
from qpiai_quantum.simulator.base_simulator import BaseSimulator
from qpiai_quantum.simulator.gates import (
    ALL_KNOWN_GATES,
    DECOMPOSED_GATES,
    H,
    S,
    SDG,
    SWAP,
    T,
    TDG,
    X,
    Y,
    Z,
    controlled,
    decompose,
    gate_spec,
    rx_matrix,
    ry_matrix,
    rz_matrix,
)
from qpiai_quantum.simulator.result import QasmSimulatorResult
from qpiai_quantum.simulator.state import QuantumState

if TYPE_CHECKING:
    from qpiai_quantum.circuit import Circuit


class DensityMatrixState(QuantumState):
    """
    Density matrix representation of a quantum state (2^n x 2^n complex matrix).

    Supports pure and mixed quantum states, single- and multi-qubit unitary
    evolutions, Kraus noise channels, partial traces, and measurement statistics.
    """

    def __init__(self, num_qubits: int, data: np.ndarray | None = None) -> None:
        """
        Initialize a DensityMatrixState for n qubits.

        Args:
            num_qubits: Number of qubits.
            data: Optional 2D complex numpy array of shape (2^n, 2^n).
                  Defaults to the pure zero state |0...0><0...0|.
        """
        if num_qubits <= 0:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")

        self._num_qubits = num_qubits
        dim = 1 << num_qubits

        if data is None:
            self._data = np.zeros((dim, dim), dtype=complex)
            self._data[0, 0] = 1.0 + 0.0j
        else:
            arr = np.asarray(data, dtype=complex)
            if arr.ndim == 1:
                # Convert statevector to density matrix
                if arr.shape[0] != dim:
                    raise ValueError(
                        f"Statevector size {arr.shape[0]} does not match 2^{num_qubits} = {dim}"
                    )
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                self._data = np.outer(arr, np.conj(arr))
            elif arr.ndim == 2:
                if arr.shape != (dim, dim):
                    raise ValueError(
                        f"Density matrix shape {arr.shape} does not match ({dim}, {dim})"
                    )
                self._data = arr.copy()
            else:
                raise ValueError(
                    f"Expected 1D statevector or 2D density matrix, got {arr.ndim}D array"
                )

    @property
    def num_qubits(self) -> int:
        """Number of qubits represented in the density matrix."""
        return self._num_qubits

    @property
    def data(self) -> np.ndarray:
        """Underlying 2D numpy array representing the density matrix."""
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        dim = 1 << self._num_qubits
        arr = np.asarray(new_data, dtype=complex)
        if arr.shape != (dim, dim):
            raise ValueError(f"Shape must be ({dim}, {dim}), got {arr.shape}")
        self._data = arr

    # ----------------------------------------------------------------------
    # Factory Constructors
    # ----------------------------------------------------------------------

    @classmethod
    def pure_state(cls, statevector: np.ndarray | list[complex]) -> DensityMatrixState:
        """
        Construct a DensityMatrixState from a pure statevector.

        Args:
            statevector: 1D array of amplitudes of length 2^n.
        """
        sv = np.asarray(statevector, dtype=complex)
        dim = sv.shape[0]
        num_qubits = int(np.log2(dim))
        if 1 << num_qubits != dim:
            raise ValueError(f"Statevector length {dim} must be a power of 2.")
        return cls(num_qubits=num_qubits, data=sv)

    @classmethod
    def maximally_mixed(cls, num_qubits: int) -> DensityMatrixState:
        """
        Construct a maximally mixed state ρ = I / 2^n.

        Args:
            num_qubits: Number of qubits.
        """
        dim = 1 << num_qubits
        matrix = np.eye(dim, dtype=complex) / dim
        return cls(num_qubits=num_qubits, data=matrix)

    # ----------------------------------------------------------------------
    # Single-Qubit Gates
    # ----------------------------------------------------------------------

    def apply_h(self, target: int) -> None:
        """Apply Hadamard gate to target qubit."""
        self.apply_unitary(H, [target])

    def apply_x(self, target: int) -> None:
        """Apply Pauli-X gate to target qubit."""
        self.apply_unitary(X, [target])

    def apply_y(self, target: int) -> None:
        """Apply Pauli-Y gate to target qubit."""
        self.apply_unitary(Y, [target])

    def apply_z(self, target: int) -> None:
        """Apply Pauli-Z gate to target qubit."""
        self.apply_unitary(Z, [target])

    def apply_s(self, target: int) -> None:
        """Apply Phase (S) gate to target qubit."""
        self.apply_unitary(S, [target])

    def apply_sdg(self, target: int) -> None:
        """Apply S† gate to target qubit."""
        self.apply_unitary(SDG, [target])

    def apply_t(self, target: int) -> None:
        """Apply T gate to target qubit."""
        self.apply_unitary(T, [target])

    def apply_tdg(self, target: int) -> None:
        """Apply T† gate to target qubit."""
        self.apply_unitary(TDG, [target])

    def apply_rx(self, target: int, theta: float) -> None:
        """Apply RX rotation gate to target qubit."""
        self.apply_unitary(rx_matrix(theta), [target])

    def apply_ry(self, target: int, theta: float) -> None:
        """Apply RY rotation gate to target qubit."""
        self.apply_unitary(ry_matrix(theta), [target])

    def apply_rz(self, target: int, theta: float) -> None:
        """Apply RZ rotation gate to target qubit."""
        self.apply_unitary(rz_matrix(theta), [target])

    # ----------------------------------------------------------------------
    # Multi-Qubit Gates
    # ----------------------------------------------------------------------

    def apply_cnot(self, control: int, target: int) -> None:
        """Apply CNOT (CX) gate between control and target qubits."""
        self.apply_unitary(controlled(X, 1), [control, target])

    def apply_cz(self, control: int, target: int) -> None:
        """Apply Controlled-Z (CZ) gate between control and target qubits."""
        self.apply_unitary(controlled(Z, 1), [control, target])

    def apply_swap(self, target1: int, target2: int) -> None:
        """Apply SWAP gate between target1 and target2 qubits."""
        self.apply_unitary(SWAP, [target1, target2])

    def apply_cswap(self, control: int, target1: int, target2: int) -> None:
        """Apply Fredkin (CSWAP) gate."""
        self.apply_unitary(controlled(SWAP, 1), [control, target1, target2])

    # ----------------------------------------------------------------------
    # Generic Unitary & Kraus Transformations
    # ----------------------------------------------------------------------

    def apply_unitary(self, matrix: np.ndarray, qubits: list[int]) -> None:
        """
        Apply a unitary transformation U to specified qubits: ρ -> U ρ U†.

        Args:
            matrix: 2^k x 2^k complex matrix where k = len(qubits).
            qubits: List of qubit indices.
        """
        n = self._num_qubits
        k = len(qubits)
        if matrix.shape != (2**k, 2**k):
            raise ValueError(
                f"Matrix shape {matrix.shape} does not match 2^{k} x 2^{k}"
            )
        for q in qubits:
            if not (0 <= q < n):
                raise ValueError(f"Qubit index {q} out of bounds for n={n}")

        self._data = self._transform_density_matrix(self._data, n, qubits, matrix)

    def apply_kraus(self, kraus_ops: list[np.ndarray], qubits: list[int]) -> None:
        """
        Apply a quantum channel represented by Kraus operators: ρ -> Σ_i K_i ρ K_i†.

        Args:
            kraus_ops: List of Kraus matrices K_i of shape (2^k, 2^k).
            qubits: List of target qubits.
        """
        n = self._num_qubits
        dim = 1 << n
        new_rho = np.zeros((dim, dim), dtype=complex)
        for K in kraus_ops:
            new_rho += self._transform_density_matrix(self._data, n, qubits, K)
        self._data = new_rho

    @staticmethod
    def _transform_density_matrix(
        rho: np.ndarray, n: int, qubits: list[int], M: np.ndarray
    ) -> np.ndarray:
        """
        Transform density matrix by operator M: ρ -> M ρ M† on target qubits.
        """
        k = len(qubits)
        # Reshape into 2n tensor: n ket axes followed by n bra axes
        tensor = rho.reshape([2] * (2 * n))

        # Little-endian convention: qubit q corresponds to axis (n - 1 - q)
        row_axes = [n - 1 - q for q in qubits]
        col_axes = [n + (n - 1 - q) for q in qubits]

        # 1. Apply M on row (ket) indices: (M @ flat_row)
        tensor = np.moveaxis(tensor, row_axes, list(range(k)))
        shape_row = tensor.shape
        flat_row = tensor.reshape(2**k, -1)
        flat_row = M @ flat_row
        tensor = flat_row.reshape(shape_row)
        tensor = np.moveaxis(tensor, list(range(k)), row_axes)

        # 2. Apply M* on col (bra) indices: equivalent to ρ' @ M†
        tensor = np.moveaxis(tensor, col_axes, list(range(k)))
        shape_col = tensor.shape
        flat_col = tensor.reshape(2**k, -1)
        flat_col = np.conj(M) @ flat_col
        tensor = flat_col.reshape(shape_col)
        tensor = np.moveaxis(tensor, list(range(k)), col_axes)

        dim = 1 << n
        return tensor.reshape((dim, dim))

    # ----------------------------------------------------------------------
    # Quantum Information Diagnostics & Operations
    # ----------------------------------------------------------------------

    def partial_trace(self, qubits_to_trace: list[int]) -> DensityMatrixState:
        """
        Compute the reduced density matrix by tracing out the specified qubits.

        Args:
            qubits_to_trace: Qubits to trace out.

        Returns:
            A new DensityMatrixState for the remaining subsystem.
        """
        n = self._num_qubits
        for q in qubits_to_trace:
            if not (0 <= q < n):
                raise ValueError(f"Qubit index {q} out of bounds for n={n}")

        remaining_qubits = [q for q in range(n) if q not in qubits_to_trace]
        if not remaining_qubits:
            raise ValueError("Cannot trace out all qubits in partial_trace.")

        tensor = self._data.reshape([2] * (2 * n))

        # For each qubit to trace, contract its row axis with its col axis
        traced_tensor = tensor
        # We trace out one by one from highest qubit index down to keep indices consistent
        for q in sorted(qubits_to_trace, reverse=True):
            current_n = traced_tensor.ndim // 2
            # Find axis of q in current subsystem
            row_axis = current_n - 1 - q
            col_axis = current_n + row_axis
            traced_tensor = np.trace(traced_tensor, axis1=row_axis, axis2=col_axis)

        rem_n = len(remaining_qubits)
        rem_dim = 1 << rem_n
        reduced_matrix = traced_tensor.reshape((rem_dim, rem_dim))
        return DensityMatrixState(num_qubits=rem_n, data=reduced_matrix)

    def purity(self) -> float:
        """Return purity γ = Tr(ρ²)."""
        return float(np.real(np.trace(self._data @ self._data)))

    def von_neumann_entropy(self) -> float:
        """Return von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ)."""
        eigenvals = np.linalg.eigvalsh(self._data)
        eigenvals = eigenvals[eigenvals > 1e-15]
        return float(-np.sum(eigenvals * np.log2(eigenvals)))

    def get_probabilities(self) -> np.ndarray:
        """Return measurement probabilities for all 2^n computational basis states."""
        probs = np.real(np.diag(self._data))
        # Clamp negative rounding errors
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return probs

    def reset(self) -> None:
        """Reset the density matrix state back to |0...0><0...0|."""
        dim = 1 << self._num_qubits
        self._data = np.zeros((dim, dim), dtype=complex)
        self._data[0, 0] = 1.0 + 0.0j


class DensityMatrixSimulator(BaseSimulator):
    """
    Local Density Matrix Quantum Simulator.

    Simulates quantum circuits using the density matrix formalism, supporting
    both pure states and mixed states/noise channels.
    """

    @property
    def name(self) -> str:
        """Name of the simulator backend."""
        return "QpiAI-QDM-Local"

    def run(
        self,
        circuit: "Circuit",
        shots: int = 1024,
        seed: int | None = None,
        name: str | None = None,
        initial_state: np.ndarray | DensityMatrixState | None = None,
    ) -> QasmSimulatorResult:
        """
        Execute a quantum circuit using density matrix simulation.

        Args:
            circuit: The quantum circuit to execute.
            shots: Number of measurement shots to sample.
            seed: Optional RNG seed.
            name: Optional name for the result object.
            initial_state: Optional initial density matrix or statevector.

        Returns:
            QasmSimulatorResult containing density matrix and measurement counts.
        """
        start_time = time.perf_counter()
        n_qubits = circuit.num_qubits
        n_cbits = circuit.num_clbits

        if n_qubits <= 0:
            raise ValueError(
                f"Circuit must have at least 1 qubit, got {n_qubits} qubits"
            )

        # Initialize density matrix state
        if initial_state is None:
            state = DensityMatrixState(n_qubits)
        elif isinstance(initial_state, DensityMatrixState):
            if initial_state.num_qubits != n_qubits:
                raise ValueError(
                    f"initial_state qubits {initial_state.num_qubits} != circuit qubits {n_qubits}"
                )
            state = DensityMatrixState(n_qubits, data=initial_state.data)
        else:
            state = DensityMatrixState(n_qubits, data=initial_state)

        measure_map: dict[int, int] = {}

        def _apply_gate(gate_name: str, params: list[float], qubits: list[int]) -> None:
            norm_name = gate_name.lower()
            if norm_name in DECOMPOSED_GATES:
                for sub_name, sub_p, sub_q in decompose(norm_name, qubits):
                    _apply_gate(sub_name, sub_p, sub_q)
                return

            if norm_name in ALL_KNOWN_GATES:
                _, matrix = gate_spec(norm_name, params, len(qubits))
                state.apply_unitary(matrix, qubits)
            elif norm_name == "barrier":
                pass
            else:
                raise ValueError(f"Unknown gate '{gate_name}'")

        def _apply_op(op) -> None:
            if op.operation_type == OperationType.MEASURE:
                if op.qubits and op.clbits:
                    for q, c in zip(op.qubits, op.clbits):
                        measure_map[q] = c
            elif op.operation_type in (
                OperationType.N_QUBIT_NON_PARAMETRIC,
                OperationType.N_QUBIT_PARAMETRIC,
                OperationType.SWAP,
            ):
                _apply_gate(op.gate_name, op.params or [], op.qubits)
            elif op.operation_type == OperationType.OPERATION:
                if hasattr(op, "order") and op.order is not None:
                    for sub_op in op.order:
                        _apply_op(sub_op)
                else:
                    _apply_gate(op.gate_name, op.params or [], op.qubits)
            else:
                raise ValueError(f"Unsupported operation type: {op.operation_type}")

        # Execute circuit evolution
        for op in circuit.icr.evolve:
            _apply_op(op)

        # Sample measurement counts if measurements were registered
        probs = state.get_probabilities()
        if n_cbits > 0 and measure_map:
            counts = self._sample_counts(
                probs, n_qubits, n_cbits, measure_map, shots, seed
            )
        else:
            counts = {}

        elapsed_time = time.perf_counter() - start_time

        return QasmSimulatorResult(
            name=name or circuit.name,
            counts=counts,
            statevector=None,
            density_matrix=state.data.tolist(),
            shots=shots,
            execution_time=elapsed_time,
            method="density_matrix",
            job_status="completed",
            n_qubits=n_qubits,
            n_cbits=n_cbits,
        )

    @staticmethod
    def _sample_counts(
        probs: np.ndarray,
        n_qubits: int,
        n_cbits: int,
        measure_map: dict[int, int],
        shots: int,
        seed: int | None = None,
    ) -> dict[str, int]:
        """Sample measurement outcomes from probability distribution."""
        rng = np.random.default_rng(seed)
        outcomes = rng.choice(len(probs), size=shots, p=probs)

        counts: dict[str, int] = {}
        for outcome in outcomes:
            cbits = ["0"] * n_cbits
            for qubit, cbit in measure_map.items():
                bit = (outcome >> qubit) & 1
                cbits[cbit] = str(bit)
            bitstring = "".join(reversed(cbits)) if n_cbits > 0 else ""
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts
