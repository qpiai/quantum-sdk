"""
Statevector Simulator
=====================
A self-contained statevector simulator that executes QpiAI Quantum
circuits locally using NumPy. It implements the BaseSimulator interface
and directly consumes IntermediateCircuitRepresentation objects without
needing QASM compilation.
"""

import time
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from qpiai_quantum.icr.circuitoperation import OperationType

if TYPE_CHECKING:
    from qpiai_quantum.circuit import Circuit

from .base_simulator import BaseSimulator
from .gates import DECOMPOSED_GATES, decompose, gate_spec
from .result import QasmSimulatorResult


class StatevectorSimulator(BaseSimulator):
    """
    Local statevector simulator for QpiAI Quantum circuits.
    Executes directly from the ICR (Intermediate Circuit Representation).
    """

    @property
    def name(self) -> str:
        return "QpiAI-QSV-Local"

    def run(
        self,
        circuit: "Circuit",
        shots: int = 1024,
        seed: int | None = None,
        name: str | None = None,
        initial_state: np.ndarray | None = None,
    ) -> QasmSimulatorResult:
        """
        Execute the given circuit locally using statevector simulation.

        Args:
            circuit: The quantum circuit to execute.
            shots: Number of measurement shots to perform.
            seed: Optional RNG seed for reproducibility.
            name: Optional name for the result object.
            initial_state: Optional initial statevector for the simulation.

        Returns:
            A QasmSimulatorResult object containing counts and statevector.
        """
        start_time = time.perf_counter()

        n_qubits = circuit.num_qubits
        n_cbits = circuit.num_clbits

        if n_qubits == 0:
            raise ValueError("Cannot simulate a circuit with 0 qubits.")

        dim = 2**n_qubits
        if initial_state is not None:
            if len(initial_state) != dim:
                raise ValueError(
                    f"Initial state dimension {len(initial_state)} does not match circuit qubit dimension {dim}."
                )
            if not np.isclose(np.linalg.norm(initial_state), 1.0):
                raise ValueError(
                    "Initial state vector must be normalized (norm must be 1.0)."
                )
            state = np.array(initial_state, dtype=complex)
        else:
            # Initialize state to |0...0>
            state = np.zeros(dim, dtype=complex)
            state[0] = 1.0

        measure_map: dict[int, int] = {}

        # Define internal applicator for recursive decomposition handling
        def _apply_gate(gate_name: str, params: list[float], qubits: list[int]) -> None:
            nonlocal state
            gate_name_lower = gate_name.lower()

            if gate_name_lower in DECOMPOSED_GATES:
                for sub_name, sub_params, sub_qubits in decompose(
                    gate_name_lower, qubits
                ):
                    _apply_gate(sub_name, sub_params, sub_qubits)
                return

            _, U = gate_spec(gate_name_lower, params, num_qubits=len(qubits))
            state = self._apply_unitary(state, n_qubits, qubits, U)

        # Define internal applicator for recursive operation handling
        def _apply_op(op) -> None:
            nonlocal state
            if op.operation_type == OperationType.BARRIER:
                return

            elif op.operation_type == OperationType.MEASURE:
                # Store the last measurement targeting this classical bit
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
                # If it's a composite gate with sub-operations, apply recursively
                if hasattr(op, "order") and op.order is not None:
                    for sub_op in op.order:
                        _apply_op(sub_op)
                else:
                    _apply_gate(op.gate_name, op.params or [], op.qubits)

            else:
                # Handle reset if we eventually add it to OperationType
                if op.gate_name.lower() == "reset" and op.qubits:
                    for q in op.qubits:
                        state = self._apply_reset(state, n_qubits, q)
                else:
                    raise ValueError(f"Unsupported operation type: {op.operation_type}")

        # Iterate directly through the circuit's evolution list
        for op in circuit.icr.evolve:
            _apply_op(op)

        # Sample measurements if requested
        if n_cbits > 0 and measure_map:
            counts = self._sample_counts(
                state, n_qubits, n_cbits, measure_map, shots, seed
            )
        else:
            counts = {}

        elapsed_time = time.perf_counter() - start_time

        return QasmSimulatorResult(
            name=name or circuit.name,
            counts=counts,
            statevector=state.tolist(),
            shots=shots,
            execution_time=elapsed_time,
            method="statevector",
            job_status="completed",
            n_qubits=n_qubits,
            n_cbits=n_cbits,
        )

    # ----------------------------------------------------------------------
    # Internal math utilities
    # ----------------------------------------------------------------------

    @staticmethod
    def _apply_unitary(
        state: np.ndarray, n: int, qubits: list[int], U: np.ndarray
    ) -> np.ndarray:
        """Apply unitary U to specified qubits within statevector."""
        k = len(qubits)
        tensor = state.reshape([2] * n)
        axes = [int(n - 1 - q) for q in qubits]
        tensor = np.moveaxis(tensor, axes, list(range(k)))
        shape = tensor.shape
        flat = tensor.reshape(2**k, -1)
        flat = U @ flat
        tensor = flat.reshape(shape)
        tensor = np.moveaxis(tensor, list(range(k)), axes)
        return tensor.reshape(-1)

    @staticmethod
    def _apply_reset(state: np.ndarray, n: int, qubit: int) -> np.ndarray:
        """Reset a single qubit to |0>, collapsing and renormalising."""
        tensor = state.reshape([2] * n)
        axis = int(n - 1 - qubit)
        tensor = np.moveaxis(tensor, axis, 0)
        tensor[1, ...] = 0.0
        tensor = np.moveaxis(tensor, 0, axis)
        flat = tensor.reshape(-1)
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat = flat / norm
        return flat

    @staticmethod
    def _sample_counts(
        state: np.ndarray,
        n_qubits: int,
        n_cbits: int,
        measure_map: dict[int, int],
        shots: int,
        seed: int | None = None,
    ) -> dict[str, int]:
        """Sample measurement outcomes."""
        probs = np.abs(state) ** 2
        probs = probs / probs.sum()  # renormalise against fp drift
        rng = np.random.default_rng(seed)
        outcomes = rng.choice(len(probs), size=shots, p=probs)

        counts: dict[str, int] = {}
        for outcome in outcomes:
            cbits = ["0"] * n_cbits
            for qubit, cbit in measure_map.items():
                bit = (outcome >> qubit) & 1
                cbits[cbit] = str(bit)
            # Qiskit-style left-to-right ordering: highest cbit index first
            bitstring = "".join(reversed(cbits)) if n_cbits > 0 else ""
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts
