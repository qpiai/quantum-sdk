import math
from collections.abc import Iterable
from typing import Any
import numpy as np
from ....circuit import Circuit
from ....circuit.quantumregister import QuantumRegister
from ..qaoa_components import InitialState, Mixer, XMixer, XYMixer, append_component
from ..parameters import count_parameters as count_parameters, parameterize


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
    param_count = 0

    # Initial layer
    for i in range(n_qubits):
        circuit.ry(i, 0.0)  # Parameterized by optimizer
        parameterize(circuit, param_count)
        param_count += 1
        circuit.rz(i, 0.0)
        parameterize(circuit, param_count)
        param_count += 1

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
                parameterize(circuit, param_count)
                param_count += 1
                circuit.rz(i, 0.0)
                parameterize(circuit, param_count)
                param_count += 1

    return circuit


def standard_qaoa_ansatz(
    n_qubits: int,
    hamiltonian: Any,
    layers: int = 1,
    mixer: str | Mixer = "x",
    parameters: Iterable[float] | None = None,
    initial_state: InitialState | None = None,
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
        mixer: ``x``, ``xy``, or a numeric mixer factory. XY uses an ordered
            chain edge sweep; supply XYMixer(edges=...) to choose a topology.
        parameters: Logical QAOA parameters in interleaved order
            ``[gamma_0, beta_0, ..., gamma_(p-1), beta_(p-1)]``. When omitted,
            zero-valued angles are used to create a circuit template.
        initial_state: Unmeasured circuit or width-to-circuit factory.
            Defaults to a uniform superposition.

    Returns:
        Parameterized QAOA circuit compatible with SDK

    Note:
        The logical parameter count is exactly ``2 * layers``. A logical gamma
        or beta is shared by every corresponding gate in that layer.
    """
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer")
    if not isinstance(layers, int) or isinstance(layers, bool) or layers < 1:
        raise ValueError("layers must be a positive integer")
    if isinstance(mixer, str):
        if mixer not in {"x", "xy"}:
            raise ValueError("mixer must be x, xy, or a mixer factory")
        mixer_factory: Mixer
        if mixer == "x":
            mixer_factory = XMixer()
        else:
            mixer_factory = XYMixer()
    elif callable(mixer):
        mixer_factory = mixer
    else:
        raise TypeError("mixer must be x, xy, or a mixer factory")

    expected_parameter_count = 2 * layers
    if parameters is None:
        resolved_parameters = [0.0] * expected_parameter_count
    else:
        resolved_parameters = [float(value) for value in parameters]
        if len(resolved_parameters) != expected_parameter_count:
            raise ValueError(
                f"Standard QAOA with {layers} layer(s) requires "
                f"{expected_parameter_count} parameters, got "
                f"{len(resolved_parameters)}"
            )
        if not all(math.isfinite(value) for value in resolved_parameters):
            raise ValueError("QAOA parameters must be finite")

    circuit = Circuit(
        n_qubits,
        metadata={
            "qaoa": {
                "parameter_count": expected_parameter_count,
                "parameter_order": [
                    name
                    for layer in range(layers)
                    for name in (f"gamma_{layer}", f"beta_{layer}")
                ],
            }
        },
    )

    if initial_state is None:
        for i in range(n_qubits):
            circuit.h(i)
    else:
        preparation = (
            initial_state(n_qubits) if callable(initial_state) else initial_state
        )
        if isinstance(preparation, Circuit) and count_parameters(preparation):
            raise ValueError(
                "initial_state must be fixed, without logical parameter annotations"
            )
        append_component(circuit, preparation)
        for operation in circuit.icr.evolve:
            if operation.parameter_index is not None:
                raise ValueError(
                    "initial_state must be fixed, without logical parameter annotations"
                )

    # Get Hamiltonian terms
    if not hasattr(hamiltonian, "get_hamiltonian_terms"):
        raise ValueError("Hamiltonian must have get_hamiltonian_terms() method")

    terms = list(hamiltonian.get_hamiltonian_terms())
    _validate_qaoa_terms(terms, n_qubits=n_qubits)

    # QAOA layers
    for layer in range(layers):
        gamma = resolved_parameters[2 * layer]
        beta = resolved_parameters[2 * layer + 1]

        # Cost Hamiltonian layer (problem-specific)
        # Each term in the Hamiltonian gets exp(-i * gamma * term)
        for ops, coeff in terms:
            if not ops:  # Constant term, skip
                continue

            # Apply cost layer evolution
            first = len(circuit.icr.evolve)
            _apply_cost_evolution(circuit, ops, coeff, gamma=gamma)
            for operation in list(circuit.icr.evolve)[first:]:
                if operation.params:
                    operation.parameter_index = 2 * layer
                    operation.parameter_scale = 2.0 * float(coeff)

        # Mixer Hamiltonian layer
        # RX(theta) = exp(-i * theta * X / 2), hence theta = 2 * beta.
        first = len(circuit.icr.evolve)
        append_component(circuit, mixer_factory(n_qubits, beta))
        for operation in list(circuit.icr.evolve)[first:]:
            if operation.parameter_index is not None:
                if operation.parameter_index != 0:
                    raise ValueError(
                        "mixer annotations must use local beta parameter_index=0"
                    )
                operation.parameter_index = 2 * layer + 1
            for argument, (
                index,
                scale,
                offset,
            ) in operation.parameter_bindings.items():
                if index != 0:
                    raise ValueError(
                        "mixer annotations must use local beta parameter_index=0"
                    )
                operation.parameter_bindings[argument] = (2 * layer + 1, scale, offset)

    return circuit


def _validate_qaoa_terms(
    terms: list[tuple[list[tuple[int, str]], float]], *, n_qubits: int
) -> None:
    """Validate the diagonal Ising Hamiltonian supported by standard QAOA."""
    for ops, coeff in terms:
        coefficient = float(coeff)
        if not math.isfinite(coefficient):
            raise ValueError("Hamiltonian coefficients must be finite")

        seen_qubits: set[int] = set()
        for qubit, operator in ops:
            if not isinstance(qubit, int) or isinstance(qubit, bool):
                raise ValueError("Hamiltonian qubit indices must be integers")
            if qubit < 0 or qubit >= n_qubits:
                raise ValueError(
                    f"Hamiltonian qubit index {qubit} is outside [0, {n_qubits})"
                )
            if qubit in seen_qubits:
                raise ValueError(
                    f"Hamiltonian term contains duplicate qubit index {qubit}"
                )
            seen_qubits.add(qubit)
            if operator not in {"I", "Z"}:
                raise ValueError(
                    "Standard QAOA requires a diagonal Ising Hamiltonian "
                    f"containing only I and Z operators, got {operator!r}"
                )


def _apply_cost_evolution(
    circuit: Circuit,
    ops: list[tuple[int, str]],
    coeff: float,
    gamma: float,
) -> None:
    """
    Apply cost evolution for a single Hamiltonian term.

    For a term like Z_i Z_j, this applies exp(-i * gamma * coeff * Z_i Z_j).

    Args:
        circuit: Circuit to modify
        ops: list of (qubit_index, pauli_op) tuples
        coeff: Coefficient of this term
        gamma: Shared cost angle for the current QAOA layer.
    """
    active_ops = [(qubit, operator) for qubit, operator in ops if operator != "I"]
    if not active_ops:
        return

    # Extract qubits and operators
    qubits = [qi for qi, _ in active_ops]
    operators = [op for _, op in active_ops]
    # RZ/RZZ(theta) implement exp(-i * theta * P / 2), while a QAOA
    # cost term is exp(-i * gamma * coeff * P).
    rotation_angle = 2.0 * float(gamma) * float(coeff)

    # Handle different Pauli string types
    if all(op == "Z" for op in operators):
        # Pure Z terms: use RZZ or RZ gates
        if len(qubits) == 1:
            # Single Z term: RZ gate
            circuit.rz(qubits[0], rotation_angle)
        elif len(qubits) == 2:
            # ZZ term: RZZ gate
            circuit.rzz(qubits[0], qubits[1], rotation_angle)
        else:
            # Multi-qubit Z term: decompose into CNOTs and RZ
            # Z_i Z_j Z_k = CNOT(j,i) CNOT(k,i) RZ(i) CNOT(k,i) CNOT(j,i)
            for i in range(1, len(qubits)):
                circuit.cx(qubits[i], qubits[0])
            circuit.rz(qubits[0], rotation_angle)
            for i in range(len(qubits) - 1, 0, -1):
                circuit.cx(qubits[i], qubits[0])

    else:  # Defensive guard; _validate_qaoa_terms rejects this earlier.
        raise ValueError(f"Unsupported QAOA cost operators: {operators}")


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
