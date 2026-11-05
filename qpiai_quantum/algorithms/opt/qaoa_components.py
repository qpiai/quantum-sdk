"""Composable QAOA states, mixers, and sampled energy objectives.

Mixer factories receive ``(n_qubits, beta)`` and return a numeric circuit.
Built-in mixers carry affine gate annotations for gate-wise differentiation.
User-defined mixers support finite differences, SPSA, and gradient callbacks.
"""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ...circuit import Circuit
from ...icr.circuitoperation import CircuitOperation, OperationType


class Mixer(Protocol):
    """Build one mixer layer at a numeric shared beta angle."""

    def __call__(self, n_qubits: int, beta: float, /) -> Circuit:
        """Return a unitary circuit on exactly ``n_qubits`` wires."""
        ...


InitialState = Circuit | Callable[[int], Circuit]
Aggregation = Callable[[Mapping[str, float], Mapping[str, float]], float]


def append_component(target: Circuit, component: Circuit) -> None:
    """Append independent operations after validating a unitary component."""
    if not isinstance(component, Circuit) or component.num_qubits != target.num_qubits:
        raise ValueError("QAOA component must be a Circuit of the problem width")

    def validate(operation: CircuitOperation) -> None:
        if operation.operation_type in {OperationType.MEASURE, OperationType.RESET}:
            raise ValueError("QAOA components must not contain measurement or reset")
        base = getattr(operation, "base_operation", None)
        if base is not None:
            validate(base)
        for child in getattr(operation, "order", None) or []:
            validate(child)

    for operation in component.icr.evolve:
        validate(operation)
    for operation in component.icr.evolve:
        target.add_operation(deepcopy(operation))


def _mark_last(circuit: Circuit, scale: float) -> None:
    operation = list(circuit.icr.evolve)[-1]
    operation.parameter_index = 0
    operation.parameter_scale = scale


@dataclass(frozen=True)
class BasisState:
    """Prepare a basis bitstring, with qubit zero at its rightmost position."""

    bitstring: str

    def __post_init__(self) -> None:
        if not self.bitstring or set(self.bitstring) - {"0", "1"}:
            raise ValueError("bitstring must be a nonempty binary string")

    def __call__(self, n_qubits: int) -> Circuit:
        """Prepare the requested bitstring without measurement."""
        if len(self.bitstring) != n_qubits:
            raise ValueError("basis bitstring must match the problem width")
        circuit = Circuit(n_qubits)
        for index, bit in enumerate(reversed(self.bitstring)):
            if bit == "1":
                circuit.x(index)
        return circuit


@dataclass(frozen=True)
class XMixer:
    """Transverse-field evolution exp(-i beta sum(X))."""

    def __call__(self, n_qubits: int, beta: float) -> Circuit:
        """Build the commuting single-qubit X rotations."""
        circuit = Circuit(n_qubits)
        for index in range(n_qubits):
            circuit.rx(index, 2.0 * beta)
            _mark_last(circuit, 2.0)
        return circuit


@dataclass(frozen=True)
class XYMixer:
    """Ordered edge sweep of exp(-i beta (XX+YY)/2), preserving Hamming weight.

    Adjacent edges need not commute. This is one ordered product of edge
    evolutions per QAOA layer, not an exact exponential of their sum.
    Without explicit edges a linear chain is used.
    """

    edges: tuple[tuple[int, int], ...] | None = None

    def __call__(self, n_qubits: int, beta: float) -> Circuit:
        """Build excitation-preserving exchanges in the supplied edge order."""
        edges = (
            self.edges
            if self.edges is not None
            else tuple((index, index + 1) for index in range(n_qubits - 1))
        )
        circuit = Circuit(n_qubits)
        seen: set[tuple[int, int]] = set()
        for left, right in edges:
            if (
                any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < n_qubits
                    for index in (left, right)
                )
                or left == right
            ):
                raise ValueError("XY edges must connect two distinct in-range qubits")
            edge = (min(left, right), max(left, right))
            if edge in seen:
                raise ValueError("XY edges must be unique")
            seen.add(edge)
            circuit.rxx(left, right, beta)
            _mark_last(circuit, 1.0)
            circuit.ryy(left, right, beta)
            _mark_last(circuit, 1.0)
        return circuit


@dataclass(frozen=True)
class WarmStart:
    """Product-state warm start and matching mixer from relaxed probabilities.

    Input index i is P(x_i=1). Values are clipped into [epsilon, 1-epsilon]
    to maintain exploration. This component does not solve a relaxation.
    epsilon=0.5 recovers the uniform initial state and standard X mixer.
    The prepared state is a +1 eigenstate of each rotated-Z mixer term;
    this sign convention agrees with standard exp(-i beta X).
    """

    probabilities: Sequence[float]
    epsilon: float = 0.01

    def __post_init__(self) -> None:
        values = np.asarray(self.probabilities, dtype=float)
        if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
            raise ValueError(
                "warm-start probabilities must be a finite nonempty vector"
            )
        if np.any((values < 0) | (values > 1)):
            raise ValueError("warm-start probabilities must lie in [0, 1]")
        if not np.isfinite(self.epsilon) or not 0 < self.epsilon <= 0.5:
            raise ValueError("epsilon must lie in (0, 0.5]")
        object.__setattr__(
            self, "probabilities", tuple(float(value) for value in values)
        )

    def _angles(self, n_qubits: int) -> np.ndarray:
        if len(self.probabilities) != n_qubits:
            raise ValueError("warm-start probabilities must match the problem width")
        return 2.0 * np.arcsin(
            np.sqrt(np.clip(self.probabilities, self.epsilon, 1.0 - self.epsilon))
        )

    def initial_state(self, n_qubits: int) -> Circuit:
        """Prepare the clipped independent Bernoulli distribution."""
        circuit = Circuit(n_qubits)
        for index, angle in enumerate(self._angles(n_qubits)):
            circuit.ry(index, float(angle))
        return circuit

    def mixer(self, n_qubits: int, beta: float) -> Circuit:
        """Evolve under sum(RY(theta) Z RY(-theta)), matching the X convention."""
        circuit = Circuit(n_qubits)
        for index, angle in enumerate(self._angles(n_qubits)):
            circuit.ry(index, -float(angle))
            circuit.rz(index, 2.0 * beta)
            _mark_last(circuit, 2.0)
            circuit.ry(index, float(angle))
        return circuit


@dataclass(frozen=True)
class CVaR:
    """Mean energy in the lowest-energy alpha probability mass.

    The boundary outcome contributes only the fraction needed to reach alpha.
    alpha=1 is the ordinary expectation. Lower alpha can discard most samples.
    """

    alpha: float = 0.25

    def __post_init__(self) -> None:
        if (
            isinstance(self.alpha, bool)
            or not np.isfinite(self.alpha)
            or not 0 < self.alpha <= 1
        ):
            raise ValueError("CVaR alpha must lie in (0, 1]")

    def __call__(
        self, probabilities: Mapping[str, float], energies: Mapping[str, float]
    ) -> float:
        """Aggregate a normalized energy distribution into a lower-tail mean."""
        if not probabilities or probabilities.keys() != energies.keys():
            raise ValueError(
                "probabilities and energies must have the same nonempty support"
            )
        if any(not np.isfinite(value) or value < 0 for value in probabilities.values()):
            raise ValueError("probabilities must be finite and nonnegative")
        if not np.isclose(sum(probabilities.values()), 1.0, rtol=0, atol=1e-10):
            raise ValueError("probabilities must sum to one")
        if any(not np.isfinite(value) for value in energies.values()):
            raise ValueError("energies must be finite")
        remaining = self.alpha
        value = 0.0
        for bitstring in sorted(probabilities, key=lambda key: (energies[key], key)):
            mass = min(remaining, probabilities[bitstring])
            value += mass * energies[bitstring]
            remaining -= mass
            if remaining <= 0.0:
                break
        return float(value / self.alpha)
