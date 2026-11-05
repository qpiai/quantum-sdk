"""Budget-compatible QAOA differentiation of scalar objectives.

Exact gate-wise parameter shift applies to expectation values, not nonlinear
distribution aggregations such as CVaR. Every probe uses the solver's normal
execution guard; no simulator-only state access or authentication bypass occurs.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from ...circuit import Circuit
from ...icr.circuitoperation import CircuitOperation, OperationType

Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray, Objective], np.ndarray]


@dataclass
class CircuitObjective:
    """Scalar objective with a separate budgeted gate-probe execution path."""

    evaluate: Callable[[np.ndarray, Circuit | None, bool], float]

    def __call__(self, parameters: np.ndarray) -> float:
        """Evaluate an ordinary candidate parameter vector."""
        return self.evaluate(parameters, None, False)

    def probe(self, parameters: np.ndarray, circuit: Circuit) -> float:
        """Evaluate a gate-shifted circuit that is not a candidate solution."""
        return self.evaluate(parameters, circuit, True)


def finite_difference(
    parameters: np.ndarray, objective: Objective, step: float
) -> np.ndarray:
    """Central differences, with two budgeted probes per logical parameter."""
    gradient = np.zeros_like(parameters, dtype=float)
    for index in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (objective(plus) - objective(minus)) / (2.0 * step)
    return gradient


def spsa_gradient(
    parameters: np.ndarray,
    objective: Objective,
    step: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Two-probe simultaneous perturbation with a caller-owned random stream."""
    direction = rng.choice([-1.0, 1.0], size=parameters.shape)
    difference = objective(parameters + step * direction) - objective(
        parameters - step * direction
    )
    return np.asarray(difference / (2.0 * step) * direction, dtype=float)


def parameter_shift(
    parameters: np.ndarray,
    circuit: Circuit,
    evaluate: Callable[[np.ndarray, Circuit], float],
) -> np.ndarray:
    """Shift each affine Pauli rotation and sum its chain-rule contribution.

    Gate angles carry ``parameter_index`` and ``parameter_scale``. Preparation
    rotations have no logical parameter index and are held fixed. Unsupported
    parameter-dependent gates are rejected before any probe is submitted.
    """
    rotations = {"RX", "RY", "RZ", "RXX", "RYY", "RZZ"}
    active = []
    for index, operation in enumerate(circuit.icr.evolve):
        parameter_index = operation.parameter_index
        if parameter_index is None:
            continue
        if (
            operation.operation_type != OperationType.N_QUBIT_PARAMETRIC
            or operation.gate_name.upper() not in rotations
            or not operation.params
            or len(operation.params) != 1
        ):
            raise ValueError(
                "parameter_shift requires supported single-angle Pauli rotations"
            )
        if not 0 <= parameter_index < len(parameters):
            raise ValueError("invalid parameter-shift gate annotation")
        if not np.isfinite(operation.parameter_scale):
            raise ValueError("parameter scales must be finite")
        active.append((index, parameter_index, operation.parameter_scale))
    gradient = np.zeros_like(parameters, dtype=float)
    for gate_index, parameter_index, scale in active:
        if scale == 0:
            continue
        plus, minus = Circuit(circuit.num_qubits), Circuit(circuit.num_qubits)
        for index, operation in enumerate(circuit.icr.evolve):
            positive, negative = deepcopy(operation), deepcopy(operation)
            if index == gate_index:
                # Rebuild the numeric rotation so cached decomposition bodies
                # (notably RZZ.order) cannot retain the original angle.
                positive = CircuitOperation(
                    operation.operation_type,
                    operation.gate_name,
                    operation.qubits,
                    [operation.params[0] + np.pi / 2],
                )
                negative = CircuitOperation(
                    operation.operation_type,
                    operation.gate_name,
                    operation.qubits,
                    [operation.params[0] - np.pi / 2],
                )
            plus.add_operation(positive)
            minus.add_operation(negative)
        gradient[parameter_index] += (
            scale * (evaluate(parameters, plus) - evaluate(parameters, minus)) / 2.0
        )
    return gradient
