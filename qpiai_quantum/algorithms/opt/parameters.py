"""Explicit variational angles shared by QAOA and VQE.

Numeric gate arguments are fixed. Mark only the arguments an optimizer should
change with :func:`parameterize`. Existing single-angle ``parameter_index`` and
``parameter_scale`` annotations express the same rule and remain supported.
"""

import copy
import math
from collections.abc import Callable, Iterable, Iterator
from typing import cast

import numpy as np

from ...circuit import Circuit
from ...icr.circuitoperation import CircuitOperation, OperationType


def _index(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def parameterize(
    circuit: Circuit,
    parameter_index: int,
    *,
    operation_index: int | None = None,
    argument_index: int = 0,
    scale: float = 1.0,
    offset: float = 0.0,
) -> None:
    """Mark one gate argument as ``scale * theta[parameter_index] + offset``.

    By default, mark the last operation added to ``circuit``. Otherwise use its
    zero-based ``operation_index``. ``argument_index`` selects an angle within a
    gate with several angles. Other arguments remain fixed. Reusing a parameter
    index shares that optimizer value across gates. Indices start at zero.

    Example::

        circuit.ry(0, 0.7)  # fixed preparation
        circuit.rz(0, 0.0)
        parameterize(circuit, 0)
        circuit.rx(0, 0.0)
        parameterize(circuit, 0, scale=2.0)

    The last two angles become theta[0] and 2 * theta[0]. Unlike the previous
    implicit solver convention, writing 0.0 alone does not make an angle
    adjustable. This helper changes the template in place, not its gate values.
    """
    parameter_index = _index(parameter_index, "parameter_index")
    argument_index = _index(argument_index, "argument_index")
    operations = list(circuit.icr.evolve)
    selected = (
        len(operations) - 1
        if operation_index is None
        else _index(operation_index, "operation_index")
    )
    if not 0 <= selected < len(operations):
        raise ValueError("operation_index must identify an existing gate")
    operation = operations[selected]
    while getattr(operation, "base_operation", None) is not None:
        operation = operation.base_operation
    if (
        operation.operation_type != OperationType.N_QUBIT_PARAMETRIC
        or not operation.params
    ):
        raise ValueError("Choose a gate with numeric angle arguments")
    if argument_index >= len(operation.params):
        raise ValueError("argument_index is outside the gate's angle arguments")
    if operation.parameter_index is not None and argument_index == 0:
        raise ValueError("This angle already has a parameter_index annotation")
    if argument_index in operation.parameter_bindings:
        raise ValueError("This angle already has a parameter binding")
    operation.parameter_bindings[argument_index] = (
        parameter_index,
        _finite(scale, "scale"),
        _finite(offset, "offset"),
    )


def _operations(operation: CircuitOperation) -> Iterator[CircuitOperation]:
    """Visit semantic gate bodies, excluding cached primitive decompositions."""
    base = getattr(operation, "base_operation", None)
    if base is not None:
        if operation.parameter_index is not None or operation.parameter_bindings:
            raise ValueError(
                "Annotate a controlled operation's base gate instead of its wrapper",
            )
        yield from _operations(base)
    elif operation.operation_type == OperationType.N_QUBIT_PARAMETRIC:
        if (
            type(operation) is CircuitOperation
            and getattr(operation, "order", None) is not None
        ):
            if operation.parameter_index is not None or operation.parameter_bindings:
                raise ValueError(
                    "Bind angles inside a custom composite's body, not its cached parent",
                )
            for child in operation.order or []:
                yield from _operations(child)
        else:
            yield operation
    elif getattr(operation, "order", None) is not None:
        if operation.parameter_index is not None or operation.parameter_bindings:
            raise ValueError("Bind angles inside the composite's body")
        for child in operation.order or []:
            yield from _operations(child)
    elif operation.parameter_index is not None or operation.parameter_bindings:
        raise ValueError(
            "Parameter bindings require a gate with numeric angle arguments"
        )


def _bindings(operation: CircuitOperation) -> dict[int, tuple[int, float, float]]:
    result = dict(operation.parameter_bindings)
    if operation.parameter_index is not None:
        if len(operation.params or []) != 1:
            raise ValueError(
                "Use parameterize(..., argument_index=...) for gates with several angles",
            )
        if 0 in result:
            raise ValueError("An angle cannot have two parameter bindings")
        result[0] = (operation.parameter_index, operation.parameter_scale, 0.0)
    validated: dict[int, tuple[int, float, float]] = {}
    for argument, binding in result.items():
        argument = _index(argument, "argument_index")
        if argument >= len(operation.params or []):
            raise ValueError("A binding selects a nonexistent gate angle")
        if not isinstance(binding, tuple | list) or len(binding) != 3:
            raise ValueError("Each angle binding needs an index, scale, and offset")
        index, scale, offset = binding
        validated[argument] = (
            _index(index, "parameter_index"),
            _finite(scale, "scale"),
            _finite(offset, "offset"),
        )
    return validated


def count_parameters(circuit: Circuit) -> int:
    """Count explicitly indexed optimizer values; unmarked angles are fixed.

    The vector spans zero through the largest declared index. Standard QAOA
    retains its declared 2p slots even when a constant cost uses no gamma gates.
    """
    count = 0
    for top in circuit.icr.evolve:
        for operation in _operations(top):
            for index, _, _ in _bindings(operation).values():
                count = max(count, index + 1)
    declared = (circuit.icr.metadata or {}).get("qaoa", {}).get("parameter_count")
    if declared is not None:
        declared = _index(declared, "parameter_count")
        if count > declared:
            raise ValueError("Gate bindings exceed the declared QAOA parameter count")
        count = declared
    return count


def bind_parameters(circuit: Circuit, parameters: Iterable[float] | None) -> Circuit:
    """Return an independent numeric circuit with all explicit bindings applied.

    ``None`` returns an independent template copy. A supplied vector must be
    finite, one-dimensional, and exactly match :func:`count_parameters`.
    Fixed angles, controls, and composite bodies are preserved. Bound gates
    lose their template annotations so binding cannot accidentally run twice.
    """
    count = count_parameters(circuit)
    values: np.ndarray | None = None
    if parameters is not None:
        try:
            values = np.asarray(list(parameters), dtype=float)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "Supply a one-dimensional vector of finite parameter values"
            )
        if values.ndim != 1 or len(values) != count or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Ansatz requires exactly {count} finite parameter values. Numeric angles are fixed; mark adjustable arguments with parameterize().",
            )

    def bind(operation: CircuitOperation) -> CircuitOperation:
        result = copy.deepcopy(operation)
        base = getattr(operation, "base_operation", None)
        if base is not None:
            result.base_operation = bind(base)
            result.params = copy.deepcopy(result.base_operation.params)
        elif operation.operation_type == OperationType.N_QUBIT_PARAMETRIC:
            if (
                type(operation) is CircuitOperation
                and getattr(operation, "order", None) is not None
            ):
                result.order = [bind(child) for child in operation.order or []]
                return result
            bindings = _bindings(operation)
            if bindings and values is not None:
                angles = list(operation.params or [])
                for argument, (index, scale, offset) in bindings.items():
                    angles[argument] = _finite(
                        scale * float(values[index]) + offset, "bound angle"
                    )
                # A cached decomposition may contain the original angles. Rebuild
                # known concrete gates through their constructor, never edit only
                # the parent's params and leave the executed body stale.
                if getattr(operation, "order", None) is not None:
                    if type(operation) is CircuitOperation:
                        raise ValueError(
                            "Bind angles inside a custom composite's body, not its cached parent",
                        )
                    try:
                        constructor = cast(
                            Callable[..., CircuitOperation], type(operation)
                        )
                        rebuilt = constructor(*operation.qubits, *angles)
                    except (TypeError, ValueError):
                        raise ValueError(
                            "Cannot rebuild this parameterized composite; bind its body gates instead",
                        )
                    result.order = rebuilt.order
                result.params = angles
                result.parameter_index = None
                result.parameter_scale = 1.0
                result.parameter_bindings = {}
        elif getattr(operation, "order", None) is not None:
            result.order = [bind(child) for child in operation.order or []]
        return result

    metadata = copy.deepcopy(circuit.icr.metadata or {})
    if values is not None:
        # QAOA's logical slot declaration describes a template, not a bound circuit.
        metadata.get("qaoa", {}).pop("parameter_count", None)
    result = Circuit(circuit.num_qubits, metadata=metadata)
    for operation in circuit.icr.evolve:
        result.add_operation(bind(operation))
    return result
