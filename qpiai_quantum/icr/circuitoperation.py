from enum import Enum
from typing import Any, Optional
from .validation import validate_index, validate_parameter


class OperationType(Enum):
    N_QUBIT_NON_PARAMETRIC = "n_qubit_non_parametric"
    N_QUBIT_PARAMETRIC = "n_qubit_parametric"
    MEASURE = "measure"
    BARRIER = "barrier"
    RESET = "reset"
    SWAP = "swap"
    OPERATION = "operation"


class CircuitOperation:
    _registry: dict[str, type["CircuitOperation"]] = {}
    order: list["CircuitOperation"] | None
    base_operation: "CircuitOperation"
    control_qubits: list[int]

    def __init_subclass__(cls):
        """
        Register the subclass into the registry.
        """

        if cls.__name__ in CircuitOperation._registry:
            raise ValueError(f"Duplicate subclass name: {cls.__name__}")

        CircuitOperation._registry[cls.__name__] = cls

    @classmethod
    def get_gate(cls, name: str):
        """
        Get the subclass from the registry by name.
        """
        return CircuitOperation._registry.get(name, None)

    @classmethod
    def list_gates(cls):
        """
        List all the registered subclasses.
        """
        return list(CircuitOperation._registry.keys())

    def __init__(
        self,
        operation_type: OperationType,
        gate_name: str,
        qubits: list[int],
        params: list[float] | None = None,
        clbits: list[int] | None = None,
        parameter_index: int | None = None,
        parameter_scale: float = 1.0,
    ):
        self.operation_type = operation_type
        self.gate_name = gate_name
        self.qubits = [validate_index(q) for q in qubits]
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError("Duplicate qubits are not allowed")
        self.params = (
            None if params is None else [validate_parameter(p) for p in params]
        )
        self.clbits = None if clbits is None else [validate_index(c) for c in clbits]
        self.parameter_index = parameter_index
        self.parameter_scale = parameter_scale
        # Per-angle explicit variational bindings: argument -> (index, scale, offset).
        # Numeric arguments without annotations remain fixed in QAOA and VQE.
        self.parameter_bindings: dict[int, tuple[int, float, float]] = {}

    def decompose(self) -> list["CircuitOperation"] | None:
        """Return an independent gate body, or None for an opaque operation.

        Composite bodies and shared named definitions use the same interface.
        Controlled wrappers remain structural: consumers must preserve their
        controls while visiting the base operation. Measurement/reset have no
        unitary decomposition.
        """
        import copy
        from .decomposition import gate_decomposition

        if self.operation_type in (OperationType.MEASURE, OperationType.RESET):
            return None
        if getattr(self, "base_operation", None) is not None:
            return None
        body = getattr(self, "order", None)
        if body is not None:
            return copy.deepcopy(body)
        return gate_decomposition(self.gate_name, self.qubits, self.params)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "operation_type": self.operation_type.value,
            "gate_name": self.gate_name,
            "qubits": self.qubits,
        }
        if self.params is not None:
            result["params"] = self.params
        if self.clbits is not None:
            result["clbits"] = self.clbits
        if self.parameter_index is not None:
            result["parameter_index"] = self.parameter_index
            result["parameter_scale"] = self.parameter_scale
        if self.parameter_bindings:
            result["parameter_bindings"] = {
                str(argument): list(binding)
                for argument, binding in self.parameter_bindings.items()
            }
        order = getattr(self, "order", None)
        if order is not None:
            result["order"] = [op.to_json() for op in order]
        base = getattr(self, "base_operation", None)
        if base is not None:
            result["base_operation"] = base.to_json()
            result["control_qubits"] = list(self.control_qubits)
        return result


# Single-qubit non-parametric gates
class HGate(CircuitOperation):
    def __init__(self, qubit: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "H", [qubit])


class XGate(CircuitOperation):
    def __init__(self, qubit: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "X", [qubit])


class IDGate(CircuitOperation):
    def __init__(self, qubit: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "ID", [qubit])


class YGate(CircuitOperation):
    def __init__(self, qubit: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "Y", [qubit])


class ZGate(CircuitOperation):
    def __init__(self, qubit: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "Z", [qubit])


class SGate(CircuitOperation):
    def __init__(self, qubit: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "S", [qubit])


class SDGGate(CircuitOperation):
    def __init__(self, qubit: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "Sdg", [qubit])


class TGate(CircuitOperation):
    def __init__(self, qubit: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "T", [qubit])


class TDGGate(CircuitOperation):
    def __init__(self, qubit: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "Tdg", [qubit])


class SXGate(CircuitOperation):
    def __init__(self, qubit: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "SX", [qubit])


class SXDGGate(CircuitOperation):
    def __init__(self, qubit: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "SXdg", [qubit])


# Single-qubit parametric gates


class RXGate(CircuitOperation):
    def __init__(self, qubit: int, theta: float):
        super().__init__(OperationType.N_QUBIT_PARAMETRIC, "RX", [qubit], [theta])


class RYGate(CircuitOperation):
    def __init__(self, qubit: int, theta: float):
        super().__init__(OperationType.N_QUBIT_PARAMETRIC, "RY", [qubit], [theta])


class RZGate(CircuitOperation):
    def __init__(self, qubit: int, theta: float):
        super().__init__(OperationType.N_QUBIT_PARAMETRIC, "RZ", [qubit], [theta])


class PGate(CircuitOperation):
    def __init__(self, qubit: int, theta: float):
        super().__init__(OperationType.N_QUBIT_PARAMETRIC, "P", [qubit], [theta])


class UGate(CircuitOperation):
    def __init__(self, qubit: int, theta: float, phi: float, lam: float):
        super().__init__(
            OperationType.N_QUBIT_PARAMETRIC, "U", [qubit], [theta, phi, lam]
        )


class U2Gate(CircuitOperation):
    def __init__(self, qubit: int, phi: float, lam: float):
        super().__init__(OperationType.N_QUBIT_PARAMETRIC, "U2", [qubit], [phi, lam])


# Two-qubit non-parametric gates
class CXGate(CircuitOperation):
    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC, "CX", [control_qubit, target_qubit]
        )


class CYGate(CircuitOperation):
    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC, "CY", [control_qubit, target_qubit]
        )


class CZGate(CircuitOperation):
    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC, "CZ", [control_qubit, target_qubit]
        )


class CHGate(CircuitOperation):
    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC, "CH", [control_qubit, target_qubit]
        )


class CSGate(CircuitOperation):
    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC, "CS", [control_qubit, target_qubit]
        )


class CSDGGate(CircuitOperation):
    def __init__(self, control_qubit: int, target_qubit: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC,
            "CSdg",
            [control_qubit, target_qubit],
        )


class DCXGate(CircuitOperation):
    def __init__(self, qubit1: int, qubit2: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "DCX", [qubit1, qubit2])


class ECRGate(CircuitOperation):
    def __init__(self, qubit1: int, qubit2: int):
        super().__init__(OperationType.N_QUBIT_NON_PARAMETRIC, "ECR", [qubit1, qubit2])


class SwapGate(CircuitOperation):
    def __init__(self, qubit1: int, qubit2: int):
        super().__init__(OperationType.SWAP, "SWAP", [qubit1, qubit2])


class ISwapGate(CircuitOperation):
    def __init__(self, qubit1: int, qubit2: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC, "iSWAP", [qubit1, qubit2]
        )


class ISwapDGGate(CircuitOperation):
    def __init__(self, qubit1: int, qubit2: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC, "iSWAPdg", [qubit1, qubit2]
        )


# Two-qubit parametric gates
class CPGate(CircuitOperation):
    def __init__(self, control_qubit: int, target_qubit: int, theta: float):
        super().__init__(
            OperationType.N_QUBIT_PARAMETRIC,
            "CP",
            [control_qubit, target_qubit],
            [theta],
        )


class RXXGate(CircuitOperation):
    def __init__(self, qubit1: int, qubit2: int, theta: float):
        super().__init__(
            OperationType.N_QUBIT_PARAMETRIC,
            "RXX",
            [qubit1, qubit2],
            [theta],
        )


class RYYGate(CircuitOperation):
    def __init__(self, qubit1: int, qubit2: int, theta: float):
        super().__init__(
            OperationType.N_QUBIT_PARAMETRIC,
            "RYY",
            [qubit1, qubit2],
            [theta],
        )


class CRXGate(CircuitOperation):
    def __init__(self, control_qubit: int, target_qubit: int, theta: float):
        super().__init__(
            OperationType.N_QUBIT_PARAMETRIC,
            "CRX",
            [control_qubit, target_qubit],
            [theta],
        )


class CRYGate(CircuitOperation):
    def __init__(self, control_qubit: int, target_qubit: int, theta: float):
        super().__init__(
            OperationType.N_QUBIT_PARAMETRIC,
            "CRY",
            [control_qubit, target_qubit],
            [theta],
        )


class CRZGate(CircuitOperation):
    def __init__(self, control_qubit: int, target_qubit: int, theta: float):
        super().__init__(
            OperationType.N_QUBIT_PARAMETRIC,
            "CRZ",
            [control_qubit, target_qubit],
            [theta],
        )


class CUGate(CircuitOperation):
    def __init__(
        self,
        control_qubit: int,
        target_qubit: int,
        theta: float,
        phi: float,
        lam: float,
        gamma: float,
    ):
        super().__init__(
            OperationType.N_QUBIT_PARAMETRIC,
            "CU",
            [control_qubit, target_qubit],
            [theta, phi, lam, gamma],
        )


class RZZGate(CircuitOperation):
    def __init__(self, qubit1: int, qubit2: int, theta: float):
        super().__init__(
            OperationType.N_QUBIT_PARAMETRIC, "RZZ", [qubit1, qubit2], [theta]
        )
        self.order = [
            CXGate(qubit1, qubit2),
            RZGate(qubit2, theta),
            CXGate(qubit1, qubit2),
        ]

    def to_json(self) -> dict[str, Any]:
        result = super().to_json()
        order = self.order
        if order is not None:
            result["order"] = [op.to_json() for op in order]
        return result


# Three-qubit gates
class CCXGate(CircuitOperation):
    def __init__(self, control_qubit1: int, control_qubit2: int, target_qubit: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC,
            "CCX",
            [control_qubit1, control_qubit2, target_qubit],
        )


class RCCXGate(CircuitOperation):
    def __init__(self, control_qubit1: int, control_qubit2: int, target_qubit: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC,
            "RCCX",
            [control_qubit1, control_qubit2, target_qubit],
        )


class CSwapGate(CircuitOperation):
    def __init__(self, control_qubit: int, target_qubit1: int, target_qubit2: int):
        super().__init__(
            OperationType.SWAP,
            "CSWAP",
            [control_qubit, target_qubit1, target_qubit2],
        )


class MCXGate(CircuitOperation):
    def __init__(self, control_qubits: list[int], target_qubit: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC,
            "MCX",
            control_qubits + [target_qubit],
        )


# Measurement and control operations
class MeasureOperation(CircuitOperation):
    def __init__(self, qubit: int, clbit: int):
        super().__init__(OperationType.MEASURE, "Measure", [qubit], None, [clbit])


class BarrierOperation(CircuitOperation):
    def __init__(self, *qubits: int | list[int]):
        qubit_list: list[int] = (
            qubits[0]
            if len(qubits) == 1 and isinstance(qubits[0], list)
            else [q for q in qubits if isinstance(q, int)]
        )
        super().__init__(OperationType.BARRIER, "Barrier", qubit_list)


# Generic operation container
class ResetOperation(CircuitOperation):
    def __init__(self, qubit: int):
        super().__init__(OperationType.RESET, "reset", [qubit])


class Operation(CircuitOperation):
    def __init__(
        self,
        name: str,
        qubits: list[int],
        params: list[float] | None = None,
        clbits: list[int] | None = None,
    ):
        super().__init__(OperationType.OPERATION, name, qubits, params, clbits)
