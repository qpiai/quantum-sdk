from enum import Enum
from typing import Dict, List, Any, Optional


class OperationType(Enum):
    N_QUBIT_NON_PARAMETRIC = "n_qubit_non_parametric"
    N_QUBIT_PARAMETRIC = "n_qubit_parametric"
    MEASURE = "measure"
    BARRIER = "barrier"
    SWAP = "swap"
    OPERATION = "operation"


class CircuitOperation:
    _registry = {}
    order: Optional[List["CircuitOperation"]]

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
        qubits: List[int],
        params: Optional[List[float]] = None,
        clbits: Optional[List[int]] = None,
    ):
        self.operation_type = operation_type
        self.gate_name = gate_name
        self.qubits = qubits
        self.params = params
        self.clbits = clbits

    def to_json(self) -> Dict[str, Any]:
        result = {
            "operation_type": self.operation_type.value,
            "gate_name": self.gate_name,
            "qubits": self.qubits,
        }
        if self.params is not None:
            result["params"] = self.params
        if self.clbits is not None:
            result["clbits"] = self.clbits
        order = getattr(self, "order", None)
        if order is not None:
            result["order"] = [op.to_json() for op in order]
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


class ECRGate(CircuitOperation):
    def __init__(self, qubit1: int, qubit2: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC, "ECR", [qubit1, qubit2]
        )


class SwapGate(CircuitOperation):
    def __init__(self, qubit1: int, qubit2: int):
        super().__init__(OperationType.SWAP, "SWAP", [qubit1, qubit2])


class ISwapGate(CircuitOperation):
    def __init__(self, qubit1: int, qubit2: int):
        super().__init__(
            OperationType.N_QUBIT_NON_PARAMETRIC, "iSWAP", [qubit1, qubit2]
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

    def to_json(self) -> Dict[str, Any]:
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


class CSwapGate(CircuitOperation):
    def __init__(self, control_qubit: int, target_qubit1: int, target_qubit2: int):
        super().__init__(
            OperationType.SWAP,
            "CSWAP",
            [control_qubit, target_qubit1, target_qubit2],
        )


class MCXGate(CircuitOperation):
    def __init__(self, control_qubits: List[int], target_qubit: int):
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
    def __init__(self, *qubits: List[int]):
        super().__init__(OperationType.BARRIER, "Barrier", qubits)


# Generic operation container
class Operation(CircuitOperation):
    def __init__(
        self,
        name: str,
        qubits: List[int],
        params: Optional[List[float]] = None,
        clbits: Optional[List[int]] = None,
    ):
        super().__init__(OperationType.OPERATION, name, qubits, params, clbits)
