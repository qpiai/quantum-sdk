import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..icr.circuitoperation import CircuitOperation, OperationType
from ..icr.icr import IntermediateCircuitRepresentation
from .classicalregister import ClassicalRegister
from .exceptions import CircuitError
from .quantumregister import QuantumRegister
from .register import Register
from ..simulator import StatevectorSimulator

if TYPE_CHECKING:
    from ..jobmanager.job_result import JobResult
    from ..results.base_result import BaseQuantumResult


def _map_device_name_to_method(device_name: str):
    """
    Internal helper to map high-level device_name to (device, method).

    Returns:
        Tuple of (device, method) where:
          - device name is "QpiAI-QSV-Simulator", "QpiAI-QDM-Simulator", "QpiAI-QTN-Simulator", "QpiAI-Indus-1", "QpiAI-QSV-Lite", or "QpiAI-QDM-Lite"
          - method is one of "statevector", "density_matrix", "tensor_network", or None (for qpu)
    """
    if device_name == "QpiAI-QSV-Simulator":
        return "statevector"
    elif device_name == "QpiAI-QDM-Simulator":
        return "density_matrix"
    elif device_name == "QpiAI-QTN-Simulator":
        return "tensor_network"
    elif device_name == "QpiAI-Indus-1":
        return "statevector"
    elif device_name == "QpiAI-QSV-Lite":
        return "statevector_sampling"
    elif device_name == "QpiAI-QDM-Lite":
        return "density_matrix"
    elif device_name == "QpiAI-QSV-Local":
        return None
    else:
        return "statevector"


class Circuit:
    def __init__(self, *regs: int | Register, name=None, metadata=None):
        """
        A controller for the Intermediate Circuit Representation (ICR). This class is used to interact with the ICR at a high level and create quantum circuits. It is not recommended to directly interact with the ICR, but rather use this class.

        Args:
            *regs (int | Register): The number of qubits or the registers to be added to the circuit. If the argument
                              is any integer, there should either only be one or only two, or a sequence of
                              registers can be passed as arguments.

            name (str): The name of the circuit.

            metadata (dict): The metadata of the circuit.
        """

        if name is None:
            import uuid

            name = f"Circuit_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        self.name = name
        self.icr: IntermediateCircuitRepresentation

        if len(regs) == 1 and isinstance(regs[0], int):
            num_qubits = regs[0]
            qregs = QuantumRegister(num_qubits, name="q")
            cregs = ClassicalRegister(num_qubits, name="c")
            self.icr = IntermediateCircuitRepresentation(
                qregs, cregs, name=name, metadata=metadata
            )

        elif len(regs) == 2 and all(isinstance(reg, int) for reg in regs):
            num_qubits: int = regs[0]  # type: ignore
            num_clbits: int = regs[1]  # type: ignore
            qregs = QuantumRegister(num_qubits, name="q")
            cregs = ClassicalRegister(num_clbits, name="c")
            self.icr = IntermediateCircuitRepresentation(
                qregs, cregs, name=name, metadata=metadata
            )

        elif all(isinstance(reg, Register) for reg in regs):
            self.icr = IntermediateCircuitRepresentation(
                *regs, name=name, metadata=metadata
            )

        else:
            raise CircuitError("Invalid arguments for Circuit initialization.")

        if self.icr is None:
            raise CircuitError(
                "Failed to initialize the Intermediate Circuit Representation, cannot create a circuit."
            )

        self.standard_gate_set = CircuitOperation.list_gates()

        self._create_gate_methods()

    def _create_gate_methods(self):
        """
        Dynamically create methods for each gate label at the time of circuit initialization. This is done to maintain a clean code base. It is not recommended to use these methods directly.
        """

        gate_classes = self.standard_gate_set

        for gate_label in gate_classes:
            gate_class = CircuitOperation.get_gate(gate_label)
            if gate_class is None:
                continue
            label = gate_class.__name__
            if "Gate" in gate_class.__name__:
                label = gate_class.__name__.replace("Gate", "")
            elif "Operation" in gate_class.__name__:
                label = gate_class.__name__.replace("Operation", "")
            setattr(self, label.upper(), self._create_gate_method(gate_class))
            # setattr(self, label.lower(), self._create_gate_method(gate_class))

    def _create_gate_method(self, gate_class):
        """
        Given a gate class which is a CircuitOperation, this function creates a method on the circuit object which allows the user to create and add frequently used custom gates to the circuit easily.

        Args:
            gate_class (CircuitOperation): The class of the gate to be added to the circuit.
        """
        if gate_class.__name__ == "BarrierOperation":

            def barrier_method(*qubits):
                self.add_operation(gate_class(*qubits))

            return barrier_method

        def gate_method(*args, **kwargs):
            gate = gate_class(*args, **kwargs)
            self.add_operation(gate)

        return gate_method

    def add_operation(self, operation: CircuitOperation):
        """
        Adds a generic operation to the circuit.

        Args:
            operation (CircuitOperation): The operation to be added to the circuit.
        """

        if not isinstance(operation, CircuitOperation):
            raise CircuitError(
                "Invalid operation type, expected a CircuitOperation object."
            )

        self.icr._add_operation(operation)

    def compose(self, other: "Circuit", qubits: Optional[List[int]] = None):
        """
        Appends the operations of another circuit onto this circuit.

        Args:
            other (Circuit): The circuit whose operations should be appended.
            qubits (List[int], optional): A mapping from the 'other' circuit's qubits to this circuit's qubits.
                                          Indices correspond to `other`'s qubits, values correspond to `self`'s qubits.
        """
        import copy

        if qubits is not None and len(qubits) != other.num_qubits:
            raise CircuitError(
                f"Length of qubits mapping ({len(qubits)}) must match the number of qubits "
                f"in the other circuit ({other.num_qubits})."
            )

        for op in other.icr.evolve:
            new_op = copy.deepcopy(op)
            if qubits is not None:
                if getattr(new_op, "qubits", None) is not None:
                    new_op.qubits = [qubits[q] for q in new_op.qubits]
                if getattr(new_op, "order", None):
                    for sub_op in new_op.order:
                        if getattr(sub_op, "qubits", None) is not None:
                            sub_op.qubits = [qubits[q] for q in sub_op.qubits]
            self.add_operation(new_op)

    def inverse(self) -> "Circuit":
        """
        Returns a new circuit with the operations inverted in reverse order.

        Returns:
            Circuit: The inverted circuit.
        """
        import copy
        from ..icr.circuitoperation import SDGGate, SGate, TDGGate, TGate, OperationType

        inv_circuit = Circuit(self.num_qubits, self.num_clbits)

        def invert_op(op: CircuitOperation) -> CircuitOperation:
            name = op.__class__.__name__

            # Self-inverse gates
            if name in [
                "HGate",
                "XGate",
                "YGate",
                "ZGate",
                "IDGate",
                "SXGate",
                "CXGate",
                "CYGate",
                "CZGate",
                "CHGate",
                "ECRGate",
                "SwapGate",
                "CCXGate",
                "CSwapGate",
                "ISwapGate",
            ]:
                return copy.deepcopy(op)

            # S / T / SX gates
            elif name == "SGate":
                return SDGGate(*op.qubits)
            elif name == "SDGGate":
                return SGate(*op.qubits)
            elif name == "TGate":
                return TDGGate(*op.qubits)
            elif name == "TDGGate":
                return TGate(*op.qubits)
            elif name == "SXDGGate":
                from ..icr.circuitoperation import SXGate as SXG

                return SXG(*op.qubits)

            # U gate inverse: U(θ, φ, λ)⁻¹ = U(-θ, -λ, -φ)
            elif name == "UGate":
                inv_op = copy.deepcopy(op)
                if inv_op.params is not None:
                    theta, phi, lam = inv_op.params
                    inv_op.params = [-theta, -lam, -phi]
                return inv_op

            # Parametric gates (negate theta)
            elif name in [
                "RXGate",
                "RYGate",
                "RZGate",
                "PGate",
                "CPGate",
                "RZZGate",
                "RXXGate",
                "RYYGate",
                "CRXGate",
                "CRYGate",
                "CRZGate",
            ]:
                inv_op = copy.deepcopy(op)
                if inv_op.params is not None:
                    inv_op.params = [-p for p in inv_op.params]
                order = getattr(inv_op, "order", None)
                if order is not None:
                    setattr(
                        inv_op,
                        "order",
                        [invert_op(sub_op) for sub_op in reversed(order)],
                    )
                return inv_op

            # Barriers and Measurements
            elif name in ["BarrierOperation", "MeasureOperation"]:
                return copy.deepcopy(op)

            # Multi-controlled X (self inverse)
            elif name == "MCXGate":
                return copy.deepcopy(op)

            else:
                raise CircuitError(f"Inverse of {name} is not currently supported.")

        for op in reversed(list(self.icr.evolve)):
            inv_circuit.add_operation(invert_op(op))

        return inv_circuit

    def to_json(self) -> Dict:
        """
        Returns the JSON representation of the circuit.

        Returns:
            dict: The JSON representation of the circuit.
        """
        return self.icr.to_json()

    def __str__(self):
        return str(self.icr)

    def _validate_qubit(self, qubit: int):
        if not (0 <= qubit < self.icr.num_qubits):
            raise CircuitError(
                f"Qubit index {qubit} out of range. Valid range: 0-{self.icr.num_qubits - 1}"
            )

    def _validate_clbit(self, clbit: int):
        if not (0 <= clbit < self.icr.num_clbits):
            raise CircuitError(
                f"Classical bit index {clbit} out of range. Valid range: 0-{self.icr.num_clbits - 1}"
            )

    def _validate_unique_qubits(self, *qubits: int):
        if len(qubits) != len(set(qubits)):
            raise CircuitError("Duplicate qubits are not allowed in the input.")

    def to_circuit_operation(self, name=None):
        """
        Convert the circuit to a Circuit Operation.
        """

        if name is None:
            name = self.name

        gates: List[CircuitOperation] = list(self.icr.evolve)

        # validate circuit can be converted to a single circuit operation

        for gate in gates:
            if gate.operation_type == OperationType.MEASURE:
                raise CircuitError("Circuit contains measurement operations.")

        qubits = []
        params = []

        for gate in gates:
            qubits.extend(gate.qubits)
            if gate.params:
                params.extend(gate.params)

        qubitMap = {}
        qubits = list(set(qubits))

        for qubit in qubits:
            qubitMap[f"qubit{qubit}"] = qubit

        paramMap = {}
        params = list(set(params))
        for index in range(len(params)):
            paramMap[f"param{index}"] = params[index]

        # circOperationType = OperationType.N_QUBIT_PARAMETRIC if isParam else OperationType.N_QUBIT_NON_PARAMETRIC
        circOperationType = OperationType.OPERATION
        circOperation = CircuitOperation(
            circOperationType, name, qubits=qubits, params=params
        )
        circOperation.order = []  # type: ignore

        for gate in gates:
            circOperation.order.append(gate)  # type: ignore

        return circOperation

    # Single-qubit non-parametric gates
    def h(self, qubit: int):
        """
        Apply a Hadamard (H) gate to the specified qubit.

        Args:
            qubit (int): Target qubit index
        """
        self._validate_qubit(qubit)
        self.H(qubit)  # type: ignore

    def x(self, qubit: int):
        """
        Apply a Pauli-X gate to the specified qubit.

        Args:
            qubit (int): Target qubit index
        """
        self._validate_qubit(qubit)
        self.X(qubit)  # type: ignore

    def id(self, qubit: int):
        """
        Apply an Identity gate to the specified qubit.

        Args:
            qubit (int): Target qubit index
        """
        self._validate_qubit(qubit)
        self.ID(qubit)  # type: ignore

    def y(self, qubit: int):
        """
        Apply a Pauli-Y gate to the specified qubit.

        Args:
            qubit (int): Target qubit index
        """
        self._validate_qubit(qubit)
        self.Y(qubit)  # type: ignore

    def z(self, qubit: int):
        """
        Apply a Pauli-Z gate to the specified qubit.

        Args:
            qubit (int): Target qubit index
        """
        self._validate_qubit(qubit)
        self.Z(qubit)  # type: ignore

    def s(self, qubit: int):
        """
        Apply an S gate to the specified qubit (√Z or phase(π/2)).

        Args:
            qubit (int): Target qubit index
        """
        self._validate_qubit(qubit)
        self.S(qubit)  # type: ignore

    def sdg(self, qubit: int):
        """
        Apply an S-dagger gate to the specified qubit (conjugate transpose of S).

        Args:
            qubit (int): Target qubit index
        """
        self._validate_qubit(qubit)
        self.SDG(qubit)  # type: ignore

    def t(self, qubit: int):
        """
        Apply a T gate to the specified qubit (√S or phase(π/4)).

        Args:
            qubit (int): Target qubit index
        """
        self._validate_qubit(qubit)
        self.T(qubit)  # type: ignore

    def tdg(self, qubit: int):
        """
        Apply a T-dagger gate to the specified qubit (conjugate transpose of T).

        Args:
            qubit (int): Target qubit index
        """
        self._validate_qubit(qubit)
        self.TDG(qubit)  # type: ignore

    def sx(self, qubit: int):
        """
        Apply an SX gate to the specified qubit (√X).

        Args:
            qubit (int): Target qubit index
        """
        self._validate_qubit(qubit)
        self.SX(qubit)  # type: ignore

    # Single-qubit parametric gates
    def rx(self, qubit: int, theta: float):
        """
        Apply a rotation around X-axis to the specified qubit.

        Args:
            qubit (int): Target qubit index
            theta (float): Rotation angle in radians
        """
        self._validate_qubit(qubit)
        self.RX(qubit, theta)  # type: ignore

    def ry(self, qubit: int, theta: float):
        """
        Apply a rotation around Y-axis to the specified qubit.

        Args:
            qubit (int): Target qubit index
            theta (float): Rotation angle in radians
        """
        self._validate_qubit(qubit)
        self.RY(qubit, theta)  # type: ignore

    def rz(self, qubit: int, theta: float):
        """
        Apply a rotation around Z-axis to the specified qubit.

        Args:
            qubit (int): Target qubit index
            theta (float): Rotation angle in radians
        """
        self._validate_qubit(qubit)
        self.RZ(qubit, theta)  # type: ignore

    def p(self, qubit: int, theta: float):
        """
        Apply a phase rotation to the specified qubit.

        Args:
            qubit (int): Target qubit index
            theta (float): Phase angle in radians
        """
        self._validate_qubit(qubit)
        self.P(qubit, theta)  # type: ignore

    # Two-qubit gates
    def cx(self, control_qubit: int, target_qubit: int):
        """
        Apply a controlled-X (CNOT) gate with the specified control and target qubits.

        Args:
            control_qubit (int): Control qubit index
            target_qubit (int): Target qubit index
        """
        self._validate_unique_qubits(control_qubit, target_qubit)
        self._validate_qubit(control_qubit)
        self._validate_qubit(target_qubit)
        self.CX(control_qubit, target_qubit)  # type: ignore

    def cy(self, control_qubit: int, target_qubit: int):
        """
        Apply a controlled-Y gate with the specified control and target qubits.

        Args:
            control_qubit (int): Control qubit index
            target_qubit (int): Target qubit index
        """
        self._validate_unique_qubits(control_qubit, target_qubit)
        self._validate_qubit(control_qubit)
        self._validate_qubit(target_qubit)
        self.CY(control_qubit, target_qubit)  # type: ignore

    def cz(self, control_qubit: int, target_qubit: int):
        """
        Apply a controlled-Z gate with the specified control and target qubits.

        Args:
            control_qubit (int): Control qubit index
            target_qubit (int): Target qubit index
        """
        self._validate_unique_qubits(control_qubit, target_qubit)
        self._validate_qubit(control_qubit)
        self._validate_qubit(target_qubit)
        self.CZ(control_qubit, target_qubit)  # type: ignore

    def swap(self, qubit1: int, qubit2: int):
        """
        Apply a SWAP gate between the specified qubits.

        Args:
            qubit1 (int): First qubit index
            qubit2 (int): Second qubit index
        """
        self._validate_unique_qubits(qubit1, qubit2)
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self.SWAP(qubit1, qubit2)  # type: ignore

    def iswap(self, qubit1: int, qubit2: int):
        """
        Apply a iSWAP gate between the specified qubits.

        Args:
            qubit1 (int): First qubit index
            qubit2 (int): Second qubit index
        """
        self._validate_unique_qubits(qubit1, qubit2)
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self.ISWAP(qubit1, qubit2)  # type: ignore

    def cp(self, control_qubit: int, target_qubit: int, theta: float):
        """
        Apply a controlled phase rotation with the specified control and target qubits.

        Args:
            control_qubit (int): Control qubit index
            target_qubit (int): Target qubit index
            theta (float): Phase angle in radians
        """
        self._validate_unique_qubits(control_qubit, target_qubit)
        self._validate_qubit(control_qubit)
        self._validate_qubit(target_qubit)
        self.CP(control_qubit, target_qubit, theta)  # type: ignore

    def rzz(self, qubit1: int, qubit2: int, theta: float):
        """
        Apply an RZZ gate (two-qubit rotation around ZZ) between the specified qubits.

        Args:
            qubit1 (int): First qubit index
            qubit2 (int): Second qubit index
            theta (float): Rotation angle in radians
        """
        self._validate_unique_qubits(qubit1, qubit2)
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self.RZZ(qubit1, qubit2, theta)  # type: ignore

    def rxx(self, qubit1: int, qubit2: int, theta: float):
        """
        Apply an RXX gate (two-qubit rotation around XX) between the specified qubits.

        Args:
            qubit1 (int): First qubit index
            qubit2 (int): Second qubit index
            theta (float): Rotation angle in radians
        """
        self._validate_unique_qubits(qubit1, qubit2)
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self.RXX(qubit1, qubit2, theta)  # type: ignore

    def ryy(self, qubit1: int, qubit2: int, theta: float):
        """
        Apply an RYY gate (two-qubit rotation around YY) between the specified qubits.

        Args:
            qubit1 (int): First qubit index
            qubit2 (int): Second qubit index
            theta (float): Rotation angle in radians
        """
        self._validate_unique_qubits(qubit1, qubit2)
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self.RYY(qubit1, qubit2, theta)  # type: ignore

    def crx(self, control_qubit: int, target_qubit: int, theta: float):
        """
        Apply a controlled RX gate.

        Args:
            control_qubit (int): Control qubit index
            target_qubit (int): Target qubit index
            theta (float): Rotation angle in radians
        """
        self._validate_unique_qubits(control_qubit, target_qubit)
        self._validate_qubit(control_qubit)
        self._validate_qubit(target_qubit)
        self.CRX(control_qubit, target_qubit, theta)  # type: ignore

    def cry(self, control_qubit: int, target_qubit: int, theta: float):
        """
        Apply a controlled RY gate.

        Args:
            control_qubit (int): Control qubit index
            target_qubit (int): Target qubit index
            theta (float): Rotation angle in radians
        """
        self._validate_unique_qubits(control_qubit, target_qubit)
        self._validate_qubit(control_qubit)
        self._validate_qubit(target_qubit)
        self.CRY(control_qubit, target_qubit, theta)  # type: ignore

    def crz(self, control_qubit: int, target_qubit: int, theta: float):
        """
        Apply a controlled RZ gate.

        Args:
            control_qubit (int): Control qubit index
            target_qubit (int): Target qubit index
            theta (float): Rotation angle in radians
        """
        self._validate_unique_qubits(control_qubit, target_qubit)
        self._validate_qubit(control_qubit)
        self._validate_qubit(target_qubit)
        self.CRZ(control_qubit, target_qubit, theta)  # type: ignore

    def ch(self, control_qubit: int, target_qubit: int):
        """
        Apply a controlled H gate.

        Args:
            control_qubit (int): Control qubit index
            target_qubit (int): Target qubit index
        """
        self._validate_unique_qubits(control_qubit, target_qubit)
        self._validate_qubit(control_qubit)
        self._validate_qubit(target_qubit)
        self.CH(control_qubit, target_qubit)  # type: ignore

    def cs(self, control_qubit: int, target_qubit: int):
        """
        Apply a controlled S gate.

        Args:
            control_qubit (int): Control qubit index
            target_qubit (int): Target qubit index
        """
        self._validate_unique_qubits(control_qubit, target_qubit)
        self._validate_qubit(control_qubit)
        self._validate_qubit(target_qubit)
        self.CS(control_qubit, target_qubit)  # type: ignore

    def ecr(self, qubit1: int, qubit2: int):
        """
        Apply an ECR (echoed cross-resonance) gate between the specified qubits.

        Args:
            qubit1 (int): First qubit index
            qubit2 (int): Second qubit index
        """
        self._validate_unique_qubits(qubit1, qubit2)
        self._validate_qubit(qubit1)
        self._validate_qubit(qubit2)
        self.ECR(qubit1, qubit2)  # type: ignore

    def u(self, qubit: int, theta: float, phi: float, lam: float):
        """
        Apply a single-qubit unitary gate specified by three Euler angles.

        The U gate has the matrix form::

            [[cos(θ/2), -e^{iλ}sin(θ/2)],
             [e^{iφ}sin(θ/2),  e^{i(φ+λ)}cos(θ/2)]]

        Args:
            qubit (int): Qubit index
            theta (float): Polar angle in radians
            phi (float): Azimuthal angle in radians
            lam (float): Phase angle in radians
        """
        self._validate_qubit(qubit)
        self.U(qubit, theta, phi, lam)  # type: ignore

    def sxdg(self, qubit: int):
        """
        Apply the inverse Sqrt(X) (SX†) gate.

        Args:
            qubit (int): Qubit index
        """
        self._validate_qubit(qubit)
        self.SXDG(qubit)  # type: ignore

    def ccx(self, control_qubit1: int, control_qubit2: int, target_qubit: int):
        """
        Apply a Toffoli (CCX) gate with the specified control and target qubits.

        Args:
            control_qubit1 (int): First control qubit index
            control_qubit2 (int): Second control qubit index
            target_qubit (int): Target qubit index
        """
        self._validate_unique_qubits(control_qubit1, control_qubit2, target_qubit)
        self._validate_qubit(control_qubit1)
        self._validate_qubit(control_qubit2)
        self._validate_qubit(target_qubit)
        self.CCX(control_qubit1, control_qubit2, target_qubit)  # type: ignore

    def cswap(self, control_qubit: int, target_qubit1: int, target_qubit2: int):
        """
        Apply a controlled SWAP (Fredkin) gate with the specified control and target qubits.

        Args:
            control_qubit (int): Control qubit index
            target_qubit1 (int): First target qubit index (to be swapped)
            target_qubit2 (int): Second target qubit index (to be swapped)
        """
        self._validate_unique_qubits(control_qubit, target_qubit1, target_qubit2)
        self._validate_qubit(control_qubit)
        self._validate_qubit(target_qubit1)
        self._validate_qubit(target_qubit2)
        self.CSWAP(control_qubit, target_qubit1, target_qubit2)  # type: ignore

    def mcx(self, control_qubits: List[int], target_qubit: int):
        """
        Apply a multi-controlled X (MCX) gate.

        Args:
            control_qubits (List[int]): List of control qubit indices
            target_qubit (int): Target qubit index
        """
        self._validate_unique_qubits(*(control_qubits + [target_qubit]))
        for ctrl in control_qubits:
            self._validate_qubit(ctrl)
        self._validate_qubit(target_qubit)
        self.MCX(control_qubits, target_qubit)  # type: ignore

    def barrier(self, *qubits: int):
        self._validate_unique_qubits(*qubits)
        for qubit in qubits:
            self._validate_qubit(qubit)
        self.BARRIER(*qubits)  # type: ignore

    def measure(self, qubit: Union[int, List[int]], clbit: Union[int, List[int]]):
        if isinstance(qubit, list) and isinstance(clbit, list):
            if len(qubit) != len(clbit):
                raise CircuitError(
                    f"Qubit and clbit lists must have the same length. "
                    f"Got {len(qubit)} qubits and {len(clbit)} classical bits."
                )
            for q, c in zip(qubit, clbit):
                self._validate_qubit(q)
                self._validate_clbit(c)
                self.MEASURE(q, c)  # type: ignore
        elif isinstance(qubit, list) or isinstance(clbit, list):
            raise CircuitError(
                "Both qubit and clbit must be either integers or lists. "
                f"Got qubit type: {type(qubit).__name__}, clbit type: {type(clbit).__name__}"
            )
        else:
            self._validate_qubit(qubit)
            self._validate_clbit(clbit)
            self.MEASURE(qubit, clbit)  # type: ignore

    def measure_all(self):
        for qubit in range(self.icr.num_qubits):
            self.measure(qubit, qubit)

    def remove_operation(self, upto: int = 1):
        self.icr._remove_operation(upto)

    def to_qasm(self):
        from qpiai_quantum.iem.qasm.v2.exporter import QASM2

        return QASM2.generate(self)

    def show(self, theme: str = "light", dpi: int = 200, use_mathtext: bool = True):
        """Render the circuit with the matplotlib visualizer."""
        from qpiai_quantum.visualization.matplotlib_visualizer.visualizer import (
            MatplotlibVisualizer,
        )

        MatplotlibVisualizer.plot_icr(
            self.icr, theme=theme, dpi=dpi, use_mathtext=use_mathtext
        )

    def run(
        self,
        shots: int = 1024,
        experiment_name: str = "Default Experiment",
        need_statevector: bool = False,
        need_density_matrix: bool = False,
        device_name: str = "QpiAI-QSV-Simulator",
        reverse_bits: bool = False,
        **kwargs,
    ) -> Union["JobResult", "BaseQuantumResult"]:
        from ..jobmanager import ExecutionEngine, Backend as BackendEnum

        if device_name not in [
            "QpiAI-QSV-Simulator",
            "QpiAI-QDM-Simulator",
            "QpiAI-QTN-Simulator",
            "QpiAI-Indus-1",
            "QpiAI-QSV-Lite",
            "QpiAI-QDM-Lite",
            "QpiAI-QSV-Local",
        ]:
            raise ValueError(
                f"Unsupported device name '{device_name}'. Supported devices are: "
                f"['QpiAI-QSV-Simulator', 'QpiAI-QDM-Simulator', "
                f"'QpiAI-QTN-Simulator', 'QpiAI-Indus-1', "
                f"'QpiAI-QSV-Lite', 'QpiAI-QDM-Lite', 'QpiAI-QSV-Local']"
            )
        if (
            device_name == "QpiAI-QDM-Simulator" or device_name == "QpiAI-QDM-Lite"
        ) and need_statevector:
            raise ValueError("Cannot get statevector from Density Matrix simulator.")

        if device_name == "QpiAI-QSV-Local":
            simulator = StatevectorSimulator()
            result = simulator.run(self, shots=shots, name=experiment_name)
            if reverse_bits and result.counts:
                result.counts = {
                    bitstring[::-1]: count for bitstring, count in result.counts.items()
                }
            return result
        circuit_name = kwargs.pop("circuit_name", None)
        overwrite = kwargs.pop("overwrite", True)
        timeout = kwargs.pop("timeout", 500)

        method = _map_device_name_to_method(device_name)

        if circuit_name is None:
            import uuid

            circuit_name = f"{self.name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        method = kwargs.pop("method", method)
        device_name = kwargs.pop("device_name", device_name)
        job_result = ExecutionEngine.execute_circuit(
            circuit=self,
            shots=shots,
            need_density_matrix=need_density_matrix,
            need_statevector=need_statevector,
            experiment_name=experiment_name,
            circuit_name=circuit_name,
            overwrite=overwrite,
            timeout=timeout,
            method=method,
            device_name=device_name,
            **kwargs,
        )
        if reverse_bits and job_result.counts:
            job_result.counts = {
                bitstring[::-1]: count for bitstring, count in job_result.counts.items()
            }

        return job_result

    @staticmethod
    def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
        from ..jobmanager import JobManager

        job_manager = JobManager()
        return job_manager.get_job_status(job_id)

    @staticmethod
    def check_job_status(job_id: str) -> Optional[Dict[str, Any]]:
        return Circuit.get_job_status(job_id)

    @staticmethod
    def get_job_results(job_id: str) -> "JobResult":
        from ..jobmanager import JobManager

        job_manager = JobManager()
        return job_manager.get_job_results(job_id)

    @staticmethod
    def cancel_job(job_id: str) -> Dict[str, Any]:
        from ..jobmanager import JobManager

        job_manager = JobManager()
        return job_manager.cancel_job(job_id)

    @staticmethod
    def delete_job(job_id: str) -> Dict[str, Any]:
        from ..jobmanager import JobManager

        job_manager = JobManager()
        return job_manager.delete_job(job_id)

    @staticmethod
    def get_current_job() -> Optional[Dict[str, Any]]:
        from ..jobmanager import JobManager

        job_manager = JobManager()
        return job_manager.get_current_job()

    @staticmethod
    def get_job_history(
        period: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        from ..jobmanager import JobManager

        job_manager = JobManager()
        return job_manager.get_job_history(period, status, page, page_size)

    @staticmethod
    def list_jobs(
        period: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        return Circuit.get_job_history(period, status, page, page_size)

    def depth(self) -> int:
        if not self.icr.evolve:
            return 0

        qubit_depths = [0] * self.icr.num_qubits
        for operation in self.icr.evolve:
            if operation.operation_type == OperationType.BARRIER:
                continue
            if operation.operation_type == OperationType.MEASURE:
                continue

            involved_qubits = operation.qubits if operation.qubits else []

            if not involved_qubits:
                continue

            max_depth = max(qubit_depths[q] for q in involved_qubits)
            for q in involved_qubits:
                qubit_depths[q] = max_depth + 1

            actual_qubit = max(qubit_depths)
        return actual_qubit if qubit_depths else 0

    def size(self) -> int:
        return len(self.icr.evolve)

    @property
    def num_qubits(self) -> int:
        return self.icr.num_qubits

    @property
    def num_clbits(self) -> int:
        return self.icr.num_clbits

    def list_gates(self) -> Dict:
        # NOTE: Clifford gates set
        clifford_gate_names = {
            "H",
            "X",
            "Y",
            "Z",
            "S",
            "SDG",
            "SX",
            "CX",
            "CY",
            "CZ",
            "SWAP",
            "ISWAP",
            "ID",
        }

        stats: Dict[str, Any] = {
            "total_operations": 0,
            "total_gates": 0,
            "single_qubit_gates": 0,
            "two_qubit_gates": 0,
            "multi_qubit_gates": 0,
            "clifford_gates": 0,
            "non_clifford_gates": 0,
            "parametric_gates": 0,
            "measurements": 0,
            "barriers": 0,
            "gate_counts": {},
        }

        for operation in self.icr.evolve:
            stats["total_operations"] += 1
            gate_name = operation.__class__.__name__
            if "Gate" in gate_name:
                gate_name = gate_name.replace("Gate", "")
            elif "Operation" in gate_name:
                gate_name = gate_name.replace("Operation", "")

            if operation.operation_type == OperationType.MEASURE:
                stats["measurements"] += 1
                stats["gate_counts"]["MEASURE"] = (
                    stats["gate_counts"].get("MEASURE", 0) + 1
                )
            elif operation.operation_type == OperationType.BARRIER:
                stats["barriers"] += 1
                stats["gate_counts"]["BARRIER"] = (
                    stats["gate_counts"].get("BARRIER", 0) + 1
                )
            else:
                stats["total_gates"] += 1
                stats["gate_counts"][gate_name.upper()] = (
                    stats["gate_counts"].get(gate_name.upper(), 0) + 1
                )
                num_qubits = len(operation.qubits) if operation.qubits else 0
                if num_qubits == 1:
                    stats["single_qubit_gates"] += 1
                elif num_qubits == 2:
                    stats["two_qubit_gates"] += 1
                elif num_qubits >= 3:
                    stats["multi_qubit_gates"] += 1
                if gate_name.upper() in clifford_gate_names:
                    stats["clifford_gates"] += 1
                else:
                    stats["non_clifford_gates"] += 1
                if operation.params and len(operation.params) > 0:
                    stats["parametric_gates"] += 1

        return stats
