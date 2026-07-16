from typing import Optional, List
from qpiai_quantum.circuit import Circuit
from .base import QuantumAlgorithm
import math


class GroverSearch(QuantumAlgorithm):
    def __init__(self, num_qubits: int, target: str | None = None):
        super().__init__(num_qubits=num_qubits, name="Grover's Search")
        self.target = target
        self.description = (
            "Grover's Search Algorithm - quadratic speedup for unstructured search"
        )

        self.num_iterations = int(math.pi / 4 * math.sqrt(2**num_qubits))

    def build_circuit(
        self, target: str | None = None, iterations: int | None = None
    ) -> Circuit:
        if target is not None:
            self.target = target

        if self.target is None:
            raise ValueError("Target bitstring must be specified")

        if len(self.target) != self.num_qubits:
            raise ValueError(
                f"Target bitstring length ({len(self.target)}) must match "
                f"num_qubits ({self.num_qubits})"
            )

        if iterations is None:
            iterations = self.num_iterations

        self.circuit = Circuit(self.num_qubits, self.num_qubits)

        self._initialize()

        for _ in range(iterations):
            self._oracle()
            self._diffusion()

        for i in range(self.num_qubits):
            self.circuit.measure(i, i)

        return self.circuit

    def _initialize(self):
        for i in range(self.num_qubits):
            self.circuit.h(i)

    def _oracle(self):
        for i, bit in enumerate(self.target):
            if bit == "0":
                self.circuit.x(i)

        if self.num_qubits == 1:
            self.circuit.z(0)
        elif self.num_qubits == 2:
            self.circuit.cz(0, 1)
        else:
            last_qubit = self.num_qubits - 1

            self.circuit.h(last_qubit)

            if self.num_qubits == 3:
                self.circuit.ccx(0, 1, 2)
            else:
                self._multi_controlled_x()

            self.circuit.h(last_qubit)

        for i, bit in enumerate(self.target):
            if bit == "0":
                self.circuit.x(i)

    def _multi_controlled_x(self):
        controls = list(range(self.num_qubits - 1))
        target = self.num_qubits - 1
        self.circuit.mcx(controls, target)

    def _diffusion(self):
        for i in range(self.num_qubits):
            self.circuit.h(i)

        for i in range(self.num_qubits):
            self.circuit.x(i)

        if self.num_qubits == 1:
            self.circuit.z(0)
        elif self.num_qubits == 2:
            self.circuit.cz(0, 1)
        else:
            last_qubit = self.num_qubits - 1
            self.circuit.h(last_qubit)

            if self.num_qubits == 3:
                self.circuit.ccx(0, 1, 2)
            else:
                self._multi_controlled_x()

            self.circuit.h(last_qubit)

        for i in range(self.num_qubits):
            self.circuit.x(i)

        for i in range(self.num_qubits):
            self.circuit.h(i)

    def get_success_probability(self, iterations: int | None = None) -> float:
        if iterations is None:
            iterations = self.num_iterations

        N = 2**self.num_qubits
        theta = math.asin(1 / math.sqrt(N))
        prob = math.sin((2 * iterations + 1) * theta) ** 2
        return prob
