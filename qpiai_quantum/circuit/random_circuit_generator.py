from typing import Optional
import random
import math
from qpiai_quantum.circuit.circuit import Circuit


class RandomCircuitGenerator:
    def __init__(
        self,
        num_qubits: int,
        depth: int,
        max_gates: int | None = None,
        allow_parametric: bool = True,
        single_qubit_gates: list[str] | None = None,
        parametric_single_qubit_gates: list[str] | None = None,
        two_qubit_gates: list[str] | None = None,
        parametric_two_qubit_gates: list[str] | None = None,
        three_qubit_gates: list[str] | None = None,
        add_measurements: bool = True,
        single_qubit_probability: float = 0.6,
        two_qubit_probability: float = 0.3,
        parametric_probability: float = 0.3,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.max_gates = max_gates

        if self.max_gates is not None:
            if self.max_gates > self.num_qubits * self.depth:
                raise ValueError(
                    f"Impossible request: max_gates={self.max_gates} exceeds "
                    f"depth × num_qubits = {self.depth * self.num_qubits}."
                )

        self.allow_parametric = allow_parametric
        self.add_measurements = add_measurements

        self.single_qubit_probability = single_qubit_probability
        self.two_qubit_probability = two_qubit_probability
        self.parametric_probability = parametric_probability

        self.single_qubit_gates = single_qubit_gates or [
            "h",
            "x",
            "t",
            "s",
            "z",
            "sx",
            "y",
        ]

        self.parametric_single_qubit_gates = parametric_single_qubit_gates or [
            "rx",
            "ry",
            "rz",
        ]

        self.two_qubit_gates = two_qubit_gates or ["cx", "cz", "swap"]

        self.parametric_two_qubit_gates = parametric_two_qubit_gates or ["rzz"]

        self.three_qubit_gates = three_qubit_gates or ["ccx"]

        self.circ = Circuit(num_qubits, num_qubits)
        self._gate_counter = 0

    def _apply_single_qubit_gate_on(self, qubit: int):
        if self.allow_parametric and random.random() < self.parametric_probability:
            gate = random.choice(self.parametric_single_qubit_gates)
            angle = random.uniform(0.0, 2.0 * math.pi)
            getattr(self.circ, gate)(qubit, angle)
        else:
            gate = random.choice(self.single_qubit_gates)
            getattr(self.circ, gate)(qubit)

    def _apply_two_qubit_gate_on(self, q1: int, q2: int):
        control, target = sorted((q1, q2))

        if self.allow_parametric and random.random() < self.parametric_probability:
            gate = random.choice(self.parametric_two_qubit_gates)
            angle = random.uniform(0.0, 2.0 * math.pi)
            getattr(self.circ, gate)(control, target, angle)
        else:
            gate = random.choice(self.two_qubit_gates)
            getattr(self.circ, gate)(control, target)

    def _apply_three_qubit_gate_on(self, q1: int, q2: int, q3: int):
        c1, c2, t = sorted((q1, q2, q3))
        gate = random.choice(self.three_qubit_gates)
        getattr(self.circ, gate)(c1, c2, t)

    def _sample_gate_arity(self):
        if self.num_qubits == 1:
            return 1

        r = random.random()

        if self.num_qubits == 2:
            return 1 if r < self.single_qubit_probability else 2

        if r < self.single_qubit_probability:
            return 1
        elif r < self.single_qubit_probability + self.two_qubit_probability:
            return 2
        else:
            return 3

    def _apply_layer(self):
        used = set()

        while True:
            if self.max_gates is not None:
                if self._gate_counter >= self.max_gates:
                    return

            free = [q for q in range(self.num_qubits) if q not in used]

            if not free:
                break

            arity = self._sample_gate_arity()

            if arity == 1 and len(free) >= 1:
                q = random.choice(free)
                self._apply_single_qubit_gate_on(q)
                used.add(q)
                self._gate_counter += 1

            elif arity == 2 and len(free) >= 2:
                q1, q2 = random.sample(free, 2)
                self._apply_two_qubit_gate_on(q1, q2)
                used.add(q1)
                used.add(q2)
                self._gate_counter += 1

            elif arity == 3 and len(free) >= 3:
                q1, q2, q3 = random.sample(free, 3)
                self._apply_three_qubit_gate_on(q1, q2, q3)
                used.add(q1)
                used.add(q2)
                used.add(q3)
                self._gate_counter += 1

            else:
                break

    def generate(self) -> Circuit:
        self._gate_counter = 0

        for _ in range(self.depth):
            self._apply_layer()

        if self.add_measurements:
            self.circ.measure_all()

        return self.circ

    def __iter__(self):
        return self

    def __next__(self):
        self.circ = Circuit(self.num_qubits, self.num_qubits)
        return self.generate()
