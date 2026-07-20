import math
import copy
from typing import Optional, Any
from collections.abc import Callable
import scipy.optimize
import numpy as np

from ..circuit import Circuit
from ..icr.circuitoperation import (
    CircuitOperation,
    OperationType,
    HGate,
    XGate,
    YGate,
    ZGate,
    IDGate,
    CXGate,
    CYGate,
    CZGate,
    SwapGate,
    CCXGate,
    CSwapGate,
    MCXGate,
    SGate,
    SDGGate,
    TGate,
    TDGGate,
    RXGate,
    RYGate,
    RZGate,
    PGate,
    CPGate,
    RZZGate,
    MeasureOperation,
    BarrierOperation,
)
from .base import QuantumAlgorithm
from .qft import QFT


class EstimationProblem:
    """
    Defines the problem for Amplitude Estimation.
    """

    def __init__(
        self,
        state_preparation: Circuit,
        objective_qubits: list[int],
        is_good_state: Callable[[str], bool] | None = None,
    ):
        self.state_preparation = state_preparation
        self.objective_qubits = objective_qubits
        self.is_good_state: Callable[[str], bool]
        if is_good_state is None:

            def default_is_good_state(bitstring: str) -> bool:
                """By default we assume if any of the objective qubits measured '1' it's good state, or better:
                typically all objective qubits being '1' indicates the marked state."""
                for q in objective_qubits:
                    # Reverse bitstring if reading right-to-left
                    idx = len(bitstring) - 1 - q
                    if idx >= 0 and bitstring[idx] == "0":
                        return False
                return True

            self.is_good_state = default_is_good_state
        else:
            self.is_good_state = is_good_state

    @property
    def num_qubits(self) -> int:
        return self.state_preparation.num_qubits


class AmplitudeEstimation(QuantumAlgorithm):
    """
    Canonical Quantum Amplitude Estimation (QAE) — **not yet implemented**.

    The canonical QPE-based variant is not yet available.  Canonical QAE
    requires constructing controlled Grover operators, which in turn need
    a general controlled-subcircuit compiler not yet present in the SDK.

    Use :class:`IterativeAmplitudeEstimation` for currently supported
    amplitude estimation workflows.
    """

    def __init__(self, num_evaluation_qubits: int):
        self.num_evaluation_qubits = num_evaluation_qubits
        super().__init__(num_qubits=0, name="Amplitude Estimation")
        self.description = "Canonical Amplitude Estimation using QPE"

    def build_circuit(self, problem: EstimationProblem) -> Circuit:
        raise NotImplementedError(
            "Canonical QAE requires native implementation of controlled circuits."
        )

    def estimate(self, problem: EstimationProblem) -> float:
        raise NotImplementedError(
            "Canonical QAE requires native implementation of controlled sub-circuits, "
            "which are not yet fully supported by the basic Circuit builder. "
            "Please use IterativeAmplitudeEstimation instead."
        )


class IterativeAmplitudeEstimation(QuantumAlgorithm):
    """
    Iterative Amplitude Estimation / ML Amplitude Estimation.
    """

    def __init__(self, epsilon_target: float, alpha: float):
        self.epsilon_target = epsilon_target
        self.alpha = alpha
        super().__init__(num_qubits=0, name="Iterative Amplitude Estimation")
        self.description = "Amplitude Estimation without QPE using Iterations"

    def _build_grover_operator(self, problem: EstimationProblem) -> Circuit:
        n = problem.num_qubits
        q_circuit = Circuit(n)

        # S_chi
        if len(problem.objective_qubits) == 1:
            q_circuit.z(problem.objective_qubits[0])
        elif len(problem.objective_qubits) == 2:
            q_circuit.cz(problem.objective_qubits[0], problem.objective_qubits[1])
        else:
            raise NotImplementedError(
                "More than 2 objective qubits is not fully mapped in S_chi yet."
            )

        # A^-1
        A_inv = problem.state_preparation.inverse()
        q_circuit.compose(A_inv)

        # S_0
        for i in range(n):
            q_circuit.x(i)

        if n == 1:
            q_circuit.z(0)
        elif n == 2:
            q_circuit.cz(0, 1)
        else:
            # Emulate MCZ
            last_q = n - 1
            controls = list(range(n - 1))
            q_circuit.h(last_q)
            q_circuit.add_operation(MCXGate(controls, last_q))
            q_circuit.h(last_q)

        for i in range(n):
            q_circuit.x(i)

        # A
        q_circuit.compose(problem.state_preparation)
        return q_circuit

    def build_circuit(self, problem: EstimationProblem, k: int) -> Circuit:
        self.num_qubits = problem.num_qubits
        self.circuit = Circuit(self.num_qubits, self.num_qubits)

        self.circuit.compose(problem.state_preparation)

        grover_op = self._build_grover_operator(problem)
        for _ in range(k):
            self.circuit.compose(grover_op)

        for i in range(self.num_qubits):
            self.circuit.measure(i, i)

        return self.circuit

    def estimate(
        self,
        problem: EstimationProblem,
        shots: int = 1000,
        device_name: str = "QpiAI-QSV-Local",
    ) -> float:
        # Schedule of iterations
        k_schedule = [0, 1, 2, 4, 8]
        h_list = []
        n_list = []

        for k in k_schedule:
            circ = self.build_circuit(problem, k)
            result = circ.run(shots=shots, device_name=device_name)
            counts = result.get()["counts"]

            h = sum(
                count
                for bitstring, count in counts.items()
                if problem.is_good_state(bitstring)
            )
            h_list.append(h)
            n_list.append(shots)

        def nll(theta: float) -> float:
            val = 0.0
            for k, h, n in zip(k_schedule, h_list, n_list):
                p_good = np.sin((2 * k + 1) * theta) ** 2
                p_good = max(min(p_good, 1 - 1e-10), 1e-10)
                val -= h * np.log(p_good) + (n - h) * np.log(1 - p_good)
            return val

        # The log-likelihood is highly oscillatory. We use grid-search to find the global minimum basin, then refine.

        grid = np.linspace(0, np.pi / 2, 1000)
        nll_values = [nll(t) for t in grid]
        best_grid_t = grid[np.argmin(nll_values)]

        res = scipy.optimize.minimize(
            nll, x0=best_grid_t, bounds=[(0, np.pi / 2)], method="L-BFGS-B"
        )
        theta_opt = res.x[0] if isinstance(res.x, np.ndarray) else res.x
        amplitude = np.sin(theta_opt) ** 2
        return float(amplitude)
