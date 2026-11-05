"""Numerical VQE regressions with circuit execution isolated from authentication."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.algorithms.opt import parameterize


class Hamiltonian:
    def __init__(self, terms: list[tuple[list[tuple[int, str]], float]]) -> None:
        self.terms = terms

    def get_hamiltonian_terms(self) -> list[tuple[list[tuple[int, str]], float]]:
        return self.terms


def evaluate_once(
    objective: Callable[[np.ndarray], float],
    initial_point: np.ndarray,
    **kwargs: Any,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    parameters = np.asarray(initial_point, dtype=float)
    energy = objective(parameters)
    return parameters, energy, {"nfev": 1, "nit": 1, "success": True}


@pytest.mark.parametrize("reverse_bits", [False, True])
@pytest.mark.parametrize("axis", ["Z", "X", "Y"])
def test_count_display_order_does_not_change_physical_energy(
    reverse_bits: bool, axis: str
) -> None:
    ansatz = Circuit(3)
    if axis == "Z":
        ansatz.x(0)
    else:
        ansatz.h(0)
        if axis == "X":
            ansatz.z(0)
        else:
            ansatz.sdg(0)
    solver = VQESolver(n_qubits=3, ansatz=lambda _: ansatz, verbose=False)
    with (
        patch.object(solver, "_optimize_parameters", side_effect=evaluate_once),
        patch.object(
            solver, "_execute_circuit", wraps=solver._execute_circuit
        ) as execute,
        patch(
            "qpiai_quantum.circuit.Circuit.run",
            autospec=True,
            side_effect=lambda circuit, **kwargs: SimpleNamespace(
                counts={"100" if kwargs.get("reverse_bits") else "001": 128},
                statevector=None,
            ),
        ),
    ):
        result = solver.run(
            hamiltonian=Hamiltonian([([(0, axis)], 1.0), ([(2, "Z")], 2.0)]),
            device_name="QpiAI-QSV-Lite",
            method="statevector_sampling",
            reverse_bits=reverse_bits,
            shots=128,
        )
    assert result.optimal_energy == pytest.approx(1.0)
    assert result.counts == {"100" if reverse_bits else "001": 128}
    assert result.bitstring == ("100" if reverse_bits else "001")
    assert [call.kwargs["reverse_bits"] for call in execute.call_args_list] == [
        False,
        False,
        reverse_bits,
    ]


@pytest.mark.parametrize("missing_call", [1, 2])
def test_missing_objective_or_fresh_energy_counts_raise(missing_call: int) -> None:
    ansatz = Circuit(1)
    solver = VQESolver(n_qubits=1, ansatz=lambda _: ansatz, verbose=False)
    responses = [SimpleNamespace(counts={"0": 128}, statevector=None) for _ in range(3)]
    responses[missing_call - 1] = SimpleNamespace(counts={}, statevector=None)
    with (
        patch.object(solver, "_optimize_parameters", side_effect=evaluate_once),
        patch(
            "qpiai_quantum.circuit.Circuit.run",
            autospec=True,
            side_effect=responses,
        ),
    ):
        with pytest.raises(RuntimeError, match="no measurement counts"):
            solver.run(
                hamiltonian=Hamiltonian([([(0, "Z")], 1.0)]),
                device_name="QpiAI-QSV-Lite",
                method="statevector_sampling",
                shots=128,
            )


def test_one_shot_grouped_energy_does_not_claim_known_uncertainty() -> None:
    ansatz = Circuit(1)
    solver = VQESolver(n_qubits=1, ansatz=lambda _: ansatz, verbose=False)
    with (
        patch.object(solver, "_optimize_parameters", side_effect=evaluate_once),
        patch(
            "qpiai_quantum.circuit.Circuit.run",
            autospec=True,
            return_value=SimpleNamespace(counts={"0": 1}, statevector=None),
        ),
    ):
        result = solver.run(
            hamiltonian=Hamiltonian([([(0, "X")], 0.5), ([(0, "Y")], 0.25)]),
            device_name="QpiAI-QSV-Lite",
            method="statevector_sampling",
            shots=1,
        )
    assert result.optimal_energy == pytest.approx(0.75)
    assert result.energy_standard_error is None


@pytest.mark.parametrize("initial_point", [[], [np.pi]])
def test_list_initial_point_returns_parameters_when_initial_is_best(
    initial_point: list[float],
) -> None:
    ansatz = Circuit(1)
    if initial_point:
        ansatz.ry(0, 0.0)
        parameterize(ansatz, 0)
    else:
        ansatz.x(0)
    solver = VQESolver(
        n_qubits=1,
        ansatz=lambda _: ansatz,
        optimizer="adam",
        initial_point=initial_point,
        verbose=False,
    )

    def analytic_counts(circuit: Circuit, shots: int, **kwargs: Any) -> SimpleNamespace:
        rotations = [op for op in circuit.icr.evolve if op.params]
        zeros = (
            round(shots * (1 + np.cos(rotations[0].params[0])) / 2) if rotations else 0
        )
        return SimpleNamespace(
            counts={"0": zeros, "1": shots - zeros}, statevector=None
        )

    with patch(
        "qpiai_quantum.circuit.Circuit.run",
        autospec=True,
        side_effect=analytic_counts,
    ):
        result = solver.run(
            hamiltonian=Hamiltonian([([(0, "Z")], 1.0)]),
            device_name="QpiAI-QSV-Lite",
            method="statevector_sampling",
            shots=128,
        )
    assert result.optimal_parameters == pytest.approx(initial_point)
    assert result.optimal_energy == pytest.approx(-1.0)


@pytest.mark.parametrize("dtype", [float, int])
def test_integer_initial_parameters_do_not_truncate_adam_gradients(
    dtype: type[float] | type[int],
) -> None:
    ansatz = Circuit(1)
    ansatz.ry(0, 0.0)
    parameterize(ansatz, 0)
    solver = VQESolver(
        n_qubits=1,
        ansatz=lambda _: ansatz,
        optimizer="adam",
        initial_point=np.array([1], dtype=dtype),
        max_iterations=2,
        verbose=False,
    )

    def analytic_counts(circuit: Circuit, shots: int, **kwargs: Any) -> SimpleNamespace:
        # For Ry(theta)|0>, measuring Z gives P(0) = (1 + cos(theta))/2.
        rotation = next(op for op in circuit.icr.evolve if op.params)
        zeros = round(shots * (1 + np.cos(rotation.params[0])) / 2)
        return SimpleNamespace(
            counts={"0": zeros, "1": shots - zeros}, statevector=None
        )

    with patch(
        "qpiai_quantum.circuit.Circuit.run",
        autospec=True,
        side_effect=analytic_counts,
    ):
        result = solver.run(
            hamiltonian=Hamiltonian([([(0, "Z")], 1.0)]),
            device_name="QpiAI-QSV-Lite",
            method="statevector_sampling",
            shots=1_000_000,
        )
    assert result.optimal_parameters[0] > 1.0
    assert result.optimal_energy < np.cos(1.0) - 1e-4
