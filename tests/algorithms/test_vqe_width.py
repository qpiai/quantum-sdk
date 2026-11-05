"""VQE register-width contracts, isolated from authenticated circuit dispatch."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver
from qpiai_quantum.circuit import Circuit


pytestmark = pytest.mark.usefixtures("backend_devices")


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
    return (
        initial_point,
        objective(initial_point),
        {"nfev": 1, "nit": 1, "success": True},
    )


@pytest.mark.parametrize("method", ["statevector", "density_matrix"])
@pytest.mark.parametrize("constant", [False, True])
def test_run_preserves_declared_width_and_solver_reuse(
    method: str, constant: bool
) -> None:
    device_name = (
        "QpiAI-QDM-Simulator" if method == "density_matrix" else "QpiAI-QSV-Simulator"
    )
    factory_widths: list[int] = []

    def ansatz_factory(n_qubits: int) -> Circuit:
        factory_widths.append(n_qubits)
        circuit = Circuit(n_qubits)
        circuit.x(2)
        return circuit

    solver = VQESolver(n_qubits=3, ansatz=ansatz_factory, verbose=False)
    first_hamiltonian = Hamiltonian([([], 2.5)] if constant else [([(0, "Z")], 1.0)])
    execution_widths: list[int] = []

    def execute(circuit: Circuit, **kwargs: Any) -> SimpleNamespace:
        # Analytic results for X(q2)|000> = |100>, with q0 at the right.
        execution_widths.append(circuit.num_qubits)
        statevector = np.zeros(8, dtype=complex)
        statevector[4] = 1.0
        return SimpleNamespace(counts={"100": 64}, statevector=statevector)

    with (
        patch.object(solver, "_optimize_parameters", side_effect=evaluate_once),
        patch.object(Circuit, "run", autospec=True, side_effect=execute),
    ):
        first = solver.run(
            hamiltonian=first_hamiltonian,
            device_name=device_name,
            method=method,
            shots=64,
        )
        assert first.optimal_energy == pytest.approx(2.5 if constant else 1.0)
        assert first.bitstring == "100"
        assert solver.num_qubits == 3

        # A later Hamiltonian may use a previously unmentioned qubit.
        second = solver.run(
            hamiltonian=Hamiltonian([([(2, "Z")], 1.0)]),
            device_name=device_name,
            method=method,
            shots=64,
        )
        assert second.optimal_energy == pytest.approx(-1.0)

    assert factory_widths == [3, 3]
    assert execution_widths and set(execution_widths) == {3}
    assert solver.num_qubits == 3
    assert solver.circuit is not None and solver.circuit.num_qubits == 3


def test_builtin_ansatz_keeps_declared_width_for_partial_hamiltonian() -> None:
    solver = VQESolver(n_qubits=3, ansatz="standard", verbose=False)
    expected_parameters = solver.num_parameters
    with (
        patch.object(solver, "_optimize_parameters", side_effect=evaluate_once),
        patch.object(
            Circuit,
            "run",
            autospec=True,
            side_effect=lambda circuit, **kwargs: SimpleNamespace(
                counts={"0" * circuit.num_qubits: 64}, statevector=None
            ),
        ) as execute,
    ):
        result = solver.run(
            hamiltonian=Hamiltonian([([(0, "Z")], 1.0)]),
            device_name="QpiAI-QDM-Simulator",
            shots=64,
        )

    assert solver.num_qubits == 3
    assert solver.num_parameters == expected_parameters
    assert len(result.optimal_parameters) == expected_parameters
    assert all(call.args[0].num_qubits == 3 for call in execute.call_args_list)


def test_hamiltonian_outside_declared_register_fails_before_execution() -> None:
    solver = VQESolver(n_qubits=2, verbose=False)
    with (
        patch.object(solver, "_optimize_parameters") as optimize,
        patch.object(Circuit, "run") as execute,
    ):
        with pytest.raises(
            ValueError, match="Hamiltonian requires 3 qubits.*declared.*2"
        ):
            solver.run(
                hamiltonian=Hamiltonian([([(2, "Z")], 1.0)]),
                device_name="QpiAI-QDM-Simulator",
            )
    optimize.assert_not_called()
    execute.assert_not_called()
    assert solver.num_qubits == 2


@pytest.mark.parametrize("ansatz_width", [1, 4])
@pytest.mark.parametrize("entrypoint", ["build_circuit", "num_parameters", "run"])
def test_custom_ansatz_width_must_match_declared_register(
    ansatz_width: int, entrypoint: str
) -> None:
    solver = VQESolver(
        n_qubits=3, ansatz=lambda _: Circuit(ansatz_width), verbose=False
    )
    with patch.object(Circuit, "run") as execute:
        with pytest.raises(
            ValueError,
            match=rf"Ansatz has {ansatz_width} qubits.*declared.*3",
        ):
            if entrypoint == "run":
                solver.run(
                    hamiltonian=Hamiltonian([([(2, "Z")], 1.0)]),
                    device_name="QpiAI-QDM-Simulator",
                )
            elif entrypoint == "build_circuit":
                solver.build_circuit()
            else:
                _ = solver.num_parameters
    execute.assert_not_called()
    assert solver.num_qubits == 3
