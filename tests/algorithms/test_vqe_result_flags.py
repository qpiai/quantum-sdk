"""Exercise result flags through VQE, the base algorithm, and Circuit.run."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver
from qpiai_quantum.circuit import Circuit


class Hamiltonian:
    def __init__(self, axis):
        self.axis = axis

    def get_hamiltonian_terms(self):
        return [([(0, self.axis)], 1.0)]


def make_solver():
    from qpiai_quantum.algorithms.opt import parameterize

    ansatz = Circuit(1)
    ansatz.ry(0, 0.0)
    parameterize(ansatz, 0)
    return VQESolver(
        n_qubits=1,
        ansatz=lambda _: ansatz,
        initial_point=[0.0],
        verbose=False,
    )


def evaluate_once(objective, initial_point, **kwargs):
    parameters = np.asarray(initial_point, dtype=float)
    energy = objective(parameters)
    return (
        parameters,
        energy,
        {
            "nfev": 1,
            "nit": 1,
            "history": [energy],
            "param_history": [parameters],
            "success": True,
        },
    )


@pytest.mark.parametrize("axis", ["Z", "X", "Y"])
@pytest.mark.parametrize(
    "device,options,expected_statevector,expected_density_matrix",
    [
        ("QpiAI-QSV-Simulator", {}, True, False),
        ("QpiAI-QSV-Simulator", {"need_density_matrix": True}, True, True),
        ("QpiAI-QDM-Simulator", {}, False, False),
        ("QpiAI-QDM-Lite", {"need_statevector": False}, False, False),
        ("QpiAI-QDM-Simulator", {"need_density_matrix": True}, False, True),
        ("QpiAI-QSV-Lite", {}, False, False),
        ("QpiAI-QTN-Simulator", {}, False, False),
        ("QpiAI-Indus-1", {}, False, False),
        (
            "QpiAI-QSV-Simulator",
            {"method": "statevector_sampling", "need_statevector": False},
            False,
            False,
        ),
    ],
)
def test_flags_reach_execution_for_objective_fresh_energy_and_final_counts(
    axis, device, options, expected_statevector, expected_density_matrix
):
    solver = make_solver()
    response = SimpleNamespace(counts={"0": 75, "1": 25}, statevector=[[1.0], [0.0]])
    with (
        patch.object(solver, "_optimize_parameters", side_effect=evaluate_once),
        patch(
            "qpiai_quantum.jobmanager.ExecutionEngine.execute_circuit",
            return_value=response,
        ) as execute,
    ):
        result = solver.run(
            hamiltonian=Hamiltonian(axis), device_name=device, shots=100, **options
        )

    exact = result.metadata["method"] == "statevector"
    assert execute.call_count == (2 if exact else 3)
    for call in execute.call_args_list:
        assert call.kwargs["need_statevector"] is expected_statevector
        assert call.kwargs["need_density_matrix"] is expected_density_matrix
        assert call.kwargs["device_name"] == device
    assert result.optimal_energy == pytest.approx(
        (1.0 if axis == "Z" else 0.0) if exact else 0.5
    )
    assert (result.energy_standard_error is None) is exact


def test_exact_vqe_rejects_disabled_statevector_before_optimization():
    solver = make_solver()
    with patch.object(solver, "_optimize_parameters") as optimize:
        with pytest.raises(ValueError, match="requires need_statevector=True"):
            solver.run(
                hamiltonian=Hamiltonian("Z"),
                device_name="QpiAI-QSV-Simulator",
                need_statevector=False,
            )
    optimize.assert_not_called()


def test_qdm_rejects_explicit_statevector_without_submitting_job():
    solver = make_solver()
    with (
        patch.object(solver, "_optimize_parameters", side_effect=evaluate_once),
        patch("qpiai_quantum.jobmanager.ExecutionEngine.execute_circuit") as execute,
    ):
        with pytest.raises(ValueError, match="QDM simulators return density matrices"):
            solver.run(
                hamiltonian=Hamiltonian("Z"),
                device_name="QpiAI-QDM-Simulator",
                need_statevector=True,
            )
    execute.assert_not_called()
