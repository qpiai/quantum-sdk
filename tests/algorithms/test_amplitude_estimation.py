import os
import sys
import pytest
from dotenv import load_dotenv

load_dotenv("qcloud.env")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.circuit import Circuit
from qpiai_quantum.algorithms.amplitude_estimation import (
    EstimationProblem,
    IterativeAmplitudeEstimation,
    AmplitudeEstimation,
)
from qpiai_quantum.icr.circuitoperation import RXGate, MeasureOperation


def test_estimation_problem():
    circuit = Circuit(1)
    circuit.ry(0, 0.5)
    problem = EstimationProblem(state_preparation=circuit, objective_qubits=[0])

    assert problem.num_qubits == 1
    assert problem.objective_qubits == [0]


from unittest.mock import MagicMock, patch
import math


@patch("qpiai_quantum.circuit.Circuit.run")
def test_iterative_amplitude_estimation(mock_run):
    theta = 0.8
    expected_prob = math.sin(theta / 2) ** 2
    shots = 2000

    def make_mock_result(counts):
        mock = MagicMock()
        mock.get.return_value = {"counts": counts}
        return mock

    # Generate ideal mock counts for k_schedule = [0, 1, 2, 4, 8]
    probs = [math.sin((2 * k + 1) * theta / 2) ** 2 for k in [0, 1, 2, 4, 8]]
    mock_run.side_effect = [
        make_mock_result({"0": int(shots * (1 - p)), "1": int(shots * p)})
        for p in probs
    ]

    circuit = Circuit(1)
    circuit.ry(0, theta)
    problem = EstimationProblem(state_preparation=circuit, objective_qubits=[0])

    iae = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05)
    estimated_prob = iae.estimate(problem, shots=shots)

    # Allow 0.05 absolute tolerance
    assert abs(estimated_prob - expected_prob) < 0.05


def test_canonical_qae_build_circuit():
    circuit = Circuit(1)
    circuit.ry(0, 0.6)
    problem = EstimationProblem(state_preparation=circuit, objective_qubits=[0])
    qae = AmplitudeEstimation(num_evaluation_qubits=3)

    built_circuit = qae.build_circuit(problem)
    # Total qubits = 3 evaluation + 1 state qubit = 4
    assert built_circuit.num_qubits == 4
    assert built_circuit.num_clbits == 3


def test_canonical_qae_estimation_local():
    theta = 0.8
    expected_prob = math.sin(theta / 2) ** 2

    circuit = Circuit(1)
    circuit.ry(0, theta)
    problem = EstimationProblem(state_preparation=circuit, objective_qubits=[0])

    qae = AmplitudeEstimation(num_evaluation_qubits=6)
    estimated_prob = qae.estimate(problem, shots=1000, device_name="QpiAI-QSV-Local")

    # Allow 0.05 tolerance with 6 evaluation qubits
    assert abs(estimated_prob - expected_prob) < 0.05


@pytest.mark.skipif(
    os.environ.get("RUN_ALGO_CORRECTNESS") != "1" or not os.getenv("API_KEY"),
    reason="Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 and API_KEY in environment to run.",
)
def test_amplitude_estimation_correctness():
    import uuid

    api_key = os.getenv("API_KEY")
    if api_key:
        from qpiai_quantum.authentication.auth import QpiAIQuantumAuth

        try:
            QpiAIQuantumAuth.login(api_key)
        except Exception:
            pass
    theta = 0.8
    expected_prob = math.sin(theta / 2) ** 2
    circuit = Circuit(1, name=f"test_qae_{uuid.uuid4().hex[:8]}")
    circuit.ry(0, theta)
    problem = EstimationProblem(state_preparation=circuit, objective_qubits=[0])
    iae = IterativeAmplitudeEstimation(epsilon_target=0.08, alpha=0.05)
    estimated_prob = iae.estimate(problem, shots=200)
    assert abs(estimated_prob - expected_prob) < 0.15


# Regression coverage for the canonical QAE controlled-power ordering.
# Evaluation qubit j must receive Q^(2^(m-1-j)) to match the inverse-QFT
# convention used by QuantumPhaseEstimation.  With the reversed assignment
# theta=0.8 still happens to pass, so the sweep below is what actually pins
# the convention down: theta=0.2 returns ~0.378 instead of ~0.010.
@pytest.mark.parametrize("theta", [0.2, 0.5, 0.8, 1.5, 2.2, 2.8])
def test_canonical_qae_amplitude_sweep(theta):
    expected_prob = math.sin(theta / 2) ** 2

    circuit = Circuit(1)
    circuit.ry(0, theta)
    problem = EstimationProblem(state_preparation=circuit, objective_qubits=[0])

    qae = AmplitudeEstimation(num_evaluation_qubits=6)
    estimated_prob = qae.estimate(problem, shots=4000, device_name="QpiAI-QSV-Local")

    assert abs(estimated_prob - expected_prob) < 0.05


@pytest.mark.parametrize("theta,expected_prob", [(0.0, 0.0), (math.pi, 1.0)])
def test_canonical_qae_boundary_amplitudes(theta, expected_prob):
    circuit = Circuit(1)
    circuit.ry(0, theta)
    problem = EstimationProblem(state_preparation=circuit, objective_qubits=[0])

    qae = AmplitudeEstimation(num_evaluation_qubits=6)
    estimated_prob = qae.estimate(problem, shots=4000, device_name="QpiAI-QSV-Local")

    assert abs(estimated_prob - expected_prob) < 0.05


def _two_qubit_problem(is_good_state):
    """RY(0.8) on qubit 0, RY(1.1) on qubit 1, with a custom predicate."""
    circuit = Circuit(2)
    circuit.ry(0, 0.8)
    circuit.ry(1, 1.1)
    return EstimationProblem(
        state_preparation=circuit,
        objective_qubits=[0],
        is_good_state=is_good_state,
    )


def test_canonical_qae_honours_custom_good_state():
    # Counts layout is MSB first, so bitstring[0] is qubit 1 and bitstring[1]
    # is qubit 0.  "exactly one qubit is 1" cannot be expressed as the default
    # all-objective-qubits-are-1 marking, so this fails outright if
    # is_good_state is ignored (it would estimate P(qubit 0 = 1) = 0.1516).
    p0 = math.sin(0.8 / 2) ** 2
    p1 = math.sin(1.1 / 2) ** 2
    expected_prob = p0 * (1 - p1) + (1 - p0) * p1

    problem = _two_qubit_problem(lambda bitstring: bitstring.count("1") == 1)
    qae = AmplitudeEstimation(num_evaluation_qubits=6)
    estimated_prob = qae.estimate(problem, shots=4000, device_name="QpiAI-QSV-Local")

    assert abs(estimated_prob - expected_prob) < 0.05


def test_canonical_qae_custom_good_state_marking_zeros():
    p0 = math.sin(0.8 / 2) ** 2
    p1 = math.sin(1.1 / 2) ** 2
    expected_prob = (1 - p0) * (1 - p1)

    problem = _two_qubit_problem(lambda bitstring: bitstring == "00")
    qae = AmplitudeEstimation(num_evaluation_qubits=6)
    estimated_prob = qae.estimate(problem, shots=4000, device_name="QpiAI-QSV-Local")

    assert abs(estimated_prob - expected_prob) < 0.05


def test_resolve_good_states_uses_compact_marking_by_default():
    qae = AmplitudeEstimation(num_evaluation_qubits=3)
    circuit = Circuit(2)
    circuit.ry(0, 0.8)
    circuit.ry(1, 1.1)

    default = EstimationProblem(state_preparation=circuit, objective_qubits=[0, 1])
    assert qae._resolve_good_states(default) is None

    # A custom predicate that happens to match the default must not pay for an
    # explicitly synthesised oracle either.
    equivalent = EstimationProblem(
        state_preparation=circuit,
        objective_qubits=[0],
        is_good_state=lambda bitstring: bitstring[-1] == "1",
    )
    assert qae._resolve_good_states(equivalent) is None


def test_resolve_good_states_enumerates_custom_predicate():
    qae = AmplitudeEstimation(num_evaluation_qubits=3)
    circuit = Circuit(2)

    problem = EstimationProblem(
        state_preparation=circuit,
        objective_qubits=[0],
        is_good_state=lambda bitstring: bitstring == "10",
    )
    # "10" is qubit 1 = 1 and qubit 0 = 0, i.e. integer 0b10.
    assert qae._resolve_good_states(problem) == [0b10]


def test_canonical_qae_rejects_oversized_state_register():
    class _OversizedStatePreparation:
        num_qubits = AmplitudeEstimation.MAX_ORACLE_QUBITS + 1

    problem = EstimationProblem(
        state_preparation=_OversizedStatePreparation(), objective_qubits=[0]
    )
    qae = AmplitudeEstimation(num_evaluation_qubits=3)

    with pytest.raises(ValueError, match="at most"):
        qae._resolve_good_states(problem)


@pytest.mark.parametrize(
    "is_good_state,expected_prob",
    [
        (lambda bitstring: False, 0.0),
        (lambda bitstring: True, 1.0),
    ],
)
def test_canonical_qae_degenerate_good_state(is_good_state, expected_prob):
    """A predicate marking no states or every state must still be exact."""
    problem = _two_qubit_problem(is_good_state)
    qae = AmplitudeEstimation(num_evaluation_qubits=6)
    estimated_prob = qae.estimate(problem, shots=4000, device_name="QpiAI-QSV-Local")

    assert abs(estimated_prob - expected_prob) < 0.05
