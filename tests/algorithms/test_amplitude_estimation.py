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


def test_canonical_qae_not_implemented():
    circuit = Circuit(1)
    problem = EstimationProblem(state_preparation=circuit, objective_qubits=[0])
    qae = AmplitudeEstimation(num_evaluation_qubits=3)

    with pytest.raises(NotImplementedError):
        qae.estimate(problem)


@pytest.mark.skipif(
    os.environ.get("RUN_ALGO_CORRECTNESS") != "1",
    reason="Skipping correctness test. Set RUN_ALGO_CORRECTNESS=1 to run.",
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
