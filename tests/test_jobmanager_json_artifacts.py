import json
from typing import Any
from types import SimpleNamespace

import numpy as np
import pytest

from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver
from qpiai_quantum.jobmanager.exceptions import InvalidResponseError
from qpiai_quantum.jobmanager.jobmanager import JobManager
from qpiai_quantum.jobmanager.result_contract import ContractValidationError
from qpiai_quantum.jobmanager.result_decoding import (
    decode_density_matrix,
    decode_statevector,
)
from qpiai_quantum.jobmanager.sse_handler import SSEResultHandler


def _complex_pair(value: complex) -> dict[str, float]:
    return {"real": value.real, "imag": value.imag}


def _json_simulator_response() -> dict:
    statevector = [2**-0.5 + 0.0j, 0.0j, 0.0j, 2**-0.5 + 0.0j]
    density_matrix = np.outer(statevector, np.conjugate(statevector))
    return {
        "results": {
            "data": {
                "counts": {"00": 512, "11": 512},
                "statevector": [_complex_pair(value) for value in statevector],
                "density_matrix": [
                    [_complex_pair(value) for value in row] for row in density_matrix
                ],
            }
        },
        "execution_time": 0.125,
    }


def test_parse_job_results_decodes_json_complex_artifacts() -> None:
    response = _json_simulator_response()
    job_payload = {
        "id": "job-123",
        "status": "completed",
        "server_response": json.dumps(response),
    }

    result = JobManager()._parse_job_results(
        job_payload,
        circuit_name="bell-state",
        shots=1024,
        device_name="QpiAI-QSV-Simulator",
        need_statevector=True,
        need_density_matrix=True,
    )

    expected_statevector = np.asarray([2**-0.5 + 0.0j, 0.0j, 0.0j, 2**-0.5 + 0.0j])
    np.testing.assert_allclose(
        np.asarray(result.statevector).reshape(-1), expected_statevector
    )
    np.testing.assert_allclose(
        np.asarray(result.density_matrix),
        np.outer(expected_statevector, np.conjugate(expected_statevector)),
    )


def test_cloud_json_statevector_can_be_used_by_vqe() -> None:
    response = _json_simulator_response()
    parsed = JobManager()._parse_simulator_response(response)
    solver = VQESolver(n_qubits=2)

    energy = solver._compute_statevector_expectation(
        parsed["statevector"],
        [([(0, "Z"), (1, "Z")], 1.0)],
    )

    np.testing.assert_allclose(energy, 1.0)


def test_sse_parser_uses_the_same_statevector_decoding() -> None:
    parsed = SSEResultHandler(
        device_name="QpiAI-QSV-Simulator"
    )._parse_simulator_response(_json_simulator_response())

    expected = np.asarray([2**-0.5, 0.0, 0.0, 2**-0.5])
    np.testing.assert_allclose(np.asarray(parsed["statevector"]).reshape(-1), expected)


def test_numeric_artifacts_decode_to_canonical_shapes() -> None:
    statevector = [[1.0 + 0.0j], [0.0 + 0.0j]]
    density_matrix = [[1.0 + 0.0j, 0.0j], [0.0j, 0.0j]]
    parsed = JobManager()._parse_simulator_response(
        {
            "statevector": statevector,
            "density_matrix": density_matrix,
        }
    )

    assert parsed["statevector"] == statevector
    assert parsed["density_matrix"] == density_matrix


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-list",
        [{"real": 1.0}],
        [{"real": "invalid", "imag": 0.0}],
        [[1.0, 0.0]],
    ],
)
def test_statevector_decoder_rejects_malformed_payloads(payload: object) -> None:
    with pytest.raises(InvalidResponseError) as exc_info:
        decode_statevector(payload)

    assert isinstance(exc_info.value.__cause__, ContractValidationError)


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-matrix",
        [1.0, 0.0],
        [[{"real": 1.0}]],
        [[1.0, 0.0], [0.0]],
    ],
)
def test_density_matrix_decoder_rejects_malformed_payloads(payload: object) -> None:
    with pytest.raises(InvalidResponseError) as exc_info:
        decode_density_matrix(payload)

    assert isinstance(exc_info.value.__cause__, ContractValidationError)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_decoders_reject_invalid_amplitudes(value):
    with pytest.raises(InvalidResponseError):
        decode_statevector([value])
    with pytest.raises(InvalidResponseError):
        decode_density_matrix([[value]])


def test_density_matrix_decoder_rejects_rectangular_matrix():
    with pytest.raises(InvalidResponseError, match="square"):
        decode_density_matrix([[1.0, 0.0]])


@pytest.mark.parametrize(
    "envelope",
    ["top", "data", "results_data", "legacy", "list", "histogram", "nested_histogram"],
)
def test_get_job_results_preserves_counts_and_nested_artifacts(
    monkeypatch: pytest.MonkeyPatch, envelope: str
) -> None:
    counts = {"001": 7, "110": 25}
    vector = np.zeros(8, dtype=complex)
    vector[1] = (7 / 32) ** 0.5
    vector[6] = (25 / 32) ** 0.5
    density = np.outer(vector, vector.conjugate())
    data: dict[str, Any] = {
        "counts": counts,
        "statevector": [_complex_pair(value) for value in vector],
        "density_matrix": [[_complex_pair(value) for value in row] for row in density],
        "probabilities": {"001": 7 / 32, "110": 25 / 32},
    }
    payload = data
    if envelope == "data":
        payload = {"data": data}
    elif envelope == "results_data":
        payload = {"results": {"data": data}}
    elif envelope == "legacy":
        payload = {"results": {"results": data}}
    elif envelope == "list":
        payload = {"results": [{"data": data}]}
    elif envelope == "histogram":
        data["histogram"] = data.pop("counts")

    elif envelope == "nested_histogram":
        data["histogram"] = {
            key: str(value) for key, value in data.pop("counts").items()
        }
        payload = {"data": data}

    manager = JobManager()
    job = {
        "id": "stored-json-job",
        "status": "Success",
        "method": "tensor_network",
        "server_response": json.dumps(payload),
    }
    monkeypatch.setattr(manager, "get_job_status", lambda job_id: job)
    monkeypatch.setattr(manager, "_get_auth_header", lambda: {})
    monkeypatch.setattr(
        "requests.get",
        lambda *args, **kwargs: SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: job
        ),
    )
    result = manager.get_job_results("stored-json-job")
    assert result.counts == counts
    assert result.shots == 32
    assert result.probabilities == {"001": 7 / 32, "110": 25 / 32}
    np.testing.assert_allclose(np.asarray(result.statevector).reshape(-1), vector)
    np.testing.assert_allclose(np.asarray(result.density_matrix), density)
