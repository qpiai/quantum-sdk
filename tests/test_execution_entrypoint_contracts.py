"""Offline public-boundary regressions for request and stored-result handling."""

import inspect
from importlib.machinery import PathFinder
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import qpiai_quantum
import qpiai_quantum.jobmanager as jobmanager_module
from qpiai_quantum.algorithms.base import QuantumAlgorithm
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager import DeviceRegistry, ExecutionEngine, JobManager
from qpiai_quantum.jobmanager.job_result import JobResult
from qpiai_quantum.state_preparation import (
    BellStateGenerator,
    GHZStateGenerator,
    ClusterStateGenerator,
)
from qpiai_quantum.jobmanager.backend import resolve_execution_request
from qpiai_quantum.jobmanager.exceptions import InvalidResponseError
from qpiai_quantum.jobmanager.result_decoding import decode_counts
from qpiai_quantum.jobmanager.sse_handler import SSEResultHandler


def _payload(envelope: str) -> dict:
    vector = np.zeros(8)
    vector[1], vector[6] = ((7 / 32) ** 0.5, (25 / 32) ** 0.5)
    data = {
        "counts": {"001": "7", "110": 25},
        "statevector": vector.reshape(-1, 1).tolist(),
        "density_matrix": np.outer(vector, vector).tolist(),
        "probabilities": {"001": 7 / 32, "110": 25 / 32},
    }
    if envelope == "histogram":
        data["histogram"] = data.pop("counts")
    if envelope in {"top", "histogram"}:
        response = data
    elif envelope == "data":
        response = {"data": data}
    elif envelope == "list":
        response = {"results": [{"data": data}]}
    elif envelope == "legacy":
        response = {"results": {"results": data}}
    else:
        response = {"results": {"data": data}}
    return {
        **response,
        "execution_time": "1e-3 seconds",
        "metadata": {"calibration": "preserved"},
    }


@pytest.mark.parametrize(
    "binary,wrapper",
    [
        (False, "nested"),
        (False, "flat"),
        (False, "string"),
        (True, "nested"),
        (True, "flat"),
    ],
)
def test_direct_stored_artifact_download_receives_job_context(
    binary: bool, wrapper: str
) -> None:
    manager = JobManager()
    vector = np.asarray([1.0, 0.0], dtype="<c16")
    density = np.outer(vector, vector.conjugate())
    body = vector.tobytes() + density.tobytes()
    pointer: dict[str, Any] = {
        "large_file": True,
        "bucket": "results",
        "key": "job.bin" if binary else "job.json",
        "format": "qpiai_binary_artifacts",
        "schema_version": 1,
        "size_bytes": len(body),
        "counts": {"0": 8},
        "execution_time": 0.25,
        "artifacts": [
            {
                "result_type": "statevector",
                "dtype": "complex128",
                "byte_order": "little",
                "shape": [2],
                "offset_bytes": 0,
                "size_bytes": vector.nbytes,
            },
            {
                "result_type": "density_matrix",
                "dtype": "complex128",
                "byte_order": "little",
                "shape": [2, 2],
                "offset_bytes": vector.nbytes,
                "size_bytes": density.nbytes,
            },
        ],
    }
    response: dict[str, Any] | str = (
        pointer if wrapper == "flat" else {"results": pointer}
    )
    if wrapper == "string":
        response = "s3://results/" + pointer["key"]
    record = {"status": "Success", "server_response": response}

    def download(
        s3_path: str, manifest: dict | None, job_id: str, result_type: str | None = None
    ) -> dict:
        assert job_id == "stored-job"
        assert s3_path == "s3://results/" + pointer["key"]
        if binary:
            assert manifest is not None
            return manager._decode_binary_artifact_payload(body, manifest)
        return {"data": {"counts": {"0": "8"}}, "execution_time": "2.5e-1 seconds"}

    with (
        patch.object(manager, "get_job_status", return_value=record),
        patch.object(
            manager, "_download_s3_result_via_backend", side_effect=download
        ) as fetch,
    ):
        result = manager.get_job_results("stored-job")
    fetch.assert_called_once()
    assert result.counts == {"0": 8}
    assert result.execution_time == pytest.approx(0.25)
    assert result.job_id == "stored-job"
    if binary:
        np.testing.assert_allclose(np.asarray(result.statevector).reshape(-1), vector)
        np.testing.assert_allclose(np.asarray(result.density_matrix), density)


@pytest.mark.parametrize(
    "value", [{"0": -1}, {"0": 1.5}, {"0": True}, {"0": "bad"}, "bad"]
)
def test_invalid_counts_are_rejected(value: object) -> None:
    manager = JobManager()
    with patch.object(
        manager,
        "get_job_status",
        return_value={"status": "Success", "server_response": {"counts": value}},
    ):
        with pytest.raises(
            InvalidResponseError, match="invalid measurement counts"
        ) as error:
            manager.get_job_results("invalid-count-job")
    assert "invalid measurement counts" in str(error.value)


def test_counts_accept_integer_numbers_and_decimal_strings() -> None:
    assert decode_counts({"counts": {"00": 0, "01": 5.0, "10": "+7"}}) == {
        "00": 0,
        "01": 5,
        "10": 7,
    }


@pytest.mark.parametrize("seconds", [0, 1.25, "1.25", "1.25e-3 seconds"])
def test_stored_seconds_field_is_normalized(seconds: str | float) -> None:
    result = JobManager()._parse_job_results(
        {"server_response": {"counts": {"0": 1}}, "execution_time": seconds},
        "circuit",
        1,
    )
    expected = 0.00125 if seconds == "1.25e-3 seconds" else float(seconds)
    assert result.execution_time == pytest.approx(expected)


def test_invalid_response_time_falls_back_to_job_seconds() -> None:
    result = JobManager()._parse_job_results(
        {
            "server_response": {"counts": {"0": 1}, "execution_time": "nan"},
            "execution_time": "1e-3",
            "execution_time_ms": 9000,
        },
        "circuit",
        1,
    )
    assert result.execution_time == pytest.approx(0.001)


def test_resource_clear_does_not_wait_for_fetch_or_recache_its_result() -> None:
    started, release = (Event(), Event())
    context = (("X-Secret-Token", "cache-account"), ("server_url", "cache-server"))

    def fetch(_context):
        started.set()
        assert release.wait(5)
        return [{"ID": "old-id", "device_name": "Chip", "method": "qpu"}]

    JobManager.clear_resource_cache()
    try:
        with (
            patch.object(DeviceRegistry, "_current_cache_key", return_value=context),
            patch.object(DeviceRegistry, "_fetch", side_effect=fetch),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            pending = pool.submit(JobManager()._get_compute_resource_id, "qpu", "Chip")
            try:
                assert started.wait(5)
                JobManager.clear_resource_cache()
            finally:
                release.set()
            assert pending.result(5) == "old-id"
        assert JobManager._resource_cache == {}
    finally:
        JobManager.clear_resource_cache()


def test_removed_backend_api_fails_before_authentication():
    assert not hasattr(qpiai_quantum, "Backend")
    assert not hasattr(jobmanager_module, "Backend")
    with pytest.raises(TypeError):
        ExecutionEngine.execute_circuit(Circuit(1), backend=object())
    with pytest.raises(TypeError):
        ExecutionEngine.execute_circuit(Circuit(1), 32, object())
    with patch.object(DeviceRegistry, "_fetch", side_effect=AssertionError("network")):
        with pytest.raises(TypeError, match="device_name"):
            Circuit(1).run(backend=object())


def test_direct_output_defaults_remain_false(backend_devices):
    for fn in (
        ExecutionEngine.execute_circuit,
        JobManager.submit_circuit_job,
        JobManager.submit_qasm_job,
        JobManager.submit_and_wait_for_results_qasm,
    ):
        for flag in ("need_statevector", "need_density_matrix"):
            assert inspect.signature(fn).parameters[flag].default is False
    explicit = resolve_execution_request(
        "QpiAI-QDM-Simulator", need_density_matrix=False
    )
    omitted = resolve_execution_request("QpiAI-QDM-Simulator")
    assert not explicit.need_density_matrix
    assert omitted.need_density_matrix


def test_shared_request_resolver_rejects_unsupported_direct_outputs(backend_devices):
    with pytest.raises(ValueError, match="cannot return a statevector"):
        JobManager().submit_qasm_job(
            "OPENQASM 2.0;", device_name="QpiAI-Indus-1", need_statevector=True
        )


@pytest.mark.parametrize("kind", ["bell", "ghz", "cluster"])
def test_state_comparison_helpers_forward_device_names(kind: str):
    counts = {"00": 4, "11": 4} if kind == "bell" else {"000": 4, "111": 4}
    result = JobResult(counts=counts, shots=8)
    with patch.object(QuantumAlgorithm, "run", return_value=result) as execute:
        if kind == "bell":
            BellStateGenerator.compare_all_bell_states(
                shots=8, device_name="QpiAI-Indus-1"
            )
        elif kind == "ghz":
            GHZStateGenerator.compare_different_sizes(
                [3], shots=8, device_name="QpiAI-Indus-1"
            )
        else:
            ClusterStateGenerator.compare_different_sizes(
                [3], shots=8, device_name="QpiAI-Indus-1"
            )
    assert execute.call_count > 0
    for call in execute.call_args_list:
        assert call.kwargs == {"shots": 8, "device_name": "QpiAI-Indus-1"}


def test_obsolete_credit_api_is_removed():
    assert not hasattr(QuantumAlgorithm, "estimate_credits")
    assert (
        PathFinder.find_spec("qpiai_quantum.creditsmanager", qpiai_quantum.__path__)
        is None
    )


def test_legacy_millisecond_field_is_only_a_fallback():
    manager = JobManager()
    record = {"server_response": {"counts": {"0": 1}}, "execution_time_ms": "1250"}
    assert manager._parse_job_results(record, "circuit", 1).execution_time == 1.25
    record["execution_time"] = 0.25
    assert manager._parse_job_results(record, "circuit", 1).execution_time == 0.25


def test_sse_preserves_stored_pointer_without_fetching():
    pointer = "s3://results/completed-job.json"
    source = MagicMock()
    source.__enter__.return_value = [
        SimpleNamespace(
            data=json.dumps(
                {"status": "Success", "job_id": "completed-job", "response": pointer}
            )
        )
    ]
    handler = SSEResultHandler()
    with (
        patch("qpiai_quantum.jobmanager.sse_handler.SSE_AVAILABLE", True),
        patch("qpiai_quantum.jobmanager.sse_handler.EventSource", return_value=source),
        patch("requests.get", side_effect=AssertionError("unexpected fetch")),
    ):
        assert handler.wait_for_events("https://offline.invalid", {})
    assert handler.get_result()["response_data"] == pointer
    assert handler.get_result()["job_id"] == "completed-job"


def test_profile_defaults_cannot_enable_unsupported_results():
    DeviceRegistry.load(
        [
            {
                "ID": "invalid-defaults",
                "device_name": "Misconfigured simulator",
                "method": "statevector",
                "supported_results": [],
                "default_results": ["statevector"],
            }
        ]
    )
    try:
        with pytest.raises(ValueError, match="cannot return a statevector"):
            resolve_execution_request("Misconfigured simulator")
    finally:
        DeviceRegistry.clear()


@pytest.mark.parametrize("replacement", ["clear", "load", "refresh"])
def test_device_fetch_cannot_overwrite_later_cache_changes(replacement: str) -> None:
    """An older request may finish, but cannot republish superseded profiles."""
    key = (("server_url", "offline"),)
    started, release = (Event(), Event())
    old = [{"device_name": "Old", "method": "qpu"}]
    new = [{"device_name": "New", "method": "qpu"}]

    def fetch(_key: object) -> list[dict]:
        started.set()
        assert release.wait(5)
        return old

    JobManager.clear_resource_cache()
    try:
        with (
            patch.object(DeviceRegistry, "_current_cache_key", return_value=key),
            patch.object(DeviceRegistry, "_fetch", side_effect=fetch),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            pending = pool.submit(DeviceRegistry._get_devices)
            try:
                assert started.wait(5)
                if replacement == "clear":
                    JobManager.clear_resource_cache()
                elif replacement == "load":
                    DeviceRegistry.load(new)
                else:
                    with patch.object(DeviceRegistry, "_fetch", return_value=new):
                        DeviceRegistry.refresh()
            finally:
                release.set()
            assert "Old" in pending.result(5)
            if replacement == "clear":
                assert key not in DeviceRegistry._entries
            else:
                assert "New" in DeviceRegistry._entries[key][1]
                assert "Old" not in DeviceRegistry._entries[key][1]
    finally:
        JobManager.clear_resource_cache()


def test_device_fetches_for_different_accounts_remain_concurrent() -> None:
    started, release = (Event(), Event())
    first_key = (("server_url", "first"),)
    second_key = (("server_url", "second"),)

    def fetch(key: object) -> list[dict]:
        if key == first_key:
            started.set()
            assert release.wait(5)
        return [{"device_name": "Chip", "method": "qpu"}]

    DeviceRegistry.clear()
    try:
        with (
            patch.object(DeviceRegistry, "_fetch", side_effect=fetch),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            first = pool.submit(DeviceRegistry._get_devices, cache_key=first_key)
            try:
                assert started.wait(5)
                second = pool.submit(DeviceRegistry._get_devices, cache_key=second_key)
                assert "Chip" in second.result(5)
            finally:
                release.set()
            assert "Chip" in first.result(5)
        assert set(DeviceRegistry._entries) == {first_key, second_key}
    finally:
        DeviceRegistry.clear()


def _read_result_record(
    manager: JobManager, record: dict, entrypoint: str
) -> JobResult:
    """Exercise conversion with submission and transport mocked below their boundaries."""
    with (
        patch.object(manager, "get_job_status", return_value=record),
        patch.object(
            manager, "submit_qasm_job", return_value={"job_id": "payload-job"}
        ),
        patch.object(manager, "_wait_with_polling", return_value={"job_info": record}),
    ):
        if entrypoint == "direct":
            return manager.get_job_results("payload-job")
        return manager.submit_and_wait_for_results_qasm(
            "OPENQASM 2.0;", use_events=False
        )


@pytest.mark.parametrize("entrypoint", ["direct", "wait"])
@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"response_data": None},
        {"response_data": ""},
        {"response_data": " ", "server_response": "\t"},
    ],
)
def test_completed_job_requires_payload(
    backend_devices, entrypoint: str, fields: dict
) -> None:
    record = {"status": "completed", **fields}
    with pytest.raises(InvalidResponseError, match="no results data") as error:
        _read_result_record(JobManager(), record, entrypoint)
    assert "payload-job" in str(error.value)


@pytest.mark.parametrize("entrypoint", ["direct", "wait"])
@pytest.mark.parametrize("empty", [None, "", " \t\n"])
def test_blank_primary_payload_uses_legacy_result(
    backend_devices, entrypoint: str, empty: object
) -> None:
    record = {
        "status": "completed",
        "response_data": empty,
        "server_response": {"counts": {"0": 8}},
    }
    assert _read_result_record(JobManager(), record, entrypoint).counts == {"0": 8}


@pytest.mark.parametrize("entrypoint", ["direct", "wait"])
@pytest.mark.parametrize("malformed", ["broken", "null", 0, False, []])
def test_nonempty_or_zero_primary_payload_is_not_hidden(
    backend_devices, entrypoint: str, malformed: object
) -> None:
    record = {
        "status": "completed",
        "response_data": malformed,
        "server_response": {"counts": {"0": 8}},
    }
    with pytest.raises(InvalidResponseError, match="unsupported format"):
        _read_result_record(JobManager(), record, entrypoint)


@pytest.mark.parametrize("entrypoint", ["direct", "wait"])
@pytest.mark.parametrize(
    "payload", [{"counts": {}}, {"counts": {"0": 0}}, {"statevector": [1, 0]}]
)
def test_valid_partial_results_preserve_primary_payload(
    backend_devices, entrypoint: str, payload: dict
) -> None:
    record = {
        "status": "completed",
        "response_data": payload,
        "server_response": {"counts": {"1": 8}},
    }
    result = _read_result_record(JobManager(), record, entrypoint)
    assert result.counts == payload.get("counts")
    if "statevector" in payload:
        np.testing.assert_array_equal(
            np.asarray(result.statevector).reshape(-1), [1, 0]
        )
