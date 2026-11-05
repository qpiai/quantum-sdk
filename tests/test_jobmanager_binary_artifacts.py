from unittest.mock import patch

import numpy as np
import pytest

from qpiai_quantum.jobmanager.exceptions import InvalidResponseError
from qpiai_quantum.jobmanager.jobmanager import JobManager
from qpiai_quantum.jobmanager.result_contract import ContractValidationError


def _dual_artifact_pointer() -> tuple[dict, bytes, np.ndarray, np.ndarray]:
    statevector = np.asarray([2**-0.5, 0.0, 0.0, 2**-0.5], dtype=np.dtype("<c16"))
    density_matrix = np.asarray(
        np.outer(statevector, np.conjugate(statevector)), dtype=np.dtype("<c16")
    )
    statevector_bytes = statevector.tobytes(order="C")
    density_matrix_bytes = density_matrix.tobytes(order="C")
    body = statevector_bytes + density_matrix_bytes
    pointer = {
        "large_file": True,
        "storage": "s3",
        "bucket": "test-bucket",
        "key": "simulator-results/job-123.bin",
        "size_bytes": len(body),
        "content_type": "application/x-qpiai-binary",
        "sha256": "unused-in-sdk-decoder-test",
        "format": "qpiai_binary_artifacts",
        "schema_version": 1,
        "counts": {"00": 512, "11": 512},
        "execution_time": 0.125,
        "artifacts": [
            {
                "result_type": "statevector",
                "dtype": "complex128",
                "byte_order": "little",
                "shape": [4],
                "offset_bytes": 0,
                "size_bytes": len(statevector_bytes),
            },
            {
                "result_type": "density_matrix",
                "dtype": "complex128",
                "byte_order": "little",
                "shape": [4, 4],
                "offset_bytes": len(statevector_bytes),
                "size_bytes": len(density_matrix_bytes),
            },
        ],
    }
    return pointer, body, statevector, density_matrix


@patch("requests.get")
@patch("requests.post")
@patch.object(JobManager, "_get_auth_header", return_value={"X-Secret-Token": "key"})
def test_resolve_new_binary_pointer_decodes_all_artifacts(
    _mock_auth, mock_post, mock_get
):
    pointer, body, expected_statevector, expected_density_matrix = (
        _dual_artifact_pointer()
    )
    mock_post.return_value.json.return_value = {
        "presigned_url": "https://s3.example/result.bin"
    }
    mock_get.return_value.content = body
    mock_get.return_value.headers = {"Content-Type": "application/x-qpiai-binary"}

    resolved = JobManager()._resolve_job_response_data(pointer)

    assert resolved["counts"] == pointer["counts"]
    assert resolved["execution_time"] == pointer["execution_time"]
    np.testing.assert_allclose(
        np.asarray(resolved["statevector"]).reshape(-1), expected_statevector
    )
    np.testing.assert_allclose(
        np.asarray(resolved["density_matrix"]), expected_density_matrix
    )
    assert resolved["results"]["data"]["counts"] == pointer["counts"]
    assert "statevector" in resolved["results"]["data"]
    assert "density_matrix" in resolved["results"]["data"]


@patch("requests.get")
@patch("requests.post")
@patch.object(JobManager, "_get_auth_header", return_value={"X-Secret-Token": "key"})
def test_resolve_nested_worker_pointer_keeps_manifest(_mock_auth, mock_post, mock_get):
    pointer, body, expected_statevector, expected_density_matrix = (
        _dual_artifact_pointer()
    )
    mock_post.return_value.json.return_value = {
        "presigned_url": "https://s3.example/result.bin"
    }
    mock_get.return_value.content = body
    mock_get.return_value.headers = {"Content-Type": "application/x-qpiai-binary"}

    resolved = JobManager()._resolve_job_response_data(
        {"results": pointer}, job_id="job-123"
    )

    assert resolved["counts"] == pointer["counts"]
    np.testing.assert_allclose(
        np.asarray(resolved["statevector"]).reshape(-1), expected_statevector
    )
    np.testing.assert_allclose(
        np.asarray(resolved["density_matrix"]), expected_density_matrix
    )
    assert mock_post.call_args.kwargs["json"] == {
        "s3_path": "s3://test-bucket/simulator-results/job-123.bin",
        "job_id": "job-123",
    }


@patch("requests.get")
@patch("requests.post")
@patch.object(JobManager, "_get_auth_header", return_value={"X-Secret-Token": "key"})
def test_resolve_binary_pointer_rejects_legacy_result_type_format(
    _mock_auth, mock_post, mock_get
):
    mock_post.return_value.json.return_value = {
        "presigned_url": "https://s3.example/result.bin"
    }
    mock_get.return_value.content = np.asarray([1.0, 0.0], dtype="<c16").tobytes()
    mock_get.return_value.headers = {"Content-Type": "application/octet-stream"}
    legacy_pointer = {
        "large_file": True,
        "bucket": "test-bucket",
        "key": "simulator-results/job-legacy.bin",
        "result_type": "statevector",
        "counts": {"0": 1024},
        "execution_time": 0.1,
    }

    with pytest.raises(InvalidResponseError, match="Unsupported binary result format"):
        JobManager()._resolve_job_response_data(legacy_pointer)


@patch("requests.get")
@patch("requests.post")
@patch.object(JobManager, "_get_auth_header", return_value={"X-Secret-Token": "key"})
def test_binary_download_requires_manifest(_mock_auth, mock_post, mock_get):
    mock_post.return_value.json.return_value = {
        "presigned_url": "https://s3.example/result.bin"
    }
    mock_get.return_value.headers = {"Content-Type": "application/octet-stream"}
    mock_get.return_value.content = np.asarray([1.0, 0.0], dtype="<c16").tobytes()

    with pytest.raises(InvalidResponseError, match="missing its artifact manifest"):
        JobManager()._resolve_job_response_data("s3://test-bucket/result.bin")


def test_binary_manifest_rejects_noncontiguous_artifact_ranges():
    pointer, body, _, _ = _dual_artifact_pointer()
    pointer["artifacts"][1]["offset_bytes"] += 16

    with pytest.raises(
        InvalidResponseError,
    ) as exc_info:
        JobManager._decode_binary_artifact_payload(body, pointer)

    assert isinstance(exc_info.value.__cause__, ContractValidationError)
    assert "contiguous" in str(exc_info.value.__cause__)


@pytest.mark.parametrize(
    "change", ["truncated", "trailing", "shape", "duplicate", "dtype", "byte_order"]
)
def test_binary_manifest_rejects_corrupted_payload_or_descriptors(change):
    pointer, body, _, _ = _dual_artifact_pointer()
    if change == "truncated":
        body = body[:-16]
    elif change == "trailing":
        body += bytes(16)
    elif change == "shape":
        pointer["artifacts"][1]["shape"] = [4, 3]
    elif change == "duplicate":
        pointer["artifacts"][1]["result_type"] = "statevector"
    else:
        pointer["artifacts"][0][change] = "unsupported"
    with pytest.raises(InvalidResponseError):
        JobManager._decode_binary_artifact_payload(body, pointer)


@pytest.mark.parametrize(
    ("manifest_update", "internal_message"),
    [
        ({"schema_version": True}, "schema version"),
        ({"execution_time": -0.1}, "finite and nonnegative"),
        ({"execution_time": float("inf")}, "finite and nonnegative"),
        ({"counts": {"00": 1.5}}, "nonnegative integers"),
        ({"counts": {"00": -1}}, "nonnegative integers"),
    ],
)
def test_binary_manifest_rejects_invalid_counts_and_time(
    manifest_update: dict, internal_message: str
):
    pointer, body, _, _ = _dual_artifact_pointer()
    pointer.update(manifest_update)

    with pytest.raises(
        InvalidResponseError,
    ) as exc_info:
        JobManager._decode_binary_artifact_payload(body, pointer)

    assert isinstance(exc_info.value.__cause__, ContractValidationError)
    assert internal_message in str(exc_info.value.__cause__)
