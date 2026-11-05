from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from qpiai_quantum.jobmanager.jobmanager import JobManager


def _configure_job_manager(
    job_manager: JobManager, monkeypatch: pytest.MonkeyPatch
) -> tuple[SimpleNamespace, MagicMock]:
    monkeypatch.setattr(
        job_manager, "submit_qasm_job", MagicMock(return_value={"job_id": "job-123"})
    )
    polling = MagicMock(
        return_value={"job_info": {"id": "job-123", "status": "completed"}}
    )
    monkeypatch.setattr(job_manager, "_wait_with_polling", polling)
    result = SimpleNamespace(job_metadata=None)
    monkeypatch.setattr(
        job_manager, "_parse_job_results", MagicMock(return_value=result)
    )
    return result, polling


@patch("qpiai_quantum.jobmanager.jobmanager.SSE_AVAILABLE", False)
def test_submit_and_wait_polls_when_sse_is_unavailable(
    backend_devices: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_manager = JobManager()
    job_result, polling = _configure_job_manager(job_manager, monkeypatch)

    result = job_manager.submit_and_wait_for_results_qasm("OPENQASM 2.0;")

    assert result is job_result
    polling.assert_called_once()
    assert polling.call_args.args[:2] == ("job-123", 300)


@patch("qpiai_quantum.jobmanager.jobmanager.SSE_AVAILABLE", True)
@patch("qpiai_quantum.jobmanager.jobmanager.SSEResultHandler")
def test_submit_and_wait_polls_when_sse_monitoring_fails(
    mock_sse_handler: MagicMock,
    backend_devices: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_sse_handler.return_value.wait_for_events.return_value = False
    mock_sse_handler.return_value.get_error.return_value = "connection reset"
    job_manager = JobManager()
    job_result, polling = _configure_job_manager(job_manager, monkeypatch)
    monkeypatch.setattr(job_manager, "_get_auth_header", MagicMock(return_value={}))

    result = job_manager.submit_and_wait_for_results_qasm("OPENQASM 2.0;")

    assert result is job_result
    mock_sse_handler.return_value.wait_for_events.assert_called_once()
    polling.assert_called_once()
