from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from qpiai_quantum.jobmanager.jobmanager import JobManager


def _configure_job_manager(job_manager: JobManager) -> MagicMock:
    job_manager.submit_qasm_job = MagicMock(return_value={"job_id": "job-123"})
    job_manager._wait_with_polling = MagicMock(
        return_value={"job_info": {"id": "job-123", "status": "completed"}}
    )
    job_manager._parse_job_results = MagicMock(
        return_value=SimpleNamespace(job_metadata=None)
    )
    return job_manager._parse_job_results.return_value


@patch("qpiai_quantum.jobmanager.jobmanager.SSE_AVAILABLE", False)
def test_submit_and_wait_polls_when_sse_is_unavailable():
    job_manager = JobManager()
    job_result = _configure_job_manager(job_manager)

    result = job_manager.submit_and_wait_for_results_qasm("OPENQASM 2.0;")

    assert result is job_result
    job_manager._wait_with_polling.assert_called_once()
    assert job_manager._wait_with_polling.call_args.args[:2] == ("job-123", 300)


@patch("qpiai_quantum.jobmanager.jobmanager.SSE_AVAILABLE", True)
@patch("qpiai_quantum.jobmanager.jobmanager.SSEResultHandler")
def test_submit_and_wait_polls_when_sse_monitoring_fails(mock_sse_handler):
    mock_sse_handler.return_value.wait_for_events.return_value = False
    mock_sse_handler.return_value.get_error.return_value = "connection reset"
    job_manager = JobManager()
    job_result = _configure_job_manager(job_manager)
    job_manager._get_auth_header = MagicMock(return_value={"X-Secret-Token": "key"})

    result = job_manager.submit_and_wait_for_results_qasm("OPENQASM 2.0;")

    assert result is job_result
    mock_sse_handler.return_value.wait_for_events.assert_called_once()
    job_manager._wait_with_polling.assert_called_once()
