import json
import logging
import time
from typing import Any, Optional
from .result_decoding import decode_execution_time_seconds, parse_circuit_response

try:
    from requests_sse import (
        EventSource as _EventSource,
        InvalidStatusCodeError,
        InvalidContentTypeError,
    )

    EventSource: Any = _EventSource
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False
    EventSource = None
logger = logging.getLogger("qpiai_quantum.jobmanager")


class SSEResultHandler:
    """
    Handles Server-Sent Events (SSE) messages and tracks execution completion.

    This class processes incoming SSE events from the quantum backend,
    tracks job status, and extracts results when jobs complete.

    Attributes:
        device_name (Optional[str]): The device name ("QpiAI-QSV-Simulator", "QpiAI-QDM-Simulator",
                                "QpiAI-QTN-Simulator",
                                "QpiAI-Indus-1",
                                "QpiAI-QSV-Lite",
                                "QpiAI-QDM-Lite") for result parsing
        is_completed (bool): Whether the job has completed
        final_result (Optional[dict]): The final execution result
        error_message (Optional[str]): Error message if job failed
        timeout (int): Timeout in seconds for SSE connection
    """

    def __init__(self, device_name: str | None = None, timeout: int = 3600):
        self.device_name = device_name
        self.is_completed = False
        self.final_result: dict | None = None
        self.error_message: str | None = None
        self.timeout = timeout

    def _parse_simulator_response(self, response_data: dict) -> dict:
        """Parse simulator response format: results.data.counts, results.data.statevector."""
        return parse_circuit_response(response_data)

    def wait_for_events(self, url: str, headers: dict) -> bool:
        """
        Connect to SSE endpoint and wait for job completion.

        Args:
            url (str): SSE endpoint URL
            headers (dict): HTTP headers (including authentication)

        Returns:
            bool: True if job completed successfully, False if failed or timed out

        Raises:
            Exception: If SSE not available or connection fails
        """
        if not SSE_AVAILABLE or EventSource is None:
            raise RuntimeError(
                "SSE library not available. Install with: pip install requests-sse"
            )
        try:
            start_time = time.time()
            with EventSource(url, headers=headers) as event_source:
                for event in event_source:
                    if time.time() - start_time > self.timeout:
                        self.error_message = "SSE connection timed out."
                        logger.warning(self.error_message)
                        return False
                    event_data = event.data
                    if event_data:
                        try:
                            if isinstance(event_data, str) and event_data.startswith(
                                "data:"
                            ):
                                event_data = event_data[5:].strip()
                            data = json.loads(event_data)
                            status = data.get("status", "")
                            job_id = data.get("job_id")
                            logger.debug("SSE event: status=%s", status)
                            if status.lower() in [
                                "success",
                                "completed",
                                "failed",
                                "error",
                                "cancelled",
                            ]:
                                self.is_completed = True
                                if status.lower() in ["success", "completed"]:
                                    response_value = data.get("response", "{}")
                                    parsed = parse_circuit_response({})
                                    try:
                                        if isinstance(response_value, dict):
                                            response_data = response_value
                                        elif isinstance(
                                            response_value, str
                                        ) and response_value.startswith("s3://"):
                                            response_data = None
                                        else:
                                            response_data = json.loads(response_value)
                                        if isinstance(response_data, dict):
                                            parsed = parse_circuit_response(
                                                response_data
                                            )
                                    except json.JSONDecodeError:
                                        logger.warning(
                                            "SSE success event had non-JSON response payload; deferring result fetch to job status API"
                                        )
                                    execution_time = decode_execution_time_seconds(
                                        parsed["execution_time"]
                                    )
                                    if execution_time is None:
                                        execution_time = decode_execution_time_seconds(
                                            data.get("execution_time")
                                        )
                                    self.final_result = {
                                        "state_vector": parsed["statevector"],
                                        "probabilities": parsed["probabilities"],
                                        "density_matrix": parsed["density_matrix"],
                                        "histogram": parsed["counts"],
                                        "method": data.get("method")
                                        or parsed.get("method"),
                                        "time": execution_time,
                                        "job_id": job_id or parsed.get("job_id"),
                                        "metadata": parsed["metadata"],
                                        "message": parsed["message"],
                                        "response_data": response_value,
                                        **{
                                            field: data.get(field)
                                            if data.get(field) is not None
                                            else parsed.get(field)
                                            for field in (
                                                "cpu_usage",
                                                "memory_usage",
                                                "gpu_memory_usage",
                                            )
                                        },
                                    }
                                    logger.info("Job completed successfully via SSE")
                                    return True
                                elif status.lower() in ["failed", "error", "cancelled"]:
                                    if status.lower() == "cancelled":
                                        self.error_message = (
                                            f"Job {job_id} was cancelled."
                                        )
                                    else:
                                        self.error_message = f"Job {job_id} failed during execution. Review the circuit, backend settings, and job configuration before retrying."
                                    return False
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse SSE event data")
                            continue
        except InvalidStatusCodeError:
            self.error_message = "Unable to open the live event stream for this job. Retry the run or check the job status manually."
            return False
        except InvalidContentTypeError:
            self.error_message = "Unable to open the live event stream for this job. Retry the run or check the job status manually."
            return False
        except Exception:
            self.error_message = "Lost the live event stream for this job. Retry the run or check the job status manually."
            return False
        return False

    def get_result(self) -> dict | None:
        """Get the final result after completion."""
        return self.final_result

    def get_error(self) -> str | None:
        """Get the error message if job failed."""
        return self.error_message
