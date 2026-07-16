import json
import logging
import time
from typing import Optional

try:
    from requests_sse import (
        EventSource,
        InvalidStatusCodeError,
        InvalidContentTypeError,
    )

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
        device_name (Optional[str]): The device name ("QpiAI-QSV-Local", "QpiAI-QSV-Simulator", "QpiAI-QDM-Simulator",
                                "QpiAI-QTN-Simulator",
                                "QpiAI-Indus-1",
                                "QpiAI-QSV-Lite",
                                "QpiAI-QDM-Lite") for result parsing
        is_completed (bool): Whether the job has completed
        final_result (Optional[dict]): The final execution result
        error_message (Optional[str]): Error message if job failed
        timeout (int): Timeout in seconds for SSE connection
    """

    def __init__(self, device_name: str | None = None, timeout: int = 300):
        self.device_name = device_name
        self.is_completed = False
        self.final_result: dict | None = None
        self.error_message: str | None = None
        self.timeout = timeout

    def _parse_simulator_response(self, response_data: dict) -> dict:
        """Parse simulator response format: results.data.counts, results.data.statevector."""
        results_data = response_data.get("results", {})
        data = results_data.get("data", {})
        statevector_raw = data.get("statevector")
        converted_statevector = None
        if statevector_raw:
            # Convert statevector from [{"real": x, "imag": y}, ...] to complex numbers
            converted_statevector = []
            for state in statevector_raw:
                if isinstance(state, dict) and "real" in state and "imag" in state:
                    complex_val = complex(state["real"], state["imag"])
                    converted_statevector.append([complex_val])
                else:
                    converted_statevector.append(state)

        return {
            "counts": data.get("counts"),
            "statevector": converted_statevector,
            "probabilities": data.get("probabilities"),
            "execution_time": response_data.get("execution_time"),
        }

    def _parse_qpu_response(self, response_data: dict) -> dict:
        """Parse QPU response format with nested structure: results.results.counts."""
        counts = None

        if isinstance(response_data, dict):
            # Newer responses can return counts directly at the top-level
            # (for example: {"counts": {...}, ...}).
            counts = response_data.get("counts")

            # Backward-compatible handling for nested shapes.
            if counts is None and "results" in response_data:
                results = response_data["results"]
                if isinstance(results, dict):
                    counts = results.get("counts")
                    if counts is None and "results" in results:
                        inner_results = results["results"]
                        if isinstance(inner_results, dict):
                            counts = inner_results.get("counts")

        return {"counts": counts, "execution_time": response_data.get("execution_time")}

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
                    # Check timeout
                    if time.time() - start_time > self.timeout:
                        logger.warning(
                            f"SSE connection timed out after {self.timeout}s"
                        )
                        return False

                    # Process event
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

                            logger.debug(f"SSE Event: job_id={job_id}, status={status}")

                            # Check for completion
                            if status.lower() in [
                                "success",
                                "completed",
                                "failed",
                                "error",
                                "cancelled",
                            ]:
                                self.is_completed = True

                                if status.lower() in ["success", "completed"]:
                                    # Parse results
                                    response_value = data.get("response", "{}")
                                    counts = None
                                    execution_time = None
                                    converted_statevector = None

                                    try:
                                        # Handle both dict (already parsed) and JSON string.
                                        # If response is an S3 path string, parsing is skipped here;
                                        # caller will fetch final results via job-status APIs.
                                        if isinstance(response_value, dict):
                                            response_data = response_value
                                        elif isinstance(
                                            response_value, str
                                        ) and response_value.startswith("s3://"):
                                            response_data = None
                                        else:
                                            response_data = json.loads(response_value)

                                        if isinstance(response_data, dict):
                                            # Use device-specific parsing
                                            if self.device_name == "QpiAI-Indus-1":
                                                parsed_data = self._parse_qpu_response(
                                                    response_data
                                                )
                                                counts = parsed_data.get("counts")
                                                execution_time = parsed_data.get(
                                                    "execution_time"
                                                )
                                            else:
                                                parsed_data = (
                                                    self._parse_simulator_response(
                                                        response_data
                                                    )
                                                )
                                                counts = parsed_data.get("counts")
                                                execution_time = parsed_data.get(
                                                    "execution_time"
                                                )
                                                converted_statevector = parsed_data.get(
                                                    "statevector"
                                                )
                                    except json.JSONDecodeError:
                                        logger.warning(
                                            "SSE success event had non-JSON response payload; "
                                            "continuing and deferring result fetch to job status API"
                                        )

                                    self.final_result = {
                                        "state_vector": converted_statevector,
                                        "probabilities": None,
                                        "histogram": counts,
                                        "time": execution_time,
                                        "cpu_usage": None,
                                        "memory_usage": None,
                                        "gpu_memory_usage": None,
                                    }

                                    logger.info(
                                        f"Job {job_id} completed successfully via SSE"
                                    )
                                    return True

                                elif status.lower() in ["failed", "error", "cancelled"]:
                                    self.error_message = (
                                        f"Job {job_id} failed with status: {status}"
                                    )
                                    error_detail = data.get(
                                        "error",
                                        data.get("error_message", "Unknown error"),
                                    )
                                    self.error_message += f"\nDetails: {error_detail}"
                                    logger.error(self.error_message)
                                    return False

                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse SSE event data: {event_data}"
                            )
                            continue

        except InvalidStatusCodeError as e:
            self.error_message = (
                f"SSE connection failed with invalid status: {e.response.status_code}"
            )
            logger.error(self.error_message)
            return False
        except InvalidContentTypeError:
            self.error_message = "SSE endpoint returned invalid content-type (expected text/event-stream)"
            logger.error(self.error_message)
            return False
        except Exception as e:
            self.error_message = f"SSE connection error: {str(e)}"
            logger.error(self.error_message)
            return False

        # Should not reach here normally
        return False

    def get_result(self) -> dict | None:
        """Get the final result after completion."""
        return self.final_result

    def get_error(self) -> str | None:
        """Get the error message if job failed."""
        return self.error_message
