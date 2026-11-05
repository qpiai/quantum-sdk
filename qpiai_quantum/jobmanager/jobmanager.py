import json
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Any, Literal
import numpy as np
from ..authentication.user import resolve_api_key
from ..circuit.circuit import Circuit
from ..config import HTTP_TIMEOUT, get_api_url, get_sse_url
from ..icr.icr import IntermediateCircuitRepresentation
from .exceptions import (
    AuthenticationError,
    CircuitNotFoundError,
    CircuitUpdateError,
    ExperimentNotFoundError,
    InvalidExperimentResponseError,
    InvalidResponseError,
    JobManagerError,
    JobSubmissionError,
)
from .backend import DeviceInfo, resolve_execution_request, DeviceRegistry
from .job_result import JobResult
from .result_decoding import (
    decode_execution_time_seconds,
    decode_density_matrix,
    decode_statevector,
    parse_circuit_response,
)
from .result_contract import ContractValidationError, validate_binary_manifest
from .sse_handler import SSE_AVAILABLE, SSEResultHandler

logger = logging.getLogger("qpiai_quantum.jobmanager")


class JobManager:
    """
    QPIAI Job Manager for handling quantum job lifecycle management.

    Provides functions for submitting, monitoring, and retrieving results
    from quantum circuit execution jobs on QPIAI backends.
    """

    _resource_cache: dict[tuple, tuple[str, float]] = {}
    _resource_cache_lock = threading.Lock()
    _resource_cache_generation = 0

    @classmethod
    def clear_resource_cache(cls) -> None:
        """Clear cached compute-resource identifiers and the loaded device list."""
        with cls._resource_cache_lock:
            cls._resource_cache_generation += 1
            cls._resource_cache.clear()
        DeviceRegistry.clear()

    @staticmethod
    def track_job(func, job_name: str):
        """
        A decorator to track the job by its name.

        Args:
            func: Function to wrap
            job_name (str): Name of the job for tracking

        Returns:
            Wrapped function that logs job tracking
        """

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    def _get_auth_header(self) -> dict:
        """
        Get authentication header using current user's API key.

        Returns:
            dict: Authentication headers for API requests

        Raises:
            BaseError: If user is not authenticated
        """
        token = resolve_api_key()
        headers = {"X-Secret-Token": token} if token else None
        if headers is None:
            raise AuthenticationError(
                "No API key found in the current user context. Please login using QpiAIQuantumAuth.login(...)."
            )
        return headers

    def _get_compute_resource_id(self, method: str, device_name: str) -> str | None:
        """
        Get compute resource ID based on method and device_name.
        Matches the device against the compute resources listed by the backend.

        Args:
            method (str): Execution method ("statevector", "density_matrix", "tensor_network",
                         or variants like "statevector_sampling")
            device_name (str): Device name as listed by the backend, e.g. "QpiAI-QSV-Simulator"

        Returns:
            Optional[str]: Compute resource ID (UUID string), or None if not found
        """
        context = DeviceRegistry._current_cache_key()
        cache_key = (*context, method, device_name)
        with self._resource_cache_lock:
            generation = self._resource_cache_generation
            cached_entry = self._resource_cache.get(cache_key)
            if cached_entry and time.monotonic() - cached_entry[1] < 300:
                return cached_entry[0]
        resources = DeviceRegistry._fetch(context)
        matches = []
        for resource in resources:
            if resource.get("execution_kind") == "emulator":
                continue
            if resource.get("status") not in {None, "active"}:
                continue
            if device_name not in {
                resource.get("backend_name"),
                resource.get("device_name"),
            }:
                continue
            profile = DeviceInfo.from_resource(device_name, resource)
            if profile.method == method and resource.get("ID"):
                matches.append(resource["ID"])
        if len(matches) != 1:
            raise JobSubmissionError(
                f"Expected one active matching resource for '{device_name}', found {len(matches)}. Refresh the device list and retry."
            )
        with self._resource_cache_lock:
            if generation == self._resource_cache_generation:
                self._resource_cache[cache_key] = (matches[0], time.monotonic())
        return matches[0]

    def list_compute_resources(self) -> list[dict[str, Any]]:
        """
        List all available compute resources for the authenticated user.

        Returns:
            List[Dict]: List of compute resources with their details

        Raises:
            JobManagerError: If the API call fails
        """
        try:
            import requests
        except ImportError:
            raise JobManagerError(
                "requests is required. Install it with `pip install requests`."
            )
        try:
            url = get_api_url("/api/compute-resources")
            headers = self._get_auth_header()
            response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            resources = response.json()
            if isinstance(resources, list):
                return resources
            if isinstance(resources, dict) and "compute_resources" in resources:
                return resources["compute_resources"]
            return []
        except JobManagerError:
            raise
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500:
                raise JobManagerError(
                    "The server encountered an internal error. Please try again later or contact support."
                ) from exc
            else:
                raise JobManagerError(
                    "Unable to load compute resources. Check that your API key is valid and retry the request."
                ) from exc
        except requests.exceptions.RequestException as exc:
            raise JobManagerError(
                "Unable to load compute resources. Check that your API key is valid and retry the request."
            ) from exc
        except Exception as exc:
            raise JobManagerError(
                "Unable to load compute resources. Check that your API key is valid and retry the request."
            ) from exc

    def get_experiment_id_by_name(self, experiment_name: str) -> str:
        """
        Get experiment ID by name from the API.
        (Consistent with Executor pattern - raises error if not found)

        Args:
            experiment_name (str): The name of the experiment

        Returns:
            str: The experiment ID

        Raises:
            BaseError: If the experiment is not found or API call fails
        """
        try:
            import requests
        except ImportError:
            raise JobManagerError(
                "requests is required. Install it with `pip install requests`."
            )
        try:
            url = get_api_url(f"/api/experiments/by-name?name={experiment_name}")
            headers = self._get_auth_header()
            response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            if response.status_code == 404:
                raise JobManagerError(
                    f"Experiment '{experiment_name}' not found. Please create the experiment first."
                )
            if response.status_code == 403:
                raise JobManagerError("Access denied for the requested experiment.")
            response.raise_for_status()
            experiment_data = response.json()
            experiment_id = experiment_data.get("experiment_id")
            if not experiment_id:
                raise InvalidExperimentResponseError(
                    "Invalid response from experiments API: missing experiment_id."
                )
            return experiment_id
        except JobManagerError:
            raise
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500:
                raise JobManagerError(
                    "The server encountered an internal error. Please try again later or contact support."
                ) from exc
            else:
                raise JobManagerError(
                    f"Unable to look up experiment '{experiment_name}'. Check that the experiment exists and that your API key can access it."
                ) from exc
        except requests.exceptions.RequestException as exc:
            raise JobManagerError(
                f"Unable to look up experiment '{experiment_name}'. Check that the experiment exists and that your API key can access it."
            ) from exc
        except Exception as exc:
            raise JobManagerError(
                f"Unable to look up experiment '{experiment_name}'. Check the experiment name and retry."
            ) from exc

    def _convert_circuit_to_qasm(self, circuit) -> str:
        """
        Convert Circuit or IntermediateCircuitRepresentation object to OpenQASM string.

        Args:
            circuit: Circuit or IntermediateCircuitRepresentation object

        Returns:
            str: OpenQASM string representation of the circuit

        Raises:
            BaseError: If circuit is not a valid Circuit or ICR object
        """
        if isinstance(circuit, Circuit):
            qasm_result = circuit.to_qasm()
            if isinstance(qasm_result, list):
                return "\n".join(qasm_result)
            return qasm_result
        elif isinstance(circuit, IntermediateCircuitRepresentation):
            temp_circuit = Circuit()
            temp_circuit.icr = circuit
            qasm_result = temp_circuit.to_qasm()
            if isinstance(qasm_result, list):
                return "\n".join(qasm_result)
            return qasm_result
        else:
            raise JobManagerError(
                f"Cannot convert circuit of type {type(circuit)}, expected a Circuit or IntermediateCircuitRepresentation object."
            )

    def submit_qasm_job(
        self,
        qasm_string: str,
        experiment_name: str = "Default Experiment",
        shots: int = 1024,
        method: str | None = None,
        need_statevector: bool = False,
        need_density_matrix: bool = False,
        device_name: str = "QpiAI-QSV-Simulator",
        circuit_name: str = "circuit",
        compute_resource_id: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Submit OpenQASM 2.0 circuit for execution.

        Args:
            qasm_string (str): OpenQASM 2.0 circuit description
            experiment_name (str): Name of experiment to associate with (default: "Default Experiment")
                                Note: Experiment must exist - create experiment first
            shots (int): Number of shots to execute (default: 1024)
            method (str): Execution method (default: "statevector")
            device_name (str): Device name (default: "QpiAI-QSV-Simulator")
            circuit_name (str): Circuit name (default: "circuit")
            compute_resource_id (str): Compute resource ID (optional)
            overwrite (bool): Whether to overwrite existing circuit with same name (default: False)

        Returns:
            Dict containing job ID and submission info:
                - job_id: str, Unique job identifier
                - status: str, Job status ("submitted")
                - experiment_id: str, Experiment ID
                - circuit_id: str, Circuit ID
                - submission_info: dict, Additional submission details

        Raises:
            BaseError: If submission fails, circuit exists and overwrite=False, QASM is invalid,
                   or experiment does not exist
        """
        request = resolve_execution_request(
            device_name,
            method=method,
            need_statevector=need_statevector,
            need_density_matrix=need_density_matrix,
        )
        profile = request.device
        method = profile.method
        need_statevector = request.need_statevector
        need_density_matrix = request.need_density_matrix
        try:
            import requests
        except ImportError:
            raise JobManagerError(
                "requests is required. Install it with `pip install requests`."
            )
        try:
            experiment_id = self.get_experiment_id_by_name(experiment_name)
            if circuit_name is None:
                import uuid

                circuit_name = f"circuit_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            associated_device_name = {
                "QpiAI-QSV-Lite": "QpiAI-QSV-Simulator",
                "QpiAI-QDM-Lite": "QpiAI-QDM-Simulator",
            }.get(device_name, device_name)
            if compute_resource_id is None:
                compute_resource_id = self._get_compute_resource_id(method, device_name)
                if compute_resource_id:
                    logger.debug(
                        "Auto-selected compute resource for method=%s device_name=%s",
                        method,
                        device_name,
                    )
            headers = self._get_auth_header()
            circuit_url = get_api_url("/api/circuits/create")
            circuit_payload: dict[str, Any] = {
                "openqasm_string": qasm_string,
                "shots": shots,
                "name": circuit_name,
                "device_name": associated_device_name,
                "source_type": "sdk",
            }
            logger.debug("Creating circuit for experiment '%s'", experiment_name)
            circuit_response = requests.post(
                circuit_url, json=circuit_payload, headers=headers, timeout=HTTP_TIMEOUT
            )
            if circuit_response.status_code == 409:
                if not overwrite:
                    raise JobManagerError(
                        f"Circuit '{circuit_name}' already exists. Set overwrite=True to replace it."
                    )
                logger.info(
                    "Circuit '%s' already exists, attempting to update...", circuit_name
                )
                try:
                    list_url = get_api_url("/api/circuits/?page=1&page_size=100")
                    if experiment_id:
                        list_url += f"&experiment_id={experiment_id}"
                    list_response = requests.get(
                        list_url, headers=headers, timeout=HTTP_TIMEOUT
                    )
                    list_response.raise_for_status()
                    circuits_data = list_response.json()
                    circuits = circuits_data.get("circuits", [])
                    existing_circuit = None
                    for circuit in circuits:
                        if circuit.get("name") == circuit_name:
                            existing_circuit = circuit
                            break
                    if existing_circuit and existing_circuit.get("id"):
                        circuit_id = existing_circuit["id"]
                        logger.info(
                            "Found existing circuit '%s' with ID %s",
                            circuit_name,
                            circuit_id,
                        )
                        update_url = get_api_url(f"/api/circuits/{circuit_id}")
                        update_payload: dict[str, Any] = {
                            "qasm": qasm_string,
                            "name": circuit_name,
                            "device_name": associated_device_name,
                            "shots": shots,
                            "method": method,
                            "need_statevector": need_statevector,
                            "need_density_matrix": need_density_matrix,
                        }
                        circuit_update_response = requests.put(
                            update_url,
                            json=update_payload,
                            headers=headers,
                            timeout=HTTP_TIMEOUT,
                        )
                        circuit_update_response.raise_for_status()
                    else:
                        raise JobManagerError(
                            f"Could not find the existing circuit '{circuit_name}' to update."
                        )
                except JobManagerError:
                    raise
                except Exception as update_error:
                    raise CircuitUpdateError(update_error) from update_error
            else:
                circuit_response.raise_for_status()
                circuit_id = circuit_response.json().get("id")
            job_url = get_api_url("/api/jobs/qasm")
            job_payload = {
                "circuit_id": circuit_id,
                "experiment_id": experiment_id,
                "method": method,
                "need_statevector": need_statevector,
                "need_density_matrix": need_density_matrix,
                "shots": shots,
                "device_name": device_name,
            }
            if compute_resource_id is not None:
                job_payload["compute_resource_id"] = compute_resource_id
            job_response = requests.post(
                job_url, json=job_payload, headers=headers, timeout=HTTP_TIMEOUT
            )
            if job_response.status_code == 400:
                raise JobSubmissionError(
                    "Job submission was rejected. Please verify the circuit, method, device, and shot count.\n\nCommon causes:\n  1. You may not have access to the specified compute resource\n  2. Invalid circuit QASM format\n  3. Invalid method or device_name value\n  4. Missing or invalid circuit_id or experiment_id\n  5. Invalid shots value (must be positive integer)"
                )
            job_response.raise_for_status()
            job_data = job_response.json()
            if "job" in job_data:
                job_info = job_data["job"]
                job_id = job_info.get("id")
                job_status = job_info.get("status", "submitted")
            else:
                job_id = job_data.get("id")
                job_status = job_data.get("status", "submitted")
            if not job_id:
                raise JobSubmissionError(
                    "Job submission failed because the server did not return a job ID. Retry the submission and contact support if this persists."
                )
            return {
                "job_id": job_id,
                "status": job_status,
                "experiment_id": experiment_id,
                "circuit_id": circuit_id,
            }
        except JobManagerError:
            raise
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500:
                raise JobSubmissionError(
                    "The server encountered an internal error. Please try again later or contact support."
                ) from exc
            else:
                raise JobSubmissionError(
                    "Unable to submit the job. Check the experiment name, circuit, backend settings, and shot count, then try again."
                ) from exc
        except requests.exceptions.RequestException as exc:
            raise JobSubmissionError(
                "Unable to submit the job. Check the experiment name, circuit, backend settings, and shot count, then try again."
            ) from exc
        except Exception as exc:
            raise JobManagerError(
                "Unable to submit the job. Check the experiment name, circuit, backend settings, and shot count, then try again."
            ) from exc

    def submit_circuit_job(
        self,
        circuit: Circuit | IntermediateCircuitRepresentation,
        experiment_name: str = "Default Experiment",
        shots: int = 1024,
        method: str | None = None,
        need_statevector: bool = False,
        need_density_matrix: bool = False,
        device_name: str = "QpiAI-QSV-Simulator",
        circuit_name: str | None = None,
        compute_resource_id: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Submit Circuit or IntermediateCircuitRepresentation for execution.

        This method provides the same functionality as submit_qasm_job() but accepts
        Circuit or ICR objects directly, converting them to QASM automatically.

        Args:
            circuit (Circuit | IntermediateCircuitRepresentation): Circuit or ICR object to execute
            experiment_name (str): Name of experiment to associate with (default: "Default Experiment")
                                Note: Experiment must exist - create experiment first
            shots (int): Number of shots to execute (default: 1024)
            method (str): Execution method (default: "statevector")
            device_name (str): Device name (default: "QpiAI-QSV-Simulator")
            circuit_name (str): Circuit name (default: "circuit")
            compute_resource_id (str): Compute resource ID (optional)
            overwrite (bool): Whether to overwrite existing circuit with same name (default: False)

        Returns:
            Dict containing job ID and submission info:
                - job_id: str, Unique job identifier
                - status: str, Job status ("submitted")
                - experiment_id: str, Experiment ID
                - circuit_id: str, Circuit ID
                - submission_info: dict, Additional submission details

        Raises:
            BaseError: If submission fails, circuit exists and overwrite=False, circuit type is invalid,
                   or experiment does not exist
        """
        qasm_string = self._convert_circuit_to_qasm(circuit)
        return self.submit_qasm_job(
            qasm_string=qasm_string,
            experiment_name=experiment_name,
            shots=shots,
            method=method,
            need_statevector=need_statevector,
            need_density_matrix=need_density_matrix,
            device_name=device_name,
            circuit_name=circuit_name or "Circuit",
            compute_resource_id=compute_resource_id,
            overwrite=overwrite,
        )

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """
        Check status of specific job by its ID.

        Args:
            job_id (str): The unique identifier of the job

        Returns:
            Dictionary containing job details or None if job not found:
                - id: str, Job unique identifier
                - status: str, Job status ("pending", "running", "completed", "success", "failed", "scheduled")
                - user_id: str, User ID who submitted the job
                - circuit_id: str, Circuit ID
                - experiment_id: str, Experiment ID
                - project_id: str, Project ID
                - workspace_id: str, Workspace ID
                - compute_resource_id: str, Compute resource ID
                - is_scheduled: bool, Whether job is scheduled
                - scheduled_at: str or None, Scheduled timestamp
                - server_response_id: str or None, Response ID
                - error: str or None, Error message if failed
                - credits_used: int, Credits consumed
                - execution_time: float, Execution time in seconds
                - created_at: str, Creation timestamp
                - updated_at: str, Last update timestamp
                - response_data: str or None, MongoDB response data

        Raises:
            BaseError: If API call fails or authentication error occurs
        """
        try:
            import requests
        except ImportError:
            raise JobManagerError(
                "requests is required. Install it with `pip install requests`."
            )
        try:
            url = get_api_url(f"/api/jobs/{job_id}")
            headers = self._get_auth_header()
            response = requests.get(
                url, headers=headers, stream=True, timeout=HTTP_TIMEOUT
            )
            if response.status_code == 404:
                logger.info("Job %s not found", job_id)
                return None
            response.raise_for_status()
            chunks = []
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
            return json.loads(b"".join(chunks))
        except JobManagerError:
            raise
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500:
                raise JobManagerError(
                    "The server encountered an internal error. Please try again later or contact support."
                ) from exc
            else:
                raise JobManagerError(
                    f"Unable to retrieve the status for job {job_id}. Confirm the job ID is correct and retry."
                ) from exc
        except requests.exceptions.RequestException as exc:
            raise JobManagerError(
                f"Unable to retrieve the status for job {job_id}. Confirm the job ID is correct and retry."
            ) from exc
        except Exception as exc:
            raise JobManagerError(
                f"Unable to retrieve the status for job {job_id}. Confirm the job ID is correct and retry."
            ) from exc

    def get_current_job(self) -> dict[str, Any] | None:
        """
        Get the currently running job for the authenticated user.

        Returns:
            Dictionary containing current running job details or None if no job is running:
                - id: str, Job unique identifier
                - status: str, Job status (should be "running")
                - user_id: str, User ID who submitted the job
                - circuit_id: str, Circuit ID
                - experiment_id: str, Experiment ID
                - project_id: str, Project ID
                - workspace_id: str, Workspace ID
                - compute_resource_id: str, Compute resource ID
                - credits_used: int, Credits consumed
                - execution_time_ms: int, Execution time in milliseconds
                - created_at: str, Creation timestamp
                - updated_at: str, Last update timestamp
                - Other job fields as documented in get_job_status()

        Raises:
            BaseError: If API call fails or authentication error occurs
        """
        try:
            import requests
        except ImportError:
            raise JobManagerError(
                "requests is required. Install it with `pip install requests`."
            )
        try:
            url = get_api_url("/api/jobs/user?status=running&page_size=1")
            headers = self._get_auth_header()
            response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            jobs = data.get("jobs", [])
            if not jobs:
                logger.info("No currently running job found")
                return None
            return jobs[0]
        except JobManagerError:
            raise
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500:
                raise JobManagerError(
                    "The server encountered an internal error. Please try again later or contact support."
                ) from exc
            else:
                raise JobManagerError(
                    "Unable to load the current job. Retry the request or query a specific job ID instead."
                ) from exc
        except requests.exceptions.RequestException as exc:
            raise JobManagerError(
                "Unable to load the current job. Retry the request or query a specific job ID instead."
            ) from exc
        except Exception as exc:
            raise JobManagerError(
                "Unable to load the current job. Retry the request or query a specific job ID instead."
            ) from exc

    def get_job_history(
        self,
        period: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """
        List recent jobs for the authenticated user with optional filtering.

        Args:
            period (str, optional): Time period filter ("daily", "weekly", "monthly", or None)
            status (str, optional): Status filter ("pending", "running", "completed", "success", "failed", "scheduled", or None)
            page (int): Page number (default: 1, 1-indexed)
            page_size (int): Jobs per page (default: 20, max 100)

        Returns:
            Dictionary containing job list and pagination information:
                - jobs: list[dict], List of job objects with following fields:
                    - id: str, Job unique identifier
                    - status: str, Job status
                    - credits_used: int, Credits consumed
                    - execution_time_ms: int, Execution time in milliseconds
                    - created_at: str, Creation timestamp
                    - updated_at: str, Last update timestamp
                    - circuit_id: str, Circuit ID
                    - experiment_id: str, Experiment ID
                    - compute_resource_id: str, Compute resource ID
                    - project_id: str, Project ID
                    - workspace_id: str, Workspace ID
                - pagination: dict, Pagination info:
                    - current_page: int, Current page number
                    - page_size: int, Jobs per page
                    - total: int, Total number of jobs

        Raises:
            BaseError: If API call fails, invalid parameters, or authentication error occurs
        """
        try:
            import requests
        except ImportError:
            raise JobManagerError(
                "requests is required. Install it with `pip install requests`."
            )
        try:
            if page < 1:
                raise JobManagerError("Page number must be >= 1.")
            if page_size < 1 or page_size > 100:
                raise JobManagerError("Page size must be between 1 and 100.")
            if period:
                valid_periods = ["daily", "weekly", "monthly"]
                if period not in valid_periods:
                    raise JobManagerError(f"Period must be one of: {valid_periods}.")
            if status:
                valid_statuses = [
                    "pending",
                    "running",
                    "completed",
                    "success",
                    "failed",
                    "scheduled",
                ]
                if status.lower() not in valid_statuses:
                    raise JobManagerError(f"Status must be one of: {valid_statuses}.")
            params: dict[str, Any] = {"page": page, "page_size": page_size}
            if period:
                params["period"] = period
            if status:
                params["status"] = status
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            url = get_api_url(f"/api/jobs/user?{query_string}")
            headers = self._get_auth_header()
            response = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            return {
                "jobs": data.get("jobs", []),
                "pagination": {
                    "current_page": data.get("pagination", {}).get(
                        "current_page", page
                    ),
                    "page_size": data.get("pagination", {}).get("page_size", page_size),
                    "total": data.get("pagination", {}).get("total", 0),
                },
            }
        except JobManagerError:
            raise
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500:
                raise JobManagerError(
                    "The server encountered an internal error. Please try again later or contact support."
                ) from exc
            else:
                raise JobManagerError(
                    "Unable to load job history. Retry the request or simplify the page, status, and period filters."
                ) from exc
        except requests.exceptions.RequestException as exc:
            raise JobManagerError(
                "Unable to load job history. Retry the request or simplify the page, status, and period filters."
            ) from exc
        except Exception as exc:
            raise JobManagerError(
                "Unable to load job history. Retry the request or simplify the page, status, and period filters."
            ) from exc

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """
        Cancel a running or pending job.

        This method attempts to cancel a job that is currently running or pending.
        Jobs that are already completed or failed cannot be cancelled.

        Args:
            job_id (str): The unique identifier of the job to cancel

        Returns:
            Dictionary containing cancellation confirmation:
                - job_id: str, Job unique identifier
                - status: str, New job status (typically "cancelled")
                - message: str, Confirmation message
                - previous_status: str, Status before cancellation

        Raises:
            BaseError: If job not found, already completed, or API error occurs

        Example:
            >>> job_manager = JobManager()
            >>> result = job_manager.cancel_job("job_12345")
            >>> print(result["status"])
            'cancelled'
        """
        try:
            import requests
        except ImportError:
            raise JobManagerError(
                "requests is required. Install it with `pip install requests`."
            )
        try:
            job_info = self.get_job_status(job_id)
            if not job_info:
                raise JobManagerError(f"Job {job_id} not found.")
            current_status = job_info.get("status", "unknown")
            if current_status.lower() in [
                "completed",
                "success",
                "failed",
                "cancelled",
            ]:
                raise JobManagerError(
                    f"Cannot cancel job {job_id} because it is already {current_status}."
                )
            url = get_api_url(f"/api/jobs/{job_id}/cancel")
            headers = self._get_auth_header()
            response = requests.post(url, headers=headers, timeout=HTTP_TIMEOUT)
            if response.status_code == 404:
                raise JobManagerError(f"Job {job_id} not found.")
            if response.status_code == 409:
                raise JobManagerError(
                    f"Job {job_id} cannot be cancelled in its current state."
                )
            response.raise_for_status()
            updated_job_info = self.get_job_status(job_id)
            new_status = (
                updated_job_info.get("status", "unknown")
                if updated_job_info
                else "unknown"
            )
            logger.info(
                "Job %s cancelled successfully (was %s, now %s)",
                job_id,
                current_status,
                new_status,
            )
            return {
                "job_id": job_id,
                "status": new_status,
                "message": f"Job {job_id} cancelled successfully",
                "previous_status": current_status,
            }
        except JobManagerError:
            raise
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500:
                raise JobManagerError(
                    "The server encountered an internal error. Please try again later or contact support."
                ) from exc
            else:
                raise JobManagerError(
                    f"Unable to cancel job {job_id}. Check that the job exists and is still running or queued."
                ) from exc
        except requests.exceptions.RequestException as exc:
            raise JobManagerError(
                f"Unable to cancel job {job_id}. Check that the job exists and is still running or queued."
            ) from exc
        except Exception as exc:
            raise JobManagerError(
                f"Unable to cancel job {job_id}. Check that the job exists and is still running or queued."
            ) from exc

    def delete_job(self, job_id: str) -> dict[str, Any]:
        """
        Delete a job from the system.

        This method permanently deletes a job and all its associated data.
        Use with caution as this operation cannot be undone.

        Args:
            job_id (str): The unique identifier of the job to delete

        Returns:
            Dictionary containing deletion confirmation:
                - job_id: str, Job unique identifier that was deleted
                - message: str, Confirmation message
                - deleted: bool, Whether deletion was successful

        Raises:
            BaseError: If job not found, deletion not allowed, or API error occurs

        Example:
            >>> job_manager = JobManager()
            >>> result = job_manager.delete_job("job_12345")
            >>> print(result["deleted"])
            True
        """
        try:
            import requests
        except ImportError:
            raise JobManagerError(
                "requests is required. Install it with `pip install requests`."
            )
        try:
            job_info = self.get_job_status(job_id)
            if not job_info:
                raise JobManagerError(f"Job {job_id} not found.")
            url = get_api_url(f"/api/jobs/{job_id}")
            headers = self._get_auth_header()
            response = requests.delete(url, headers=headers, timeout=HTTP_TIMEOUT)
            if response.status_code == 404:
                raise JobManagerError(f"Job {job_id} not found.")
            if response.status_code == 403:
                raise JobManagerError(
                    f"You do not have permission to delete job {job_id}."
                )
            if response.status_code == 409:
                raise JobManagerError(
                    f"Job {job_id} cannot be deleted in its current state."
                )
            response.raise_for_status()
            logger.info("Job %s deleted successfully", job_id)
            return {
                "job_id": job_id,
                "message": f"Job {job_id} deleted successfully",
                "deleted": True,
            }
        except JobManagerError:
            raise
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500:
                raise JobManagerError(
                    "The server encountered an internal error. Please try again later or contact support."
                ) from exc
            else:
                raise JobManagerError(
                    f"Unable to delete job {job_id}. Check that the job exists and is already finished or cancelled."
                ) from exc
        except requests.exceptions.RequestException as exc:
            raise JobManagerError(
                f"Unable to delete job {job_id}. Check that the job exists and is already finished or cancelled."
            ) from exc
        except Exception as exc:
            raise JobManagerError(
                f"Unable to delete job {job_id}. Check that the job exists and is already finished or cancelled."
            ) from exc

    def get_job_results(self, job_id: str, device_name: str | None = None) -> JobResult:
        """Retrieve a completed job with decoded artifacts and execution time in seconds.

        The job endpoint is read once. JSON and stored artifacts use the same
        conversion as submission/wait, including job context and metadata.
        """
        try:
            job_info = self.get_job_status(job_id)
            if not job_info:
                raise JobManagerError(f"Job {job_id} not found.")
            status = str(job_info.get("status") or "unknown")
            if status.lower() not in {"completed", "success"}:
                if status.lower() in {"failed", "error", "cancelled"}:
                    raise JobManagerError(
                        f"Job {job_id} {status.lower()}. Results are unavailable."
                    )
                raise JobManagerError(
                    f"Job {job_id} is {status}. Results are not yet available."
                )
            return self._parse_job_results(
                job_info,
                circuit_name=job_info.get("name") or f"Job_{job_id}",
                shots=job_info.get("shots") or 0,
                device_name=device_name,
                job_id=job_id,
            )
        except (JobManagerError, InvalidResponseError):
            raise
        except Exception as exc:
            raise InvalidResponseError(
                f"Results for job {job_id} could not be loaded. Retry the request or contact support if this persists."
            ) from exc

    def _extract_s3_result_reference(
        self, response_data: Any
    ) -> tuple[str, dict[str, Any] | None] | None:
        """Recursively find an S3 result path and its binary manifest."""
        if isinstance(response_data, str) and response_data.startswith("s3://"):
            return response_data, None

        if isinstance(response_data, dict):
            if (
                response_data.get("large_file")
                and response_data.get("bucket")
                and response_data.get("key")
            ):
                s3_path = f"s3://{response_data['bucket']}/{response_data['key']}"
                return s3_path, response_data

            for key in ["s3_path", "s3_uri", "s3_url"]:
                value = response_data.get(key)
                if isinstance(value, str) and value.startswith("s3://"):
                    manifest = (
                        response_data
                        if "format" in response_data or "artifacts" in response_data
                        else None
                    )
                    return value, manifest

            for nested_key in [
                "response_data",
                "server_response",
                "result",
                "results",
                "data",
            ]:
                nested_value = response_data.get(nested_key)
                nested_reference = self._extract_s3_result_reference(nested_value)
                if nested_reference:
                    return nested_reference

        return None

    def _extract_s3_path_from_response(self, response_data: Any) -> str | None:
        """Return only the S3 path from a nested result response."""
        reference = self._extract_s3_result_reference(response_data)
        return reference[0] if reference else None

    @staticmethod
    def _decode_binary_artifact_payload(
        raw_bytes: bytes, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate and decode a versioned multi-artifact binary payload."""
        try:
            counts, execution_time, artifacts = validate_binary_manifest(
                len(raw_bytes), manifest
            )
        except ContractValidationError as exc:
            raise InvalidResponseError(str(exc)) from exc

        decoded_artifacts: dict[str, Any] = {}
        for artifact in artifacts:
            end_offset = artifact.offset_bytes + artifact.size_bytes
            complex_values = np.frombuffer(
                raw_bytes[artifact.offset_bytes : end_offset], dtype=np.dtype("<c16")
            )
            if artifact.result_type == "statevector":
                decoded_artifacts[artifact.result_type] = np.reshape(
                    complex_values, (-1, 1)
                ).tolist()
            else:
                decoded_artifacts[artifact.result_type] = np.reshape(
                    complex_values, artifact.shape
                ).tolist()

        result_data = {"counts": counts, **decoded_artifacts}
        return {
            "results": {"data": result_data},
            **result_data,
            "execution_time": execution_time,
            "message": "Job completed successfully",
        }

    def _download_s3_result_via_backend(
        self,
        s3_path: str,
        manifest: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            import requests
        except ImportError:
            raise JobManagerError(
                "Failed to import requests, please install requests using `pip install requests`"
            )

        headers = self._get_auth_header()
        download_url = get_api_url("/api/storage/download/by-s3-path")

        try:
            response = requests.post(
                download_url,
                json={"s3_path": s3_path, **({"job_id": job_id} if job_id else {})},
                headers=headers,
                timeout=HTTP_TIMEOUT,
            )
            response.raise_for_status()
            presigned_data = response.json()
        except requests.exceptions.RequestException as e:
            raise JobManagerError(
                f"Failed to get presigned URL for S3 path '{s3_path}': {e}"
            )

        presigned_url = (
            presigned_data.get("presigned_url")
            if isinstance(presigned_data, dict)
            else None
        )
        if not presigned_url:
            raise InvalidResponseError(
                f"Invalid presigned URL response for S3 path '{s3_path}': {presigned_data}"
            )

        try:
            file_response = requests.get(presigned_url, timeout=HTTP_TIMEOUT)
            file_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise JobManagerError(
                f"Failed to download S3 result file for '{s3_path}': {e}"
            )

        content_type = file_response.headers.get("Content-Type", "").lower()
        if (
            "application/octet-stream" in content_type
            or "application/x-qpiai-binary" in content_type
            or s3_path.endswith(".bin")
        ):
            if manifest is None:
                raise InvalidResponseError(
                    "The binary job result is missing its artifact manifest. "
                    "Retry the job or contact support if this persists."
                )
            return self._decode_binary_artifact_payload(file_response.content, manifest)

        try:
            return file_response.json()
        except ValueError:
            try:
                return json.loads(file_response.text)
            except (json.JSONDecodeError, TypeError) as e:
                raise InvalidResponseError(
                    f"Downloaded S3 result file is not valid JSON for '{s3_path}': {e}"
                )

    def _resolve_job_response_data(
        self, response_data: Any, job_id: str | None = None
    ) -> Any:
        parsed_data = response_data

        if isinstance(parsed_data, str):
            if parsed_data.startswith("data:"):
                parsed_data = parsed_data[5:].strip()
            try:
                parsed_data = json.loads(parsed_data)
            except (json.JSONDecodeError, TypeError):
                pass

        s3_reference = self._extract_s3_result_reference(parsed_data)
        if s3_reference:
            s3_path, manifest = s3_reference
            return self._download_s3_result_via_backend(
                s3_path,
                manifest=manifest,
                job_id=job_id,
            )

        return parsed_data

    def _parse_simulator_response(self, response_data: dict) -> dict:
        """
        Parse simulator response format: results.data.counts, results.data.statevector, etc.

        Args:
            response_data: Server response dictionary from simulator

        Returns:
            Dictionary with extracted quantum results and metadata
        """
        return parse_circuit_response(response_data)

    def _parse_job_results(
        self,
        job_result: dict[str, Any],
        circuit_name: str,
        shots: int,
        device_name: str | None = None,
        need_statevector: bool = False,
        need_density_matrix: bool = False,
        *,
        job_id: str | None = None,
    ) -> JobResult:
        """Convert a stored job into the result shared by retrieval and submit/wait."""
        job_id = (
            job_id
            or job_result.get("id")
            or job_result.get("job_id")
            or job_result.get("proxy_id")
        )
        response_data = None
        for field in ("response_data", "server_response"):
            candidate = job_result.get(field)
            if candidate is not None and (
                not (isinstance(candidate, str) and (not candidate.strip()))
            ):
                response_data = candidate
                break
        if response_data is None:
            raise InvalidResponseError(
                f"Job {job_id} completed but no results data are available."
            )
        resolved_response = self._resolve_job_response_data(
            response_data, job_id=job_id
        )
        if not isinstance(resolved_response, dict):
            raise InvalidResponseError(
                "The job returned results in an unsupported format. Retry the request or contact support if this persists."
            )
        parsed = parse_circuit_response(resolved_response)
        counts = parsed["counts"]
        execution_time = decode_execution_time_seconds(parsed["execution_time"])
        if execution_time is None:
            execution_time = decode_execution_time_seconds(
                job_result.get("execution_time")
            )
        if execution_time is None:
            legacy_ms = decode_execution_time_seconds(
                job_result.get("execution_time_ms")
            )
            execution_time = legacy_ms / 1000.0 if legacy_ms is not None else 0.0
        name = job_result.get("name") or parsed.get("name") or circuit_name
        method = job_result.get("method") or parsed.get("method") or "unknown"
        credits_used = job_result.get("credits_used", 0)
        job_metadata = {
            "server_response": resolved_response,
            "status": job_result.get("status"),
            "circuit_id": job_result.get("circuit_id"),
            "experiment_id": job_result.get("experiment_id"),
            "experiment_name": job_result.get("experiment_name"),
            "compute_resource_name": job_result.get("compute_resource_name"),
            "server_response_id": job_result.get("server_response_id"),
            "method": method,
            "credits_used": credits_used,
            "created_at": job_result.get("created_at"),
            "updated_at": job_result.get("updated_at"),
            "device_name": device_name or job_result.get("device_name"),
            "circuit_info": job_result.get("circuit_info")
            or parsed.get("circuit_info")
            or {
                "name": name,
                "qubits": parsed.get("num_qubits") or 0,
                "depth": parsed.get("depth") or 0,
            },
            "timing": {
                "created_at": job_result.get("created_at"),
                "started_at": job_result.get("started_at"),
                "completed_at": job_result.get("completed_at")
                or job_result.get("updated_at"),
            },
        }
        response_metadata = dict(parsed["metadata"])
        if parsed.get("job_id"):
            response_metadata["job_id"] = parsed["job_id"]
        if response_metadata:
            job_metadata["qpu_metadata"] = response_metadata
        metrics: dict[str, Any] = {
            field: job_result.get(field)
            if job_result.get(field) is not None
            else parsed.get(field) or 0.0
            for field in ("cpu_usage", "memory_usage", "gpu_memory_usage")
        }
        return JobResult(
            name=name,
            counts=counts,
            statevector=parsed["statevector"],
            density_matrix=parsed["density_matrix"],
            probabilities=parsed["probabilities"],
            message=parsed["message"]
            or job_result.get("message")
            or "Job completed successfully",
            execution_time=execution_time,
            shots=sum(counts.values()) if counts else shots,
            job_id=job_id,
            job_status=job_result.get("status"),
            method=method,
            credits_used=credits_used,
            job_metadata=job_metadata,
            **metrics,
        )

    def submit_and_wait_for_results_qasm(
        self,
        qasm_string_or_circuit: str | Circuit | IntermediateCircuitRepresentation,
        experiment_name: str = "Default Experiment",
        shots: int = 1024,
        method: str | None = None,
        need_statevector: bool = False,
        need_density_matrix: bool = False,
        device_name: str = "QpiAI-QSV-Simulator",
        circuit_name: str = "circuit",
        compute_resource_id: str | None = None,
        overwrite: bool = False,
        use_events: bool = True,
        timeout: int = 300,
        access_token: str | None = None,
    ) -> JobResult:
        """
        All-in-one QASM/Circuit submission that waits for completion and returns job results.

        This method combines job submission, status monitoring, and results retrieval
        into a single convenient function call with SSE support for real-time status updates.

        Args:
            qasm_string_or_circuit (str | Circuit | IntermediateCircuitRepresentation):
                                    OpenQASM 2.0 circuit description OR Circuit/ICR object
            experiment_name (str): Name of experiment to associate with (default: "Default Experiment")
                                Note: Experiment must exist - create experiment first
            shots (int): Number of shots to execute (default: 1024)
            method (str): Execution method (default: "statevector")
            device_name (str): Device name (default: "QpiAI-QSV-Simulator")
            circuit_name (str): Circuit name (default: "circuit")
            compute_resource_id (str): Compute resource ID (optional)
            overwrite (bool): Whether to overwrite existing circuit with same name (default: False)
            use_events (bool): Whether to use SSE for real-time status updates (default: True)
            timeout (int): Maximum time to wait for completion in seconds (default: 300)

        Returns:
            JobResult object containing job information and quantum execution results:
                - Can use .get() method to retrieve all fields as a dictionary
                - Can use .get('field_name') to retrieve specific fields
                - Can use .plot() method to visualize probability distributions
                - Includes job metadata, measurement counts, statevector, and execution statistics

        Raises:
            BaseError: If submission fails, job doesn't complete, timeout occurs, or experiment does not exist
        """
        import time

        if isinstance(qasm_string_or_circuit, str):
            qasm_string = qasm_string_or_circuit
        else:
            qasm_string = self._convert_circuit_to_qasm(qasm_string_or_circuit)
        request = resolve_execution_request(
            device_name,
            method=method,
            need_statevector=need_statevector,
            need_density_matrix=need_density_matrix,
        )
        profile = request.device
        method = profile.method
        need_statevector = request.need_statevector
        need_density_matrix = request.need_density_matrix
        sse_handler = (
            SSEResultHandler(device_name=device_name, timeout=timeout)
            if use_events and SSE_AVAILABLE
            else None
        )
        try:
            submission_result = self.submit_qasm_job(
                qasm_string=qasm_string,
                experiment_name=experiment_name,
                shots=shots,
                method=method,
                need_statevector=need_statevector,
                need_density_matrix=need_density_matrix,
                device_name=device_name,
                circuit_name=circuit_name or "Circuit",
                compute_resource_id=compute_resource_id,
                overwrite=overwrite,
            )
            job_id = submission_result.get("job_id")
            if not job_id:
                raise JobSubmissionError(
                    "Job submission failed because the server did not return a job ID. Retry the submission and contact support if this persists."
                )
            start_time = time.time()
            if use_events and sse_handler:
                sse_url = get_sse_url(job_id)
                headers = self._get_auth_header()
                if sse_handler.wait_for_events(sse_url, headers):
                    result = self.get_job_status(job_id)
                else:
                    logger.warning(
                        "SSE monitoring failed; falling back to polling: %s",
                        sse_handler.get_error() or "no completion event received",
                    )
                    poll_res = self._wait_with_polling(
                        job_id,
                        timeout,
                        poll_interval=2,
                        start_time=start_time,
                        initial_poll_count=0,
                    )
                    result = poll_res.get("job_info") or self.get_job_status(job_id)
            else:
                poll_res = self._wait_with_polling(
                    job_id,
                    timeout,
                    poll_interval=2,
                    start_time=start_time,
                    initial_poll_count=0,
                )
                result = poll_res.get("job_info") or self.get_job_status(job_id)
            if not result:
                raise JobManagerError(
                    f"Unable to retrieve the final status for job {job_id}. Retry the status check or call get_job_status(job_id) manually."
                )
            job_result = self._parse_job_results(
                result,
                circuit_name,
                shots,
                device_name,
                need_statevector=need_statevector,
                need_density_matrix=need_density_matrix,
                job_id=job_id,
            )
            if job_result.job_metadata is None:
                job_result.job_metadata = {}
            job_result.job_metadata["submission_params"] = {
                "shots": shots,
                "circuit_name": circuit_name,
                "experiment_name": experiment_name,
                "overwrite": overwrite,
                "use_events": use_events,
                "timeout": timeout,
            }
            return job_result
        except (JobManagerError, JobSubmissionError):
            raise
        except Exception as exc:
            raise JobManagerError(
                "Unable to monitor job completion. Retry the run or check the job later with get_job_status(job_id)."
            ) from exc

    def _wait_with_polling(
        self,
        job_id: str,
        timeout: int,
        poll_interval: int,
        start_time: float,
        initial_poll_count: int,
    ) -> dict[str, Any]:
        """
        Helper method to wait for job completion using HTTP polling.

        Args:
            job_id (str): Job ID to monitor
            timeout (int): Maximum wait time in seconds
            poll_interval (int): Time between polls in seconds
            start_time (float): Start time for timeout calculation
            initial_poll_count (int): Starting poll count

        Returns:
            Dict with at minimum {"status": "completed", "job_id": job_id}
        """
        try:
            poll_count = initial_poll_count
            while True:
                poll_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    raise JobManagerError(
                        f"Job {job_id} did not complete within {timeout} seconds."
                    )
                job_info = self.get_job_status(job_id)
                if not job_info:
                    raise JobManagerError(
                        f"Job {job_id} was not found while checking its status."
                    )
                current_status = job_info.get("status", "unknown")
                logger.debug("Poll %s: job status = %s", poll_count, current_status)
                if current_status and current_status.lower() in [
                    "completed",
                    "success",
                ]:
                    logger.info("Job completed successfully after %.1fs", elapsed_time)
                    return {
                        "status": current_status,
                        "job_id": job_id,
                        "job_info": job_info,
                        "polling_checks": poll_count,
                    }
                if current_status and current_status.lower() == "failed":
                    raise JobManagerError(f"Job {job_id} failed during execution.")
                if not current_status or current_status.lower() not in [
                    "pending",
                    "running",
                    "scheduled",
                    "submitted",
                    "queued",
                ]:
                    raise JobManagerError(
                        f"Job {job_id} entered an unexpected status: {current_status}."
                    )
                time.sleep(poll_interval)
        except JobManagerError:
            raise
        except Exception as exc:
            raise JobManagerError(
                f"Unable to keep polling job {job_id}. Retry the status check manually with get_job_status(job_id)."
            ) from exc
