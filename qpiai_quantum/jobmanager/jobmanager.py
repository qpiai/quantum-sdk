import json
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np  # type: ignore

from ..authentication.user import get_user
from ..circuit.circuit import Circuit
from ..config import get_api_url, get_sse_url
from ..icr.icr import IntermediateCirucitRepresentation
from .exceptions import (
    AuthenticationError,
    CircuitNotFoundError,
    CircuitUpdateError,
    ExperimentNotFoundError,
    InvalidExperimentResponseError,
    InvalidResponseError,
    JobManagerError,
    JobStatusError,
    JobSubmissionError,
)
from .job_result import JobResult
from .sse_handler import SSEResultHandler, SSE_AVAILABLE

logger = logging.getLogger("qpiai_quantum.jobmanager")


class JobManager:
    """
    QPIAI Job Manager for handling quantum job lifecycle management.

    Provides functions for submitting, monitoring, and retrieving results
    from quantum circuit execution jobs on QPIAI backends.
    """

    @staticmethod
    def track_job(func, jobName: str):
        """
        A decorator to track the job by its name.

        Args:
            func: Function to wrap
            jobName (str): Name of the job for tracking

        Returns:
            Wrapped function that logs job tracking
        """

        def wrapper(*args, **kwargs):
            print(f"Tracking job: {jobName}")
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
        user = get_user()
        if not user or not user.api_key:
            logger.error("No API key in user context. Login required.")
            raise AuthenticationError()
        return {"X-Secret-Token": user.api_key}

    def _get_compute_resource_id(self, method: str, device_name: str) -> Optional[str]:
        """
        Get compute resource ID based on method and device_name.
        Maps to the appropriate backend resource name.

        Args:
            method (str): Execution method ("statevector", "density_matrix", "tensor_network",
                         or variants like "statevector_sampling")
            device_name (str): Device name ("QpiAI-QSV-Simulator", "QpiAI-QDM-Simulator", "QpiAI-QTN-Simulator", "QpiAI-Indus-1", "QpiAI-QSV-Lite", "QpiAI-QDM-Lite")

        Returns:
            Optional[str]: Compute resource ID (UUID string), or None if not found
        """
        try:
            import requests
        except ImportError:
            logger.warning("requests not available, cannot fetch compute resource")
            return None

        resource_name_map = {
            ("statevector", "QpiAI-QSV-Simulator"): "QpiAI-QSV-Simulator",
            ("density_matrix", "QpiAI-QDM-Simulator"): "QpiAI-QDM-Simulator",
            ("tensor_network", "QpiAI-QTN-Simulator"): "QpiAI-QTN-Simulator",
            ("statevector", "QpiAI-Indus-1"): "QpiAI-Indus-1",
            ("statevector_sampling", "QpiAI-QSV-Lite"): "QpiAI-QSV-Lite",
            ("density_matrix", "QpiAI-QDM-Lite"): "QpiAI-QDM-Lite",
        }
        # Try with normalized values
        resource_name = resource_name_map.get((method, device_name))

        if not resource_name:
            logger.warning(
                f"No compute resource mapping for method={method}), "
                f"device_name={device_name})"
            )
            return None

        try:
            # Fetch compute resources from API
            url = get_api_url("/api/compute-resources")
            headers = self._get_auth_header()
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                resources = response.json()
                # API returns list of resources with structure:
                # [{'ID': 'uuid', device_name: 'QpiAI-QSV-Simulator', ...}, ...]
                if isinstance(resources, list):
                    for resource in resources:
                        backend_name = resource.get("backend_name")
                        if backend_name == resource_name:
                            resource_id = resource.get("ID")
                            logger.debug(
                                f"Found compute resource: {backend_name} -> {resource_id}"
                            )
                            return resource_id
                elif isinstance(resources, dict) and "compute_resources" in resources:
                    for resource in resources["compute_resources"]:
                        backend_name = resource.get("backend_name")
                        if backend_name == resource_name:
                            resource_id = resource.get("ID")
                            logger.debug(
                                f"Found compute resource: {backend_name} -> {resource_id}"
                            )
                            return resource_id

            logger.warning(f"Could not find compute resource ID for {resource_name}")
            return None

        except Exception as e:
            logger.warning(f"Error fetching compute resources: {e}")
            return None

    def list_compute_resources(self) -> List[Dict[str, Any]]:
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
                "Failed to import requests, please install requests using `pip install requests`"
            )

        try:
            url = get_api_url("/api/compute-resources")
            headers = self._get_auth_header()
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            resources = response.json()
            if isinstance(resources, list):
                return resources
            elif isinstance(resources, dict) and "compute_resources" in resources:
                return resources["compute_resources"]
            else:
                return []

        except requests.exceptions.RequestException as e:
            raise JobManagerError(f"Failed to list compute resources: {e}")

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
                "Failed to import requests, please install requests using `pip install requests`"
            )

        try:
            url = get_api_url(f"/api/experiments/by-name?name={experiment_name}")
            headers = self._get_auth_header()
            response = requests.get(url, headers=headers)

            if response.status_code == 404:
                raise ExperimentNotFoundError(experiment_name)

            if response.status_code == 403:
                error_msg = (
                    f"Access denied (403 Forbidden) for experiment '{experiment_name}'.\n"
                    "Possible causes:\n"
                    "  1. Your API key doesn't have permission to access this experiment\n"
                    "  2. The experiment belongs to a different user/account\n"
                    "  3. Your API key is invalid or expired\n"
                    "  4. The experiment doesn't exist - create it in the QpiAI web UI first\n\n"
                    "Please verify:\n"
                    "  - Your API key is correct and active\n"
                    "  - The experiment exists and you have access to it\n"
                    "  - You created the experiment in the QpiAI platform UI"
                )
                raise JobManagerError(error_msg)

            response.raise_for_status()

            experiment_data = response.json()
            experiment_id = experiment_data.get("experiment_id")

            if not experiment_id:
                raise InvalidExperimentResponseError()

            return experiment_id

        except requests.exceptions.RequestException as e:
            if "403" in str(e):
                # Already handled above, re-raise
                raise
            raise JobManagerError(
                f"Failed to get experiment ID for '{experiment_name}': {e}"
            )

    def _convert_circuit_to_qasm(self, circuit) -> str:
        """
        Convert Circuit or IntermediateCirucitRepresentation object to OpenQASM string.

        Args:
            circuit: Circuit or IntermediateCirucitRepresentation object

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
        elif isinstance(circuit, IntermediateCirucitRepresentation):
            temp_circuit = Circuit()
            temp_circuit.icr = circuit
            qasm_result = temp_circuit.to_qasm()
            if isinstance(qasm_result, list):
                return "\n".join(qasm_result)
            return qasm_result
        else:
            raise JobManagerError(
                f"Cannot convert circuit of type {type(circuit)}, expected a Circuit or IntermediateCirucitRepresentation object."
            )

    def submit_qasm_job(
        self,
        qasm_string: str,
        experiment_name: str = "Default Experiment",
        shots: int = 1024,
        method: str = "statevector",
        need_statevector: bool = False,
        need_density_matrix: bool = False,
        device_name: str = "QpiAI-QSV-Simulator",
        circuit_name: str = "circuit",
        compute_resource_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
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
        try:
            import requests
        except ImportError:
            raise JobManagerError(
                "Failed to import requests, please install requests using `pip install requests`"
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

            # Auto-determine compute_resource_id if not provided
            if compute_resource_id is None:
                compute_resource_id = self._get_compute_resource_id(method, device_name)
                if compute_resource_id:
                    logger.debug(
                        f"Auto-selected compute resource: {compute_resource_id}"
                    )

            headers = self._get_auth_header()

            circuit_url = get_api_url("/api/circuits/create")

            circuit_payload = {
                "openqasm_string": qasm_string,
                "shots": shots,
                "name": circuit_name,
                "device_name": associated_device_name,
                "source_type": "sdk",
            }
            logger.debug(f"Creating circuit with payload: {circuit_payload}")

            circuit_response = requests.post(
                circuit_url, json=circuit_payload, headers=headers
            )

            if circuit_response.status_code == 409:
                if not overwrite:
                    raise JobManagerError(
                        f"Circuit '{circuit_name}' already exists (HTTP 409 Conflict). "
                        f"To overwrite the existing circuit, set overwrite=True."
                    )

                logger.info(
                    f"Circuit '{circuit_name}' already exists, attempting to update..."
                )

                try:
                    list_url = get_api_url("/api/circuits/?page=1&page_size=100")
                    if experiment_id:
                        list_url += f"&experiment_id={experiment_id}"

                    list_response = requests.get(list_url, headers=headers)
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
                            f"Found existing circuit '{circuit_name}' with ID: {circuit_id}"
                        )

                        update_url = get_api_url(f"/api/circuits/{circuit_id}")
                        update_payload = {
                            "qasm": qasm_string,
                            "name": circuit_name,
                            "device_name": associated_device_name,
                            "shots": shots,
                            "method": method,
                            "need_statevector": need_statevector,
                            "need_density_matrix": need_density_matrix,
                        }
                        circuit_update_response = requests.put(
                            update_url, json=update_payload, headers=headers
                        )
                        circuit_update_response.raise_for_status()
                    else:
                        raise CircuitNotFoundError(circuit_name)

                except Exception as update_error:
                    raise CircuitUpdateError(update_error)
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
            # Only include compute_resource_id if it's provided

            if compute_resource_id is not None:
                job_payload["compute_resource_id"] = compute_resource_id

            job_response = requests.post(job_url, json=job_payload, headers=headers)

            # Enhanced error handling for 400 Bad Request
            if job_response.status_code == 400:
                try:
                    error_detail = job_response.json()
                    error_msg = (
                        f"Bad Request (400) when submitting job.\n"
                        f"Server response: {error_detail}\n\n"
                        "Common causes:\n"
                        "  1. You may not have access to the specified compute resource\n"
                        "     (if part of an organization, contact your administrator)\n"
                        "  2. Invalid circuit QASM format\n"
                        "  3. Invalid method or device_name value\n"
                        "  4. Missing or invalid circuit_id or experiment_id\n"
                        "  5. Invalid shots value (must be positive integer)\n\n"
                        f"Request payload:\n"
                        f"  - circuit_id: {circuit_id}\n"
                        f"  - experiment_id: {experiment_id}\n"
                        f"  - method: {method}\n"
                        f"  - device_name: {device_name}\n"
                        f"  - shots: {shots}\n"
                        f"  - compute_resource_id: {compute_resource_id}"
                    )
                except Exception:
                    error_msg = f"Bad Request (400) when submitting job. Response: {job_response.text}"
                raise JobSubmissionError(error_msg)

            job_response.raise_for_status()
            job_data = job_response.json()

            if "job" in job_data:
                job_info = job_data["job"]
                job_id = job_info.get("id")
                job_status = job_info.get("status", "submitted")
                job_info.get("created_at")
            else:
                job_id = job_data.get("id")
                job_status = job_data.get("status", "submitted")
                job_data.get("created_at")

            if not job_id:
                raise JobSubmissionError()

            return {
                "job_id": job_id,
                "status": job_status,
                "experiment_id": experiment_id,
                "circuit_id": circuit_id,
            }

        except requests.exceptions.RequestException as e:
            raise JobSubmissionError(f"Failed to submit QASM job: {e}")
        except Exception as e:
            raise JobManagerError(f"Error during QASM job submission: {e}")

    def submit_circuit_job(
        self,
        circuit: Circuit | IntermediateCirucitRepresentation,
        experiment_name: str = "Default Experiment",
        shots: int = 1024,
        method: str = "statevector",
        need_statevector: bool = False,
        need_density_matrix: bool = False,
        device_name: str = "QpiAI-QSV-Simulator",
        circuit_name: Optional[str] = None,
        compute_resource_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Submit Circuit or IntermediateCirucitRepresentation for execution.

        This method provides the same functionality as submit_qasm_job() but accepts
        Circuit or ICR objects directly, converting them to QASM automatically.

        Args:
            circuit (Circuit | IntermediateCirucitRepresentation): Circuit or ICR object to execute
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
            circuit_name=circuit_name or "circuit",
            compute_resource_id=compute_resource_id,
            overwrite=overwrite,
        )

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
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
                - execution_time_ms: int, Execution time in milliseconds
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
                "Failed to import requests, please install requests using `pip install requests`"
            )
        try:
            import requests
            import json

            url = get_api_url(f"/api/jobs/{job_id}")
            headers = self._get_auth_header()
            response = requests.get(url, headers=headers, stream=True)
            if response.status_code == 404:
                logger.info(f"Job {job_id} not found")
                return None
            response.raise_for_status()
            chunks = []
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
            return json.loads(b"".join(chunks))
        except requests.exceptions.RequestException as e:
            raise JobStatusError(job_id, e)
        except Exception as e:
            raise JobManagerError(f"Unexpected error while getting job status: {e}")

    def get_current_job(self) -> Optional[Dict[str, Any]]:
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
                "Failed to import requests, please install requests using `pip install requests`"
            )

        try:
            # Query for currently running jobs
            url = get_api_url("/api/jobs/user?status=running&page_size=1")
            headers = self._get_auth_header()
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            jobs = data.get("jobs", [])

            if not jobs:
                logger.info("No currently running job found")
                return None

            # Return the first (and only) running job
            return jobs[0]

        except requests.exceptions.RequestException as e:
            raise JobManagerError(f"Failed to get current job: {e}")
        except Exception as e:
            raise JobManagerError(f"Unexpected error while getting current job: {e}")

    def get_job_history(
        self,
        period: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
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
            import requests  # type: ignore
        except ImportError:
            raise JobManagerError(
                "Failed to import requests, please install requests using `pip install requests`"
            )

        try:
            # Validate parameters
            if page < 1:
                raise JobManagerError("Page number must be >= 1")
            if page_size < 1 or page_size > 100:
                raise JobManagerError("Page size must be between 1 and 100")

            if period:
                valid_periods = ["daily", "weekly", "monthly"]
                if period not in valid_periods:
                    raise JobManagerError(f"Period must be one of: {valid_periods}")

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
                    raise JobManagerError(f"Status must be one of: {valid_statuses}")

            # Build query parameters
            params: Dict[str, Any] = {"page": page, "page_size": page_size}

            if period:
                params["period"] = period
            if status:
                params["status"] = status

            # Convert params dict to query string
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            url = get_api_url(f"/api/jobs/user?{query_string}")

            headers = self._get_auth_header()
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()

            # Ensure expected structure
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

        except requests.exceptions.RequestException as e:
            raise JobManagerError(f"Failed to get job history: {e}")
        except Exception as e:
            raise JobManagerError(f"Unexpected error while getting job history: {e}")

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
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
                "Failed to import requests, please install requests using `pip install requests`"
            )

        try:
            # First check if job exists and get its current status
            job_info = self.get_job_status(job_id)
            if not job_info:
                raise JobManagerError(f"Job {job_id} not found")

            current_status = job_info.get("status", "unknown")

            # Check if job can be cancelled
            if current_status.lower() in [
                "completed",
                "success",
                "failed",
                "cancelled",
            ]:
                raise JobManagerError(
                    f"Cannot cancel job {job_id}: Job is already {current_status}. "
                    f"Only pending, running, or scheduled jobs can be cancelled."
                )

            # Send cancellation request
            url = get_api_url(f"/api/jobs/{job_id}/cancel")
            headers = self._get_auth_header()
            response = requests.post(url, headers=headers)

            if response.status_code == 404:
                raise JobManagerError(f"Job {job_id} not found")
            elif response.status_code == 409:
                # Job might already be in a non-cancellable state
                raise JobManagerError(
                    f"Job {job_id} cannot be cancelled (HTTP 409). "
                    f"It may have already completed or is in a non-cancellable state."
                )

            response.raise_for_status()

            # Get updated job status to confirm cancellation
            updated_job_info = self.get_job_status(job_id)
            new_status = (
                updated_job_info.get("status", "unknown")
                if updated_job_info
                else "unknown"
            )

            logger.info(
                f"Job {job_id} cancelled successfully (was {current_status}, now {new_status})"
            )

            return {
                "job_id": job_id,
                "status": new_status,
                "message": f"Job {job_id} cancelled successfully",
                "previous_status": current_status,
            }

        except requests.exceptions.RequestException as e:
            raise JobManagerError(f"Failed to cancel job {job_id}: {e}")
        except Exception as e:
            raise JobManagerError(
                f"Unexpected error while cancelling job {job_id}: {e}"
            )

    def delete_job(self, job_id: str) -> Dict[str, Any]:
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
                "Failed to import requests, please install requests using `pip install requests`"
            )

        try:
            # First check if job exists
            job_info = self.get_job_status(job_id)
            if not job_info:
                raise JobManagerError(f"Job {job_id} not found")

            # Send deletion request
            url = get_api_url(f"/api/jobs/{job_id}")
            headers = self._get_auth_header()
            response = requests.delete(url, headers=headers)

            if response.status_code == 404:
                raise JobManagerError(f"Job {job_id} not found")
            elif response.status_code == 403:
                raise JobManagerError(
                    f"Permission denied: Cannot delete job {job_id}. "
                    f"You may not have permission to delete this job."
                )
            elif response.status_code == 409:
                raise JobManagerError(
                    f"Job {job_id} cannot be deleted (HTTP 409). "
                    f"It may be currently running or in a protected state."
                )

            response.raise_for_status()

            logger.info(f"Job {job_id} deleted successfully")

            return {
                "job_id": job_id,
                "message": f"Job {job_id} deleted successfully",
                "deleted": True,
            }

        except requests.exceptions.RequestException as e:
            raise JobManagerError(f"Failed to delete job {job_id}: {e}")
        except Exception as e:
            raise JobManagerError(f"Unexpected error while deleting job {job_id}: {e}")

    def get_job_results(
        self, job_id: str, device_name: Optional[str] = None
    ) -> "JobResult":
        """
        Retrieve quantum computation results for a completed job.

        Args:
            job_id (str): The unique identifier of the job
            device_name (Optional[str]): Device name for format-specific parsing

        Returns:
            Results dictionary with measurement counts and metadata:
                - job_id: str, Job unique identifier
                - status: str, Job status (should be "completed")
                - results: dict, Quantum measurement results:
                    - counts: dict, Measurement counts for each basis state
                        - "00": int, Count for 00 state
                        - "01": int, Count for 01 state
                        - "10": int, Count for 10 state
                        - "11": int, Count for 11 state
                        - ... (for higher qubit counts)
                    - shots: int, Total number of shots executed
                    - execution_metadata: dict:
                        - execution_time_ms: int, Execution time in milliseconds
                        - device_name: str, Device name
                        - method: str, Execution method ("statevector" or "qasm")
                        - credits_used: int, Credits consumed for execution
                - circuit_info: dict, Circuit information:
                    - name: str, Circuit name
                    - qubits: int, Number of qubits
                    - depth: int, Circuit depth
                - timing: dict, Execution timing information:
                    - created_at: str, Job creation timestamp
                    - started_at: str, Job start timestamp
                    - completed_at: str, Job completion timestamp

        Raises:
            BaseError: If job not found, not completed, results unavailable, or API error occurs

        Note:
            Returns error information if job failed.
            Times out or returns partial results if job not yet completed.
        """
        try:
            import requests
        except ImportError:
            raise JobManagerError(
                "Failed to import requests, please install requests using `pip install requests`"
            )

        try:
            # First get job status to check completion
            job_info = self.get_job_status(job_id)
            if not job_info:
                raise JobManagerError(f"Job {job_id} not found")

            job_status = job_info.get("status", "unknown")

            if job_status.lower() not in ["completed", "success"]:
                if job_status.lower() == "failed":
                    raise JobManagerError(
                        f"Job {job_id} failed with error: {job_info.get('error', 'Unknown error')}"
                    )
                elif job_status.lower() in [
                    "pending",
                    "running",
                    "scheduled",
                    "success",
                ]:
                    raise JobManagerError(
                        f"Job {job_id} is still {job_status}, results not yet available"
                    )
                else:
                    raise JobManagerError(
                        f"Job {job_id} has status '{job_status}', cannot retrieve results"
                    )

            # Get job details (includes results if completed)
            url = get_api_url(f"/api/jobs/{job_id}")
            headers = self._get_auth_header()
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            job_data = response.json()

            # Extract response_data or server_response which contains the actual results
            response_data = job_data.get(
                "response_data", job_data.get("server_response")
            )
            if not response_data:
                raise InvalidResponseError(
                    f"Job {job_id} completed but no results data available"
                )

            # Parse results data (supports direct JSON or S3 path indirection)
            try:
                results_data = self._resolve_job_response_data(response_data)
            except Exception as e:
                raise InvalidResponseError(f"Failed to resolve results data: {e}")

            if not isinstance(results_data, dict):
                raise InvalidResponseError(
                    f"Resolved results data must be a dictionary, got {type(results_data).__name__}"
                )

            # Extract measurement counts with fallback for different response formats
            counts = results_data.get("counts", results_data.get("histogram", {}))

            # Handle new nested structure: results.data.counts
            if not counts and "results" in results_data:
                results_section = results_data["results"]
                if "data" in results_section:
                    data_section = results_section["data"]
                    counts = data_section.get(
                        "counts", data_section.get("histogram", {})
                    )

            # Convert statevector format to match Executor format if needed
            statevector = results_data.get("statevector")
            if (
                statevector
                and len(statevector) > 0
                and isinstance(statevector[0], dict)
                and "real" in statevector[0]
                and "imag" in statevector[0]
            ):
                # Convert from {"real": x, "imag": y} format to complex numbers
                converted_statevector = []
                for state in statevector:
                    complex_val = complex(state["real"], state["imag"])
                    converted_statevector.append([np.complex128(complex_val)])
                statevector = converted_statevector

            # Create JobResult object with rich job metadata
            execution_result = JobResult(
                name=results_data.get("name", f"Job_{job_id}"),
                counts=counts,
                statevector=statevector,
                probabilities=results_data.get("probabilities"),
                message="Job completed successfully",
                executionTime=job_data.get("execution_time_ms", 0) / 1000.0
                if job_data.get("execution_time_ms")
                else 0.0,
                shots=sum(counts.values()) if counts else 0,
                cpu_usage=job_data.get("cpu_usage", 0.0),
                memory_usage=job_data.get("memory_usage", 0.0),
                gpu_memory_usage=job_data.get("gpu_memory_usage", 0.0),
                job_id=job_id,
                job_status=job_status,
                method=results_data.get("method", job_data.get("method", "unknown")),
                credits_used=job_data.get("credits_used", 0),
            )

            # Store additional metadata in the JobResult object
            execution_result.job_metadata = {
                "device_name": job_data.get("device_name", "unknown"),
                "method": results_data.get("method", job_data.get("method", "unknown")),
                "credits_used": job_data.get("credits_used", 0),
                "circuit_info": results_data.get(
                    "circuit_info",
                    {
                        "name": results_data.get("name", "unknown"),
                        "qubits": results_data.get("num_qubits", 0),
                        "depth": results_data.get("depth", 0),
                    },
                ),
                "timing": {
                    "created_at": job_data.get("created_at"),
                    # "started_at": job_data.get("started_at"),
                    "completed_at": job_data.get("updated_at"),
                },
            }

            # Return JobResult directly for easy migration
            return execution_result

        except JobManagerError:
            raise
        except requests.exceptions.RequestException as e:
            raise JobManagerError(f"Failed to retrieve results for job {job_id}: {e}")
        except Exception as e:
            raise JobManagerError(f"Unexpected error while getting job results: {e}")

    def _extract_s3_path_from_response(self, response_data: Any) -> Optional[str]:
        if isinstance(response_data, str) and response_data.startswith("s3://"):
            return response_data

        if isinstance(response_data, dict):
            for key in ["s3_path", "s3_uri", "s3_url"]:
                value = response_data.get(key)
                if isinstance(value, str) and value.startswith("s3://"):
                    return value

            for nested_key in [
                "response_data",
                "server_response",
                "result",
                "results",
                "data",
            ]:
                nested_value = response_data.get(nested_key)
                if isinstance(nested_value, str) and nested_value.startswith("s3://"):
                    return nested_value
                if isinstance(nested_value, dict):
                    nested_s3_path = self._extract_s3_path_from_response(nested_value)
                    if nested_s3_path:
                        return nested_s3_path

        return None

    def _download_s3_result_via_backend(self, s3_path: str) -> Dict[str, Any]:
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
                json={"s3_path": s3_path},
                headers=headers,
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
            file_response = requests.get(presigned_url)
            file_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise JobManagerError(
                f"Failed to download S3 result file for '{s3_path}': {e}"
            )

        try:
            return file_response.json()
        except ValueError:
            try:
                return json.loads(file_response.text)
            except (json.JSONDecodeError, TypeError) as e:
                raise InvalidResponseError(
                    f"Downloaded S3 result file is not valid JSON for '{s3_path}': {e}"
                )

    def _resolve_job_response_data(self, response_data: Any) -> Any:
        parsed_data = response_data

        if isinstance(parsed_data, str):
            if parsed_data.startswith("data:"):
                parsed_data = parsed_data[5:].strip()
            try:
                parsed_data = json.loads(parsed_data)
            except (json.JSONDecodeError, TypeError):
                pass

        s3_path = self._extract_s3_path_from_response(parsed_data)
        if s3_path:
            return self._download_s3_result_via_backend(s3_path)

        return parsed_data

    def _parse_simulator_response(self, response_data: dict) -> dict:
        """
        Parse simulator response format: results.data.counts, results.data.statevector, etc.

        Args:
            response_data: Server response dictionary from simulator

        Returns:
            Dictionary with extracted quantum results and metadata
        """
        # Extract data from server_response.results.data
        results_obj = response_data.get("results")

        if results_obj is not None:
            # Handle case where 'results' is a list (common in some API versions)
            if isinstance(results_obj, list) and len(results_obj) > 0:
                results_data = results_obj[0]
            else:
                results_data = results_obj
            data = results_data.get("data", {})
        else:
            # results is missing, look for data at top level
            data = response_data.get("data", {})

        if not data and isinstance(response_data, dict):
            # Fallback for alternative structures
            data = response_data

        return {
            "counts": data.get("counts"),
            "statevector": data.get("statevector"),
            "density_matrix": data.get("density_matrix"),
            "probabilities": data.get("probabilities"),
            "execution_time": response_data.get("execution_time"),
            "message": response_data.get("message"),
        }

    def _parse_qpu_response(self, response_data: dict) -> dict:
        """
        Parse QPU response format with nested structure: results.results.counts

        Args:
            response_data: Server response dictionary from QPU

        Returns:
            Dictionary with extracted quantum results and QPU-specific metadata
        """
        counts = None
        qpu_metadata = {}

        if isinstance(response_data, dict):
            # Newer QPU payloads may return counts at top-level under "counts".
            counts = response_data.get("counts")

            # Backward-compatible handling for older nested structures.
            if counts is None and "results" in response_data:
                results = response_data["results"]
                if isinstance(results, dict):
                    counts = results.get("counts")
                    if counts is None and "results" in results:
                        nested_results = results["results"]
                        if isinstance(nested_results, dict):
                            counts = nested_results.get("counts")

        qpu_metadata = response_data.get("metadata", {})
        qpu_metadata["job_id"] = response_data.get("job_id")

        return {
            "counts": counts,
            "metadata": qpu_metadata,
            "execution_time": response_data.get("execution_time"),
            "message": response_data.get("message"),
            "qpu_specific": True,
        }

    def _parse_job_results(
        self,
        job_result: Dict[str, Any],
        circuit_name: str,
        shots: int,
        device_name: Optional[str] = None,
        need_statevector=False,
        need_density_matrix=False,
    ) -> JobResult:
        """
        Parse job results from API response into a JobResult object.

        Args:
            job_result: Job status dictionary from API
            circuit_name: Name of the circuit
            shots: Number of shots executed
            device_name: Device name for format-specific parsing

        Returns:
            JobResult object with parsed quantum execution results

        Raises:
            BaseError: If results parsing fails or data is invalid
        """

        # Parse server_response to get quantum computation results
        server_response_str = job_result.get("server_response")
        parsed_server_response = None
        counts = None
        statevector = None
        probabilities = None
        density_matrix = None
        execution_time_seconds = None
        message = None
        qpu_metadata = {}

        if server_response_str:
            try:
                parsed_server_response = self._resolve_job_response_data(
                    server_response_str
                )
                if not isinstance(parsed_server_response, dict):
                    raise InvalidResponseError(
                        f"Resolved server_response must be a dictionary, got {type(parsed_server_response).__name__}"
                    )

                # Use device-specific parsing
                if device_name == "QpiAI-Indus-1":
                    # Use QPU-specific parsing
                    parsed_data = self._parse_qpu_response(parsed_server_response)
                    counts = parsed_data.get("counts")
                    execution_time_seconds = parsed_data.get("execution_time")
                    message = parsed_data.get("message")
                    qpu_metadata = parsed_data.get("metadata", {})

                    # For QPU, statevector and other data might not be available
                    # Keep defaults as None unless found in fallback parsing
                else:
                    # Use simulator parsing (default)
                    parsed_data = self._parse_simulator_response(parsed_server_response)
                    counts = parsed_data.get("counts")
                    statevector = parsed_data.get("statevector")
                    probabilities = parsed_data.get("probabilities")
                    density_matrix = parsed_data.get("density_matrix")
                    execution_time_seconds = parsed_data.get("execution_time")
                    message = parsed_data.get("message")

                # Convert counts to proper integer values if they're strings
                if counts and isinstance(counts, dict):
                    counts = {k: int(v) for k, v in counts.items()}

            except (
                json.JSONDecodeError,
                TypeError,
                InvalidResponseError,
                JobManagerError,
            ) as e:
                logger.warning(f"Failed to parse server_response: {e}")
                raise

        # Calculate total shots from counts
        total_shots = sum(counts.values()) if counts else shots

        # Use execution_time from server_response if available, otherwise fall back to job's execution_time
        if execution_time_seconds is not None:
            try:
                if isinstance(execution_time_seconds, str):
                    # Handle cases like '0.1225 seconds'
                    import re

                    match = re.search(r"[-+]?\d*\.?\d+", execution_time_seconds)
                    job_execution_time = float(match.group()) if match else 0.0
                else:
                    job_execution_time = float(execution_time_seconds)
            except (ValueError, TypeError, AttributeError):
                job_execution_time = 0.0
        else:
            # job has execution_time in seconds
            job_execution_time = job_result.get("execution_time", 0.0)

        # Build job metadata with QPU-specific data if available
        job_metadata = {
            "server_response": parsed_server_response,
            "status": job_result.get("status"),
            "circuit_id": job_result.get("circuit_id"),
            "experiment_id": job_result.get("experiment_id"),
            "experiment_name": job_result.get("experiment_name"),
            "compute_resource_name": job_result.get("compute_resource_name"),
            "server_response_id": job_result.get("server_response_id"),
            "method": job_result.get("method"),
            "created_at": job_result.get("created_at"),
            "updated_at": job_result.get("updated_at"),
            "device_name": device_name,
        }

        # Add QPU-specific metadata if available
        if qpu_metadata:
            job_metadata["qpu_metadata"] = qpu_metadata

        # Create JobResult object
        return JobResult(
            name=job_result.get("name", circuit_name),
            counts=counts,
            statevector=statevector,
            message=message,
            probabilities=probabilities,
            density_matrix=density_matrix,
            executionTime=job_execution_time,
            shots=total_shots,
            job_id=job_result.get("id"),
            job_status=job_result.get("status"),
            method=job_result.get("method"),
            credits_used=job_result.get("credits_used", 0),
            job_metadata=job_metadata,
        )

    def submit_and_wait_for_results_qasm(
        self,
        qasm_string_or_circuit: str | Circuit | IntermediateCirucitRepresentation,
        experiment_name: str = "Default Experiment",
        shots: int = 1024,
        method: str = "statevector",
        need_statevector: bool = False,
        need_density_matrix: bool = False,
        device_name: str = "QpiAI-QSV-Simulator",
        circuit_name: str = "circuit",
        compute_resource_id: Optional[str] = None,
        overwrite: bool = False,
        use_events: bool = True,
        timeout: int = 300,
    ) -> JobResult:
        """
        All-in-one QASM/Circuit submission that waits for completion and returns job results.

        This method combines job submission, status monitoring, and results retrieval
        into a single convenient function call with SSE support for real-time status updates.

        Args:
            qasm_string_or_circuit (str | Circuit | IntermediateCirucitRepresentation):
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

        # Handle circuit or QASM string input
        if isinstance(qasm_string_or_circuit, str):
            qasm_string = qasm_string_or_circuit
        else:
            # Convert Circuit or ICR to QASM string
            qasm_string = self._convert_circuit_to_qasm(qasm_string_or_circuit)

        # SSE setup
        sse_handler = None

        if use_events:
            if not SSE_AVAILABLE:
                raise RuntimeError(
                    "SSE handler not available. Install with: pip install requests-sse"
                )

            sse_handler = SSEResultHandler(device_name=device_name, timeout=timeout)
        try:
            # Submit the job
            submission_result = self.submit_qasm_job(
                qasm_string=qasm_string,
                experiment_name=experiment_name,
                shots=shots,
                method=method,
                need_statevector=need_statevector,
                need_density_matrix=need_density_matrix,
                device_name=device_name,
                circuit_name=circuit_name,
                compute_resource_id=compute_resource_id,
                overwrite=overwrite,
            )
            job_id = submission_result.get("job_id")
            if not job_id:
                raise JobSubmissionError("Job submission failed: no job ID returned")

            # Use SSE for real-time updates
            if use_events and sse_handler:
                sse_url = get_sse_url(job_id)
                headers = self._get_auth_header()

                if not sse_handler.wait_for_events(sse_url, headers):
                    # SSE failed or timed out
                    error = sse_handler.get_error()
                    if error:
                        raise JobManagerError(f"SSE execution failed: {error}")
                    raise JobManagerError(
                        f"SSE connection timed out for job {job_id} after {timeout}s"
                    )
            else:
                raise ValueError("use_events must be True - SSE only mode")

            # Retrieve job status after SSE completion
            result = self.get_job_status(job_id)
            if not result:
                raise JobStatusError(job_id, Exception("Failed to retrieve job status"))

            # Parse job results using helper method
            job_result = self._parse_job_results(
                result,
                circuit_name,
                shots,
                device_name,
                need_statevector=need_statevector,
                need_density_matrix=need_density_matrix,
            )
            # Add submission parameters to metadata
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

        finally:
            pass

    def _wait_with_polling(
        self,
        job_id: str,
        timeout: int,
        poll_interval: int,
        start_time: float,
        initial_poll_count: int,
    ) -> Dict[str, Any]:
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
                        f"Job {job_id} did not complete within {timeout} seconds (timeout)"
                    )

                # Check job status
                job_info = self.get_job_status(job_id)

                if not job_info:
                    raise JobManagerError(f"Job {job_id} not found during polling")

                current_status = job_info.get("status", "unknown")
                logger.debug(
                    f"Poll {poll_count}: Job {job_id} status = {current_status}"
                )

                # Check for completion (accept multiple success states)
                if current_status and current_status.lower() in [
                    "completed",
                    "success",
                ]:
                    logger.info(
                        f"Job {job_id} completed successfully after {elapsed_time:.1f}s"
                    )
                    result = {
                        "status": current_status,
                        "job_id": job_id,
                        "job_info": job_info,
                        "polling_checks": poll_count,
                    }
                    return result
                elif current_status and current_status.lower() == "failed":
                    error_message = (
                        job_info.get("error")
                        or job_info.get("error_message")
                        or "Unknown error"
                    )
                    # Include more details from job_info
                    error_details = {
                        "error": error_message,
                        "job_id": job_id,
                        "full_job_info": job_info,
                    }
                    raise JobManagerError(
                        f"Job {job_id} failed: {error_message}\n\nFull details: {error_details}"
                    )
                elif not current_status or current_status.lower() not in [
                    "pending",
                    "running",
                    "scheduled",
                    "submitted",
                    "queued",
                ]:
                    raise JobManagerError(
                        f"Job {job_id} in unexpected status: {current_status}"
                    )

                # Wait before next check
                time.sleep(poll_interval)

        except JobManagerError:
            raise
        except Exception as e:
            raise JobManagerError(f"Unexpected error during job polling: {e}")
