from typing import Any, Dict, List, Optional
from ..results.base_result import BaseQuantumResult
import numpy as np


class JobResult(BaseQuantumResult):
    __slots__ = (
        "name",
        "_counts",
        "_statevector",
        "message",
        "_probabilities",
        "density_matrix",
        "_execution_time",
        "cpu_usage",
        "memory_usage",
        "gpu_memory_usage",
        "_shots",
        "_job_id",
        "_job_status",
        "method",
        "credits_used",
        "job_metadata",
    )

    def __init__(
        self,
        name: str | None = None,
        counts: dict[str, int] | None = None,
        statevector: list[list[complex]] | None = None,
        probabilities: dict[str, float] | None = None,
        density_matrix: list[list[float]] | None = None,
        message: str | None = "Job executed successfully",
        execution_time: float = 0.0,
        cpu_usage: float = 0.0,
        memory_usage: float = 0.0,
        gpu_memory_usage: float = 0.0,
        shots: int | None = 0,
        job_id: str | None = None,
        job_status: str | None = None,
        method: str | None = None,
        credits_used: int = 0,
        job_metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self._counts = counts
        self._statevector = statevector
        self.message = message
        self._probabilities = probabilities
        self.density_matrix = density_matrix
        self._execution_time = execution_time
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage
        self.gpu_memory_usage = gpu_memory_usage
        self._shots = shots
        self._job_id = job_id
        self._job_status: str | None = job_status or "completed"
        # Default to completed for result objects
        self.method = method
        self.credits_used = credits_used
        self.job_metadata = job_metadata or {}

    @property
    def execution_time(self) -> float:
        return self._execution_time

    @property
    def statevector(self) -> list[list[complex]] | None:
        return self._statevector

    @statevector.setter
    def statevector(self, value: list[list[complex]] | None):
        self._statevector = value

    @property
    def probabilities(self) -> dict[str, float] | None:
        if self._probabilities is not None:
            return self._probabilities

        if self._counts is None or not self._shots or self._shots == 0:
            return None

        total_shots = self._shots
        return {outcome: count / total_shots for outcome, count in self._counts.items()}

    @probabilities.setter
    def probabilities(self, value: dict[str, float] | None):
        self._probabilities = value

    @property
    def shots(self) -> int | None:
        return self._shots

    @shots.setter
    def shots(self, value: int | None):
        self._shots = value

    @property
    def counts(self) -> dict[str, int] | None:
        return self._counts

    @counts.setter
    def counts(self, value: dict[str, int] | None):
        self._counts = value

    @property
    def job_id(self) -> str | None:
        """Get the job ID."""
        return self._job_id

    @job_id.setter
    def job_id(self, value: str | None):
        """Set the job ID."""
        self._job_id = value

    @property
    def job_status(self) -> str | None:
        """Get the job status."""
        return self._job_status

    @job_status.setter
    def job_status(self, value: str | None):
        """Set the job status."""
        self._job_status = value

    def set_name(self, name: str):
        self.name = name

    def get_job_id(self) -> str | None:
        """
        Get job ID.

        Returns:
            Optional[str]: Job ID if available

        Example:
            >>> job_id = result.get_job_id()
        """
        return self.job_id

    def get_job_status(self) -> str | None:
        """
        Get job status.

        Returns:
            Optional[str]: Job status if available

        Example:
            >>> status = result.get_job_status()
        """
        return self.job_status

    def get(self, *param) -> dict[str, Any]:
        if len(param) == 0:
            return {
                "name": self.name,
                "counts": self.counts,
                "statevector": self.statevector,
                "message": self.message,
                "probabilities": self.probabilities,
                "density_matrix": self.density_matrix,
                "execution_time": self._execution_time,
                "cpu_usage": self.cpu_usage,
                "memory_usage": self.memory_usage,
                "gpu_memory_usage": self.gpu_memory_usage,
                "shots": self.shots,
                "job_id": self.job_id,
                "job_status": self.job_status,
                "method": self.method,
                "credits_used": self.credits_used,
                "job_metadata": self.job_metadata,
            }

        res_object = {}
        for p in param:
            if hasattr(self, p):
                res_object[p] = getattr(self, p)
            else:
                raise AttributeError(f"JobResult object has no attribute {p}")

        return res_object

    def __repr__(self):
        status_info = f", status='{self.job_status}'" if self.job_status else ""
        lines = [f"JobResult(job_id='{self.job_id}'{status_info})"]

        if self.counts:
            preview_items = list(self.counts.items())[:10]
            preview_text = ", ".join([f"{k}:{v}" for k, v in preview_items])
            suffix = " ..." if len(self.counts) > 10 else ""
            lines.append(
                f"  counts: {len(self.counts)} entries ({preview_text}{suffix})"
            )

        if self.statevector:
            sv_length = (
                len(self.statevector)
                if isinstance(self.statevector, list)
                else "unknown"
            )
            lines.append(f"  statevector: available (dim={sv_length})")

        if self.density_matrix:
            dm_length = (
                len(self.density_matrix)
                if isinstance(self.density_matrix, list)
                else "unknown"
            )
            lines.append(f"  density_matrix: available (dim={dm_length})")

        if self.shots and self.shots > 0:
            lines.append(f"  shots: {self.shots}")

        if self._execution_time and self._execution_time > 0:
            lines.append(f"  execution_time: {self._execution_time:.4f}s")

        return "\n".join(lines)

    def to_json(self, indent=2):
        import json

        def json_encoder(obj):
            if isinstance(obj, np.complexfloating):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                converted = []
                for item in obj:
                    if isinstance(item, np.complexfloating):
                        converted.append(str(item))
                    elif isinstance(item, list):
                        converted.append(
                            [
                                str(x) if isinstance(x, np.complexfloating) else x
                                for x in item
                            ]
                        )
                    else:
                        converted.append(item)
                return converted
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        data = {
            "job_id": self.job_id,
            "job_status": self.job_status,
            "name": self.name,
            "message": self.message,
        }

        if self.counts:
            data["counts"] = self.counts

        if self.statevector:
            data["statevector"] = self.statevector

        if self.probabilities:
            data["probabilities"] = self.probabilities

        if self.density_matrix:
            data["density_matrix"] = self.density_matrix

        execution_metrics = {}
        if self._execution_time and self._execution_time > 0:
            execution_metrics["execution_time"] = self._execution_time
        if self.shots and self.shots > 0:
            execution_metrics["shots"] = self.shots
        if self.credits_used:
            execution_metrics["credits_used"] = self.credits_used

        if execution_metrics:
            data["execution_metrics"] = execution_metrics

        if self.job_metadata:
            job_metadata_filtered = dict(self.job_metadata)

            if "server_response" in job_metadata_filtered:
                del job_metadata_filtered["server_response"]

            data["job_metadata"] = job_metadata_filtered

        resource_usage = {}
        if self.cpu_usage and self.cpu_usage > 0:
            resource_usage["cpu_usage"] = self.cpu_usage
        if self.memory_usage and self.memory_usage > 0:
            resource_usage["memory_usage"] = self.memory_usage
        if self.gpu_memory_usage and self.gpu_memory_usage > 0:
            resource_usage["gpu_memory_usage"] = self.gpu_memory_usage

        if resource_usage:
            data["resource_usage"] = resource_usage

        return json.dumps(data, indent=indent, default=json_encoder)


# NOTE: Alias for easy migration - can use JobResult as drop-in replacement
JobExecutionResult = JobResult
