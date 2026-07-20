"""
QASM Simulator Result
======================
Concrete :class:`~qpiai_quantum.results.base_result.BaseQuantumResult`
implementation for the local QASM statevector simulator.

The API deliberately mirrors
:class:`~qpiai_quantum.jobmanager.job_result.JobResult` so that code which
does ``result.get()["counts"]`` or ``result.counts`` works identically
regardless of whether the circuit was executed on the cloud or locally.
"""

from typing import Any, Optional

import numpy as np

from ..results.base_result import BaseQuantumResult


class QasmSimulatorResult(BaseQuantumResult):
    """Result object returned by :class:`QasmSimulator`.

    Drop-in compatible with
    :class:`~qpiai_quantum.jobmanager.job_result.JobResult`:
    all the same properties, ``.get()``, ``.plot()``, ``.to_json()``
    work identically.
    """

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
        "n_qubits",
        "n_cbits",
    )

    def __init__(
        self,
        name: str | None = None,
        counts: dict[str, int] | None = None,
        statevector: list | None = None,
        probabilities: dict[str, float] | None = None,
        density_matrix=None,
        message: str | None = "Job executed successfully",
        execution_time: float = 0.0,
        cpu_usage: float = 0.0,
        memory_usage: float = 0.0,
        gpu_memory_usage: float = 0.0,
        shots: int | None = 0,
        job_id: str | None = None,
        job_status: str | None = None,
        method: str | None = "qasm_simulator",
        credits_used: int = 0,
        job_metadata: dict[str, Any] | None = None,
        n_qubits: int | None = None,
        n_cbits: int | None = None,
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
        self.method = method
        self.credits_used = credits_used
        self.job_metadata = job_metadata or {}
        self.n_qubits = n_qubits
        self.n_cbits = n_cbits

    # -- BaseQuantumResult required properties ------------------------------

    @property
    def execution_time(self) -> float:
        return self._execution_time

    @property
    def statevector(self) -> list | None:
        return self._statevector

    @statevector.setter
    def statevector(self, value: list | None):
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
        return self._job_id

    @job_id.setter
    def job_id(self, value: str | None):
        self._job_id = value

    @property
    def job_status(self) -> str | None:
        return self._job_status

    @job_status.setter
    def job_status(self, value: str | None):
        self._job_status = value

    # -- Convenience methods ------------------------------------------------

    def set_name(self, name: str):
        self.name = name

    def get_job_id(self) -> str | None:
        return self.job_id

    def get_job_status(self) -> str | None:
        return self.job_status

    # -- get(): zero-arg returns everything (matches JobResult.get()) -------

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
                "n_qubits": self.n_qubits,
                "n_cbits": self.n_cbits,
            }

        res_object: dict[str, Any] = {}
        for p in param:
            if hasattr(self, p):
                res_object[p] = getattr(self, p)
            else:
                raise AttributeError(f"QasmSimulatorResult object has no attribute {p}")
        return res_object

    # -- repr ---------------------------------------------------------------

    def __repr__(self):
        status_info = f", status='{self.job_status}'" if self.job_status else ""
        lines = [f"QasmSimulatorResult(job_id='{self.job_id}'{status_info})"]

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

        if self.shots and self.shots > 0:
            lines.append(f"  shots: {self.shots}")

        if self._execution_time and self._execution_time > 0:
            lines.append(f"  execution_time: {self._execution_time:.4f}s")

        return "\n".join(lines)

    # -- JSON serialisation -------------------------------------------------

    def to_json(self, indent: int = 2) -> str:
        import json as _json

        def _json_encoder(obj):
            if isinstance(obj, complex):
                return str(obj)
            if isinstance(obj, np.complexfloating):
                return str(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        data: dict[str, Any] = {
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
        if self.shots and self.shots > 0:
            data["shots"] = self.shots
        if self._execution_time and self._execution_time > 0:
            data["execution_time"] = self._execution_time
        if self.n_qubits is not None:
            data["n_qubits"] = self.n_qubits
        if self.n_cbits is not None:
            data["n_cbits"] = self.n_cbits

        return _json.dumps(data, indent=indent, default=_json_encoder)
