import time
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..circuit.circuit import Circuit
    from .job_result import JobResult


class ExecutionEngine:
    @staticmethod
    def execute_circuit(
        circuit: "Circuit",
        shots: int = 1024,
        *,
        need_statevector: bool = False,
        need_density_matrix: bool = False,
        experiment_name: str | None = None,
        circuit_name: str | None = None,
        access_token: str | None = None,
        overwrite: bool = True,
        timeout: int = 300,
        method: str | None = None,
        device_name: str = "QpiAI-QSV-Local",
    ) -> "JobResult":
        """Submit a circuit using a device profile and wait for its result."""
        import uuid
        from ..authentication.user import user_context
        from .jobmanager import JobManager

        if access_token is None:
            access_token = ExecutionEngine._get_api_key()
        if experiment_name is None:
            experiment_name = "Default Experiment"
        if circuit_name is None:
            circuit_name = (
                f"{experiment_name}_circuit_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            )
        with user_context(access_token):
            job_manager = JobManager()
            result = job_manager.submit_and_wait_for_results_qasm(
                qasm_string_or_circuit=circuit,
                shots=shots,
                experiment_name=experiment_name,
                need_statevector=need_statevector,
                need_density_matrix=need_density_matrix,
                method=method,
                device_name=device_name,
                circuit_name=circuit_name,
                use_events=True,
                overwrite=overwrite,
                timeout=timeout,
            )
        return result

    @staticmethod
    def _get_api_key(access_token: str | None = None) -> str:
        from ..authentication.user import resolve_api_key

        api_key = resolve_api_key(access_token)
        if api_key is not None:
            return api_key
        raise ValueError(
            "API_KEY not found. Please either:\n  1. Call QpiAIQuantumAuth.login() to authenticate, or\n  2. Set os.environ['API_KEY'] = 'your_key', or\n  3. Pass access_token='your_key' as an argument"
        )
