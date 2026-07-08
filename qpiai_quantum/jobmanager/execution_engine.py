import os
import time
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..circuit.circuit import Circuit
    from .job_result import JobResult


class ExecutionEngine:
    @staticmethod
    def execute_circuit(
        circuit: "Circuit",
        shots: int = 1024,
        need_statevector: bool = False,
        need_density_matrix: bool = False,
        experiment_name: Optional[str] = None,
        circuit_name: Optional[str] = None,
        access_token: Optional[str] = None,
        overwrite: bool = True,
        timeout: int = 300,
        method: str = "statevector",
        device_name: str = "QpiAI-QSV-Local",
        **kwargs,
    ) -> "JobResult":
        from .jobmanager import JobManager

        if access_token is None:
            access_token = ExecutionEngine._get_api_key(**kwargs)

        if experiment_name is None:
            experiment_name = "Default Experiment"

        if circuit_name is None:
            import uuid

            circuit_name = (
                f"{experiment_name}_circuit_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            )
        # print(f"Running on device '{device_name}' with method '{method}'")
        job_manager = JobManager()
        try:
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
        except Exception as e:
            print(f"Error : {str(e)}")
            raise
        return result

    @staticmethod
    def _get_api_key(**kwargs) -> str:
        if "access_token" in kwargs:
            return kwargs["access_token"]

        try:
            from ..authentication.user import get_user

            user = get_user()
            if user and hasattr(user, "api_key") and user.api_key:
                return user.api_key
        except (ImportError, Exception):
            pass

        api_key = os.getenv("API_KEY")
        if api_key:
            return api_key

        raise ValueError(
            "API_KEY not found. Please either:\n"
            "  1. Call QpiAIQuantumAuth.login() to authenticate, or\n"
            "  2. Add API_KEY='your_key' to qcloud.env, or\n"
            "  3. Pass access_token='your_key' as an argument"
        )
