from qpiai_quantum.authentication.user import get_user
from qpiai_quantum.jobmanager.backend import Backend, ResolvedBackend
from qpiai_quantum.config import HTTP_TIMEOUT, get_server_url
import requests


class QuantumAppAdapter:
    """
    Connects to the QpiAI Quantum App server.
    """

    def __init__(self):
        user = get_user()

        if user is None:
            raise ValueError(
                "User is not authenticated. Please log in to access the Quantum App and Compute Resources"
            )

        # Setup the request template to the quantum app

        self.API_URL = get_server_url()
        self.headers = {
            "X-Secret-Token": f"{user.api_key}",
        }

    def resolve_backend(self, backend: Backend):
        """
        Resolves the backend name to a full URL.
        """

        if backend is None:
            raise ValueError("Backend cannot be None. Please provide a valid backend.")
        if not isinstance(backend, Backend):
            raise TypeError("Backend must be an instance of the Backend class.")
        response = requests.get(
            f"{self.API_URL}/api/compute-resources/sdk/resolve/{backend.value}",
            headers=self.headers,
            timeout=HTTP_TIMEOUT,
        )
        if response.status_code != 200:
            raise ValueError(f"Failed to resolve backend: {response.text}")

        resolved_backend = response.json()
        return ResolvedBackend(
            ID=resolved_backend["ID"],
            backend_name=resolved_backend["backend_name"],
            address=resolved_backend["address"],
            health_check=resolved_backend["health_check"],
            run=resolved_backend["run"],
        )
