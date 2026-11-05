from qpiai_quantum.authentication.user import get_active_auth_header
from qpiai_quantum.jobmanager.backend import ResolvedBackend
from qpiai_quantum.config import HTTP_TIMEOUT, get_server_url
import requests
from urllib.parse import quote


class QuantumAppAdapter:
    """
    Connects to the QpiAI Quantum App server.
    """

    def __init__(self):
        self.API_URL = get_server_url()
        self.headers = get_active_auth_header()
        if self.headers is None:
            raise ValueError(
                "User is not authenticated. Log in before accessing Quantum App compute resources."
            )

    def resolve_backend(self, device_name: str) -> ResolvedBackend:
        """
        Resolve a device name to its backend endpoint configuration.
        """
        if not isinstance(device_name, str) or not device_name.strip():
            raise ValueError("device_name must be a nonempty string.")
        encoded_name = quote(device_name.strip(), safe="")
        try:
            response = requests.get(
                f"{self.API_URL}/api/compute-resources/sdk/resolve/{encoded_name}",
                headers=self.headers,
                timeout=HTTP_TIMEOUT,
            )
            if response.status_code != 200:
                raise ValueError(
                    "Unable to resolve the requested backend. Check that the backend is supported and that your API key can access it."
                )
            resolved_backend = response.json()
            return ResolvedBackend(
                ID=resolved_backend["ID"],
                backend_name=resolved_backend["backend_name"],
                address=resolved_backend["address"],
                health_check=resolved_backend["health_check"],
                run=resolved_backend["run"],
            )
        except ValueError:
            raise
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500:
                raise ValueError(
                    "The server encountered an internal error. Please try again later or contact support."
                ) from exc
            else:
                raise ValueError(
                    "Unable to resolve the requested backend. Check that the backend is supported and that your API key can access it."
                ) from exc
        except requests.exceptions.RequestException as exc:
            raise ValueError(
                "Unable to resolve the requested backend. Check that the backend is supported and that your API key can access it."
            ) from exc
        except Exception as exc:
            raise ValueError(
                "Unable to resolve the requested backend. Check the backend value and try again."
            ) from exc
