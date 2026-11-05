import os

import pytest
from dotenv import load_dotenv

from qpiai_quantum import QpiAIQuantumAuth

load_dotenv("qcloud.env")
load_dotenv()


@pytest.fixture(scope="session")
def authenticated_sdk_user():
    """Authenticate tests that exercise the access-controlled local backend."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        pytest.skip("API_KEY is required for authenticated QSV-Local tests")

    try:
        return QpiAIQuantumAuth.login(api_key)
    except Exception:
        pytest.fail(
            "QSV-Local test authentication failed; credentials were not displayed",
            pytrace=False,
        )


from qpiai_quantum.jobmanager import DeviceRegistry

BACKEND_DEVICES = [
    {
        "ID": "qsv-id",
        "backend_name": "QpiAI-QSV-Simulator",
        "device_name": "QpiAI-QSV-Simulator",
        "status": "active",
        "method": "statevector",
        "supported_results": ["statevector", "density_matrix"],
        "default_results": [],
    },
    {
        "ID": "qsv-lite-id",
        "backend_name": "QpiAI-QSV-Lite",
        "device_name": "QpiAI-QSV-Lite",
        "status": "active",
        "method": "statevector_sampling",
        "supported_results": ["statevector", "density_matrix"],
        "default_results": [],
    },
    {
        "ID": "qdm-id",
        "backend_name": "QpiAI-QDM-Simulator",
        "device_name": "QpiAI-QDM-Simulator",
        "status": "active",
        "method": "density_matrix",
        "supported_results": ["density_matrix"],
        "default_results": ["density_matrix"],
    },
    {
        "ID": "qdm-lite-id",
        "backend_name": "QpiAI-QDM-Lite",
        "device_name": "QpiAI-QDM-Lite",
        "status": "active",
        "method": "density_matrix",
        "supported_results": ["density_matrix"],
        "default_results": ["density_matrix"],
    },
    {
        "ID": "qtn-id",
        "backend_name": "QpiAI-QTN-Simulator",
        "device_name": "QpiAI-QTN-Simulator",
        "status": "active",
        "method": "tensor_network",
        "supported_results": ["statevector", "density_matrix"],
        "default_results": ["statevector", "density_matrix"],
    },
    {
        "ID": "indus-id",
        "backend_name": "QpiAI-Indus-1",
        "device_name": "QpiAI-Indus-1",
        "status": "active",
        "method": "qpu",
        "supported_results": ["statevector", "density_matrix"],
        "default_results": [],
    },
]


@pytest.fixture
def backend_devices():
    """Load the standard backend device list into the SDK device registry."""
    DeviceRegistry.load(BACKEND_DEVICES)
    yield BACKEND_DEVICES
    DeviceRegistry.clear()
