from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager import DeviceRegistry
from qpiai_quantum.jobmanager.backend import get_device_info, get_device_method

AUTH_KEY = (("X-Secret-Token", "key"),)


@pytest.fixture(autouse=True)
def clear_registry():
    DeviceRegistry.clear()
    yield
    DeviceRegistry.clear()


def _response(resources: list[dict]) -> MagicMock:
    response = MagicMock(status_code=200)
    response.json.return_value = resources
    return response


def test_unknown_device_refreshes_once_then_fails() -> None:
    """Unknown names trigger one reload before reporting the available devices."""
    resources = [{"backend_name": "QpiAI-QSV-Simulator", "method": "statevector"}]
    with (
        patch.object(DeviceRegistry, "_current_cache_key", return_value=AUTH_KEY),
        patch("requests.get", return_value=_response(resources)) as mock_get,
    ):
        assert get_device_method("QpiAI-QSV-Simulator") == "statevector"
        with pytest.raises(ValueError, match="Unsupported device name 'Missing'"):
            get_device_method("Missing")
    assert mock_get.call_count == 2


def test_device_list_is_cached_per_api_key() -> None:
    """Repeated lookups reuse the cache; switching accounts reloads the device list."""
    resources = [{"backend_name": "QpiAI-QSV-Simulator", "method": "statevector"}]
    active = {"key": AUTH_KEY}
    with (
        patch.object(
            DeviceRegistry, "_current_cache_key", side_effect=lambda: active["key"]
        ),
        patch("requests.get", return_value=_response(resources)) as mock_get,
    ):
        get_device_info("QpiAI-QSV-Simulator")
        get_device_info("QpiAI-QSV-Simulator")
        active["key"] = (("X-Secret-Token", "other"),)
        get_device_info("QpiAI-QSV-Simulator")
    assert mock_get.call_count == 2


def test_legacy_qpu_type_is_supported_without_name_guesses() -> None:
    """Only an explicit legacy QPU type can supply a missing method."""
    DeviceRegistry.load(
        [
            {
                "backend_name": "QpiAI-QTN-Simulator",
                "device_name": "QpiAI-QTN-Simulator",
                "method": "tensor_network",
            },
            {
                "backend_name": "QpiAI-Indus-1",
                "device_name": "QpiAI-Indus-1",
                "device_type": "qpu",
            },
        ]
    )
    qtn = get_device_info("QpiAI-QTN-Simulator")
    assert qtn.method == "tensor_network"
    assert qtn.default_results == frozenset({"statevector", "density_matrix"})
    assert get_device_info("QpiAI-Indus-1").is_hardware


def test_local_device_needs_no_backend() -> None:
    with patch("requests.get") as mock_get:
        assert get_device_method("QpiAI-QSV-Local") == "statevector"
    mock_get.assert_not_called()


def test_loading_without_login_asks_for_authentication() -> None:
    with (
        patch.object(DeviceRegistry, "_current_cache_key", return_value=None),
        pytest.raises(ValueError, match="Authentication required"),
    ):
        get_device_info("QpiAI-QSV-Simulator")


@patch("qpiai_quantum.jobmanager.ExecutionEngine.execute_circuit")
def test_run_uses_backend_method_and_rejects_unsupported_results(mock_execute) -> None:
    DeviceRegistry.load(
        [
            {
                "backend_name": "QpiAI-Counts-Only",
                "method": "statevector_sampling",
                "supported_results": [],
                "default_results": [],
            }
        ]
    )
    mock_execute.return_value = SimpleNamespace(counts=None)
    Circuit(1).run(device_name="QpiAI-Counts-Only")
    assert mock_execute.call_args.kwargs["method"] == "statevector_sampling"
    with pytest.raises(ValueError, match="cannot return a statevector"):
        Circuit(1).run(device_name="QpiAI-Counts-Only", need_statevector=True)
    with pytest.raises(ValueError, match="cannot return a density matrix"):
        Circuit(1).run(device_name="QpiAI-Counts-Only", need_density_matrix=True)
