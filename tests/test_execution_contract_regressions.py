from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import json
import pytest
from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver
from qpiai_quantum.authentication.user import resolve_api_key
from qpiai_quantum.jobmanager import DeviceRegistry, JobManager, SSEResultHandler
from qpiai_quantum.jobmanager.backend import DeviceInfo, get_device_info
from qpiai_quantum.jobmanager.exceptions import JobSubmissionError
from qpiai_quantum.jobmanager.result_decoding import decode_counts


@pytest.fixture(autouse=True)
def clean_caches():
    JobManager.clear_resource_cache()
    yield
    JobManager.clear_resource_cache()


def test_discovery_uses_shared_credential_priority(monkeypatch):
    monkeypatch.setenv("API_KEY", "environment")
    monkeypatch.setenv("QPIAI_SERVER_URL", "https://test.example/")
    with patch(
        "qpiai_quantum.authentication.user.get_active_api_key", return_value="active"
    ):
        assert resolve_api_key(" explicit ") == "explicit"
        assert dict(DeviceRegistry._current_cache_key(" explicit ")) == {
            "X-Secret-Token": "explicit",
            "server_url": "https://test.example",
        }
        assert dict(DeviceRegistry._current_cache_key())["X-Secret-Token"] == "active"
    with patch(
        "qpiai_quantum.authentication.user.get_active_api_key", return_value=None
    ):
        assert (
            dict(DeviceRegistry._current_cache_key())["X-Secret-Token"] == "environment"
        )
        assert JobManager()._get_auth_header()["X-Secret-Token"] == "environment"


def test_concurrent_discovery_does_not_mix_accounts():
    barrier = Barrier(2)

    def fetch(key):
        token = dict(key)["X-Secret-Token"]
        barrier.wait(timeout=5)
        return [{"ID": token, "backend_name": "Shared", "method": token}]

    with patch.object(DeviceRegistry, "_fetch", side_effect=fetch):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(get_device_info, "Shared", token)
                for token in ("statevector", "density_matrix")
            ]
            assert [f.result().method for f in futures] == [
                "statevector",
                "density_matrix",
            ]


def test_ambiguous_resources_are_rejected():
    resources = [
        {"ID": str(i), "device_name": "Chip", "method": "qpu"} for i in range(2)
    ]
    with pytest.raises(ValueError, match="Ambiguous"):
        DeviceRegistry.load(resources)
    with (
        patch.object(DeviceRegistry, "_fetch", return_value=resources),
        pytest.raises(JobSubmissionError, match="found 2"),
    ):
        JobManager()._get_compute_resource_id("qpu", "Chip")


def test_missing_simulator_configuration_is_not_guessed():
    with pytest.raises(ValueError, match="no method"):
        DeviceInfo.from_resource("QpiAI-QDM-Simulator", {})


def test_exact_estimation_rejects_hardware_before_execution(backend_devices):
    with pytest.raises(ValueError, match="Exact estimation"):
        VQESolver(n_qubits=1).run(device_name="QpiAI-Indus-1", estimator="exact")


def test_estimator_selection_is_separate_from_execution_method(backend_devices):
    for device, estimator, exact in [
        ("QpiAI-Indus-1", "auto", False),
        ("QpiAI-QSV-Local", "sampled", False),
        ("QpiAI-QSV-Local", "exact", True),
    ]:
        solver = VQESolver(n_qubits=1)
        with pytest.raises(ValueError, match="hamiltonian"):
            solver.run(device_name=device, estimator=estimator)
        assert solver._requested_outputs["need_statevector"] is exact
