from types import SimpleNamespace
from unittest.mock import patch

import pytest

from qpiai_quantum.circuit import Circuit

pytestmark = pytest.mark.usefixtures("backend_devices")


@pytest.mark.parametrize(
    ("device_name", "expected_statevector", "expected_density_matrix"),
    [
        ("QpiAI-QDM-Simulator", False, True),
        ("QpiAI-QDM-Lite", False, True),
        ("QpiAI-QTN-Simulator", True, True),
        ("QpiAI-QSV-Simulator", False, False),
        ("QpiAI-QSV-Lite", False, False),
    ],
)
@patch("qpiai_quantum.jobmanager.ExecutionEngine.execute_circuit")
def test_run_uses_backend_aware_result_defaults(
    mock_execute,
    device_name: str,
    expected_statevector: bool,
    expected_density_matrix: bool,
):
    mock_execute.return_value = SimpleNamespace(counts=None)

    Circuit(1).run(device_name=device_name)

    assert mock_execute.call_args.kwargs["need_statevector"] is expected_statevector
    assert (
        mock_execute.call_args.kwargs["need_density_matrix"] is expected_density_matrix
    )


@pytest.mark.parametrize("device_name", ["QpiAI-QDM-Simulator", "QpiAI-QDM-Lite"])
@patch("qpiai_quantum.jobmanager.ExecutionEngine.execute_circuit")
def test_run_rejects_qdm_statevector_request(mock_execute, device_name: str):
    with pytest.raises(ValueError, match="returns density matrices only"):
        Circuit(1).run(
            device_name=device_name,
            need_statevector=True,
            need_density_matrix=True,
        )

    mock_execute.assert_not_called()


@pytest.mark.parametrize("device_name", ["QpiAI-QDM-Simulator", "QpiAI-QTN-Simulator"])
@patch("qpiai_quantum.jobmanager.ExecutionEngine.execute_circuit")
def test_explicit_false_disables_backend_default_artifacts(mock_execute, device_name):
    mock_execute.return_value = SimpleNamespace(counts=None)
    Circuit(1).run(
        device_name=device_name, need_statevector=False, need_density_matrix=False
    )
    assert mock_execute.call_args.kwargs["need_statevector"] is False
    assert mock_execute.call_args.kwargs["need_density_matrix"] is False
