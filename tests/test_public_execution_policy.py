"""Regression checks for public execution policy and authentication convenience."""

import inspect
from unittest.mock import patch

import numpy as np
import pytest

from qpiai_quantum import QpiAIQuantumAuth
from qpiai_quantum.authentication.user import clear_user, get_user, set_user
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.algorithms.opt.solvers.qaoa import QAOASolver
from qpiai_quantum.algorithms.opt.solvers.vqe import VQESolver
from qpiai_quantum.jobmanager import DeviceRegistry, ExecutionEngine, JobManager
from qpiai_quantum.jobmanager.backend import get_device_info
from qpiai_quantum.utils.gate_matrices import GateMatrices


def test_public_entrypoint_defaults_remain_distinct():
    for entrypoint in (QAOASolver.run, VQESolver.run, ExecutionEngine.execute_circuit):
        assert (
            inspect.signature(entrypoint).parameters["device_name"].default
            == "QpiAI-QSV-Local"
        )
    assert (
        inspect.signature(Circuit.run).parameters["device_name"].default
        == "QpiAI-QSV-Simulator"
    )
    assert (
        inspect.signature(ExecutionEngine.execute_circuit).parameters["timeout"].default
        == 300
    )
    assert (
        inspect.signature(JobManager.submit_and_wait_for_results_qasm)
        .parameters["timeout"]
        .default
        == 300
    )


@pytest.mark.parametrize("name", ["RY", "RZ"])
def test_matrix_helpers_support_class_and_instance_calls(name):
    np.testing.assert_array_equal(
        getattr(GateMatrices, name)(0.37), getattr(GateMatrices(), name)(0.37)
    )


def test_login_resolves_environment_and_preserves_explicit_precedence(monkeypatch):
    previous = get_user()
    try:
        clear_user()
        monkeypatch.setenv("API_KEY", "environment-key")
        with patch.object(
            QpiAIQuantumAuth,
            "me",
            return_value={"name": "Auth test", "email": "auth@example.test"},
        ) as me:
            QpiAIQuantumAuth.login()
            me.assert_called_with(api_key="environment-key")
            QpiAIQuantumAuth.login("explicit-key")
            me.assert_called_with(api_key="explicit-key")
            QpiAIQuantumAuth.login()
            me.assert_called_with(api_key="explicit-key")
    finally:
        clear_user() if previous is None else set_user(previous)


@pytest.mark.parametrize("argument", ["execution_kind", "result_transport"])
def test_excluded_options_are_rejected_before_execution(argument):
    for entrypoint in (Circuit(1).run, QAOASolver().run, VQESolver(1).run):
        with pytest.raises(TypeError, match="not supported"):
            entrypoint(**{argument: "unsupported"})
    for entrypoint in (
        DeviceRegistry.get,
        get_device_info,
        ExecutionEngine.execute_circuit,
        JobManager.submit_qasm_job,
        JobManager.submit_circuit_job,
        JobManager.submit_and_wait_for_results_qasm,
    ):
        assert argument not in inspect.signature(entrypoint).parameters


def test_public_registry_ignores_emulator_variants(backend_devices):
    resources = [
        {
            "ID": "emulated",
            "device_name": "QpiAI-Indus-1",
            "method": "qpu",
            "execution_kind": "emulator",
        },
        *backend_devices,
    ]
    DeviceRegistry.load(resources)
    assert get_device_info("QpiAI-Indus-1").is_hardware
    with patch.object(DeviceRegistry, "_fetch", return_value=resources):
        assert (
            JobManager()._get_compute_resource_id("qpu", "QpiAI-Indus-1") == "indus-id"
        )
