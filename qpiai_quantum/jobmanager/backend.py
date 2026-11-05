import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

LOCAL_DEVICE_NAME: Final[str] = "QpiAI-QSV-Local"
RESULT_STATEVECTOR: Final[str] = "statevector"
RESULT_DENSITY_MATRIX: Final[str] = "density_matrix"
DEVICE_CACHE_TTL_SECONDS: Final[float] = 300.0


@dataclass(frozen=True)
class DeviceInfo:
    """Execution profile of a device, as configured on the QpiAI backend.

    Attributes:
        name: Device name users pass as ``device_name``.
        method: Execution method sent with jobs on this device.
        is_hardware: Whether this profile targets physical QPU hardware.
        supported_results: Extra results the device can return besides counts.
        default_results: Results requested when the caller does not choose.
        max_qubits: Largest circuit the device accepts, or 0 when unlimited.
    """

    name: str
    method: str
    is_hardware: bool
    supported_results: frozenset[str]
    default_results: frozenset[str]
    max_qubits: int = 0

    @classmethod
    def from_resource(cls, name: str, resource: dict[str, Any]) -> "DeviceInfo":
        method = resource.get("method") or _infer_method(resource)
        supported = resource.get("supported_results")
        defaults = resource.get("default_results")
        if method == "qpu":
            supported, defaults = ([], [])
        if supported is None:
            supported = (
                [RESULT_DENSITY_MATRIX]
                if method == "density_matrix"
                else [RESULT_STATEVECTOR, RESULT_DENSITY_MATRIX]
            )
        if defaults is None:
            defaults = {
                "density_matrix": [RESULT_DENSITY_MATRIX],
                "tensor_network": [RESULT_STATEVECTOR, RESULT_DENSITY_MATRIX],
            }.get(method, [])
        return cls(
            name,
            method,
            method == "qpu",
            frozenset(supported),
            frozenset(defaults),
            int(resource.get("max_qubits") or 0),
        )


LOCAL_DEVICE: Final[DeviceInfo] = DeviceInfo(
    name=LOCAL_DEVICE_NAME,
    method="statevector",
    is_hardware=False,
    supported_results=frozenset({RESULT_STATEVECTOR, RESULT_DENSITY_MATRIX}),
    default_results=frozenset(),
)


def _infer_method(resource: dict[str, Any]) -> str:
    """Accept an explicit legacy QPU type; simulator methods must be configured."""
    if resource.get("device_type") == "qpu":
        return "qpu"
    raise ValueError(
        "Device profile has no method. Ask the platform administrator to configure its execution profile."
    )


class DeviceRegistry:
    """Devices available to the logged-in user, loaded from the QpiAI backend.

    Devices are fetched from ``/api/compute-resources`` on first use and cached per
    server and resolved credential for ``DEVICE_CACHE_TTL_SECONDS``. Looking up an unknown device refreshes
    the cache once, so newly onboarded devices work without an SDK release. Only the
    in-process ``QpiAI-QSV-Local`` simulator is built in.
    """

    _lock = threading.Lock()
    _revisions: dict[tuple[tuple[str, str], ...] | None, object] = {}
    _entries: dict[
        tuple[tuple[str, str], ...] | None, tuple[float, dict[str, DeviceInfo]]
    ] = {}

    @classmethod
    def get(cls, device_name: str, access_token: str | None = None) -> DeviceInfo:
        if device_name == LOCAL_DEVICE_NAME:
            return LOCAL_DEVICE
        cache_key = (
            cls._current_cache_key(access_token)
            if access_token
            else cls._current_cache_key()
        )
        devices = cls._get_devices(cache_key=cache_key)
        if device_name not in devices:
            devices = cls._get_devices(force=True, cache_key=cache_key)
        try:
            return devices[device_name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported device name '{device_name}'. Supported devices are: {', '.join(sorted({*devices, LOCAL_DEVICE_NAME}))}"
            ) from exc

    @classmethod
    def available(cls, access_token: str | None = None) -> list[DeviceInfo]:
        devices = cls._get_devices(access_token=access_token)
        return [LOCAL_DEVICE, *sorted(devices.values(), key=lambda info: info.name)]

    @classmethod
    def refresh(cls, access_token: str | None = None) -> None:
        """Reload the device list from the backend."""
        cls._get_devices(force=True, access_token=access_token)

    @classmethod
    def clear(cls) -> None:
        """Drop the cached device list."""
        with cls._lock:
            cls._entries.clear()
            cls._revisions.clear()

    @classmethod
    def load(
        cls, resources: list[dict[str, Any]], access_token: str | None = None
    ) -> None:
        """Replace the cached devices with ``resources`` from ``/api/compute-resources``."""
        cache_key = (
            cls._current_cache_key(access_token)
            if access_token
            else cls._current_cache_key()
        )
        devices = cls._index(resources)
        with cls._lock:
            cls._revisions[cache_key] = object()
            cls._entries[cache_key] = (time.monotonic(), devices)

    @staticmethod
    def _index(resources: list[dict[str, Any]]) -> dict[str, DeviceInfo]:
        devices: dict[str, DeviceInfo] = {}
        identities: dict[str, object] = {}
        for field in ("device_name", "backend_name"):
            for resource in resources:
                if resource.get("status") not in {None, "active"}:
                    continue
                if resource.get("execution_kind") == "emulator":
                    continue
                name = resource.get(field)
                if not name:
                    continue
                info = DeviceInfo.from_resource(name, resource)
                identity = resource.get("ID", id(resource))
                if name in devices and (
                    devices[name] != info or identities[name] != identity
                ):
                    raise ValueError(f"Ambiguous device profile for '{name}'.")
                devices[name] = info
                identities[name] = identity
        return devices

    @staticmethod
    def _current_cache_key(
        access_token: str | None = None,
    ) -> tuple[tuple[str, str], ...]:
        from ..authentication.user import resolve_api_key
        from ..config import get_server_url

        token = resolve_api_key(access_token) or ""
        return (("X-Secret-Token", token), ("server_url", get_server_url().rstrip("/")))

    @classmethod
    def _get_devices(
        cls,
        force: bool = False,
        access_token: str | None = None,
        cache_key: tuple[tuple[str, str], ...] | None = None,
    ) -> dict[str, DeviceInfo]:
        if cache_key is None:
            cache_key = (
                cls._current_cache_key(access_token)
                if access_token
                else cls._current_cache_key()
            )
        with cls._lock:
            entry = cls._entries.get(cache_key)
            if (
                entry
                and (not force)
                and (time.monotonic() - entry[0] < DEVICE_CACHE_TTL_SECONDS)
            ):
                return entry[1]
            revision = object()
            cls._revisions[cache_key] = revision
        resources = cls._fetch(cache_key)
        devices = cls._index(resources)
        with cls._lock:
            if cls._revisions.get(cache_key) is revision:
                cls._entries[cache_key] = (time.monotonic(), devices)
        return devices

    @staticmethod
    def _fetch(cache_key: tuple[tuple[str, str], ...] | None) -> list[dict[str, Any]]:
        if cache_key is None or not dict(cache_key).get("X-Secret-Token"):
            raise ValueError(
                "Authentication required to load devices. Call QpiAIQuantumAuth.login(api_key) first."
            )
        import requests
        from ..config import HTTP_TIMEOUT, get_api_url

        try:
            response = requests.get(
                dict(cache_key).get("server_url", get_api_url("").rstrip("/"))
                + "/api/compute-resources",
                headers={"X-Secret-Token": dict(cache_key)["X-Secret-Token"]},
                timeout=HTTP_TIMEOUT,
            )
            response.raise_for_status()
            resources = response.json()
        except Exception as exc:
            raise ValueError(
                "Could not load the device list from QpiAI cloud. Check your connection and try again."
            ) from exc
        if isinstance(resources, dict):
            resources = (
                resources.get("compute_resources") or resources.get("data") or []
            )
        if not isinstance(resources, list):
            raise ValueError("Unexpected device list returned by QpiAI cloud.")
        return [r for r in resources if isinstance(r, dict)]


def get_device_info(device_name: str, access_token: str | None = None) -> DeviceInfo:
    """Read the device's normal execution profile."""
    return DeviceRegistry.get(device_name, access_token)


def list_devices(access_token: str | None = None) -> list[DeviceInfo]:
    """Return every device available to the current user."""
    return DeviceRegistry.available(access_token)


def get_device_method(device_name: str) -> str:
    """Return the execution method associated with a supported device."""
    return get_device_info(device_name).method


@dataclass(frozen=True)
class ResolvedExecutionRequest:
    """Validated device profile and requested outputs for a circuit execution."""

    device: DeviceInfo
    need_statevector: bool
    need_density_matrix: bool


def resolve_execution_request(
    device_name: str,
    *,
    method: str | None = None,
    need_statevector: bool | None = None,
    need_density_matrix: bool | None = None,
    access_token: str | None = None,
) -> ResolvedExecutionRequest:
    """Resolve omitted choices and reject conflicts against one device profile.

    Explicit False output flags remain False. Only None selects profile defaults.
    """
    device = get_device_info(device_name, access_token)
    if need_statevector is None:
        need_statevector = RESULT_STATEVECTOR in device.default_results
    if need_density_matrix is None:
        need_density_matrix = RESULT_DENSITY_MATRIX in device.default_results
    if method is not None and method != device.method:
        raise ValueError(
            f"method does not match the selected device profile. Device '{device_name}' requires method='{device.method}'."
        )
    if need_statevector and RESULT_STATEVECTOR not in device.supported_results:
        if RESULT_DENSITY_MATRIX in device.supported_results:
            raise ValueError(
                f"{device_name} returns density matrices only. For a pure result, convert it with DensityMatrix(...).to_statevector()."
            )
        raise ValueError(f"{device_name} cannot return a statevector.")
    if need_density_matrix and RESULT_DENSITY_MATRIX not in device.supported_results:
        raise ValueError(f"{device_name} cannot return a density matrix.")
    return ResolvedExecutionRequest(
        device=device,
        need_statevector=need_statevector,
        need_density_matrix=need_density_matrix,
    )


@dataclass(frozen=True)
class ResolvedBackend:
    ID: str
    backend_name: str
    address: str
    health_check: str
    run: str
