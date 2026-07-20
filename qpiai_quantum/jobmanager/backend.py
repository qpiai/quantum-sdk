from dataclasses import dataclass
from enum import Enum


class Backend(Enum):
    STATEVECTOR_SIMULATOR_CPU = "QpiAI-QSV-Simulator"
    DENSITY_MATRIX_SIMULATOR_CPU = "QpiAI-QDM-Simulator"
    INDUS_QPU = "QpiAI-Indus-1"
    TENSOR_NETWORK_SIMULATOR_CPU = "QpiAI-QTN-Simulator"
    LOCAL_SIMULATOR = "QpiAI-QSV-Local"

    def to_method_and_device(self) -> tuple[str, str]:
        """
        Convert Backend enum to method and device name.
        """
        mapping = {
            Backend.STATEVECTOR_SIMULATOR_CPU: ("statevector", "QpiAI-QSV-Simulator"),
            Backend.DENSITY_MATRIX_SIMULATOR_CPU: (
                "density_matrix",
                "QpiAI-QDM-Simulator",
            ),
            Backend.INDUS_QPU: ("qpu", "QpiAI-Indus-1"),
            Backend.TENSOR_NETWORK_SIMULATOR_CPU: (
                "tensor_network",
                "QpiAI-QTN-Simulator",
            ),
            Backend.LOCAL_SIMULATOR: ("statevector", "QpiAI-QSV-Local"),
        }
        return mapping.get(self, ("statevector", self.value))


@dataclass(frozen=True)
class ResolvedBackend:
    ID: str
    backend_name: str
    address: str
    health_check: str
    run: str
