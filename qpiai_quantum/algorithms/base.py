from abc import ABC, abstractmethod
from typing import Optional, Any
import time
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager import Backend
from qpiai_quantum.results.base_result import BaseQuantumResult
from qpiai_quantum.circuit.exceptions import CircuitError


class QuantumAlgorithm(ABC):
    def __init__(self, num_qubits: int, name: str = "QuantumAlgorithm"):
        self.num_qubits = num_qubits
        self.name = name
        self.circuit: Circuit | None = None
        self.description = "Base quantum algorithm"

    @abstractmethod
    def build_circuit(self, *args, **kwargs) -> Circuit:
        pass

    def run(
        self,
        shots: int = 1024,
        experiment_name: str = "Default Experiment",
        need_statevector: bool = True,
        need_density_matrix: bool = False,
        device_name: str = "QpiAI-QSV-Local",
        reverse_bits: bool = False,
        **kwargs,
    ) -> "BaseQuantumResult":
        if self.circuit is None:
            raise ValueError(
                f"Circuit not built. Call build_circuit() first for {self.name}"
            )

        return self.circuit.run(
            shots=shots,
            experiment_name=experiment_name,
            need_statevector=need_statevector,
            need_density_matrix=need_density_matrix,
            device_name=device_name,
            reverse_bits=reverse_bits,
            **kwargs,
        )

    def _backend_to_method_and_device(self, backend: Backend) -> tuple[str, str]:
        return backend.to_method_and_device()

    def visualize(
        self,
        plot: str = "circuit",
        result: BaseQuantumResult | None = None,
        **kwargs,
    ):
        if self.circuit is None:
            raise ValueError(
                f"Circuit not built. Call build_circuit() first for {self.name}"
            )

        plot = plot.lower()

        if plot == "circuit":
            self.circuit.show()

        elif plot in ["histogram", "probabilities", "state", "counts"]:
            if result is None:
                shots = kwargs.get("shots", 1024)
                result = self.run(shots=shots)

            self._plot_results(result, plot_type=plot, **kwargs)

        else:
            raise ValueError(
                f"Unknown plot type: '{plot}'. "
                f"Valid options: 'circuit', 'histogram', 'probabilities', 'state'"
            )

    def _plot_results(
        self, result: BaseQuantumResult, plot_type: str = "histogram", **kwargs
    ):
        import matplotlib.pyplot as plt

        data = result.get()
        counts = data.get("counts", {})

        if not counts:
            print("No measurement data available to plot")
            return

        figsize = kwargs.get("figsize", (10, 6))
        title = kwargs.get("title", None)
        _, ax = plt.subplots(figsize=figsize)

        if plot_type in ["histogram", "counts"]:
            states = list(counts.keys())
            values = list(counts.values())

            ax.bar(states, values, color="skyblue", edgecolor="black", alpha=0.7)
            ax.set_xlabel("Quantum State", fontsize=12)
            ax.set_ylabel("Counts", fontsize=12)
            ax.set_title(
                title or f"{self.name} - Measurement Counts",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(axis="y", alpha=0.3)

            if len(states) > 8:
                plt.xticks(rotation=45, ha="right")

        elif plot_type == "probabilities":
            total_shots = sum(counts.values())
            states = list(counts.keys())
            probabilities = [count / total_shots for count in counts.values()]

            ax.bar(
                states, probabilities, color="lightcoral", edgecolor="black", alpha=0.7
            )
            ax.set_xlabel("Quantum State", fontsize=12)
            ax.set_ylabel("Probability", fontsize=12)
            ax.set_title(
                title or f"{self.name} - Probability Distribution",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.3)

            for i, prob in enumerate(probabilities):
                if prob > 0.02:  # NOTE: Only show labels for significant probabilities
                    ax.text(
                        i, prob, f"{prob:.1%}", ha="center", va="bottom", fontsize=9
                    )

            if len(states) > 8:
                plt.xticks(rotation=45, ha="right")

        elif plot_type == "state":
            state_vector = data.get("state", None)

            if state_vector is not None:
                amplitudes = [abs(complex(s[0])) for s in state_vector]
                states = [
                    bin(i)[2:].zfill(self.num_qubits) for i in range(len(amplitudes))
                ]
                ax.bar(
                    states, amplitudes, color="lightgreen", edgecolor="black", alpha=0.7
                )
                ax.set_xlabel("Quantum State", fontsize=12)
                ax.set_ylabel("Amplitude", fontsize=12)
                ax.set_title(
                    title or f"{self.name} - State Vector Amplitudes",
                    fontsize=14,
                    fontweight="bold",
                )
                ax.grid(axis="y", alpha=0.3)

                if len(states) > 8:
                    plt.xticks(rotation=45, ha="right")
            else:
                print("State vector not available, showing probabilities instead")
                self._plot_results(result, plot_type="probabilities", **kwargs)
                return

        plt.tight_layout()
        plt.show()

    def to_qasm(self):
        """
        Convert the algorithm's circuit to OpenQASM representation.

        Returns:
            str | list[str]: The OpenQASM representation of the circuit
        """
        if self.circuit is None:
            raise ValueError(
                f"Circuit not built. Call build_circuit() first for {self.name}"
            )

        return self.circuit.to_qasm()

    def get_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "description": self.description,
            "circuit_built": self.circuit is not None,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"num_qubits={self.num_qubits}, "
            f"circuit_built={self.circuit is not None})"
        )
