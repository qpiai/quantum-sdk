import time
import numpy as np
from typing import Union, Optional
from ..circuit.circuit import Circuit
from ..jobmanager import JobManager
import os


class Statevector:
    """
    Statevector class for representing quantum states.

    Similar to Qiskit's Statevector, this class represents a quantum state
    as a complex vector and provides methods for manipulation and analysis.

    Attributes:
        data: Complex numpy array representing the statevector
        num_qubits: Number of qubits in the system
    """

    def __init__(
        self,
        data: Union[list, np.ndarray, Circuit, "Statevector"],
        dims: list[int] | None = None,
        experiment_name: str = "Default Experiment",
        device_name: str = "QpiAI-QSV-Local",
    ):
        """
        Initialize a Statevector.

        Args:
            data: Input data, can be:
                  - List or numpy array of complex amplitudes
                  - Circuit object (will be simulated)
                  - Another Statevector (copy constructor)
            dims: Dimensions of subsystems (optional)
            experiment_name: Name for the experiment when simulating circuits (default: "Default Experiment")

        Example:
            >>> # From array
            >>> sv = Statevector([1, 0, 0, 1]) / np.sqrt(2)

            >>> # From circuit (default experiment name)
            >>> qc = Circuit(2, 2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)
            >>> sv = Statevector(qc)

            >>> # From circuit (custom experiment name)
            >>> sv = Statevector(qc, experiment_name="Default Experiment")
        """
        self.data: np.ndarray
        self.num_qubits: int
        if isinstance(data, Circuit):
            # Simulate the circuit to get statevector
            self._init_from_circuit(data, experiment_name, device_name)
        elif isinstance(data, Statevector):
            # Copy constructor
            self.data = data.data.copy()
            self.num_qubits = data.num_qubits
        else:
            # From array or list
            self._init_from_array(data)

        self.dims = dims or [2] * self.num_qubits

    def _init_from_array(self, data: list | np.ndarray):
        """Initialize from array or list."""
        # Handle nested list format [[c1], [c2], ...]
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                # Nested list format from JobManager
                self.data = np.array([item[0] for item in data], dtype=complex)
            elif isinstance(data[0], dict) and "real" in data[0] and "imag" in data[0]:
                # Dict format
                self.data = np.array(
                    [complex(item["real"], item["imag"]) for item in data],
                    dtype=complex,
                )
            else:
                self.data = np.array(data, dtype=complex)
        else:
            self.data = np.array(data, dtype=complex)

        # Normalize
        norm = np.linalg.norm(self.data)
        if norm > 0:
            self.data = self.data / norm

        # Determine number of qubits
        dim = len(self.data)
        self.num_qubits = int(np.log2(dim))

        if 2**self.num_qubits != dim:
            raise ValueError(f"Statevector length must be a power of 2, got {dim}")

    def _init_from_circuit(
        self,
        circuit: Circuit,
        experiment_name: str,
        device_name: str = "QpiAI-QSV-Local",
    ):
        """Initialize by simulating a circuit."""
        # Get API key from environment
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError(
                "API_KEY not found in environment. Please set it or pass it explicitly."
            )

        circuit_name = f"statevector_circuit_{int(time.time())}"
        result = circuit.run(
            shots=1024,
            experiment_name=experiment_name,
            device_name=device_name,
            need_statevector=True,
            circuit_name=circuit_name,
        )

        # Extract statevector from result
        if hasattr(result, "statevector") and result.statevector is not None:
            self._init_from_array(result.statevector)
        elif hasattr(result, "state") and result.state is not None:
            self._init_from_array(result.state)
        else:
            raise RuntimeError("Failed to get statevector from circuit simulation")

    @classmethod
    def from_label(cls, label: str) -> "Statevector":
        """
        Create a statevector from a computational basis state label.

        Args:
            label: String like '0', '1', '00', '01', '10', '11', etc.

        Returns:
            Statevector in the specified computational basis state

        Example:
            >>> sv = Statevector.from_label("10")  # |10⟩ state
        """
        num_qubits = len(label)
        dim = 2**num_qubits
        data = np.zeros(dim, dtype=complex)

        # Convert binary label to index
        index = int(label, 2)
        data[index] = 1.0

        return cls(data)

    @classmethod
    def from_circuit_object(
        cls,
        circuit: Circuit,
        experiment_name: str = "Default Experiment",
        device_name: str = "QpiAI-QSV-Local",
    ) -> "Statevector":
        """
        Create a statevector by simulating a quantum circuit.

        This method provides an alternative way to create a Statevector from a Circuit
        object, similar to how from_label creates from a basis state label.

        Args:
            circuit: QuantumCircuit object to simulate
            experiment_name: Name for the experiment when simulating (default: "Default Experiment")

        Returns:
            Statevector representing the final quantum state after applying the circuit

        Example:
            >>> qc = Circuit(2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)
            >>> state = Statevector.from_circuit_object(qc)
            >>> print(state.to_dict())  # {'00': (0.707...), '11': (0.707...)}
        """
        if not isinstance(circuit, Circuit):
            raise TypeError(f"Expected Circuit object, got {type(circuit).__name__}")

        # Create a new instance using the circuit
        return cls(circuit, experiment_name=experiment_name, device_name=device_name)

    def to_dict(self, decimals: int = 8) -> dict:
        """
        Convert statevector to dictionary representation.

        Args:
            decimals: Number of decimal places to round to (default: None, no rounding)

        Returns:
            Dictionary mapping basis states to complex amplitudes (as Python complex)

        Example:
            >>> state.to_dict()  # Full precision
            {'0': (0.7071067811865476+0j), '1': (4.33e-17+0.7071067811865475j)}

            >>> state.to_dict(decimals=8)  # Rounded
            {'0': (0.70710678+0j), '1': 0.70710678j}
        """
        result = {}
        for i, amplitude in enumerate(self.data):
            if np.abs(amplitude) > 1e-10:  # Only include non-zero amplitudes
                basis_state = format(i, f"0{self.num_qubits}b")
                amp = complex(amplitude)

                if decimals is not None:
                    # Round both real and imaginary parts
                    real_part = round(amp.real, decimals)
                    imag_part = round(amp.imag, decimals)

                    # Clean up values very close to zero
                    if abs(real_part) < 10 ** (-decimals):
                        real_part = 0.0
                    if abs(imag_part) < 10 ** (-decimals):
                        imag_part = 0.0

                    # Create clean complex number
                    if real_part == 0.0 and imag_part != 0.0:
                        amp = imag_part * 1j  # Pure imaginary (e.g., 0.707j)
                    elif imag_part == 0.0 and real_part != 0.0:
                        amp = real_part + 0j  # Pure real
                    elif real_part == 0.0 and imag_part == 0.0:
                        amp = 0j
                    else:
                        amp = complex(real_part, imag_part)

                result[basis_state] = amp
        return result

    def probabilities(self) -> np.ndarray:
        """
        Get measurement probabilities for all basis states.

        Returns:
            Array of probabilities
        """
        return np.abs(self.data) ** 2

    def probabilities_dict(self, decimals: int | None = None) -> dict:
        """
        Get measurement probabilities as a dictionary.

        Args:
            decimals: Number of decimal places to round to (default: None, no rounding)

        Returns:
            Dictionary mapping basis states to probabilities (as Python float)

        Example:
            >>> state.probabilities_dict()  # Full precision
            {'0': 0.5000000000000001, '1': 0.4999999999999999}

            >>> state.probabilities_dict(decimals=8)  # Rounded
            {'0': 0.5, '1': 0.5}
        """
        result = {}
        probs = self.probabilities()
        for i, prob in enumerate(probs):
            if prob > 1e-10:
                basis_state = format(i, f"0{self.num_qubits}b")
                # Convert numpy float to Python float for cleaner output
                prob_value = float(prob)

                if decimals is not None:
                    prob_value = round(prob_value, decimals)

                result[basis_state] = prob_value
        return result

    def purity(self) -> float:
        """
        Calculate the purity of the state.

        For a pure state, purity = 1.

        Returns:
            Purity value between 0 and 1
        """
        return float(np.abs(np.vdot(self.data, self.data)))

    def is_valid(self, atol: float = 1e-8) -> bool:
        """
        Check if the statevector is normalized.

        Args:
            atol: Absolute tolerance

        Returns:
            True if normalized, False otherwise
        """
        norm = np.linalg.norm(self.data)
        return np.abs(norm - 1.0) < atol

    def to_density_matrix(self):
        """
        Convert statevector to density matrix.

        Returns:
            DensityMatrix object
        """
        from .density_matrix import DensityMatrix

        rho = np.outer(self.data, np.conj(self.data))
        return DensityMatrix(rho)

    def evolve(self, other: Circuit | np.ndarray) -> "Statevector":
        """
        Evolve the statevector by a circuit or unitary matrix.

        Args:
            other: Circuit or unitary matrix to apply

        Returns:
            New evolved Statevector
        """
        if isinstance(other, Circuit):
            from qpiai_quantum.simulator.statevector import StatevectorSimulator

            simulator = StatevectorSimulator()
            result = simulator.run(other, initial_state=self.data)
            return Statevector(result.statevector)
        else:
            # Assume it's a unitary matrix
            new_data = other @ self.data
            return Statevector(new_data)

    def __repr__(self) -> str:
        """String representation in Qiskit-style format."""
        # Format the data array similar to numpy's array repr
        dims_str = f"dims={tuple(self.dims)}"

        # Use numpy's array formatting
        with np.printoptions(precision=8, suppress=True, threshold=1000):
            array_str = repr(self.data)

        return f"Statevector({array_str},\n            {dims_str})"

    def __len__(self) -> int:
        """Return the dimension of the statevector."""
        return len(self.data)

    def __getitem__(self, key):
        """Index into the statevector."""
        return self.data[key]

    def __array__(self) -> np.ndarray:
        """Return as numpy array."""
        return self.data

    def copy(self) -> "Statevector":
        """Create a copy of the statevector."""
        return Statevector(self.data.copy())

    def __truediv__(self, scalar) -> "Statevector":
        """Division operator to maintain Statevector type."""
        new_data = self.data / scalar
        return Statevector(new_data)

    def __mul__(self, scalar) -> "Statevector":
        """Multiplication operator to maintain Statevector type."""
        new_data = self.data * scalar
        return Statevector(new_data)

    def __rmul__(self, scalar) -> "Statevector":
        """Right multiplication operator."""
        return self.__mul__(scalar)

    def __add__(self, other) -> "Statevector":
        """Addition operator."""
        if isinstance(other, Statevector):
            new_data = self.data + other.data
        else:
            new_data = self.data + other
        return Statevector(new_data)

    def __sub__(self, other) -> "Statevector":
        """Subtraction operator."""
        if isinstance(other, Statevector):
            new_data = self.data - other.data
        else:
            new_data = self.data - other
        return Statevector(new_data)
