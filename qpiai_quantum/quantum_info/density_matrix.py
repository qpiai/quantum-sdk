"""DensityMatrix class for quantum state representation."""

import time
import numpy as np
from typing import Union, List, Optional
from ..circuit.circuit import Circuit
from ..jobmanager import JobManager
from ..formalism.density_matrix.base_density_matrix import BaseDensityMatrix
import os

try:
    from ..formalism import DensityMatrix as FormalismDensityMatrix

    _HAS_FORMALISM = True
except ImportError:
    _HAS_FORMALISM = False


class DensityMatrix(BaseDensityMatrix):
    def __init__(
        self,
        data: Union[list, np.ndarray, Circuit, "DensityMatrix"],
        dims: list[int] | None = None,
        experiment_name: str = "Default Experiment",
        device_name: str = "QpiAI-QSV-Local",
    ):
        self.data: np.ndarray
        self.num_qubits: int
        if isinstance(data, Circuit):
            # Simulate the circuit to get density matrix
            self._init_from_circuit(data, experiment_name, device_name)
        elif isinstance(data, DensityMatrix):
            # Copy constructor
            self.data = data.data.copy()
            self.num_qubits = data.num_qubits
        else:
            # From array
            self._init_from_array(data)

        self.dims = dims or [2] * self.num_qubits

    def _get_data(self) -> np.ndarray:
        """Return the underlying density matrix data."""
        return self.data

    def _get_num_qubits(self) -> int:
        """Return the number of qubits."""
        return self.num_qubits

    def _init_from_array(self, data: list | np.ndarray):
        """Initialize from array."""
        # Handle nested list format
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                # Could be density matrix or nested statevector
                if isinstance(data[0], list):
                    # Check for dict-based density matrix
                    if (
                        isinstance(data[0][0], dict)
                        and "real" in data[0][0]
                        and "imag" in data[0][0]
                    ):
                        self._init_from_dict_density_matrix(data)
                        return

                    # 2D numeric density matrix
                    if isinstance(data[0][0], list):
                        self.data = np.array(data, dtype=complex)
                else:
                    # Nested statevector [[c1], [c2], ...]
                    statevector = np.array([item[0] for item in data], dtype=complex)
                    self._init_from_statevector(statevector)
                    return
            elif isinstance(data[0], dict) and "real" in data[0] and "imag" in data[0]:
                # Dict format statevector
                statevector = np.array(
                    [complex(item["real"], item["imag"]) for item in data],
                    dtype=complex,
                )
                self._init_from_statevector(statevector)
                return
            else:
                self.data = np.array(data, dtype=complex)
        else:
            self.data = np.array(data, dtype=complex)

        # Check dimensions
        if self.data.ndim == 1:
            # It's a statevector, convert to density matrix
            self._init_from_statevector(self.data)
        elif self.data.ndim == 2:
            # It's already a density matrix
            if self.data.shape[0] != self.data.shape[1]:
                raise ValueError(
                    f"Density matrix must be square, got shape {self.data.shape}"
                )

            dim = self.data.shape[0]
            self.num_qubits = int(np.log2(dim))

            if 2**self.num_qubits != dim:
                raise ValueError(
                    f"Density matrix dimension must be a power of 2, got {dim}"
                )
        else:
            raise ValueError(f"Data must be 1D or 2D, got {self.data.ndim}D")

    def _init_from_statevector(self, statevector: np.ndarray):
        """Initialize from a statevector."""
        # Normalize
        norm = np.linalg.norm(statevector)
        if norm > 0:
            statevector = statevector / norm

        # Create density matrix: ρ = |ψ⟩⟨ψ|
        self.data = np.outer(statevector, np.conj(statevector))

        # Determine number of qubits
        dim = len(statevector)
        self.num_qubits = int(np.log2(dim))

        if 2**self.num_qubits != dim:
            raise ValueError(f"Statevector length must be a power of 2, got {dim}")

    def _init_from_dict_density_matrix(self, data: list):
        """
        Initialize from density matrix in dict format:
        [
            [ {'real': r, 'imag': i}, ... ],
            ...
        ]
        """
        if not isinstance(data, list) or not isinstance(data[0], list):
            raise ValueError("Invalid density matrix dict format")

        dim = len(data)

        matrix = np.zeros((dim, dim), dtype=np.complex128)

        for i in range(dim):
            for j in range(dim):
                elem = data[i][j]

                if (
                    not isinstance(elem, dict)
                    or "real" not in elem
                    or "imag" not in elem
                ):
                    raise ValueError("Invalid element in density matrix dict format")

                matrix[i, j] = complex(elem["real"], elem["imag"])

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Density matrix must be square, got {matrix.shape}")

        self.data = matrix

        self.num_qubits = int(np.log2(dim))

        if 2**self.num_qubits != dim:
            raise ValueError(
                f"Density matrix dimension must be a power of 2, got {dim}"
            )

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

        circuit_name = f"density_matrix_circuit_{int(time.time())}"
        result = circuit.run(
            shots=1024,
            experiment_name=experiment_name,
            device_name=device_name,
            need_statevector=True,
            circuit_name=circuit_name,
        )

        # Extract statevector and convert to density matrix
        if hasattr(result, "statevector") and result.statevector is not None:
            self._init_from_array(result.statevector)
        elif hasattr(result, "state") and result.state is not None:
            self._init_from_array(result.state)
        elif hasattr(result, "density_matrix") and result.density_matrix is not None:
            self._init_from_array(result.density_matrix)
        else:
            raise RuntimeError("Failed to get state from circuit simulation")

    @classmethod
    def from_label(cls, label: str) -> "DensityMatrix":
        """
        Create a density matrix from a computational basis state label.

        Args:
            label: String like '0', '1', '00', '01', '10', '11', etc.

        Returns:
            DensityMatrix in the specified computational basis state

        Example:
            >>> dm = DensityMatrix.from_label("10")  # |10⟩⟨10|
        """
        num_qubits = len(label)
        dim = 2**num_qubits
        statevector = np.zeros(dim, dtype=complex)

        # Convert binary label to index
        index = int(label, 2)
        statevector[index] = 1.0

        return cls(statevector)

    @classmethod
    def from_circuit_object(
        cls,
        circuit: Circuit,
        experiment_name: str = "Default Experiment",
        device_name: str = "QpiAI-QSV-Local",
    ) -> "DensityMatrix":
        """
        Create a density matrix by simulating a quantum circuit.

        This method provides an alternative way to create a DensityMatrix from a Circuit
        object, similar to how from_label creates from a basis state label.

        Args:
            circuit: QuantumCircuit object to simulate
            experiment_name: Name for the experiment when simulating (default: "Default Experiment")
            device_name: Name of the quantum device or simulator to run on (default: "QpiAI-QSV-Local")

        Returns:
            DensityMatrix representing the final quantum state after applying the circuit

        Example:
            >>> qc = Circuit(2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)
            >>> dm = DensityMatrix.from_circuit_object(qc)
            >>> print(dm.is_pure())  # True (pure Bell state)
        """
        if not isinstance(circuit, Circuit):
            raise TypeError(f"Expected Circuit object, got {type(circuit).__name__}")

        # Create a new instance using the circuit
        return cls(circuit, experiment_name=experiment_name, device_name=device_name)

    def probabilities(self) -> np.ndarray:
        """
        Get measurement probabilities for all basis states.

        Returns:
            Array of probabilities (diagonal elements of density matrix)
        """
        return np.real(np.diag(self.data))

    def probabilities_dict(self, decimals: int | None = None) -> dict:
        """
        Get measurement probabilities as a dictionary.

        Args:
            decimals: Number of decimal places to round to (default: None, no rounding)

        Returns:
            Dictionary mapping basis states to probabilities (as Python float)

        Example:
            >>> dm.probabilities_dict()  # Full precision
            {'0': 0.5000000000000001, '1': 0.4999999999999999}

            >>> dm.probabilities_dict(decimals=8)  # Rounded
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

    # Methods purity(), trace(), is_valid(), von_neumann_entropy(), is_pure(), fidelity()
    # are inherited from BaseDensityMatrix

    def entropy(self) -> float:
        """
        Alias for von_neumann_entropy() for backward compatibility.

        Returns:
            Entropy value (0 for pure states)
        """
        return self.von_neumann_entropy()

    def to_statevector(self):
        """
        Convert to statevector (only for pure states).

        Returns:
            Statevector object

        Raises:
            ValueError: If state is not pure
        """
        if not self.is_pure():
            raise ValueError("Cannot convert mixed state to statevector")

        from .statevector import Statevector

        # Get the eigenvector corresponding to eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eigh(self.data)
        max_idx = np.argmax(eigenvalues)

        return Statevector(eigenvectors[:, max_idx])

    # is_pure() is inherited from BaseDensityMatrix

    def __repr__(self) -> str:
        """String representation in Qiskit-style format."""
        # Format the data array similar to numpy's array repr
        dims_str = f"dims={tuple(self.dims)}"

        # Use numpy's array formatting
        with np.printoptions(precision=8, suppress=True, threshold=1000):
            array_str = repr(self.data)

        return f"DensityMatrix({array_str},\n              {dims_str})"

    def __array__(self) -> np.ndarray:
        """Return as numpy array."""
        return self.data

    def copy(self) -> "DensityMatrix":
        """Create a copy of the density matrix."""
        return DensityMatrix(self.data.copy())

    def to_formalism(self):
        """
        Convert to formalism.DensityMatrix for advanced operations.

        The formalism module provides advanced quantum operations like
        noise channels (ADC, depolarizing), etc.

        Returns:
            formalism.DensityMatrix object

        Example:
            >>> dm = DensityMatrix(circuit)
            >>> fdm = dm.to_formalism()
            >>> noisy = fdm.depol(0.1)  # Apply depolarizing noise
        """
        if not _HAS_FORMALISM:
            raise ImportError("formalism module not available")

        return FormalismDensityMatrix(self.data)

    @classmethod
    def from_formalism(cls, formalism_dm) -> "DensityMatrix":
        """
        Create from a formalism.DensityMatrix object.

        Args:
            formalism_dm: A formalism.DensityMatrix object

        Returns:
            quantum_info.DensityMatrix object

        Example:
            >>> from qpiai_quantum.formalism import DensityMatrix as FDM
            >>> fdm = FDM(some_matrix)
            >>> fdm_noisy = fdm.depol(0.1)
            >>> dm = DensityMatrix.from_formalism(fdm_noisy)
            >>> plot_bloch_multivector(dm)
        """
        return cls(formalism_dm.state)

    def __truediv__(self, scalar) -> "DensityMatrix":
        """Division operator to maintain DensityMatrix type."""
        new_data = self.data / scalar
        return DensityMatrix(new_data)

    def __mul__(self, scalar) -> "DensityMatrix":
        """Multiplication operator to maintain DensityMatrix type."""
        new_data = self.data * scalar
        return DensityMatrix(new_data)

    def __rmul__(self, scalar) -> "DensityMatrix":
        """Right multiplication operator."""
        return self.__mul__(scalar)

    def __add__(self, other) -> "DensityMatrix":
        """Addition operator."""
        if isinstance(other, DensityMatrix):
            new_data = self.data + other.data
        else:
            new_data = self.data + other
        return DensityMatrix(new_data)

    def __sub__(self, other) -> "DensityMatrix":
        """Subtraction operator."""
        if isinstance(other, DensityMatrix):
            new_data = self.data - other.data
        else:
            new_data = self.data - other
        return DensityMatrix(new_data)
