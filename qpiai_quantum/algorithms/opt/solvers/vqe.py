from typing import Any, Dict, Optional, Callable, Union, List, Tuple
import numpy as np
import warnings
from dataclasses import dataclass
from qpiai_quantum.results.base_result import BaseQuantumResult
from ....circuit import Circuit
from ....jobmanager import JobManager
from ....jobmanager.job_result import JobResult
from ....icr.circuitoperation import OperationType, CircuitOperation
from ...base import QuantumAlgorithm
from ..ansatz.hardware_efficient import hardware_efficient_ansatz, two_local_ansatz
from ..ansatz.standard import standard_vqe_ansatz, count_parameters
from ...utils.optimizers import (
    adam_optimize,
    gradient_descent_optimize,
    nelder_mead_optimize,
    l_bfgs_b_optimize,
    spsa_optimize,
    slsqp_optimize,
    genetic_algorithm_optimize,
    particle_swarm_optimize,
    cobyla_optimize,
    powell_optimize,
)


@dataclass
class VQEResult:
    """Result from VQE execution. Supports both attribute and dictionary-like access."""

    optimal_parameters: List[float]
    optimal_energy: float
    energy_history: List[float]
    param_history: List[List[float]]
    bitstring: str
    counts: Dict[str, int]
    metadata: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-like access for backward compatibility."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{key}' not found in VQEResult")

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for backward compatibility."""
        return hasattr(self, key)


class VQESolver(QuantumAlgorithm):
    """
    Variational Quantum Eigensolver (VQE) implementation using QpiAI Quantum SDK.

    VQE is a hybrid quantum-classical algorithm that finds the ground state energy
    of a Hamiltonian by optimizing a parameterized quantum circuit (ansatz).

    This implementation follows the standard run() pattern used by other algorithms,
    ensuring consistency across the SDK.

    Args:
        n_qubits: Number of qubits in the system
        ansatz: Quantum circuit ansatz ('standard', 'hardware_efficient', 'two_local', or callable, default: 'standard')
        optimizer: Optimization method ('adam', 'gradient_descent', 'cobyla', 'spsa', etc.)
        max_iterations: Maximum number of optimization iterations (default: 100)
        initial_point: Initial parameters for optimization
        verbose: Enable printing of optimization progress and completion summaries (default: True)
        name: Name of this VQE instance

    Note:
        max_iterations is ONLY set here in __init__(). Use with_max_iterations()
        to change it.

    Example:
        >>> from qpiai_quantum.algorithms.opt.solvers import VQESolver
        >>> from qpiai_quantum.algorithms.opt.problems import MaxCutProblem
        >>>
        >>> # Define problem
        >>> problem = MaxCutProblem(graph)
        >>>
        >>> # Create solver with standard ansatz (default)
        >>> solver = VQESolver(
        ...     n_qubits=problem.n_qubits,
        ...     ansatz="standard",
        ...     optimizer="spsa",
        ...     max_iterations=5,
        ... )
        >>>
        >>> # Or use hardware_efficient ansatz
        >>> solver = VQESolver(
        ...     n_qubits=problem.n_qubits,
        ...     ansatz="hardware_efficient",
        ...     optimizer="cobyla",
        ...     max_iterations=10,
        ... )
        >>>
        >>> # Run optimization using standard run() method
        >>> result = solver.run(
        ...     hamiltonian=problem,
        ...     shots=10000,
        ...     experiment_name="Default Experiment",
        ...     method="statevector",
        ...     device_name="QpiAI-QSV-Local",
        ... )
        >>>
        >>> print(f"Ground state energy: {result.optimal_energy}")
    """

    def __init__(
        self,
        n_qubits: int,
        hamiltonian: Any = None,
        ansatz: Union[str, Callable] = "standard",
        optimizer: str = "adam",
        max_iterations: int = 100,
        initial_point: Optional[np.ndarray] = None,
        verbose: bool = True,
        name: str = "VQE",
    ):
        """Initialize VQE solver with algorithm parameters."""
        super().__init__(num_qubits=n_qubits, name=name)
        self.ansatz = ansatz
        self.optimizer = optimizer.lower()
        self.max_iterations = max_iterations
        self.initial_point = initial_point
        self.hamiltonian = hamiltonian
        self.verbose = verbose
        self._executor: Optional[JobManager] = None
        self.shots = 1024
        self.description = (
            "Variational Quantum Eigensolver for finding ground state energies (verbose progress printing is enabled by default)"
        )

    def build_circuit(self, parameters: Optional[np.ndarray] = None) -> Circuit:
        """
        Public method: Build the VQE circuit with given parameters.

        Args:
            parameters: Circuit parameters (optional)

        Returns:
            Parameterized quantum circuit
        """
        ansatz = self._get_ansatz()
        self.circuit = self._build_circuit(ansatz, parameters)
        return self.circuit

    def _get_ansatz(self) -> Circuit:
        """
        Private method: Select and validate ansatz based on configuration.

        Returns:
            Validated ansatz circuit
        """
        if self.ansatz == "standard" or self.ansatz is None:
            # Use standard VQE ansatz (default)
            ansatz = standard_vqe_ansatz(
                self.num_qubits, layers=2, entanglement="linear"
            )
        elif self.ansatz == "hardware_efficient":
            ansatz = hardware_efficient_ansatz(self.num_qubits, depth=2)
        elif self.ansatz == "two_local":
            ansatz = two_local_ansatz(self.num_qubits)
        elif callable(self.ansatz):
            ansatz = self.ansatz(self.num_qubits)
        else:
            raise ValueError(f"Unknown ansatz type: {self.ansatz}")

        if not isinstance(ansatz, Circuit):
            raise ValueError("Ansatz must be a Circuit object")

        if not hasattr(ansatz, "icr") or not ansatz.icr:
            raise ValueError("Ansatz circuit must have ICR representation")

        if not hasattr(ansatz.icr, "evolve"):
            raise ValueError("Ansatz circuit must have valid ICR evolution operations")

        return ansatz

    def _build_circuit(
        self, ansatz: Circuit, parameters: Optional[np.ndarray] = None
    ) -> Circuit:
        """
        Private method: Apply parameters to ansatz and build executable circuit.

        Args:
            ansatz: Pre-validated ansatz circuit template
            parameters: Parameter values to apply (optional)

        Returns:
            Parameterized circuit ready for execution
        """
        circuit = Circuit(self.num_qubits)
        param_idx = 0

        try:
            if ansatz.icr is None:
                raise ValueError("Circuit has no ICR representation")

            if not hasattr(ansatz.icr, "evolve"):
                raise ValueError("ICR has no evolve attribute")

            for op in ansatz.icr.evolve:
                if isinstance(op, CircuitOperation):
                    if op.operation_type == OperationType.N_QUBIT_PARAMETRIC:
                        if parameters is not None:
                            param_value = float(parameters[param_idx])
                            new_op = CircuitOperation(
                                operation_type=op.operation_type,
                                gate_name=op.gate_name,
                                qubits=op.qubits,
                                params=[param_value],
                                clbits=op.clbits,
                            )
                            circuit.add_operation(new_op)
                            param_idx += 1
                        else:
                            circuit.add_operation(op)
                    else:
                        circuit.add_operation(op)

            circuit.measure_all()
            return circuit

        except (AttributeError, TypeError, IndexError) as e:
            raise ValueError(f"Error building circuit: {str(e)}")

    def _compute_expectation(self, result: BaseQuantumResult) -> float:
        """
        Compute expectation value ⟨ψ|H|ψ⟩ from measurement results.
        """
        if not result or not result.counts:
            raise ValueError("Invalid execution result")

        counts = {k: int(v) for k, v in result.counts.items()}

        if self.hamiltonian is None:
            raise ValueError("Hamiltonian not set")

        if not hasattr(self.hamiltonian, "get_hamiltonian_terms"):
            raise ValueError("Hamiltonian must have get_hamiltonian_terms method")

        terms = self.hamiltonian.get_hamiltonian_terms()
        if terms:
            max_qubit_idx = max((qi for ops, _ in terms for qi, _ in ops), default=0)
            expected_n_qubits = max_qubit_idx + 1
        else:
            expected_n_qubits = 1

        # NOTE: Pad bitstrings to expected length
        padded_counts = {}
        for bitstr, count in counts.items():
            if len(bitstr) < expected_n_qubits:
                bitstr = bitstr.zfill(expected_n_qubits)
            padded_counts[bitstr] = count
        counts = padded_counts

        total_shots = sum(counts.values())
        if total_shots == 0:
            raise ValueError("No measurements recorded")

        expectation = 0.0
        for ops, coeff in terms:
            if not ops:
                expectation += coeff
                continue

            term_exp = 0.0
            for bitstr, count in counts.items():
                eigenvalue = 1.0
                bitstr = str(bitstr)
                for qubit_idx, op_name in ops:
                    bit = int(bitstr[-(qubit_idx + 1)])
                    if op_name == "Z":
                        eigenvalue *= 1.0 if bit == 0 else -1.0
                    elif op_name == "I":
                        pass
                    elif op_name in ["X", "Y"]:
                        raise NotImplementedError(
                            f"{op_name} operator requires basis rotation measurements"
                        )
                    else:
                        raise ValueError(f"Unknown Pauli operator: {op_name}")
                term_exp += eigenvalue * count
            expectation += coeff * term_exp / total_shots

        return float(expectation)

    def _compute_gradient(
        self, params: np.ndarray, cost_function: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """
        Compute gradient using parameter shift rule.

        WARNING: This requires 2N circuit executions where N = number of parameters!

        For a parameterized gate U(θ), the gradient is:
        ∂⟨H⟩/∂θ = [⟨H⟩(θ + π/2) - ⟨H⟩(θ - π/2)] / 2

        Args:
            params: Current parameter values
            cost_function: Objective function to differentiate

        Returns:
            Gradient vector
        """
        grad = np.zeros_like(params)
        shift = np.pi / 2

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += shift
            energy_plus = cost_function(params_plus)

            params_minus = params.copy()
            params_minus[i] -= shift
            energy_minus = cost_function(params_minus)

            grad[i] = (energy_plus - energy_minus) / 2

        return grad

    def _get_job_manager(self) -> JobManager:
        """Get or create quantum job manager."""
        if self._executor is None:
            self._executor = JobManager()
        return self._executor

    def _execute_circuit(
        self,
        circuit: Circuit,
        method: Optional[str] = None,
        device_name: Optional[str] = None,
        shots: Optional[int] = None,
        **kwargs,
    ) -> BaseQuantumResult:
        """
        Execute a quantum circuit using the base class run() method.

        This method uses the run() execution from the base class,
        ensuring consistent behavior with other algorithms in the SDK.

        Args:
            circuit: Circuit to execute
            method: Simulation method ('statevector', 'density_matrix', 'tensor_network')
            device_name: Device name ('QpiAI-QSV-Local', 'QpiAI-QSV-Simulator', 'QpiAI-QDM-Simulator', 'QpiAI-QTN-Simulator', etc)
            shots: Number of measurement shots
            **kwargs: Additional parameters passed to run()

        Returns:
            Execution result with measurement counts
        """
        actual_shots = shots if shots is not None else self.shots

        # Set the circuit on this instance for base class run() to use
        self.circuit = circuit

        # Use provided method/device_name or defaults
        actual_method = (
            method if method is not None else getattr(self, "method", "statevector")
        )
        actual_device_name = (
            device_name
            if device_name is not None
            else getattr(self, "device_name", "QpiAI-QSV-Local")
        )

        # Extract run parameters from kwargs
        run_kwargs = {
            "shots": actual_shots,
            "method": actual_method,
            "device_name": actual_device_name,
        }

        # Add other valid run parameters
        valid_run_params = ["experiment_name", "reverse_bits"]
        for param in valid_run_params:
            if param in kwargs:
                run_kwargs[param] = kwargs[param]

        # Call parent class run() method (QuantumAlgorithm.run())
        return super().run(**run_kwargs)  # type: ignore

    def _print_optimization_info(self, n_params: int, optimizer_name: str):
        """
        Print detailed information about the optimization before it starts.

        Args:
            n_params: Number of parameters to optimize
            optimizer_name: Name of the optimizer being used
        """
        print("\n" + "=" * 70)
        print("VQE OPTIMIZATION STARTING")
        print("=" * 70)
        print(f"Optimizer:        {optimizer_name.upper()}")
        print(f"Parameters:       {n_params}")
        print(f"Max Iterations:   {self.max_iterations}")
        print(f"Shots per eval:   {self.shots}")

        # Calculate expected circuit executions based on optimizer type
        GRADIENT_FREE_OPTIMIZERS = {
            "cobyla",
            "nelder_mead",
            "spsa",
            "slsqp",
            "differential_evolution",
            "genetic_algorithm",
            "particle_swarm",
        }

        if optimizer_name in GRADIENT_FREE_OPTIMIZERS:
            if optimizer_name == "cobyla":
                # COBYLA typically does (n_params + 2) evaluations per iteration
                base_per_iteration = n_params + 2
                base_evaluations = self.max_iterations * base_per_iteration

                if self.max_iterations <= 5:
                    buffer = 10 + self.max_iterations * 2
                elif self.max_iterations <= 20:
                    buffer = 20 + self.max_iterations
                else:
                    buffer = 50

                expected_evals = base_evaluations + buffer
                print(
                    f"Expected Evals:   ~{expected_evals} (may stop early if converged)"
                )
                print(
                    f"  |-- {self.max_iterations} iterations x {base_per_iteration} evals/iter + {buffer} buffer"
                )

            elif optimizer_name == "spsa":
                # SPSA does 3 evaluations per iteration (baseline + 2 perturbed)
                expected_evals = 1 + self.max_iterations * 3
                print(
                    f"Expected Evals:   ~{expected_evals} (may stop early if converged)"
                )
                print(
                    f"  |-- 1 initial + {self.max_iterations} iterations x 3 evals/iter"
                )

            elif optimizer_name == "nelder_mead":
                # Nelder-Mead is unpredictable but typically 1-3 per iteration
                expected_evals_min = self.max_iterations
                expected_evals_max = self.max_iterations * 3
                print(
                    f"Expected Evals:   {expected_evals_min}-{expected_evals_max} (variable)"
                )

            elif optimizer_name in [
                "differential_evolution",
                "genetic_algorithm",
                "particle_swarm",
            ]:
                # Population-based methods
                if optimizer_name == "differential_evolution":
                    popsize = 15  # default
                    expected_evals = popsize * n_params * self.max_iterations
                    print(f"Expected Evals:   ~{expected_evals}")
                    print(
                        f"  |-- Population-based: {popsize} x {n_params} x {self.max_iterations}"
                    )
                elif optimizer_name == "genetic_algorithm":
                    popsize = 50  # default
                    expected_evals = popsize * self.max_iterations
                    print(f"Expected Evals:   ~{expected_evals}")
                    print(
                        f"  |-- Population: {popsize} x {self.max_iterations} generations"
                    )
                else:  # particle_swarm
                    n_particles = 30  # default
                    expected_evals = n_particles * self.max_iterations
                    print(f"Expected Evals:   ~{expected_evals}")
                    print(
                        f"  |-- Swarm: {n_particles} particles x {self.max_iterations} iterations"
                    )
            else:
                expected_evals = self.max_iterations
                print(f"Expected Evals:   ~{expected_evals}")
        else:
            # Gradient-based optimizers
            expected_evals = self.max_iterations * (2 * n_params + 1) + 1
            print(f"Expected Evals:   ~{expected_evals} (GRADIENT-BASED - EXPENSIVE!)")
            print(
                f"  |-- {self.max_iterations} iter x (2x{n_params} for gradient + 1 for energy) + 1 final"
            )
            print(
                "  ⚠️  Consider gradient-free optimizers (cobyla, spsa) for fewer evaluations"
            )

        print("\nNote: Optimizer may stop earlier if convergence criteria are met.")
        print("=" * 70 + "\n")

    def _optimize_parameters(
        self,
        objective: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        n_params: int,
        **kwargs,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Run optimization loop to minimize objective function.

        Supports optimizers from both gradient-free and gradient-based categories:

        Gradient-Free:
            - cobyla, nelder_mead, spsa, slsqp
            - differential_evolution, genetic_algorithm, particle_swarm

        Gradient-Based:
            - adam, gradient_descent, adagrad, l_bfgs_b, cg

        Args:
            objective: Cost function to minimize
            initial_point: Starting parameter values
            n_params: Number of parameters
            **kwargs: Additional optimizer parameters

        Returns:
            Tuple of (optimal_params, optimal_value, optimization_info)
        """
        GRADIENT_FREE_OPTIMIZERS = {
            "cobyla",
            "spsa",
            "genetic_algorithm",
            "particle_swarm",
            "slsqp",
            "nelder_mead",
            "powell",
        }

        GRADIENT_BASED_OPTIMIZERS = {"adam", "gradient_descent", "l_bfgs_b"}

        optimizer_name = self.optimizer.lower()

        if self.verbose:
            self._print_optimization_info(n_params, optimizer_name)

        if optimizer_name in GRADIENT_FREE_OPTIMIZERS:
            if optimizer_name == "cobyla":
                base_per_iteration = n_params + 2
                base_evaluations = self.max_iterations * base_per_iteration

                if self.max_iterations <= 5:
                    buffer = 10 + self.max_iterations * 2
                elif self.max_iterations <= 20:
                    buffer = 20 + self.max_iterations
                else:
                    buffer = 50

                maxfun = base_evaluations + buffer

                return cobyla_optimize(
                    objective=objective,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    maxfun=maxfun,
                    rhobeg=1.0,
                    rhoend=1e-4,
                    **kwargs,
                )

            elif optimizer_name == "nelder_mead":
                return nelder_mead_optimize(
                    objective=objective,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "powell":
                return powell_optimize(
                    objective=objective,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "slsqp":
                return slsqp_optimize(
                    objective=objective,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "spsa":
                return spsa_optimize(
                    objective=objective,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "genetic_algorithm":
                bounds = kwargs.get("bounds", [(0, 2 * np.pi)] * n_params)

                return genetic_algorithm_optimize(
                    objective=objective,
                    bounds=bounds,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "particle_swarm":
                bounds = kwargs.get("bounds", [(0, 2 * np.pi)] * n_params)

                return particle_swarm_optimize(
                    objective=objective,
                    bounds=bounds,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

        # ========== GRADIENT-BASED OPTIMIZERS ==========
        elif optimizer_name in GRADIENT_BASED_OPTIMIZERS:

            def gradient_wrapper(params: np.ndarray) -> np.ndarray:
                return self._compute_gradient(params, objective)

            if optimizer_name == "adam":
                return adam_optimize(
                    objective=objective,
                    gradient=gradient_wrapper,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "gradient_descent":
                return gradient_descent_optimize(
                    objective=objective,
                    gradient=gradient_wrapper,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "l_bfgs_b":
                return l_bfgs_b_optimize(
                    objective=objective,
                    gradient=gradient_wrapper,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            # If we reach here, the optimizer is in GRADIENT_BASED_OPTIMIZERS but not handled
            raise ValueError(
                f"Gradient-based optimizer '{optimizer_name}' is not fully implemented yet"
            )

        # Unknown optimizer
        raise ValueError(
            f"Unknown optimizer: '{self.optimizer}'\n\n"
            f"Supported gradient-free optimizers:\n"
            f"  {', '.join(sorted(GRADIENT_FREE_OPTIMIZERS))}\n\n"
            f"Supported gradient-based optimizers:\n"
            f"  {', '.join(sorted(GRADIENT_BASED_OPTIMIZERS))}\n\n"
            f"Example: VQESolver(n_qubits=2, optimizer='cobyla')"
        )

    def _count_parameters(self) -> int:
        """Count the number of parameters in the ansatz circuit."""
        circuit = self.build_circuit()

        if not hasattr(circuit, "icr") or not circuit.icr:
            return 0

        if not hasattr(circuit.icr, "evolve"):
            return 0

        return sum(
            1
            for op in circuit.icr.evolve
            if isinstance(op, CircuitOperation)
            and op.operation_type == OperationType.N_QUBIT_PARAMETRIC
        )

    def _get_n_qubits_from_hamiltonian(self, hamiltonian: Any) -> int:
        """Determine number of qubits needed from Hamiltonian structure."""
        n_qubits = 0
        try:
            for term, _ in hamiltonian.get_hamiltonian_terms():
                for qubit_idx, _ in term:
                    n_qubits = max(n_qubits, qubit_idx + 1)
        except (AttributeError, TypeError) as e:
            raise ValueError(f"Invalid Hamiltonian structure: {str(e)}")

        return n_qubits

    @property
    def executor(self):
        """Get the job manager instance."""
        if self._executor is None:
            self._executor = JobManager()
        return self._executor

    def with_ansatz(self, ansatz):
        """Set ansatz and return self for method chaining."""
        self.ansatz = ansatz
        return self

    def with_optimizer(self, optimizer):
        """Set optimizer and return self for method chaining."""
        self.optimizer = optimizer.lower()
        return self

    def with_initial_point(self, point):
        """Set initial parameters and return self for method chaining."""
        self.initial_point = point
        return self

    def with_max_iterations(self, max_iter):
        """Set max iterations and return self for method chaining."""
        self.max_iterations = max_iter
        return self

    def with_backend_config(
        self, method: str = "statevector", device_name: str = "sim"
    ):
        """Set backend configuration and return self for method chaining."""
        self.method = method
        self.device_name = device_name
        return self

    def with_method(self, method: str):
        """Set simulation method and return self for method chaining."""
        self.method = method
        return self

    def with_device_name(self, device_name: str):
        """Set device name and return self for method chaining."""
        self.device_name = device_name
        return self

    def with_experiment_name(self, experiment_name: str):
        """Set experiment name and return self for method chaining."""
        self.experiment_name = experiment_name
        return self

    def with_reverse_bits(self, reverse_bits: bool):
        """Set reverse bits flag and return self for method chaining."""
        self.reverse_bits = reverse_bits
        return self

    def with_shots(self, shots):
        """Set number of shots and return self for method chaining."""
        self.shots = shots
        return self

    def with_verbose(self, verbose: bool):
        """Set verbose flag and return self for method chaining."""
        self.verbose = verbose
        return self

    def run(  # type: ignore[override]
        self,
        shots: int = 1024,
        experiment_name: str = "Default Experiment",
        need_statevector: bool = True,
        need_density_matrix: bool = False,
        device_name: str = "QpiAI-QSV-Local",
        reverse_bits: bool = False,
        **kwargs,
    ) -> VQEResult:
        """
        Execute VQE optimization by repeatedly evaluating a variational circuit
        on a backend determined by the high-level device_name.

        Args:
            hamiltonian: Problem Hamiltonian (required for VQE)
            shots: Number of measurement shots
            experiment_name: Name of the experiment (default: "Default Experiment")
            device_name: Target device name ('QpiAI-QSV-Local', 'QpiAI-QSV-Simulator', 'QpiAI-QDM-Simulator', 'QpiAI-QTN-Simulator', 'QpiAI-Indus-1')
            reverse_bits: Reverse bit order in results
            **kwargs: Additional parameters

        Returns:
            VQEResult: Optimization results including optimal energy and parameters
        """
        # Map high-level device_name to low-level (device, method)
        from qpiai_quantum.circuit.circuit import _map_device_name_to_method

        method = _map_device_name_to_method(device_name)

        if "hamiltonian" in kwargs:
            self.hamiltonian = kwargs["hamiltonian"]

        if self.hamiltonian is None:
            raise ValueError(
                "VQE requires a hamiltonian parameter (either in __init__ or run)"
            )

        self.shots = shots

        # Determine number of qubits from Hamiltonian
        n_qubits = self._get_n_qubits_from_hamiltonian(self.hamiltonian)
        self.num_qubits = n_qubits

        # Get and validate ansatz
        ansatz = self._get_ansatz()

        if not isinstance(ansatz, Circuit):
            raise ValueError("Ansatz must return a Circuit object")

        if not hasattr(ansatz, "icr") or not ansatz.icr:
            raise ValueError("Ansatz circuit must have ICR representation")

        if not hasattr(ansatz.icr, "evolve"):
            raise ValueError("Ansatz circuit must have valid ICR evolution operations")

        # Count parameters
        n_params = sum(
            1
            for op in ansatz.icr.evolve
            if isinstance(op, CircuitOperation)
            and op.operation_type == OperationType.N_QUBIT_PARAMETRIC
        )

        # Initialize parameters
        if self.initial_point is not None:
            params = np.array(self.initial_point, dtype=float)
            if len(params) != n_params:
                raise ValueError(
                    f"initial_point has {len(params)} parameters "
                    f"but ansatz requires {n_params} parameters"
                )
        else:
            params = np.random.uniform(0, 2 * np.pi, n_params)
            self.initial_point = params

        def objective(theta: np.ndarray) -> float:
            """Compute energy for given parameters."""
            circuit = self._build_circuit(ansatz, theta)
            result = self._execute_circuit(
                circuit,
                method=method,
                device_name=device_name,
                shots=shots,
                experiment_name=experiment_name,
                reverse_bits=reverse_bits,
            )
            return self._compute_expectation(result)

        # Run optimization (info is printed in _optimize_parameters)
        opt_result = self._optimize_parameters(
            objective=objective,
            initial_point=self.initial_point,
            n_params=n_params,
            **kwargs,
        )

        best_params, best_energy, opt_info = opt_result
        actual_iterations = opt_info.get("nit", 0)
        actual_nfev = opt_info.get("nfev", 0)

        # Print completion summary
        if self.verbose:
            print("\n" + "=" * 70)
            print("VQE OPTIMIZATION COMPLETED")
            print("=" * 70)
            print(f"Actual Iterations:   {actual_iterations}")
            print(f"Max Iterations:      {self.max_iterations}")
            print(f"Circuit Evaluations: {actual_nfev}")
            print(f"Final Energy:        {best_energy:.6f}")
            print(f"Success:             {opt_info.get('success', False)}")

            # Check if iterations match expectations
            if actual_iterations < self.max_iterations:
                print(
                    f"\n Note: Optimizer converged early (stopped at iteration {actual_iterations})"
                )
            elif actual_iterations == self.max_iterations:
                print("\n Note: Optimizer reached max_iterations limit")

            print("=" * 70 + "\n")

        # Get final measurement results
        final_circuit = self._build_circuit(ansatz, best_params)
        final_result = self._execute_circuit(
            final_circuit,
            method=method,
            device_name=device_name,
            shots=shots,
            experiment_name=experiment_name,
            reverse_bits=reverse_bits,
        )

        final_counts = (
            {k: int(v) for k, v in final_result.counts.items()}
            if final_result.counts
            else {}
        )

        # Handle empty counts case
        if final_counts:
            best_bitstring = max(final_counts.items(), key=lambda x: x[1])[0]
        else:
            # Fallback: generate default bitstring of all zeros
            best_bitstring = "0" * self.num_qubits
            warnings.warn(
                f"No measurement counts returned. Using default bitstring: {best_bitstring}",
                UserWarning,
                stacklevel=2,
            )

        return VQEResult(
            optimal_parameters=best_params.tolist(),
            optimal_energy=float(best_energy),
            energy_history=opt_info.get("history", []),
            param_history=[p.tolist() for p in opt_info.get("param_history", [])],
            bitstring=best_bitstring,
            counts=final_counts,
            metadata={
                "success": opt_info.get("success", False),
                "actual_iterations": actual_iterations,
                "max_iterations_limit": self.max_iterations,
                "circuit_evaluations": actual_nfev,
                "message": opt_info.get("message", ""),
                "method": method,
                "device_name": device_name,
                "experiment_name": experiment_name,
                "reverse_bits": reverse_bits,
                "shots": shots,
                "n_parameters": n_params,
                "optimizer": self.optimizer,
                "converged_early": actual_iterations < self.max_iterations,
            },
        )

    def to_dict(self, result: "VQEResult") -> Dict[str, Any]:
        """
        Convert VQEResult to dictionary format for backward compatibility.

        Args:
            result: VQEResult object to convert

        Returns:
            Dictionary containing VQE results
        """
        return {
            "optimal_parameters": result.optimal_parameters,
            "optimal_energy": result.optimal_energy,
            "energy_history": result.energy_history,
            "param_history": result.param_history,
            "bitstring": result.bitstring,
            "counts": {k: int(v) for k, v in result.counts.items()},
            "metadata": result.metadata,
        }
