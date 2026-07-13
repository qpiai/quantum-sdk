from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from qpiai_quantum.results.base_result import BaseQuantumResult
import numpy as np
import warnings
from dataclasses import dataclass
from ....circuit import Circuit
from ....jobmanager import JobManager
from ....jobmanager.job_result import JobResult
from ....icr.circuitoperation import OperationType, CircuitOperation
from ...base import QuantumAlgorithm
from ..ansatz.standard import standard_qaoa_ansatz, count_parameters
from ..ansatz.hardware_efficient import hardware_efficient_ansatz
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
class QAOAResult:
    """Result from QAOA execution. Supports both attribute and dictionary-like access."""

    optimal_parameters: List[float]
    optimal_energy: float
    energy_history: List[float]
    param_history: List[List[float]]
    solution: Any
    bitstring: str
    counts: Dict[str, int]
    is_valid: bool
    validation_message: str
    quality_metrics: Dict[str, Any]
    metadata: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-like access for backward compatibility."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"'{key}' not found in QAOAResult")

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for backward compatibility."""
        return hasattr(self, key)


class QAOASolver(QuantumAlgorithm):
    """
    Quantum Approximate Optimization Algorithm (QAOA) implementation using QpiAI Quantum SDK.

    QAOA is a hybrid quantum-classical algorithm for solving combinatorial optimization problems.
    It uses a parameterized quantum circuit with alternating cost and mixer layers.

    This implementation follows the same pattern as VQESolver for consistency across the SDK.

    Args:
        layers: Number of QAOA layers (p parameter, default: 1)
        optimizer: Optimization method ('cobyla', 'spsa', 'nelder_mead', etc., default: 'COBYLA')
        max_iterations: Maximum number of optimization iterations (default: 100)
        initial_params: Initial parameters for optimization (optional)
        ansatz: Ansatz type ('standard', 'hardware_efficient', or callable, default: 'standard')
        mixer: Mixer type ('x' or 'xy', default: 'x')
        verbose: Enable printing of optimization progress and completion summaries (default: True)
        name: Name of this QAOA instance
    """

    def __init__(
        self,
        problem: Any = None,
        layers: int = 1,
        optimizer: str = "COBYLA",
        max_iterations: int = 100,
        initial_params: Optional[np.ndarray] = None,
        ansatz: Union[str, Callable] = "standard",
        mixer: str = "x",
        verbose: bool = True,
        name: str = "QAOA",
    ):
        """Initialize QAOA solver with algorithm parameters."""
        n_qubits = problem.n_qubits if problem and hasattr(problem, "n_qubits") else 1
        super().__init__(num_qubits=n_qubits, name=name)
        self.layers = layers
        self.optimizer = optimizer.upper()
        self.max_iterations = max_iterations
        self.initial_params = initial_params
        self.ansatz = ansatz
        self.mixer = mixer
        self.problem = problem
        self.verbose = verbose
        self._executor: Optional[JobManager] = None
        self.shots = 1024
        self.description = (
            "Quantum Approximate Optimization Algorithm for combinatorial problems (verbose progress printing is enabled by default)"
        )

    def build_circuit(self, parameters: Optional[np.ndarray] = None) -> Circuit:
        """
        Public method: Build the QAOA circuit with given parameters.

        Args:
            parameters: Circuit parameters (optional)

        Returns:
            Parameterized quantum circuit
        """
        if self.problem is None:
            raise ValueError("Problem not set. Call run() with a problem first.")

        ansatz = self._get_ansatz()
        self.num_qubits = self.problem.n_qubits
        self.circuit = self._build_circuit(ansatz, parameters)
        return self.circuit

    def _get_ansatz(self) -> Circuit:
        """
        Private method: Select and validate ansatz based on configuration.

        Returns:
            Validated ansatz circuit
        """
        if self.problem is None:
            raise ValueError("Problem must be set before getting ansatz")

        if not hasattr(self.problem, "n_qubits"):
            raise ValueError("Problem must have n_qubits attribute")

        n_qubits = self.problem.n_qubits

        if self.ansatz == "standard" or self.ansatz is None:
            ansatz = standard_qaoa_ansatz(
                n_qubits=n_qubits,
                hamiltonian=self.problem,
                layers=self.layers,
                mixer=self.mixer,
            )
        elif self.ansatz == "hardware_efficient":
            # Use hardware-efficient ansatz as alternative
            ansatz = hardware_efficient_ansatz(
                n_qubits=n_qubits, depth=self.layers, entanglement="linear"
            )
        elif callable(self.ansatz):
            ansatz = self.ansatz(n_qubits, self.layers)
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

        This is the cost function value for the current parameters.
        """
        if not result or not result.counts:
            raise ValueError("Invalid execution result")

        counts = {k: int(v) for k, v in result.counts.items()}

        if self.problem is None:
            raise ValueError("Problem not set")

        if not hasattr(self.problem, "get_hamiltonian_terms"):
            raise ValueError("Problem must have get_hamiltonian_terms method")

        terms = self.problem.get_hamiltonian_terms()
        if terms:
            max_qubit_idx = max((qi for ops, _ in terms for qi, _ in ops), default=0)
            expected_n_qubits = max_qubit_idx + 1
        else:
            expected_n_qubits = 1

        # Pad bitstrings to expected length
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

        Args:
            circuit: Circuit to execute
            method: Simulation method
            device_name: Device name
            shots: Number of measurement shots
            **kwargs: Additional parameters

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

        # Call parent class run() method
        return super().run(**run_kwargs)  # type: ignore

    def _print_optimization_info(self, n_params: int, optimizer_name: str):
        """Print detailed information about the optimization before it starts."""
        print("\n" + "=" * 70)
        print("QAOA OPTIMIZATION STARTING")
        print("=" * 70)
        print(f"Optimizer:        {optimizer_name.upper()}")
        print(f"Parameters:       {n_params}")
        print(f"QAOA Layers:      {self.layers}")
        print(f"Max Iterations:   {self.max_iterations}")
        print(f"Shots per eval:   {self.shots}")

        # Calculate expected circuit executions
        GRADIENT_FREE_OPTIMIZERS = {
            "cobyla",
            "nelder_mead",
            "spsa",
            "slsqp",
            "differential_evolution",
            "genetic_algorithm",
            "particle_swarm",
        }

        if optimizer_name.lower() in GRADIENT_FREE_OPTIMIZERS:
            if optimizer_name.lower() == "cobyla":
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

            elif optimizer_name.lower() == "spsa":
                expected_evals = 1 + self.max_iterations * 3
                print(
                    f"Expected Evals:   ~{expected_evals} (may stop early if converged)"
                )
                print(
                    f"  |-- 1 initial + {self.max_iterations} iterations x 3 evals/iter"
                )

            else:
                expected_evals = self.max_iterations
                print(f"Expected Evals:   ~{expected_evals}")
        else:
            # Gradient-based optimizers
            expected_evals = self.max_iterations * (2 * n_params + 1) + 1
            print(f"Expected Evals:   ~{expected_evals} (GRADIENT-BASED)")
            print(
                f"  |-- {self.max_iterations} iter x (2x{n_params} for gradient + 1) + 1"
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

        Args:
            objective: Cost function to minimize
            initial_point: Starting parameter values
            n_params: Number of parameters
            **kwargs: Additional optimizer parameters

        Returns:
            Tuple of (optimal_params, optimal_value, optimization_info)
        """
        GRADIENT_FREE_OPTIMIZERS = {
            "COBYLA",
            "SPSA",
            "GENETIC_ALGORITHM",
            "PARTICLE_SWARM",
            "SLSQP",
            "NELDER_MEAD",
            "POWELL",
        }

        GRADIENT_BASED_OPTIMIZERS = {"ADAM", "GRADIENT_DESCENT", "L_BFGS_B"}

        optimizer_name = self.optimizer.upper()

        if self.verbose:
            self._print_optimization_info(n_params, optimizer_name)

        if optimizer_name in GRADIENT_FREE_OPTIMIZERS:
            if optimizer_name == "COBYLA":
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

            elif optimizer_name == "NELDER_MEAD":
                return nelder_mead_optimize(
                    objective=objective,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "SLSQP":
                return slsqp_optimize(
                    objective=objective,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "SPSA":
                return spsa_optimize(
                    objective=objective,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "POWELL":
                return powell_optimize(
                    objective=objective,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "GENETIC_ALGORITHM":
                bounds = kwargs.get("bounds", [(0, 2 * np.pi)] * n_params)
                return genetic_algorithm_optimize(
                    objective=objective,
                    bounds=bounds,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "PARTICLE_SWARM":
                bounds = kwargs.get("bounds", [(0, 2 * np.pi)] * n_params)
                return particle_swarm_optimize(
                    objective=objective,
                    bounds=bounds,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

        elif optimizer_name in GRADIENT_BASED_OPTIMIZERS:

            def gradient_wrapper(params: np.ndarray) -> np.ndarray:
                return self._compute_gradient(params, objective)

            if optimizer_name == "ADAM":
                return adam_optimize(
                    objective=objective,
                    gradient=gradient_wrapper,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "GRADIENT_DESCENT":
                return gradient_descent_optimize(
                    objective=objective,
                    gradient=gradient_wrapper,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            elif optimizer_name == "L_BFGS_B":
                return l_bfgs_b_optimize(
                    objective=objective,
                    gradient=gradient_wrapper,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    **kwargs,
                )

            raise ValueError(
                f"Gradient-based optimizer '{optimizer_name}' is not fully implemented yet"
            )

        raise ValueError(
            f"Unknown optimizer: '{self.optimizer}'\\n\\n"
            f"Supported gradient-free optimizers:\\n"
            f"  {', '.join(sorted(o.lower() for o in GRADIENT_FREE_OPTIMIZERS))}\\n\\n"
            f"Supported gradient-based optimizers:\\n"
            f"  {', '.join(sorted(o.lower() for o in GRADIENT_BASED_OPTIMIZERS))}\\n\\n"
            f"Example: QAOASolver(layers=2, optimizer='cobyla')"
        )

    def _count_parameters(self) -> int:
        """Count the number of parameters in the ansatz circuit."""
        circuit = self.build_circuit()
        return count_parameters(circuit)

    @property
    def executor(self):
        """Get the job manager instance."""
        if self._executor is None:
            self._executor = JobManager()
        return self._executor

    # Fluent API methods for method chaining
    def with_layers(self, layers: int):
        """Set QAOA layers and return self for method chaining."""
        self.layers = layers
        return self

    def with_optimizer(self, optimizer: str):
        """Set optimizer and return self for method chaining."""
        self.optimizer = optimizer.upper()
        return self

    def with_initial_params(self, params: np.ndarray):
        """Set initial parameters and return self for method chaining."""
        self.initial_params = params
        return self

    def with_max_iterations(self, max_iter: int):
        """Set max iterations and return self for method chaining."""
        self.max_iterations = max_iter
        return self

    def with_ansatz(self, ansatz: Union[str, Callable]):
        """Set ansatz and return self for method chaining."""
        self.ansatz = ansatz
        return self

    def with_mixer(self, mixer: str):
        """Set mixer type and return self for method chaining."""
        self.mixer = mixer
        return self

    def with_shots(self, shots: int):
        """Set number of shots and return self for method chaining."""
        self.shots = shots
        return self

    def with_verbose(self, verbose: bool):
        """Set verbose flag and return self for method chaining."""
        self.verbose = verbose
        return self

    def with_backend_config(
        self, method: str = "statevector", device_name: str = "sim"
    ):
        """Set backend configuration and return self for method chaining."""
        self.method = method
        self.device_name = device_name
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
    ) -> QAOAResult:
        """
        Execute QAOA optimization by repeatedly evaluating a parametrized circuit
        on a backend determined by the high-level device_name.

        Args:
            problem: Optimization problem (required, must have n_qubits and get_hamiltonian_terms)
            shots: Number of measurement shots
            experiment_name: Name of the experiment (default: "Default Experiment")
            device_name: Target device name ('QpiAI-QSV-Local', 'QpiAI-QSV-Simulator', 'QpiAI-QDM-Simulator', 'QpiAI-QTN-Simulator', 'QpiAI-Indus-1')
            reverse_bits: Reverse bit order in results
            **kwargs: Additional parameters

        Returns:
            QAOAResult: Optimization results including solution and quality metrics
        """
        # Map high-level device_name to low-level (device, method)
        from qpiai_quantum.circuit.circuit import _map_device_name_to_method

        method = _map_device_name_to_method(device_name)

        if "problem" in kwargs:
            self.problem = kwargs["problem"]

        if self.problem is None:
            raise ValueError(
                "QAOA requires a problem parameter (either in __init__ or run)"
            )

        self.shots = shots

        # Validate problem
        if not hasattr(self.problem, "n_qubits"):
            raise ValueError("Problem must have n_qubits attribute")

        if not hasattr(self.problem, "get_hamiltonian_terms"):
            raise ValueError("Problem must have get_hamiltonian_terms() method")

        self.num_qubits = self.problem.n_qubits

        # Get and validate ansatz
        ansatz = self._get_ansatz()

        # Count parameters (2 per layer for standard QAOA: gamma and beta)
        n_params = count_parameters(ansatz)

        # Initialize parameters
        if self.initial_params is not None:
            params = np.array(self.initial_params, dtype=float)
            if len(params) != n_params:
                raise ValueError(
                    f"initial_params has {len(params)} parameters "
                    f"but ansatz requires {n_params} parameters"
                )
        else:
            # Initialize with small random values for QAOA
            params = np.random.uniform(0, np.pi / 2, n_params)
            self.initial_params = params

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

        # Run optimization
        opt_result = self._optimize_parameters(
            objective=objective,
            initial_point=self.initial_params,
            n_params=n_params,
            **kwargs,
        )

        best_params, best_energy, opt_info = opt_result
        actual_iterations = opt_info.get("nit", 0)
        actual_nfev = opt_info.get("nfev", 0)

        # Print completion summary
        if self.verbose:
            print("\n" + "=" * 70)
            print("QAOA OPTIMIZATION COMPLETED")
            print("=" * 70)
            print(f"Actual Iterations:   {actual_iterations}")
            print(f"Max Iterations:      {self.max_iterations}")
            print(f"Circuit Evaluations: {actual_nfev}")
            print(f"Final Energy:        {best_energy:.6f}")
            print(f"Success:             {opt_info.get('success', False)}")

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

        # Get best bitstring
        if final_counts:
            best_bitstring = max(final_counts.items(), key=lambda x: x[1])[0]
        else:
            best_bitstring = "0" * self.num_qubits
            warnings.warn(
                f"No measurement counts returned. Using default bitstring: {best_bitstring}",
                UserWarning,
                stacklevel=2,
            )

        # Decode solution using problem's decoder
        if hasattr(self.problem, "decode_solution"):
            solution = self.problem.decode_solution(best_bitstring)
        else:
            solution = best_bitstring

        # Validate solution
        if hasattr(self.problem, "validate_solution"):
            is_valid, validation_message = self.problem.validate_solution(solution)
        else:
            is_valid = True
            validation_message = "Validation not available"

        # Compute quality metrics
        if hasattr(self.problem, "compute_solution_quality"):
            quality_metrics = self.problem.compute_solution_quality(solution)
        else:
            quality_metrics = {}

        return QAOAResult(
            optimal_parameters=best_params.tolist(),
            optimal_energy=float(best_energy),
            energy_history=opt_info.get("history", []),
            param_history=[p.tolist() for p in opt_info.get("param_history", [])],
            solution=solution,
            bitstring=best_bitstring,
            counts=final_counts,
            is_valid=is_valid,
            validation_message=validation_message,
            quality_metrics=quality_metrics,
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
                "layers": self.layers,
                "ansatz": str(self.ansatz),
                "mixer": self.mixer,
                "converged_early": actual_iterations < self.max_iterations,
            },
        )

    def to_dict(self, result: "QAOAResult") -> Dict[str, Any]:
        """
        Convert QAOAResult to dictionary format for backward compatibility.

        Args:
            result: QAOAResult object to convert

        Returns:
            Dictionary containing QAOA results
        """
        return {
            "optimal_parameters": result.optimal_parameters,
            "optimal_energy": result.optimal_energy,
            "energy_history": result.energy_history,
            "param_history": result.param_history,
            "solution": result.solution,
            "bitstring": result.bitstring,
            "counts": result.counts,
            "is_valid": result.is_valid,
            "validation_message": result.validation_message,
            "quality_metrics": result.quality_metrics,
            "metadata": result.metadata,
        }

    def get_name(self) -> str:
        """Get the name of this QAOA instance."""
        return f"QAOA (layers={self.layers}, optimizer={self.optimizer})"
