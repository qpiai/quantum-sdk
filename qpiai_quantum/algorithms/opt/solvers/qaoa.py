from typing import Any
from collections.abc import Callable
from qpiai_quantum.results.base_result import BaseQuantumResult
import numpy as np
from dataclasses import asdict, dataclass, field
from ....circuit import Circuit
from ....jobmanager import JobManager
from ....jobmanager.job_result import JobResult
from ...base import QuantumAlgorithm
from ..ansatz.standard import standard_qaoa_ansatz
from ..parameters import bind_parameters, count_parameters
from ..ansatz.hardware_efficient import hardware_efficient_ansatz
from ...utils.optimizers import (
    adam_optimize,
    gradient_descent_optimize,
    nelder_mead_optimize,
    l_bfgs_b_optimize,
    spsa_optimize,
    slsqp_optimize,
    differential_evolution_optimize,
    genetic_algorithm_optimize,
    particle_swarm_optimize,
    cobyla_optimize,
    powell_optimize,
)
from ...utils.optimizers.progress import OptimizationProgress
from ..qaoa_components import (
    Aggregation,
    CVaR,
    InitialState,
    Mixer,
    WarmStart,
    XMixer,
    XYMixer,
)
from ..differentiation import (
    CircuitObjective,
    Gradient,
    finite_difference,
    parameter_shift,
    spsa_gradient,
)


class _EvaluationBudgetExceeded(RuntimeError):
    """Stop an optimizer after QAOA consumes its circuit-evaluation budget."""


@dataclass(frozen=True)
class QAOAEvaluation:
    """One budgeted execution, with objective score, expectation, and probe kind.

    ``energy`` is the optimization objective (expectation by default).
    A gate-shift ``gradient_probe`` does not represent a candidate at the
    recorded logical parameters and is never selected as an optimum.
    """

    start_index: int
    evaluation_index: int
    parameters: list[float]
    energy: float
    shots: int
    expectation: float | None = None
    kind: str = "candidate"


@dataclass(frozen=True)
class QAOAStartResult:
    """Summary of one optimizer start.

    ``iterations`` counts completed SDK steps or a native backend count.
    It is ``None`` when an interruption prevents recovery of that count.
    ``callback_reports`` counts notifications separately and is not an
    iteration estimate. ``termination_reason`` distinguishes convergence,
    iteration limits, evaluation budgets, and other optimizer stops.
    ``optimal_energy`` is the expectation at the selected candidate, while
    ``optimal_objective`` is the score used to select it.
    """

    start_index: int
    initial_parameters: list[float]
    optimal_parameters: list[float]
    optimal_energy: float
    evaluations: int
    iterations: int | None
    success: bool
    message: str
    termination_reason: str = "unknown"
    callback_reports: int = 0
    optimal_objective: float | None = None


@dataclass
class QAOAResult:
    """Result from QAOA execution.

    ``optimal_energy`` is retained for compatibility and represents the
    optimized estimated expectation value. It is not a proof of global
    optimality. ``best_sampled_energy`` and ``bitstring`` describe the best
    feasible measured sample returned by the final circuit execution.
    ``optimal_objective`` is the optimized score, which may be CVaR or a custom
    aggregation. Histories contain objective scores, not necessarily means.
    """

    optimal_parameters: list[float]
    optimal_energy: float
    energy_history: list[float]
    param_history: list[list[float]]
    solution: Any
    bitstring: str
    counts: dict[str, int]
    is_valid: bool
    validation_message: str
    quality_metrics: dict[str, Any]
    metadata: dict[str, Any]
    most_probable_bitstring: str | None = None
    best_sampled_energy: float | None = None
    sampled_energies: dict[str, float] = field(default_factory=dict)
    evaluations: list[QAOAEvaluation] = field(default_factory=list)
    start_results: list[QAOAStartResult] = field(default_factory=list)
    optimal_objective: float | None = None

    @property
    def estimated_expectation(self) -> float:
        """Return the optimized QAOA expectation-value estimate."""
        return self.optimal_energy

    @property
    def best_sampled_bitstring(self) -> str:
        """Return the lowest-energy feasible bitstring observed at the end."""
        return self.bitstring

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
        ansatz: ``standard`` (default), ``hardware_efficient``, or a builder
            ``(n_qubits, layers) -> Circuit``. Custom builders own the whole
            circuit. Numeric angles are fixed. Mark adjustable arguments with
            ``qpiai_quantum.algorithms.opt.parameterize``. Shared indices, scaling,
            offsets, and separate arguments in multi-angle gates are supported.
        mixer: ``x``, ``xy``, or a ``(n_qubits, beta) -> Circuit`` factory.
        initial_state: Numeric unmeasured Circuit or width-to-circuit factory.
            Used only with the standard ansatz. Default: uniform superposition.
        warm_start: WarmStart component supplying a state and matching mixer.
            Cannot be combined with explicit state or nondefault mixer inputs.
        aggregation: Callable receiving immutable probability and energy maps.
            Defaults to expectation. Use CVaR(alpha) for lower-tail optimization.
        gradient_method: ``finite_difference`` (default), ``parameter_shift``,
            ``spsa``, ``auto``, or a ``(parameters, objective) -> gradient`` callable.
            Auto uses gate-wise shifts for supported expectation circuits and
            central differences otherwise. Gate shifts require built-in mixers.
        seed: Seed for parameter initialization and stochastic optimizers. Backend
            measurement randomness is controlled by the selected backend.
        num_starts: Number of optimizer starts. A supplied ``initial_params`` is
            used for the first start and remaining starts are generated.
        max_evaluations: Maximum objective-circuit evaluations across all starts.
            The final sampling circuit is not included in this limit.
        max_shots: Maximum requested shots for objective evaluations plus the final
            sampling circuit. One final sampling execution is always reserved.
        verbose: Enable printing of optimization progress and completion summaries (default: True)
        name: Name of this QAOA instance
        gradient_step: Positive finite central-difference or SPSA probe step for gradient-based
            optimizers (default: 0.01 radians). Smaller steps reduce truncation
            error but amplify shot noise. Increase the step or the shot count
            when using sampled objectives; use a smaller step for exact energies.
    """

    def __init__(
        self,
        problem: Any = None,
        layers: int = 1,
        optimizer: str = "COBYLA",
        max_iterations: int = 100,
        initial_params: np.ndarray | None = None,
        ansatz: str | Callable | None = "standard",
        mixer: str | Mixer = "x",
        seed: int | None = None,
        num_starts: int = 1,
        max_evaluations: int | None = None,
        max_shots: int | None = None,
        verbose: bool = True,
        name: str = "QAOA",
        gradient_step: float = 0.01,
        initial_state: InitialState | None = None,
        warm_start: WarmStart | None = None,
        aggregation: Aggregation | None = None,
        gradient_method: str | Gradient = "finite_difference",
    ) -> None:
        """Initialize QAOA solver with algorithm parameters."""
        if (
            isinstance(gradient_step, bool)
            or not isinstance(gradient_step, int | float)
            or (not np.isfinite(gradient_step))
            or (gradient_step <= 0)
        ):
            raise ValueError("gradient_step must be a positive finite number")
        if not isinstance(layers, int) or isinstance(layers, bool) or layers < 1:
            raise ValueError("layers must be a positive integer")
        if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if (
            not isinstance(num_starts, int)
            or isinstance(num_starts, bool)
            or num_starts < 1
        ):
            raise ValueError("num_starts must be a positive integer")
        for value, label in (
            (max_evaluations, "max_evaluations"),
            (max_shots, "max_shots"),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
            ):
                raise ValueError(f"{label} must be a positive integer when provided")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise ValueError("seed must be an integer or None")
        if not callable(ansatz) and ansatz not in {
            None,
            "standard",
            "hardware_efficient",
        }:
            raise ValueError(f"Unknown ansatz type: {ansatz}")
        if isinstance(mixer, str) and mixer not in {"x", "xy"}:
            raise ValueError("mixer must be x, xy, or a mixer factory")
        if not isinstance(mixer, str) and (not callable(mixer)):
            raise TypeError("mixer must be x, xy, or a mixer factory")
        if (
            initial_state is not None
            and (not isinstance(initial_state, Circuit))
            and (not callable(initial_state))
        ):
            raise TypeError("initial_state must be a Circuit or factory")
        if warm_start is not None:
            if not isinstance(warm_start, WarmStart):
                raise TypeError("warm_start must be a WarmStart component")
            if initial_state is not None or mixer != "x":
                raise ValueError(
                    "warm_start supplies both initial_state and mixer; configure them separately to override"
                )
            initial_state, mixer = (warm_start.initial_state, warm_start.mixer)
        if ansatz not in (None, "standard") and (
            initial_state is not None or mixer != "x"
        ):
            raise ValueError(
                "initial_state and mixer components require the standard ansatz"
            )
        if aggregation is not None and (not callable(aggregation)):
            raise TypeError("aggregation must be callable or None")
        if not callable(gradient_method) and gradient_method not in {
            "finite_difference",
            "parameter_shift",
            "spsa",
            "auto",
        }:
            raise ValueError("unknown gradient_method")
        if gradient_method == "parameter_shift" and aggregation is not None:
            raise ValueError(
                "parameter_shift requires the ordinary expectation objective"
            )
        n_qubits = problem.n_qubits if problem and hasattr(problem, "n_qubits") else 1
        super().__init__(num_qubits=n_qubits, name=name)
        self.layers = layers
        self.gradient_step = float(gradient_step)
        self.gradient_method = gradient_method
        self.initial_state = initial_state
        self.aggregation = aggregation
        self._gradient_rng = np.random.default_rng(seed)
        self.optimizer = optimizer.upper()
        self.max_iterations = max_iterations
        self.initial_params = initial_params
        self.ansatz = ansatz
        self.mixer = mixer
        self.seed = seed
        self.num_starts = num_starts
        self.max_evaluations = max_evaluations
        self.max_shots = max_shots
        self.problem = problem
        self.verbose = verbose
        self._executor: JobManager | None = None
        self.shots = 1024
        self.description = "Quantum Approximate Optimization Algorithm for combinatorial problems (verbose progress printing is enabled by default)"

    def build_circuit(self, parameters: np.ndarray | None = None) -> Circuit:
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
                initial_state=self.initial_state,
            )
        elif self.ansatz == "hardware_efficient":
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
        self, ansatz: Circuit, parameters: np.ndarray | None = None
    ) -> Circuit:
        """Bind explicit custom angles, or rebuild standard QAOA components.

        Standard custom mixer factories are numeric functions of beta and must
        be called at the current beta. Complete custom ansatz templates instead
        use the same explicit parameter binder as VQE. Unmarked angles stay fixed.
        """
        if self.ansatz == "standard" or self.ansatz is None:
            circuit = standard_qaoa_ansatz(
                n_qubits=self.num_qubits,
                hamiltonian=self.problem,
                layers=self.layers,
                mixer=self.mixer,
                parameters=parameters,
                initial_state=self.initial_state,
            )
        else:
            if ansatz.num_qubits != self.num_qubits:
                raise ValueError(
                    "Custom ansatz width must match the problem's number of qubits"
                )
            circuit = bind_parameters(ansatz, parameters)
        circuit.measure_all()
        return circuit

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

    def _bitstring_energy(self, bitstring: str) -> float:
        """Evaluate a measured bitstring against the diagonal cost Hamiltonian."""
        if self.problem is None:
            raise ValueError("Problem not set")
        normalized = str(bitstring).replace(" ", "")
        if not normalized or any(bit not in {"0", "1"} for bit in normalized):
            raise ValueError(f"Invalid measured bitstring: {bitstring!r}")
        if len(normalized) > self.num_qubits:
            raise ValueError(
                f"Measured bitstring {bitstring!r} exceeds {self.num_qubits} qubits"
            )
        normalized = normalized.zfill(self.num_qubits)
        energy = 0.0
        for ops, coeff in self.problem.get_hamiltonian_terms():
            eigenvalue = 1.0
            for qubit_idx, op_name in ops:
                if op_name == "I":
                    continue
                if op_name != "Z":
                    raise ValueError(
                        "Standard QAOA sample energies require an I/Z Hamiltonian"
                    )
                bit = int(normalized[-(qubit_idx + 1)])
                eigenvalue *= 1.0 if bit == 0 else -1.0
            energy += float(coeff) * eigenvalue
        return float(energy)

    def _select_best_sample(
        self, counts: dict[str, int]
    ) -> tuple[str, str, dict[str, float]]:
        """Select the lowest-energy feasible sample and report the mode separately."""
        if not counts:
            raise ValueError("QAOA execution returned no measurement counts")
        normalized_counts: dict[str, int] = {}
        for raw_bitstring, raw_count in counts.items():
            bitstring = str(raw_bitstring).replace(" ", "").zfill(self.num_qubits)
            if len(bitstring) != self.num_qubits or any(
                bit not in {"0", "1"} for bit in bitstring
            ):
                raise ValueError(f"Invalid measured bitstring: {raw_bitstring!r}")
            count = int(raw_count)
            if count < 0:
                raise ValueError("Measurement counts cannot be negative")
            if count == 0:
                continue
            normalized_counts[bitstring] = normalized_counts.get(bitstring, 0) + count
        if sum(normalized_counts.values()) <= 0:
            raise ValueError("QAOA execution returned zero total measurement shots")
        sampled_energies = {
            bitstring: self._bitstring_energy(bitstring)
            for bitstring in normalized_counts
        }
        most_probable = min(
            normalized_counts,
            key=lambda bitstring: (-normalized_counts[bitstring], bitstring),
        )
        feasible: list[str] = []
        for bitstring in normalized_counts:
            solution = (
                self.problem.decode_solution(bitstring)
                if hasattr(self.problem, "decode_solution")
                else bitstring
            )
            if hasattr(self.problem, "validate_solution"):
                is_valid, _ = self.problem.validate_solution(solution)
            else:
                is_valid = True
            if is_valid:
                feasible.append(bitstring)
        candidates = feasible or list(normalized_counts)
        best = min(
            candidates,
            key=lambda bitstring: (
                sampled_energies[bitstring],
                -normalized_counts[bitstring],
                bitstring,
            ),
        )
        return (best, most_probable, sampled_energies)

    def _resolved_gradient_method(self) -> str | Gradient:
        """Resolve and validate the configured differentiation strategy.

        ``auto`` follows a fixed compatibility rule, not a measured comparison
        of methods. It does not consider shots, measurement noise, remaining
        budget, or relative execution cost when making its choice.
        """
        method = self.gradient_method
        supports_shift = self.ansatz in (None, "standard") and (
            isinstance(self.mixer, str)
            or isinstance(self.mixer, XMixer | XYMixer)
            or isinstance(getattr(self.mixer, "__self__", None), WarmStart)
        )
        if method == "auto":
            return (
                "parameter_shift"
                if supports_shift and self.aggregation is None
                else "finite_difference"
            )
        if method == "parameter_shift" and (
            not supports_shift or self.aggregation is not None
        ):
            raise ValueError(
                "parameter_shift requires a standard expectation ansatz with a built-in mixer"
            )
        return method

    def _compute_gradient(
        self, params: np.ndarray, cost_function: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """Differentiate using budgeted central, SPSA, gate-shift, or user probes.

        Gate shifts use the chain rule for shared weighted parameters and are
        restricted to expectation objectives with annotated built-in mixers.
        Central differences also support nonlinear sampled objectives; their
        error and shot-noise sensitivity depend on ``gradient_step``.
        """
        parameters = np.asarray(params, dtype=float)
        method = self._resolved_gradient_method()
        if callable(method):
            gradient = method(parameters.copy(), cost_function)
        elif method == "parameter_shift":
            if not isinstance(cost_function, CircuitObjective):
                raise ValueError(
                    "parameter_shift requires a circuit-aware budgeted objective"
                )
            gradient = parameter_shift(
                parameters, self.build_circuit(parameters), cost_function.probe
            )
        elif method == "spsa":
            gradient = spsa_gradient(
                parameters, cost_function, self.gradient_step, self._gradient_rng
            )
        else:
            gradient = finite_difference(parameters, cost_function, self.gradient_step)
        result = np.asarray(gradient, dtype=float)
        if result.shape != parameters.shape or not np.all(np.isfinite(result)):
            raise ValueError("gradient must be a finite vector matching the parameters")
        return result

    def _aggregate_objective(
        self, result: BaseQuantumResult, expectation: float
    ) -> float:
        """Aggregate measured probabilities without changing expectation semantics."""
        if self.aggregation is None:
            return expectation
        from types import MappingProxyType

        counts = result.counts or {}
        normalized: dict[str, int] = {}
        for key, value in counts.items():
            bitstring = str(key).replace(" ", "").zfill(self.num_qubits)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError("measurement counts must be nonnegative integers")
            if value:
                normalized[bitstring] = normalized.get(bitstring, 0) + int(value)
        total = sum(normalized.values())
        if not total:
            raise ValueError("No measurements recorded")
        energies = {key: self._bitstring_energy(key) for key in normalized}
        probabilities = {key: value / total for key, value in normalized.items()}
        aggregated_value = float(
            self.aggregation(
                MappingProxyType(probabilities), MappingProxyType(energies)
            )
        )
        if not np.isfinite(aggregated_value):
            raise ValueError("aggregation must return a finite scalar")
        return aggregated_value

    def _get_job_manager(self) -> JobManager:
        """Get or create quantum job manager."""
        if self._executor is None:
            self._executor = JobManager()
        return self._executor

    def _execute_circuit(
        self,
        circuit: Circuit,
        method: str | None = None,
        device_name: str | None = None,
        shots: int | None = None,
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
        self.circuit = circuit
        actual_method = (
            method if method is not None else getattr(self, "method", "statevector")
        )
        actual_device_name = (
            device_name
            if device_name is not None
            else getattr(self, "device_name", "QpiAI-QSV-Local")
        )
        run_kwargs: dict[str, Any] = {
            "shots": actual_shots,
            "method": actual_method,
            "device_name": actual_device_name,
        }
        valid_run_params = ["experiment_name", "reverse_bits", "access_token"]
        for param in valid_run_params:
            if param in kwargs:
                run_kwargs[param] = kwargs[param]
        run_kwargs.update(getattr(self, "_execution_options", {}))
        run_kwargs.update({"need_statevector": False, "need_density_matrix": False})
        return super().run(**run_kwargs)

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
        elif self._resolved_gradient_method() == "finite_difference":
            expected_evals = self.max_iterations * (2 * n_params + 1) + 1
            print(f"Expected Evals:   ~{expected_evals} (GRADIENT-BASED)")
            print(
                f"  |-- {self.max_iterations} iter x (2x{n_params} for gradient + 1) + 1"
            )
        else:
            strategy = self._resolved_gradient_method()
            print(
                f"Gradient method:  {(strategy if isinstance(strategy, str) else 'custom')}"
            )
            print(
                "Circuit evaluations and shots are recorded for every gradient probe."
            )
        print("\nNote: Optimizer may stop earlier if convergence criteria are met.")
        print("=" * 70 + "\n")

    def _optimize_parameters(
        self,
        objective: Callable[[np.ndarray], float],
        initial_point: np.ndarray,
        n_params: int,
        **kwargs,
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """
        Run optimization loop to minimize objective function.

        Args:
            objective: Cost function to minimize
            initial_point: Starting parameter values
            n_params: Number of parameters
            **kwargs: Additional optimizer parameters

        Returns:
            tuple of (optimal_params, optimal_value, optimization_info)
        """
        if n_params == 0:
            point = np.asarray(initial_point, dtype=float).copy()
            return (
                point,
                float(objective(point)),
                {
                    "success": True,
                    "nfev": 1,
                    "nit": 0,
                    "message": "Fixed circuit evaluated; no adjustable angles",
                },
            )
        GRADIENT_FREE_OPTIMIZERS = {
            "COBYLA",
            "SPSA",
            "DIFFERENTIAL_EVOLUTION",
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
                optimizer_kwargs = dict(kwargs)
                maxfun = optimizer_kwargs.pop("maxfun", base_evaluations + buffer)
                return cobyla_optimize(
                    objective=objective,
                    initial_point=initial_point,
                    maxiter=self.max_iterations,
                    maxfun=maxfun,
                    rhobeg=optimizer_kwargs.pop("rhobeg", 1.0),
                    **optimizer_kwargs,
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
                optimizer_kwargs = dict(kwargs)
                bounds = optimizer_kwargs.pop("bounds", [(0, 2 * np.pi)] * n_params)
                return genetic_algorithm_optimize(
                    objective=objective,
                    bounds=bounds,
                    maxiter=self.max_iterations,
                    **optimizer_kwargs,
                )
            elif optimizer_name == "PARTICLE_SWARM":
                optimizer_kwargs = dict(kwargs)
                bounds = optimizer_kwargs.pop("bounds", [(0, 2 * np.pi)] * n_params)
                return particle_swarm_optimize(
                    objective=objective,
                    bounds=bounds,
                    maxiter=self.max_iterations,
                    **optimizer_kwargs,
                )
            elif optimizer_name == "DIFFERENTIAL_EVOLUTION":
                optimizer_kwargs = dict(kwargs)
                bounds = optimizer_kwargs.pop("bounds", [(0, 2 * np.pi)] * n_params)
                return differential_evolution_optimize(
                    objective=objective,
                    bounds=bounds,
                    maxiter=self.max_iterations,
                    **optimizer_kwargs,
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
            f"Unknown optimizer: '{self.optimizer}'\\n\\nSupported gradient-free optimizers:\\n  {', '.join(sorted(o.lower() for o in GRADIENT_FREE_OPTIMIZERS))}\\n\\nSupported gradient-based optimizers:\\n  {', '.join(sorted(o.lower() for o in GRADIENT_BASED_OPTIMIZERS))}\\n\\nExample: QAOASolver(layers=2, optimizer='cobyla')"
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

    def with_layers(self, layers: int):
        """Set QAOA layers and return self for method chaining."""
        if not isinstance(layers, int) or isinstance(layers, bool) or layers < 1:
            raise ValueError("layers must be a positive integer")
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
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.max_iterations = max_iter
        return self

    def with_ansatz(self, ansatz: str | Callable | None):
        """Set ansatz and return self for method chaining."""
        if not callable(ansatz) and ansatz not in {
            None,
            "standard",
            "hardware_efficient",
        }:
            raise ValueError(f"Unknown ansatz type: {ansatz}")
        if ansatz not in (None, "standard") and (
            self.initial_state is not None or self.mixer != "x"
        ):
            raise ValueError(
                "initial_state and mixer components require the standard ansatz"
            )
        self.ansatz = ansatz
        return self

    def with_mixer(self, mixer: str | Mixer) -> "QAOASolver":
        """Set mixer type and return self for method chaining."""
        if isinstance(mixer, str) and mixer not in {"x", "xy"}:
            raise ValueError("mixer must be x, xy, or a mixer factory")
        if not isinstance(mixer, str) and (not callable(mixer)):
            raise TypeError("mixer must be x, xy, or a mixer factory")
        if self.ansatz not in (None, "standard") and mixer != "x":
            raise ValueError("mixer components require the standard ansatz")
        self.mixer = mixer
        return self

    def with_shots(self, shots: int):
        """Set number of shots and return self for method chaining."""
        if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
            raise ValueError("shots must be a positive integer")
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

    def _make_start_points(
        self, n_params: int, rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Create validated, reproducible starting points for optimization."""
        starts: list[np.ndarray] = []
        if self.initial_params is not None:
            initial = np.asarray(self.initial_params, dtype=float)
            if initial.ndim != 1 or initial.size != n_params:
                raise ValueError(
                    f"initial_params has {initial.size} parameters but ansatz requires {n_params} parameters"
                )
            if not np.all(np.isfinite(initial)):
                raise ValueError("initial_params must contain only finite values")
            starts.append(initial.copy())
        while len(starts) < self.num_starts:
            starts.append(rng.uniform(0.0, np.pi / 2.0, n_params))
        return starts

    def _objective_evaluation_limit(self, shots: int) -> int | None:
        """Return the objective-evaluation limit after reserving final sampling."""
        limit = self.max_evaluations
        if self.max_shots is not None:
            shot_limited_evaluations = self.max_shots // shots - 1
            if shot_limited_evaluations < 1:
                raise ValueError(
                    "max_shots must cover at least one objective evaluation and one final sampling execution"
                )
            limit = (
                shot_limited_evaluations
                if limit is None
                else min(limit, shot_limited_evaluations)
            )
        return limit

    def run(  # type: ignore[override]
        self,
        shots: int = 1024,
        experiment_name: str = "Default Experiment",
        need_statevector: bool | None = None,
        need_density_matrix: bool | None = None,
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
        if "execution_kind" in kwargs or "result_transport" in kwargs:
            raise TypeError(
                "execution_kind and result_transport are not supported by this SDK"
            )
        from qpiai_quantum.jobmanager.backend import get_device_info

        profile = get_device_info(device_name, kwargs.get("access_token"))
        method = profile.method
        self._execution_options = {
            key: kwargs.pop(key) for key in ("access_token",) if key in kwargs
        }
        if "problem" in kwargs:
            self.problem = kwargs.pop("problem")
        if self.problem is None:
            raise ValueError(
                "QAOA requires a problem parameter (either in __init__ or run)"
            )
        if not isinstance(shots, int) or isinstance(shots, bool) or shots < 1:
            raise ValueError("shots must be a positive integer")
        self.shots = shots
        if not hasattr(self.problem, "n_qubits"):
            raise ValueError("Problem must have n_qubits attribute")
        if not hasattr(self.problem, "get_hamiltonian_terms"):
            raise ValueError("Problem must have get_hamiltonian_terms() method")
        self.num_qubits = self.problem.n_qubits
        effective_gradient_method = self._resolved_gradient_method()
        ansatz = self._get_ansatz()
        n_params = count_parameters(ansatz)
        rng = np.random.default_rng(self.seed)
        self._gradient_rng = np.random.default_rng(self.seed)
        start_points = self._make_start_points(n_params, rng)
        evaluation_limit = self._objective_evaluation_limit(shots)
        evaluation_callback = kwargs.pop("evaluation_callback", None)
        if evaluation_callback is not None and (not callable(evaluation_callback)):
            raise TypeError("evaluation_callback must be callable")
        evaluations: list[QAOAEvaluation] = []
        start_results: list[QAOAStartResult] = []
        optimization_results: list[tuple[np.ndarray, float, dict[str, Any]]] = []
        stochastic_optimizers = {
            "SPSA",
            "DIFFERENTIAL_EVOLUTION",
            "GENETIC_ALGORITHM",
            "PARTICLE_SWARM",
        }
        for start_index, initial_point in enumerate(start_points):
            if evaluation_limit is not None and len(evaluations) >= evaluation_limit:
                break
            first_evaluation = len(evaluations)
            progress = OptimizationProgress()
            start_evaluation_limit: int | None = None
            if evaluation_limit is not None:
                remaining_evaluations = evaluation_limit - len(evaluations)
                remaining_starts = len(start_points) - start_index
                start_allowance = max(1, remaining_evaluations // remaining_starts)
                start_evaluation_limit = len(evaluations) + start_allowance

            def evaluate(
                theta: np.ndarray,
                circuit_override: Circuit | None,
                gradient_probe: bool,
                current_start: int = start_index,
                current_limit: int | None = start_evaluation_limit,
            ) -> float:
                """Compute and record energy for one parameter vector."""
                if (
                    evaluation_limit is not None
                    and len(evaluations) >= evaluation_limit
                    or (current_limit is not None and len(evaluations) >= current_limit)
                ):
                    raise _EvaluationBudgetExceeded
                parameter_array = np.asarray(theta, dtype=float)
                circuit = (
                    circuit_override
                    if circuit_override is not None
                    else self._build_circuit(ansatz, parameter_array)
                )
                result = self._execute_circuit(
                    circuit,
                    method=method,
                    device_name=device_name,
                    shots=shots,
                    experiment_name=experiment_name,
                    reverse_bits=reverse_bits,
                )
                expectation = self._compute_expectation(result)
                energy = self._aggregate_objective(result, expectation)
                evaluation = QAOAEvaluation(
                    start_index=current_start,
                    evaluation_index=len(evaluations),
                    parameters=parameter_array.tolist(),
                    energy=float(energy),
                    shots=shots,
                    expectation=float(expectation),
                    kind="gradient_probe" if gradient_probe else "candidate",
                )
                evaluations.append(evaluation)
                if evaluation_callback is not None:
                    evaluation_callback(evaluation)
                return float(energy)

            objective = CircuitObjective(evaluate)
            optimizer_kwargs = dict(kwargs)
            optimizer_kwargs["progress"] = progress
            if self.optimizer in stochastic_optimizers:
                optimizer_kwargs.setdefault("seed", int(rng.integers(0, 2**63 - 1)))
            if evaluation_limit is not None and self.optimizer == "COBYLA":
                if start_evaluation_limit is None:
                    raise RuntimeError("missing per-start evaluation limit")
                start_allowance = start_evaluation_limit - len(evaluations)
                requested = optimizer_kwargs.get("maxfun", start_allowance)
                if requested is None:
                    requested = start_allowance
                if (
                    not isinstance(requested, int)
                    or isinstance(requested, bool)
                    or requested < 1
                ):
                    raise ValueError("maxfun must be a positive integer")
                optimizer_kwargs["maxfun"] = min(requested, start_allowance)
            try:
                opt_result = self._optimize_parameters(
                    objective=objective,
                    initial_point=initial_point,
                    n_params=n_params,
                    **optimizer_kwargs,
                )
            except _EvaluationBudgetExceeded:
                start_evaluations = evaluations[first_evaluation:]
                if not start_evaluations:
                    break
                candidates = [
                    item for item in start_evaluations if item.kind == "candidate"
                ]
                if not candidates:
                    raise RuntimeError(
                        "gradient consumed the budget before a candidate was evaluated"
                    )
                best_evaluation = min(candidates, key=lambda item: item.energy)
                opt_result = (
                    np.asarray(best_evaluation.parameters, dtype=float),
                    best_evaluation.energy,
                    {
                        "nfev": len(start_evaluations),
                        "nit": progress.completed_iterations,
                        "callback_reports": progress.callback_reports,
                        "history": [item.energy for item in candidates],
                        "param_history": [
                            np.asarray(item.parameters, dtype=float)
                            for item in candidates
                        ],
                        "success": False,
                        "message": "QAOA objective-evaluation budget reached",
                        "termination_reason": "evaluation_budget",
                    },
                )
            params_for_start, energy_for_start, info_for_start = opt_result
            info_for_start = dict(info_for_start)
            info_for_start.setdefault("callback_reports", progress.callback_reports)
            reason = info_for_start.get("termination_reason")
            if reason is None:
                if (
                    not info_for_start.get("success", False)
                    and start_evaluation_limit is not None
                    and (len(evaluations) >= start_evaluation_limit)
                ):
                    reason = "evaluation_budget"
                elif (
                    progress.completed_iterations is not None
                    and progress.completed_iterations >= self.max_iterations
                ):
                    reason = "iteration_limit"
                elif info_for_start.get("success", False):
                    reason = "converged"
                else:
                    reason = "optimizer_stopped"
                info_for_start["termination_reason"] = reason
            params_for_start = np.asarray(params_for_start, dtype=float)
            energy_for_start = float(energy_for_start)
            if params_for_start.shape != (n_params,) or not np.all(
                np.isfinite(params_for_start)
            ):
                raise ValueError(
                    "optimizer returned parameters with an invalid shape or non-finite values"
                )
            if not np.isfinite(energy_for_start):
                raise ValueError("optimizer returned a non-finite energy")
            expectation_for_start = energy_for_start
            if self.aggregation is not None:
                matching = [
                    item
                    for item in evaluations[first_evaluation:]
                    if item.kind == "candidate"
                    and item.parameters == params_for_start.tolist()
                ]
                if not matching:
                    raise ValueError(
                        "optimizer must return an evaluated candidate for a custom objective"
                    )
                selected = min(
                    matching, key=lambda item: abs(item.energy - energy_for_start)
                )
                if selected.expectation is None:
                    raise RuntimeError("candidate is missing its expectation estimate")
                expectation_for_start = float(selected.expectation)
                info_for_start["expectation_at_optimum"] = expectation_for_start
            normalized_result = (params_for_start, energy_for_start, info_for_start)
            optimization_results.append(normalized_result)
            start_results.append(
                QAOAStartResult(
                    start_index=start_index,
                    initial_parameters=initial_point.tolist(),
                    optimal_parameters=np.asarray(params_for_start).tolist(),
                    optimal_energy=float(expectation_for_start),
                    optimal_objective=float(energy_for_start),
                    evaluations=len(evaluations) - first_evaluation,
                    iterations=info_for_start.get("nit"),
                    success=bool(info_for_start.get("success", False)),
                    message=str(info_for_start.get("message", "")),
                    termination_reason=reason,
                    callback_reports=int(info_for_start["callback_reports"]),
                )
            )
        if not optimization_results:
            raise RuntimeError("QAOA did not perform any objective evaluations")
        best_start_index = min(
            range(len(optimization_results)),
            key=lambda index: float(optimization_results[index][1]),
        )
        best_params, best_energy, opt_info = optimization_results[best_start_index]
        iteration_counts = [item.iterations for item in start_results]
        actual_iterations = (
            sum(count for count in iteration_counts if count is not None)
            if all(count is not None for count in iteration_counts)
            else None
        )
        actual_nfev = len(evaluations)
        if self.verbose:
            print("\n" + "=" * 70)
            print("QAOA OPTIMIZATION COMPLETED")
            print("=" * 70)
            print(f"Actual Iterations:   {actual_iterations}")
            print(f"Iterations per Start: {self.max_iterations}")
            print(f"Circuit Evaluations: {actual_nfev}")
            print(f"Final Objective:     {best_energy:.6f}")
            if self.aggregation is not None:
                print(f"Expectation:         {opt_info['expectation_at_optimum']:.6f}")
            print(f"Success:             {opt_info.get('success', False)}")
            print(f"Termination:         {opt_info['termination_reason']}")
            print("=" * 70 + "\n")
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
        best_bitstring, most_probable_bitstring, sampled_energies = (
            self._select_best_sample(final_counts)
        )
        if hasattr(self.problem, "decode_solution"):
            solution = self.problem.decode_solution(best_bitstring)
        else:
            solution = best_bitstring
        if hasattr(self.problem, "validate_solution"):
            is_valid, validation_message = self.problem.validate_solution(solution)
        else:
            is_valid = True
            validation_message = "Validation not available"
        if hasattr(self.problem, "compute_solution_quality"):
            quality_metrics = self.problem.compute_solution_quality(solution)
        else:
            quality_metrics = {}
        return QAOAResult(
            optimal_parameters=best_params.tolist(),
            optimal_energy=float(opt_info.get("expectation_at_optimum", best_energy)),
            optimal_objective=float(best_energy),
            energy_history=opt_info.get("history", []),
            param_history=[p.tolist() for p in opt_info.get("param_history", [])],
            solution=solution,
            bitstring=best_bitstring,
            counts=final_counts,
            is_valid=is_valid,
            validation_message=validation_message,
            quality_metrics=quality_metrics,
            most_probable_bitstring=most_probable_bitstring,
            best_sampled_energy=sampled_energies[best_bitstring],
            sampled_energies=sampled_energies,
            evaluations=evaluations,
            start_results=start_results,
            metadata={
                "success": opt_info.get("success", False),
                "actual_iterations": actual_iterations,
                "iterations_per_start_limit": self.max_iterations,
                "max_iterations_limit": self.max_iterations * len(start_results),
                "objective_evaluations": actual_nfev,
                "final_sampling_evaluations": 1,
                "circuit_evaluations": actual_nfev + 1,
                "optimization_shots": actual_nfev * shots,
                "final_sampling_shots": shots,
                "total_shots": (actual_nfev + 1) * shots,
                "max_evaluations": self.max_evaluations,
                "effective_objective_evaluation_limit": evaluation_limit,
                "max_shots": self.max_shots,
                "message": opt_info.get("message", ""),
                "termination_reason": opt_info["termination_reason"],
                "callback_reports": sum(
                    item.callback_reports for item in start_results
                ),
                "budget_exhausted": any(
                    item.termination_reason == "evaluation_budget"
                    for item in start_results
                ),
                "method": method,
                "device_name": device_name,
                "experiment_name": experiment_name,
                "reverse_bits": reverse_bits,
                "shots": shots,
                "n_parameters": n_params,
                "optimizer": self.optimizer,
                "seed": self.seed,
                "requested_starts": self.num_starts,
                "completed_starts": len(start_results),
                "best_start_index": start_results[best_start_index].start_index,
                "layers": self.layers,
                "ansatz": str(self.ansatz),
                "mixer": self.mixer
                if isinstance(self.mixer, str)
                else type(self.mixer).__name__,
                "aggregation": "expectation"
                if self.aggregation is None
                else type(self.aggregation).__name__,
                "cvar_alpha": self.aggregation.alpha
                if isinstance(self.aggregation, CVaR)
                else None,
                "gradient_method": self.gradient_method
                if isinstance(self.gradient_method, str)
                else "custom",
                "effective_gradient_method": effective_gradient_method
                if isinstance(effective_gradient_method, str)
                else "custom",
                "gradient_probe_evaluations": sum(
                    item.kind == "gradient_probe" for item in evaluations
                ),
                "converged_early": any(
                    item.termination_reason == "converged"
                    and item.iterations is not None
                    and (item.iterations < self.max_iterations)
                    for item in start_results
                ),
                "energy_semantics": "estimated_expectation",
                "history_semantics": "optimization_objective",
                "sample_selection": "lowest_energy_feasible_then_highest_count",
            },
        )

    def to_dict(self, result: "QAOAResult") -> dict[str, Any]:
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
            "optimal_objective": result.optimal_objective,
            "energy_history": result.energy_history,
            "param_history": result.param_history,
            "solution": result.solution,
            "bitstring": result.bitstring,
            "counts": result.counts,
            "is_valid": result.is_valid,
            "validation_message": result.validation_message,
            "quality_metrics": result.quality_metrics,
            "most_probable_bitstring": result.most_probable_bitstring,
            "best_sampled_energy": result.best_sampled_energy,
            "sampled_energies": result.sampled_energies,
            "evaluations": [asdict(item) for item in result.evaluations],
            "start_results": [asdict(item) for item in result.start_results],
            "metadata": result.metadata,
        }

    def get_name(self) -> str:
        """Get the name of this QAOA instance."""
        return f"QAOA (layers={self.layers}, optimizer={self.optimizer})"
