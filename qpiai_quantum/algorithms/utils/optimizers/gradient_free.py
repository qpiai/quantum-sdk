# Gradient-free optimizers for quantum algorithms
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize, differential_evolution


class HistoryTracker:
    """
    Tracks energy and parameter history during optimization.
    This can be shared across all optimizers.
    """

    def __init__(self):
        self.energy_history = []
        self.param_history = []
        self.iteration_count = 0

    def reset(self):
        """Reset all history."""
        self.energy_history = []
        self.param_history = []
        self.iteration_count = 0

    def create_wrapped_objective(self, objective: Callable[[np.ndarray], float]):
        """Create a wrapped objective function that tracks history."""

        def wrapped_objective(params):
            energy = objective(params)
            self.energy_history.append(float(energy))
            self.param_history.append(params.copy())
            return energy

        return wrapped_objective

    def create_callback(self, user_callback: Optional[Callable] = None):
        """Create a callback that tracks iterations."""

        def callback_wrapper(xk):
            self.iteration_count += 1
            if user_callback is not None:
                try:
                    user_callback(xk)
                except Exception:
                    pass

        return callback_wrapper


def cobyla_optimize(
    objective: Callable[[np.ndarray], float],
    initial_point: np.ndarray,
    bounds: Optional[List[Tuple[float, float]]] = None,
    maxiter: int = 100,
    maxfun: Optional[int] = None,
    tol: float = 1e-6,
    **kwargs,
) -> Tuple[np.ndarray, float, Dict]:
    """
    COBYLA (Constrained Optimization BY Linear Approximation) optimizer.
    """
    n_params = len(initial_point)
    if maxfun is None:
        if maxiter <= 5:
            buffer = 10 + maxiter * 2
        elif maxiter <= 20:
            buffer = 20 + maxiter
        else:
            buffer = 50
        maxfun = maxiter * (n_params + 2) + buffer

    # Use the history tracker
    tracker = HistoryTracker()
    wrapped_objective = tracker.create_wrapped_objective(objective)
    callback_wrapper = tracker.create_callback(kwargs.get("callback"))

    constraints = []
    if bounds is not None:
        for i, (lower, upper) in enumerate(bounds):
            if lower is not None:
                constraints.append(
                    {"type": "ineq", "fun": lambda x, idx=i, lb=lower: x[idx] - lb}
                )
            if upper is not None:
                constraints.append(
                    {"type": "ineq", "fun": lambda x, idx=i, ub=upper: ub - x[idx]}
                )

    result = minimize(
        wrapped_objective,
        initial_point,
        method="COBYLA",
        constraints=constraints,
        callback=callback_wrapper,
        options={"maxiter": maxiter, "tol": tol, **kwargs.get("options", {})},
    )

    if tracker.iteration_count > 0:
        actual_nit = tracker.iteration_count
    elif hasattr(result, "nit") and result.nit > 0:
        actual_nit = result.nit
    elif result.nfev > 0:
        actual_nit = max(1, min(result.nfev // (n_params + 2), maxiter))
    else:
        actual_nit = max(1, len(tracker.energy_history))

    return (
        result.x,
        result.fun,
        {
            "nfev": result.nfev,
            "nit": actual_nit,
            "history": tracker.energy_history,
            "param_history": tracker.param_history,
            "status": result.status,
            "success": result.success,
            "message": result.message,
        },
    )


def nelder_mead_optimize(
    objective: Callable[[np.ndarray], float],
    initial_point: np.ndarray,
    bounds: Optional[List[Tuple[float, float]]] = None,
    maxiter: int = 100,
    tol: float = 1e-6,
    **kwargs,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Nelder-Mead simplex optimizer.
    """
    # Use the history tracker
    tracker = HistoryTracker()
    wrapped_objective = tracker.create_wrapped_objective(objective)
    callback_wrapper = tracker.create_callback(kwargs.get("callback"))

    constraints = []
    if bounds is not None:
        for i, (lower, upper) in enumerate(bounds):
            if lower is not None:
                constraints.append(
                    {"type": "ineq", "fun": lambda x, idx=i, lb=lower: x[idx] - lb}
                )
            if upper is not None:
                constraints.append(
                    {"type": "ineq", "fun": lambda x, idx=i, ub=upper: ub - x[idx]}
                )

    result = minimize(
        wrapped_objective,
        initial_point,
        method="Nelder-Mead",
        constraints=constraints,
        callback=callback_wrapper,
        options={
            "maxiter": maxiter,
            "xatol": tol,
            "fatol": tol,
            **kwargs.get("options", {}),
        },
    )

    return (
        result.x,
        result.fun,
        {
            "nfev": result.nfev,
            "nit": result.nit,
            "history": tracker.energy_history,
            "param_history": tracker.param_history,
            "status": result.status,
            "success": result.success,
            "message": result.message,
        },
    )


def spsa_optimize(
    objective: Callable[[np.ndarray], float],
    initial_point: np.ndarray,
    maxiter: int = 100,
    tol: float = 1e-6,
    learning_rate: float = 0.1,
    perturbation_scale: float = 0.1,
    **kwargs,
) -> Tuple[np.ndarray, float, Dict]:
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.

    This optimizer performs exactly maxiter iterations unless it converges early.
    Each iteration evaluates the objective 3 times (plus, minus, and current).

    Args:
        objective: Function to minimize
        initial_point: Starting point for optimization
        maxiter: Maximum number of iterations (will run exactly this many unless converged)
        tol: Tolerance for early stopping (if improvement < tol, stop)
        learning_rate: Step size for parameter updates
        perturbation_scale: Size of perturbations for gradient estimation
        **kwargs: Additional arguments

    Returns:
        Tuple containing:
        - Optimal parameters
        - Final objective value
        - Dictionary with optimization details including:
            - nit: Actual number of iterations performed (≤ maxiter)
            - nfev: Total number of function evaluations
            - history: Energy values at each iteration
            - param_history: Parameter values at each iteration
    """
    # Use the history tracker
    tracker = HistoryTracker()

    params = initial_point.copy()
    best_params = params.copy()
    best_value = objective(params)
    tracker.energy_history.append(best_value)
    tracker.param_history.append(params.copy())

    nfev = 1
    converged = False
    actual_iterations = 0

    for iteration in range(maxiter):
        actual_iterations = iteration + 1

        # Generate random perturbation
        delta = 2 * np.random.randint(0, 2, size=len(params)) - 1

        # Evaluate at perturbed points
        params_plus = params + perturbation_scale * delta
        params_minus = params - perturbation_scale * delta

        value_plus = objective(params_plus)
        value_minus = objective(params_minus)
        nfev += 2

        # Approximate gradient
        gradient_approx = (value_plus - value_minus) / (2 * perturbation_scale * delta)

        # Update parameters
        params = params - learning_rate * gradient_approx

        # Evaluate new parameters
        current_value = objective(params)
        nfev += 1
        tracker.energy_history.append(float(current_value))
        tracker.param_history.append(params.copy())

        # Track best
        improvement = best_value - current_value
        if current_value < best_value:
            best_value = current_value
            best_params = params.copy()

        # Check convergence
        if abs(improvement) < tol and iteration > 0:
            converged = True
            break

    return (
        best_params,
        best_value,
        {
            "nfev": nfev,
            "nit": actual_iterations,
            "history": tracker.energy_history,
            "param_history": tracker.param_history,
            "success": converged or actual_iterations == maxiter,
            "message": f"SPSA optimization {'converged' if converged else 'completed'} after {actual_iterations} iterations.",
        },
    )


def slsqp_optimize(
    objective: Callable[[np.ndarray], float],
    initial_point: np.ndarray,
    bounds: Optional[List[Tuple[float, float]]] = None,
    constraints: Optional[List[Dict]] = None,
    maxiter: int = 100,
    tol: float = 1e-6,
    **kwargs,
) -> Tuple[np.ndarray, float, Dict]:
    """
    SLSQP (Sequential Least SQuares Programming) optimizer.
    """
    # Use the history tracker
    tracker = HistoryTracker()
    wrapped_objective = tracker.create_wrapped_objective(objective)
    callback_wrapper = tracker.create_callback(kwargs.get("callback"))

    result = minimize(
        wrapped_objective,
        initial_point,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=callback_wrapper,
        options={"maxiter": maxiter, "ftol": tol, **kwargs.get("options", {})},
    )

    return (
        result.x,
        result.fun,
        {
            "nfev": result.nfev,
            "nit": result.nit,
            "history": tracker.energy_history,
            "param_history": tracker.param_history,
            "status": result.status,
            "success": result.success,
            "message": result.message,
        },
    )


def differential_evolution_optimize(
    objective: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    maxiter: int = 100,
    tol: float = 1e-6,
    popsize: int = 15,
    **kwargs,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Differential Evolution optimizer.
    """
    # Use the history tracker
    tracker = HistoryTracker()
    wrapped_objective = tracker.create_wrapped_objective(objective)

    # Note: differential_evolution doesn't support callbacks in the same way
    # But we can track through the wrapped objective
    result = differential_evolution(
        wrapped_objective, bounds, maxiter=maxiter, tol=tol, popsize=popsize, **kwargs
    )

    return (
        result.x,
        result.fun,
        {
            "nfev": result.nfev,
            "nit": result.nit,
            "history": tracker.energy_history,
            "param_history": tracker.param_history,
            "success": result.success,
            "message": result.message,
        },
    )


def genetic_algorithm_optimize(
    objective: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    maxiter: int = 100,
    population_size: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    **kwargs,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Simple Genetic Algorithm optimizer.

    This optimizer performs exactly maxiter generations.
    Each generation evaluates population_size individuals.

    Args:
        objective: Function to minimize
        bounds: List of (lower, upper) bounds for each parameter
        maxiter: Number of generations (runs exactly this many)
        population_size: Size of the population
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        **kwargs: Additional arguments

    Returns:
        Tuple containing:
        - Optimal parameters
        - Final objective value
        - Dictionary with optimization details including:
            - nit: Number of generations = maxiter
            - nfev: Total function evaluations = maxiter × population_size
            - history: Best fitness in each generation
            - param_history: Best parameters in each generation
    """
    n_params = len(bounds)

    # Use the history tracker
    tracker = HistoryTracker()

    population = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(population_size, n_params),
    )

    best_params = None
    best_fitness = float("inf")
    nfev = 0

    # Run exactly maxiter generations
    for generation in range(maxiter):
        fitness = np.array([objective(individual) for individual in population])
        nfev += population_size

        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_params = population[current_best_idx].copy()

        # Track best in each generation
        tracker.energy_history.append(float(best_fitness))
        tracker.param_history.append(best_params.copy())

        # Selection (tournament selection)
        new_population = []
        for _ in range(population_size):
            idx1, idx2 = np.random.choice(population_size, 2, replace=False)
            if fitness[idx1] < fitness[idx2]:
                new_population.append(population[idx1].copy())
            else:
                new_population.append(population[idx2].copy())

        population = np.array(new_population)

        # Crossover
        for i in range(0, population_size - 1, 2):
            if np.random.random() < crossover_rate:
                crossover_point = np.random.randint(1, n_params)
                temp = population[i][crossover_point:].copy()
                population[i][crossover_point:] = population[i + 1][crossover_point:]
                population[i + 1][crossover_point:] = temp

        # Mutation
        for individual in population:
            for j in range(n_params):
                if np.random.random() < mutation_rate:
                    individual[j] = np.random.uniform(bounds[j][0], bounds[j][1])

    return (
        best_params,
        best_fitness,
        {
            "nfev": nfev,
            "nit": maxiter,  # Always runs exactly maxiter generations
            "history": tracker.energy_history,
            "param_history": tracker.param_history,
            "success": True,
            "message": f"Genetic algorithm completed {maxiter} generations.",
        },
    )


def particle_swarm_optimize(
    objective: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    maxiter: int = 100,
    n_particles: int = 30,
    w: float = 0.9,
    c1: float = 2.0,
    c2: float = 2.0,
    **kwargs,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Particle Swarm Optimization.

    This optimizer performs exactly maxiter iterations.
    Each iteration evaluates n_particles particles.

    Args:
        objective: Function to minimize
        bounds: List of (lower, upper) bounds for each parameter
        maxiter: Number of iterations (runs exactly this many)
        n_particles: Number of particles in the swarm
        w: Inertia weight (0.4-0.9 typical)
        c1: Cognitive parameter - attraction to personal best (2.0 typical)
        c2: Social parameter - attraction to global best (2.0 typical)
        **kwargs: Additional arguments

    Returns:
        Tuple containing:
        - Optimal parameters
        - Final objective value
        - Dictionary with optimization details including:
            - nit: Number of iterations = maxiter
            - nfev: Total function evaluations
            - history: Global best fitness at each iteration
            - param_history: Global best parameters at each iteration
    """
    n_params = len(bounds)

    # Use the history tracker
    tracker = HistoryTracker()

    # Initialize particles
    particles = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(n_particles, n_params),
    )

    velocities = np.random.uniform(-1, 1, size=(n_particles, n_params))

    # Evaluate initial particles
    personal_best = particles.copy()
    personal_best_fitness = np.array([objective(p) for p in particles])

    global_best_idx = np.argmin(personal_best_fitness)
    global_best = personal_best[global_best_idx].copy()
    global_best_fitness = personal_best_fitness[global_best_idx]

    nfev = n_particles

    # Track initial best
    tracker.energy_history.append(float(global_best_fitness))
    tracker.param_history.append(global_best.copy())

    # Run exactly maxiter iterations
    for iteration in range(maxiter):
        for i in range(n_particles):
            # Update velocity
            r1, r2 = np.random.random(), np.random.random()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (personal_best[i] - particles[i])
                + c2 * r2 * (global_best - particles[i])
            )

            # Update position
            particles[i] += velocities[i]

            # Apply bounds
            for j in range(n_params):
                particles[i][j] = np.clip(particles[i][j], bounds[j][0], bounds[j][1])

            # Evaluate
            fitness = objective(particles[i])
            nfev += 1

            # Update personal best
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best[i] = particles[i].copy()

                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best = particles[i].copy()

        # Track global best at end of each iteration
        tracker.energy_history.append(float(global_best_fitness))
        tracker.param_history.append(global_best.copy())

    return (
        global_best,
        global_best_fitness,
        {
            "nfev": nfev,
            "nit": maxiter,  # Always runs exactly maxiter iterations
            "history": tracker.energy_history,
            "param_history": tracker.param_history,
            "success": True,
            "message": f"Particle swarm optimization completed {maxiter} iterations.",
        },
    )


def powell_optimize(
    objective: Callable[[np.ndarray], float],
    initial_point: np.ndarray,
    bounds: Optional[List[Tuple[float, float]]] = None,
    maxiter: int = 100,
    tol: float = 1e-6,
    **kwargs,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Powell optimizer.
    """
    tracker = HistoryTracker()
    wrapped_objective = tracker.create_wrapped_objective(objective)
    callback_wrapper = tracker.create_callback(kwargs.get("callback"))

    result = minimize(
        wrapped_objective,
        initial_point,
        method="Powell",
        bounds=bounds,
        callback=callback_wrapper,
        options={
            "maxiter": maxiter,
            "xtol": tol,
            "ftol": tol,
            **kwargs.get("options", {}),
        },
    )

    return (
        result.x,
        result.fun,
        {
            "nfev": result.nfev,
            "nit": result.nit,
            "history": tracker.energy_history,
            "param_history": tracker.param_history,
            "status": result.status,
            "success": result.success,
            "message": result.message,
        },
    )
