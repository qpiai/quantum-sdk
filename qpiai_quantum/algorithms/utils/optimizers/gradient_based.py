"""Gradient-based optimization methods with complete history tracking."""

from typing import Dict, List, Optional, Tuple, Union
from collections.abc import Callable
import numpy as np
from scipy.optimize import minimize


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

    def create_callback(self, user_callback: Callable | None = None):
        """Create a callback that tracks iterations."""

        def callback_wrapper(xk):
            self.iteration_count += 1
            if user_callback is not None:
                try:
                    user_callback(xk)
                except Exception:
                    pass

        return callback_wrapper


def gradient_descent_optimize(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    initial_point: np.ndarray,
    learning_rate: float = 0.01,
    maxiter: int = 100,
    tol: float = 1e-6,
    **kwargs,
) -> tuple[np.ndarray, float, dict]:
    """
    Basic gradient descent optimizer with complete history tracking.

    This optimizer performs exactly maxiter iterations unless it converges early.
    Each iteration computes gradient (1 eval) and evaluates objective (1 eval).

    Args:
        objective: Function to minimize
        gradient: Function that computes gradient
        initial_point: Starting point
        learning_rate: Step size for updates
        maxiter: Maximum number of iterations (will run exactly this many unless converged)
        tol: Convergence tolerance (if improvement < tol, stop early)
        **kwargs: Additional arguments (callback, etc.)

    Returns:
        Tuple containing:
        - Optimal parameters
        - Final objective value
        - Dictionary with optimization details including:
            - nit: Actual number of iterations performed (≤ maxiter)
            - nfev: Total number of function evaluations
            - history: Energy values at each iteration
            - param_history: Parameter values at each iteration
            - success: Whether optimization succeeded
    """
    # Use the history tracker
    tracker = HistoryTracker()

    params = initial_point.copy()
    best_params = params.copy()
    best_value = objective(params)

    # Track initial point
    tracker.energy_history.append(float(best_value))
    tracker.param_history.append(params.copy())

    nfev = 1  # Number of function evaluations
    converged = False
    actual_iterations = 0

    user_callback = kwargs.get("callback")

    for iteration in range(maxiter):
        actual_iterations = iteration + 1

        # Compute gradient
        grad = gradient(params)
        nfev += 1

        # Update parameters
        params = params - learning_rate * grad

        # Evaluate objective
        value = objective(params)
        nfev += 1

        # Track history
        tracker.energy_history.append(float(value))
        tracker.param_history.append(params.copy())

        # Update best
        if value < best_value:
            best_value = value
            best_params = params.copy()

        # User callback
        if user_callback is not None:
            try:
                user_callback(params)
            except Exception:
                pass

        # Check convergence
        if len(tracker.energy_history) > 1:
            improvement = abs(tracker.energy_history[-2] - tracker.energy_history[-1])
            if improvement < tol:
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
            "status": 0 if converged else 1,
            "success": converged or actual_iterations == maxiter,
            "message": f"Gradient descent {'converged' if converged else 'completed'} after {actual_iterations} iterations.",
        },
    )


def adam_optimize(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    initial_point: np.ndarray,
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    maxiter: int = 100,
    tol: float = 1e-6,
    **kwargs,
) -> tuple[np.ndarray, float, dict]:
    """
    Adam optimizer (Adaptive Moment Estimation) with complete history tracking.

    Adam uses adaptive learning rates for each parameter based on first and second
    moments of the gradients. This makes it very effective for many problems.

    This optimizer performs exactly maxiter iterations unless it converges early.
    Each iteration computes gradient (1 eval) and evaluates objective (1 eval).

    Args:
        objective: Function to minimize
        gradient: Function that computes gradient
        initial_point: Starting point
        learning_rate: Base learning rate (alpha)
        beta1: Exponential decay rate for first moment estimates (typically 0.9)
        beta2: Exponential decay rate for second moment estimates (typically 0.999)
        epsilon: Small constant for numerical stability (typically 1e-8)
        maxiter: Maximum number of iterations (will run exactly this many unless converged)
        tol: Convergence tolerance (if improvement < tol, stop early)
        **kwargs: Additional arguments (callback, etc.)

    Returns:
        Tuple containing:
        - Optimal parameters
        - Final objective value
        - Dictionary with optimization details including:
            - nit: Actual number of iterations performed (≤ maxiter)
            - nfev: Total number of function evaluations
            - history: Energy values at each iteration
            - param_history: Parameter values at each iteration
            - success: Whether optimization succeeded
    """
    # Use the history tracker
    tracker = HistoryTracker()

    params = initial_point.copy()
    best_params = params.copy()
    best_value = objective(params)

    # Track initial point
    tracker.energy_history.append(float(best_value))
    tracker.param_history.append(params.copy())

    nfev = 1
    converged = False
    actual_iterations = 0

    # Initialize moment vectors
    m = np.zeros_like(params)  # First moment
    v = np.zeros_like(params)  # Second moment

    user_callback = kwargs.get("callback")

    for iteration in range(maxiter):
        actual_iterations = iteration + 1

        # Compute gradient
        grad = gradient(params)
        nfev += 1

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad

        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * grad**2

        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** (iteration + 1))

        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** (iteration + 1))

        # Update parameters
        params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # Evaluate objective
        value = objective(params)
        nfev += 1

        # Track history
        tracker.energy_history.append(float(value))
        tracker.param_history.append(params.copy())

        # Update best
        if value < best_value:
            best_value = value
            best_params = params.copy()

        # User callback
        if user_callback is not None:
            try:
                user_callback(params)
            except Exception:
                pass

        # Check convergence
        if len(tracker.energy_history) > 1:
            improvement = abs(tracker.energy_history[-2] - tracker.energy_history[-1])
            if improvement < tol:
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
            "status": 0 if converged else 1,
            "success": converged or actual_iterations == maxiter,
            "message": f"Adam optimizer {'converged' if converged else 'completed'} after {actual_iterations} iterations.",
        },
    )


def adagrad_optimize(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    initial_point: np.ndarray,
    learning_rate: float = 0.01,
    epsilon: float = 1e-8,
    maxiter: int = 100,
    tol: float = 1e-6,
    **kwargs,
) -> tuple[np.ndarray, float, dict]:
    """
    Adagrad optimizer (Adaptive Gradient Algorithm) with complete history tracking.

    Adagrad adapts the learning rate for each parameter based on historical gradients.
    It performs larger updates for infrequent parameters and smaller updates for frequent ones.

    This optimizer performs exactly maxiter iterations unless it converges early.
    Each iteration computes gradient (1 eval) and evaluates objective (1 eval).

    Args:
        objective: Function to minimize
        gradient: Function that computes gradient
        initial_point: Starting point
        learning_rate: Initial learning rate
        epsilon: Small constant for numerical stability (typically 1e-8)
        maxiter: Maximum number of iterations (will run exactly this many unless converged)
        tol: Convergence tolerance (if improvement < tol, stop early)
        **kwargs: Additional arguments (callback, etc.)

    Returns:
        Tuple containing:
        - Optimal parameters
        - Final objective value
        - Dictionary with optimization details including:
            - nit: Actual number of iterations performed (≤ maxiter)
            - nfev: Total number of function evaluations
            - history: Energy values at each iteration
            - param_history: Parameter values at each iteration
            - success: Whether optimization succeeded
    """
    # Use the history tracker
    tracker = HistoryTracker()

    params = initial_point.copy()
    best_params = params.copy()
    best_value = objective(params)

    # Track initial point
    tracker.energy_history.append(float(best_value))
    tracker.param_history.append(params.copy())

    nfev = 1
    converged = False
    actual_iterations = 0

    # Initialize accumulated squared gradients
    G = np.zeros_like(params)

    user_callback = kwargs.get("callback")

    for iteration in range(maxiter):
        actual_iterations = iteration + 1

        # Compute gradient
        grad = gradient(params)
        nfev += 1

        # Accumulate squared gradients
        G += grad**2

        # Update parameters with adapted learning rate
        params = params - learning_rate * grad / (np.sqrt(G) + epsilon)

        # Evaluate objective
        value = objective(params)
        nfev += 1

        # Track history
        tracker.energy_history.append(float(value))
        tracker.param_history.append(params.copy())

        # Update best
        if value < best_value:
            best_value = value
            best_params = params.copy()

        # User callback
        if user_callback is not None:
            try:
                user_callback(params)
            except Exception:
                pass

        # Check convergence
        if len(tracker.energy_history) > 1:
            improvement = abs(tracker.energy_history[-2] - tracker.energy_history[-1])
            if improvement < tol:
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
            "status": 0 if converged else 1,
            "success": converged or actual_iterations == maxiter,
            "message": f"Adagrad optimizer {'converged' if converged else 'completed'} after {actual_iterations} iterations.",
        },
    )


def l_bfgs_b_optimize(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray] | None = None,
    initial_point: np.ndarray = None,
    bounds: list[tuple[float, float]] | None = None,
    maxiter: int = 100,
    tol: float = 1e-6,
    **kwargs,
) -> tuple[np.ndarray, float, dict]:
    """
    L-BFGS-B optimizer (Limited-memory BFGS with bounds) with complete history tracking.

    L-BFGS-B is a quasi-Newton method that approximates the Hessian using a limited
    amount of memory. It's very efficient for large-scale problems and supports
    box constraints.

    Args:
        objective: Function to minimize
        gradient: Function that computes gradient (optional, will use finite differences if None)
        initial_point: Starting point
        bounds: List of (lower, upper) bounds for each parameter (optional)
        maxiter: Maximum number of iterations
        tol: Tolerance for termination (both function and gradient)
        **kwargs: Additional arguments (callback, options, etc.)

    Returns:
        Tuple containing:
        - Optimal parameters
        - Final objective value
        - Dictionary with optimization details including:
            - nit: Number of iterations performed
            - nfev: Total number of function evaluations
            - history: Energy values at each evaluation (wrapped objective)
            - param_history: Parameter values at each evaluation
            - success: Whether optimization succeeded
    """
    # Use the history tracker
    tracker = HistoryTracker()
    wrapped_objective = tracker.create_wrapped_objective(objective)
    callback_wrapper = tracker.create_callback(kwargs.get("callback"))

    result = minimize(
        wrapped_objective,
        initial_point,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds,
        callback=callback_wrapper,
        options={
            "maxiter": maxiter,
            "ftol": tol,
            "gtol": tol,
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


def cg_optimize(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray] | None = None,
    initial_point: np.ndarray = None,
    maxiter: int = 100,
    tol: float = 1e-6,
    **kwargs,
) -> tuple[np.ndarray, float, dict]:
    """
    Conjugate Gradient optimizer with complete history tracking.

    CG is an iterative method for solving systems of linear equations and
    optimization problems. It's particularly effective for quadratic objectives.

    Args:
        objective: Function to minimize
        gradient: Function that computes gradient (optional, will use finite differences if None)
        initial_point: Starting point
        maxiter: Maximum number of iterations
        tol: Tolerance for gradient norm at termination
        **kwargs: Additional arguments (callback, options, etc.)

    Returns:
        Tuple containing:
        - Optimal parameters
        - Final objective value
        - Dictionary with optimization details including:
            - nit: Number of iterations performed
            - nfev: Total number of function evaluations
            - history: Energy values at each evaluation (wrapped objective)
            - param_history: Parameter values at each evaluation
            - success: Whether optimization succeeded
    """
    # Use the history tracker
    tracker = HistoryTracker()
    wrapped_objective = tracker.create_wrapped_objective(objective)
    callback_wrapper = tracker.create_callback(kwargs.get("callback"))

    result = minimize(
        wrapped_objective,
        initial_point,
        method="CG",
        jac=gradient,
        callback=callback_wrapper,
        options={"maxiter": maxiter, "gtol": tol, **kwargs.get("options", {})},
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
