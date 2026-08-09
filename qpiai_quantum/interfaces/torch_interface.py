import torch
import numpy as np
from typing import Callable, Any, Tuple

class ParameterShiftFunction(torch.autograd.Function):
    """
    PyTorch autograd function that implements the parameter-shift rule
    for quantum circuits in the QpiAI Quantum SDK.
    """

    @staticmethod
    def forward(ctx: Any, cost_function: Callable[[np.ndarray], float], parameters: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the quantum cost function for the given parameters.

        Args:
            ctx (Any): PyTorch context for saving information for the backward pass.
            cost_function (Callable[[np.ndarray], float]): A classical function that takes a NumPy 
                array of parameters and returns the expectation value as a float. This abstracts 
                away the circuit building and execution (since `Circuit.execute` does not exist).
            parameters (torch.Tensor): The PyTorch tensor of parameters.

        Returns:
            torch.Tensor: The evaluated expectation value.
        """
        # Save necessary context(ctx) for the backward pass
        ctx.cost_function = cost_function

        params_np = parameters.detach().numpy()

        # Execute the cost function and get expectation value
        expectation_value = cost_function(params_np)

        ctx.save_for_backward(parameters)

        return torch.tensor(expectation_value, dtype=torch.float32)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Compute gradients using the Parameter-Shift Rule.

        Args:
            ctx (Any): PyTorch context containing saved tensors and cost function.
            grad_outputs (Any): The upstream gradients.

        Returns:
            Tuple[None, torch.Tensor]: Tuple containing None for the non-tensor 
            argument (cost_function) and the computed gradients for the parameters.
        """
        grad_output = grad_outputs[0]
        parameters, = ctx.saved_tensors
        cost_function = ctx.cost_function

        params_np = parameters.detach().numpy()
        gradients = torch.zeros_like(parameters)

        shift = torch.pi / 2.0

        # Note: The SDK currently simulates individual circuits sequentially. 
        # To improve this, one could use Python's `concurrent.futures` to evaluate 
        # the forward and backward shifted circuits in parallel across multiple CPU cores.
        for i in range(len(params_np)):
            # Forward shift: 
            params_forward = params_np.copy()
            params_forward[i] += shift
            exp_forward = cost_function(params_forward)

            # Backward shift:
            params_backwards = params_np.copy()
            params_backwards[i] -= shift
            exp_backwards = cost_function(params_backwards)

            # The parameter shift rule:
            gradients[i] = 0.5 * (exp_forward - exp_backwards)

        # returning None for non-tensor arguments (cost_function)
        return None, gradients * grad_output
