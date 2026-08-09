import torch
import numpy as np
import pytest
from torch.autograd import gradcheck

# Assuming the user has these available in their package
from qpiai_quantum.interfaces.torch_interface import ParameterShiftFunction as SequentialFunction
# We will import the batched one as well
try:
    from qpiai_quantum.interfaces.torch_integration import ParameterShiftFunction as BatchedFunction
except ImportError:
    # Fallback in case it's not saved properly yet
    BatchedFunction = None


def dummy_cost_function(params_np: np.ndarray) -> float:
    """
    A simple dummy cost function: f(x) = sum(sin(x_i)).
    The parameter-shift rule exactly computes the derivative of this function
    because sin(x + pi/2) - sin(x - pi/2) = 2 * cos(x).
    """
    return float(np.sum(np.sin(params_np)))


def test_sequential_gradients():
    """
    Verify that the sequential PyTorch interface computes gradients correctly.
    """
    # Create random parameters requiring gradients
    params = torch.rand(4, dtype=torch.float64, requires_grad=True)

    # 1. Forward pass
    expectation = SequentialFunction.apply(dummy_cost_function, params)
    
    # 2. Backward pass
    expectation.backward()
    
    # Analytical gradient for sum(sin(x)) is cos(x)
    expected_grad = torch.cos(params)
    
    # Check if gradients match the analytical ones
    assert params.grad is not None, "Gradient is None"
    assert torch.allclose(params.grad, expected_grad, atol=1e-5), \
        f"Sequential Gradients do not match! Expected: {expected_grad}, Got: {params.grad}"


@pytest.mark.skipif(BatchedFunction is None, reason="torch_integration not found")
def test_batched_gradients():
    """
    Verify that the batched (concurrent) PyTorch interface computes gradients correctly.
    """
    assert BatchedFunction is not None
    params = torch.rand(4, dtype=torch.float64, requires_grad=True)

    # 1. Forward pass using batched function
    expectation = BatchedFunction.apply(dummy_cost_function, params)
    
    # 2. Backward pass
    expectation.backward()
    
    # Analytical gradient
    expected_grad = torch.cos(params)
    
    # Check if gradients match the analytical ones
    assert params.grad is not None, "Gradient is None"
    assert torch.allclose(params.grad, expected_grad, atol=1e-5), \
        f"Batched Gradients do not match! Expected: {expected_grad}, Got: {params.grad}"


def test_gradcheck_sequential():
    """
    Use PyTorch's built-in gradcheck to verify the Jacobian numerically via finite differences.
    """
    # PyTorch gradcheck requires float64 to ensure numerical stability during finite difference checks
    params = torch.rand(3, dtype=torch.float64, requires_grad=True)
    
    # Wrapper for gradcheck
    def apply_fn(p):
        return SequentialFunction.apply(dummy_cost_function, p)
        
    test_passed = gradcheck(apply_fn, (params,), eps=1e-4, atol=1e-3)
    assert test_passed, "Gradcheck failed for SequentialFunction"


@pytest.mark.skipif(BatchedFunction is None, reason="torch_integration not found")
def test_gradcheck_batched():
    """
    Use PyTorch's built-in gradcheck for the batched interface.
    """
    assert BatchedFunction is not None
    params = torch.rand(3, dtype=torch.float64, requires_grad=True)
    
    def apply_fn(p):
        return BatchedFunction.apply(dummy_cost_function, p)
        
    test_passed = gradcheck(apply_fn, (params,), eps=1e-4, atol=1e-3)
    assert test_passed, "Gradcheck failed for BatchedFunction"


@pytest.mark.skipif(BatchedFunction is None, reason="torch_integration not found")
def test_quantum_layer():
    """
    Test the QuantumLayer with a real Circuit and Z-observable to ensure it mathematically 
    matches analytical gradients using the QpiAI Quantum Simulator.
    """
    from qpiai_quantum.circuit import Circuit
    # The QuantumLayer is defined in torch_integration.py
    from qpiai_quantum.interfaces.torch_integration import QuantumLayer

    # 1. Create a 1-qubit circuit with a single parametric RX gate
    circuit = Circuit(1)
    circuit.rx(0, 0.0) # Dummy parameter, will be replaced by bind_parameters
    
    # 2. Observable: Measure Pauli-Z on qubit 0
    observables = [(0, 'Z')]
    
    # 3. Initialize QuantumLayer
    layer = QuantumLayer(circuit, observables, num_params=1)
    
    # 4. Set the parameter to a known value for testing
    # Let's set it to pi/4
    with torch.no_grad():
        layer.q_params[0] = torch.pi / 4.0
    
    # 5. The analytical expectation for RX(theta) measuring Z is cos(theta).
    expectation = layer()
    expectation.backward()
    
    # Analytical gradient for cos(theta) is -sin(theta)
    expected_grad = -torch.sin(layer.q_params)
    
    assert layer.q_params.grad is not None, "Gradient is None"
    assert torch.allclose(layer.q_params.grad, expected_grad, atol=1e-5), \
        f"QuantumLayer Gradients do not match! Expected: {expected_grad}, Got: {layer.q_params.grad}"
