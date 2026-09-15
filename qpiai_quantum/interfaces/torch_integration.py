import torch
import torch.nn as nn
import numpy as np
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.icr.circuitoperation import CircuitOperation, OperationType
from typing import Any
from collections.abc import Callable
import concurrent.futures


class ParameterShiftFunction(torch.autograd.Function):
    """
    PyTorch autograd function that implements the parameter-shift rule
    for quantum circuits in the QpiAI Quantum SDK.
    """

    @staticmethod
    def forward(
        ctx: Any, cost_function: Callable[[np.ndarray], float], parameters: torch.Tensor
    ) -> torch.Tensor:
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
            tuple[None, torch.Tensor]: Tuple containing None for the non-tensor
            argument (cost_function) and the computed gradients for the parameters.
        """
        grad_output = grad_outputs[0]
        (parameters,) = ctx.saved_tensors
        cost_function = ctx.cost_function

        params_np = parameters.detach().numpy()
        gradients = torch.zeros_like(parameters)

        shift = torch.pi / 2.0

        # Note: The SDK currently simulates individual circuits sequentially.
        # To improve this, one could use Python's `concurrent.futures` to evaluate
        # the forward and backward shifted circuits in parallel across multiple CPU cores.
        def evaluate_shift(index, shift_type):
            params_shifted = params_np.copy()
            if shift_type == "forward":
                params_shifted[index] += shift
            else:
                params_shifted[index] -= shift
            return index, shift_type, cost_function(params_shifted)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(len(params_np)):
                futures.append(executor.submit(evaluate_shift, i, "forward"))
                futures.append(executor.submit(evaluate_shift, i, "backward"))

            results = {"forward": {}, "backward": {}}
            for future in concurrent.futures.as_completed(futures):
                idx, s_type, exp_val = future.result()
                results[s_type][idx] = exp_val
        for i in range(len(params_np)):
            gradients[i] = 0.5 * (results["forward"][i] - results["backward"][i])

        # returning None for non-tensor arguments (cost_function)
        return None, gradients * grad_output


class QuantumLayer(nn.Module):
    """
    A PyTorch neural network module that wraps a parameterized QpiAI Quantum Circuit.

    This abstracts away the parameter binding and expectation computation, allowing
    users to easily drop a quantum circuit into standard PyTorch models by passing
    the circuit and observables directly.
    """

    def __init__(
        self,
        circuit: Circuit,
        observables: list,
        num_params: int,
        device_name: str = "QpiAI-QSV-Local",
        shots: int = 1024,
        **run_kwargs,
    ):
        """
        Args:
            circuit (Circuit): The parameterized quantum circuit template.
            observables (list): A list of tuples specifying observables, e.g., [(qubit_idx, 'Z')].
            num_params (int): The number of trainable parameters in the circuit.
            device_name (str): The execution device (e.g., 'QpiAI-QSV-Local' or 'QpiAI-Indus-1').
            shots (int): The number of measurement shots.
        """
        super().__init__()
        self.circuit = circuit
        self.observables = observables
        self.device_name = device_name
        self.shots = shots
        self.q_params = nn.Parameter(torch.rand(num_params))

    def _bind_parameters(self, params_np: np.ndarray) -> Circuit:
        """Bind numpy parameters to the circuit by replacing parametric gates."""
        bound_circuit = Circuit(self.circuit.num_qubits)
        param_idx = 0

        for op in self.circuit.icr.evolve:
            if op.operation_type == OperationType.N_QUBIT_PARAMETRIC:
                if param_idx < len(params_np):
                    new_op = CircuitOperation(
                        operation_type=op.operation_type,
                        gate_name=op.gate_name,
                        qubits=op.qubits,
                        params=[float(params_np[param_idx])],
                        clbits=op.clbits,
                    )
                    bound_circuit.add_operation(new_op)
                    param_idx += 1
                else:
                    bound_circuit.add_operation(op)
            else:
                bound_circuit.add_operation(op)
        return bound_circuit

    def forward(self, inputs: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass for the quantum layer.
        """

        def cost_fn(params_np: np.ndarray) -> float:
            bound_circuit = self._bind_parameters(params_np)

            # If local simulator, we can optionally use exact statevectors.
            # Otherwise (e.g. real hardware), we must use measurement counts.
            need_sv = self.device_name == "QpiAI-QSV-Local"

            # Note: For real hardware, `bound_circuit` must have measurement gates.
            if not need_sv:
                bound_circuit.measure_all()

            result = bound_circuit.run(
                shots=self.shots, need_statevector=need_sv, device_name=self.device_name
            )
            expectation = 0.0

            if (
                need_sv
                and hasattr(result, "statevector")
                and result.statevector is not None
            ):
                # EXACT EXPECTATION FROM STATEVECTOR
                state = np.array(result.statevector, dtype=complex)
                n_qubits = bound_circuit.num_qubits

                for qubit_idx, op_name in self.observables:
                    if op_name == "Z":
                        evolved_state = state.copy()
                        shape = (2 ** (n_qubits - 1 - qubit_idx), 2, 2**qubit_idx)
                        state_tensor = evolved_state.reshape(shape)
                        state_tensor[:, 1, :] *= -1
                        evolved_state = state_tensor.reshape(-1)
                        expectation += np.vdot(state, evolved_state).real
            else:
                # COUNTS-BASED EXPECTATION (FOR HARDWARE OR NOISE MODELS)
                counts = result.counts
                total_shots = sum(counts.values()) if counts else 0
                if total_shots > 0:
                    for qubit_idx, op_name in self.observables:
                        term_exp = 0.0
                        assert counts is not None, (
                            "Counts should not be None if total_shots > 0"
                        )
                        for bitstr, count in counts.items():
                            eigenvalue = 1.0
                            # Left-pad bitstr if needed
                            if len(bitstr) < bound_circuit.num_qubits:
                                bitstr = bitstr.zfill(bound_circuit.num_qubits)

                            if op_name == "Z":
                                bit = int(bitstr[-(qubit_idx + 1)])
                                eigenvalue *= 1.0 if bit == 0 else -1.0

                            term_exp += eigenvalue * count
                        expectation += term_exp / total_shots

            return float(expectation)

        return ParameterShiftFunction.apply(cost_fn, self.q_params)
