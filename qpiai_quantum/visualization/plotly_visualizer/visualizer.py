from qpiai_quantum.icr.icr import IntermediateCircuitRepresentation as ICR
from qpiai_quantum.circuit.circuit import Circuit
from .plotter import Plotter
from qpiai_quantum.icr.circuitoperation import OperationType


class PlotlyVisualizer:
    """
    Creates a plotly visualization of the circuit or ICR
    """

    @staticmethod
    def plot_icr(icr: Circuit | ICR, theme: str = "dark") -> None:
        """
        Controller for the plotting of the circuit or ICR. Directly interacts with the Plotly
        plotter to create the visualization.

        Args:
            icr (Circuit | ICR): The circuit or ICR to be visualized.
            theme (str): The theme to use for visualization. Default is "light".
        """
        if not isinstance(icr, Circuit) and not isinstance(icr, ICR):
            raise TypeError(
                f"Cannot visualize object of type {type(icr)}, expected a Circuit or ICR object."
            )

        if isinstance(icr, Circuit):
            icr = icr.icr

        # Track the depth for each qubit
        depth_per_qubit = [1.0] * (icr.num_qubits + icr.num_clbits)
        max_depth = 1.0

        # First pass to calculate max_depth and positions
        operations_with_positions = []

        for op in icr.evolve:
            # Convert operation to JSON dict
            op = op.to_json()
            x_pos = 1.0

            if (
                op["operation_type"] == OperationType.N_QUBIT_NON_PARAMETRIC.value
                or op["operation_type"] == OperationType.N_QUBIT_PARAMETRIC.value
            ):
                qubits = op["qubits"]
                # Calculate the position as the maximum depth of all affected qubits
                if len(qubits) > 0:
                    x_pos = max(
                        depth_per_qubit[q] for q in range(min(qubits), max(qubits) + 1)
                    )
                    # Update the depth for all affected qubits
                    for q in range(min(qubits), max(qubits) + 1):
                        depth_per_qubit[q] = x_pos + 1

            elif op["operation_type"] == OperationType.MEASURE.value:
                qubit = op["qubits"][0]  # Assuming first qubit in the list is measured
                x_pos = max(depth_per_qubit[qubit:])
                for q in range(qubit, icr.num_qubits):
                    depth_per_qubit[q] = x_pos + 1

            elif op["operation_type"] == OperationType.SWAP.value:
                qubits = op["qubits"]
                x_pos = max(
                    depth_per_qubit[q] for q in range(min(qubits), max(qubits) + 1)
                )
                for q in range(min(qubits), max(qubits) + 1):
                    depth_per_qubit[q] = x_pos + 1

            elif op["operation_type"] == OperationType.BARRIER.value:
                barrier_qubits = op["qubits"]
                x_pos = max(depth_per_qubit[q] for q in barrier_qubits)
                for q in barrier_qubits:
                    depth_per_qubit[q] = x_pos + 1

            elif op["operation_type"] == OperationType.OPERATION.value:
                qubits = op["qubits"]
                width = max(1, (0.09 * len(op["gate_name"])))
                x_pos = max(
                    depth_per_qubit[q] for q in range(min(qubits), max(qubits) + 1)
                )
                for q in range(min(qubits), max(qubits) + 1):
                    depth_per_qubit[q] = x_pos + (width + 1 / 2)

            max_depth = max(max_depth, x_pos + 1)
            operations_with_positions.append((op, x_pos))

        # Create plotter with calculated dimensions
        plotter = Plotter(max_depth, icr.num_qubits, theme)

        # Draw wires
        plotter.draw_qubit_wires(0, max_depth + 1, icr.num_qubits)
        plotter.draw_clbit_wires(0, max_depth + 1, icr.num_qubits + 1)

        # Draw gates using the enhanced plotter methods
        for op, x_pos in operations_with_positions:
            if op["operation_type"] == OperationType.N_QUBIT_NON_PARAMETRIC.value:
                qubits = op["qubits"]

                if op["gate_name"].upper() in ["ID", "I"]:
                    gate_name = "I"
                else:
                    gate_name = op["gate_name"].replace("C", "")

                # Use the unified n_qubit_gate method
                plotter.draw_n_qubit_gate(x_pos, qubits, gate_name)

            elif op["operation_type"] == OperationType.N_QUBIT_PARAMETRIC.value:
                qubits = op["qubits"]
                params = op["params"]

                gate_name = op["gate_name"].replace("C", "")

                # Use the unified n_qubit_gate method with parameters
                plotter.draw_n_qubit_gate(x_pos, qubits, gate_name, params)

            elif op["operation_type"] == OperationType.SWAP.value:
                qubits = op["qubits"]
                if len(qubits) == 2:
                    plotter.draw_swap(x_pos, qubits[0], qubits[1])
                elif len(qubits) == 3:
                    # Use the specialized controlled-swap method
                    plotter.draw_controlled_swap(x_pos, qubits[0], qubits[1], qubits[2])
                else:
                    # Multi-qubit swap (if ever implemented)
                    plotter.draw_multi_qubit_gate(x_pos, qubits, op["gate_name"])

            elif op["operation_type"] == OperationType.MEASURE.value:
                qubit = op["qubits"][0]
                clbit = op["clbits"][0] if "clbits" in op and op["clbits"] else 0
                plotter.draw_measure(x_pos, qubit, icr.num_qubits, clbit)

            elif op["operation_type"] == OperationType.BARRIER.value:
                barrier_qubits = op["qubits"]
                for q in barrier_qubits:
                    plotter.draw_barrier(x_pos, q)

            elif op["operation_type"] == OperationType.OPERATION.value:
                # Use the multi_qubit_gate method for custom operations
                plotter.draw_multi_qubit_gate(x_pos, op["qubits"], op["gate_name"])

        plotter.show()
