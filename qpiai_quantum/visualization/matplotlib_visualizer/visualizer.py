from qpiai_quantum.icr.icr import IntermediateCircuitRepresentation as ICR
from qpiai_quantum.circuit.circuit import Circuit
from .plotter import Plotter
from qpiai_quantum.icr.circuitoperation import OperationType
from typing import Any
import matplotlib.pyplot as plt


class MatplotlibVisualizer:
    """
    Creates a matplotlib visualization of the circuit or ICR
    """

    @staticmethod
    def _target_gate_label(gate_name: str) -> str:
        controlled_gate_labels = {
            "CX": "X",
            "CY": "Y",
            "CZ": "Z",
            "CP": "P",
            "CCX": "X",
            "MCX": "X",
            "id": "I",
            "ID": "I",
        }
        return controlled_gate_labels.get(gate_name, gate_name)

    @staticmethod
    def plot_icr(
        icr: Circuit | ICR,
        theme: str = "light",
        dpi: int = 200,
        use_mathtext: bool = True,
    ) -> None:
        """
        Controller for the plotting of the circuit or ICR. Directly interacts with the Matplotlib
        plotter to create the visualization.

        Args:
            icr (Circuit | ICR): The circuit or ICR to be visualized.
            theme (str): The theme to use for visualization. Default is "light".
            dpi (int): The DPI (dots per inch) of the visualization. Default is 200.
            use_mathtext (bool): When True, render labels with Matplotlib's built-in
                Mathtext engine. Default is True.
        """

        if not isinstance(icr, Circuit) and not isinstance(icr, ICR):
            raise TypeError(
                f"Cannot visualize object of type {type(icr)}, expected a Circuit or ICR object."
            )

        if isinstance(icr, Circuit):
            icr = icr.icr

        # Track the layer index for each qubit to form strict vertical columns
        layer_per_qubit = [0] * (icr.num_qubits + icr.num_clbits)
        layers: list[Any] = []

        for op in icr.evolve:
            op_dict = op.to_json()
            qubits = op_dict.get("qubits") or []

            if op_dict["operation_type"] == OperationType.MEASURE.value:
                affected = list(range(qubits[0], icr.num_qubits)) if qubits else []
            elif op_dict["operation_type"] == OperationType.BARRIER.value:
                affected = qubits
            elif op_dict["operation_type"] in (
                OperationType.N_QUBIT_NON_PARAMETRIC.value,
                OperationType.N_QUBIT_PARAMETRIC.value,
                OperationType.OPERATION.value,
                OperationType.SWAP.value,
            ):
                affected = list(range(min(qubits), max(qubits) + 1)) if qubits else []
            else:
                affected = []

            if affected:
                layer_idx = max(layer_per_qubit[q] for q in affected)
            else:
                layer_idx = max(layer_per_qubit) if layer_per_qubit else 0

            while len(layers) <= layer_idx:
                layers.append([])

            layers[layer_idx].append(op_dict)

            for q in affected:
                layer_per_qubit[q] = layer_idx + 1

        operations_with_positions = []
        current_x = 0.5

        for layer in layers:
            max_step = 0.0

            for op in layer:
                step = Plotter.STANDARD_COLUMN_WIDTH
                if op["operation_type"] == OperationType.N_QUBIT_NON_PARAMETRIC.value:
                    gate_label = MatplotlibVisualizer._target_gate_label(
                        op["gate_name"]
                    )
                    if (
                        len(op.get("qubits", [])) == 1
                        and gate_label.upper()
                        in Plotter.SQUARE_SAFE_SINGLE_QUBIT_LABELS
                    ):
                        step = Plotter.COMPACT_COLUMN_WIDTH
                    else:
                        calc_step = Plotter.operation_column_width_for_label(
                            gate_label, use_mathtext=use_mathtext
                        )
                        step = max(step, calc_step)
                elif op["operation_type"] == OperationType.N_QUBIT_PARAMETRIC.value:
                    gate_label = MatplotlibVisualizer._target_gate_label(
                        op["gate_name"]
                    )
                    params = op.get("params")
                    if params:
                        param_str = f"{params[0]:.2f}"
                        label_for_width = f"{gate_label}_{{{param_str}}}"
                        calc_step = Plotter.operation_column_width_for_label(
                            label_for_width, use_mathtext=use_mathtext
                        )
                        step = max(step, calc_step)
                    else:
                        calc_step = Plotter.operation_column_width_for_label(
                            gate_label, use_mathtext=use_mathtext
                        )
                        step = max(step, calc_step)
                elif op["operation_type"] == OperationType.OPERATION.value:
                    calc_step = Plotter.operation_column_width_for_label(
                        op["gate_name"], use_mathtext=use_mathtext
                    )
                    step = max(step, calc_step)

                max_step = max(max_step, step)

            x_pos = current_x + max_step / 2
            for op in layer:
                operations_with_positions.append((op, x_pos))

            current_x += max_step

        max_depth = current_x

        break_point = 16

        # Sort operations by x_pos to ensure they are drawn in order
        operations_with_positions.sort(key=lambda item: item[1])

        # Break the circuit into multiple parts if length is longer than break_point
        max_x_pos = operations_with_positions[-1][1] if operations_with_positions else 0
        num_chunks = (
            int(max_x_pos // break_point) + 1 if operations_with_positions else 1
        )
        broken_circuits: list[Any] = [[] for _ in range(num_chunks)]
        for op_pos in operations_with_positions:
            op, x_pos = op_pos
            chunk_idx = int(x_pos // break_point)
            broken_circuits[chunk_idx].append(op_pos)

        # Create plotter with calculated dimensions
        x_offset = 0
        for circs in broken_circuits:
            operations_with_positions = circs

            current_width = min(break_point, max_depth)

            plotter = Plotter(
                current_width,
                icr.num_qubits,
                break_point,
                len(broken_circuits),
                use_mathtext=use_mathtext,
            )
            plotter.set_theme(theme)

            # Draw wires
            plotter.draw_qubit_wires(0, current_width + 1, icr.num_qubits)
            plotter.draw_clbit_wires(0, current_width + 1, icr.num_qubits + 1)

            # Draw gates using the enhanced plotter methods
            for op, x_pos in operations_with_positions:
                if op["operation_type"] == OperationType.N_QUBIT_NON_PARAMETRIC.value:
                    qubits = op["qubits"]

                    gate_name = MatplotlibVisualizer._target_gate_label(op["gate_name"])

                    # Use the unified n_qubit_gate method
                    plotter.draw_n_qubit_gate(x_pos - x_offset, qubits, gate_name)

                elif op["operation_type"] == OperationType.N_QUBIT_PARAMETRIC.value:
                    qubits = op["qubits"]
                    params = op["params"]

                    gate_name = MatplotlibVisualizer._target_gate_label(op["gate_name"])

                    # Use the unified n_qubit_gate method with parameters
                    plotter.draw_n_qubit_gate(
                        x_pos - x_offset, qubits, gate_name, params
                    )

                elif op["operation_type"] == OperationType.SWAP.value:
                    qubits = op["qubits"]
                    if len(qubits) == 2:
                        plotter.draw_swap(x_pos - x_offset, qubits[0], qubits[1])
                    elif len(qubits) == 3:
                        # Use the specialized controlled-swap method
                        plotter.draw_controlled_swap(
                            x_pos - x_offset, qubits[0], qubits[1], qubits[2]
                        )
                    else:
                        # Multi-qubit swap (if ever implemented)
                        plotter.draw_multi_qubit_gate(
                            x_pos - x_offset, qubits, op["gate_name"]
                        )

                elif op["operation_type"] == OperationType.MEASURE.value:
                    qubit = op["qubits"][0]
                    clbit = op["clbits"][0] if "clbits" in op and op["clbits"] else 0
                    plotter.draw_measure(x_pos - x_offset, qubit, icr.num_qubits, clbit)

                elif op["operation_type"] == OperationType.BARRIER.value:
                    barrier_qubits = op["qubits"]
                    for q in barrier_qubits:
                        plotter.draw_barrier(x_pos - x_offset, q)

                elif op["operation_type"] == OperationType.OPERATION.value:
                    # Use the multi_qubit_gate method for custom operations
                    plotter.draw_multi_qubit_gate(
                        x_pos - x_offset, op["qubits"], op["gate_name"]
                    )

            # Render at user-specified DPI as PNG but display at original size for sharpness
            try:
                import io
                from IPython.display import display, Image
                from PIL import Image as PILImage

                buf = io.BytesIO()
                plotter.fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
                buf.seek(0)
                actual_width = PILImage.open(buf).width
                buf.seek(0)
                # Compute display width to maintain original figsize visual size
                display_width = int(actual_width * 100 / dpi)
                display(Image(data=buf.getvalue(), width=display_width))
                plt.close(plotter.fig)
            except ImportError:
                plt.show()
                plt.close()

            x_offset += break_point
