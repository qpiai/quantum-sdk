import plotly.graph_objects as go  # type: ignore[import-untyped]
import numpy as np
from typing import List, Optional
import math


class Plotter:
    """
    An abstraction which directly plots operations on a Plotly graph to generate
    circuit visualizations.
    """

    def __init__(self, maxDepth, numQubits, theme="light", scale=1):
        self.theme = theme
        self.maxDepth = maxDepth
        self.numQubits = numQubits
        self.scale = scale

        # Fixed size parameters
        self.min_gate_width = 0.8  # Minimum width for readability
        self.max_gates_per_row = 15  # Maximum number of gates per row

        # Fixed spacing parameters
        self.gate_width = 0.9 * self.scale  # Width of gate
        self.gate_padding = 0.3 * self.scale  # Padding between gates
        self.gate_spacing = self.gate_width + self.gate_padding  # Total space per gate
        self.row_spacing = 2.0  # Vertical space between rows

        # Fixed height parameters
        self.qubit_height = 1.0 * self.scale  # Height per qubit - fixed
        self.qubit_padding = 0.5 * self.scale  # Padding between qubits
        self.row_height = self.numQubits * self.qubit_height + self.qubit_padding * (
            self.numQubits - 1
        )

        # Margins for better layout
        self.left_margin = 1.5 * self.scale  # Space for qubit labels
        self.right_margin = 1.0 * self.scale  # Right padding

        # Style options (keeping the same structure for compatibility)
        self.lightStyleOptions = {
            "bg_color": "#ffffff",
            "text_color": "#1a1a1a",
            "font_size": 10,
            "font_family": "Times New Roman",
            "qubit_wires": {
                "primary": "#ababab",
                "label": "#1a1a1a",
            },
            "cbit_wires": {
                "primary": "#ababab",
                "label": "#1a1a1a",
            },
            "one_qubit_gate_color": {
                "primary": "#FF6500",
                "label": "#FF6500",
            },
            "two_qubit_gate_color": {
                "primary": "#1E3E62",
                "line": "#1E3E62",
                "label": "#1E3E62",
            },
            "three_qubit_gate_color": {
                "primary": "#0B192C",
                "line": "#0B192C",
                "label": "#0B192C",
            },
            "measurement_color": {
                "primary": "#1a1a1a",
                "line": "#1a1a1a",
                "label": "#1a1a1a",
            },
            "barrier_color": "#e0c4b0",  # Slightly darker color
        }

        self.darkStyleOptions = {
            "bg_color": "#1a1a1a",
            "text_color": "#ababab",
            "font_size": 10,
            "font_family": "Times New Roman",
            "qubit_wires": {
                "primary": "#cacaca",
                "label": "#cacaca",
            },
            "cbit_wires": {
                "primary": "#cacaca",
                "label": "#cacaca",
            },
            "one_qubit_gate_color": {
                "primary": "#FF6500",
                "label": "#FF6500",
            },
            "two_qubit_gate_color": {
                "primary": "#50739a",
                "line": "#50739a",
                "label": "#50739a",
            },
            "three_qubit_gate_color": {
                "primary": "#0B192C",
                "line": "#0B192C",
                "label": "#0B192C",
            },
            "measurement_color": {
                "primary": "#cacaca",
                "line": "#cacaca",
                "label": "#cacaca",
            },
            "barrier_color": "#e0c4b0",  # Slightly darker color
        }

        self.styleOptions = (
            self.lightStyleOptions if self.theme == "light" else self.darkStyleOptions
        )

        # Calculate dimensions
        self.base_width = (
            max(8, self.gate_spacing * 5) * self.scale
        )  # Min width for 5 gates
        self.base_height = max(6, self.row_height) * self.scale

        self._initialize_figure()

    def _initialize_figure(self):
        """Initialize the figure."""
        self.fig = go.Figure()

        # Set figure background and remove axes
        self.fig.update_layout(
            plot_bgcolor=self.styleOptions["bg_color"],
            paper_bgcolor=self.styleOptions["bg_color"],
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
                range=[0, self.maxDepth],
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
                scaleanchor="x",
                scaleratio=1,
                range=[-(self.numQubits + 1), 1],
            ),
        )

    def show(self):
        """Display the constructed circuit with proper configuration."""
        config = {
            "scrollZoom": True,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": [
                "zoom2d",
                "pan2d",
                "resetScale2d",
                "zoomIn2d",
                "zoomOut2d",
            ],
            # Disable autoscale which would resize everything to fit
            "modeBarButtonsToRemove": ["autoScale2d", "lasso2d", "select2d"],
        }
        self.fig.show(config=config)

    def _layout_circuit(self, gates):
        """
        Determine the layout of gates across multiple rows.
        Returns:
            - list of gates with their positions and rows
            - number of rows
            - width of each row
        """
        num_gates = len(gates)

        # Calculate optimal number of rows to maintain aspect ratio
        if num_gates <= self.max_gates_per_row:
            # Single row layout
            positions = []
            for i, gate in enumerate(gates):
                positions.append(
                    {
                        "gate": gate,
                        "x": self.left_margin
                        + i * self.gate_spacing
                        + self.gate_width / 2,
                        "row": 0,
                    }
                )
            return (
                positions,
                1,
                self.left_margin + num_gates * self.gate_spacing + self.right_margin,
            )
        else:
            # Multi-row layout
            positions = []
            rows = int(np.ceil(num_gates / self.max_gates_per_row))
            gates_per_row = int(np.ceil(num_gates / rows))

            for i, gate in enumerate(gates):
                row = i // gates_per_row
                col = i % gates_per_row
                positions.append(
                    {
                        "gate": gate,
                        "x": self.left_margin
                        + col * self.gate_spacing
                        + self.gate_width / 2,
                        "row": row,
                    }
                )

            row_width = (
                self.left_margin + gates_per_row * self.gate_spacing + self.right_margin
            )
            return positions, rows, row_width

    def draw_circuit(self, gates):
        """Draw the entire circuit, breaking into multiple rows if necessary."""
        # Determine layout
        gate_positions, num_rows, row_width = self._layout_circuit(gates)

        # Calculate required figure dimensions
        # Set a fixed height per row
        row_height_with_spacing = (
            self.row_height + (1 if num_rows > 1 else 0) * self.row_spacing
        )
        total_height = row_height_with_spacing * num_rows

        # Calculate fixed pixel size for each gate
        px_per_unit = 80  # Fixed number of pixels per unit of our coordinate system

        # Create a fresh figure with calculated dimensions - width will expand as needed
        self.fig = go.Figure()

        # Fixed height in pixels, width scales with content
        fig_height_px = max(
            600, (total_height + 2) * px_per_unit
        )  # Fixed minimum height
        fig_width_px = max(
            1000, (row_width + 1) * px_per_unit
        )  # Width scales with content

        # Padding for y-axis range to ensure we see all content
        y_padding = 1.0

        # Update figure layout with explicit size and no auto-scaling
        self.fig.update_layout(
            width=fig_width_px,  # Width in pixels based on content
            height=fig_height_px,  # Fixed height in pixels
            plot_bgcolor=self.styleOptions["bg_color"],
            paper_bgcolor=self.styleOptions["bg_color"],
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
                range=[0, row_width],
                constrain="domain",  # Maintain axis scaling
                scaleratio=1,  # Fixed scale ratio
                fixedrange=False,  # Allow x-axis zooming
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                showline=False,
                range=[-total_height - y_padding, y_padding],
                fixedrange=False,  # Allow y-axis zooming
                scaleanchor="x",  # Link scale to x-axis
                scaleratio=1,  # 1:1 ratio ensures circle gates look circular
            ),
            # Enable scrolling and zooming
            dragmode="pan",
            autosize=False,  # Disable autosize to prevent browser rescaling
            # Configure unified data ratio to maintain scale
            uniformtext=dict(minsize=10, mode="hide"),
        )

        # Draw each row of the circuit
        for row in range(num_rows):
            # Calculate y offset for this row
            y_offset = row * (self.row_height + self.row_spacing)

            # Draw qubit wires for this row
            self.draw_qubit_wires(
                self.left_margin,
                row_width - self.right_margin,
                self.numQubits,
                y_offset,
            )

            # Draw gates in this row
            row_gates = [pos for pos in gate_positions if pos["row"] == row]
            for gate_pos in row_gates:
                # Call the gate drawing method with the proper offset
                self._draw_gate(gate_pos["gate"], gate_pos["x"], y_offset)

        # Set config options for showing the figure
        # config = {
        #     "scrollZoom": True,
        #     "displayModeBar": True,
        #     "editable": False,
        #     "displaylogo": False,
        #     "modeBarButtonsToAdd": [
        #         "zoom2d",
        #         "pan2d",
        #         "resetScale2d",
        #         "zoomIn2d",
        #         "zoomOut2d",
        #     ],
        #     "modeBarButtonsToRemove": ["autoScale2d", "lasso2d", "select2d"],
        #     "toImageButtonOptions": {
        #         "format": "png",
        #         "filename": "quantum_circuit",
        #         "scale": 2,  # Higher resolution for export
        #     },
        # }

        # Add instructions for scrolling if the circuit is wide
        if row_width * px_per_unit > 1000:
            self.fig.add_annotation(
                x=0.5,
                y=-total_height - 0.5,
                text="Use mouse wheel to zoom or drag to pan",
                showarrow=False,
                font=dict(size=10),
                xref="paper",
                yref="y",
            )

        return self.fig

    def _draw_gate(self, gate, x, y_offset=0):
        """Draw a gate at the specified position with row offset."""
        # This method would dispatch to the appropriate gate drawing method
        # based on the gate type, with the y_offset added to the qubit positions
        pass

    def _draw_gates(self, gates, y_offset=0):
        """Draw the gates on the current figure with an optional row offset."""
        for i, gate in enumerate(gates):
            x = (i + 0.5) * self.gate_spacing + self.left_margin
            self._draw_gate(gate, x, y_offset)

    def set_theme(self, theme: str):
        """Set the theme for the circuit diagram."""
        self.theme = theme
        self.styleOptions = (
            self.lightStyleOptions if self.theme == "light" else self.darkStyleOptions
        )

    def draw_qubit_wires(self, x_start, x_end, num_qubits, y_offset=0):
        """Draw quantum circuit wires with an optional vertical offset."""
        for y in range(num_qubits):
            # Calculate qubit position using fixed spacing
            actual_y = y * (self.qubit_height + self.qubit_padding) + y_offset

            # Draw wire as a line
            self.fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[-actual_y, -actual_y],
                    mode="lines",
                    line=dict(
                        color=self.styleOptions["qubit_wires"]["primary"], width=1.5
                    ),
                    hoverinfo="skip",
                )
            )

            # Add qubit label
            self.fig.add_annotation(
                x=x_start - 0.5,
                y=-actual_y,
                text=f"q{y}",
                showarrow=False,
                font=dict(color=self.styleOptions["qubit_wires"]["label"], size=12),
                xanchor="right",
                yanchor="middle",
            )

    def draw_clbit_wires(self, x_start, x_end, y_pos, y_offset=0):
        """Draw classical register as a single double line with number of bits as superscript."""
        # Calculate position to be below the last qubit wire with proper spacing
        # Add extra spacing after the last qubit to avoid overlap
        clbit_extra_spacing = (
            1.0  # Extra spacing between last qubit and first classical bit
        )

        # Position classical bit wire below the last qubit wire
        # numQubits - 1 is the index of the last qubit
        last_qubit_y = (self.numQubits - 1) * (
            self.qubit_height + self.qubit_padding
        ) + y_offset
        y_bottom = -last_qubit_y - clbit_extra_spacing  # Position below the last qubit

        # Draw parallel lines for classical bits
        self.fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[y_bottom, y_bottom],
                mode="lines",
                line=dict(color=self.styleOptions["cbit_wires"]["primary"], width=1.5),
                hoverinfo="skip",
            )
        )

        self.fig.add_trace(
            go.Scatter(
                x=[x_start, x_end],
                y=[y_bottom + 0.1, y_bottom + 0.1],
                mode="lines",
                line=dict(color=self.styleOptions["cbit_wires"]["primary"], width=1.5),
                hoverinfo="skip",
            )
        )

        # Add classical register label with number of bits as superscript
        num_clbits = y_pos - 1  # Calculate number of classical bits
        self.fig.add_annotation(
            x=x_start - 0.5,
            y=y_bottom + 0.1,
            text=f"c<sup>{num_clbits}</sup>",
            showarrow=False,
            font=dict(color=self.styleOptions["cbit_wires"]["label"], size=12),
            xanchor="right",
            yanchor="middle",
        )

    def draw_one_qubit_non_parametric(self, x, y, label, y_offset=0):
        """Draw single qubit gate box with label."""
        actual_y = y * (self.qubit_height + self.qubit_padding) + y_offset

        # Add rectangle shape for the gate
        self.fig.add_shape(
            type="rect",
            x0=x - self.gate_width / 2,
            y0=-actual_y - 0.3,
            x1=x + self.gate_width / 2,
            y1=-actual_y + 0.3,
            line=dict(
                color=self.styleOptions["one_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor=self.styleOptions["bg_color"],
        )

        # Add label
        self.fig.add_annotation(
            x=x,
            y=-actual_y,
            text=label,
            showarrow=False,
            font=dict(
                color=self.styleOptions["one_qubit_gate_color"]["label"], size=16
            ),
            xanchor="center",
            yanchor="middle",
        )

    def draw_one_qubit_parametric(self, x, y, label, parameter, y_offset=0):
        """Draw parametric single qubit gate."""
        self.draw_one_qubit_non_parametric(
            x, y, f"{label}<sub>{parameter}</sub>", y_offset
        )

    def draw_two_qubit_non_parametric(self, x, control_y, target_y, label, y_offset=0):
        """Draw two qubit gate with control and target."""
        actual_control_y = (
            control_y * (self.qubit_height + self.qubit_padding) + y_offset
        )
        actual_target_y = target_y * (self.qubit_height + self.qubit_padding) + y_offset

        # Draw vertical line
        self.fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[-actual_control_y, -actual_target_y],
                mode="lines",
                line=dict(
                    color=self.styleOptions["two_qubit_gate_color"]["line"], width=2
                ),
                hoverinfo="skip",
            )
        )

        # Draw control point (filled circle)
        self.fig.add_shape(
            type="circle",
            x0=x - 0.08,
            y0=-actual_control_y - 0.08,
            x1=x + 0.08,
            y1=-actual_control_y + 0.08,
            line=dict(
                color=self.styleOptions["two_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor=self.styleOptions["two_qubit_gate_color"]["primary"],
        )

        # Draw target (empty circle)
        self.fig.add_shape(
            type="circle",
            x0=x - 0.25,
            y0=-actual_target_y - 0.25,
            x1=x + 0.25,
            y1=-actual_target_y + 0.25,
            line=dict(
                color=self.styleOptions["two_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor=self.styleOptions["bg_color"],
        )

        # Add label
        self.fig.add_annotation(
            x=x,
            y=-actual_target_y,
            text=label,
            showarrow=False,
            font=dict(
                color=self.styleOptions["two_qubit_gate_color"]["label"], size=16
            ),
            xanchor="center",
            yanchor="middle",
        )

    def draw_two_qubit_parametric(
        self, x, control_y, target_y, label, parameter, y_offset=0
    ):
        """Draw parametric two qubit gate."""
        actual_control_y = (
            control_y * (self.qubit_height + self.qubit_padding) + y_offset
        )
        actual_target_y = target_y * (self.qubit_height + self.qubit_padding) + y_offset

        # Draw vertical line
        self.fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[-actual_control_y, -actual_target_y],
                mode="lines",
                line=dict(
                    color=self.styleOptions["two_qubit_gate_color"]["line"], width=2
                ),
                hoverinfo="skip",
            )
        )

        # Draw control point (filled circle)
        self.fig.add_shape(
            type="circle",
            x0=x - 0.08,
            y0=-actual_control_y - 0.08,
            x1=x + 0.08,
            y1=-actual_control_y + 0.08,
            line=dict(
                color=self.styleOptions["two_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor=self.styleOptions["two_qubit_gate_color"]["primary"],
        )

        # Draw target as rectangle
        self.fig.add_shape(
            type="rect",
            x0=x - 0.45,
            y0=-actual_target_y - 0.35,
            x1=x + 0.45,
            y1=-actual_target_y + 0.35,
            line=dict(
                color=self.styleOptions["two_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor=self.styleOptions["bg_color"],
        )

        # Add label with parameter
        self.fig.add_annotation(
            x=x,
            y=-actual_target_y,
            text=f"{label}<sub>{parameter}</sub>",
            showarrow=False,
            font=dict(
                color=self.styleOptions["two_qubit_gate_color"]["label"], size=12
            ),
            xanchor="center",
            yanchor="middle",
        )

    def draw_three_qubit_non_parametric(
        self, x, control1_y, control2_y, target_y, label, y_offset=0
    ):
        """Draw three qubit gate with two controls and target."""
        actual_control1_y = (
            control1_y * (self.qubit_height + self.qubit_padding) + y_offset
        )
        actual_control2_y = (
            control2_y * (self.qubit_height + self.qubit_padding) + y_offset
        )
        actual_target_y = target_y * (self.qubit_height + self.qubit_padding) + y_offset

        # Calculate min and max y for the vertical line
        min_y = min(actual_control1_y, actual_control2_y, actual_target_y)
        max_y = max(actual_control1_y, actual_control2_y, actual_target_y)

        # Draw vertical line
        self.fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[-min_y, -max_y],
                mode="lines",
                line=dict(
                    color=self.styleOptions["three_qubit_gate_color"]["line"], width=2
                ),
                hoverinfo="skip",
            )
        )

        # Draw first control point
        self.fig.add_shape(
            type="circle",
            x0=x - 0.08,
            y0=-actual_control1_y - 0.08,
            x1=x + 0.08,
            y1=-actual_control1_y + 0.08,
            line=dict(
                color=self.styleOptions["three_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor=self.styleOptions["three_qubit_gate_color"]["primary"],
        )

        # Draw second control point
        self.fig.add_shape(
            type="circle",
            x0=x - 0.08,
            y0=-actual_control2_y - 0.08,
            x1=x + 0.08,
            y1=-actual_control2_y + 0.08,
            line=dict(
                color=self.styleOptions["three_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor=self.styleOptions["three_qubit_gate_color"]["primary"],
        )

        # Draw target (empty circle)
        self.fig.add_shape(
            type="circle",
            x0=x - 0.25,
            y0=-actual_target_y - 0.25,
            x1=x + 0.25,
            y1=-actual_target_y + 0.25,
            line=dict(
                color=self.styleOptions["three_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor=self.styleOptions["bg_color"],
        )

        # Add label
        self.fig.add_annotation(
            x=x,
            y=-actual_target_y,
            text=label,
            showarrow=False,
            font=dict(
                color=self.styleOptions["three_qubit_gate_color"]["label"], size=16
            ),
            xanchor="center",
            yanchor="middle",
        )

    def draw_n_qubit_gate(
        self,
        x: float,
        qubits: List[int],
        label: str,
        params: Optional[List[float]] = None,
        y_offset=0,
    ):
        """Draw a gate that operates on n qubits with optional parameters."""
        if len(qubits) == 1:
            # Single qubit gate
            if params:
                self.draw_one_qubit_parametric(
                    x, qubits[0], label, f"{params[0]:.2f}", y_offset
                )
            else:
                self.draw_one_qubit_non_parametric(x, qubits[0], label, y_offset)
        elif len(qubits) == 2:
            # Two qubit gate
            if params:
                self.draw_two_qubit_parametric(
                    x, qubits[0], qubits[1], label, f"{params[0]:.2f}", y_offset
                )
            else:
                self.draw_two_qubit_non_parametric(
                    x, qubits[0], qubits[1], label, y_offset
                )
        elif len(qubits) == 3:
            # Three qubit gate
            self.draw_three_qubit_non_parametric(
                x, qubits[0], qubits[1], qubits[2], label, y_offset
            )
        else:
            # Multi-qubit gate - draw as a box covering all qubits
            min_y = min(qubits)
            max_y = max(qubits)
            self.draw_operation(x, min_y, max_y + 1, label, y_offset)

    def draw_measure(self, x, qubit_y, clbit_y, target, basis="z", y_offset=0):
        """Draw measurement operation."""
        actual_qubit_y = qubit_y * (self.qubit_height + self.qubit_padding) + y_offset

        # Calculate the actual position of the classical bit wire using the same logic as in draw_clbit_wires
        clbit_extra_spacing = 1.0
        last_qubit_y = (self.numQubits - 1) * (
            self.qubit_height + self.qubit_padding
        ) + y_offset
        actual_clbit_y = last_qubit_y + clbit_extra_spacing

        # Draw measurement box
        self.fig.add_shape(
            type="rect",
            x0=x - 0.45,
            y0=-actual_qubit_y - 0.3,
            x1=x + 0.45,
            y1=-actual_qubit_y + 0.3,
            line=dict(
                color=self.styleOptions["measurement_color"]["primary"],
                width=2,
            ),
            fillcolor=self.styleOptions["bg_color"],
        )

        # Draw meter arc (semi-circle)
        # Plotly doesn't have direct arc shapes, so we'll use SVG path
        path = f"M {x - 0.25} {-actual_qubit_y - 0.1} A 0.25 0.25 0 0 1 {x + 0.25} {-actual_qubit_y - 0.1}"
        self.fig.add_shape(
            type="path",
            path=path,
            line=dict(
                color=self.styleOptions["measurement_color"]["primary"],
                width=2,
            ),
            fillcolor="#ffffff",  # Transparent fill
        )

        # Draw line inside semi-circle
        angle = 45  # degrees
        line_length = 0.3
        x_end = x + line_length * math.cos(math.radians(angle))
        y_end = -actual_qubit_y - 0.1 + line_length * math.sin(math.radians(angle))

        self.fig.add_trace(
            go.Scatter(
                x=[x, x_end],
                y=[-actual_qubit_y - 0.1, y_end],
                mode="lines",
                line=dict(
                    color=self.styleOptions["measurement_color"]["primary"], width=2
                ),
                hoverinfo="skip",
            )
        )

        # Draw vertical line to classical bit (dashed)
        self.fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[-actual_qubit_y, -actual_clbit_y],
                mode="lines",
                line=dict(
                    color=self.styleOptions["measurement_color"]["line"],
                    width=1.5,
                    dash="dash",
                ),
                hoverinfo="skip",
            )
        )

        # Add basis label
        self.fig.add_annotation(
            x=x - 0.3,
            y=-actual_qubit_y + 0.15,
            text=basis,
            showarrow=False,
            font=dict(color=self.styleOptions["measurement_color"]["label"], size=10),
            xanchor="center",
            yanchor="middle",
        )

        # Add target label
        self.fig.add_annotation(
            x=x + 0.2,
            y=-actual_clbit_y + 0.3,
            text=f"c{target}",
            showarrow=False,
            font=dict(color=self.styleOptions["measurement_color"]["label"], size=10),
            xanchor="left",
            yanchor="middle",
        )

    def draw_barrier(self, x, y, y_offset=0):
        """Draw barrier line for a single qubit."""
        actual_y = y * (self.qubit_height + self.qubit_padding) + y_offset

        # Draw vertical line for barrier
        self.fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[-actual_y - 0.5, -actual_y + 0.5],
                mode="lines",
                line=dict(color=self.styleOptions["barrier_color"], width=4),
                hoverinfo="skip",
            )
        )

    def draw_reset(self, x, y, y_offset=0):
        """Draw reset operation."""
        self.draw_one_qubit_non_parametric(x, y, "|0⟩", y_offset)

    def draw_swap(self, x, y1, y2, y_offset=0):
        """Draw swap operation."""
        actual_y1 = y1 * (self.qubit_height + self.qubit_padding) + y_offset
        actual_y2 = y2 * (self.qubit_height + self.qubit_padding) + y_offset

        # Draw vertical line
        self.fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[-actual_y1, -actual_y2],
                mode="lines",
                line=dict(
                    color=self.styleOptions["two_qubit_gate_color"]["line"], width=2
                ),
                hoverinfo="skip",
            )
        )

        # Draw first swap point
        self.fig.add_shape(
            type="circle",
            x0=x - 0.08,
            y0=-actual_y1 - 0.08,
            x1=x + 0.08,
            y1=-actual_y1 + 0.08,
            line=dict(
                color=self.styleOptions["two_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor=self.styleOptions["two_qubit_gate_color"]["primary"],
        )

        # Draw second swap point
        self.fig.add_shape(
            type="circle",
            x0=x - 0.08,
            y0=-actual_y2 - 0.08,
            x1=x + 0.08,
            y1=-actual_y2 + 0.08,
            line=dict(
                color=self.styleOptions["two_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor=self.styleOptions["two_qubit_gate_color"]["primary"],
        )

    def draw_controlled_swap(self, x, control_y, swap_y1, swap_y2, y_offset=0):
        """Draw a controlled swap (Fredkin) gate."""
        actual_control_y = (
            control_y * (self.qubit_height + self.qubit_padding) + y_offset
        )
        actual_swap_y1 = swap_y1 * (self.qubit_height + self.qubit_padding) + y_offset
        actual_swap_y2 = swap_y2 * (self.qubit_height + self.qubit_padding) + y_offset

        # Draw vertical line connecting all qubits
        min_y = min(actual_control_y, actual_swap_y1, actual_swap_y2)
        max_y = max(actual_control_y, actual_swap_y1, actual_swap_y2)

        self.fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[-min_y, -max_y],
                mode="lines",
                line=dict(
                    color=self.styleOptions["three_qubit_gate_color"]["line"], width=2
                ),
                hoverinfo="skip",
            )
        )

        # Draw control point
        self.fig.add_shape(
            type="circle",
            x0=x - 0.08,
            y0=-actual_control_y - 0.08,
            x1=x + 0.08,
            y1=-actual_control_y + 0.08,
            line=dict(
                color=self.styleOptions["three_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor=self.styleOptions["three_qubit_gate_color"]["primary"],
        )

        # Draw X symbols for the swap points
        self._draw_x_symbol(x, actual_swap_y1, 0.2)
        self._draw_x_symbol(x, actual_swap_y2, 0.2)

    def _draw_x_symbol(self, x, y, size):
        """Helper method to draw X symbol for swap gates."""
        # Draw diagonal lines to form X
        self.fig.add_trace(
            go.Scatter(
                x=[x - size / 2, x + size / 2],
                y=[-y - size / 2, -y + size / 2],
                mode="lines",
                line=dict(
                    color=self.styleOptions["two_qubit_gate_color"]["primary"], width=2
                ),
                hoverinfo="skip",
            )
        )

        self.fig.add_trace(
            go.Scatter(
                x=[x - size / 2, x + size / 2],
                y=[-y + size / 2, -y - size / 2],
                mode="lines",
                line=dict(
                    color=self.styleOptions["two_qubit_gate_color"]["primary"], width=2
                ),
                hoverinfo="skip",
            )
        )

    def draw_operation(self, x, y_start, y_end, label, y_offset=0):
        """Draw custom operation box."""
        actual_y_start = y_start * (self.qubit_height + self.qubit_padding) + y_offset
        actual_y_end = y_end * (self.qubit_height + self.qubit_padding) + y_offset

        # Draw the operation box
        self.fig.add_shape(
            type="rect",
            x0=x - 0.45,
            y0=-actual_y_end,
            x1=x + 0.45,
            y1=-actual_y_start,
            line=dict(
                color=self.styleOptions["one_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor="white",
        )

        # Add label
        self.fig.add_annotation(
            x=x,
            y=-(actual_y_start + actual_y_end) / 2,
            text=label,
            showarrow=False,
            font=dict(
                color=self.styleOptions["one_qubit_gate_color"]["label"], size=12
            ),
            xanchor="center",
            yanchor="middle",
        )

    def draw_multi_qubit_gate(self, x, qubits, label, y_offset=0):
        """Draw a box around multiple qubits for a multi-qubit gate."""
        # Calculate actual y positions for min and max qubits
        min_y_idx = min(qubits)
        max_y_idx = max(qubits)
        min_actual_y = min_y_idx * (self.qubit_height + self.qubit_padding) + y_offset
        max_actual_y = max_y_idx * (self.qubit_height + self.qubit_padding) + y_offset

        # Draw vertical line connecting all qubits
        self.fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[-min_actual_y, -max_actual_y],
                mode="lines",
                line=dict(
                    color=self.styleOptions["three_qubit_gate_color"]["line"], width=2
                ),
                hoverinfo="skip",
            )
        )

        # Calculate width based on label length
        min_width = 0.5
        char_width = 0.09
        width = max(min_width, (char_width * len(label)))
        half_width = width / 2

        # Create a box spanning all qubits
        self.fig.add_shape(
            type="rect",
            x0=x - half_width,
            y0=-max_actual_y - 0.3,
            x1=x + half_width,
            y1=-min_actual_y + 0.3,
            line=dict(
                color=self.styleOptions["three_qubit_gate_color"]["primary"],
                width=2,
            ),
            fillcolor="white",
        )

        # Add label in the center
        self.fig.add_annotation(
            x=x,
            y=-(min_actual_y + max_actual_y) / 2,
            text=label,
            showarrow=False,
            font=dict(
                color=self.styleOptions["three_qubit_gate_color"]["label"], size=12
            ),
            xanchor="center",
            yanchor="middle",
        )
