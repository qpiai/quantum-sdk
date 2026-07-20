import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle, Circle, Arc
import numpy as np
from typing import Optional
import re


class Plotter:
    """
    An abstraction to which directly plots operations on a matplotlib graph to generate
    circuit visualizations.
    """

    SQUARE_SAFE_SINGLE_QUBIT_LABELS = {
        "H",
        "I",
        "S",
        "SDG",
        "SX",
        "T",
        "TDG",
        "X",
        "Y",
        "Z",
    }
    COMPACT_COLUMN_WIDTH = 0.8
    STANDARD_COLUMN_WIDTH = 1.1
    CUSTOM_OPERATION_PADDING = 0.2

    @staticmethod
    def _extract_math_group(text: str, start: int) -> tuple[str, int]:
        """Extract a single Mathtext token or grouped expression starting at start."""
        if start >= len(text):
            return "", start

        if text[start] != "{":
            return text[start], start + 1

        depth = 0
        index = start
        while index < len(text):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start + 1 : index], index + 1
            index += 1

        return text[start + 1 :], len(text)

    @staticmethod
    def _plain_text_units(text: str, font_scale: float = 1.0) -> float:
        """Estimate visible width for a plain text fragment with length penalty."""
        units = 0.0
        char_count = 0
        index = 0
        while index < len(text):
            char = text[index]
            is_space = False

            if char == "\\" and index + 1 < len(text):
                escaped = text[index + 1]
                is_space = escaped.isspace()
                index += 2
            elif char not in "{}":
                is_space = char.isspace()
                index += 1
            else:
                index += 1
                continue

            if is_space:
                units += 0.3 * font_scale
            else:
                char_width = 0.5 if char_count < 4 else 0.8
                units += char_width * font_scale
                char_count += 1

        return units

    @staticmethod
    def _label_units_for_width(label: str, use_mathtext: bool = True) -> float:
        """Estimate visible label width, compacting Mathtext control syntax."""
        normalized = label.strip()
        if not normalized:
            return 1.0

        if not use_mathtext:
            return Plotter._plain_text_units(normalized)

        units = 0.0
        char_count = 0
        index = 0
        while index < len(normalized):
            char = normalized[index]

            if char == "\\":
                command_end = index + 1
                while (
                    command_end < len(normalized) and normalized[command_end].isalpha()
                ):
                    command_end += 1

                if command_end > index + 1:
                    command = normalized[index + 1 : command_end]
                    if command == "mathrm" and command_end < len(normalized):
                        token, index = Plotter._extract_math_group(
                            normalized, command_end
                        )
                        units += Plotter._plain_text_units(token)
                        continue
                    if command == "sqrt" and command_end < len(normalized):
                        token, index = Plotter._extract_math_group(
                            normalized, command_end
                        )
                        units += 1.2 + 0.35 * Plotter._plain_text_units(token)
                        continue
                    char_width = 0.5 if char_count < 4 else 0.8
                    units += char_width
                    char_count += 1
                    index = command_end
                    continue

                char_width = 0.5 if char_count < 4 else 0.8
                units += char_width
                char_count += 1
                index = min(len(normalized), index + 2)
                continue

            if char in "_^":
                token, index = Plotter._extract_math_group(normalized, index + 1)
                units += max(0.25, 0.25 * Plotter._plain_text_units(token))
                continue

            if char not in "{}":
                if char.isspace():
                    units += 0.3
                else:
                    char_width = 0.5 if char_count < 4 else 0.2
                    units += char_width
                    char_count += 1
            index += 1

        return max(1.0, units)

    @staticmethod
    def operation_width_for_label(label: str, use_mathtext: bool = True) -> float:
        """Return the width used to lay out and draw a custom operation box."""
        units = Plotter._label_units_for_width(label, use_mathtext=use_mathtext)
        return max(0.65, units * 0.65)

    @staticmethod
    def operation_column_width_for_label(
        label: str, use_mathtext: bool = True
    ) -> float:
        """Return the horizontal span a custom operation occupies in the layout."""
        return (
            Plotter.operation_width_for_label(label, use_mathtext=use_mathtext)
            + Plotter.CUSTOM_OPERATION_PADDING
        )

    def __init__(
        self,
        max_depth,
        num_qubits,
        break_point,
        num_broken_circuit,
        scale=1,
        use_mathtext: bool = True,
    ):
        self.theme = "light"
        self.max_depth = max_depth
        self.num_qubits = num_qubits
        self.scale = scale
        self.break_point = break_point
        self.num_broken_circuit = num_broken_circuit
        self.min_gate_width = 0.8  # Minimum width for each gate
        self.single_gate_height = 0.6
        self.square_gate_size = self.single_gate_height
        self.square_safe_single_qubit_labels = self.SQUARE_SAFE_SINGLE_QUBIT_LABELS
        self.use_mathtext = use_mathtext
        self.gate_linewidth = 1.2
        self.control_linewidth = 1.2
        self.connector_linewidth = 1.2
        self.gate_font_size = 13
        self.gate_text_stroke_width = 0.05

        self.lightStyleOptions = {
            "bg_color": "#ffffff",
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
                "primary": "#CB2E92",
                "label": "#CB2E92",
            },
            "two_qubit_gate_color": {
                "primary": "#245EAF",
                "line": "#245EAF",
                "label": "#245EAF",
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
            "bg_color": "#ffffff",
            "font_size": 10,
            "font_family": "Times New Roman",
            "qubit_wires": {
                "primary": "#ababab",
                "label": "#ababab",
            },
            "cbit_wires": {
                "primary": "#ababab",
                "label": "#ababab",
            },
            "one_qubit_gate_color": {
                "primary": "#FF6500",
                "label": "#FF6500",
            },
            "two_qubit_gate_color": {
                "primary": "#245EAF",
                "line": "#245EAF",
                "label": "#245EAF",
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

        self.styleOptions = (
            self.lightStyleOptions if self.theme == "light" else self.darkStyleOptions
        )

        # self.special_maps = {
        #     "iSWAP":
        # }

        self.fig_width = self.break_point + 1 * self.scale
        self.fig_height = self.num_qubits * self.num_broken_circuit * self.scale

        self._initialize_figure()

    def _gate_text_options(self, color: str) -> dict:
        """Return shared text options for symbols drawn inside gates."""
        return {
            "fontsize": self.gate_font_size,
            "fontweight": "normal",
            "path_effects": [
                path_effects.Stroke(
                    linewidth=self.gate_text_stroke_width,
                    foreground=color,
                ),
                path_effects.Normal(),
            ],
        }

    def _initialize_figure(self):
        """Initialize the figure and axes."""
        self.fig, self.ax = plt.subplots(figsize=(self.fig_width, self.fig_height))

        self.ax.set_facecolor(self.styleOptions["bg_color"])
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        self.ax.set_aspect("equal", adjustable="box")

    def set_theme(self, theme: str):
        self.theme = theme
        self.styleOptions = (
            self.lightStyleOptions if self.theme == "light" else self.darkStyleOptions
        )

    def _mathtext_expression(self, label: str) -> str:
        normalized = label.strip()

        if normalized.startswith("$") and normalized.endswith("$"):
            return normalized[1:-1]

        replacements = {
            "Sdg": r"\mathrm{S}^{\dagger}",
            "Tdg": r"\mathrm{T}^{\dagger}",
            "SX": r"\sqrt{\mathrm{X}}",
        }
        if normalized in replacements:
            return replacements[normalized]

        register_match = re.fullmatch(r"([qc])(\d+)", normalized)
        if register_match:
            register, index = register_match.groups()
            return rf"\mathrm{{{register}}}_{{{index}}}"

        if any(token in normalized for token in ("\\", "_", "^")):
            normalized = normalized.replace(r"\theta", "θ")
            normalized = normalized.replace(r"\phi", "φ")
            normalized = normalized.replace(r"\lambda", "λ")
            normalized = normalized.replace(r"\gamma", "γ")
            normalized = normalized.replace(r"\alpha", "α")
            normalized = normalized.replace(r"\beta", "β")
            normalized = normalized.replace(r"\pi", "π")
            return normalized

        safe_label = normalized.replace("{", r"\{").replace("}", r"\}")
        safe_label = safe_label.replace(" ", r"\ ")
        return rf"\mathrm{{{safe_label}}}"

    def _plain_unicode_label(self, label: str) -> str | None:
        """Return an upright plain-text label for simple Greek Mathtext labels."""
        normalized = label.strip()
        greek_replacements = {
            r"\alpha": "α",
            r"\beta": "β",
            r"\gamma": "γ",
            r"\theta": "θ",
            r"\phi": "φ",
            r"\lambda": "λ",
            r"\pi": "π",
        }
        for command, glyph in greek_replacements.items():
            if normalized == command:
                return glyph

            subscript_match = re.fullmatch(
                rf"{re.escape(command)}_\{{?([0-9+\-=()]+)\}}?",
                normalized,
            )
            if subscript_match:
                return rf"{glyph}$_{{{subscript_match.group(1)}}}$"

        return None

    def _format_label(self, label: str) -> str:
        if not self.use_mathtext or not label:
            return label
        plain_label = self._plain_unicode_label(label)
        if plain_label is not None:
            return plain_label
        return f"${self._mathtext_expression(label)}$"

    def _format_parametric_label(self, label: str, parameter: str) -> str:
        if not self.use_mathtext:
            return f"${label}_{{{parameter}}}$"
        return f"${self._mathtext_expression(label)}_{{{parameter}}}$"

    def operation_width(self, label: str) -> float:
        """Return the width used to lay out and draw a custom operation box."""
        return self.operation_width_for_label(label, use_mathtext=self.use_mathtext)

    def operation_column_width(self, label: str) -> float:
        """Return the horizontal span a custom operation occupies in the layout."""
        return self.operation_column_width_for_label(
            label, use_mathtext=self.use_mathtext
        )

    def draw_qubit_wires(self, x_start, x_end, num_qubits):
        """Draw quantum circuit wires."""
        for y in range(num_qubits):
            self.ax.plot(
                [x_start, (x_end)],
                [-y, -y],
                color=self.styleOptions["qubit_wires"]["primary"],
                lw=1.5,
                zorder=1,
            )
            self.ax.text(
                x_start - 0.5,
                -y,
                self._format_label(f"q{y}"),
                ha="right",
                va="center",
                color=self.styleOptions["qubit_wires"]["label"],
                zorder=1,
            )

    def draw_clbit_wires(self, x_start, x_end, y_pos):
        """Draw classical register as a single double line with number of bits as superscript."""
        y_bottom = -y_pos + 1  # Position at the bottom

        # Draw single set of parallel lines for all classical bits
        self.ax.plot(
            [x_start, x_end],
            [y_bottom, y_bottom],
            color=self.styleOptions["cbit_wires"]["primary"],
            lw=1.5,
            zorder=1,
        )
        self.ax.plot(
            [x_start, x_end],
            [y_bottom + 0.1, y_bottom + 0.1],
            color=self.styleOptions["cbit_wires"]["primary"],
            lw=1.5,
            zorder=1,
        )

        # Add classical register label with number of bits as superscript
        num_clbits = y_pos - 1  # Calculate number of classical bits
        clbit_label = f"c$^{num_clbits}$"
        if self.use_mathtext:
            clbit_label = rf"$\mathrm{{c}}^{{{num_clbits}}}$"
        self.ax.text(
            x_start - 0.5,
            y_bottom + 0.1,
            clbit_label,
            ha="right",
            va="center",
            color=self.styleOptions["cbit_wires"]["label"],
            zorder=1,
        )

    def draw_one_qubit_non_parametric(self, x, y, label):
        """Draw single qubit gate box with label."""
        if label.upper() in self.square_safe_single_qubit_labels:
            x_start = x - self.square_gate_size / 2
            width = self.square_gate_size
            height = self.square_gate_size
        else:
            width = self.operation_width(label)
            width = max(self.min_gate_width, width)
            x_start = x - width / 2
            height = self.single_gate_height

        rect = Rectangle(
            (x_start, -y - height / 2),
            width,
            height,
            fc="white",
            ec=self.styleOptions["one_qubit_gate_color"]["primary"],
            linewidth=self.gate_linewidth,
            zorder=2,
        )
        self.ax.add_patch(rect)
        self.ax.text(
            x,
            -y,
            self._format_label(label),
            ha="center",
            va="center",
            color=self.styleOptions["one_qubit_gate_color"]["label"],
            zorder=3,
            **self._gate_text_options(
                self.styleOptions["one_qubit_gate_color"]["label"]
            ),
        )

    def draw_one_qubit_parametric(self, x, y, label, parameter):
        """Draw parametric single qubit gate."""
        raw_label_for_width = f"{label}_{{{parameter}}}"
        width = self.operation_width(raw_label_for_width)
        width = max(self.min_gate_width, width)
        x_start = x - width / 2
        height = self.single_gate_height

        rect = Rectangle(
            (x_start, -y - height / 2),
            width,
            height,
            fc="white",
            ec=self.styleOptions["one_qubit_gate_color"]["primary"],
            linewidth=self.gate_linewidth,
            zorder=2,
        )
        self.ax.add_patch(rect)
        self.ax.text(
            x,
            -y,
            self._format_parametric_label(label, parameter),
            ha="center",
            va="center",
            color=self.styleOptions["one_qubit_gate_color"]["label"],
            zorder=3,
            **self._gate_text_options(
                self.styleOptions["one_qubit_gate_color"]["label"]
            ),
        )

    def draw_two_qubit_non_parametric(self, x, control_y, target_y, label):
        """Draw two qubit gate with control and target."""
        # Draw vertical line
        self.ax.plot(
            [x, x],
            [-control_y, -target_y],
            color=self.styleOptions["two_qubit_gate_color"]["line"],
            linewidth=self.connector_linewidth,
            zorder=1,
        )

        # Draw control point
        self.ax.add_patch(
            Circle(
                (x, -control_y),
                0.08,  # Smaller size
                fc=self.styleOptions["two_qubit_gate_color"]["primary"],  # Darker color
                ec=self.styleOptions["two_qubit_gate_color"]["primary"],
                linewidth=self.control_linewidth,
                zorder=2,
            )
        )
        # Draw target
        self.ax.add_patch(
            Circle(
                (x, -target_y),
                0.25,  # Bigger size
                fc="white",
                ec=self.styleOptions["two_qubit_gate_color"]["primary"],
                linewidth=self.gate_linewidth,
                zorder=2,
            )
        )
        self.ax.text(
            x,
            -target_y,
            self._format_label(label),
            ha="center",
            va="center",
            color=self.styleOptions["two_qubit_gate_color"]["label"],
            zorder=3,
            **self._gate_text_options(
                self.styleOptions["two_qubit_gate_color"]["label"]
            ),
        )

    def draw_two_qubit_parametric(self, x, control_y, target_y, label, parameter):
        """Draw parametric two qubit gate."""
        # Draw vertical line
        self.ax.plot(
            [x, x],
            [-control_y, -target_y],
            color=self.styleOptions["two_qubit_gate_color"]["line"],
            linewidth=self.connector_linewidth,
            zorder=1,
        )
        # Draw control point
        self.ax.add_patch(
            Circle(
                (x, -control_y),
                0.08,  # Smaller size
                fc=self.styleOptions["two_qubit_gate_color"]["primary"],  # Darker color
                ec=self.styleOptions["two_qubit_gate_color"]["primary"],
                linewidth=self.control_linewidth,
                zorder=2,
            )
        )

        # Draw target as a wider rectangle
        raw_label_for_width = f"{label}_{{{parameter}}}"
        width = self.operation_width(raw_label_for_width)
        width = max(0.9, width)

        rect = Rectangle(
            (x - width / 2, -target_y - 0.35),
            width,
            0.7,  # Wider rectangle
            fc="white",
            ec=self.styleOptions["two_qubit_gate_color"]["primary"],
            linewidth=self.gate_linewidth,
            zorder=2,
        )
        self.ax.add_patch(rect)
        self.ax.text(
            x,
            -target_y,
            self._format_parametric_label(label, parameter),
            ha="center",
            va="center",
            color=self.styleOptions["two_qubit_gate_color"]["label"],
            zorder=3,
            **self._gate_text_options(
                self.styleOptions["two_qubit_gate_color"]["label"]
            ),
        )

    def draw_three_qubit_non_parametric(
        self, x, control1_y, control2_y, target_y, label
    ):
        """Draw three qubit gate with two controls and target."""
        # Draw vertical lines
        self.ax.plot(
            [x, x],
            [
                -min(control1_y, control2_y, target_y),
                -max(control1_y, control2_y, target_y),
            ],
            color=self.styleOptions["three_qubit_gate_color"]["line"],
            linewidth=self.connector_linewidth,
            zorder=1,
        )
        # Draw control points
        self.ax.add_patch(
            Circle(
                (x, -control1_y),
                0.08,  # Smaller size
                fc=self.styleOptions["three_qubit_gate_color"][
                    "primary"
                ],  # Darker color
                ec=self.styleOptions["three_qubit_gate_color"]["primary"],
                linewidth=self.control_linewidth,
                zorder=2,
            )
        )
        self.ax.add_patch(
            Circle(
                (x, -control2_y),
                0.08,  # Smaller size
                fc=self.styleOptions["three_qubit_gate_color"][
                    "primary"
                ],  # Darker color
                ec=self.styleOptions["three_qubit_gate_color"]["primary"],
                linewidth=self.control_linewidth,
                zorder=2,
            )
        )
        # Draw target
        self.ax.add_patch(
            Circle(
                (x, -target_y),
                0.25,  # Bigger size
                fc="white",
                ec=self.styleOptions["three_qubit_gate_color"]["primary"],
                linewidth=self.gate_linewidth,
                zorder=2,
            )
        )
        self.ax.text(
            x,
            -target_y,
            self._format_label(label),
            ha="center",
            va="center",
            color=self.styleOptions["three_qubit_gate_color"]["label"],
            zorder=3,
            **self._gate_text_options(
                self.styleOptions["three_qubit_gate_color"]["label"]
            ),
        )

    def draw_n_qubit_gate(
        self,
        x: float,
        qubits: list[int],
        label: str,
        params: list[float] | None = None,
    ):
        """Draw a gate that operates on n qubits with optional parameters."""
        if len(qubits) == 1:
            # Single qubit gate
            if params:
                self.draw_one_qubit_parametric(x, qubits[0], label, f"{params[0]:.2f}")
            else:
                self.draw_one_qubit_non_parametric(x, qubits[0], label)
        elif len(qubits) == 2:
            # Two qubit gate
            if params:
                self.draw_two_qubit_parametric(
                    x, qubits[0], qubits[1], label, f"{params[0]:.2f}"
                )
            else:
                self.draw_two_qubit_non_parametric(x, qubits[0], qubits[1], label)
        elif len(qubits) == 3:
            # Three qubit gate
            self.draw_three_qubit_non_parametric(
                x, qubits[0], qubits[1], qubits[2], label
            )
        else:
            # Multi-qubit gate - draw as a box covering all qubits
            min_y = min(qubits)
            max_y = max(qubits)
            self.draw_operation(x, min_y, max_y + 1, label)

    def draw_measure(self, x, qubit_y, clbit_y, target, basis="z"):
        """Draw measurement operation."""
        # Draw measurement box
        rect = Rectangle(
            (x - 0.45, -qubit_y - 0.3),
            0.9,
            0.6,  # Increased width
            fc="white",
            ec=self.styleOptions["measurement_color"]["primary"],
            linewidth=self.gate_linewidth,
            zorder=2,
        )
        self.ax.add_patch(rect)
        # Draw meter arc
        arc = Arc(
            (x, -qubit_y - 0.1),
            0.5,
            0.5,
            theta1=0,
            theta2=180,
            color=self.styleOptions["measurement_color"]["primary"],
            linewidth=self.connector_linewidth,
            zorder=2,
        )
        self.ax.add_patch(arc)
        # Draw line inside the semi-circle at an angle
        angle = np.deg2rad(45)  # 45 degrees
        line_length = 0.3  # Slightly shorter than the radius
        x_end = x + line_length * np.cos(angle)
        y_end = -qubit_y + line_length * np.sin(angle)
        self.ax.plot(
            [x, x_end],
            [-qubit_y - 0.1, y_end],
            color=self.styleOptions["measurement_color"]["primary"],
            linewidth=self.connector_linewidth,
            zorder=2,
        )
        # Draw vertical line to classical bit
        self.ax.plot(
            [x, x],
            [-qubit_y, -clbit_y],
            "--",
            color=self.styleOptions["measurement_color"]["line"],
            linewidth=self.connector_linewidth,
            zorder=1,
        )
        # Draw basis of measurement in the left corner
        self.ax.text(
            x - 0.30,
            -qubit_y + 0.15,
            self._format_label(basis),
            ha="center",
            va="center",
            color=self.styleOptions["measurement_color"]["primary"],
            zorder=3,
            **self._gate_text_options(
                self.styleOptions["measurement_color"]["primary"]
            ),
        )
        # Draw target value beside the dropdown on the clbit wire
        self.ax.text(
            x + 0.2,
            -clbit_y + 0.3,
            self._format_label(f"c{target}"),
            ha="left",
            va="center",
            color=self.styleOptions["measurement_color"]["primary"],
            zorder=3,
        )

    def draw_barrier(self, x, y):
        """Draw barrier line for a single qubit."""
        self.ax.plot(
            [x, x],
            [-y - 0.5, -y + 0.5],  # Extend from wire to +0.5 and -0.5
            color=self.styleOptions["barrier_color"],
            linewidth=4,  # Thicker line
            zorder=2,
        )

    def draw_reset(self, x, y):
        """Draw reset operation."""
        self.draw_one_qubit_non_parametric(x, y, r"$\vert 0 \rangle$")

    def draw_swap(self, x, y1, y2):
        """Draw swap operation."""
        # Draw vertical line
        self.ax.plot(
            [x, x],
            [-y1, -y2],
            color=self.styleOptions["two_qubit_gate_color"]["line"],
            linewidth=self.connector_linewidth,
            zorder=1,
        )
        # Draw control point
        self.ax.add_patch(
            Circle(
                (x, -y1),
                0.08,  # Smaller size
                fc=self.styleOptions["two_qubit_gate_color"]["primary"],  # Darker color
                ec=self.styleOptions["two_qubit_gate_color"]["primary"],
                linewidth=self.control_linewidth,
                zorder=2,
            )
        )
        # Draw target point
        self.ax.add_patch(
            Circle(
                (x, -y2),
                0.08,  # Smaller size
                fc=self.styleOptions["two_qubit_gate_color"]["primary"],  # Darker color
                ec=self.styleOptions["two_qubit_gate_color"]["primary"],
                linewidth=self.control_linewidth,
                zorder=2,
            )
        )

    def draw_controlled_swap(self, x, control_y, swap_y1, swap_y2):
        """Draw a controlled swap (Fredkin) gate."""
        # Draw vertical line connecting all qubits
        self.ax.plot(
            [x, x],
            [-min(control_y, swap_y1, swap_y2), -max(control_y, swap_y1, swap_y2)],
            color=self.styleOptions["three_qubit_gate_color"]["line"],
            linewidth=self.connector_linewidth,
            zorder=1,
        )
        # Draw control point
        self.ax.add_patch(
            Circle(
                (x, -control_y),
                0.08,  # Smaller size
                fc=self.styleOptions["three_qubit_gate_color"]["primary"],
                ec=self.styleOptions["three_qubit_gate_color"]["primary"],
                linewidth=self.control_linewidth,
                zorder=2,
            )
        )
        # Draw X symbols for swap
        self._draw_x_symbol(x, -swap_y1, 0.2)
        self._draw_x_symbol(x, -swap_y2, 0.2)

    def draw_operation(self, x, y_start, y_end, label):
        """Draw custom operation box."""
        height = y_end - y_start
        width = self.operation_width(label)
        rect = Rectangle(
            (x - width / 2, -y_end),
            width,
            height,  # Increased width
            fc="white",
            ec=self.styleOptions["one_qubit_gate_color"]["primary"],
            linewidth=self.gate_linewidth,
            zorder=2,
        )
        self.ax.add_patch(rect)
        self.ax.text(
            x,
            -(y_start + y_end) / 2,
            self._format_label(label),
            ha="center",
            va="center",
            color=self.styleOptions["one_qubit_gate_color"]["label"],
            zorder=3,
            **self._gate_text_options(
                self.styleOptions["one_qubit_gate_color"]["label"]
            ),
        )

    def _draw_x_symbol(self, x, y, size):
        """Helper method to draw X symbol for swap gates."""
        self.ax.plot(
            [x - size / 2, x + size / 2],
            [y - size / 2, y + size / 2],
            color=self.styleOptions["two_qubit_gate_color"]["primary"],
            linewidth=self.connector_linewidth,
            zorder=2,
        )
        self.ax.plot(
            [x - size / 2, x + size / 2],
            [y + size / 2, y - size / 2],
            color=self.styleOptions["two_qubit_gate_color"]["primary"],
            linewidth=self.connector_linewidth,
            zorder=2,
        )

    def draw_multi_qubit_gate(self, x, qubits, label):
        """Draw a box around multiple qubits for a multi-qubit gate."""
        min_y = min(qubits)
        max_y = max(qubits)

        # Draw vertical line connecting all qubits
        self.ax.plot(
            [x, x],
            [-min_y, -max_y],
            color=self.styleOptions["three_qubit_gate_color"]["line"],
            linewidth=self.connector_linewidth,
            zorder=1,
        )

        # Create a box spanning all qubits using the same width the scheduler uses.
        width = self.operation_width(label)
        height = max_y - min_y + 0.6  # Add padding
        rect = Rectangle(
            (x - width / 2, -max_y - 0.3),
            width,
            height,
            fc="white",
            ec=self.styleOptions["three_qubit_gate_color"]["primary"],
            linewidth=self.gate_linewidth,
            zorder=2,
        )
        self.ax.add_patch(rect)

        # Add label in the center
        self.ax.text(
            x,
            -(min_y + max_y) / 2,
            self._format_label(label),
            ha="center",
            va="center",
            color=self.styleOptions["three_qubit_gate_color"]["label"],
            zorder=3,
            **self._gate_text_options(
                self.styleOptions["three_qubit_gate_color"]["label"]
            ),
        )
