"""Interactive Bloch Sphere visualization using Plotly."""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Union


class PlotlyBlochSphere:
    """
    A class for creating interactive Bloch sphere visualizations using Plotly.

    The Bloch sphere is a geometrical representation of the pure state space of a
    two-level quantum mechanical system (qubit). This implementation provides
    interactive 3D visualization with rotation, zoom, and hover capabilities.
    """

    def __init__(self):
        """Initialize the Plotly Bloch sphere visualization."""
        self.traces = []
        self.vectors = []
        self.colors = []
        self.labels = []

    def _create_sphere_trace(self) -> go.Surface:
        """
        Create the sphere surface trace.

        Returns:
            Plotly Surface trace for the Bloch sphere
        """
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        return go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=[[0, "lightblue"], [1, "lightblue"]],
            showscale=False,
            opacity=0.2,
            name="Bloch Sphere",
            hoverinfo="skip",
        )

    def _create_axes_traces(self) -> list[go.Scatter3d]:
        """
        Create traces for the X, Y, Z axes.

        Returns:
            list of Plotly Scatter3d traces for axes
        """
        axis_length = 1.3
        traces = []

        # X-axis (red)
        traces.append(
            go.Scatter3d(
                x=[0, axis_length],
                y=[0, 0],
                z=[0, 0],
                mode="lines",
                line=dict(color="red", width=4),
                name="X-axis",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Y-axis (green)
        traces.append(
            go.Scatter3d(
                x=[0, 0],
                y=[0, axis_length],
                z=[0, 0],
                mode="lines",
                line=dict(color="green", width=4),
                name="Y-axis",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Z-axis (blue)
        traces.append(
            go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[0, axis_length],
                mode="lines",
                line=dict(color="blue", width=4),
                name="Z-axis",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        return traces

    def _create_axis_labels(self) -> list[go.Scatter3d]:
        """
        Create text annotations for axes and quantum states.

        Returns:
            list of Plotly Scatter3d traces for labels
        """
        axis_length = 1.4
        labels = []

        # Axis labels
        axis_positions = [
            (axis_length, 0, 0, "X", "red"),
            (0, axis_length, 0, "Y", "green"),
            (0, 0, axis_length, "Z", "blue"),
        ]

        for x, y, z, text, color in axis_positions:
            labels.append(
                go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode="text",
                    text=[text],
                    textfont=dict(size=16, color=color),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # State labels at key positions
        state_positions = [
            (0, 0, 1.15, "|0⟩"),
            (0, 0, -1.15, "|1⟩"),
            (1.15, 0, 0, "|+⟩"),
            (-1.15, 0, 0, "|-⟩"),
            (0, 1.15, 0, "|+i⟩"),
            (0, -1.15, 0, "|-i⟩"),
        ]

        for x, y, z, text in state_positions:
            labels.append(
                go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode="text",
                    text=[text],
                    textfont=dict(size=14, color="black"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        return labels

    def _create_circle_trace(
        self, radius: float = 1.0, axis: str = "z", color: str = "gray"
    ) -> go.Scatter3d:
        """
        Create a circle trace (equator or meridian).

        Args:
            radius: Radius of the circle
            axis: Axis perpendicular to the circle plane ('x', 'y', or 'z')
            color: Color of the circle line

        Returns:
            Plotly Scatter3d trace for the circle
        """
        theta = np.linspace(0, 2 * np.pi, 100)

        if axis == "z":  # Equator
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = np.zeros_like(theta)
        elif axis == "x":  # YZ plane
            x = np.zeros_like(theta)
            y = radius * np.cos(theta)
            z = radius * np.sin(theta)
        else:  # axis == 'y', XZ plane
            x = radius * np.cos(theta)
            y = np.zeros_like(theta)
            z = radius * np.sin(theta)

        return go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color=color, width=2),
            opacity=0.4,
            showlegend=False,
            hoverinfo="skip",
        )

    def add_vector(
        self,
        vector: list[float] | np.ndarray,
        color: str = "red",
        label: str | None = None,
    ):
        """
        Add a vector to the Bloch sphere.

        Args:
            vector: 3D vector [x, y, z] representing a point on the Bloch sphere
            color: Color of the vector arrow
            label: Optional label for the vector
        """
        vector = np.array(vector)
        # Normalize the vector to lie on the unit sphere
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            vector = vector / norm

        self.vectors.append(vector)
        self.colors.append(color)
        self.labels.append(label if label else f"Vector {len(self.vectors)}")

    def add_state(
        self,
        state_vector: list[complex] | np.ndarray,
        color: str = "red",
        label: str | None = None,
    ):
        """
        Add a quantum state to the Bloch sphere.

        Args:
            state_vector: Complex state vector [alpha, beta] where |ψ⟩ = alpha|0⟩ + beta|1⟩
            color: Color of the vector arrow
            label: Optional label for the state
        """
        state_vector = np.array(state_vector, dtype=complex)

        # Normalize the state vector
        norm = np.linalg.norm(state_vector)
        if norm > 1e-10:
            state_vector = state_vector / norm

        alpha, beta = state_vector[0], state_vector[1]

        # Convert to Bloch vector coordinates
        x = 2 * np.real(np.conj(alpha) * beta)
        y = 2 * np.imag(np.conj(alpha) * beta)
        z = np.abs(alpha) ** 2 - np.abs(beta) ** 2

        bloch_vector = np.array([x, y, z])
        self.add_vector(bloch_vector, color=color, label=label)

    def add_density_matrix(
        self, rho: np.ndarray, color: str = "red", label: str | None = None
    ):
        """
        Add a quantum state from its density matrix representation.

        Args:
            rho: 2x2 density matrix
            color: Color of the vector arrow
            label: Optional label for the state
        """
        rho = np.array(rho, dtype=complex)

        # Extract Bloch vector from density matrix
        x = 2 * np.real(rho[0, 1])
        y = 2 * np.imag(rho[0, 1])
        z = np.real(rho[0, 0] - rho[1, 1])

        bloch_vector = np.array([x, y, z])
        self.add_vector(bloch_vector, color=color, label=label)

    def add_state_from_angles(
        self, theta: float, phi: float, color: str = "red", label: str | None = None
    ):
        """
        Add a quantum state using spherical coordinates.

        The state is |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩

        Args:
            theta: Polar angle in radians (0 to π)
            phi: Azimuthal angle in radians (0 to 2π)
            color: Color of the vector arrow
            label: Optional label for the state
        """
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        bloch_vector = np.array([x, y, z])
        self.add_vector(bloch_vector, color=color, label=label)

    def _create_arrow(
        self, vector: np.ndarray, color: str, label: str
    ) -> list[go.Scatter3d]:
        """
        Create an arrow trace for a state vector.

        Args:
            vector: 3D vector coordinates
            color: Color of the arrow
            label: Label for the arrow

        Returns:
            list of Plotly traces forming the arrow
        """
        traces = []

        # Arrow shaft
        traces.append(
            go.Scatter3d(
                x=[0, vector[0]],
                y=[0, vector[1]],
                z=[0, vector[2]],
                mode="lines",
                line=dict(color=color, width=6),
                name=label,
                hovertemplate=f"<b>{label}</b><br>"
                + f"x: {vector[0]:.3f}<br>"
                + f"y: {vector[1]:.3f}<br>"
                + f"z: {vector[2]:.3f}<extra></extra>",
                showlegend=True,
            )
        )

        # Arrow head (cone)
        arrow_length = 0.15
        direction = vector / np.linalg.norm(vector)

        traces.append(
            go.Cone(
                x=[vector[0] - direction[0] * arrow_length * 0.3],
                y=[vector[1] - direction[1] * arrow_length * 0.3],
                z=[vector[2] - direction[2] * arrow_length * 0.3],
                u=[direction[0]],
                v=[direction[1]],
                w=[direction[2]],
                sizemode="absolute",
                sizeref=arrow_length,
                anchor="tail",
                colorscale=[[0, color], [1, color]],
                showscale=False,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        return traces

    def clear(self):
        """Clear all vectors from the Bloch sphere."""
        self.vectors = []
        self.colors = []
        self.labels = []
        self.traces = []

    def show(
        self,
        title: str = "Interactive Bloch Sphere",
        width: int = 800,
        height: int = 800,
    ):
        """
        Display the interactive Bloch sphere with all added vectors.

        Args:
            title: Title of the plot
            width: Width of the figure in pixels
            height: Height of the figure in pixels
        """
        # Create figure
        fig = go.Figure()

        # Add sphere
        fig.add_trace(self._create_sphere_trace())

        # Add circles (equator and meridians)
        fig.add_trace(self._create_circle_trace(axis="z", color="gray"))
        fig.add_trace(self._create_circle_trace(axis="x", color="gray"))
        fig.add_trace(self._create_circle_trace(axis="y", color="gray"))

        # Add axes
        for trace in self._create_axes_traces():
            fig.add_trace(trace)

        # Add labels
        for trace in self._create_axis_labels():
            fig.add_trace(trace)

        # Add all state vectors
        for vector, color, label in zip(self.vectors, self.colors, self.labels):
            for trace in self._create_arrow(vector, color, label):
                fig.add_trace(trace)

        # Update layout
        fig.update_layout(
            title=dict(
                text=title, font=dict(size=20, color="black"), x=0.5, xanchor="center"
            ),
            scene=dict(
                xaxis=dict(visible=False, range=[-1.5, 1.5]),
                yaxis=dict(visible=False, range=[-1.5, 1.5]),
                zaxis=dict(visible=False, range=[-1.5, 1.5]),
                aspectmode="cube",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            ),
            width=width,
            height=height,
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(255, 255, 255, 0.8)"),
            hovermode="closest",
        )

        fig.show()

    def save(self, filename: str, **kwargs):
        """
        Save the Bloch sphere to an HTML file.

        Args:
            filename: Output filename (should end with .html)
            **kwargs: Additional arguments passed to fig.write_html()
        """
        # Create the figure using the same process as show()
        fig = go.Figure()
        fig.add_trace(self._create_sphere_trace())
        fig.add_trace(self._create_circle_trace(axis="z", color="gray"))
        fig.add_trace(self._create_circle_trace(axis="x", color="gray"))
        fig.add_trace(self._create_circle_trace(axis="y", color="gray"))

        for trace in self._create_axes_traces():
            fig.add_trace(trace)
        for trace in self._create_axis_labels():
            fig.add_trace(trace)
        for vector, color, label in zip(self.vectors, self.colors, self.labels):
            for trace in self._create_arrow(vector, color, label):
                fig.add_trace(trace)

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, range=[-1.5, 1.5]),
                yaxis=dict(visible=False, range=[-1.5, 1.5]),
                zaxis=dict(visible=False, range=[-1.5, 1.5]),
                aspectmode="cube",
            ),
            showlegend=True,
        )

        fig.write_html(filename, **kwargs)


def create_interactive_bloch_sphere(
    states: list[np.ndarray] | None = None,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    title: str = "Interactive Bloch Sphere",
) -> PlotlyBlochSphere:
    """
    Convenience function to create and display an interactive Bloch sphere.

    Args:
        states: list of state vectors (each is [alpha, beta])
        colors: list of colors for each state
        labels: list of labels for each state
        title: Title of the plot

    Returns:
        PlotlyBlochSphere object
    """
    bloch = PlotlyBlochSphere()

    if states is not None:
        if colors is None:
            colors = ["red", "blue", "green", "orange", "purple"] * (
                len(states) // 5 + 1
            )
        if labels is None:
            labels = [f"State {i + 1}" for i in range(len(states))]

        for state, color, label in zip(states, colors, labels):
            bloch.add_state(state, color=color, label=label)

    bloch.show(title=title)
    return bloch
