"""
Comprehensive visualization suite for VQE results.
Provides multiple plot types for analyzing optimization convergence.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# Try to import seaborn, but make it optional
try:
    import seaborn as sns  # type: ignore[import-untyped]

    HAS_SEABORN = True
    # Set nice default style
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False
    # Use matplotlib defaults if seaborn is not available
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

plt.rcParams["figure.dpi"] = 100


def plot_vqe_results_comprehensive(result, figsize=(16, 12)):
    """
    Create a comprehensive 6-panel visualization of VQE results.

    Args:
        result: VQEResult object with optimal_parameters, optimal_energy,
                energy_history, param_history, counts, metadata
        figsize: Figure size (width, height)
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Determine appropriate window size based on history length
    history_len = len(result.energy_history)
    window_size = min(10, max(3, history_len // 3))  # Adaptive window

    # 1. Measurement Probability Distribution (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_measurement_distribution(result, ax=ax1)

    # 2. Energy Convergence (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_energy_convergence(result, ax=ax2)

    # 3. Energy Improvement per Iteration (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_energy_improvements(result, ax=ax3)

    # 4. Parameter Evolution (middle left - spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    plot_parameter_evolution(result, ax=ax4)

    # 5. Convergence Metrics (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    plot_convergence_metrics(result, ax=ax5)

    # 6. Parameter Heatmap (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    plot_parameter_heatmap(result, ax=ax6)

    # 7. Rolling Statistics (bottom middle) - with adaptive window
    ax7 = fig.add_subplot(gs[2, 1])
    plot_rolling_statistics(result, ax=ax7, window=window_size)

    fig.suptitle(
        f"VQE Optimization Analysis - {result.metadata.get('optimizer', 'Unknown').upper()}",
        fontsize=16,
        fontweight="bold",
    )


def plot_measurement_distribution(result, ax=None, top_n=8):
    """
    Plot measurement probability distribution.

    Args:
        result: VQEResult object
        ax: Matplotlib axis (creates new if None)
        top_n: Number of top states to show
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    counts = result.counts
    total_shots = sum(counts.values())

    # Calculate probabilities
    probs = {state: count / total_shots for state, count in counts.items()}

    # Sort by probability and take top N
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    states, probabilities = zip(*sorted_probs)

    # Create bar plot
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(states)))
    bars = ax.bar(
        range(len(states)),
        probabilities,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )

    # Highlight the most probable state
    bars[0].set_color("red")
    bars[0].set_alpha(0.8)

    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, rotation=45, ha="right")
    ax.set_ylabel("Probability", fontsize=11, fontweight="bold")
    ax.set_xlabel("Quantum State", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Measurement Distribution (Top {top_n} States)\nMost Probable: {states[0]}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add probability values on top of bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{prob:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def plot_energy_convergence(result, ax=None):
    """
    Plot energy convergence over iterations with key markers.

    Args:
        result: VQEResult object
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    history = result.energy_history
    iterations = range(len(history))

    # Main energy line
    ax.plot(iterations, history, "b-", linewidth=2, alpha=0.7, label="Energy")

    # Mark start and end
    ax.plot(
        0, history[0], "go", markersize=12, label=f"Start: {history[0]:.4f}", zorder=5
    )
    ax.plot(
        len(history) - 1,
        history[-1],
        "r*",
        markersize=15,
        label=f"Final: {history[-1]:.4f}",
        zorder=5,
    )

    # Add horizontal line at final energy
    ax.axhline(
        y=result.optimal_energy,
        color="r",
        linestyle="--",
        alpha=0.5,
        linewidth=1.5,
        label="Optimal",
    )

    # Fill area showing improvement
    ax.fill_between(
        iterations,
        history,
        result.optimal_energy,
        where=(np.array(history) > result.optimal_energy),
        alpha=0.2,
        color="orange",
        label="Optimization region",
    )

    ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Energy", fontsize=11, fontweight="bold")
    ax.set_title("Energy Convergence", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_energy_improvements(result, ax=None):
    """
    Plot energy improvement per iteration (delta energy).

    Args:
        result: VQEResult object
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    history = result.energy_history
    improvements = [history[i] - history[i + 1] for i in range(len(history) - 1)]
    iterations = range(1, len(history))

    # Color bars based on positive/negative improvement
    colors = ["green" if imp > 0 else "red" for imp in improvements]

    ax.bar(
        iterations,
        improvements,
        color=colors,
        alpha=0.6,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Add threshold line for significant improvement
    threshold = np.std(improvements) * 0.5
    ax.axhline(
        y=threshold,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label=f"Significance threshold: {threshold:.4f}",
    )

    ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Energy Improvement (ΔE)", fontsize=11, fontweight="bold")
    ax.set_title("Energy Improvement per Iteration", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add text for total improvement
    total_improvement = history[0] - history[-1]
    ax.text(
        0.02,
        0.98,
        f"Total Improvement: {total_improvement:.6f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
        fontweight="bold",
    )


def plot_parameter_evolution(result, ax=None, max_params=6):
    """
    Plot evolution of parameters over iterations.

    Args:
        result: VQEResult object
        ax: Matplotlib axis
        max_params: Maximum number of parameters to plot (for readability)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    param_history = np.array(result.param_history)
    n_params = param_history.shape[1]
    iterations = range(len(param_history))

    # If too many parameters, select a subset or plot all with thin lines
    if n_params > max_params:
        # Plot all with thin lines
        for i in range(n_params):
            ax.plot(iterations, param_history[:, i], alpha=0.3, linewidth=1)
        # Highlight first few
        colors = plt.cm.tab10(np.arange(max_params))
        for i in range(min(max_params, n_params)):
            ax.plot(
                iterations,
                param_history[:, i],
                linewidth=2,
                color=colors[i],
                label=f"θ_{i}",
                marker="o",
                markersize=3,
                markevery=max(1, len(iterations) // 20),
            )
    else:
        colors = plt.cm.tab10(np.arange(n_params))
        for i in range(n_params):
            ax.plot(
                iterations,
                param_history[:, i],
                linewidth=2,
                color=colors[i],
                label=f"θ_{i}",
                marker="o",
                markersize=4,
                markevery=max(1, len(iterations) // 20),
            )

    ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Parameter Value (radians)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Parameter Evolution ({n_params} parameters)", fontsize=12, fontweight="bold"
    )
    ax.legend(loc="best", ncol=min(3, (n_params + 2) // 3), fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_convergence_metrics(result, ax=None):
    """
    Display convergence metrics as text.

    Args:
        result: VQEResult object
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.axis("off")

    # Calculate metrics
    metadata = result.metadata
    history = result.energy_history

    total_improvement = history[0] - history[-1]
    avg_improvement = total_improvement / len(history)

    # Calculate convergence rate (exponential fit of improvement)
    improvements = np.abs(np.diff(history))
    if len(improvements) > 1:
        convergence_rate = (
            np.mean(improvements[-10:]) / np.mean(improvements[:10])
            if np.mean(improvements[:10]) > 0
            else 0
        )
    else:
        convergence_rate = 0

    metrics_text = f"""
╔═══════════════════════════════╗
║   CONVERGENCE METRICS         ║
╚═══════════════════════════════╝

Optimizer:          {metadata.get("optimizer", "N/A").upper()}
Iterations:         {metadata.get("actual_iterations", len(history))}
Max Iterations:     {metadata.get("max_iterations_limit", "N/A")}
Circuit Evals:      {metadata.get("circuit_evaluations", "N/A")}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Initial Energy:     {history[0]:.6f}
Final Energy:       {history[-1]:.6f}
Total Improvement:  {total_improvement:.6f}
Avg per Iteration:  {avg_improvement:.6f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Converged Early:    {metadata.get("converged_early", False)}
Success:            {metadata.get("success", False)}
Convergence Rate:   {convergence_rate:.4f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Parameters:         {metadata.get("n_parameters", len(result.optimal_parameters))}
Backend:            {str(metadata.get("backend", "N/A")).split(".")[-1]}
Shots:              {metadata.get("shots", "N/A")}
"""

    ax.text(
        0.1,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )


def plot_parameter_heatmap(result, ax=None):
    """
    Plot heatmap of parameter values over iterations.

    Args:
        result: VQEResult object
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    param_history = np.array(result.param_history).T  # Transpose for heatmap

    im = ax.imshow(
        param_history, aspect="auto", cmap="RdYlBu_r", interpolation="nearest"
    )

    ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Parameter Index", fontsize=11, fontweight="bold")
    ax.set_title("Parameter Values Heatmap", fontsize=12, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Value (radians)", rotation=270, labelpad=20, fontweight="bold")

    # Set y-ticks to parameter indices
    ax.set_yticks(range(param_history.shape[0]))
    ax.set_yticklabels([f"θ_{i}" for i in range(param_history.shape[0])])


def plot_rolling_statistics(result, ax=None, window=10):
    """
    Plot rolling mean and standard deviation of energy.

    Args:
        result: VQEResult object
        ax: Matplotlib axis
        window: Window size for rolling statistics
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    history = np.array(result.energy_history)
    iterations = range(len(history))

    # Adjust window size if history is too short
    effective_window = min(window, len(history))

    # If history is very short, just plot the energy
    if len(history) < 3:
        ax.plot(iterations, history, "b-", linewidth=2, marker="o", label="Energy")
        ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
        ax.set_ylabel("Energy", fontsize=11, fontweight="bold")
        ax.set_title(
            "Energy History (too few points for rolling statistics)",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        return

    # Calculate rolling statistics
    rolling_mean = np.convolve(
        history, np.ones(effective_window) / effective_window, mode="valid"
    )
    rolling_std = np.array(
        [
            np.std(history[max(0, i - effective_window) : i + 1])
            for i in range(len(history))
        ]
    )

    # Plot actual energy
    ax.plot(iterations, history, "b-", alpha=0.3, linewidth=1, label="Energy")

    # Plot rolling mean
    if len(rolling_mean) > 0:
        rolling_x = range(effective_window - 1, len(history))
        ax.plot(
            rolling_x,
            rolling_mean,
            "r-",
            linewidth=2,
            label=f"{effective_window}-iter Moving Avg",
        )

        # Plot confidence band (rolling_mean ± rolling_std)
        rolling_std_subset = rolling_std[effective_window - 1 :]
        if len(rolling_std_subset) == len(rolling_mean):
            ax.fill_between(
                rolling_x,
                rolling_mean - rolling_std_subset,
                rolling_mean + rolling_std_subset,
                alpha=0.2,
                color="red",
                label="±1 Std Dev",
            )

    ax.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax.set_ylabel("Energy", fontsize=11, fontweight="bold")

    if effective_window < window:
        ax.set_title(
            f"Rolling Statistics (window={effective_window}, reduced from {window})",
            fontsize=12,
            fontweight="bold",
        )
    else:
        ax.set_title(
            f"Rolling Statistics (window={window})", fontsize=12, fontweight="bold"
        )

    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_summary_statistics(result, ax=None):
    """
    Display statistical summary of the optimization.

    Args:
        result: VQEResult object
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.axis("off")

    history = result.energy_history
    improvements = np.abs(np.diff(history))
    param_history = np.array(result.param_history)

    # Calculate statistics
    stats_text = f"""
╔═══════════════════════════════╗
║   STATISTICAL SUMMARY         ║
╚═══════════════════════════════╝

ENERGY STATISTICS:
──────────────────────────────
Min Energy:         {np.min(history):.6f}
Max Energy:         {np.max(history):.6f}
Mean Energy:        {np.mean(history):.6f}
Std Dev:            {np.std(history):.6f}
Range:              {np.max(history) - np.min(history):.6f}

IMPROVEMENT STATISTICS:
──────────────────────────────
Mean Improvement:   {np.mean(improvements):.6f}
Max Improvement:    {np.max(improvements):.6f}
Min Improvement:    {np.min(improvements):.6f}
Last 5 Avg:         {np.mean(improvements[-5:]):.6f}

PARAMETER STATISTICS:
──────────────────────────────
Param Mean:         {np.mean(param_history[-1]):.4f}
Param Std:          {np.std(param_history[-1]):.4f}
Param Range:        {np.max(param_history[-1]) - np.min(param_history[-1]):.4f}
"""

    ax.text(
        0.1,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
    )


# ============================================================================
# SPECIALIZED PLOTS FOR SPECIFIC ANALYSES
# ============================================================================


def plot_parameter_trajectory_2d(result, param_indices=(0, 1), ax=None):
    """
    Plot 2D trajectory of two parameters in parameter space.

    Args:
        result: VQEResult object
        param_indices: tuple of (param1_idx, param2_idx)
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    param_history = np.array(result.param_history)
    idx1, idx2 = param_indices

    # Plot trajectory
    ax.plot(
        param_history[:, idx1],
        param_history[:, idx2],
        "b-",
        alpha=0.5,
        linewidth=2,
        label="Trajectory",
    )

    # Mark start and end
    ax.plot(
        param_history[0, idx1],
        param_history[0, idx2],
        "go",
        markersize=15,
        label="Start",
        zorder=5,
    )
    ax.plot(
        param_history[-1, idx1],
        param_history[-1, idx2],
        "r*",
        markersize=20,
        label="End",
        zorder=5,
    )

    # Add arrows to show direction
    n_arrows = min(10, len(param_history) - 1)
    arrow_indices = np.linspace(0, len(param_history) - 2, n_arrows, dtype=int)
    for i in arrow_indices:
        dx = param_history[i + 1, idx1] - param_history[i, idx1]
        dy = param_history[i + 1, idx2] - param_history[i, idx2]
        ax.arrow(
            param_history[i, idx1],
            param_history[i, idx2],
            dx,
            dy,
            head_width=0.1,
            head_length=0.1,
            fc="blue",
            ec="blue",
            alpha=0.3,
        )

    ax.set_xlabel(f"Parameter θ_{idx1} (radians)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Parameter θ_{idx2} (radians)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Parameter Space Trajectory (θ_{idx1} vs θ_{idx2})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")


def plot_comparison_multi_runs(results_dict, figsize=(14, 6)):
    """
    Compare multiple VQE runs (e.g., different optimizers or parameters).

    Args:
        results_dict: Dictionary of {label: result} pairs
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    # Plot 1: Energy convergence comparison
    for (label, result), color in zip(results_dict.items(), colors):
        history = result.energy_history
        ax1.plot(history, linewidth=2, label=label, color=color, alpha=0.7)

    ax1.set_xlabel("Iteration", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Energy", fontsize=11, fontweight="bold")
    ax1.set_title("Energy Convergence Comparison", fontsize=12, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final energy and iterations comparison
    labels = list(results_dict.keys())
    final_energies = [r.optimal_energy for r in results_dict.values()]
    iterations = [
        r.metadata.get("actual_iterations", len(r.energy_history))
        for r in results_dict.values()
    ]

    x = np.arange(len(labels))
    width = 0.35

    ax2_twin = ax2.twinx()

    ax2.bar(
        x - width / 2,
        final_energies,
        width,
        label="Final Energy",
        color=colors,
        alpha=0.7,
    )
    ax2_twin.bar(
        x + width / 2,
        iterations,
        width,
        label="Iterations",
        color=colors,
        alpha=0.4,
        edgecolor="black",
        linewidth=2,
    )

    ax2.set_xlabel("Optimizer", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Final Energy", fontsize=11, fontweight="bold")
    ax2_twin.set_ylabel("Iterations", fontsize=11, fontweight="bold")
    ax2.set_title("Final Results Comparison", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    return fig
