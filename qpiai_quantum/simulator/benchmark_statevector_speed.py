import cProfile
import csv
import io
import json
import pstats
import statistics
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[2]
if "qpiai_quantum" not in sys.modules:
    qpiai_quantum = types.ModuleType("qpiai_quantum")
    qpiai_quantum.__path__ = [str(ROOT / "qpiai_quantum")]
    sys.modules["qpiai_quantum"] = qpiai_quantum

from qpiai_quantum.circuit.circuit import Circuit
from qpiai_quantum.icr.circuitoperation import OperationType
from qpiai_quantum.simulator.gates import DECOMPOSED_GATES, decompose, gate_spec
from qpiai_quantum.simulator.result import QasmSimulatorResult
from qpiai_quantum.simulator.statevector import StatevectorSimulator


OPTIMIZED_GATES = ("h", "x", "y", "z", "s", "rx", "ry", "rz")
PARAMETRIC_GATES = {"rx", "ry", "rz"}
ITERATIONS = 30
WARMUPS = 5
RNG_SEED = 20260711


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    workload: str
    n_qubits: int
    specs: tuple[tuple[str, tuple[int, ...], tuple[float, ...]], ...]


class OriginalStatevectorSimulator(StatevectorSimulator):
    def run(
        self,
        circuit: Circuit,
        shots: int = 1024,
        seed: int | None = None,
        name: str | None = None,
    ) -> QasmSimulatorResult:
        start_time = time.perf_counter()
        n_qubits = circuit.num_qubits
        n_cbits = circuit.num_clbits

        if n_qubits == 0:
            raise ValueError("Cannot simulate a circuit with 0 qubits.")

        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1.0
        measure_map: dict[int, int] = {}

        def _apply_gate(gate_name: str, params: list[float], qubits: list[int]) -> None:
            nonlocal state
            gate_name_lower = gate_name.lower()

            if gate_name_lower in DECOMPOSED_GATES:
                for sub_name, sub_params, sub_qubits in decompose(
                    gate_name_lower, qubits
                ):
                    _apply_gate(sub_name, sub_params, sub_qubits)
                return

            _, unitary = gate_spec(gate_name_lower, params, num_qubits=len(qubits))
            state = self._apply_unitary(state, n_qubits, qubits, unitary)

        def _apply_op(op) -> None:
            nonlocal state
            if op.operation_type == OperationType.BARRIER:
                return

            if op.operation_type == OperationType.MEASURE:
                if op.qubits and op.clbits:
                    for q, c in zip(op.qubits, op.clbits):
                        measure_map[q] = c
                return

            if op.operation_type in (
                OperationType.N_QUBIT_NON_PARAMETRIC,
                OperationType.N_QUBIT_PARAMETRIC,
                OperationType.SWAP,
            ):
                _apply_gate(op.gate_name, op.params or [], op.qubits)
                return

            if op.operation_type == OperationType.OPERATION:
                if hasattr(op, "order") and op.order is not None:
                    for sub_op in op.order:
                        _apply_op(sub_op)
                else:
                    _apply_gate(op.gate_name, op.params or [], op.qubits)
                return

            if op.gate_name.lower() == "reset" and op.qubits:
                for q in op.qubits:
                    state = self._apply_reset(state, n_qubits, q)
            else:
                raise ValueError(f"Unsupported operation type: {op.operation_type}")

        for op in circuit.icr.evolve:
            _apply_op(op)

        counts = (
            self._sample_counts(state, n_qubits, n_cbits, measure_map, shots, seed)
            if n_cbits > 0 and measure_map
            else {}
        )

        return QasmSimulatorResult(
            name=name or circuit.name,
            counts=counts,
            statevector=state.tolist(),
            shots=shots,
            executionTime=time.perf_counter() - start_time,
            method="statevector-original",
            job_status="completed",
            n_qubits=n_qubits,
            n_cbits=n_cbits,
        )


def apply_spec(circuit: Circuit, spec):
    gate, qubits, params = spec
    if gate in PARAMETRIC_GATES:
        getattr(circuit, gate)(qubits[0], params[0])
    elif gate == "cx":
        circuit.cx(qubits[0], qubits[1])
    elif gate == "cz":
        circuit.cz(qubits[0], qubits[1])
    elif gate == "swap":
        circuit.swap(qubits[0], qubits[1])
    else:
        getattr(circuit, gate)(qubits[0])


def build_circuit(n_qubits: int, specs) -> Circuit:
    circuit = Circuit(n_qubits)
    for spec in specs:
        apply_spec(circuit, spec)
    return circuit


def statevector(simulator, circuit: Circuit) -> np.ndarray:
    return np.asarray(simulator.run(circuit, shots=1).statevector, dtype=complex)


def describe_specs(specs) -> str:
    lines = []
    for index, (gate, qubits, params) in enumerate(specs, start=1):
        suffix = f", params={list(params)}" if params else ""
        lines.append(f"{index}. {gate.upper()} qubits={list(qubits)}{suffix}")
    return "\n".join(lines)


def first_divergence(n_qubits: int, specs) -> str:
    original = OriginalStatevectorSimulator()
    optimized = StatevectorSimulator()
    for index in range(1, len(specs) + 1):
        prefix = specs[:index]
        circuit = build_circuit(n_qubits, prefix)
        original_state = statevector(original, circuit)
        optimized_state = statevector(optimized, circuit)
        if not np.allclose(original_state, optimized_state):
            gate, qubits, params = specs[index - 1]
            return (
                f"first divergent gate {index}: "
                f"{gate.upper()} qubits={list(qubits)} params={list(params)}"
            )
    return "no divergent prefix found"


def verify_case(case: BenchmarkCase):
    circuit = build_circuit(case.n_qubits, case.specs)
    original_state = statevector(OriginalStatevectorSimulator(), circuit)
    optimized_state = statevector(StatevectorSimulator(), circuit)

    if not np.allclose(original_state, optimized_state):
        detail = first_divergence(case.n_qubits, case.specs)
        raise RuntimeError(
            f"Verification failed for {case.name}\n"
            f"Circuit:\n{describe_specs(case.specs)}\n"
            f"Original statevector:\n{original_state}\n"
            f"Optimized statevector:\n{optimized_state}\n"
            f"{detail}\n"
            "Reason: the optimized pair-wise kernel is not mathematically "
            "equivalent to the original matrix path for this prefix."
        )

    if case.workload == "Bell state":
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[3] = 1 / np.sqrt(2)
        if not np.allclose(original_state, expected) or not np.allclose(
            optimized_state, expected
        ):
            raise RuntimeError(
                "Bell-state expected-value verification failed.\n"
                f"Original statevector: {original_state}\n"
                f"Optimized statevector: {optimized_state}\n"
                f"Expected statevector: {expected}"
            )

    return circuit


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "stddev": statistics.stdev(samples),
    }


def benchmark_case(case: BenchmarkCase) -> dict[str, object]:
    circuit = verify_case(case)
    original = OriginalStatevectorSimulator()
    optimized = StatevectorSimulator()

    for _ in range(WARMUPS):
        original.run(circuit, shots=1)
        optimized.run(circuit, shots=1)

    original_times = []
    optimized_times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        original.run(circuit, shots=1)
        original_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        optimized.run(circuit, shots=1)
        optimized_times.append(time.perf_counter() - start)

    original_stats = summarize(original_times)
    optimized_stats = summarize(optimized_times)
    return {
        "name": case.name,
        "workload": case.workload,
        "n_qubits": case.n_qubits,
        "depth": len(case.specs),
        "iterations": ITERATIONS,
        "warmups": WARMUPS,
        "original": original_stats,
        "optimized": optimized_stats,
        "speedup": original_stats["mean"] / optimized_stats["mean"],
    }


def repeated_gate_specs(gate: str, n_qubits: int, depth: int, seed: int):
    rng = np.random.default_rng(seed)
    specs = []
    for index in range(depth):
        if n_qubits >= 15:
            target = n_qubits // 2 + (index % max(1, n_qubits - n_qubits // 2))
        else:
            target = index % n_qubits
        params = (
            (float(rng.uniform(-2 * np.pi, 2 * np.pi)),)
            if gate in PARAMETRIC_GATES
            else ()
        )
        specs.append((gate, (target,), params))
    return tuple(specs)


def random_single_qubit_specs(n_qubits: int, depth: int, seed: int):
    rng = np.random.default_rng(seed)
    low_target = n_qubits // 2 if n_qubits >= 15 else 0
    specs = []
    for _ in range(depth):
        gate = str(rng.choice(OPTIMIZED_GATES))
        target = int(rng.integers(low_target, n_qubits))
        params = (
            (float(rng.uniform(-2 * np.pi, 2 * np.pi)),)
            if gate in PARAMETRIC_GATES
            else ()
        )
        specs.append((gate, (target,), params))
    return tuple(specs)


def mixed_specs(n_qubits: int, depth: int, seed: int):
    rng = np.random.default_rng(seed)
    low_target = n_qubits // 2 if n_qubits >= 15 else 0
    specs = []
    for index in range(depth):
        if n_qubits > 1 and index % 9 == 8:
            first = int(rng.integers(0, n_qubits))
            second = int((first + rng.integers(1, n_qubits)) % n_qubits)
            specs.append(("swap", (first, second), ()))
        elif n_qubits > 1 and index % 5 == 4:
            control = int(rng.integers(0, n_qubits))
            target = int((control + rng.integers(1, n_qubits)) % n_qubits)
            specs.append(("cx", (control, target), ()))
        elif n_qubits > 1 and index % 7 == 6:
            control = int(rng.integers(0, n_qubits))
            target = int((control + rng.integers(1, n_qubits)) % n_qubits)
            specs.append(("cz", (control, target), ()))
        else:
            gate = str(rng.choice(OPTIMIZED_GATES))
            target = int(rng.integers(low_target, n_qubits))
            params = (
                (float(rng.uniform(-2 * np.pi, 2 * np.pi)),)
                if gate in PARAMETRIC_GATES
                else ()
            )
            specs.append((gate, (target,), params))
    return tuple(specs)


def build_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    stress_depths = {5: 120, 10: 60, 15: 8, 20: 1}
    random_depths = {5: 140, 10: 70, 15: 10, 20: 1}
    mixed_depths = {5: 100, 10: 50, 15: 10, 20: 1}

    for n_qubits, depth in stress_depths.items():
        for gate in OPTIMIZED_GATES:
            cases.append(
                BenchmarkCase(
                    name=f"{gate.upper()} repeated, {n_qubits}q",
                    workload="Single-gate stress",
                    n_qubits=n_qubits,
                    specs=repeated_gate_specs(
                        gate, n_qubits, depth, RNG_SEED + n_qubits
                    ),
                )
            )

    for n_qubits, depth in random_depths.items():
        cases.append(
            BenchmarkCase(
                name=f"Random optimized gates, {n_qubits}q",
                workload="Random single-qubit",
                n_qubits=n_qubits,
                specs=random_single_qubit_specs(
                    n_qubits, depth, RNG_SEED + 100 + n_qubits
                ),
            )
        )

    for n_qubits, depth in mixed_depths.items():
        cases.append(
            BenchmarkCase(
                name=f"Mixed 1q + CX/CZ/SWAP, {n_qubits}q",
                workload="Mixed",
                n_qubits=n_qubits,
                specs=mixed_specs(n_qubits, depth, RNG_SEED + 200 + n_qubits),
            )
        )

    cases.extend(
        [
            BenchmarkCase(
                name="Bell state H(0), CX(0, 1)",
                workload="Bell state",
                n_qubits=2,
                specs=(("h", (0,), ()), ("cx", (0, 1), ())),
            ),
            BenchmarkCase(
                name="Large random optimized gates, 12q depth 240",
                workload="Large stress",
                n_qubits=12,
                specs=random_single_qubit_specs(12, 240, RNG_SEED + 300),
            ),
            BenchmarkCase(
                name="Large random optimized gates, 15q depth 16",
                workload="Large stress",
                n_qubits=15,
                specs=random_single_qubit_specs(15, 16, RNG_SEED + 301),
            ),
            BenchmarkCase(
                name="Large mixed circuit, 12q depth 160",
                workload="Large stress",
                n_qubits=12,
                specs=mixed_specs(12, 160, RNG_SEED + 302),
            ),
        ]
    )
    return cases


def profile_optimized() -> dict[str, object]:
    case = BenchmarkCase(
        name="Profile random optimized gates, 12q depth 240",
        workload="Profiling",
        n_qubits=12,
        specs=random_single_qubit_specs(12, 240, RNG_SEED + 400),
    )
    circuit = verify_case(case)
    simulator = StatevectorSimulator()
    for _ in range(3):
        simulator.run(circuit, shots=1)

    profiler = cProfile.Profile()
    profiler.enable()
    simulator.run(circuit, shots=1)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(12)
    entries = []
    total = stats.total_tt
    for key, value in stats.stats.items():
        filename, line, func_name = key
        call_count, primitive, total_time, cumulative_time, callers = value
        entries.append(
            {
                "function": f"{func_name} ({filename}:{line})",
                "calls": call_count,
                "total_s": total_time,
                "cumulative_s": cumulative_time,
                "percent_total": (total_time / total * 100) if total else 0.0,
            }
        )
    entries.sort(key=lambda item: item["cumulative_s"], reverse=True)
    return {
        "case": case.name,
        "total_profiled_s": total,
        "top_entries": entries[:12],
        "raw": stream.getvalue(),
    }


def machine_info() -> dict[str, str]:
    import platform

    cpu = platform.processor() or platform.machine() or "not reported by OS"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": cpu,
    }


def fmt_ms(value: float) -> str:
    return f"{value * 1000:.3f}"


def write_csv(results: list[dict[str, object]], output: Path):
    with output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "workload",
                "name",
                "n_qubits",
                "depth",
                "iterations",
                "warmups",
                "original_mean_s",
                "optimized_mean_s",
                "original_median_s",
                "optimized_median_s",
                "original_min_s",
                "optimized_min_s",
                "original_max_s",
                "optimized_max_s",
                "original_stddev_s",
                "optimized_stddev_s",
                "speedup",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row["workload"],
                    row["name"],
                    row["n_qubits"],
                    row["depth"],
                    row["iterations"],
                    row["warmups"],
                    row["original"]["mean"],
                    row["optimized"]["mean"],
                    row["original"]["median"],
                    row["optimized"]["median"],
                    row["original"]["min"],
                    row["optimized"]["min"],
                    row["original"]["max"],
                    row["optimized"]["max"],
                    row["original"]["stddev"],
                    row["optimized"]["stddev"],
                    row["speedup"],
                ]
            )


def draw_text_page(pdf, title: str, lines: list[str]):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    fig.text(0.07, 0.96, title, fontsize=15, fontweight="bold", va="top")
    y = 0.91
    for line in lines:
        if y < 0.05:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            plt.axis("off")
            y = 0.96
        is_heading = line.endswith(":")
        fig.text(
            0.08,
            y,
            line,
            fontsize=10.2,
            fontweight="bold" if is_heading else "normal",
            va="top",
        )
        y -= 0.034 if line else 0.018
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_table_page(pdf, title: str, rows: list[dict[str, object]]):
    columns = [
        "Workload",
        "Case",
        "Qubits",
        "Depth",
        "Orig mean ms",
        "Opt mean ms",
        "Speedup",
        "Opt std ms",
    ]
    per_page = 17
    for start in range(0, len(rows), per_page):
        chunk = rows[start : start + per_page]
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
        data = [
            [
                row["workload"],
                row["name"],
                row["n_qubits"],
                row["depth"],
                fmt_ms(row["original"]["mean"]),
                fmt_ms(row["optimized"]["mean"]),
                f"{row['speedup']:.3f}x",
                fmt_ms(row["optimized"]["stddev"]),
            ]
            for row in chunk
        ]
        table = ax.table(
            cellText=data,
            colLabels=columns,
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.4)
        table.scale(1, 1.48)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def draw_runtime_plot(pdf, results: list[dict[str, object]]):
    representative = [
        row
        for row in results
        if row["workload"] in ("Random single-qubit", "Mixed", "Large stress")
    ]
    labels = [row["name"].replace(" gates, ", "\n") for row in representative]
    x = np.arange(len(labels))
    original = [row["original"]["mean"] * 1000 for row in representative]
    optimized = [row["optimized"]["mean"] * 1000 for row in representative]

    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    width = 0.38
    ax.bar(x - width / 2, original, width, label="Original")
    ax.bar(x + width / 2, optimized, width, label="Optimized pair-wise")
    ax.set_yscale("log")
    ax.set_ylabel("Mean runtime (ms, log scale)")
    ax.set_title("Runtime comparison for representative workloads")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_speedup_plot(pdf, results: list[dict[str, object]]):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    for workload in ("Random single-qubit", "Mixed"):
        rows = [row for row in results if row["workload"] == workload]
        rows.sort(key=lambda row: row["n_qubits"])
        ax.plot(
            [row["n_qubits"] for row in rows],
            [row["speedup"] for row in rows],
            marker="o",
            label=workload,
        )
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", label="break-even")
    ax.set_xlabel("Qubit count")
    ax.set_ylabel("Speedup (Original / Optimized)")
    ax.set_title("Speedup versus qubit count")
    ax.legend()
    ax.grid(alpha=0.25)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_report(
    results: list[dict[str, object]],
    profile: dict[str, object],
    output: Path,
):
    info = machine_info()
    speedups = [float(row["speedup"]) for row in results]
    faster = [row for row in results if float(row["speedup"]) > 1.0]
    materially_faster = [row for row in results if float(row["speedup"]) >= 1.10]
    random_rows = [row for row in results if row["workload"] == "Random single-qubit"]
    mixed_rows = [row for row in results if row["workload"] == "Mixed"]
    large_rows = [row for row in results if row["workload"] == "Large stress"]
    recommendation = (
        "Submit as a PR only if the goal is code clarity or a foundation for future "
        "vectorized kernels; based on these measurements, the performance case alone "
        "is not strong enough."
        if len(materially_faster) <= len(results) // 2
        else "The optimization shows enough broad performance improvement to justify a PR."
    )

    with PdfPages(output) as pdf:
        intro = [
            "Benchmark methodology:",
            "Original path: every gate uses gate_spec(...) followed by _apply_unitary(...).",
            "Optimized path: H, X, Y, Z, S, RX, RY, and RZ use the current pair-wise traversal kernel.",
            "Each benchmark uses the same Circuit object for both simulators.",
            "Before timing, each case verifies np.allclose(original_statevector, optimized_statevector).",
            "The Bell benchmark also checks the expected (|00> + |11>) / sqrt(2) state.",
            f"Warm-up runs per case: {WARMUPS}.",
            f"Measured iterations per case: {ITERATIONS}.",
            "Timer: time.perf_counter().",
            "Random seeds are fixed for reproducibility.",
            "For 15- and 20-qubit workloads, single-qubit targets are limited to middle/high indices so the pair-wise implementation is computationally feasible under 30 iterations.",
            "This is still a fair original-vs-optimized comparison because both simulators execute the same circuits.",
            "",
            "Environment:",
            f"Python version: {info['python']}",
            f"NumPy version: {info['numpy']}",
            f"Platform: {info['platform']}",
            f"CPU: {info['cpu']}",
            "",
            "Correctness verification summary:",
            f"Verified benchmark cases: {len(results)}.",
            "All benchmark cases passed original-vs-optimized statevector equality.",
            "No benchmark timings were recorded for an incorrect case.",
        ]
        draw_text_page(
            pdf,
            "QpiAI Quantum SDK: Pair-Wise Single-Qubit Gate Benchmark",
            intro,
        )

        summary = [
            "Summary statistics:",
            f"Cases faster than original: {len(faster)} / {len(results)}.",
            f"Cases at least 10 percent faster: {len(materially_faster)} / {len(results)}.",
            f"Median speedup: {statistics.median(speedups):.3f}x.",
            f"Mean speedup: {statistics.fmean(speedups):.3f}x.",
            f"Best speedup: {max(speedups):.3f}x.",
            f"Worst speedup: {min(speedups):.3f}x.",
            "",
            "Interpretation:",
            "Speedup above 1.0 means the pair-wise optimized implementation is faster.",
            "Speedup below 1.0 means the original NumPy matrix path is faster.",
            "The pair-wise traversal removes redundant pair visits and avoids per-amplitude bit/flipped calculations.",
            "However, this implementation still allocates a result array per gate and creates offset index arrays inside each stride block.",
            "For low target qubits, many small np.arange/indexing operations create noticeable overhead.",
            "For high target qubits, fewer but larger indexed slices are used, shifting cost toward array allocation and vectorized arithmetic.",
            "The original _apply_unitary path delegates most arithmetic and reshaping work to optimized NumPy internals.",
            "",
            "Final recommendation:",
            recommendation,
            "The improvement is statistically meaningful only for cases where the speedup is consistently above 1.0 and materially beyond run-to-run noise.",
            "The added code complexity is justified only if follow-up work removes allocation/index creation overhead or specializes common targets further.",
        ]
        draw_text_page(pdf, "Analysis and Recommendation", summary)

        draw_table_page(pdf, "Runtime and speedup table", results)
        draw_runtime_plot(pdf, results)
        draw_speedup_plot(pdf, results)

        profile_lines = [
            "Profiling setup:",
            f"Profiled case: {profile['case']}.",
            f"Profiled total Python time: {profile['total_profiled_s']:.6f} seconds.",
            "",
            "Top cumulative-time functions:",
        ]
        for entry in profile["top_entries"]:
            profile_lines.append(
                f"{entry['function']}: calls={entry['calls']}, "
                f"total={entry['total_s']:.6f}s, cumulative={entry['cumulative_s']:.6f}s"
            )
        profile_lines.extend(
            [
                "",
                "Profiling interpretation:",
                "The dominant cost is the optimized gate dispatcher/kernel and the NumPy operations it performs inside each gate.",
                "The main bottlenecks are repeated result allocation, offset index construction, advanced indexing gathers/scatters, and arithmetic on temporary arrays.",
                "Dispatcher overhead is present but smaller than per-gate array work for larger statevectors.",
            ]
        )
        draw_text_page(pdf, "Profiling Summary", profile_lines)

        draw_table_page(pdf, "Random-circuit workloads", random_rows)
        draw_table_page(pdf, "Mixed workloads", mixed_rows)
        draw_table_page(pdf, "Large stress workloads", large_rows)


def main():
    output = Path(__file__).with_name("result_speed.pdf")
    cases = build_cases()
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.name}")
        results.append(benchmark_case(case))

    print("[profile] optimized implementation")
    profile = profile_optimized()

    payload = {
        "machine": machine_info(),
        "iterations": ITERATIONS,
        "warmups": WARMUPS,
        "results": results,
        "profile": profile,
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(results, output.with_suffix(".csv"))
    write_report(results, profile, output)
    print(output)


if __name__ == "__main__":
    main()
