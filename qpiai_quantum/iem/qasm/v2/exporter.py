"""
OpenQASM 2.0 Exporter.

This module provides functionality to export a Circuit or Intermediate Circuit Representation
object into OpenQASM 2.0 format.
"""

from qpiai_quantum.circuit.circuit import Circuit
from qpiai_quantum.icr.circuitoperation import CircuitOperation
from typing import List, Dict, Any


class QASM2:
    @staticmethod
    def generate(circuit: Circuit) -> str | list[str]:
        """
        Generates an OpenQASM 2.0 representation of the given circuit.

        Args:
            circuit (Circuit): The quantum circuit to export.

        Returns:
            str: The OpenQASM 2.0 string.
        """
        qasm_lines = []
        qasm_lines.append("OPENQASM 2.0;")
        qasm_lines.append('include "qelib1.inc";')
        qasm_lines.append(f"qreg q[{circuit.icr.num_qubits}];")
        qasm_lines.append(f"creg c[{circuit.icr.num_clbits}];")

        for op in circuit.icr.evolve:
            operation: dict[str, Any] = op.to_json()

            operation["gate_name"] = operation["gate_name"].lower()
            if operation["gate_name"] == "measure":
                qasm_statement = f"measure q[{operation['qubits'][0]}] -> c[{operation['clbits'][0]}];"
            elif operation["gate_name"] == "reset":
                qasm_statement = f"reset q[{operation['qubits'][0]}];"
            elif operation["gate_name"] == "barrier":
                qasm_statement = (
                    "barrier "
                    + ", ".join([f"q[{q}]" for q in operation["qubits"]])
                    + ";"
                )
            else:
                # Handle parametrized gates (e.g., rx, ry, rz, cp, crx, cry, crz)
                params = operation.get("params", [])
                if params:
                    params_str = "(" + ", ".join([str(p) for p in params]) + ")"
                else:
                    params_str = ""

                if len(operation["qubits"]) > 1:
                    qubits_str = ", ".join([f"q[{op}]" for op in operation["qubits"]])
                    qasm_statement = (
                        f"{operation['gate_name']}{params_str} {qubits_str};"
                    )
                else:
                    qasm_statement = f"{operation['gate_name']}{params_str} q[{operation['qubits'][0]}];"

            qasm_lines.append(qasm_statement)

        return "\n".join(qasm_lines)
