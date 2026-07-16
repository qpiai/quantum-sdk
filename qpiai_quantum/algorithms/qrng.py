"""
Quantum Random Number Generator (QRNG)

Generates random numbers by exploiting quantum mechanical uncertainty.
Each qubit is placed in an equal superposition via a Hadamard gate, so
measurement yields 0 or 1 with exactly 50 % probability.

When executed on real quantum hardware (QPU), the output is fundamentally
unpredictable — true quantum randomness.  On the local statevector
simulator the sampling step uses NumPy's pseudo-random number generator,
so the output is *not* genuinely random in the information-theoretic sense.

"""

from typing import Any, Dict, List, Optional, Union
from qpiai_quantum.results.base_result import BaseQuantumResult
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager.job_result import JobResult
from .base import QuantumAlgorithm


class QRNG(QuantumAlgorithm):
    """
    Quantum Random Number Generator.

    Uses Hadamard sampling to generate random numbers.

    When executed on quantum hardware, produces hardware-certified true
    random numbers.  On the local simulator, output is pseudo-random
    (sampled via NumPy from the ideal probability distribution).

    Each call to ``generate()`` builds a circuit of ``n_bits`` qubits,
    applies H gates, measures, and converts the bitstring to the desired
    output format.

    Args:
        n_bits (int): Number of random bits per sample (= number of qubits).
            Must be ≥ 1. Default: 8 (one random byte).

    Example:
        >>> from qpiai_quantum.algorithms import QRNG
        >>> rng = QRNG(n_bits=8)
        >>> rng.generate()  # returns a random int 0–255
        >>> rng.generate(output_format="bitstring")  # e.g. "10110011"
        >>> rng.generate(output_format="bytes")  # e.g. b'\\xb3'
        >>> rng.generate_batch(count=5)  # [42, 179, 3, 200, 91]
    """

    VALID_FORMATS = ("int", "bytes", "bitstring")

    def __init__(self, n_bits: int = 8):
        if not isinstance(n_bits, int) or n_bits < 1:
            raise ValueError(f"n_bits must be a positive integer, got {n_bits!r}")

        super().__init__(num_qubits=n_bits, name="QRNG")
        self.n_bits = n_bits
        self.description = (
            "Quantum Random Number Generator — generates random numbers "
            "via Hadamard sampling (true randomness on QPU hardware; "
            "pseudo-random on local simulator)"
        )

    # ------------------------------------------------------------------ #
    #  Circuit construction                                                #
    # ------------------------------------------------------------------ #

    def build_circuit(self, **kwargs) -> Circuit:
        """
        Build the QRNG circuit.

        The circuit applies a Hadamard gate to every qubit, creating an
        equal superposition of all 2^n_bits basis states, and then measures
        each qubit.

        Returns:
            Circuit: The constructed QRNG circuit.
        """
        self.circuit = Circuit(self.n_bits, self.n_bits)

        # Apply Hadamard to every qubit → equal superposition
        for i in range(self.n_bits):
            self.circuit.h(i)

        # Measure every qubit
        for i in range(self.n_bits):
            self.circuit.measure(i, i)

        return self.circuit

    # ------------------------------------------------------------------ #
    #  Output conversion                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_format(output_format: str) -> None:
        """Raise ValueError if *output_format* is not recognised."""
        if output_format not in QRNG.VALID_FORMATS:
            raise ValueError(
                f"Unknown output_format {output_format!r}. "
                f"Choose from {QRNG.VALID_FORMATS}"
            )

    def _convert_output(self, bitstring: str, output_format: str) -> int | bytes | str:
        """
        Convert a raw measurement bitstring to the requested format.

        Args:
            bitstring: Binary string, e.g. ``"10110011"``.
            output_format: One of ``"int"``, ``"bytes"``, ``"bitstring"``.

        Returns:
            The converted value.
        """
        if output_format == "bitstring":
            return bitstring
        value = int(bitstring, 2)
        if output_format == "int":
            return value
        # output_format == "bytes"
        byte_length = (self.n_bits + 7) // 8
        return value.to_bytes(byte_length, byteorder="big")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        shots: int = 1,
        output_format: str = "int",
    ) -> int | bytes | str | list:
        """
        Generate quantum random number(s).

        Args:
            shots (int): How many random values to produce. If ``shots=1``
                a single value is returned; otherwise a list is returned.
            output_format (str): ``"int"`` (default), ``"bytes"``, or
                ``"bitstring"``.

        Returns:
            A single value when *shots=1*, or a list of values otherwise.
        """
        self._validate_format(output_format)

        # Build circuit if not already built
        if self.circuit is None:
            self.build_circuit()

        result: BaseQuantumResult = self.run(
            shots=shots,
            experiment_name="Default Experiment",
        )
        counts = result.get()["counts"]

        # Expand the counts dict into individual samples
        samples: list = []
        for bitstring, count in counts.items():
            converted = self._convert_output(bitstring, output_format)
            samples.extend([converted] * count)

        if shots == 1:
            return samples[0]
        return samples

    def generate_batch(
        self,
        count: int,
        output_format: str = "int",
    ) -> list[int | bytes | str]:
        """
        Generate a batch of quantum random values in a single backend call.

        This is a convenience wrapper around ``generate(shots=count, ...)``.

        Args:
            count (int): Number of random values to produce.
            output_format (str): ``"int"`` (default), ``"bytes"``, or
                ``"bitstring"``.

        Returns:
            list: A list of *count* random values.
        """
        result = self.generate(shots=count, output_format=output_format)
        if count == 1:
            return [result]  # type: ignore
        return result  # type: ignore

    # ------------------------------------------------------------------ #
    #  Info helpers                                                         #
    # ------------------------------------------------------------------ #

    def get_info(self):
        """Return metadata about this QRNG instance."""
        info = super().get_info()
        info.update(
            {
                "n_bits": self.n_bits,
                "max_value": 2**self.n_bits - 1,
                "output_bytes": (self.n_bits + 7) // 8,
            }
        )
        return info

    def __repr__(self) -> str:
        return (
            f"QRNG(n_bits={self.n_bits}, "
            f"max_value={2**self.n_bits - 1}, "
            f"circuit_built={self.circuit is not None})"
        )
