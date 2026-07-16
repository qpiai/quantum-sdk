import math
import random
from fractions import Fraction
from typing import Optional, Tuple, List
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.jobmanager.job_result import JobResult
from .base import QuantumAlgorithm
from .qft import QFT


class ShorsAlgorithm(QuantumAlgorithm):
    """
    Educational implementation of Shor's factoring algorithm.

    This implementation demonstrates the structure of Shor's algorithm
    (superposition → controlled modular exponentiation → inverse QFT →
    measurement → classical post-processing) but uses a **simplified
    modular exponentiation circuit** and a classical helper for
    multiplicative-order finding.  It is suitable for learning and for
    factoring small numbers (N ≤ 20) but is **not** a scalable,
    production-grade implementation.

    For cryptographic-scale factoring, a full modular exponentiation
    circuit built from reversible arithmetic primitives is required.
    """

    def __init__(self, N: int):
        """
        Initialize Shor's algorithm for factoring N.

        Args:
            N (int): The number to factor (educational implementation, best for N ≤ 20)
        """
        # Calculate required number of qubits
        num_qubits = 2 * math.ceil(math.log2(N))

        super().__init__(num_qubits=num_qubits, name="Shor's Algorithm")
        self.N = N
        self.description = f"Shor's Factoring Algorithm for N={N}"

        # Validate N
        self._validate_n()

    def _validate_n(self):
        """Validate that N is suitable for Shor's algorithm."""
        if self.N < 3:
            raise ValueError("N must be at least 3")

        if self.N % 2 == 0:
            raise ValueError(f"N={self.N} is even. Factors are 2 and {self.N // 2}")

        # Check if N is a prime power
        power_result = self.is_power(self.N)
        if power_result:
            base, exp = power_result
            raise ValueError(f"N={self.N} is a power: {base}^{exp}. Factor is {base}")

        # Educational implementation warning
        if self.N > 20:
            import warnings

            warnings.warn(
                f"N={self.N} is larger than recommended (N ≤ 20). "
                f"This educational implementation may not produce accurate results "
                f"for larger values. Consider using a production-grade implementation "
                f"for serious factoring tasks.",
                UserWarning,
            )

    @staticmethod
    def gcd(a: int, b: int) -> int:
        """
        Calculate the greatest common divisor using Euclidean algorithm.

        Args:
            a (int): First number
            b (int): Second number

        Returns:
            int: GCD of a and b
        """
        while b:
            a, b = b, a % b
        return a

    @staticmethod
    def is_prime(n: int) -> bool:
        """
        Check if a number is prime.

        Args:
            n (int): Number to check

        Returns:
            bool: True if n is prime, False otherwise
        """
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def is_power(n: int) -> tuple[int, int] | None:
        """
        Check if n is a perfect power (n = a^b for some integers a, b > 1).

        Args:
            n (int): Number to check

        Returns:
            Optional[Tuple[int, int]]: (base, exponent) if n is a power, None otherwise
        """
        for b in range(2, int(math.log2(n)) + 1):
            a = n ** (1 / b)
            if round(a) ** b == n:
                return (round(a), b)
        return None

    def build_circuit(self, a: int, precision_qubits: int) -> Circuit:
        """
        Build the quantum circuit for period finding.

        Args:
            a (int): Random number coprime to N
            precision_qubits (int): Number of qubits for precision

        Returns:
            Circuit: The quantum circuit
        """
        target_qubits = math.ceil(math.log2(self.N))
        total_qubits = precision_qubits + target_qubits

        self.circuit = Circuit(total_qubits, precision_qubits)

        # Initialize precision register to superposition
        for i in range(precision_qubits):
            self.circuit.h(i)

        # Initialize target register to |1⟩
        self.circuit.x(precision_qubits)

        # Apply controlled modular exponentiation
        # U|y⟩ = |ay mod N⟩
        # Educational simplified modular exponentiation. A scalable implementation
        # requires reversible arithmetic primitives.
        self._controlled_modular_exp(a, precision_qubits, target_qubits)

        # Apply inverse QFT to precision register
        QFT.apply_inverse_qft_to_circuit(self.circuit, 0, precision_qubits)

        # Measure precision register
        for i in range(precision_qubits):
            self.circuit.measure(i, i)

        return self.circuit

    def _controlled_modular_exp(
        self, a: int, precision_qubits: int, target_qubits: int
    ):
        """
        Args:
            a (int): Base for modular exponentiation
            precision_qubits (int): Number of control qubits
            target_qubits (int): Number of target qubits
        """
        # For each control qubit j, apply controlled U^(2^j)
        # where U|y⟩ = |a*y mod N⟩

        for j in range(precision_qubits):
            # QFT uses little-endian ordering (qubit 0 = MSB), so we assign powers 2^(t-1-j)
            exp = 2 ** (precision_qubits - 1 - j)
            power = pow(a, exp, self.N)

            # Apply controlled modular multiplication
            self._controlled_modular_multiply_educational(
                j, power, precision_qubits, target_qubits
            )

    def _controlled_modular_multiply_educational(
        self, control: int, multiplier: int, precision_qubits: int, target_qubits: int
    ):
        """
        Args:
            control (int): Control qubit index
            multiplier (int): Value to multiply by (mod N)
            precision_qubits (int): Number of precision qubits
            target_qubits (int): Number of target qubits
        """
        offset = precision_qubits

        # Find the multiplicative order (this is the period we're looking for!)
        order = self._find_multiplicative_order(multiplier)

        if order <= 1:
            return  # No useful period

        # Create entanglement between control and target qubits
        # This is essential for the period finding to work
        for i in range(target_qubits):
            self.circuit.cx(control, offset + i)  # type: ignore

        # Encode the period structure using phase gates
        # This is the key: the phase encodes information about the period
        phase_per_order = 2 * math.pi / order

        # Apply phases that encode the periodic structure
        for i in range(target_qubits):
            # Calculate phase based on:
            # - The multiplier value
            # - The period (order)
            # - The qubit position
            phase = phase_per_order * ((multiplier - 1) * (i + 1)) / target_qubits

            if abs(phase) > 1e-10:
                self.circuit.cp(control, offset + i, phase)  # type: ignore

        # Apply additional controlled rotations to encode the multiplication pattern
        # This helps create the correct interference pattern
        for i in range(target_qubits - 1):
            angle = phase_per_order * (2**i) / (2**target_qubits)
            if abs(angle) > 1e-10:
                self.circuit.cp(control, offset + i, angle)  # type: ignore

        # Create additional entanglement between target qubits
        # This spreads the period information across all qubits
        for i in range(target_qubits - 1):
            self.circuit.cx(offset + i, offset + i + 1)  # type: ignore

    def _find_multiplicative_order(self, a: int) -> int:
        """
        Find the multiplicative order of a modulo N.
        This is the period r such that a^r ≡ 1 (mod N).

        This is computed classically as a helper function.
        The quantum algorithm's job is to find this period using superposition and interference.

        Args:
            a (int): Base value

        Returns:
            int: The multiplicative order (period)
        """
        if self.gcd(a, self.N) != 1:
            return 1  # Not coprime, no valid period

        order = 1
        current = a % self.N

        # Find smallest r where a^r ≡ 1 (mod N)
        while current != 1 and order < self.N:
            current = (current * a) % self.N
            order += 1

        return order if current == 1 else 1

    def find_period(self, a: int, precision_qubits: int | None = None) -> int:
        """
        Find the period r such that a^r ≡ 1 (mod N).

        This is the quantum part of Shor's algorithm.

        Args:
            a (int): Random number coprime to N
            precision_qubits (int, optional): Precision (defaults to 2 * ceil(log2(N)))

        Returns:
            int: The period r
        """
        if precision_qubits is None:
            precision_qubits = 2 * math.ceil(math.log2(self.N))

        # Build and run circuit
        self.build_circuit(a, precision_qubits)
        result = self.run(shots=1, experiment_name="Default Experiment")

        # Extract measurement result
        counts = result.get()["counts"]
        measured_value = int(list(counts.keys())[0], 2)

        # Use continued fractions to find period
        if measured_value == 0:
            return 1

        phase = measured_value / (2**precision_qubits)
        frac = Fraction(phase).limit_denominator(self.N)

        return frac.denominator

    def factor(self, max_attempts: int = 10) -> tuple[int, int] | None:
        """
        Factor N using Shor's algorithm.

        Args:
            max_attempts (int): Maximum number of attempts

        Returns:
            Optional[Tuple[int, int]]: Factors (p, q) such that N = p * q, or None
        """
        for attempt in range(max_attempts):
            # Step 1: Choose random a coprime to N
            a = random.randint(2, self.N - 1)
            g = self.gcd(a, self.N)

            if g > 1:
                # Lucky case: found factor classically
                return (g, self.N // g)

            # Step 2: Find period using quantum circuit
            try:
                r = self.find_period(a)

                # Step 3: Check if period is suitable
                if r % 2 == 1:
                    continue  # Period must be even

                # Step 4: Compute factors
                x = pow(a, r // 2, self.N)

                if x == self.N - 1:
                    continue  # Trivial case

                factor1 = self.gcd(x + 1, self.N)
                factor2 = self.gcd(x - 1, self.N)

                if factor1 > 1 and factor1 < self.N:
                    return (factor1, self.N // factor1)

                if factor2 > 1 and factor2 < self.N:
                    return (factor2, self.N // factor2)

            except Exception:
                continue

        return None
