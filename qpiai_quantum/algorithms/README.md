# QpiAI Quantum Algorithms Module

A comprehensive collection of modular quantum algorithm implementations built on the QpiAI Quantum SDK.

## Available Algorithms

### 1. Quantum Fourier Transform (QFT)
Quantum analog of the classical discrete Fourier transform.

```python
from qpiai_quantum.algorithms import QFT

# Create QFT instance
qft = QFT(num_qubits=3)

# Build circuit
circuit = qft.build_circuit(initialize_superposition=True)

# Run on quantum simulator
result = qft.run(shots=1024)
print(result.get()["counts"])

# Or use inverse QFT
iqft = QFT(num_qubits=3, inverse=True)
```

### 2. Grover's Search Algorithm
Quadratic speedup for searching unstructured databases.

```python
from qpiai_quantum.algorithms import GroverSearch

# Search for target "101" in 3-qubit space
grover = GroverSearch(num_qubits=3, target="101")

# Build and run
result = grover.run(shots=1024)
print(result.get()["counts"])

# Check success probability
prob = grover.get_success_probability()
print(f"Success probability: {prob:.2%}")
```

### 3. Shor's Factoring Algorithm
Educational implementation demonstrating quantum period-finding for integer factorization.
Best suited for small numbers (N ≤ 20); uses a simplified modular exponentiation circuit.

```python
from qpiai_quantum.algorithms import ShorsAlgorithm

# Factor the number 15
shor = ShorsAlgorithm(N=15)

# Find factors
factors = shor.factor()
print(f"Factors of 15: {factors}")  # Output: (3, 5) or (5, 3)
```

### 4. Simon's Algorithm
Exponential speedup for finding hidden bitstrings.

```python
from qpiai_quantum.algorithms import SimonAlgorithm

# Find hidden bitstring
simon = SimonAlgorithm(num_qubits=3, hidden_string="110")

# Run algorithm
result = simon.run(shots=1024)
print(result.get()["counts"])

# Or let it find the hidden string automatically
hidden = simon.find_hidden_string()
print(f"Hidden string: {hidden}")
```

### 5. Bernstein-Vazirani Algorithm
Determine a hidden bitstring in a single query.

```python
from qpiai_quantum.algorithms import BernsteinVazirani

# Find hidden bitstring "1011"
bv = BernsteinVazirani(num_qubits=4, hidden_string="1011")

# Build and run
result = bv.run(shots=1024)
print(result.get()["counts"])  # Should show "1011" with ~100% probability

# Or use the convenience method
hidden = bv.find_hidden_string(shots=1024)
print(f"Hidden string: {hidden}")  # Output: "1011"

# Check theoretical result
theory = bv.get_theoretical_result()
print(f"Speedup: {theory['speedup']}")
```

### 6. Deutsch-Jozsa Algorithm
Determine whether a boolean function is constant or balanced in a single query.

```python
from qpiai_quantum.algorithms import DeutschJozsa

# Test a balanced oracle on 3 qubits
dj = DeutschJozsa(num_qubits=3, oracle_type="balanced")

# Build and run
result = dj.run(shots=1024)
print(result.get()["counts"])  # Should show non-zero bitstring

# Interpret result
print(DeutschJozsa.interpret_result(result))  # Output: "balanced"

# Or use the convenience method
result_type = dj.determine_function_type(shots=1024)
print(f"Function is: {result_type}")  # Output: "balanced"

# Check theoretical result
theory = dj.get_theoretical_result()
print(f"Speedup: {theory['speedup']}")
```

### 7. Quantum Phase Estimation
Estimate phase of a unitary operator's eigenvalue.

```python
from qpiai_quantum.algorithms import QuantumPhaseEstimation

# Estimate phase for T gate (phase = 1/8)
qpe = QuantumPhaseEstimation(precision_qubits=4, eigenstate_qubits=1)

# Run estimation
phase = qpe.estimate_phase(unitary='T')
print(f"Estimated phase: {phase}")  # Should be close to 0.125 (1/8)

# Compare with theoretical value
theoretical = qpe.get_theoretical_phase('T')
print(f"Theoretical phase: {theoretical}")
```

### 8. Quantum Random Number Generator (QRNG)
Generate random numbers via Hadamard sampling.  Output is true quantum
random when executed on QPU hardware; on the local simulator, sampling
uses NumPy's pseudo-random number generator.

```python
from qpiai_quantum.algorithms import QRNG

# Create QRNG for 8-bit random numbers (0–255)
rng = QRNG(n_bits=8)

# Generate a single random integer
value = rng.generate()
print(f"Random int: {value}")

# Different output formats
bits = rng.generate(output_format="bitstring")   # e.g. "10110011"
raw  = rng.generate(output_format="bytes")       # e.g. b'\xb3'

# Batch generation (single backend call)
values = rng.generate_batch(count=10)
print(f"10 random values: {values}")
```

### 9. Amplitude Estimation
Quantum speedup for Monte Carlo estimation.  The SDK provides the **iterative
(maximum-likelihood) variant**; the canonical QPE-based variant is planned but
not yet implemented.

```python
from qpiai_quantum.algorithms.amplitude_estimation import (
    IterativeAmplitudeEstimation,
    EstimationProblem,
)
from qpiai_quantum.circuit import Circuit

# Prepare a state whose amplitude we want to estimate
A = Circuit(2)
A.ry(0, 0.6)   # rotate qubit 0
A.cx(0, 1)      # entangle

# Define the estimation problem
problem = EstimationProblem(
    state_preparation=A,
    objective_qubits=[1],
)

# Run iterative amplitude estimation
iae = IterativeAmplitudeEstimation(epsilon_target=0.01, alpha=0.05)
amplitude = iae.estimate(problem, shots=2000)
print(f"Estimated amplitude: {amplitude:.4f}")
```

> **Note:** The canonical QPE-based `AmplitudeEstimation` class is not yet
> available and raises `NotImplementedError`.  Use
> `IterativeAmplitudeEstimation` for currently supported amplitude estimation
> workflows.

## Common Operations

All algorithms inherit from the base `QuantumAlgorithm` class and support:

### Building Circuits
```python
from qpiai_quantum.algorithms import QFT

qft = QFT(num_qubits=3)
circuit = qft.build_circuit()
```

### Visualization
```python
# 1. Circuit diagram (default)
qft.visualize()
qft.visualize(plot='circuit')

# 2. Measurement histogram
qft.visualize(plot='histogram', shots=1024)

# 3. Probability distribution
qft.visualize(plot='probabilities', shots=1024)

# 4. State vector amplitudes
qft.visualize(plot='state', shots=1024)

# 5. Custom styling
qft.visualize(
    plot='histogram',
    shots=2048,
    figsize=(12, 6),
    title='My Custom Title'
)

# 6. Export to OpenQASM
qasm_code = qft.to_qasm()
print(qasm_code)
```

### Execution
```python
from qpiai_quantum.executor import Backend

# Execute on specific backend
result = qft.execute(
    shots=1024,
    backend=Backend.STATEVECTOR_SIMULATOR_CPU
)

# Or use the convenient run() method
result = qft.run(shots=1024)
```

### Algorithm Information
```python
# Get algorithm details
info = qft.get_info()
print(info)
# Output: {
#     'name': 'QFT',
#     'num_qubits': 3,
#     'description': '...',
#     'circuit_built': True
# }
```

## Advanced Usage

### Embedding Algorithms as Subroutines

```python
from qpiai_quantum import Circuit
from qpiai_quantum.algorithms import QFT

# Create a larger circuit
circuit = Circuit(6, 6)

# Add some gates
circuit.h(0)
circuit.cx(0, 1)

# Embed QFT on qubits 2-4 (3 qubits)
QFT.apply_qft_to_circuit(circuit, start=2, n=3)

# Continue building circuit
circuit.cx(4, 5)
```

### Custom Backend Configuration

```python
from qpiai_quantum.executor import Backend
from qpiai_quantum.algorithms import GroverSearch

grover = GroverSearch(num_qubits=3, target="101")

# Use different backends
result_cpu = grover.run(backend=Backend.STATEVECTOR_SIMULATOR_CPU)
result_gpu = grover.run(backend=Backend.STATEVECTOR_SIMULATOR_GPU)
```

### Batch Execution

```python
from qpiai_quantum.algorithms import QFT

# Run multiple sizes
for n in [2, 3, 4, 5]:
    qft = QFT(num_qubits=n)
    result = qft.run(shots=1024)
    print(f"QFT-{n}: {result.get()['counts']}")
```

## Algorithm Descriptions

### QFT (Quantum Fourier Transform)
- **Complexity**: O(n²) gates for n qubits
- **Use Cases**: Phase estimation, Shor's algorithm, quantum simulation
- **Key Feature**: Basis transformation from computational to Fourier basis

### Grover's Search
- **Complexity**: O(√N) queries vs O(N) classical
- **Use Cases**: Database search, solving SAT problems, optimization
- **Key Feature**: Quadratic speedup via amplitude amplification

### Shor's Algorithm (Educational)
- **Complexity**: Polynomial time vs exponential classical
- **Use Cases**: Integer factorization, cryptanalysis
- **Key Feature**: Combines period finding (quantum) with classical number theory
- **Note**: This SDK provides an educational implementation with a simplified modular exponentiation circuit. It is suitable for learning and small N (≤ 20).

### Simon's Algorithm
- **Complexity**: O(n) queries vs O(2^(n/2)) classical
- **Use Cases**: Finding hidden structures, cryptanalysis
- **Key Feature**: Exponential speedup for hidden bitstring problems

### Bernstein-Vazirani Algorithm
- **Complexity**: 1 query vs n classical queries
- **Use Cases**: Hidden linear function recovery, oracle identification
- **Key Feature**: Deterministic single-query extraction via phase kickback

### Deutsch-Jozsa Algorithm
- **Complexity**: 1 query vs 2^(n-1)+1 deterministic classical queries
- **Use Cases**: Function classification, oracle identification, quantum advantage demonstration
- **Key Feature**: Exponential speedup for distinguishing constant vs balanced functions

### Quantum Phase Estimation
- **Complexity**: O(n) gates for n-bit precision
- **Use Cases**: Eigenvalue problems, quantum chemistry, machine learning
- **Key Feature**: Fundamental subroutine in many quantum algorithms

### QRNG (Quantum Random Number Generator)
- **Complexity**: O(n) gates for n random bits
- **Use Cases**: Cryptographic key generation, Monte Carlo sampling, gaming
- **Key Feature**: Randomness from quantum measurement (true randomness on QPU hardware; pseudo-random on simulator)

### Amplitude Estimation
- **Complexity**: O(1/ε) queries vs O(1/ε²) classical Monte Carlo
- **Use Cases**: Option pricing, risk analysis, counting problems
- **Key Feature**: Quadratic speedup for estimating expectation values
- **Note**: Only the iterative (maximum-likelihood) variant is currently implemented. The canonical QPE-based variant is planned.

## Performance Tips

1. **Choose appropriate precision**: More qubits = higher precision but slower execution
2. **Use optimal iterations**: Grover's algorithm has an optimal number of iterations
3. **Batch similar runs**: Reuse circuit construction when possible
4. **Select appropriate backend**: GPU backends for larger circuits

## Contributing

To add a new algorithm:

1. Create a new file in `qpiai_quantum/algorithms/`
2. Inherit from `QuantumAlgorithm` base class
3. Implement `build_circuit()` and `run()` methods
4. Add to `__init__.py` exports
5. Update this README

## References

- Nielsen & Chuang: "Quantum Computation and Quantum Information"
- QpiAI Quantum SDK Documentation
- Qiskit Textbook

## License

Part of the QpiAI Quantum SDK

