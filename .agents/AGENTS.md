# QpiAI Quantum SDK - Agent Context

This repository contains the **QpiAI Quantum SDK**, a comprehensive Python-based quantum computing framework providing modular implementations of quantum circuits, simulation backends, and quantum algorithms.

## Project Structure

- `qpiai_quantum/`: The core source code directory.
  - `circuit/`: Quantum circuit representation and gate definitions.
  - `simulator/`: Simulation backend (Statevector).
  - `algorithms/`: Implementations of quantum algorithms (Grover, QFT, Shor, etc.).
  - `authentication/`: Cloud authentication and API key management.
- `tests/`: Unit tests using `pytest`.
- `sdk_notebooks/`: Jupyter notebooks providing tutorials and examples for end-users.

## Technology Stack

- **Language**: Python 3.10+
- **Core Dependencies**: `numpy`, `scipy`, `matplotlib`, `plotly`, `networkx`, `pandas`, `requests`
- **Linting & Formatting**: Configured via `ruff` in `pyproject.toml` (target version py310).
- **Type Checking**: Static typing with `mypy` is heavily used across the codebase.

## Development Guidelines for AI Agents

1. **Type Annotations**: Always include accurate type hints for function arguments and return types. The project uses static type checking, and new code must be compatible with existing type definitions.
2. **Testing**: Write or update tests in the `tests/` directory when adding new features or fixing bugs. Use `pytest` for testing.
3. **Documentation**: Ensure docstrings are provided for public classes and methods. If modifying public APIs, consider if updates to `sdk_notebooks` are necessary.
4. **Code Quality**: Keep code modular. Respect the existing formatting rules as defined in `pyproject.toml`.
5. **Core Abstractions**: Be cautious when modifying core abstractions like `Circuit`, `CircuitOperation`, or simulation backends. Avoid breaking changes as many algorithms depend on these foundational classes.
6. **Commit Message Conventions**: Always write commit messages that adhere to the Conventional Commits specification (e.g., `feat:`, `fix:`, `refactor:`, `docs:`, `test:`). Clearly describe the scope, nature of the changes, and reference any relevant issue numbers (e.g., `Closes #123`).

## Key Abstractions

- `Circuit`: The central class for building quantum circuits.
- `Statevector`: Class for quantum state representation and manipulation.
- `Backend`: Abstraction for executing circuits, including local simulators and cloud endpoints.
