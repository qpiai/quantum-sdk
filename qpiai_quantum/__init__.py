"""
QpiAI Quantum SDK

A comprehensive Python-based quantum computing framework providing
modular implementations of quantum algorithms and optimization solvers.

Copyright (c) 2026 QpiAI
License: Apache-2.0
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("qpiai-quantum")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"
__author__ = "QpiAI"
__email__ = "support@qcloud.qpiai.tech"
__license__ = "Apache-2.0"
__description__ = (
    "A comprehensive quantum computing framework for research, "
    "education, and quantum application development"
)

# ============================================================================
# Core imports
# ============================================================================
from .authentication import QpiAIQuantumAuth
from .circuit import Circuit, ClassicalRegister, QuantumRegister
from .exceptions import BaseError
from .jobmanager import Backend, JobManager, JobResult

# Quantum algorithms
from .algorithms import (
    BernsteinVazirani,
    DeutschJozsa,
    QFT,
    QRNG,
    GroverSearch,
    ShorsAlgorithm,
    SimonAlgorithm,
    phase_estimation,
)

# State preparation
from .state_preparation import (
    BellStateGenerator,
    ClusterStateGenerator,
    GHZStateGenerator,
    WStateGenerator,
)

# Optimization algorithms
from .algorithms.opt.solvers import VQESolver

# Quantum information
from .quantum_info import DensityMatrix, Statevector

# Applications
from . import applications

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "Circuit",
    "ClassicalRegister",
    "QuantumRegister",
    "Backend",
    "JobManager",
    "JobResult",
    "QpiAIQuantumAuth",
    "BaseError",
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    # Quantum algorithms
    "BernsteinVazirani",
    "DeutschJozsa",
    "QFT",
    "GroverSearch",
    "ShorsAlgorithm",
    "SimonAlgorithm",
    "phase_estimation",
    "QRNG",
    # State preparation
    "BellStateGenerator",
    "ClusterStateGenerator",
    "GHZStateGenerator",
    "WStateGenerator",
    # Optimization algorithms
    "VQESolver",
    # Quantum information
    "Statevector",
    "DensityMatrix",
    # Applications
    "applications",
]
