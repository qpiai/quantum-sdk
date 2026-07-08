from typing import Dict, Any, List, Tuple
import numpy as np
import networkx as nx  # type: ignore[import-untyped]
from itertools import combinations
from .base import OptimizationProblem


class MaxCutProblem(OptimizationProblem):
    def __init__(self, graph: nx.Graph):
        super().__init__()
        self.graph = graph
        self.n_qubits = len(graph.nodes)

    def _construct_hamiltonian(self) -> Any:
        """Construct MaxCut Hamiltonian."""
        return self._get_hamiltonian_terms()

    def _get_hamiltonian_terms(self) -> List[Tuple[List[Tuple[int, str]], float]]:
        """Get MaxCut Hamiltonian terms in library-agnostic format."""
        terms = []

        for i, j in self.graph.edges:
            terms.append(([(i, "Z"), (j, "Z")], 0.5))

        terms.append(([], 0.5 * len(self.graph.edges)))

        return terms

    # NOTE: To Extract this Hamiltonian for USERS
    def get_hamiltonian_terms(self) -> List[Tuple[List[Tuple[int, str]], float]]:
        return self._get_hamiltonian_terms()

    def decode_solution(self, bitstring: str) -> Dict[str, Any]:
        """Decode bitstring into MaxCut partitions."""
        partition_0 = []
        partition_1 = []

        for i, bit in enumerate(bitstring[::-1]):
            if bit == "0":
                partition_0.append(i)
            else:
                partition_1.append(i)

        return {"partition_0": partition_0, "partition_1": partition_1}

    def validate_solution(self, solution: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate MaxCut solution."""
        if "partition_0" not in solution or "partition_1" not in solution:
            return False, "Solution missing partitions"

        all_nodes = set(solution["partition_0"]) | set(solution["partition_1"])
        if all_nodes != set(self.graph.nodes):
            return False, "Not all nodes are assigned to partitions"

        if set(solution["partition_0"]) & set(solution["partition_1"]):
            return False, "Partitions overlap"

        return True, "Valid MaxCut solution"

    def compute_solution_quality(self, solution: Dict[str, Any]) -> Dict[str, float]:
        """Compute comprehensive MaxCut quality metrics."""
        if not self.validate_solution(solution)[0]:
            return {"cut_value": 0.0, "is_valid": 0.0}

        cut_value = 0
        for i, j in self.graph.edges:
            is_cut = (
                i in solution["partition_0"] and j in solution["partition_1"]
            ) or (i in solution["partition_1"] and j in solution["partition_0"])
            if is_cut:
                cut_value += 1

        return {
            "cut_value": float(cut_value),
            "is_valid": 1.0,
            "cut_ratio": cut_value / len(self.graph.edges),
            "balance_ratio": min(
                len(solution["partition_0"]), len(solution["partition_1"])
            )
            / (len(self.graph.nodes) / 2),
        }


class PortfolioOptimizationProblem(OptimizationProblem):
    def __init__(
        self,
        returns: np.ndarray,
        risk_matrix: np.ndarray,
        budget: float = 1.0,
        risk_tolerance: float = 1.0,
    ):
        super().__init__()

        if (
            len(returns) != risk_matrix.shape[0]
            or risk_matrix.shape[0] != risk_matrix.shape[1]
        ):
            raise ValueError("Inconsistent dimensions between returns and risk matrix")

        self.returns = returns
        self.risk_matrix = risk_matrix
        self.budget = budget
        self.risk_tolerance = risk_tolerance
        self.n_assets = len(returns)
        self.n_qubits = self.n_assets

    def _construct_hamiltonian(self) -> Any:
        """Construct Portfolio Optimization Hamiltonian."""
        return self._get_hamiltonian_terms()

    def _get_hamiltonian_terms(self) -> List[Tuple[List[Tuple[int, str]], float]]:
        """Get Portfolio Optimization Hamiltonian terms in library-agnostic format."""
        terms = []

        A = 1.0
        for i in range(self.n_assets):
            terms.append(([(i, "Z")], A * self.returns[i] / 2))
            terms.append(([], -A * self.returns[i] / 2))

        B = self.risk_tolerance
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if i == j:
                    terms.append(([(i, "Z")], -B * self.risk_matrix[i, i] / 4))
                    terms.append(([], B * self.risk_matrix[i, i] / 4))
                else:
                    terms.append(([(i, "Z"), (j, "Z")], B * self.risk_matrix[i, j] / 4))
                    terms.append(([(i, "Z")], -B * self.risk_matrix[i, j] / 4))
                    terms.append(([(j, "Z")], -B * self.risk_matrix[i, j] / 4))
                    terms.append(([], B * self.risk_matrix[i, j] / 4))

        C = 2.0
        target_sum = 2 * self.budget - self.n_assets

        for i, j in combinations(range(self.n_assets), 2):
            terms.append(([(i, "Z"), (j, "Z")], C))

        for i in range(self.n_assets):
            terms.append(([(i, "Z")], -C * target_sum))

        terms.append(([], C * target_sum * target_sum / 2))

        return terms

    # NOTE: To Extract this Hamiltonian for USERS
    def get_hamiltonian_terms(self) -> List[Tuple[List[Tuple[int, str]], float]]:
        """Public method to get Hamiltonian terms for VQE."""
        return self._get_hamiltonian_terms()

    def decode_solution(self, bitstring: str) -> Dict[str, Any]:
        """Decode bitstring into portfolio allocation."""
        allocation = np.array([int(b) for b in bitstring[::-1]])

        if sum(allocation) > 0:
            allocation = allocation * (self.budget / sum(allocation))

        return {
            "allocation": allocation,
            "selected_assets": np.where(allocation > 0)[0].tolist(),
        }

    def validate_solution(self, solution: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate portfolio solution."""
        if "allocation" not in solution:
            return False, "Solution missing allocation"

        allocation = solution["allocation"]

        if len(allocation) != self.n_assets:
            return (
                False,
                f"Allocation length {len(allocation)} != number of assets {self.n_assets}",
            )

        total_allocation = sum(allocation)
        if not np.isclose(total_allocation, self.budget, rtol=1e-5):
            return (
                False,
                f"Budget constraint violated: {total_allocation} != {self.budget}",
            )

        if any(a < 0 for a in allocation):
            return False, "Negative allocations not allowed"

        return True, "Valid portfolio allocation"

    def compute_solution_quality(self, solution: Dict[str, Any]) -> Dict[str, float]:
        """Compute comprehensive portfolio quality metrics."""
        if not self.validate_solution(solution)[0]:
            return {
                "expected_return": 0.0,
                "portfolio_risk": float("inf"),
                "is_valid": 0.0,
            }

        allocation = solution["allocation"]

        expected_return = np.dot(allocation, self.returns)
        portfolio_risk = np.sqrt(
            np.dot(allocation, np.dot(self.risk_matrix, allocation))
        )

        risk_free_rate = 0.0
        sharpe_ratio = (
            (expected_return - risk_free_rate) / portfolio_risk
            if portfolio_risk > 0
            else 0
        )

        return {
            "expected_return": float(expected_return),
            "portfolio_risk": float(portfolio_risk),
            "is_valid": 1.0,
            "sharpe_ratio": float(sharpe_ratio),
            "diversification_ratio": float(1 - sum(a**2 for a in allocation)),
        }
