import unittest
import warnings
import networkx as nx
import numpy as np
from qpiai_quantum.applications.optimization import (
    MaxCutProblem,
    PortfolioOptimizationProblem,
)


class TestOptimizationMigration(unittest.TestCase):
    def test_direct_imports_do_not_warn(self):
        """Verify that importing and instantiating from the new applications namespace does not raise warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # 1. MaxCut
            g = nx.Graph()
            g.add_edge(0, 1)
            problem_mc = MaxCutProblem(g)
            self.assertEqual(problem_mc.n_qubits, 2)

            # 2. Portfolio
            returns = np.array([0.1, 0.2])
            risk = np.array([[0.05, 0.01], [0.01, 0.08]])
            problem_pf = PortfolioOptimizationProblem(returns, risk)
            self.assertEqual(problem_pf.n_qubits, 2)

            # Confirm no deprecation warnings were raised
            dep_warnings = [
                warn for warn in w if issubclass(warn.category, DeprecationWarning)
            ]
            self.assertEqual(len(dep_warnings), 0)

    def test_deprecated_imports_warn(self):
        """Verify that importing from the old algorithms namespace raises a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Accessing from the deprecated module namespace
            import qpiai_quantum.algorithms.opt.problems.optimization as old_opt

            # Instantiate MaxCutProblem via the old module
            g = nx.Graph()
            g.add_edge(0, 1)
            mc_instance = old_opt.MaxCutProblem(g)
            self.assertIsInstance(mc_instance, MaxCutProblem)

            # Verify deprecation warning was captured
            dep_warnings = [
                warn
                for warn in w
                if issubclass(warn.category, DeprecationWarning)
                and "deprecated" in str(warn.message)
            ]
            self.assertGreaterEqual(len(dep_warnings), 1)

            # Verify warning message content
            warning_msg = str(dep_warnings[0].message)
            self.assertIn(
                "Importing MaxCutProblem from `qpiai_quantum.algorithms.opt.problems` is deprecated",
                warning_msg,
            )
