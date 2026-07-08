"""
Unit tests for the optimizer wrappers in qpiai_quantum/algorithms/utils/optimizers/.
"""

import unittest
import numpy as np
import sys
import os

# Adjust import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qpiai_quantum.algorithms.utils.optimizers import (
    gradient_descent_optimize,
    adam_optimize,
    adagrad_optimize,
    l_bfgs_b_optimize,
    cg_optimize,
    cobyla_optimize,
    nelder_mead_optimize,
    spsa_optimize,
    slsqp_optimize,
    differential_evolution_optimize,
    genetic_algorithm_optimize,
    particle_swarm_optimize,
    powell_optimize,
)


# Check if the user specifically asked to run the optimizer tests.
# We consider it an explicit run if:
# 1. "test_optimizers" is in the command line arguments (e.g. pytest tests/test_optimizers.py or python tests/test_optimizers.py).
# 2. The environment variable TEST_OPTIMIZERS is set (e.g. TEST_OPTIMIZERS=1 pytest).
explicit_run = (
    any("test_optimizers" in arg for arg in sys.argv)
    or os.environ.get("TEST_OPTIMIZERS") == "1"
)


@unittest.skipUnless(
    explicit_run,
    "Skipping optimizer tests by default. Run with `pytest tests/test_optimizers.py` or set `TEST_OPTIMIZERS=1` to run them.",
)
class TestOptimizers(unittest.TestCase):
    def setUp(self):
        # A simple 2D quadratic function minimized at params = [1.0, 1.0]
        # Minimum value is 0.0
        self.initial_point = np.array([0.0, 0.0])
        self.bounds = [(-2.0, 2.0), (-2.0, 2.0)]

    def objective(self, params):
        return float(np.sum((params - 1.0) ** 2))

    def gradient(self, params):
        return 2.0 * (params - 1.0)

    def _verify_result_structure(self, res, has_nit=True):
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 3)
        opt_params, opt_val, opt_dict = res

        self.assertIsInstance(opt_params, np.ndarray)
        self.assertEqual(opt_params.shape, (2,))
        self.assertIsInstance(opt_val, (float, np.floating))

        self.assertIsInstance(opt_dict, dict)
        self.assertIn("nfev", opt_dict)
        if has_nit:
            self.assertIn("nit", opt_dict)
        self.assertIn("history", opt_dict)
        self.assertIn("param_history", opt_dict)
        self.assertIn("success", opt_dict)

        # Optimization should decrease the objective value from initial point (which is 2.0)
        self.assertLess(opt_val, 2.0)

    def test_gradient_descent(self):
        res = gradient_descent_optimize(
            objective=self.objective,
            gradient=self.gradient,
            initial_point=self.initial_point,
            learning_rate=0.1,
            maxiter=20,
        )
        self._verify_result_structure(res)

    def test_adam(self):
        res = adam_optimize(
            objective=self.objective,
            gradient=self.gradient,
            initial_point=self.initial_point,
            learning_rate=0.2,
            maxiter=30,
        )
        self._verify_result_structure(res)

    def test_adagrad(self):
        res = adagrad_optimize(
            objective=self.objective,
            gradient=self.gradient,
            initial_point=self.initial_point,
            learning_rate=0.5,
            maxiter=30,
        )
        self._verify_result_structure(res)

    def test_l_bfgs_b(self):
        res = l_bfgs_b_optimize(
            objective=self.objective,
            gradient=self.gradient,
            initial_point=self.initial_point,
            bounds=self.bounds,
            maxiter=20,
        )
        self._verify_result_structure(res)

    def test_cg(self):
        res = cg_optimize(
            objective=self.objective,
            gradient=self.gradient,
            initial_point=self.initial_point,
            maxiter=20,
        )
        self._verify_result_structure(res)

    def test_cobyla(self):
        res = cobyla_optimize(
            objective=self.objective,
            initial_point=self.initial_point,
            bounds=self.bounds,
            maxiter=30,
        )
        self._verify_result_structure(res)

    def test_nelder_mead(self):
        res = nelder_mead_optimize(
            objective=self.objective,
            initial_point=self.initial_point,
            bounds=self.bounds,
            maxiter=30,
        )
        self._verify_result_structure(res)

    def test_spsa(self):
        res = spsa_optimize(
            objective=self.objective,
            initial_point=np.array([0.0, 0.5]),
            maxiter=50,
            learning_rate=0.1,
            perturbation_scale=0.1,
        )
        self._verify_result_structure(res)

    def test_slsqp(self):
        res = slsqp_optimize(
            objective=self.objective,
            initial_point=self.initial_point,
            bounds=self.bounds,
            maxiter=20,
        )
        self._verify_result_structure(res)

    def test_differential_evolution(self):
        res = differential_evolution_optimize(
            objective=self.objective,
            bounds=self.bounds,
            maxiter=5,
        )
        # Note: Scipy's differential_evolution might not support iterations in some scipy versions,
        # but we track success and structures.
        self._verify_result_structure(res)

    def test_genetic_algorithm(self):
        res = genetic_algorithm_optimize(
            objective=self.objective,
            bounds=self.bounds,
            maxiter=10,
            population_size=10,
        )
        self._verify_result_structure(res)

    def test_particle_swarm(self):
        res = particle_swarm_optimize(
            objective=self.objective,
            bounds=self.bounds,
            maxiter=10,
            n_particles=10,
        )
        self._verify_result_structure(res)

    def test_powell(self):
        res = powell_optimize(
            objective=self.objective,
            initial_point=self.initial_point,
            bounds=self.bounds,
            maxiter=30,
        )
        self._verify_result_structure(res)


if __name__ == "__main__":
    unittest.main()
