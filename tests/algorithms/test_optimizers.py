"""
Unit tests for the optimizer wrappers in qpiai_quantum/algorithms/utils/optimizers/.
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch
from qpiai_quantum.algorithms.utils.optimizers.progress import OptimizationProgress

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

    def test_cobyla_honors_function_evaluation_limit(self):
        _, _, info = cobyla_optimize(
            objective=self.objective,
            initial_point=self.initial_point,
            bounds=self.bounds,
            maxiter=100,
            maxfun=7,
        )

        self.assertLessEqual(info["nfev"], 7)
        self.assertEqual(info["max_evaluations"], 7)

    def test_cobyla_nested_options_cannot_expand_maxfun(self):
        _, _, info = cobyla_optimize(
            objective=self.objective,
            initial_point=self.initial_point,
            maxfun=7,
            options={"maxiter": 100},
        )

        self.assertLessEqual(info["nfev"], 7)
        self.assertEqual(info["max_evaluations"], 7)

    def test_adagrad_preserves_progress_when_objective_interrupts(self):
        progress = OptimizationProgress()
        evaluations = 0

        def interrupted_objective(params):
            nonlocal evaluations
            if evaluations == 2:
                raise RuntimeError("evaluation budget")
            evaluations += 1
            return self.objective(params)

        with self.assertRaisesRegex(RuntimeError, "evaluation budget"):
            adagrad_optimize(
                objective=interrupted_objective,
                gradient=self.gradient,
                initial_point=self.initial_point,
                maxiter=10,
                tol=0.0,
                progress=progress,
            )

        self.assertEqual(progress.completed_iterations, 1)

    def test_nelder_mead(self):
        res = nelder_mead_optimize(
            objective=self.objective,
            initial_point=self.initial_point,
            bounds=self.bounds,
            maxiter=30,
        )
        self._verify_result_structure(res)

    def test_nelder_mead_respects_bounds(self):
        params, _, _ = nelder_mead_optimize(
            objective=lambda values: float((values[0] - 10.0) ** 2),
            initial_point=np.array([0.0]),
            bounds=[(-1.0, 1.0)],
            maxiter=50,
        )

        self.assertGreaterEqual(params[0], -1.0)
        self.assertLessEqual(params[0], 1.0)

    def test_spsa(self):
        res = spsa_optimize(
            objective=self.objective,
            initial_point=np.array([0.0, 0.5]),
            maxiter=50,
            learning_rate=0.1,
            perturbation_scale=0.1,
        )
        self._verify_result_structure(res)

    def test_spsa_seed_is_reproducible(self):
        kwargs = {
            "objective": self.objective,
            "initial_point": np.array([0.0, 0.5]),
            "maxiter": 8,
            "seed": 123,
        }

        first = spsa_optimize(**kwargs)
        second = spsa_optimize(**kwargs)

        np.testing.assert_allclose(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2]["history"], second[2]["history"])

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

    def test_differential_evolution_seed_is_reproducible(self):
        kwargs = {
            "objective": self.objective,
            "bounds": self.bounds,
            "maxiter": 2,
            "popsize": 5,
            "seed": 321,
            "polish": False,
        }

        first = differential_evolution_optimize(**kwargs)
        second = differential_evolution_optimize(**kwargs)

        np.testing.assert_allclose(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_differential_evolution_preserves_both_callback_forms(self):
        for form in ("legacy", "intermediate_result"):
            with self.subTest(form=form):
                observed = []
                progress = OptimizationProgress()

                def legacy_callback(params, convergence):
                    observed.append((params.copy(), convergence))
                    return True

                def result_callback(intermediate_result):
                    observed.append(intermediate_result)
                    return True

                _, _, info = differential_evolution_optimize(
                    objective=self.objective,
                    bounds=self.bounds,
                    maxiter=5,
                    popsize=5,
                    polish=False,
                    seed=42,
                    callback=(legacy_callback if form == "legacy" else result_callback),
                    progress=progress,
                )

                self.assertEqual(len(observed), 1)
                self.assertEqual(info["nit"], 1)
                self.assertEqual(progress.completed_iterations, 1)
                self.assertEqual(progress.callback_reports, 1)

    def test_scipy_wrappers_preserve_callback_reports_when_interrupted(self):
        methods = [
            (cobyla_optimize, "gradient_free"),
            (nelder_mead_optimize, "gradient_free"),
            (slsqp_optimize, "gradient_free"),
            (powell_optimize, "gradient_free"),
            (l_bfgs_b_optimize, "gradient_based"),
            (cg_optimize, "gradient_based"),
        ]
        for optimize, module in methods:
            with self.subTest(optimizer=optimize.__name__):
                progress = OptimizationProgress()
                user_callback = MagicMock()

                def interrupt_after_callback(
                    objective, initial_point, callback, **_kwargs
                ):
                    objective(initial_point)
                    callback(initial_point)
                    raise RuntimeError("evaluation budget")

                with (
                    patch(
                        f"qpiai_quantum.algorithms.utils.optimizers.{module}.minimize",
                        side_effect=interrupt_after_callback,
                    ),
                    self.assertRaisesRegex(RuntimeError, "evaluation budget"),
                ):
                    optimize(
                        objective=self.objective,
                        initial_point=self.initial_point,
                        callback=user_callback,
                        progress=progress,
                    )

                self.assertIsNone(progress.completed_iterations)
                self.assertEqual(progress.callback_reports, 1)
                user_callback.assert_called_once()

    def test_genetic_algorithm(self):
        res = genetic_algorithm_optimize(
            objective=self.objective,
            bounds=self.bounds,
            maxiter=10,
            population_size=10,
        )
        self._verify_result_structure(res)

    def test_genetic_algorithm_seed_is_reproducible(self):
        kwargs = {
            "objective": self.objective,
            "bounds": self.bounds,
            "maxiter": 3,
            "population_size": 6,
            "seed": 456,
        }

        first = genetic_algorithm_optimize(**kwargs)
        second = genetic_algorithm_optimize(**kwargs)

        np.testing.assert_allclose(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_particle_swarm(self):
        res = particle_swarm_optimize(
            objective=self.objective,
            bounds=self.bounds,
            maxiter=10,
            n_particles=10,
        )
        self._verify_result_structure(res)

    def test_particle_swarm_seed_is_reproducible(self):
        kwargs = {
            "objective": self.objective,
            "bounds": self.bounds,
            "maxiter": 3,
            "n_particles": 6,
            "seed": 789,
        }

        first = particle_swarm_optimize(**kwargs)
        second = particle_swarm_optimize(**kwargs)

        np.testing.assert_allclose(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_callback_exceptions_are_not_hidden(self):
        def fail_callback(_params):
            raise RuntimeError("stop optimization")

        with self.assertRaisesRegex(RuntimeError, "stop optimization"):
            spsa_optimize(
                objective=self.objective,
                initial_point=self.initial_point,
                maxiter=2,
                callback=fail_callback,
                seed=1,
            )

        with self.assertRaisesRegex(RuntimeError, "stop optimization"):
            nelder_mead_optimize(
                objective=self.objective,
                initial_point=self.initial_point,
                maxiter=2,
                callback=fail_callback,
            )

        with self.assertRaisesRegex(RuntimeError, "stop optimization"):
            gradient_descent_optimize(
                objective=self.objective,
                gradient=self.gradient,
                initial_point=self.initial_point,
                maxiter=2,
                callback=fail_callback,
            )

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
