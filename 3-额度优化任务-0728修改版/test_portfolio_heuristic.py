import unittest

import numpy as np

from portfolio_milp_optimizer import (
    _bidirectional_local_search,
    _candidate_index_table,
    _feasible_growth_from_minimum,
    _prepare_heuristic_problem,
    audit_portfolio_constraints,
    solve_discrete_portfolio_heuristic,
    solve_discrete_portfolio_lagrangian,
)


class PortfolioHeuristicTests(unittest.TestCase):
    def _solve(self, risk_budget=30.0):
        grid = np.asarray([0.0, 100.0, 200.0, 300.0])
        levels = np.asarray([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        n_customers = len(levels)
        p_usage = np.tile(np.asarray([0.0, 0.80, 0.85, 0.90]), (n_customers, 1))
        p_default = np.tile(np.asarray([0.0, 0.01, 0.015, 0.02]), (n_customers, 1))
        min_limits = {level: 0.0 for level in range(1, 6)}
        max_limits = {level: 300.0 for level in range(1, 6)}
        result = solve_discrete_portfolio_heuristic(
            grid=grid,
            p_usage=p_usage,
            p_default=p_default,
            levels=levels,
            min_limits=min_limits,
            max_limits=max_limits,
            interest_rates=np.full(n_customers, 0.08),
            lgd_coefficients=np.full(n_customers, 0.45),
            linear_cost=0.005,
            quadratic_cost=0.0,
            total_budget=1200.0,
            risk_budget=risk_budget,
            base_limits=np.full(n_customers, 100.0),
            enforce_group_mean_monotonic=True,
            group_mean_min_ratio={2: 1.0, 3: 1.0, 4: 0.8, 5: 0.8},
            max_variables=400000,
            candidates_per_customer=4,
            max_rounds=20,
        )
        return result, grid, levels, p_default, min_limits, max_limits

    def test_returns_feasible_solution_without_claiming_mip_gap(self):
        result, grid, levels, p_default, min_limits, max_limits = self._solve()
        selected_pd = np.asarray([
            p_default[i, int(np.where(grid == result.limits[i])[0][0])]
            for i in range(len(levels))
        ])
        audit = audit_portfolio_constraints(
            result.limits,
            result.limits * selected_pd,
            levels,
            min_limits,
            max_limits,
            total_budget=1200.0,
            risk_budget=30.0,
            enforce_group_mean_monotonic=True,
            group_mean_min_ratio={2: 1.0, 3: 1.0, 4: 0.8, 5: 0.8},
        )

        self.assertEqual(result.solver_backend, "heuristic")
        self.assertTrue(result.solver_success)
        self.assertTrue(result.constraint_feasible)
        self.assertTrue(np.isnan(result.mip_gap))
        self.assertTrue(audit["feasible"])
        self.assertLessEqual(result.total_limit, 1200.0)
        self.assertLessEqual(result.weighted_default_risk, 30.0)

    def test_tighter_risk_budget_remains_feasible(self):
        result, _, _, _, _, _ = self._solve(risk_budget=10.0)
        self.assertTrue(result.constraint_feasible)
        self.assertLessEqual(result.weighted_default_risk, 10.0)

    def test_lagrangian_keeps_greedy_as_a_feasible_floor(self):
        greedy, grid, levels, p_default, min_limits, max_limits = self._solve()
        n_customers = len(levels)
        p_usage = np.tile(np.asarray([0.0, 0.80, 0.85, 0.90]), (n_customers, 1))
        lagrangian = solve_discrete_portfolio_lagrangian(
            grid=grid,
            p_usage=p_usage,
            p_default=p_default,
            levels=levels,
            min_limits=min_limits,
            max_limits=max_limits,
            interest_rates=np.full(n_customers, 0.08),
            lgd_coefficients=np.full(n_customers, 0.45),
            linear_cost=0.005,
            quadratic_cost=0.0,
            total_budget=1200.0,
            risk_budget=30.0,
            base_limits=np.full(n_customers, 100.0),
            enforce_group_mean_monotonic=True,
            group_mean_min_ratio={2: 1.0, 3: 1.0, 4: 0.8, 5: 0.8},
            max_variables=400000,
            candidates_per_customer=4,
            max_rounds=20,
            lagrangian_iterations=20,
            lagrangian_time_limit_seconds=10.0,
        )

        self.assertTrue(lagrangian.constraint_feasible)
        self.assertGreaterEqual(lagrangian.objective_value, greedy.objective_value - 1e-9)
        self.assertGreaterEqual(lagrangian.dual_upper_bound, lagrangian.objective_value)
        self.assertGreaterEqual(lagrangian.heuristic_dual_bound_gap, 0.0)
        self.assertEqual(lagrangian.feasible_start_count, 2)

    def test_lagrangian_can_take_a_negative_paving_move_for_joint_gain(self):
        grid = np.asarray([0.0, 100.0])
        levels = np.asarray([1, 2])
        p_usage = np.asarray([[0.0, 0.9], [0.0, 0.0]])
        p_default = np.zeros((2, 2), dtype=float)
        common = dict(
            grid=grid,
            p_usage=p_usage,
            p_default=p_default,
            levels=levels,
            min_limits={1: 0.0, 2: 0.0},
            max_limits={1: 100.0, 2: 100.0},
            interest_rates=np.asarray([0.2, 0.2]),
            lgd_coefficients=np.zeros(2),
            linear_cost=0.05,
            quadratic_cost=0.0,
            total_budget=200.0,
            risk_budget=0.0,
            base_limits=np.zeros(2),
            enforce_group_mean_monotonic=True,
            group_mean_min_ratio={2: 1.0},
            max_variables=100,
            candidates_per_customer=4,
            max_rounds=10,
        )
        greedy = solve_discrete_portfolio_heuristic(**common)
        lagrangian = solve_discrete_portfolio_lagrangian(
            lagrangian_iterations=20,
            lagrangian_time_limit_seconds=10.0,
            lagrangian_step_size=0.5,
            **common
        )

        np.testing.assert_array_equal(greedy.limits, np.asarray([0.0, 0.0]))
        np.testing.assert_array_equal(lagrangian.limits, np.asarray([100.0, 100.0]))
        self.assertGreater(lagrangian.objective_value, greedy.objective_value)

    def test_bidirectional_local_search_can_exchange_two_customers(self):
        grid = np.asarray([0.0, 50.0, 100.0])
        levels = np.asarray([1, 2])
        p_usage = np.asarray([
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ])
        p_default = np.zeros((2, 3), dtype=float)
        min_limits = {1: 0.0, 2: 0.0}
        max_limits = {1: 50.0, 2: 100.0}
        problem = _prepare_heuristic_problem(
            grid=grid,
            p_usage=p_usage,
            p_default=p_default,
            levels=levels,
            min_limits=min_limits,
            max_limits=max_limits,
            interest_rates=np.asarray([0.12, 0.10]),
            lgd_coefficients=np.zeros(2),
            linear_cost=0.0,
            quadratic_cost=0.0,
            base_limits=np.zeros(2),
            max_variables=100,
            candidates_per_customer=3,
        )
        greedy_growth = _feasible_growth_from_minimum(
            problem=problem,
            min_limits=min_limits,
            max_limits=max_limits,
            total_budget=100.0,
            risk_budget=0.0,
            enforce_group_mean_monotonic=False,
            group_mean_min_ratio={},
            max_rounds=10,
        )
        improved = _bidirectional_local_search(
            problem=problem,
            growth=greedy_growth,
            total_budget=100.0,
            risk_budget=0.0,
            enforce_group_mean_monotonic=False,
            group_mean_min_ratio={},
            max_passes=3,
            time_limit_seconds=10.0,
            pair_candidate_pool=10,
        )
        improved_limits = problem["candidate_limits"][improved["selected_var"]]

        np.testing.assert_array_equal(
            problem["candidate_limits"][greedy_growth["selected_var"]],
            np.asarray([50.0, 0.0]),
        )
        np.testing.assert_array_equal(improved_limits, np.asarray([0.0, 100.0]))
        self.assertAlmostEqual(greedy_growth["objective_value"], 6.0)
        self.assertAlmostEqual(improved["objective_value"], 10.0)
        self.assertEqual(improved["pair_exchanges"], 1)
        self.assertAlmostEqual(improved["local_search_objective_gain"], 4.0)

    def test_c2_sensitivity_uses_one_common_reduced_candidate_set(self):
        grid = np.arange(0.0, 1050.0, 50.0)
        levels = np.asarray([1, 2])
        p_usage = np.tile(np.linspace(0.0, 1.0, len(grid)), (2, 1))
        p_default = np.zeros((2, len(grid)), dtype=float)
        common = dict(
            grid=grid,
            p_usage=p_usage,
            p_default=p_default,
            levels=levels,
            min_limits={1: 0.0, 2: 0.0},
            max_limits={1: 1000.0, 2: 1000.0},
            rates=np.asarray([0.10, 0.12]),
            lgd=np.zeros(2),
            linear_cost=0.0,
            base_limits=np.asarray([250.0, 750.0]),
            max_variables=8,
            candidates_per_customer=4,
            candidate_reference_quadratic_costs=(0.0, 0.001),
        )
        low_c2 = _candidate_index_table(quadratic_cost=0.0, **common)
        high_c2 = _candidate_index_table(quadratic_cost=0.001, **common)

        np.testing.assert_array_equal(
            low_c2["grid_index"], high_c2["grid_index"]
        )
        np.testing.assert_array_equal(low_c2["offsets"], high_c2["offsets"])
        self.assertTrue(low_c2["candidate_reduced"])
        self.assertEqual(len(low_c2["grid_index"]), 8)
        cached_result = solve_discrete_portfolio_heuristic(
            grid=grid,
            p_usage=p_usage,
            p_default=p_default,
            levels=levels,
            min_limits={1: 0.0, 2: 0.0},
            max_limits={1: 1000.0, 2: 1000.0},
            interest_rates=np.asarray([0.10, 0.12]),
            lgd_coefficients=np.zeros(2),
            linear_cost=0.0,
            quadratic_cost=0.001,
            total_budget=1000.0,
            risk_budget=0.0,
            base_limits=np.asarray([250.0, 750.0]),
            enforce_group_mean_monotonic=False,
            group_mean_min_ratio={},
            max_variables=8,
            candidates_per_customer=4,
            max_rounds=10,
            candidate_reference_quadratic_costs=(0.0, 0.001),
            candidate_index_table=low_c2,
        )
        self.assertTrue(cached_result.constraint_feasible)
        self.assertEqual(cached_result.variable_count, 8)


if __name__ == "__main__":
    unittest.main()
