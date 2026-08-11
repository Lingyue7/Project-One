import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from optimal_credit_limit_precomputed_grid_large import (
    LargePrecomputedGridCalculator,
    LargePrecomputedGridConfig,
)
from parameter_selection import derive_tier_limit_policy
from optimization_sampling import build_optimization_sample_plan


class CreditLimitPipelineTests(unittest.TestCase):
    def test_probability_grid_sample_plan_is_deterministic_and_scoped(self):
        work = pd.DataFrame({
            "cst_id": [f"c{i}" for i in range(30)],
            "档位": np.repeat(["F3", "F2", "F1", "E", "D"], 6),
        })
        fit_idx = np.arange(24, dtype=int)
        test_idx = np.arange(24, 30, dtype=int)
        first = build_optimization_sample_plan(
            work, sample_size=10, random_state=42,
            fit_indices=fit_idx, test_indices=test_idx,
        )
        second = build_optimization_sample_plan(
            work, sample_size=10, random_state=42,
            fit_indices=fit_idx, test_indices=test_idx,
        )

        self.assertEqual(len(first["all"]), 10)
        self.assertEqual(len(first["fit"]), 10)
        self.assertEqual(len(first["test"]), len(test_idx))
        self.assertTrue(set(first["fit"]).issubset(set(fit_idx)))
        self.assertTrue(set(first["test"]).issubset(set(test_idx)))
        for scope in ("all", "fit", "test"):
            np.testing.assert_array_equal(first[scope], second[scope])

    def test_tier_caps_follow_technical_document_without_extra_gap(self):
        work = pd.DataFrame({
            "档位": np.repeat(["F3", "F2", "F1", "E", "D"], 20),
            "credamt": np.repeat(100.0, 100),
        })

        min_limits, uncapped, _ = derive_tier_limit_policy(
            work,
            talent_col="档位",
            credit_limit_col="credamt",
            grid_step=10.0,
            grid_max=None,
        )
        _, capped, _ = derive_tier_limit_policy(
            work,
            talent_col="档位",
            credit_limit_col="credamt",
            grid_step=10.0,
            grid_max=100.0,
        )

        self.assertEqual(min_limits, {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0})
        self.assertEqual(uncapped, {1: 100.0, 2: 100.0, 3: 100.0, 4: 110.0, 5: 120.0})
        self.assertEqual(capped, {1: 100.0, 2: 100.0, 3: 100.0, 4: 100.0, 5: 100.0})
        self.assertTrue(all(capped[level] <= capped[level + 1] for level in range(1, 5)))

    def test_scope_and_tier_stratified_sample_keep_all_inputs_aligned(self):
        n_rows = 10
        work = pd.DataFrame({
            "cst_id": [f"c{number}" for number in range(n_rows)],
            "档位": ["F3", "F2"] * 5,
            "credamt": np.repeat(1000.0, n_rows),
            "y_freq": [0, 1] * 5,
            "y_dq_risk": [1, 0] * 5,
        })
        config = LargePrecomputedGridConfig()
        config.optimization_sample_enabled = True
        config.optimization_sample_size = 4
        config.optimization_sample_random_state = 42
        calculator = LargePrecomputedGridCalculator(config)
        calculator.data_info = {
            "df_raw": work.copy(),
            "df_aggregated": work.copy(),
            "df_cleaned": work.copy(),
            "X_usage": pd.DataFrame({"x": np.arange(n_rows)}),
            "y_usage": work["y_freq"].copy(),
            "X_default": pd.DataFrame({"x": np.arange(n_rows)}),
            "y_default": work["y_dq_risk"].copy(),
        }
        grid = np.asarray([0.0, 1000.0, 2000.0])
        probabilities = np.tile(np.asarray([0.1, 0.2, 0.3]), (n_rows, 1))
        calculator.prob_grid = {
            "grid": grid,
            "p_usage": probabilities.copy(),
            "p_default": probabilities.copy(),
            "p_usage_raw": probabilities.copy(),
            "p_default_raw": probabilities.copy(),
        }
        calculator.models = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            config.prob_grid_path = temp_dir
            np.save(Path(temp_dir) / "fit_idx.npy", np.arange(8, dtype=int))
            calculator.apply_optimization_scope("fit")

        self.assertEqual(len(calculator._extract_talent_levels()), 4)
        calculator.apply_optimization_sample(True, sample_size=4, random_state=42)

        for key in (
            "df_raw", "df_aggregated", "df_cleaned",
            "X_usage", "y_usage", "X_default", "y_default",
        ):
            self.assertEqual(len(calculator.data_info[key]), 4)
        for key in ("p_usage", "p_default", "p_usage_raw", "p_default_raw"):
            self.assertEqual(calculator.prob_grid[key].shape, (4, 3))
        sampled_levels = calculator._extract_talent_levels()
        self.assertEqual(set(sampled_levels), {1, 2})
        self.assertEqual(len(sampled_levels), len(calculator.data_info["df_aggregated"]))

    def test_presampled_probability_grid_scope_is_not_sampled_twice(self):
        n_rows = 6
        work = pd.DataFrame({
            "cst_id": [f"c{i}" for i in range(n_rows)],
            "档位": ["F3", "F2", "F1", "E", "D", "F3"],
            "credamt": np.repeat(1000.0, n_rows),
            "y_freq": [0, 1, 0, 1, 0, 1],
            "y_dq_risk": [1, 0, 1, 0, 1, 0],
        })
        config = LargePrecomputedGridConfig()
        config.optimization_sample_enabled = True
        config.optimization_sample_size = 4
        config.optimization_sample_random_state = 42
        calculator = LargePrecomputedGridCalculator(config)
        calculator.data_info = {
            "df_raw": work.copy(), "df_aggregated": work.copy(),
            "df_cleaned": work.copy(),
            "X_usage": pd.DataFrame({"x": np.arange(n_rows)}),
            "y_usage": work["y_freq"].copy(),
            "X_default": pd.DataFrame({"x": np.arange(n_rows)}),
            "y_default": work["y_dq_risk"].copy(),
        }
        probabilities = np.tile(np.asarray([0.1, 0.2]), (n_rows, 1))
        calculator.prob_grid = {
            "grid": np.asarray([0.0, 1000.0]),
            "p_usage": probabilities.copy(), "p_default": probabilities.copy(),
            "p_usage_raw": probabilities.copy(), "p_default_raw": probabilities.copy(),
        }
        calculator.models = {}
        calculator.source_row_indices = np.arange(n_rows, dtype=int)
        calculator._probability_grid_pre_sampled = True

        with tempfile.TemporaryDirectory() as temp_dir:
            config.prob_grid_path = temp_dir
            np.save(Path(temp_dir) / "fit_idx.npy", np.arange(4, dtype=int))
            calculator.apply_optimization_scope("fit")
        ids_after_scope = calculator.data_info["df_aggregated"]["cst_id"].tolist()
        calculator.apply_optimization_sample(True, sample_size=4, random_state=42)

        self.assertEqual(ids_after_scope, calculator.data_info["df_aggregated"]["cst_id"].tolist())
        self.assertEqual(len(ids_after_scope), 4)

    def test_missing_talent_level_is_not_randomly_filled(self):
        config = LargePrecomputedGridConfig()
        calculator = LargePrecomputedGridCalculator(config)
        work = pd.DataFrame({"cst_id": ["a"], "credamt": [1000.0]})
        calculator.data_info = {
            "df_aggregated": work,
            "df_cleaned": work.copy(),
        }

        with self.assertRaisesRegex(ValueError, "缺少可用的 档位/talent_level"):
            calculator._extract_talent_levels()


if __name__ == "__main__":
    unittest.main()
