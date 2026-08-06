import unittest

import numpy as np
import pandas as pd

from sampling_methods import (
    BalanceCascade, MixedSMOTE, RandomOverSampler, RandomUnderSampler,
    TomekLinks, _MetricSpace, _cat_names, _classes,
    _nearest_neighbors_excluding_self,
)


class SamplingMethodsTests(unittest.TestCase):
    def test_tomek_uses_the_actual_nearest_non_self_neighbor(self):
        X = pd.DataFrame({"x": [0.0, 0.1, 10.0]})
        y = pd.Series([0, 1, 0])

        X_resampled, y_resampled = TomekLinks().fit_resample(X, y)

        self.assertEqual(len(X_resampled), 2)
        self.assertEqual(y_resampled.tolist(), [1, 0])
        self.assertEqual(X_resampled["x"].tolist(), [0.1, 10.0])

    def test_smotenc_uses_neighbor_mode_for_categorical_features(self):
        X = pd.DataFrame({
            "continuous": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                           0.0, 1.0, 2.0, 3.0],
            "category": [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1],
        })
        y = pd.Series([0] * 8 + [1] * 4)
        sampler = MixedSMOTE(
            variant="smote",
            sampling_strategy=1.0,
            k_neighbors=3,
            random_state=7,
            categorical_features=["category"],
        )

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        synthetic = X_resampled.iloc[len(X):]

        self.assertEqual(y_resampled.value_counts().to_dict(), {0: 8, 1: 8})
        self.assertEqual(synthetic["category"].tolist(), [0, 0, 0, 0])
        self.assertTrue(synthetic["continuous"].between(0.0, 3.0).all())

    def test_smotenc_categorical_distance_is_tied_to_scaled_continuous_spread(self):
        X = pd.DataFrame({
            "continuous": [0.0, 1.0, 2.0, 100.0],
            "category": [0, 0, 1, 1],
        })
        metric = _MetricSpace(X, ["category"])
        expected = np.median(np.std(metric.numeric_scaled(X), axis=0)) / np.sqrt(2.0)

        self.assertAlmostEqual(metric.categorical_scale(X), expected)
        self.assertEqual(metric.scaler.center_.tolist(), [1.5])

    def test_categorical_distance_is_not_zero_when_continuous_features_are_constant(self):
        X = pd.DataFrame({
            "continuous": [1.0, 1.0, 1.0],
            "category": [0, 1, 2],
        })
        metric = _MetricSpace(X, ["category"])

        self.assertAlmostEqual(metric.categorical_scale(X), 1.0 / np.sqrt(2.0))

    def test_duplicate_coordinates_still_exclude_the_actual_query_row(self):
        Z = np.array([[0.0], [0.0], [1.0]])
        neighbors = _nearest_neighbors_excluding_self(
            Z, Z, np.arange(len(Z)), n_neighbors=1
        )

        self.assertNotEqual(neighbors[0, 0], 0)
        self.assertNotEqual(neighbors[1, 0], 1)
        self.assertNotEqual(neighbors[2, 0], 2)

    def test_balanced_binary_classes_remain_distinct(self):
        minority, majority, n_min, n_maj = _classes(pd.Series([0, 1, 0, 1]))

        self.assertNotEqual(minority, majority)
        self.assertEqual((n_min, n_maj), (2, 2))

    def test_random_samplers_are_no_ops_when_target_is_current_ratio(self):
        X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0, 1])

        for sampler in (
            RandomOverSampler(sampling_strategy=1.0, random_state=7),
            RandomUnderSampler(sampling_strategy=1.0, random_state=7),
        ):
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            pd.testing.assert_frame_equal(X_resampled, X)
            pd.testing.assert_series_equal(y_resampled, y)

    def test_categorical_feature_indices_are_validated(self):
        X = pd.DataFrame({"a": [0], "b": [1]})

        with self.assertRaisesRegex(ValueError, "索引越界"):
            _cat_names(X, [-1])
        with self.assertRaisesRegex(ValueError, "不能重复"):
            _cat_names(X, [0, 0])

    def test_balance_cascade_follows_target_fpr_schedule(self):
        X = pd.DataFrame({"x": np.arange(45, dtype=float)})
        y = pd.Series([0] * 40 + [1] * 5)
        cascade = BalanceCascade(n_estimators=4, ratio=1.0, random_state=3).initialize(X, y)

        self.assertEqual(cascade.effective_n_estimators_, 4)

        while cascade.has_next_subset():
            cascade.next_subset()
            # 分数越大越像少数类；每轮应按目标 FPR 保留分数最高的多数类样本。
            cascade.update(np.arange(len(cascade.remaining_X()), dtype=float), score_label=1)

        self.assertAlmostEqual(cascade.target_fpr_, 0.5)
        # 4 个子模型之间仅有 3 次级联转移；最后一轮训练后不再多淘汰一次。
        self.assertEqual(cascade.remaining_majority_counts_, [40, 20, 10, 5])
        self.assertEqual(len(cascade.thresholds_), 3)
        self.assertEqual(len(cascade.subsets_), 4)
        for _, y_subset in cascade.subsets_:
            counts = pd.Series(np.asarray(y_subset)).value_counts().to_dict()
            self.assertEqual(counts, {0: 5, 1: 5})

    def test_balance_cascade_uses_one_stage_for_already_balanced_data(self):
        X = pd.DataFrame({"x": np.arange(10, dtype=float)})
        y = pd.Series([0] * 5 + [1] * 5)
        cascade = BalanceCascade(n_estimators=10, ratio=1.0, random_state=3).initialize(X, y)

        self.assertEqual(cascade.effective_n_estimators_, 1)
        X_subset, y_subset = cascade.next_subset()
        self.assertEqual(len(X_subset), 10)
        self.assertEqual(pd.Series(y_subset).value_counts().to_dict(), {0: 5, 1: 5})
        final_info = cascade.update(
            np.linspace(0, 1, len(cascade.remaining_X())), score_label=1
        )
        self.assertFalse(cascade.has_next_subset())
        self.assertEqual(len(cascade.subsets_), 1)
        self.assertFalse(final_info["pruned_for_next_stage"])
        self.assertTrue(np.isnan(final_info["threshold"]))


if __name__ == "__main__":
    unittest.main()
