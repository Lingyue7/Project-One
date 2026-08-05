import unittest

import numpy as np
import pandas as pd

from load_kechuang_potential_data import (
    PotentialFeaturePreprocessor,
    aggregate_by_customer_potential,
    build_labels_potential,
    cast_cat_cols,
    clean_data_potential,
    deduplicate_exact_cst_loan,
    filter_by_eff_date,
    filter_by_maturity,
    get_customer_static_conflicts,
    get_feature_descriptive_stats,
)


class PotentialFeaturePreprocessorTests(unittest.TestCase):
    def test_all_missing_customer_sums_stay_missing(self):
        raw = pd.DataFrame({
            "cst_id": ["a", "a"],
            "ba_out_bal_diff": [np.nan, np.nan],
            "credamt": [np.nan, np.nan],
        })

        aggregated = aggregate_by_customer_potential(raw)

        self.assertTrue(pd.isna(aggregated.loc[0, "ba_out_bal_diff"]))
        self.assertTrue(pd.isna(aggregated.loc[0, "credamt"]))

    def test_label_builders_exclude_customers_with_undefined_outcomes(self):
        freq = pd.DataFrame({
            "cst_id": ["a", "b", "c"],
            "ac_curr_bal_diff": [np.nan, 1.0, 10.0],
        })
        dq = pd.DataFrame({
            "cst_id": ["a", "b", "c", "d"],
            "rt_acct_stat_2_end": [None, "1", "3", "unknown"],
        })

        freq_labeled, _ = build_labels_potential(
            freq, target="y_freq", y_freq_mode="curr_p80_only"
        )
        dq_labeled, _ = build_labels_potential(dq, target="y_dq_risk")

        self.assertEqual(freq_labeled["cst_id"].tolist(), ["b", "c"])
        self.assertEqual(freq_labeled["y_freq"].tolist(), [0, 1])
        self.assertEqual(dq_labeled["cst_id"].tolist(), ["b", "c"])
        self.assertEqual(dq_labeled["y_dq_risk"].tolist(), [0, 1])

    def test_account_deduplication_rejects_conflicting_duplicates(self):
        exact = pd.DataFrame({
            "cst_id": ["a", "a"],
            "loanacctno": ["l1", "l1"],
            "amount": [5.0, 5.0],
        })
        conflict = exact.copy()
        conflict.loc[1, "amount"] = 6.0

        self.assertEqual(len(deduplicate_exact_cst_loan(exact)), 1)
        with self.assertRaisesRegex(ValueError, "字段冲突"):
            deduplicate_exact_cst_loan(conflict)

    def test_enabled_date_filters_require_their_source_columns(self):
        raw = pd.DataFrame({"cst_id": ["a"]})

        with self.assertRaises(KeyError):
            filter_by_maturity(raw, apply_filter=True)
        with self.assertRaises(KeyError):
            filter_by_eff_date(raw)

    def test_customer_aggregation_rejects_missing_customer_ids(self):
        raw = pd.DataFrame({"cst_id": ["a", None], "value": [1, 2]})

        with self.assertRaisesRegex(ValueError, "cst_id 为空"):
            aggregate_by_customer_potential(raw)

    def test_customer_static_conflicts_are_reported_before_aggregation(self):
        raw = pd.DataFrame({
            "cst_id": ["a", "a", "b"],
            "age": [30, 31, 40],
            "gnd_cd": ["1", "1", "2"],
        })

        conflicts = get_customer_static_conflicts(raw, ["age", "gnd_cd"])

        self.assertEqual(conflicts["字段"].tolist(), ["age"])
        self.assertEqual(conflicts.loc[0, "冲突客户数"], 1)

    def test_age_remains_a_continuous_numeric_feature(self):
        raw = pd.DataFrame({
            "cst_id": ["a", "b", "c"],
            "age": [30.0, "41", "not_available"],
        })

        cleaned = clean_data_potential(cast_cat_cols(raw), target="y_freq")

        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["age"]))
        self.assertEqual(cleaned.loc[0, "age"], 30.0)
        self.assertEqual(cleaned.loc[1, "age"], 41.0)
        self.assertTrue(pd.isna(cleaned.loc[2, "age"]))

    def test_text_categories_are_not_destroyed_by_code_normalization(self):
        raw = pd.DataFrame({
            "cst_id": ["a", "b", "c"],
            "gnd_cd": ["1.0", "男", " F "],
        })

        casted = cast_cat_cols(raw)
        cleaned = clean_data_potential(casted, target="y_freq")

        self.assertEqual(casted["gnd_cd"].tolist(), ["1", "男", "F"])
        self.assertEqual(cleaned["gnd_cd"].tolist(), ["1", "男", "F"])

    def test_future_customer_date_does_not_create_negative_tenure(self):
        raw = pd.DataFrame({
            "cst_id": ["a", "b"],
            "bank_cust_become_date": ["2025-01-01", "2027-01-01"],
        })

        cleaned = clean_data_potential(
            raw, snapshot_date="2026-01-01", target="y_freq"
        )

        self.assertEqual(cleaned.loc[0, "days_since_become_cust"], 365)
        self.assertTrue(pd.isna(cleaned.loc[1, "days_since_become_cust"]))

    def test_fit_statistics_come_only_from_training_rows(self):
        train = pd.DataFrame({
            "y_freq": [0, 1, 0],
            "numeric": [1.0, np.nan, 3.0],
            "mostly_missing": [np.nan, np.nan, 9.0],
            "category": ["a", None, "b"],
            "mostly_missing_category": [None, "x", None],
            "event_date": pd.to_datetime(["2026-01-01"] * 3),
        }, index=[10, 11, 12])
        validation = pd.DataFrame({
            "y_freq": [1, 0],
            "numeric": [100.0, np.nan],
            "mostly_missing": [4.0, 5.0],
            "category": ["new_category", None],
            "mostly_missing_category": ["x", "y"],
            "event_date": pd.to_datetime(["2026-02-01"] * 2),
        }, index=[20, 21])

        preprocessor = PotentialFeaturePreprocessor().fit(train)
        transformed = preprocessor.transform(validation)

        self.assertEqual(preprocessor.numeric_fill_values_["numeric"], 2.0)
        self.assertIn("mostly_missing", preprocessor.dropped_high_missing_)
        self.assertIn("mostly_missing_category", preprocessor.dropped_high_missing_)
        self.assertNotIn("mostly_missing", transformed.columns)
        self.assertNotIn("mostly_missing_category", transformed.columns)
        self.assertNotIn("event_date", transformed.columns)
        self.assertEqual(transformed.loc[21, "numeric"], 2.0)
        self.assertEqual(transformed.loc[20, "category"], -1)
        self.assertGreaterEqual(transformed.loc[21, "category"], 0)
        self.assertEqual(transformed.columns.tolist(), preprocessor.feature_names_)

    def test_transform_does_not_change_the_fitted_category_vocabulary(self):
        train = pd.DataFrame({
            "y_freq": [0, 1],
            "category": ["x", "y"],
        })
        validation = pd.DataFrame({
            "y_freq": [0],
            "category": ["z"],
        })
        preprocessor = PotentialFeaturePreprocessor().fit(train)
        classes_before = preprocessor.label_encoders_["category"].classes_.copy()

        transformed = preprocessor.transform(validation)

        np.testing.assert_array_equal(
            preprocessor.label_encoders_["category"].classes_, classes_before
        )
        self.assertEqual(transformed.loc[0, "category"], -1)

    def test_infinite_numeric_and_blank_categories_are_treated_as_missing(self):
        train = pd.DataFrame({
            "y_freq": [0, 1, 0, 1],
            "numeric": [1.0, np.inf, 3.0, 5.0],
            "category": ["a", "   ", "b", "a"],
        })
        validation = pd.DataFrame({
            "y_freq": [0],
            "numeric": [-np.inf],
            "category": [""],
        })

        preprocessor = PotentialFeaturePreprocessor().fit(train)
        transformed = preprocessor.transform(validation)

        self.assertEqual(preprocessor.numeric_fill_values_["numeric"], 3.0)
        self.assertEqual(transformed.loc[0, "numeric"], 3.0)
        self.assertGreaterEqual(transformed.loc[0, "category"], 0)

    def test_numeric_category_codes_use_mode_encoding_not_numeric_median(self):
        train = pd.DataFrame({
            "y_freq": [0, 1, 0, 1],
            "education_cd": [1.0, 1.0, np.nan, 3.0],
        })
        validation = pd.DataFrame({
            "y_freq": [0, 1],
            "education_cd": [np.nan, 2.0],
        })

        preprocessor = PotentialFeaturePreprocessor().fit(train)
        transformed = preprocessor.transform(validation)

        self.assertIn("education_cd", preprocessor.categorical_features_)
        self.assertNotIn("education_cd", preprocessor.numeric_features_)
        # 缺失值用训练集众数填充；未见代码 2.0 不扩充训练词表。
        self.assertGreaterEqual(transformed.loc[0, "education_cd"], 0)
        self.assertEqual(transformed.loc[1, "education_cd"], -1)

    def test_age_is_reported_as_numeric_in_descriptive_statistics(self):
        df = pd.DataFrame({"age": [20, 30], "gnd_cd": ["1", "2"]})

        numeric, categorical = get_feature_descriptive_stats(
            df, ["age", "gnd_cd"]
        )

        self.assertIn("age", numeric["字段"].tolist())
        self.assertNotIn("age", categorical["字段"].tolist())


if __name__ == "__main__":
    unittest.main()
