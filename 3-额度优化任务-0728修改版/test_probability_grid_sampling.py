import json
import contextlib
import io
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


class _FakeDataset:
    def __init__(self, data, label=None, **_):
        self.data = data
        self.label = np.asarray(label, dtype=float)


class _FakeBooster:
    def __init__(self, train_mean=0.5):
        self.train_mean = float(train_mean)

    def predict(self, X):
        values = np.asarray(X, dtype=float)
        signal = values[:, 0]
        scale = float(np.std(signal)) or 1.0
        z = np.clip((signal - np.median(signal)) / scale, -20.0, 20.0)
        probability = 0.15 + 0.60 / (1.0 + np.exp(-z)) + 0.10 * self.train_mean
        return np.clip(probability, 1e-6, 1.0 - 1e-6)

    def save_model(self, path):
        Path(path).write_text("fake booster\n", encoding="utf-8")


def _fake_lightgbm_module():
    module = types.ModuleType("lightgbm")
    module.Dataset = _FakeDataset
    module.Booster = _FakeBooster
    module.seen_params = []

    def train(params, train_set, **_):
        module.seen_params.append(dict(params))
        return _FakeBooster(np.mean(train_set.label))

    module.train = train
    module.log_evaluation = lambda **_: (lambda *args, **kwargs: None)
    return module


class ProbabilityGridSamplingIntegrationTests(unittest.TestCase):
    def test_balance_cascade_mixed_features_and_probability_grid(self):
        script_path = Path(__file__).with_name("generate_probability_grid.py")
        rows = 40
        data = pd.DataFrame({
            "cst_id": ["C%03d" % i for i in range(rows)],
            "split_eff_date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "y_freq": [(i % 4) == 0 for i in range(rows)],
            "y_dq_risk": [(i % 5) == 0 for i in range(rows)],
            "credamt": np.where(np.arange(rows) % 2 == 0, 1000.0, 1500.0),
            "age": [str(25 + i % 20) for i in range(rows)],
            "gnd_cd": [1 if i % 2 == 0 else 2 for i in range(rows)],
            "档位": ["F3" if i % 2 == 0 else "F2" for i in range(rows)],
            "continuous_feature": np.linspace(0.0, 10.0, rows),
        })
        fake_lgb = _fake_lightgbm_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cleaned_path = temp / "data_cleaned.csv"
            data.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
            old_cwd = os.getcwd()
            try:
                os.chdir(temp)
                with mock.patch.dict(sys.modules, {"lightgbm": fake_lgb}):
                    with contextlib.redirect_stdout(io.StringIO()):
                        runpy.run_path(
                            str(script_path),
                            init_globals={
                            "CLEANED_FILE_OVERRIDE": str(cleaned_path),
                            "CSV_ENCODING_OVERRIDE": "utf-8-sig",
                            "OUT_DIR_OVERRIDE": str(temp / "grid"),
                            "DEV_CALIBRATED_DIR_OVERRIDE": str(temp / "grid_cal"),
                            "CROSS_FIT_MODE_OVERRIDE": False,
                            "SPLIT_RATIOS_OVERRIDE": (0.60, 0.15, 0.15, 0.10),
                            "SAMPLING_METHOD_USAGE_OVERRIDE": "balance_cascade",
                            "SAMPLING_STRATEGY_USAGE_OVERRIDE": 1.0,
                            "SAMPLING_N_ESTIMATORS_USAGE_OVERRIDE": 3,
                            "SAMPLING_ENSEMBLE_RATIO_USAGE_OVERRIDE": 1.0,
                            "SAMPLING_METHOD_DEFAULT_OVERRIDE": "random_under",
                            "SAMPLING_STRATEGY_DEFAULT_OVERRIDE": 1.0,
                            "HANDLE_IMBALANCE_OVERRIDE": True,
                            "GRID_MIN_OVERRIDE": 1000.0,
                            "GRID_MAX_OVERRIDE": 1500.0,
                            "GRID_STEP_OVERRIDE": 500.0,
                            "INCLUDE_ZERO_OVERRIDE": False,
                            "LGB_PARAMS_OVERRIDE": {
                                "objective": "binary",
                                "metric": "auc",
                                "n_estimators": 2,
                                "verbose": -1,
                            },
                            },
                        )
            finally:
                os.chdir(old_cwd)

            metadata = json.loads(
                (temp / "grid" / "feature_preprocessing.json").read_text(encoding="utf-8")
            )
            self.assertIn("gnd_cd", metadata["categorical_features"])
            self.assertNotIn("age", metadata["categorical_features"])
            self.assertEqual(
                metadata["sampling_config"]["usage"]["method"],
                "balance_cascade",
            )
            self.assertEqual(
                metadata["sampling_config"]["default"]["method"],
                "random_under",
            )
            self.assertEqual(np.load(temp / "grid" / "p_usage.npy").shape, (rows, 2))
            self.assertTrue((temp / "saved_models" / "booster_usage_ensemble.json").is_file())
            self.assertTrue((temp / "saved_models" / "booster_default.txt").is_file())
            self.assertFalse((temp / "saved_models" / "booster_default_ensemble.json").exists())
            self.assertGreater(len(fake_lgb.seen_params), 2)
            self.assertTrue(all("scale_pos_weight" not in params for params in fake_lgb.seen_params))


if __name__ == "__main__":
    unittest.main()
