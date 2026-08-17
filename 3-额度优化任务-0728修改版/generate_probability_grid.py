#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_probability_grid.py
============================
额度优化专用：读取 Cell 3 清洗后数据（或兜底重跑预处理），
训练 y_freq / y_dq_risk 双 LightGBM 模型，生成概率网格 .npy 文件。

参数由 notebook Cell 1 通过 *_OVERRIDE 全局变量传入。
预处理逻辑对齐 load_kechuang_potential_data.py + kechuang_potential_preprocessing.ipynb。
"""

import os
import sys
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from numpy.lib.format import open_memmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd())

from load_kechuang_potential_data import (
    read_data,
    report_cst_loan_duplicates,
    deduplicate_exact_cst_loan,
    cast_cat_cols,
    rename_kechuang_cols,
    filter_by_maturity,
    filter_by_eff_date,
    filter_dq_start_customers,
    aggregate_by_customer_potential,
    clean_data_potential,
    build_labels_potential,
    drop_post_label_cols,
    PotentialFeaturePreprocessor,
    stratified_dual_target_partition_indices,
)
from sampling_methods import BalanceCascade, sampler_factory, print_sampling_summary
from optimization_sampling import build_optimization_sample_plan

# ── 参数（可被 notebook *_OVERRIDE 覆盖）────────────────────────────
CLEANED_FILE          = globals().get("CLEANED_FILE_OVERRIDE",          "data_cleaned.csv")
EXCEL_PATH            = globals().get("EXCEL_PATH_OVERRIDE",            "kechuang_merged0722.csv")
CSV_ENCODING          = globals().get("CSV_ENCODING_OVERRIDE",          "utf-8-sig")
OUT_DIR               = globals().get("OUT_DIR_OVERRIDE",               "probability_grid_large")
DEV_CALIBRATED_DIR = globals().get(
    "DEV_CALIBRATED_DIR_OVERRIDE",
    OUT_DIR + "_dev_calibrated",
)
GRID_MIN              = float(globals().get("GRID_MIN_OVERRIDE",          1000.0))
GRID_MAX              = float(globals().get("GRID_MAX_OVERRIDE",          1000000.0))
GRID_STEP             = float(globals().get("GRID_STEP_OVERRIDE",         500.0))
INCLUDE_ZERO          = bool(globals().get("INCLUDE_ZERO_OVERRIDE",       True))
SNAPSHOT_DATE         = globals().get("SNAPSHOT_DATE_OVERRIDE",          "2026-01-31")
APPLY_MATURITY_FILTER = bool(globals().get("APPLY_MATURITY_FILTER_OVERRIDE", False))
MATURITY_CUTOFF       = globals().get("MATURITY_CUTOFF_OVERRIDE",        "2026-07-21")
APPLY_EFF_DATE_FILTER = bool(globals().get("APPLY_EFF_DATE_FILTER_OVERRIDE", True))
EFF_DATE_LOWER        = globals().get("EFF_DATE_LOWER_OVERRIDE",         "2025-01-01")
EFF_DATE_UPPER        = globals().get("EFF_DATE_UPPER_OVERRIDE",         "2026-03-31")
DEDUP_CST_LOAN        = bool(globals().get("DEDUP_CST_LOAN_OVERRIDE", False))
Y_FREQ_MODE           = globals().get("Y_FREQ_MODE_OVERRIDE",            "bout_gt0_and_curr_p80")
HANDLE_IMBALANCE      = bool(globals().get("HANDLE_IMBALANCE_OVERRIDE",   True))
EARLY_STOPPING_ROUNDS = int(globals().get("EARLY_STOPPING_ROUNDS_OVERRIDE", 50))
RANDOM_STATE          = int(globals().get("RANDOM_STATE_OVERRIDE",         42))
SPLIT_RATIOS          = tuple(globals().get("SPLIT_RATIOS_OVERRIDE", (0.60, 0.15, 0.15, 0.10)))
CROSS_FIT_MODE        = bool(globals().get("CROSS_FIT_MODE_OVERRIDE", False))
CROSS_FIT_FOLDS       = int(globals().get("CROSS_FIT_FOLDS_OVERRIDE", 5))
INNER_CROSS_FIT_FOLDS = int(globals().get("INNER_CROSS_FIT_FOLDS_OVERRIDE", 5))
CROSS_FIT_DIR         = globals().get("CROSS_FIT_DIR_OVERRIDE", OUT_DIR + "_crossfit_calibrated")
PROBABILITY_GRID_SAMPLE_ENABLED = bool(
    globals().get("PROBABILITY_GRID_SAMPLE_ENABLED_OVERRIDE", False)
)
PROBABILITY_GRID_SAMPLE_SIZE = globals().get(
    "PROBABILITY_GRID_SAMPLE_SIZE_OVERRIDE", None
)
PROBABILITY_GRID_SAMPLE_RANDOM_STATE = int(
    globals().get("PROBABILITY_GRID_SAMPLE_RANDOM_STATE_OVERRIDE", RANDOM_STATE)
)
_LEGACY_SAMPLING_METHOD = globals().get("SAMPLING_METHOD_OVERRIDE", None)
_LEGACY_SAMPLING_STRATEGY = globals().get("SAMPLING_STRATEGY_OVERRIDE", 1.0)
_LEGACY_SAMPLING_N_ESTIMATORS = int(
    globals().get("SAMPLING_N_ESTIMATORS_OVERRIDE", 10)
)
_LEGACY_SAMPLING_ENSEMBLE_RATIO = float(
    globals().get("SAMPLING_ENSEMBLE_RATIO_OVERRIDE", 1.0)
)
SAMPLING_CONFIG = {
    "usage": {
        "method": globals().get(
            "SAMPLING_METHOD_USAGE_OVERRIDE", _LEGACY_SAMPLING_METHOD
        ),
        "strategy": globals().get(
            "SAMPLING_STRATEGY_USAGE_OVERRIDE", _LEGACY_SAMPLING_STRATEGY
        ),
        "n_estimators": int(globals().get(
            "SAMPLING_N_ESTIMATORS_USAGE_OVERRIDE",
            _LEGACY_SAMPLING_N_ESTIMATORS,
        )),
        "ensemble_ratio": float(globals().get(
            "SAMPLING_ENSEMBLE_RATIO_USAGE_OVERRIDE",
            _LEGACY_SAMPLING_ENSEMBLE_RATIO,
        )),
    },
    "default": {
        "method": globals().get(
            "SAMPLING_METHOD_DEFAULT_OVERRIDE", _LEGACY_SAMPLING_METHOD
        ),
        "strategy": globals().get(
            "SAMPLING_STRATEGY_DEFAULT_OVERRIDE", _LEGACY_SAMPLING_STRATEGY
        ),
        "n_estimators": int(globals().get(
            "SAMPLING_N_ESTIMATORS_DEFAULT_OVERRIDE",
            _LEGACY_SAMPLING_N_ESTIMATORS,
        )),
        "ensemble_ratio": float(globals().get(
            "SAMPLING_ENSEMBLE_RATIO_DEFAULT_OVERRIDE",
            _LEGACY_SAMPLING_ENSEMBLE_RATIO,
        )),
    },
}
SAMPLING_CATEGORICAL_CANDIDATES = list(globals().get(
    "SAMPLING_CATEGORICAL_CANDIDATES_OVERRIDE",
    [
        "gnd_cd", "mar_sttn_cd", "education_cd", "occup_cd", "cst_star_cd",
        "busikind",
    ],
))

_DEFAULT_LGB = {
    "objective":         "binary",
    "metric":            "auc",
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "max_depth":         -1,
    "min_child_samples": 20,
    "subsample":         0.8,
    "bagging_freq":      1,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "verbose":           -1,
}
LGB_PARAMS = globals().get("LGB_PARAMS_OVERRIDE", _DEFAULT_LGB)

QUOTA_COL = "credamt"


def _stratified_split_indices(work: pd.DataFrame):
    """Create the task-2-compatible joint-stratified random development split."""
    if len(SPLIT_RATIOS) != 4 or not np.isclose(sum(SPLIT_RATIOS), 1.0):
        raise ValueError("SPLIT_RATIOS must contain four values summing to 1")
    work = work.reset_index(drop=True)
    train_idx, validation_idx, cal_idx, test_idx = (
        stratified_dual_target_partition_indices(
            work,
            SPLIT_RATIOS,
            random_state=RANDOM_STATE,
        )
    )
    fit_idx = np.sort(np.concatenate([train_idx, validation_idx])).astype(int)

    print(
        "  joint-stratified random split train/validation/cal/test = %d/%d/%d/%d "
        "(%.0f%%/%.0f%%/%.0f%%/%.0f%%)"
        % (
            len(train_idx), len(validation_idx), len(cal_idx), len(test_idx),
            SPLIT_RATIOS[0] * 100, SPLIT_RATIOS[1] * 100,
            SPLIT_RATIOS[2] * 100, SPLIT_RATIOS[3] * 100,
        )
    )
    for name, idx in [
        ("train", train_idx),
        ("validation", validation_idx),
        ("cal", cal_idx),
        ("test", test_idx),
    ]:
        print(
            "    %-10s y_freq=%6.2f%% | y_dq_risk=%6.2f%%"
            % (
                name,
                work.iloc[idx]["y_freq"].mean() * 100,
                work.iloc[idx]["y_dq_risk"].mean() * 100,
            )
        )
    return work, train_idx, validation_idx, cal_idx, test_idx, fit_idx


def _label_threshold_reference_mask(work: pd.DataFrame):
    """Select the earliest 60% only for fitting data-driven y_freq thresholds."""
    if "split_eff_date" not in work.columns:
        raise ValueError(
            "data_cleaned.csv is missing split_eff_date; cannot fit the y_freq threshold"
        )
    dates = pd.to_datetime(work["split_eff_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(
            "split_eff_date contains %d invalid rows; cannot fit the y_freq threshold"
            % int(dates.isna().sum())
        )
    order = np.argsort(dates.to_numpy(), kind="mergesort")
    reference_count = int(len(work) * SPLIT_RATIOS[0])
    if reference_count <= 0:
        raise ValueError("The y_freq threshold reference cohort is empty")
    mask = np.zeros(len(work), dtype=bool)
    mask[np.asarray(order[:reference_count], dtype=int)] = True
    return mask


def _fit_transform_features_train_only(
    work: pd.DataFrame,
    train_idx,
    fit_scope: str = "train_only",
):
    """直接复用任务2的训练集拟合预处理器，再转换全部样本。"""
    preprocessor = PotentialFeaturePreprocessor(
        target="y_freq",
        add_quota_sq=False,
        add_quota_cube=False,
        add_quota_log=False,
        categorical_features=SAMPLING_CATEGORICAL_CANDIDATES,
    )
    preprocessor.fit(work.iloc[train_idx])
    X = preprocessor.transform(work)

    category_metadata = {}
    for col in preprocessor.categorical_features_:
        unseen_count = int((X[col] == -1).sum())
        category_metadata[col] = {
            "fill_value": preprocessor.categorical_fill_values_[col],
            "classes": preprocessor.label_encoders_[col].classes_.tolist(),
            "unseen_full_rows": unseen_count,
        }
        if unseen_count:
            print(
                "  category %s: %d unseen values encoded as -1"
                % (col, unseen_count)
            )

    sampling_metadata = {}
    for target_key in ("usage", "default"):
        config = SAMPLING_CONFIG[target_key]
        sampling_metadata[target_key] = {
            "method": _sampling_method_key(target_key),
            "strategy": config["strategy"],
            "n_estimators": config["n_estimators"],
            "ensemble_majority_to_minority_ratio": config["ensemble_ratio"],
        }
    metadata = {
        "fit_scope": fit_scope,
        "split_ratios": list(SPLIT_RATIOS),
        "preprocessor": "PotentialFeaturePreprocessor (task2 synchronized)",
        "dropped_high_missing_columns": preprocessor.dropped_high_missing_,
        "numeric_medians": preprocessor.numeric_fill_values_,
        "categorical_rules": category_metadata,
        "categorical_features": preprocessor.categorical_features_,
        "feature_names": X.columns.tolist(),
        "sampling_config": sampling_metadata,
    }
    if preprocessor.dropped_high_missing_:
        print(
            "  %s missingness removed %d columns (>=40%%): %s"
            % (
                fit_scope,
                len(preprocessor.dropped_high_missing_),
                preprocessor.dropped_high_missing_,
            )
        )
    print("  feature engineering fit_scope=%s: %d features" % (fit_scope, X.shape[1]))
    return X, metadata


def _load_or_preprocess() -> pd.DataFrame:
    """优先读 Cell 3 已保存的清洗数据；不存在则兜底重跑双标签预处理。"""
    if os.path.isfile(CLEANED_FILE):
        print(f"[1/5] 读取清洗后数据: {CLEANED_FILE}")
        work = pd.read_csv(CLEANED_FILE, encoding=CSV_ENCODING)
        print(f"  shape={work.shape}")
        if "y_freq" not in work.columns or "y_dq_risk" not in work.columns:
            raise ValueError(
                f"{CLEANED_FILE} 缺少 y_freq 或 y_dq_risk，请先运行 Cell 3 重新清洗"
            )
        return work

    print(f"[1/5] 未找到 {CLEANED_FILE}，从原始文件重跑预处理: {EXCEL_PATH}")
    df0 = read_data(EXCEL_PATH, csv_encoding=CSV_ENCODING)
    df0 = rename_kechuang_cols(df0)
    df0 = report_cst_loan_duplicates(df0)
    if DEDUP_CST_LOAN:
        df0 = deduplicate_exact_cst_loan(df0)
    df0 = cast_cat_cols(df0)
    df0 = filter_by_maturity(
        df0, apply_filter=APPLY_MATURITY_FILTER, maturity_cutoff=MATURITY_CUTOFF
    )
    if APPLY_EFF_DATE_FILTER:
        df0 = filter_by_eff_date(
            df0, eff_date_lower=EFF_DATE_LOWER, eff_date_upper=EFF_DATE_UPPER
        )
    else:
        print("  apply_eff_date_filter=False，跳过生效日筛选")
    df0 = filter_dq_start_customers(df0)
    df_agg = aggregate_by_customer_potential(df0)
    df_clean = clean_data_potential(
        df_agg, snapshot_date=SNAPSHOT_DATE, target="y_dq_risk"
    )
    label_train_mask = _label_threshold_reference_mask(df_clean)
    df_labeled, _ = build_labels_potential(
        df_clean,
        target="y_freq",
        y_freq_mode=Y_FREQ_MODE,
        threshold_fit_mask=label_train_mask,
    )
    df_labeled, _ = build_labels_potential(df_labeled, target="y_dq_risk")
    work = drop_post_label_cols(df_labeled, target="y_dq_risk")
    print(f"  预处理后: {len(work):,} 客户")
    return work


def _sampling_method_key(target_key):
    """Return one target's canonical task-2 sampling method name."""
    if target_key not in SAMPLING_CONFIG:
        raise KeyError("target_key must be 'usage' or 'default'")
    method_value = SAMPLING_CONFIG[target_key]["method"]
    if method_value is None:
        return "baseline"
    method = str(method_value).strip().lower().replace("-", "_").replace(" ", "_")
    if method in ("", "none", "baseline"):
        return "baseline"
    aliases = {
        "balancecascade": "balance_cascade",
        "easyensemble": "easy_ensemble",
        "ensemble": "easy_ensemble",
    }
    return aliases.get(method, method)


def _build_lgb_params(y: pd.Series, target_key: str) -> dict:
    """从 LGB_PARAMS 构造 LightGBM 训练参数字典，支持 HANDLE_IMBALANCE。"""
    params = dict(LGB_PARAMS)
    n_estimators = int(params.pop("n_estimators", 500))
    params.pop("verbose", None)
    params["verbosity"] = -1
    params.pop("scale_pos_weight", None)

    method = _sampling_method_key(target_key)
    if method != "baseline":
        print(
            "    %s sampling=%s，禁用 scale_pos_weight，避免重复处理类别不平衡"
            % (target_key, method)
        )
    elif HANDLE_IMBALANCE:
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        if n_pos > 0:
            spw = round(n_neg / n_pos, 2)
            params["scale_pos_weight"] = spw
            print(f"    scale_pos_weight = {n_neg}/{n_pos} = {spw:.2f}")
        else:
            print("    ⚠️  正样本数为 0，无法设置 scale_pos_weight")
    else:
        print("    HANDLE_IMBALANCE=False，不使用 scale_pos_weight")

    return params, n_estimators


def _train_lgb(
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
    categorical_features=None,
    target_key: str = "usage",
) -> lgb.Booster:
    print(f"  → 训练 {label} 模型…")
    params, n_rounds = _build_lgb_params(y, target_key)
    categorical_features = [
        col for col in (categorical_features or []) if col in X.columns
    ]
    ds = lgb.Dataset(
        X,
        label=y,
        feature_name=X.columns.tolist(),
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    train_kwargs = {
        "num_boost_round": n_rounds,
        "valid_sets": [ds],
    }
    if hasattr(lgb, "log_evaluation"):
        train_kwargs["callbacks"] = [lgb.log_evaluation(period=50)]
    else:
        train_kwargs["verbose_eval"] = 50
    booster = lgb.train(params, ds, **train_kwargs)
    return booster


def _predict_model(model, X):
    """Predict with one LightGBM booster or average an ensemble's probabilities."""
    if isinstance(model, (list, tuple)):
        if not model:
            raise ValueError("模型集成不能为空。")
        return np.mean([member.predict(X) for member in model], axis=0)
    return model.predict(X)


def _save_model_bundle(model, path, target_key):
    """Preserve the old single-model path and save ensemble members with a manifest."""
    if not isinstance(model, (list, tuple)):
        model.save_model(path)
        return [path]

    stem, ext = os.path.splitext(path)
    ext = ext or ".txt"
    member_paths = []
    for index, member in enumerate(model, start=1):
        member_path = "%s_member_%02d%s" % (stem, index, ext)
        member.save_model(member_path)
        member_paths.append(member_path)
    manifest_path = stem + "_ensemble.json"
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(
            {
                "aggregation": "mean_probability",
                "sampling_method": _sampling_method_key(target_key),
                "members": member_paths,
            },
            manifest_file,
            ensure_ascii=False,
            indent=2,
        )
    return member_paths + [manifest_path]


def _build_grid() -> np.ndarray:
    _base_grid = np.arange(GRID_MIN, GRID_MAX + GRID_STEP, GRID_STEP, dtype=np.float32)
    _base_grid = _base_grid[_base_grid >= GRID_MIN]
    if INCLUDE_ZERO:
        grid = np.concatenate([[np.float32(0.0)], _base_grid]) if GRID_MIN > 0 else _base_grid
    else:
        grid = _base_grid[_base_grid > 0.0]
    grid = np.sort(np.unique(grid)).astype(np.float32)
    print(f"  网格: {float(grid[0]):,.0f} ~ {float(grid[-1]):,.0f}, "
          f"步长={GRID_STEP:.0f}, 点数={len(grid)}")
    return grid


def _predict_probability_grid(
    X: pd.DataFrame,
    booster_usage: lgb.Booster,
    booster_default: lgb.Booster,
    grid: np.ndarray,
):
    feat_names = X.columns.tolist()
    if QUOTA_COL not in feat_names:
        raise ValueError(f"特征中找不到额度列 {QUOTA_COL}，实际特征: {feat_names}")
    quota_col_idx = feat_names.index(QUOTA_COL)

    X_base_np = X.to_numpy(dtype=np.float32)
    n, m = X_base_np.shape[0], len(grid)
    p_usage = np.empty((n, m), dtype=np.float32)
    p_default = np.empty((n, m), dtype=np.float32)

    print(f"  PDP 扫描: {n:,} 客户 x {m} 额度点...")
    X_scan = X_base_np.copy()
    for j, L in enumerate(grid):
        X_scan[:, quota_col_idx] = float(L)
        p_usage[:, j] = _predict_model(booster_usage, X_scan).astype(np.float32)
        p_default[:, j] = _predict_model(booster_default, X_scan).astype(np.float32)
        if (j + 1) % 200 == 0 or j + 1 == m:
            print(f"    进度: {j+1}/{m} ({(j+1)/m*100:.1f}%)")

    return np.clip(p_usage, 0.0, 1.0), np.clip(p_default, 0.0, 1.0)


def _predict_probabilities_at_grid_indices(
    X: pd.DataFrame,
    booster_usage,
    booster_default,
    grid: np.ndarray,
    grid_indices,
):
    """只在每位客户的一个指定额度点预测，用于完整校准集拟合 Isotonic。"""
    feat_names = X.columns.tolist()
    if QUOTA_COL not in feat_names:
        raise ValueError(f"特征中找不到额度列 {QUOTA_COL}")
    grid_indices = np.asarray(grid_indices, dtype=int).reshape(-1)
    if len(grid_indices) != len(X):
        raise ValueError("单点概率预测的客户数与额度下标数不一致")
    X_point = X.to_numpy(dtype=np.float32)
    X_point[:, feat_names.index(QUOTA_COL)] = grid[grid_indices]
    usage = _predict_model(booster_usage, X_point).astype(np.float32)
    default = _predict_model(booster_default, X_point).astype(np.float32)
    return np.clip(usage, 0.0, 1.0), np.clip(default, 0.0, 1.0)


def _relative_indices(source_rows, selected_source_rows):
    """把完整客户表行号转换为抽样概率网格中的相对行号。"""
    source_rows = np.asarray(source_rows, dtype=int).reshape(-1)
    selected = set(np.asarray(selected_source_rows, dtype=int).reshape(-1).tolist())
    return np.asarray(
        [position for position, source in enumerate(source_rows) if int(source) in selected],
        dtype=int,
    )


def _grid_metadata(source_customer_count: int, grid_customer_count: int, scope: str):
    return {
        "probability_grid_sampled": bool(PROBABILITY_GRID_SAMPLE_ENABLED),
        "sample_size": (
            None if PROBABILITY_GRID_SAMPLE_SIZE is None
            else int(PROBABILITY_GRID_SAMPLE_SIZE)
        ),
        "sample_random_state": int(PROBABILITY_GRID_SAMPLE_RANDOM_STATE),
        "source_customer_count": int(source_customer_count),
        "grid_customer_count": int(grid_customer_count),
        "grid_scope": str(scope),
        "training_customer_scope": "full configured train/build folds",
        "calibration_customer_scope": "full configured calibration/build rows",
    }


def _save_grid_lineage(
    output_dir: str,
    work: pd.DataFrame,
    source_rows,
    scope_labels,
    metadata,
):
    """保存抽样概率网格的原始行号、客户ID、用途和配置。"""
    source_rows = np.asarray(source_rows, dtype=int).reshape(-1)
    np.save(os.path.join(output_dir, "source_row_indices.npy"), source_rows)
    manifest = pd.DataFrame({
        "row_index": np.arange(len(source_rows), dtype=int),
        "source_row_index": source_rows,
        "cst_id": work.iloc[source_rows]["cst_id"].astype(str).to_numpy(),
        "optimization_scope": np.asarray(scope_labels, dtype=object),
    })
    manifest.to_csv(
        os.path.join(output_dir, "optimization_sample_manifest.csv"),
        index=False,
        encoding=CSV_ENCODING,
    )
    with open(os.path.join(output_dir, "grid_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _nearest_grid_indices(values, grid: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    idx = np.searchsorted(grid, values)
    idx = np.clip(idx, 0, len(grid) - 1)
    left = np.clip(idx - 1, 0, len(grid) - 1)
    use_left = np.abs(grid[left] - values) <= np.abs(grid[idx] - values)
    return np.where(use_left, left, idx).astype(int)


def _fit_isotonic_compat(raw_probability, target, label):
    """Fit Isotonic with matching dtypes for older scikit-learn/Cython builds."""
    x = np.ascontiguousarray(np.asarray(raw_probability, dtype=np.float64).reshape(-1))
    y = np.ascontiguousarray(np.asarray(target, dtype=np.float64).reshape(-1))
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "%s Isotonic 输入长度不一致: probability=%d, target=%d"
            % (label, x.shape[0], y.shape[0])
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("%s Isotonic 输入包含 NaN/Inf" % label)
    if np.unique(y).size < 2:
        raise ValueError("%s Isotonic 标签至少需要同时包含正负样本" % label)
    return IsotonicRegression(out_of_bounds="clip").fit(x, y)


def _calibrate_matrix(calibrator, matrix, chunk_size=1000000):
    flat = np.asarray(matrix).reshape(-1)
    out = np.empty(flat.shape, dtype=np.float32)
    for start in range(0, len(flat), chunk_size):
        end = min(start + chunk_size, len(flat))
        out[start:end] = calibrator.predict(flat[start:end]).astype(np.float32)
    return out.reshape(matrix.shape)


def _sample_training_only(X, y, label, categorical_features, target_key):
    """Apply the fixed task-2 sampler to model-training rows only."""
    config = SAMPLING_CONFIG[target_key]
    method = _sampling_method_key(target_key)
    if method == "baseline":
        print("    %s: no resampling" % label)
        return X, y

    if method in ("balance_cascade", "easy_ensemble"):
        sampler = sampler_factory(
            method,
            random_state=RANDOM_STATE,
            n_estimators=config["n_estimators"],
            ratio=config["ensemble_ratio"],
            categorical_features=categorical_features,
        )
    else:
        sampler = sampler_factory(
            method,
            sampling_strategy=config["strategy"],
            random_state=RANDOM_STATE,
            categorical_features=categorical_features,
        )

    if method == "balance_cascade":
        # The actual LightGBM members trained below must drive each pruning transition.
        return sampler.initialize(X, y)

    sampled = sampler.fit_resample(X, y)
    if isinstance(sampled, list):
        for index, (_, y_subset) in enumerate(sampled, start=1):
            print_sampling_summary(y, y_subset, "%s/%s subset%d" % (label, method, index))
        return sampled

    X_res, y_res = sampled
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(np.asarray(y_res, dtype=int))
    print_sampling_summary(y, y_res, "%s/%s" % (label, method))
    return X_res, y_res


def _train_sampled(sampled, label, categorical_features, target_key):
    """Train one model, Easy Ensemble members, or a classifier-driven cascade."""
    if isinstance(sampled, BalanceCascade):
        models = []
        while sampled.has_next_subset():
            model_index = len(models) + 1
            X_subset, y_subset = sampled.next_subset()
            print_sampling_summary(
                sampled._y,
                y_subset,
                "%s/balance_cascade subset%d" % (label, model_index),
            )
            model = _train_lgb(
                X_subset,
                y_subset,
                "%s cascade %d/%d" % (
                    label,
                    model_index,
                    sampled.effective_n_estimators_,
                ),
                categorical_features,
                target_key,
            )
            models.append(model)
            remaining_probability = _predict_model(model, sampled.remaining_X())
            cascade_info = sampled.update(remaining_probability, score_label=1)
            if cascade_info["pruned_for_next_stage"]:
                print(
                    "    cascade stage %d: target_fpr=%.4f, threshold=%.6f, "
                    "next majority=%d"
                    % (
                        cascade_info["stage"],
                        cascade_info["target_fpr"],
                        cascade_info["threshold"],
                        cascade_info["remaining_majority"],
                    )
                )
        return models

    if isinstance(sampled, list):
        return [
            _train_lgb(
                X_subset,
                y_subset,
                "%s ensemble %d/%d" % (label, index, len(sampled)),
                categorical_features,
                target_key,
            )
            for index, (X_subset, y_subset) in enumerate(sampled, start=1)
        ]

    X_train, y_train = sampled
    return _train_lgb(
        X_train, y_train, label, categorical_features, target_key
    )


def _crossfit_stratified_folds(work):
    """Create mutually exclusive joint-stratified random outer folds."""
    fold_parts = stratified_dual_target_partition_indices(
        work,
        np.repeat(1.0 / CROSS_FIT_FOLDS, CROSS_FIT_FOLDS),
        random_state=RANDOM_STATE,
    )
    fold_id = np.empty(len(work), dtype=np.int16)
    for fold, idx in enumerate(fold_parts):
        fold_id[idx] = fold
    if "split_eff_date" in work.columns:
        dates = pd.to_datetime(work["split_eff_date"], errors="coerce")
    else:
        dates = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns]")
    return dates, fold_id


def _run_crossfit():
    if CROSS_FIT_FOLDS != 5:
        print("  NOTE: requested folds=%d (method description normally uses 5)" % CROSS_FIT_FOLDS)
    if INNER_CROSS_FIT_FOLDS < 2:
        raise ValueError("INNER_CROSS_FIT_FOLDS must be at least 2")

    work = _load_or_preprocess().reset_index(drop=True)
    dates, fold_id = _crossfit_stratified_folds(work)
    y_usage = work["y_freq"].astype(int).reset_index(drop=True)
    y_default = work["y_dq_risk"].astype(int).reset_index(drop=True)
    customer_id_full = work["cst_id"].astype(str).to_numpy()
    grid = _build_grid()
    os.makedirs(CROSS_FIT_DIR, exist_ok=True)

    if PROBABILITY_GRID_SAMPLE_ENABLED:
        if PROBABILITY_GRID_SAMPLE_SIZE is None or int(PROBABILITY_GRID_SAMPLE_SIZE) <= 0:
            raise ValueError("概率网格抽样开启时 sample_size 必须为正整数")
        sample_plan = build_optimization_sample_plan(
            work,
            sample_size=int(PROBABILITY_GRID_SAMPLE_SIZE),
            random_state=PROBABILITY_GRID_SAMPLE_RANDOM_STATE,
        )
        output_source_idx = np.asarray(sample_plan["all"], dtype=int)
        print(
            "  [概率网格抽样] crossfit all: %d -> %d 客户"
            % (len(work), len(output_source_idx))
        )
    else:
        output_source_idx = np.arange(len(work), dtype=int)
    customer_id = customer_id_full[output_source_idx]
    source_to_output = np.full(len(work), -1, dtype=int)
    source_to_output[output_source_idx] = np.arange(len(output_source_idx), dtype=int)

    # Disk-backed matrices avoid holding all-customer x all-limit grids in RAM.
    p_usage_out = open_memmap(
        os.path.join(CROSS_FIT_DIR, "p_usage.npy"), mode="w+", dtype="float32",
        shape=(len(output_source_idx), len(grid)),
    )
    p_default_out = open_memmap(
        os.path.join(CROSS_FIT_DIR, "p_default.npy"), mode="w+", dtype="float32",
        shape=(len(output_source_idx), len(grid)),
    )
    p_usage_raw_out = open_memmap(
        os.path.join(CROSS_FIT_DIR, "p_usage_raw.npy"), mode="w+", dtype="float32",
        shape=(len(output_source_idx), len(grid)),
    )
    p_default_raw_out = open_memmap(
        os.path.join(CROSS_FIT_DIR, "p_default_raw.npy"), mode="w+", dtype="float32",
        shape=(len(output_source_idx), len(grid)),
    )
    manifest_rows = []
    preprocessing_all = {}

    all_rows = np.arange(len(work), dtype=int)
    for fold in range(CROSS_FIT_FOLDS):
        pred_idx = all_rows[fold_id == fold]
        build_idx = all_rows[fold_id != fold]
        print("\n[OUTER CROSS-FIT %d/%d] build/predict=%d/%d; inner calibration folds=%d" % (
            fold + 1, CROSS_FIT_FOLDS, len(build_idx), len(pred_idx), INNER_CROSS_FIT_FOLDS
        ))

        # 在外层建模数据内部再次交叉拟合，得到每个 build 客户在历史额度点的
        # 内部折外未校准概率；目标外层折标签不参与这里任何训练。
        raw_u_inner = np.full(len(work), np.nan, dtype=float)
        raw_d_inner = np.full(len(work), np.nan, dtype=float)
        inner_relative_parts = stratified_dual_target_partition_indices(
            work.iloc[build_idx].reset_index(drop=True),
            np.repeat(1.0 / INNER_CROSS_FIT_FOLDS, INNER_CROSS_FIT_FOLDS),
            random_state=RANDOM_STATE + fold + 1,
        )
        inner_parts = [build_idx[np.asarray(x, dtype=int)] for x in inner_relative_parts]
        inner_metadata = {}
        for inner_fold, inner_pred_idx in enumerate(inner_parts):
            inner_train_idx = np.setdiff1d(build_idx, inner_pred_idx, assume_unique=False)
            print("  [INNER %d/%d] train/predict=%d/%d" % (
                inner_fold + 1, INNER_CROSS_FIT_FOLDS, len(inner_train_idx), len(inner_pred_idx)
            ))
            X_inner, metadata_inner = _fit_transform_features_train_only(
                work,
                inner_train_idx,
                fit_scope="outer_%d_inner_%d_train_only" % (fold + 1, inner_fold + 1),
            )
            categorical_inner = metadata_inner["categorical_features"]
            sampled_u_inner = _sample_training_only(
                X_inner.iloc[inner_train_idx],
                y_usage.iloc[inner_train_idx],
                "usage",
                categorical_inner,
                "usage",
            )
            sampled_d_inner = _sample_training_only(
                X_inner.iloc[inner_train_idx],
                y_default.iloc[inner_train_idx],
                "default",
                categorical_inner,
                "default",
            )
            booster_u_inner = _train_sampled(
                sampled_u_inner,
                "usage outer %d inner %d" % (fold + 1, inner_fold + 1),
                categorical_inner,
                "usage",
            )
            booster_d_inner = _train_sampled(
                sampled_d_inner,
                "default outer %d inner %d" % (fold + 1, inner_fold + 1),
                categorical_inner,
                "default",
            )
            inner_limit_idx = _nearest_grid_indices(work.iloc[inner_pred_idx][QUOTA_COL], grid)
            X_inner_pred = X_inner.iloc[inner_pred_idx].copy()
            X_inner_pred.loc[:, QUOTA_COL] = grid[inner_limit_idx]
            raw_u_inner[inner_pred_idx] = _predict_model(booster_u_inner, X_inner_pred)
            raw_d_inner[inner_pred_idx] = _predict_model(booster_d_inner, X_inner_pred)
            inner_metadata["inner_%d" % (inner_fold + 1)] = metadata_inner

        if np.isnan(raw_u_inner[build_idx]).any() or np.isnan(raw_d_inner[build_idx]).any():
            raise RuntimeError("outer fold %d 内部折外概率未完整覆盖 build 数据" % (fold + 1))
        if y_usage.iloc[build_idx].nunique() < 2 or y_default.iloc[build_idx].nunique() < 2:
            raise ValueError("outer fold %d 建模数据标签只有一个类别，无法校准" % (fold + 1))
        iso_u = _fit_isotonic_compat(
            raw_u_inner[build_idx], y_usage.iloc[build_idx],
            "outer fold %d usage" % (fold + 1),
        )
        iso_d = _fit_isotonic_compat(
            raw_d_inner[build_idx], y_default.iloc[build_idx],
            "outer fold %d default" % (fold + 1),
        )

        # 校准器固定后，在全部外层建模数据上训练最终外层模型，再为目标折生成网格。
        X_outer, metadata_outer = _fit_transform_features_train_only(
            work, build_idx, fit_scope="outer_%d_all_build" % (fold + 1)
        )
        categorical_outer = metadata_outer["categorical_features"]
        sampled_u_outer = _sample_training_only(
            X_outer.iloc[build_idx],
            y_usage.iloc[build_idx],
            "usage",
            categorical_outer,
            "usage",
        )
        sampled_d_outer = _sample_training_only(
            X_outer.iloc[build_idx],
            y_default.iloc[build_idx],
            "default",
            categorical_outer,
            "default",
        )
        booster_u = _train_sampled(
            sampled_u_outer,
            "usage outer %d final" % (fold + 1),
            categorical_outer,
            "usage",
        )
        booster_d = _train_sampled(
            sampled_d_outer,
            "default outer %d final" % (fold + 1),
            categorical_outer,
            "default",
        )
        grid_pred_idx = pred_idx[source_to_output[pred_idx] >= 0]
        print(
            "  [概率网格目标行] outer fold %d: %d/%d"
            % (fold + 1, len(grid_pred_idx), len(pred_idx))
        )
        if len(grid_pred_idx):
            raw_u_pred, raw_d_pred = _predict_probability_grid(
                X_outer.iloc[grid_pred_idx], booster_u, booster_d, grid
            )
            output_positions = source_to_output[grid_pred_idx]
            p_usage_raw_out[output_positions, :] = raw_u_pred
            p_default_raw_out[output_positions, :] = raw_d_pred
            p_usage_out[output_positions, :] = _calibrate_matrix(iso_u, raw_u_pred)
            p_default_out[output_positions, :] = _calibrate_matrix(iso_d, raw_d_pred)
        p_usage_out.flush()
        p_default_out.flush()
        p_usage_raw_out.flush()
        p_default_raw_out.flush()
        preprocessing_all["fold_%d" % (fold + 1)] = {
            "outer_final": metadata_outer,
            "inner_crossfit": inner_metadata,
        }
        for idx in grid_pred_idx:
            manifest_rows.append({
                "row_index": int(source_to_output[idx]),
                "source_row_index": int(idx),
                "cst_id": customer_id_full[idx],
                "fold": fold + 1, "role": "out_of_fold_prediction",
                "split_eff_date": dates.iloc[idx],
            })

    np.save(os.path.join(CROSS_FIT_DIR, "customer_id.npy"), customer_id)
    np.save(os.path.join(CROSS_FIT_DIR, "grid.npy"), grid)
    work.iloc[output_source_idx].reset_index(drop=True).to_csv(
        os.path.join(CROSS_FIT_DIR, "work_features.csv"),
        index=False,
        encoding=CSV_ENCODING,
    )
    pd.DataFrame(manifest_rows).sort_values("row_index").to_csv(
        os.path.join(CROSS_FIT_DIR, "split_manifest.csv"), index=False, encoding=CSV_ENCODING
    )
    _save_grid_lineage(
        CROSS_FIT_DIR,
        work,
        output_source_idx,
        np.repeat("all", len(output_source_idx)),
        _grid_metadata(len(work), len(output_source_idx), "all"),
    )
    with open(os.path.join(CROSS_FIT_DIR, "feature_preprocessing.json"), "w", encoding="utf-8") as f:
        json.dump(preprocessing_all, f, ensure_ascii=False, indent=2)
    with open(os.path.join(CROSS_FIT_DIR, "calibration_method.txt"), "w", encoding="utf-8") as f:
        f.write(
            "%d-fold joint-stratified random outer cross-fitting; "
            "%d-fold joint-stratified random inner OOF calibration; isotonic\n"
            % (CROSS_FIT_FOLDS, INNER_CROSS_FIT_FOLDS)
        )
    print("\nCross-fitted calibrated probability grid saved to %s" % CROSS_FIT_DIR)


if CROSS_FIT_MODE:
    _run_crossfit()


# ── 主流程 ───────────────────────────────────────────────────────────
if not CROSS_FIT_MODE:
    work = _load_or_preprocess()
    work, train_idx, validation_idx, cal_idx, test_idx, fit_idx = _stratified_split_indices(work)
    print(f"  y_freq 正例: {work['y_freq'].mean():.2%}  |  "
          f"y_dq_risk 正例: {work['y_dq_risk'].mean():.2%}")
    print(f"  y_freq 模式: {Y_FREQ_MODE}")

    print("\n[2/5] 无泄漏特征工程（采样方法确定后，仅在训练集+验证集75%上拟合处理规则）…")
    X_usage, preprocessing_metadata = _fit_transform_features_train_only(
        work, fit_idx, fit_scope="train_plus_validation_after_sampling_selection"
    )
    X_default = X_usage
    y_usage = work["y_freq"].astype(int)
    y_default = work["y_dq_risk"].astype(int)
    feat_names = X_usage.columns.tolist()
    print(f"  特征数={len(feat_names)}")

    print("\n[3/5] 训练 LightGBM（使用训练集+验证集75%，固定采样仅作用于开发拟合集）…")
    categorical_features = preprocessing_metadata["categorical_features"]
    sampled_usage = _sample_training_only(
        X_usage.iloc[fit_idx], y_usage.iloc[fit_idx], "usage", categorical_features,
        "usage",
    )
    sampled_default = _sample_training_only(
        X_default.iloc[fit_idx], y_default.iloc[fit_idx], "default", categorical_features,
        "default",
    )
    booster_usage = _train_sampled(
        sampled_usage, "y_freq (usage)", categorical_features, "usage"
    )
    booster_default = _train_sampled(
        sampled_default, "y_dq_risk (default)", categorical_features, "default"
    )

    os.makedirs("saved_models", exist_ok=True)
    _save_model_bundle(booster_usage, "saved_models/booster_usage.txt", "usage")
    _save_model_bundle(booster_default, "saved_models/booster_default.txt", "default")
    print("  模型已保存到 saved_models/")

    print("\n[4/6] 固定概率网格客户名单…")
    if PROBABILITY_GRID_SAMPLE_ENABLED:
        if PROBABILITY_GRID_SAMPLE_SIZE is None or int(PROBABILITY_GRID_SAMPLE_SIZE) <= 0:
            raise ValueError("概率网格抽样开启时 sample_size 必须为正整数")
        sample_plan = build_optimization_sample_plan(
            work,
            sample_size=int(PROBABILITY_GRID_SAMPLE_SIZE),
            random_state=PROBABILITY_GRID_SAMPLE_RANDOM_STATE,
            fit_indices=fit_idx,
            test_indices=test_idx,
        )
        selected_fit_source = np.asarray(sample_plan["fit"], dtype=int)
        selected_test_source = np.asarray(sample_plan["test"], dtype=int)
        output_source_idx = np.sort(np.unique(np.concatenate([
            selected_fit_source, selected_test_source,
        ]))).astype(int)
        print(
            "  [概率网格抽样] fit=%d/%d, test=%d/%d, dev grid union=%d"
            % (
                len(selected_fit_source), len(fit_idx),
                len(selected_test_source), len(test_idx),
                len(output_source_idx),
            )
        )
    else:
        selected_fit_source = np.asarray(fit_idx, dtype=int)
        selected_test_source = np.asarray(test_idx, dtype=int)
        output_source_idx = np.arange(len(work), dtype=int)

    grid_train_idx = _relative_indices(output_source_idx, train_idx)
    grid_validation_idx = _relative_indices(output_source_idx, validation_idx)
    grid_fit_idx = _relative_indices(output_source_idx, selected_fit_source)
    grid_cal_idx = _relative_indices(output_source_idx, cal_idx)
    grid_test_idx = _relative_indices(output_source_idx, selected_test_source)
    customer_id = work.iloc[output_source_idx]["cst_id"].astype(str).to_numpy()

    print("\n[5/6] 使用完整校准集拟合 Isotonic，再为固定客户名单生成概率网格…")
    grid = _build_grid()
    original_limits = pd.to_numeric(work[QUOTA_COL], errors="coerce").fillna(0.0).to_numpy()
    cal_grid_idx = _nearest_grid_indices(original_limits[cal_idx], grid)
    raw_usage_cal, raw_default_cal = _predict_probabilities_at_grid_indices(
        X_usage.iloc[cal_idx], booster_usage, booster_default, grid, cal_grid_idx
    )
    if y_usage.iloc[cal_idx].nunique() < 2 or y_default.iloc[cal_idx].nunique() < 2:
        raise ValueError("开发阶段校准集至少需要同时包含正负样本")
    usage_dev_calibrator = _fit_isotonic_compat(
        raw_usage_cal, y_usage.iloc[cal_idx], "development usage"
    )
    default_dev_calibrator = _fit_isotonic_compat(
        raw_default_cal, y_default.iloc[cal_idx], "development default"
    )
    p_usage, p_default = _predict_probability_grid(
        X_usage.iloc[output_source_idx], booster_usage, booster_default, grid
    )
    p_usage_dev_calibrated = _calibrate_matrix(usage_dev_calibrator, p_usage)
    p_default_dev_calibrated = _calibrate_matrix(default_dev_calibrator, p_default)

    split_name_full = np.empty(len(work), dtype=object)
    split_name_full[train_idx] = "train"
    split_name_full[validation_idx] = "validation"
    split_name_full[cal_idx] = "cal"
    split_name_full[test_idx] = "test"
    split_manifest_full = pd.DataFrame({
        "row_index": np.arange(len(work), dtype=int),
        "cst_id": work["cst_id"].astype(str).to_numpy(),
        "split": split_name_full,
        "split_eff_date": pd.to_datetime(work["split_eff_date"], errors="coerce"),
        "y_freq": y_usage.to_numpy(dtype=int),
        "y_dq_risk": y_default.to_numpy(dtype=int),
    })
    split_manifest = split_manifest_full.iloc[output_source_idx].copy().reset_index(drop=True)
    split_manifest.insert(0, "source_row_index", output_source_idx)
    split_manifest["row_index"] = np.arange(len(split_manifest), dtype=int)

    scope_labels = np.full(len(output_source_idx), "calibration_only", dtype=object)
    scope_labels[np.isin(output_source_idx, selected_fit_source)] = "fit"
    scope_labels[np.isin(output_source_idx, selected_test_source)] = "test"
    metadata = _grid_metadata(len(work), len(output_source_idx), "fit_plus_test")

    print("\n[6/6] 保存抽样口径、校准前后概率网格和完整开发参数样本元数据…")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "customer_id.npy"), customer_id)
    np.save(os.path.join(OUT_DIR, "grid.npy"), grid)
    np.save(os.path.join(OUT_DIR, "p_usage.npy"), p_usage)
    np.save(os.path.join(OUT_DIR, "p_default.npy"), p_default)
    for name, value in [
        ("train_idx.npy", grid_train_idx),
        ("validation_idx.npy", grid_validation_idx),
        ("fit_idx.npy", grid_fit_idx),
        ("cal_idx.npy", grid_cal_idx),
        ("test_idx.npy", grid_test_idx),
    ]:
        np.save(os.path.join(OUT_DIR, name), value)
    grid_work = work.iloc[output_source_idx].reset_index(drop=True)
    grid_work.to_csv(
        os.path.join(OUT_DIR, "work_features.csv"), index=False, encoding=CSV_ENCODING
    )
    work.to_csv(
        os.path.join(OUT_DIR, "full_work_features.csv"), index=False, encoding=CSV_ENCODING
    )
    for name, value in [
        ("full_train_idx.npy", train_idx),
        ("full_validation_idx.npy", validation_idx),
        ("full_fit_idx.npy", fit_idx),
        ("full_cal_idx.npy", cal_idx),
        ("full_test_idx.npy", test_idx),
    ]:
        np.save(os.path.join(OUT_DIR, name), np.asarray(value, dtype=int))
    split_manifest.to_csv(
        os.path.join(OUT_DIR, "split_manifest.csv"), index=False, encoding=CSV_ENCODING
    )
    split_manifest_full.to_csv(
        os.path.join(OUT_DIR, "full_split_manifest.csv"), index=False, encoding=CSV_ENCODING
    )
    _save_grid_lineage(OUT_DIR, work, output_source_idx, scope_labels, metadata)
    with open(os.path.join(OUT_DIR, "feature_preprocessing.json"), "w", encoding="utf-8") as f:
        json.dump(preprocessing_metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feat_names))

    os.makedirs(DEV_CALIBRATED_DIR, exist_ok=True)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "customer_id.npy"), customer_id)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "grid.npy"), grid)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "p_usage.npy"), p_usage_dev_calibrated)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "p_default.npy"), p_default_dev_calibrated)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "p_usage_raw.npy"), p_usage)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "p_default_raw.npy"), p_default)
    for name, value in [
        ("train_idx.npy", grid_train_idx),
        ("validation_idx.npy", grid_validation_idx),
        ("fit_idx.npy", grid_fit_idx),
        ("cal_idx.npy", grid_cal_idx),
        ("test_idx.npy", grid_test_idx),
    ]:
        np.save(os.path.join(DEV_CALIBRATED_DIR, name), value)
    grid_work.to_csv(
        os.path.join(DEV_CALIBRATED_DIR, "work_features.csv"),
        index=False,
        encoding=CSV_ENCODING,
    )
    work.to_csv(
        os.path.join(DEV_CALIBRATED_DIR, "full_work_features.csv"),
        index=False,
        encoding=CSV_ENCODING,
    )
    for name, value in [
        ("full_train_idx.npy", train_idx),
        ("full_validation_idx.npy", validation_idx),
        ("full_fit_idx.npy", fit_idx),
        ("full_cal_idx.npy", cal_idx),
        ("full_test_idx.npy", test_idx),
    ]:
        np.save(os.path.join(DEV_CALIBRATED_DIR, name), np.asarray(value, dtype=int))
    split_manifest.to_csv(
        os.path.join(DEV_CALIBRATED_DIR, "split_manifest.csv"),
        index=False,
        encoding=CSV_ENCODING,
    )
    split_manifest_full.to_csv(
        os.path.join(DEV_CALIBRATED_DIR, "full_split_manifest.csv"),
        index=False,
        encoding=CSV_ENCODING,
    )
    _save_grid_lineage(
        DEV_CALIBRATED_DIR, work, output_source_idx, scope_labels, metadata
    )
    with open(os.path.join(DEV_CALIBRATED_DIR, "feature_preprocessing.json"), "w", encoding="utf-8") as f:
        json.dump(preprocessing_metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DEV_CALIBRATED_DIR, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feat_names))
    with open(os.path.join(DEV_CALIBRATED_DIR, "calibration_method.txt"), "w", encoding="utf-8") as f:
        f.write(
            "train_plus_validation_model_after_sampling_selection; "
            "independent_calibration_set; isotonic; final_test_untouched\n"
        )

    print(f"\n✓ 开发阶段校准概率网格已保存到 {DEV_CALIBRATED_DIR}/")
    print(f"  p_usage.npy      shape={p_usage_dev_calibrated.shape}")
    print(f"  p_default.npy    shape={p_default_dev_calibrated.shape}")
    print(
        "  模型拟合范围: train+validation；校准器拟合范围: cal；"
        "test 仅供一次性离线评价"
    )
    print(
        "  概率网格客户范围: %d/%d；模型训练和校准仍使用完整配置范围"
        % (len(output_source_idx), len(work))
    )
