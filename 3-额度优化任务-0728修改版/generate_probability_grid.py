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
    dedup_by_cst_loan,
    cast_cat_cols,
    rename_kechuang_cols,
    filter_by_maturity,
    filter_by_eff_date,
    filter_dq_start_customers,
    aggregate_by_customer_potential,
    clean_data_potential,
    build_labels_potential,
    drop_post_label_cols,
)
from sampling_methods import sampler_factory, print_sampling_summary

# ── 参数（可被 notebook *_OVERRIDE 覆盖）────────────────────────────
CLEANED_FILE          = globals().get("CLEANED_FILE_OVERRIDE",          "data_cleaned.csv")
EXCEL_PATH            = globals().get("EXCEL_PATH_OVERRIDE",            "kechuang_talent_full_data0527v1.csv")
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
MATURITY_CUTOFF       = globals().get("MATURITY_CUTOFF_OVERRIDE",        "2025-12-31")
APPLY_EFF_DATE_FILTER = bool(globals().get("APPLY_EFF_DATE_FILTER_OVERRIDE", True))
EFF_DATE_LOWER        = globals().get("EFF_DATE_LOWER_OVERRIDE",         "2024-10-01")
EFF_DATE_UPPER        = globals().get("EFF_DATE_UPPER_OVERRIDE",         "2026-03-31")
Y_FREQ_MODE           = globals().get("Y_FREQ_MODE_OVERRIDE",            "bout_gt0_and_curr_p80")
HANDLE_IMBALANCE      = bool(globals().get("HANDLE_IMBALANCE_OVERRIDE",   True))
EARLY_STOPPING_ROUNDS = int(globals().get("EARLY_STOPPING_ROUNDS_OVERRIDE", 50))
RANDOM_STATE          = int(globals().get("RANDOM_STATE_OVERRIDE",         42))
SPLIT_RATIOS          = tuple(globals().get("SPLIT_RATIOS_OVERRIDE", (0.60, 0.15, 0.15, 0.10)))
CROSS_FIT_MODE        = bool(globals().get("CROSS_FIT_MODE_OVERRIDE", False))
CROSS_FIT_FOLDS       = int(globals().get("CROSS_FIT_FOLDS_OVERRIDE", 5))
INNER_CROSS_FIT_FOLDS = int(globals().get("INNER_CROSS_FIT_FOLDS_OVERRIDE", 5))
CROSS_FIT_DIR         = globals().get("CROSS_FIT_DIR_OVERRIDE", OUT_DIR + "_crossfit_calibrated")
SAMPLING_METHOD       = globals().get("SAMPLING_METHOD_OVERRIDE", None)
SAMPLING_STRATEGY     = globals().get("SAMPLING_STRATEGY_OVERRIDE", 1.0)

_DEFAULT_LGB = {
    "objective":         "binary",
    "metric":            "auc",
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "max_depth":         -1,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "verbose":           -1,
}
LGB_PARAMS = globals().get("LGB_PARAMS_OVERRIDE", _DEFAULT_LGB)

QUOTA_COL = "credamt"


def _time_split_indices(work: pd.DataFrame):
    """Create a stable 60/15/15/10 chronological split before feature fitting."""
    if len(SPLIT_RATIOS) != 4 or not np.isclose(sum(SPLIT_RATIOS), 1.0):
        raise ValueError("SPLIT_RATIOS must contain four values summing to 1")
    if "split_eff_date" not in work.columns:
        raise ValueError(
            "data_cleaned.csv is missing split_eff_date; rerun Cell 3 before Cell 5"
        )

    dates = pd.to_datetime(work["split_eff_date"], errors="coerce")
    valid = dates.notna()
    if not valid.all():
        print("  WARNING: dropping %d rows with invalid split_eff_date" % int((~valid).sum()))
        work = work.loc[valid].reset_index(drop=True)
        dates = dates.loc[valid].reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)
        dates = dates.reset_index(drop=True)

    order = np.argsort(dates.to_numpy(), kind="mergesort")
    n = len(work)
    n_train = int(n * SPLIT_RATIOS[0])
    n_validation = int(n * SPLIT_RATIOS[1])
    n_cal = int(n * SPLIT_RATIOS[2])
    train_idx = np.asarray(order[:n_train], dtype=int)
    validation_end = n_train + n_validation
    cal_end = validation_end + n_cal
    validation_idx = np.asarray(order[n_train:validation_end], dtype=int)
    cal_idx = np.asarray(order[validation_end:cal_end], dtype=int)
    test_idx = np.asarray(order[cal_end:], dtype=int)
    fit_idx = np.sort(np.concatenate([train_idx, validation_idx])).astype(int)

    if min(len(train_idx), len(validation_idx), len(cal_idx), len(test_idx)) == 0:
        raise ValueError("At least one chronological split is empty")

    print(
        "  chronological split train/validation/cal/test = %d/%d/%d/%d "
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
        split_dates = dates.iloc[idx]
        print("    %-10s %s -> %s" % (name, split_dates.min(), split_dates.max()))
    return work, train_idx, validation_idx, cal_idx, test_idx, fit_idx


def _fit_transform_features_train_only(
    work: pd.DataFrame,
    train_idx,
    fit_scope: str = "train_only",
):
    """
    Fit every data-dependent feature rule on train only, then transform all rows.

    This mirrors feature_engineering_potential without leaking calibration/test
    missingness, medians, modes, or category vocabularies into model training.
    """
    drop_cols = {
        "cst_id", "cst_id0", "y_freq", "y_dq_risk",
        "rt_acct_stat_2_end", "split_eff_date",
    }
    X = work.drop(columns=[c for c in drop_cols if c in work.columns]).copy()
    dt_cols = list(
        X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    )
    if dt_cols:
        X = X.drop(columns=dt_cols)

    tier_order = {
        "F3": 1, "F3级": 1, "f3": 1,
        "F2": 2, "F2级": 2, "f2": 2,
        "F1": 3, "F1级": 3, "f1": 3,
        "E": 4, "E级": 4, "e": 4,
        "D": 5, "D级": 5, "d": 5,
        "C": 6, "C级": 6, "c": 6,
        "B": 7, "B级": 7, "b": 7,
        "A": 8, "A级": 8, "a": 8,
    }
    if "档位" in X.columns:
        X["档位"] = (
            X["档位"].astype(str).str.strip().map(tier_order).fillna(0).astype(int)
        )

    X_train = X.iloc[train_idx]
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    dropped_high_missing = []
    numeric_medians = {}
    for col in numeric_cols:
        miss_ratio = float(X_train[col].isna().mean())
        if miss_ratio >= 0.40:
            dropped_high_missing.append(col)
            continue
        median = X_train[col].median()
        numeric_medians[col] = float(median) if pd.notna(median) else 0.0

    if dropped_high_missing:
        X = X.drop(columns=dropped_high_missing)
        print(
            "  %s missingness removed %d columns (>=40%%): %s"
            % (fit_scope, len(dropped_high_missing), dropped_high_missing)
        )
    for col, median in numeric_medians.items():
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(median)

    categorical_cols = list(X.select_dtypes(include=["object"]).columns)
    category_metadata = {}
    for col in categorical_cols:
        train_col = X.iloc[train_idx][col]
        mode = train_col.mode(dropna=True)
        fill_value = str(mode.iloc[0]) if len(mode) else "未知"
        train_values = train_col.fillna(fill_value).astype(str)
        classes = sorted(train_values.unique().tolist())
        mapping = {value: i for i, value in enumerate(classes)}
        transformed = X[col].fillna(fill_value).astype(str).map(mapping)
        unseen_count = int(transformed.isna().sum())
        X[col] = transformed.fillna(-1).astype(int)
        category_metadata[col] = {
            "fill_value": fill_value,
            "classes": classes,
            "unseen_full_rows": unseen_count,
        }
        if unseen_count:
            print("  category %s: %d unseen values encoded as -1" % (
                col, unseen_count))

    X = X.select_dtypes(include=[np.number]).copy()
    metadata = {
        "fit_scope": fit_scope,
        "split_ratios": list(SPLIT_RATIOS),
        "dropped_high_missing_columns": dropped_high_missing,
        "numeric_medians": numeric_medians,
        "categorical_rules": category_metadata,
        "feature_names": X.columns.tolist(),
    }
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
    df0 = dedup_by_cst_loan(df0)
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
    df_clean, label_train_idx, _, _, _, _ = _time_split_indices(df_clean)
    label_train_mask = np.zeros(len(df_clean), dtype=bool)
    label_train_mask[label_train_idx] = True
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


def _build_lgb_params(y: pd.Series) -> dict:
    """从 LGB_PARAMS 构造 LightGBM 训练参数字典，支持 HANDLE_IMBALANCE。"""
    params = dict(LGB_PARAMS)
    n_estimators = int(params.pop("n_estimators", 500))
    params.pop("verbose", None)
    params["verbosity"] = -1

    if HANDLE_IMBALANCE:
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


def _train_lgb(X: pd.DataFrame, y: pd.Series, label: str) -> lgb.Booster:
    print(f"  → 训练 {label} 模型…")
    params, n_rounds = _build_lgb_params(y)
    ds = lgb.Dataset(X, label=y, feature_name=X.columns.tolist(), free_raw_data=False)
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
        p_usage[:, j] = booster_usage.predict(X_scan).astype(np.float32)
        p_default[:, j] = booster_default.predict(X_scan).astype(np.float32)
        if (j + 1) % 200 == 0 or j + 1 == m:
            print(f"    进度: {j+1}/{m} ({(j+1)/m*100:.1f}%)")

    return np.clip(p_usage, 0.0, 1.0), np.clip(p_default, 0.0, 1.0)


def _nearest_grid_indices(values, grid: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    idx = np.searchsorted(grid, values)
    idx = np.clip(idx, 0, len(grid) - 1)
    left = np.clip(idx - 1, 0, len(grid) - 1)
    use_left = np.abs(grid[left] - values) <= np.abs(grid[idx] - values)
    return np.where(use_left, left, idx).astype(int)


def _calibrate_matrix(calibrator, matrix, chunk_size=1000000):
    flat = np.asarray(matrix).reshape(-1)
    out = np.empty(flat.shape, dtype=np.float32)
    for start in range(0, len(flat), chunk_size):
        end = min(start + chunk_size, len(flat))
        out[start:end] = calibrator.predict(flat[start:end]).astype(np.float32)
    return out.reshape(matrix.shape)


def _sample_training_only(X, y, label):
    """Apply the development-stage sampler to model-training rows only."""
    method = None if SAMPLING_METHOD in (None, "", "none", "None") else str(SAMPLING_METHOD)
    if method is None:
        print("    %s: no resampling" % label)
        return X, y
    if method.lower() in ("balance_cascade", "easy_ensemble", "easyensemble", "ensemble"):
        raise ValueError(
            "Cross-fit currently expects a single resampled training set; "
            "choose a non-ensemble development method (for example smote/borderline_smote/random_under)."
        )
    sampler = sampler_factory(
        method,
        sampling_strategy=SAMPLING_STRATEGY,
        random_state=RANDOM_STATE,
    )
    X_res, y_res = sampler.fit_resample(X, y)
    X_res = pd.DataFrame(X_res, columns=X.columns)
    y_res = pd.Series(np.asarray(y_res, dtype=int))
    print_sampling_summary(y, y_res, "%s/%s" % (label, method))
    return X_res, y_res


def _crossfit_time_folds(work):
    """Five mutually exclusive, near-equal chronological folds by account opening date."""
    if "split_eff_date" not in work.columns:
        raise ValueError("data_cleaned.csv missing split_eff_date; rerun Cell 3")
    dates = pd.to_datetime(work["split_eff_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("split_eff_date contains %d invalid rows" % int(dates.isna().sum()))
    order = np.argsort(dates.to_numpy(), kind="mergesort")
    fold_id = np.empty(len(work), dtype=np.int16)
    for fold, idx in enumerate(np.array_split(order, CROSS_FIT_FOLDS)):
        fold_id[idx] = fold
    return dates, fold_id


def _run_crossfit():
    if CROSS_FIT_FOLDS != 5:
        print("  NOTE: requested folds=%d (method description normally uses 5)" % CROSS_FIT_FOLDS)
    if INNER_CROSS_FIT_FOLDS < 2:
        raise ValueError("INNER_CROSS_FIT_FOLDS must be at least 2")

    work = _load_or_preprocess().reset_index(drop=True)
    dates, fold_id = _crossfit_time_folds(work)
    y_usage = work["y_freq"].astype(int).reset_index(drop=True)
    y_default = work["y_dq_risk"].astype(int).reset_index(drop=True)
    customer_id = work["cst_id"].astype(str).to_numpy()
    grid = _build_grid()
    os.makedirs(CROSS_FIT_DIR, exist_ok=True)

    # Disk-backed matrices avoid holding all-customer x all-limit grids in RAM.
    p_usage_out = open_memmap(
        os.path.join(CROSS_FIT_DIR, "p_usage.npy"), mode="w+", dtype="float32",
        shape=(len(work), len(grid)),
    )
    p_default_out = open_memmap(
        os.path.join(CROSS_FIT_DIR, "p_default.npy"), mode="w+", dtype="float32",
        shape=(len(work), len(grid)),
    )
    p_usage_raw_out = open_memmap(
        os.path.join(CROSS_FIT_DIR, "p_usage_raw.npy"), mode="w+", dtype="float32",
        shape=(len(work), len(grid)),
    )
    p_default_raw_out = open_memmap(
        os.path.join(CROSS_FIT_DIR, "p_default_raw.npy"), mode="w+", dtype="float32",
        shape=(len(work), len(grid)),
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
        build_sorted = build_idx[np.argsort(dates.iloc[build_idx].to_numpy(), kind="mergesort")]
        inner_parts = [np.asarray(x, dtype=int) for x in np.array_split(build_sorted, INNER_CROSS_FIT_FOLDS)]
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
            X_u_inner, y_u_inner = _sample_training_only(
                X_inner.iloc[inner_train_idx], y_usage.iloc[inner_train_idx], "usage"
            )
            X_d_inner, y_d_inner = _sample_training_only(
                X_inner.iloc[inner_train_idx], y_default.iloc[inner_train_idx], "default"
            )
            booster_u_inner = _train_lgb(
                X_u_inner, y_u_inner, "usage outer %d inner %d" % (fold + 1, inner_fold + 1)
            )
            booster_d_inner = _train_lgb(
                X_d_inner, y_d_inner, "default outer %d inner %d" % (fold + 1, inner_fold + 1)
            )
            inner_limit_idx = _nearest_grid_indices(work.iloc[inner_pred_idx][QUOTA_COL], grid)
            X_inner_pred = X_inner.iloc[inner_pred_idx].copy()
            X_inner_pred.loc[:, QUOTA_COL] = grid[inner_limit_idx]
            raw_u_inner[inner_pred_idx] = booster_u_inner.predict(X_inner_pred)
            raw_d_inner[inner_pred_idx] = booster_d_inner.predict(X_inner_pred)
            inner_metadata["inner_%d" % (inner_fold + 1)] = metadata_inner

        if np.isnan(raw_u_inner[build_idx]).any() or np.isnan(raw_d_inner[build_idx]).any():
            raise RuntimeError("outer fold %d 内部折外概率未完整覆盖 build 数据" % (fold + 1))
        if y_usage.iloc[build_idx].nunique() < 2 or y_default.iloc[build_idx].nunique() < 2:
            raise ValueError("outer fold %d 建模数据标签只有一个类别，无法校准" % (fold + 1))
        iso_u = IsotonicRegression(out_of_bounds="clip").fit(
            raw_u_inner[build_idx], y_usage.iloc[build_idx]
        )
        iso_d = IsotonicRegression(out_of_bounds="clip").fit(
            raw_d_inner[build_idx], y_default.iloc[build_idx]
        )

        # 校准器固定后，在全部外层建模数据上训练最终外层模型，再为目标折生成网格。
        X_outer, metadata_outer = _fit_transform_features_train_only(
            work, build_idx, fit_scope="outer_%d_all_build" % (fold + 1)
        )
        X_u_outer, y_u_outer = _sample_training_only(
            X_outer.iloc[build_idx], y_usage.iloc[build_idx], "usage"
        )
        X_d_outer, y_d_outer = _sample_training_only(
            X_outer.iloc[build_idx], y_default.iloc[build_idx], "default"
        )
        booster_u = _train_lgb(X_u_outer, y_u_outer, "usage outer %d final" % (fold + 1))
        booster_d = _train_lgb(X_d_outer, y_d_outer, "default outer %d final" % (fold + 1))
        raw_u_pred, raw_d_pred = _predict_probability_grid(
            X_outer.iloc[pred_idx], booster_u, booster_d, grid
        )
        p_usage_raw_out[pred_idx, :] = raw_u_pred
        p_default_raw_out[pred_idx, :] = raw_d_pred
        p_usage_out[pred_idx, :] = _calibrate_matrix(iso_u, raw_u_pred)
        p_default_out[pred_idx, :] = _calibrate_matrix(iso_d, raw_d_pred)
        p_usage_out.flush()
        p_default_out.flush()
        p_usage_raw_out.flush()
        p_default_raw_out.flush()
        preprocessing_all["fold_%d" % (fold + 1)] = {
            "outer_final": metadata_outer,
            "inner_crossfit": inner_metadata,
        }
        for idx in pred_idx:
            manifest_rows.append({
                "row_index": int(idx), "cst_id": customer_id[idx],
                "fold": fold + 1, "role": "out_of_fold_prediction",
                "split_eff_date": dates.iloc[idx],
            })

    np.save(os.path.join(CROSS_FIT_DIR, "customer_id.npy"), customer_id)
    np.save(os.path.join(CROSS_FIT_DIR, "grid.npy"), grid)
    work.to_csv(os.path.join(CROSS_FIT_DIR, "work_features.csv"), index=False, encoding=CSV_ENCODING)
    pd.DataFrame(manifest_rows).sort_values("row_index").to_csv(
        os.path.join(CROSS_FIT_DIR, "split_manifest.csv"), index=False, encoding=CSV_ENCODING
    )
    with open(os.path.join(CROSS_FIT_DIR, "feature_preprocessing.json"), "w", encoding="utf-8") as f:
        json.dump(preprocessing_all, f, ensure_ascii=False, indent=2)
    with open(os.path.join(CROSS_FIT_DIR, "calibration_method.txt"), "w", encoding="utf-8") as f:
        f.write(
            "%d-fold chronological outer cross-fitting; %d-fold inner OOF calibration; isotonic\n"
            % (CROSS_FIT_FOLDS, INNER_CROSS_FIT_FOLDS)
        )
    print("\nCross-fitted calibrated probability grid saved to %s" % CROSS_FIT_DIR)


if CROSS_FIT_MODE:
    _run_crossfit()


# ── 主流程 ───────────────────────────────────────────────────────────
if not CROSS_FIT_MODE:
    work = _load_or_preprocess()
    work, train_idx, validation_idx, cal_idx, test_idx, fit_idx = _time_split_indices(work)
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
    X_usage_train, y_usage_train = _sample_training_only(
        X_usage.iloc[fit_idx], y_usage.iloc[fit_idx], "usage"
    )
    X_default_train, y_default_train = _sample_training_only(
        X_default.iloc[fit_idx], y_default.iloc[fit_idx], "default"
    )
    booster_usage = _train_lgb(
        X_usage_train, y_usage_train, "y_freq (usage)"
    )
    booster_default = _train_lgb(
        X_default_train, y_default_train, "y_dq_risk (default)"
    )

    os.makedirs("saved_models", exist_ok=True)
    booster_usage.save_model("saved_models/booster_usage.txt")
    booster_default.save_model("saved_models/booster_default.txt")
    print("  模型已保存到 saved_models/")

    print("\n[4/5] 生成概率网格…")
    grid = _build_grid()
    p_usage, p_default = _predict_probability_grid(
        X_usage, booster_usage, booster_default, grid
    )

    print("\n[5/5] 保存 .npy 文件…")
    os.makedirs(OUT_DIR, exist_ok=True)
    customer_id = work["cst_id"].astype(str).to_numpy()

    np.save(os.path.join(OUT_DIR, "customer_id.npy"), customer_id)
    np.save(os.path.join(OUT_DIR, "grid.npy"),        grid)
    np.save(os.path.join(OUT_DIR, "p_usage.npy"),      p_usage)
    np.save(os.path.join(OUT_DIR, "p_default.npy"),    p_default)
    np.save(os.path.join(OUT_DIR, "train_idx.npy"),    train_idx)
    np.save(os.path.join(OUT_DIR, "validation_idx.npy"), validation_idx)
    np.save(os.path.join(OUT_DIR, "fit_idx.npy"),      fit_idx)
    np.save(os.path.join(OUT_DIR, "cal_idx.npy"),      cal_idx)
    np.save(os.path.join(OUT_DIR, "test_idx.npy"),     test_idx)
    work.to_csv(os.path.join(OUT_DIR, "work_features.csv"), index=False, encoding=CSV_ENCODING)

    split_name = np.empty(len(work), dtype=object)
    split_name[train_idx] = "train"
    split_name[validation_idx] = "validation"
    split_name[cal_idx] = "cal"
    split_name[test_idx] = "test"
    split_manifest = pd.DataFrame({
        "row_index": np.arange(len(work), dtype=int),
        "cst_id": customer_id,
        "split": split_name,
        "split_eff_date": pd.to_datetime(work["split_eff_date"], errors="coerce"),
        "y_freq": y_usage.to_numpy(dtype=int),
        "y_dq_risk": y_default.to_numpy(dtype=int),
    })
    split_manifest.to_csv(
        os.path.join(OUT_DIR, "split_manifest.csv"),
        index=False,
        encoding=CSV_ENCODING,
    )
    with open(os.path.join(OUT_DIR, "feature_preprocessing.json"), "w", encoding="utf-8") as f:
        json.dump(preprocessing_metadata, f, ensure_ascii=False, indent=2)

    # 保存特征名供 scorecard 使用
    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feat_names))

    print(f"\n✓ 已保存到 {OUT_DIR}/")
    print(f"  customer_id.npy  shape={customer_id.shape}")
    print(f"  grid.npy         shape={grid.shape}")
    print(f"  p_usage.npy      shape={p_usage.shape}")
    print(f"  p_default.npy    shape={p_default.shape}")
    print(
        "  split train/validation/cal/test: "
        f"{len(train_idx)}/{len(validation_idx)}/{len(cal_idx)}/{len(test_idx)}"
    )
    print("  feature preprocessing fit scope: train_plus_validation_after_sampling_selection")
    print(f"  y_freq 模式      : {Y_FREQ_MODE}")
    print(f"  HANDLE_IMBALANCE : {HANDLE_IMBALANCE}")


    print("\n[6/6] 在独立校准集上拟合 Isotonic，并保存开发阶段校准概率网格…")
    original_limits = pd.to_numeric(work[QUOTA_COL], errors="coerce").fillna(0.0).to_numpy()
    cal_grid_idx = _nearest_grid_indices(original_limits[cal_idx], grid)
    cal_rows = np.asarray(cal_idx, dtype=int)
    raw_usage_cal = np.asarray(p_usage[cal_rows, cal_grid_idx], dtype=float)
    raw_default_cal = np.asarray(p_default[cal_rows, cal_grid_idx], dtype=float)
    if y_usage.iloc[cal_idx].nunique() < 2 or y_default.iloc[cal_idx].nunique() < 2:
        raise ValueError("开发阶段校准集至少需要同时包含正负样本")
    usage_dev_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        raw_usage_cal, y_usage.iloc[cal_idx]
    )
    default_dev_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
        raw_default_cal, y_default.iloc[cal_idx]
    )
    p_usage_dev_calibrated = _calibrate_matrix(usage_dev_calibrator, p_usage)
    p_default_dev_calibrated = _calibrate_matrix(default_dev_calibrator, p_default)

    os.makedirs(DEV_CALIBRATED_DIR, exist_ok=True)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "customer_id.npy"), customer_id)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "grid.npy"), grid)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "p_usage.npy"), p_usage_dev_calibrated)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "p_default.npy"), p_default_dev_calibrated)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "p_usage_raw.npy"), p_usage)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "p_default_raw.npy"), p_default)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "train_idx.npy"), train_idx)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "validation_idx.npy"), validation_idx)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "fit_idx.npy"), fit_idx)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "cal_idx.npy"), cal_idx)
    np.save(os.path.join(DEV_CALIBRATED_DIR, "test_idx.npy"), test_idx)
    work.to_csv(
        os.path.join(DEV_CALIBRATED_DIR, "work_features.csv"),
        index=False,
        encoding=CSV_ENCODING,
    )
    split_manifest.to_csv(
        os.path.join(DEV_CALIBRATED_DIR, "split_manifest.csv"),
        index=False,
        encoding=CSV_ENCODING,
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
