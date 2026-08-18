#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_credit_limit.py - Credit Limit Optimization Visualization
==============================================================
Fig 1: Boxplot - original vs optimized credit limit by talent tier
Fig 2: Bar chart - avg default probability by limit bucket x tier
Fig 3: Bar chart - avg usage probability by limit bucket x tier
Fig 4: Bar chart - unit profit before vs after optimization by tier
Fig 5: Summary table - tier stats (true rates + predicted probabilities)
Fig 6: Distribution comparison chart - original vs optimized limits
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False

# -- Parameters (can be overridden from notebook via *_OVERRIDE globals) --
RESULTS_CSV = globals().get("RESULTS_CSV_OVERRIDE", "reports/credit_limit_large_grid_results.csv")
TALENT_NPY = globals().get("TALENT_NPY_OVERRIDE", "reports/talent_levels.npy")
PROB_GRID_DIR = globals().get("PROB_GRID_DIR_OVERRIDE", "probability_grid_large")
OUT_DIR = globals().get("VIZ_OUT_DIR_OVERRIDE", "reports/viz")
INTEREST_RATE = float(globals().get("INTEREST_RATE_OVERRIDE", 0.03))
LGD_COEFFICIENT = float(globals().get("LGD_COEFFICIENT_OVERRIDE", 0.45))
LINEAR_COST = float(globals().get("LINEAR_COST_OVERRIDE", 0.005))
QUADRATIC_COST = float(globals().get("QUADRATIC_COST_OVERRIDE", 1e-9))
DPI = int(globals().get("DPI_OVERRIDE", 150))

# 校准概率网格目录: 若 notebook 传入则用校准网格做概率查表，否则与 PROB_GRID_DIR 相同
# notebook Cell 7 里加: PROB_GRID_DIR_CALIBRATED_OVERRIDE = PROB_GRID_DIR + "_calibrated"
_PROB_GRID_DIR_CALIBRATED = globals().get("PROB_GRID_DIR_CALIBRATED_OVERRIDE", PROB_GRID_DIR)

TIER_LABEL = {8: "A", 7: "B", 6: "C", 5: "D", 4: "E", 3: "F1", 2: "F2", 1: "F3"}
ALL_TIER_ORDER = ["F3", "F2", "F1", "E", "D", "C", "B", "A"]
TIER_ORDER = list(ALL_TIER_ORDER)
TIER_COLORS = {
    "F3": "#b2182b",
    "F2": "#d6604d",
    "F1": "#f4a582",
    "E": "#4dac26",
    "D": "#2166ac",
    "C": "#1a4d80",
    "B": "#123564",
    "A": "#0a1d42",
}

RUN_ID = globals().get("DEBUG_RUN_ID_OVERRIDE", "run1")
LOG_PATH = globals().get("DEBUG_LOG_PATH_OVERRIDE", "debug-d90a1c.log")
SESSION_ID = "d90a1c"


def _dbg(hypothesis_id, location, message, data=None):
    payload = {
        "sessionId": SESSION_ID,
        "runId": RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _savefig(fig, filename):
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=DPI, bbox_inches="tight")


# Limit buckets for figures 2 & 3 (unit: yuan). Keep business-friendly
# buckets near the dense area, then extend dynamically to cover high limits.
_base_bin_edges = [0, 200_000, 400_000, 700_000, 1_000_000]

# -- Load data --
print("Loading results: %s" % RESULTS_CSV)
df_results = pd.read_csv(RESULTS_CSV)

# Read tier labels from df_results when available to stay row-aligned
if "talent_level" in df_results.columns:
    talent_levels = df_results["talent_level"].values
    print("  [tier] using df_results['talent_level'] column")
else:
    print("Loading talent levels: %s" % TALENT_NPY)
    talent_levels = np.load(TALENT_NPY, allow_pickle=True)
    print("  [tier] WARNING: falling back to talent_levels.npy (row order may mismatch)")

tier_labels = np.array([TIER_LABEL.get(int(t), str(t)) for t in talent_levels])
_tier_counts = pd.Series(tier_labels).value_counts(dropna=False)
TIER_ORDER = [t for t in ALL_TIER_ORDER if int(_tier_counts.get(t, 0)) > 0]
if not TIER_ORDER:
    TIER_ORDER = list(ALL_TIER_ORDER)
print("  [tier] visible tiers in charts: %s" % ", ".join(TIER_ORDER))

orig_L = df_results["original_credit_limit"].values.astype(float)
opt_L = df_results["credit_limit"].values.astype(float)
_usage_cal_col = "optimized_usage_prob_calibrated" if "optimized_usage_prob_calibrated" in df_results.columns else "predicted_usage_prob"
_default_cal_col = "optimized_default_prob_calibrated" if "optimized_default_prob_calibrated" in df_results.columns else "predicted_default_prob"
p_u_o = df_results[_usage_cal_col].values.astype(float)
p_d_o = df_results[_default_cal_col].values.astype(float)

_observed_limit_max = float(np.nanmax([np.nanmax(orig_L), np.nanmax(opt_L)]))
_bin_upper = max(1_500_000.0, np.ceil(_observed_limit_max / 500_000.0) * 500_000.0)
BIN_EDGES = list(_base_bin_edges)
while BIN_EDGES[-1] < _bin_upper:
    BIN_EDGES.append(BIN_EDGES[-1] + 500_000)
BIN_EDGES[-1] += 1.0

def _fmt_w(v):
    return "%dw" % int(round(v / 10000.0))

BIN_LABELS = []
for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
    hi_show = hi - 1.0 if hi == BIN_EDGES[-1] else hi
    if lo >= 1_000_000:
        BIN_LABELS.append("%s-%s" % (_fmt_w(lo), _fmt_w(hi_show)))
    else:
        BIN_LABELS.append("%s-%s" % (_fmt_w(lo), _fmt_w(hi_show)))

_dbg("A", "viz_credit_limit.py:load", "loaded core result arrays", {
    "rows": int(len(df_results)),
    "orig_min": float(np.nanmin(orig_L)),
    "orig_max": float(np.nanmax(orig_L)),
    "opt_min": float(np.nanmin(opt_L)),
    "opt_max": float(np.nanmax(opt_L)),
    "orig_ge_110w": int(np.sum(orig_L >= 1_100_000)),
    "opt_ge_110w": int(np.sum(opt_L >= 1_100_000)),
    "tier_counts": {str(k): int(v) for k, v in pd.Series(tier_labels).value_counts(dropna=False).to_dict().items()},
})

# Load probability grid for base-limit lookup (unit profit comparison in Fig 4)
# 优先使用校准概率网格，保持与 Fig 5 预测概率一致
_grid_dir_used = _PROB_GRID_DIR_CALIBRATED
if not os.path.exists(os.path.join(_grid_dir_used, "grid.npy")):
    raise FileNotFoundError("未找到校准后概率网格目录: %s" % _grid_dir_used)
print("  概率网格目录: %s" % _grid_dir_used)
_cal_method_path = os.path.join(_grid_dir_used, "calibration_method.txt")
_cal_method = "unknown"
if os.path.exists(_cal_method_path):
    try:
        with open(_cal_method_path, "r", encoding="utf-8") as _f:
            _cal_method = _f.read().strip() or "unknown"
    except Exception:
        _cal_method = "unknown"
print("  概率校准方法: %s" % _cal_method)
if "isotonic" not in _cal_method.lower():
    raise ValueError("可视化要求使用 Isotonic 校准后概率，当前方法为: %s" % _cal_method)
print("  Fig 2/3 概率口径: 五折样本外 + Isotonic 校准后（优化额度点）")
_grid = np.load(os.path.join(_grid_dir_used, "grid.npy"))
_p_usage = np.load(os.path.join(_grid_dir_used, "p_usage.npy"), mmap_mode="r")
_p_default = np.load(os.path.join(_grid_dir_used, "p_default.npy"), mmap_mode="r")
_p_usage_raw = np.load(os.path.join(_grid_dir_used, "p_usage_raw.npy"), mmap_mode="r")
_p_default_raw = np.load(os.path.join(_grid_dir_used, "p_default_raw.npy"), mmap_mode="r")

_dbg("B", "viz_credit_limit.py:grid", "loaded probability grid", {
    "grid_min": float(np.nanmin(_grid)),
    "grid_max": float(np.nanmax(_grid)),
    "grid_len": int(len(_grid)),
    "usage_shape": list(np.shape(_p_usage)),
    "default_shape": list(np.shape(_p_default)),
})


def _lookup(L_array):
    idx = np.searchsorted(_grid, L_array)
    idx = np.clip(idx, 0, len(_grid) - 1)
    left = np.clip(idx - 1, 0, len(_grid) - 1)
    use_left = np.abs(_grid[left] - L_array) <= np.abs(_grid[idx] - L_array)
    idx = np.where(use_left, left, idx).astype(int)
    rows = np.arange(len(idx))
    return (
        np.asarray(_p_usage[rows, idx], dtype=float),
        np.asarray(_p_default[rows, idx], dtype=float),
    )


def _lookup_raw(L_array):
    idx = np.searchsorted(_grid, L_array)
    idx = np.clip(idx, 0, len(_grid) - 1)
    left = np.clip(idx - 1, 0, len(_grid) - 1)
    use_left = np.abs(_grid[left] - L_array) <= np.abs(_grid[idx] - L_array)
    idx = np.where(use_left, left, idx).astype(int)
    rows = np.arange(len(idx))
    return (
        np.asarray(_p_usage_raw[rows, idx], dtype=float),
        np.asarray(_p_default_raw[rows, idx], dtype=float),
    )


p_u_base, p_d_base = _lookup(orig_L)
# Fig 2/3/5 必须直接从原始概率网格取频繁支用概率，不能使用优化结果表中的
# utilization_used（历史支用率）。下列四个数组均对应优化后额度点。
p_u_o, p_d_o = _lookup(opt_L)
p_u_raw_o, p_d_raw_o = _lookup_raw(opt_L)

# -- Load true labels (y_freq / y_dq_risk) from work_features.csv --
_work_path = os.path.join(_grid_dir_used, "work_features.csv")
y_freq_true = None
y_dq_true = None
if os.path.exists(_work_path):
    print("Loading work_features: %s" % _work_path)
    _usecols = lambda c: c in {"cst_id", "y_freq", "y_dq_risk", "credamt", "授信额度", "档位"}
    _df_work = pd.read_csv(_work_path, usecols=_usecols)
    _df_work["cst_id"] = _df_work["cst_id"].astype(str)

    _id_col = next((c for c in ["customer_id", "cst_id"] if c in df_results.columns), None)
    if _id_col is not None:
        _df_r = df_results[[_id_col]].copy()
        _df_r[_id_col] = _df_r[_id_col].astype(str)
        _merged = _df_r.merge(_df_work, left_on=_id_col, right_on="cst_id", how="left")
        if "y_freq" in _merged.columns:
            y_freq_true = _merged["y_freq"].values.astype(float)
        if "y_dq_risk" in _merged.columns:
            y_dq_true = _merged["y_dq_risk"].values.astype(float)
        print("  True labels aligned: y_freq pos_rate=%.2f%%, y_dq_risk pos_rate=%.2f%%" % (
            np.nanmean(y_freq_true) * 100, np.nanmean(y_dq_true) * 100))

        _work_cred_col = "credamt" if "credamt" in _df_work.columns else ("授信额度" if "授信额度" in _df_work.columns else None)
        if _work_cred_col is not None:
            _merged_limit = _df_r.merge(
                _df_work[["cst_id", _work_cred_col]],
                left_on=_id_col,
                right_on="cst_id",
                how="left",
            )
            _work_limits_aligned = pd.to_numeric(_merged_limit[_work_cred_col], errors="coerce").to_numpy(dtype=float)
            _limit_check = pd.DataFrame([
                {
                    "Source": "optimization_results.original_credit_limit",
                    "N": int(np.isfinite(orig_L).sum()),
                    "Min(w)": round(float(np.nanmin(orig_L)) / 10000, 2),
                    "P50(w)": round(float(np.nanmedian(orig_L)) / 10000, 2),
                    "Max(w)": round(float(np.nanmax(orig_L)) / 10000, 2),
                    ">=110w": int(np.sum(orig_L >= 1_100_000)),
                    ">=550w": int(np.sum(orig_L >= 5_500_000)),
                },
                {
                    "Source": "probability_grid.work_features.%s" % _work_cred_col,
                    "N": int(np.isfinite(_work_limits_aligned).sum()),
                    "Min(w)": round(float(np.nanmin(_work_limits_aligned)) / 10000, 2),
                    "P50(w)": round(float(np.nanmedian(_work_limits_aligned)) / 10000, 2),
                    "Max(w)": round(float(np.nanmax(_work_limits_aligned)) / 10000, 2),
                    ">=110w": int(np.sum(_work_limits_aligned >= 1_100_000)),
                    ">=550w": int(np.sum(_work_limits_aligned >= 5_500_000)),
                },
            ])
            os.makedirs(OUT_DIR, exist_ok=True)
            _limit_check.to_csv(os.path.join(OUT_DIR, "limit_source_consistency.csv"), index=False, encoding="utf-8-sig")
            print("  Limit source consistency check:")
            print(_limit_check.to_string(index=False))
    else:
        print("  [WARNING] customer_id / cst_id column not found in df_results")
else:
    print("  [WARNING] %s not found, skipping true label loading" % _work_path)

os.makedirs(OUT_DIR, exist_ok=True)
# Keep the notebook preview focused on the current run only.
for _name in os.listdir(OUT_DIR):
    if _name.lower().endswith((".png", ".csv")):
        try:
            os.remove(os.path.join(OUT_DIR, _name))
        except Exception:
            pass

print("Output directory: %s" % OUT_DIR)

_tier_diag_rows = []
for _tier in ALL_TIER_ORDER:
    _mask = tier_labels == _tier
    _tier_diag_rows.append({
        "Tier": _tier,
        "Count": int(_mask.sum()),
        "Orig Min(w)": round(float(np.nanmin(orig_L[_mask])) / 10000, 2) if _mask.any() else np.nan,
        "Orig Mean(w)": round(float(np.nanmean(orig_L[_mask])) / 10000, 2) if _mask.any() else np.nan,
        "Orig Max(w)": round(float(np.nanmax(orig_L[_mask])) / 10000, 2) if _mask.any() else np.nan,
        "Opt Min(w)": round(float(np.nanmin(opt_L[_mask])) / 10000, 2) if _mask.any() else np.nan,
        "Opt Mean(w)": round(float(np.nanmean(opt_L[_mask])) / 10000, 2) if _mask.any() else np.nan,
        "Opt Max(w)": round(float(np.nanmax(opt_L[_mask])) / 10000, 2) if _mask.any() else np.nan,
        "Exists": bool(_mask.any()),
    })
df_tier_diag = pd.DataFrame(_tier_diag_rows)
df_tier_diag.to_csv(os.path.join(OUT_DIR, "tier_limit_diagnostics.csv"), index=False, encoding="utf-8-sig")
print("\nTier existence and limit diagnostics:")
print(df_tier_diag.to_string(index=False))
print("A/B/C existence:")
print(df_tier_diag[df_tier_diag["Tier"].isin(["A", "B", "C"])][["Tier", "Count", "Exists"]].to_string(index=False))


# ============================================================
# Fig 1: Boxplot - original vs optimized credit limit by tier
# ============================================================
print("\n[Fig 1] Credit limit distribution boxplot (before vs after)...")
fig, ax = plt.subplots(figsize=(14, 6), dpi=DPI)
ax.set_title("Credit Limit Distribution by Talent Tier: Before vs After Optimization", fontsize=14, pad=12)

x = np.arange(len(TIER_ORDER))
width = 0.28
rng = np.random.RandomState(42)  # 兼容旧版 numpy（<1.17 没有 default_rng）


def _visible_box_stats(arr, min_vis_height=8000.0):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    whislo, whishi = np.percentile(arr, [5, 95])
    vis_q1, vis_q3 = q1, q3
    if vis_q3 - vis_q1 < min_vis_height:
        half = max(min_vis_height / 2.0, abs(med) * 0.01)
        vis_q1 = max(0.0, med - half)
        vis_q3 = med + half
        whislo = min(whislo, vis_q1)
        whishi = max(whishi, vis_q3)
    return {
        "med": float(med),
        "q1": float(vis_q1),
        "q3": float(vis_q3),
        "whislo": float(whislo),
        "whishi": float(whishi),
        "fliers": [],
    }


for i, tier in enumerate(TIER_ORDER):
    mask = tier_labels == tier
    color = TIER_COLORS[tier]
    _dbg("C", "viz_credit_limit.py:fig1", "tier sample size and ranges", {
        "tier": tier,
        "count": int(mask.sum()),
        "orig_min": float(np.nanmin(orig_L[mask])) if mask.any() else None,
        "orig_max": float(np.nanmax(orig_L[mask])) if mask.any() else None,
        "opt_min": float(np.nanmin(opt_L[mask])) if mask.any() else None,
        "opt_max": float(np.nanmax(opt_L[mask])) if mask.any() else None,
        "orig_iqr": float(np.percentile(orig_L[mask], 75) - np.percentile(orig_L[mask], 25)) if mask.any() else None,
        "opt_iqr": float(np.percentile(opt_L[mask], 75) - np.percentile(opt_L[mask], 25)) if mask.any() else None,
    })

    for grp, offset, hatch, alpha_val, lbl, zorder in [
        (orig_L[mask], -width / 2 - 0.02, "", 0.58, "Before", 3),
        (opt_L[mask], width / 2 + 0.02, "//", 0.42, "After", 4),
    ]:
        if len(grp) == 0:
            continue

        pos = x[i] + offset
        sample = grp
        if len(grp) > 800:
            sample = rng.choice(grp, size=800, replace=False)
        jitter = rng.normal(0, width * 0.06, size=len(sample)) if len(sample) > 1 else np.array([0.0])
        ax.scatter(np.full(len(sample), pos) + jitter, sample, s=6, color=color, alpha=0.06, linewidths=0, zorder=1)

        bp = ax.bxp(
            [_visible_box_stats(grp)],
            positions=[pos],
            widths=width * 0.72,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", linewidth=2.2),
            whiskerprops=dict(linewidth=1.4, color="#444444"),
            capprops=dict(linewidth=1.4, color="#444444"),
            boxprops=dict(linewidth=1.5, color="#333333"),
        )
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(alpha_val)
        bp["boxes"][0].set_edgecolor("#333333")
        bp["boxes"][0].set_zorder(zorder)

        mv = float(np.mean(grp))
        ax.scatter(pos, mv, marker="^", color="#111111", s=42, zorder=6)
        ax.text(pos, mv + 15000, "avg\n%.0fw" % (mv / 10000),
                ha="center", va="bottom", fontsize=7.5, color="#111111", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(TIER_ORDER, fontsize=11)
ax.set_xlim(-0.65, len(TIER_ORDER) - 0.35)
ax.set_xlabel("Talent Tier", fontsize=11)
ax.set_ylabel("Credit Limit (yuan)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: "%.0fw" % (v / 10000)))
ax.grid(axis="y", linestyle="--", alpha=0.5)
ax.legend(handles=[
    mpatches.Patch(facecolor="gray", alpha=0.58, label="Before"),
    mpatches.Patch(facecolor="gray", alpha=0.42, hatch="//", label="After"),
], fontsize=10)
plt.tight_layout()
_savefig(fig, "limit_boxplot.png")
plt.close(fig)
print("  Saved: limit_boxplot.png")


# ============================================================
# Fig 2: Avg default probability by limit bucket x tier
# ============================================================
print("[Fig 2] Avg default probability by limit bucket x tier...")

df_p = pd.DataFrame({
    "tier": tier_labels,
    "p_default": p_d_o,
    "opt_L": opt_L,
})
df_p["bin"] = pd.cut(
    df_p["opt_L"],
    bins=BIN_EDGES,
    labels=BIN_LABELS,
    right=False,
    include_lowest=True,
)

_dbg("D", "viz_credit_limit.py:fig2", "default probability bin coverage", {
    "bin_counts": {str(k): int(v) for k, v in df_p["bin"].value_counts(dropna=False).to_dict().items()},
    "tier_bin_table": {
        tier: {str(bl): int(len(df_p[(df_p["bin"] == bl) & (df_p["tier"] == tier)])) for bl in BIN_LABELS}
        for tier in TIER_ORDER
    },
})

n_bins = len(BIN_LABELS)
n_tier = len(TIER_ORDER)
x = np.arange(n_bins)
bar_w = 0.15
offsets = np.linspace(-(n_tier - 1) / 2 * bar_w, (n_tier - 1) / 2 * bar_w, n_tier)

fig, ax = plt.subplots(figsize=(13, 6), dpi=DPI)
ax.set_title("Avg Predicted Default Probability by Limit Bucket x Talent Tier", fontsize=13, pad=10)

_default_rows = []
for j, tier in enumerate(TIER_ORDER):
    means = []
    counts = []
    for bl in BIN_LABELS:
        sub = df_p[(df_p["bin"] == bl) & (df_p["tier"] == tier)]["p_default"]
        counts.append(int(len(sub)))
        means.append(sub.mean() if len(sub) > 0 else 0.0)
        _default_rows.append({
            "Limit Bucket": str(bl),
            "Talent Tier": tier,
            "N": int(len(sub)),
            "Pred Default Mean": float(sub.mean()) if len(sub) > 0 else np.nan,
        })
    bars = ax.bar(
        x + offsets[j],
        means,
        width=bar_w,
        color=TIER_COLORS[tier],
        alpha=0.85,
        label=tier,
        edgecolor="white",
    )
    for bar, val, cnt in zip(bars, means, counts):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0005,
                "%.2f%%\nn=%d" % (val * 100, cnt),
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=TIER_COLORS[tier],
                fontweight="bold",
            )

ax.set_xticks(x)
ax.set_xticklabels(BIN_LABELS, fontsize=11)
ax.set_xlabel("Optimized Limit Bucket", fontsize=11)
ax.set_ylabel("Avg Default Probability (model)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
_leg2 = ax.legend(title="Talent Tier", fontsize=9)
_leg2.get_title().set_fontsize(9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
_savefig(fig, "default_prob_bar.png")
plt.close(fig)
pd.DataFrame(_default_rows).to_csv(
    os.path.join(OUT_DIR, "default_prob_by_limit_bucket_tier.csv"),
    index=False,
    encoding="utf-8-sig",
)
print("  Saved: default_prob_bar.png")


# ============================================================
# Fig 3: Avg usage probability by limit bucket x tier
# ============================================================
print("[Fig 3] Avg usage probability by limit bucket x tier...")

df_u = pd.DataFrame({
    "tier": tier_labels,
    "p_usage": p_u_o,
    "opt_L": opt_L,
})
df_u["bin"] = pd.cut(
    df_u["opt_L"],
    bins=BIN_EDGES,
    labels=BIN_LABELS,
    right=False,
    include_lowest=True,
)

_dbg("E", "viz_credit_limit.py:fig3", "usage probability bin coverage", {
    "bin_counts": {str(k): int(v) for k, v in df_u["bin"].value_counts(dropna=False).to_dict().items()},
    "tier_bin_table": {
        tier: {str(bl): int(len(df_u[(df_u["bin"] == bl) & (df_u["tier"] == tier)])) for bl in BIN_LABELS}
        for tier in TIER_ORDER
    },
})

fig, ax = plt.subplots(figsize=(13, 6), dpi=DPI)
ax.set_title("Avg Predicted Usage Probability by Limit Bucket x Talent Tier", fontsize=13, pad=10)

bar_w = 0.15
offsets = np.linspace(-(n_tier - 1) / 2 * bar_w, (n_tier - 1) / 2 * bar_w, n_tier)

_usage_rows = []
for j, tier in enumerate(TIER_ORDER):
    means = []
    counts = []
    for bl in BIN_LABELS:
        sub = df_u[(df_u["bin"] == bl) & (df_u["tier"] == tier)]["p_usage"]
        counts.append(int(len(sub)))
        means.append(sub.mean() if len(sub) > 0 else 0.0)
        _usage_rows.append({
            "Limit Bucket": str(bl),
            "Talent Tier": tier,
            "N": int(len(sub)),
            "Pred Usage Mean": float(sub.mean()) if len(sub) > 0 else np.nan,
        })
    bars = ax.bar(
        np.arange(n_bins) + offsets[j],
        means,
        width=bar_w,
        color=TIER_COLORS[tier],
        alpha=0.85,
        label=tier,
        edgecolor="white",
    )
    for bar, val, cnt in zip(bars, means, counts):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0005,
                "%.2f%%\nn=%d" % (val * 100, cnt),
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=TIER_COLORS[tier],
                fontweight="bold",
            )

ax.set_xticks(range(n_bins))
ax.set_xticklabels(BIN_LABELS, fontsize=11)
ax.set_xlabel("Optimized Limit Bucket", fontsize=11)
ax.set_ylabel("Avg Usage Probability (model)", fontsize=11)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
_leg3 = ax.legend(title="Talent Tier", fontsize=9)
_leg3.get_title().set_fontsize(9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
_savefig(fig, "usage_prob_bar.png")
plt.close(fig)
pd.DataFrame(_usage_rows).to_csv(
    os.path.join(OUT_DIR, "usage_prob_by_limit_bucket_tier.csv"),
    index=False,
    encoding="utf-8-sig",
)
print("  Saved: usage_prob_bar.png")


# ============================================================
# Fig 4: Unit risk-adjusted objective before vs after optimization by tier
# ============================================================
print("[Fig 4] Unit risk-adjusted objective before vs after optimization by tier...")

if "utilization_used" not in df_results.columns:
    raise ValueError("优化结果缺少 utilization_used，无法按当前目标函数计算 Fig 4/5")
historical_utilization = pd.to_numeric(
    df_results["utilization_used"], errors="coerce"
).to_numpy(dtype=float)
if not np.isfinite(historical_utilization).all():
    raise ValueError("utilization_used 存在空值或非有限值")

unit_base = (
    INTEREST_RATE * historical_utilization
    - LGD_COEFFICIENT * p_d_base * historical_utilization
    - LINEAR_COST
    - QUADRATIC_COST * orig_L
)
unit_opt = (
    INTEREST_RATE * historical_utilization
    - LGD_COEFFICIENT * p_d_o * historical_utilization
    - LINEAR_COST
    - QUADRATIC_COST * opt_L
)
unit_base = np.where(orig_L > 0, unit_base, 0.0)
unit_opt = np.where(opt_L > 0, unit_opt, 0.0)

df_up = pd.DataFrame({"tier": tier_labels, "base": unit_base, "opt": unit_opt})
grp = df_up.groupby("tier")[["base", "opt"]].mean()
grp = grp.reindex([t for t in TIER_ORDER if t in grp.index])

xp = np.arange(len(grp))
bw2 = 0.35
fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)

color_before = "#aec7e8"
color_after = "#1f77b4"
bb = ax.bar(xp - bw2 / 2, grp["base"], width=bw2, color=color_before, edgecolor="white", label="Before")
bo = ax.bar(xp + bw2 / 2, grp["opt"], width=bw2, color=color_after, edgecolor="white", label="After")

for bar in list(bb) + list(bo):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + (0.0002 if h >= 0 else -0.0004),
        "%.4f" % h,
        ha="center",
        va="bottom" if h >= 0 else "top",
        fontsize=8,
    )

ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xticks(xp)
ax.set_xticklabels(grp.index, fontsize=11)
ax.set_xlabel("Talent Tier", fontsize=11)
ax.set_ylabel("Avg Unit Risk-adjusted Objective", fontsize=11)
ax.set_title("Risk-adjusted Objective by Talent Tier: Before vs After", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
_savefig(fig, "unit_profit.png")
plt.close(fig)
print("  Saved: unit_profit.png")


# ============================================================
# Fig 5: Tier summary table
# ============================================================
print("[Fig 5] Tier summary table (raw/calibrated frequency-usage and default probabilities)...")

_rows = []
for tier in TIER_ORDER:
    mask = tier_labels == tier
    n = int(mask.sum())
    if n == 0:
        continue
    row = {
        "Tier": tier,
        "Count": n,
        "Orig Mean (w)": round(orig_L[mask].mean() / 10000, 2),
        "Orig Median (w)": round(np.median(orig_L[mask]) / 10000, 2),
        "Opt Mean (w)": round(opt_L[mask].mean() / 10000, 2),
        "Opt Median (w)": round(np.median(opt_L[mask]) / 10000, 2),
        "True Freq Rate": ("%.2f%%" % (np.nanmean(y_freq_true[mask]) * 100)) if y_freq_true is not None else "N/A",
        "True Default Rate": ("%.2f%%" % (np.nanmean(y_dq_true[mask]) * 100)) if y_dq_true is not None else "N/A",
        "Avg Utilization": "%.2f%%" % (historical_utilization[mask].mean() * 100),
        "Pred Freq Raw (opt)": "%.2f%%" % (p_u_raw_o[mask].mean() * 100),
        "Pred Default Raw (opt)": "%.2f%%" % (p_d_raw_o[mask].mean() * 100),
        "Pred Freq Cal (opt)": "%.2f%%" % (p_u_o[mask].mean() * 100),
        "Pred Default Cal (opt)": "%.2f%%" % (p_d_o[mask].mean() * 100),
    }
    _avg_u = float(historical_utilization[mask].mean())
    _avg_pd_cal = float(p_d_o[mask].mean())
    _avg_opt_l = float(opt_L[mask].mean())
    _income_term = INTEREST_RATE * _avg_u
    _risk_term = LGD_COEFFICIENT * _avg_u * _avg_pd_cal
    _c1_term = LINEAR_COST
    _c2_term = QUADRATIC_COST * _avg_opt_l
    row.update({
        "Revenue r*AvgU": round(_income_term, 6),
        "Risk lambda*AvgU*AvgPD": round(_risk_term, 6),
        "Cost c1": round(_c1_term, 6),
        "Cost c2*AvgL": round(_c2_term, 6),
        "Net Revenue-Risk-c1-c2": round(
            _income_term - _risk_term - _c1_term - _c2_term, 6
        ),
    })
    _rows.append(row)

df_summary = pd.DataFrame(_rows)
df_summary.to_csv(os.path.join(OUT_DIR, "tier_summary.csv"), index=False, encoding="utf-8-sig")
print(df_summary.to_string(index=False))

_cols5 = list(df_summary.columns)
_data5 = df_summary.values.tolist()
_nrow5 = len(_data5)
_ncol5 = len(_cols5)

fig5, ax5 = plt.subplots(
    figsize=(max(16, _ncol5 * 1.7), max(3, _nrow5 * 0.55 + 1.2)),
    dpi=DPI,
)
ax5.axis("off")
ax5.set_title(
    "Tier Summary: Limits, Raw/Calibrated Model Probabilities, Objective Components",
    fontsize=12,
    pad=10,
    fontweight="bold",
)

tbl5 = ax5.table(
    cellText=_data5,
    colLabels=_cols5,
    cellLoc="center",
    loc="center",
)
tbl5.auto_set_font_size(False)
tbl5.set_fontsize(9)
tbl5.auto_set_column_width(list(range(_ncol5)))

for col_idx in range(_ncol5):
    cell = tbl5[0, col_idx]
    cell.set_facecolor("#2166ac")
    cell.set_text_props(color="white", fontweight="bold")

_tier_bg = {"F3": "#fddbc7", "F2": "#f4a582", "F1": "#fddbc7", "E": "#d9f0d3", "D": "#c6dbef", "C": "#b3d1f0", "B": "#a0c7f0", "A": "#8dbdf0"}
for row_idx, row in enumerate(_rows):
    bg = _tier_bg.get(row["Tier"], "#f8f8f8")
    for col_idx in range(_ncol5):
        tbl5[row_idx + 1, col_idx].set_facecolor(bg)

plt.tight_layout()
_savefig(fig5, "tier_summary.png")
plt.close(fig5)
print("  Saved: tier_summary.png  (CSV: tier_summary.csv)")


# ============================================================
# Fig 6: Distribution comparison chart (original vs optimized)
# ============================================================
print("[Fig 6] Credit limit distribution comparison (original vs optimized)...")

# Use the same 55w bucket width as the notebook text histogram.
# This keeps the 110w+ area visible and comparable to the notebook preview.
DIST_STEP = 550_000  # 55w
max_val = float(np.nanmax([np.nanmax(orig_L), np.nanmax(opt_L), 5_500_000]))
dist_max = int(np.ceil(max_val / DIST_STEP) * DIST_STEP)
dist_edges = np.arange(0, dist_max + DIST_STEP, DIST_STEP)
dist_labels = ["%dw-%dw" % (int(lo / 10000), int(hi / 10000)) for lo, hi in zip(dist_edges[:-1], dist_edges[1:])]

orig_cat = pd.cut(orig_L, bins=dist_edges, labels=dist_labels, right=False, include_lowest=True)
opt_cat = pd.cut(opt_L, bins=dist_edges, labels=dist_labels, right=False, include_lowest=True)
orig_vc = orig_cat.value_counts().reindex(dist_labels, fill_value=0)
opt_vc = opt_cat.value_counts().reindex(dist_labels, fill_value=0)

_dbg("F", "viz_credit_limit.py:fig6", "limit distribution coverage", {
    "orig_bucket_counts": {str(k): int(v) for k, v in orig_vc.to_dict().items()},
    "opt_bucket_counts": {str(k): int(v) for k, v in opt_vc.to_dict().items()},
    "orig_above_110w": int(np.sum(orig_L >= 1_100_000)),
    "opt_above_110w": int(np.sum(opt_L >= 1_100_000)),
    "orig_max": float(np.nanmax(orig_L)),
    "opt_max": float(np.nanmax(opt_L)),
})

_orig_counts = orig_vc.astype(int).values.tolist()
_opt_counts = opt_vc.astype(int).values.tolist()

# Save the raw counts for debugging and downstream use.
df_dist = pd.DataFrame({
    "Limit Bucket": dist_labels,
    "Orig Count": _orig_counts,
    "Opt Count": _opt_counts,
})
_n_orig = len(orig_L)
_n_opt = len(opt_L)
df_dist["Orig %"] = ["%.2f%%" % (c / _n_orig * 100) if _n_orig > 0 else "0.00%" for c in _orig_counts]
df_dist["Opt %"] = ["%.2f%%" % (c / _n_opt * 100) if _n_opt > 0 else "0.00%" for c in _opt_counts]

df_dist.to_csv(os.path.join(OUT_DIR, "limit_distribution.csv"), index=False, encoding="utf-8-sig")
print(df_dist.to_string(index=False))

_thresholds = [1_100_000, 1_500_000, 2_000_000, 5_500_000]
df_high_limit = pd.DataFrame([
    {
        "Threshold": ">=%dw" % int(t / 10000),
        "Original Count": int(np.sum(orig_L >= t)),
        "Original %": "%.2f%%" % (np.sum(orig_L >= t) / max(1, _n_orig) * 100),
        "Optimized Count": int(np.sum(opt_L >= t)),
        "Optimized %": "%.2f%%" % (np.sum(opt_L >= t) / max(1, _n_opt) * 100),
    }
    for t in _thresholds
])
df_high_limit.to_csv(os.path.join(OUT_DIR, "high_limit_threshold_summary.csv"), index=False, encoding="utf-8-sig")
print("\nHigh-limit threshold summary:")
print(df_high_limit.to_string(index=False))

fig6, ax6 = plt.subplots(figsize=(max(12, len(dist_labels) * 0.72), 6.5), dpi=DPI)
bar_w = 0.42
x6 = np.arange(len(dist_labels))

b1 = ax6.bar(x6 - bar_w / 2, _orig_counts, width=bar_w, color="#9ecae1", edgecolor="white", label="Original")
b2 = ax6.bar(x6 + bar_w / 2, _opt_counts, width=bar_w, color="#3182bd", edgecolor="white", label="Optimized")

for bars, counts, color in [(b1, _orig_counts, "#1f4e79"), (b2, _opt_counts, "#08306b")]:
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax6.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(_n_orig, _n_opt) * 0.005,
                f"{cnt:,}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
                fontweight="bold",
            )

ax6.axvline(dist_labels.index("110w-165w") if "110w-165w" in dist_labels else max(0, int(1100000 / DIST_STEP) - 1), color="#666666", linestyle="--", linewidth=0.8, alpha=0.4)
ax6.set_xticks(x6)
ax6.set_xticklabels(dist_labels, rotation=45, ha="right", fontsize=9)
ax6.set_xlabel("Credit Limit Bucket (w)", fontsize=11)
ax6.set_ylabel("Count", fontsize=11)
ax6.set_title("Credit Limit Distribution: Original vs Optimized", fontsize=13, pad=10)
ax6.grid(axis="y", linestyle="--", alpha=0.35)
ax6.legend(fontsize=10)

note = (
    f"Orig max: {np.nanmax(orig_L)/10000:.0f}w\n"
    f"Opt max: {np.nanmax(opt_L)/10000:.0f}w\n"
    f"Orig ≥110w: {int(np.sum(orig_L >= 1_100_000)):,}\n"
    f"Opt ≥110w: {int(np.sum(opt_L >= 1_100_000)):,}"
)
ax6.text(
    0.99,
    0.98,
    note,
    transform=ax6.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.95),
)

plt.tight_layout()
_savefig(fig6, "limit_distribution.png")
plt.close(fig6)
print("  Saved: limit_distribution.png  (CSV: limit_distribution.csv)")


print("\nAll figures saved to %s/" % OUT_DIR)
print("  Note: predicted usage/default probabilities come from %s (calibration_method=%s)." % (_grid_dir_used, _cal_method))
print("  Note: true usage/default rates are actual positive-label fractions in the training set.")
