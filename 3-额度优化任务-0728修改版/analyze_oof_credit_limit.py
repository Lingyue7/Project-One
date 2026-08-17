"""五折 OOF 概率、ALE、候选额度曲线和预算前后额度诊断。"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIER_LABEL = {8: "A", 7: "B", 6: "C", 5: "D", 4: "E", 3: "F1", 2: "F2", 1: "F3"}
TIER_ORDER = ["A", "B", "C", "D", "E", "F1", "F2", "F3"]
TARGETS = {
    "usage": ("Frequent Utilization", "y_freq", "p_usage_raw.npy", "p_usage.npy"),
    "default": ("Default", "y_dq_risk", "p_default_raw.npy", "p_default.npy"),
}


def _nearest_grid_index(values, grid):
    values = np.asarray(values, dtype=float)
    right = np.searchsorted(grid, values, side="left")
    right = np.clip(right, 0, len(grid) - 1)
    left = np.clip(right - 1, 0, len(grid) - 1)
    return np.where(np.abs(values - grid[left]) <= np.abs(values - grid[right]), left, right)


def _quantile_edges(values, n_bins):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("额度字段没有有效数值")
    edges = np.unique(np.percentile(values, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 3:
        raise ValueError("额度唯一值太少，无法分箱")
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _load_inputs(grid_dir):
    required = [
        "grid.npy", "customer_id.npy", "work_features.csv", "split_manifest.csv",
        "p_usage.npy", "p_default.npy", "p_usage_raw.npy", "p_default_raw.npy",
    ]
    missing = [name for name in required if not os.path.isfile(os.path.join(grid_dir, name))]
    if missing:
        raise FileNotFoundError("Cell 5 五折结果不完整，缺少: %s" % missing)

    grid = np.load(os.path.join(grid_dir, "grid.npy"))
    ids = np.load(os.path.join(grid_dir, "customer_id.npy"), allow_pickle=True).astype(str)
    work = pd.read_csv(os.path.join(grid_dir, "work_features.csv"), encoding="utf-8-sig")
    manifest = pd.read_csv(os.path.join(grid_dir, "split_manifest.csv"), encoding="utf-8-sig")
    manifest = manifest.sort_values("row_index").reset_index(drop=True)
    if not np.array_equal(manifest["row_index"].to_numpy(), np.arange(len(work))):
        raise ValueError("split_manifest.csv 的 row_index 与概率矩阵行顺序不一致")
    if len(ids) != len(work) or not np.array_equal(ids, work["cst_id"].astype(str).to_numpy()):
        raise ValueError("customer_id.npy 与 work_features.csv 客户顺序不一致")
    if "credamt" not in work.columns:
        raise ValueError("work_features.csv 缺少 credamt")
    arrays = {}
    for key, (_, _, raw_name, cal_name) in TARGETS.items():
        arrays[key] = {
            "raw": np.load(os.path.join(grid_dir, raw_name), mmap_mode="r"),
            "calibrated": np.load(os.path.join(grid_dir, cal_name), mmap_mode="r"),
        }
        for matrix in arrays[key].values():
            if matrix.shape != (len(work), len(grid)):
                raise ValueError("%s 概率矩阵维度异常: %s" % (key, matrix.shape))
    return grid, ids, work, manifest, arrays


def _save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_oof_bin_plots(grid, work, arrays, out_dir, n_bins=10):
    limits = pd.to_numeric(work["credamt"], errors="coerce").to_numpy(float)
    observed_idx = _nearest_grid_index(limits, grid)
    edges = _quantile_edges(limits, n_bins)
    bin_id = np.clip(np.digitize(limits, edges[1:-1]), 0, len(edges) - 2)
    all_tables = []

    for key, (label, y_col, _, _) in TARGETS.items():
        y = pd.to_numeric(work[y_col], errors="coerce").to_numpy(float)
        raw = np.asarray(arrays[key]["raw"][np.arange(len(work)), observed_idx])
        cal = np.asarray(arrays[key]["calibrated"][np.arange(len(work)), observed_idx])
        rows = []
        for b in range(bin_id.max() + 1):
            mask = bin_id == b
            if not mask.any():
                continue
            rows.append({
                "target": key, "bin": b + 1, "sample_count": int(mask.sum()),
                "credit_min": float(np.nanmin(limits[mask])),
                "credit_max": float(np.nanmax(limits[mask])),
                "credit_mean": float(np.nanmean(limits[mask])),
                "raw_probability": float(np.nanmean(raw[mask])),
                "calibrated_probability": float(np.nanmean(cal[mask])),
                "actual_positive_rate": float(np.nanmean(y[mask])),
            })
        table = pd.DataFrame(rows)
        all_tables.append(table)

        x = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(11, 5.6))
        ax2 = ax.twinx()
        ax2.bar(x, table["sample_count"], color="#D9E6F2", alpha=0.72, label="Sample Count")
        ax.plot(x, table["raw_probability"], "o-", label="Raw Predicted Probability")
        ax.plot(x, table["calibrated_probability"], "s-", label="Calibrated Probability")
        ax.plot(x, table["actual_positive_rate"], "^-", label="Observed Positive Rate")
        # 兼容服务器上的旧版 Matplotlib：旧版 set_xticks 不接受 labels 参数。
        ax.set_xticks(x)
        ax.set_xticklabels(["Q%d" % v for v in table["bin"]])
        ax.set_xlabel("Original Credit Limit Quantile (Low to High)")
        ax.set_ylabel("Probability / Observed Positive Rate")
        ax2.set_ylabel("Sample Count")
        ax.set_title("Five-Fold OOF: %s Probability by Original Limit" % label)
        lines, labels = ax.get_legend_handles_labels()
        bars, bar_labels = ax2.get_legend_handles_labels()
        ax.legend(lines + bars, labels + bar_labels, loc="best")
        ax.grid(alpha=0.2)
        _save_figure(fig, os.path.join(out_dir, "part1_oof_bins_%s.png" % key))

    result = pd.concat(all_tables, ignore_index=True)
    result.to_csv(os.path.join(out_dir, "part1_oof_bin_summary.csv"), index=False, encoding="utf-8-sig")
    return result


def _ale_one_curve(matrix, rows, observed_limits, grid, edges):
    rows = np.asarray(rows, dtype=int)
    local = np.zeros(len(edges) - 1, dtype=float)
    counts = np.zeros(len(edges) - 1, dtype=int)
    finite_edges = np.clip(edges, grid[0], grid[-1])
    low_idx = _nearest_grid_index(finite_edges[:-1], grid)
    high_idx = _nearest_grid_index(finite_edges[1:], grid)
    bin_id = np.clip(np.digitize(observed_limits[rows], edges[1:-1]), 0, len(edges) - 2)
    for b in range(len(local)):
        use = rows[bin_id == b]
        counts[b] = len(use)
        if len(use):
            local[b] = float(np.mean(matrix[use, high_idx[b]] - matrix[use, low_idx[b]]))
    accumulated = np.cumsum(local)
    if counts.sum():
        accumulated -= np.average(accumulated, weights=counts)
    centers = (finite_edges[:-1] + finite_edges[1:]) / 2.0
    return centers, accumulated, local, counts


def make_ale_plots(grid, work, manifest, arrays, out_dir, n_bins=20):
    limits = pd.to_numeric(work["credamt"], errors="coerce").to_numpy(float)
    edges = _quantile_edges(limits, n_bins)
    folds = sorted(pd.unique(manifest["fold"]))
    detail_rows = []

    for key, (label, _, _, _) in TARGETS.items():
        curves = {kind: [] for kind in ["raw", "calibrated"]}
        centers_ref = None
        for fold in folds:
            rows = np.flatnonzero(manifest["fold"].to_numpy() == fold)
            for kind in curves:
                centers, ale, local, counts = _ale_one_curve(
                    arrays[key][kind], rows, limits, grid, edges
                )
                centers_ref = centers
                curves[kind].append(ale)
                for i in range(len(ale)):
                    detail_rows.append({
                        "target": key, "probability": kind, "fold": int(fold),
                        "bin": i + 1, "credit_center": centers[i],
                        "local_effect": local[i], "ale": ale[i], "sample_count": counts[i],
                    })

        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        ax.plot(centers_ref, curves["raw"][0], "o-", label="Fold 1: Raw-Model ALE")
        ax.plot(centers_ref, curves["calibrated"][0], "s-", label="Fold 1: Calibrated ALE")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title="Fold 1 Quick ALE: %s" % label, xlabel="Credit Limit (credamt)", ylabel="ALE")
        ax.legend(); ax.grid(alpha=0.2)
        _save_figure(fig, os.path.join(out_dir, "part2_first_fold_ale_%s.png" % key))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
        for ax, kind, title in zip(axes, ["raw", "calibrated"], ["Raw Model", "Model + Calibrator"]):
            stack = np.vstack(curves[kind])
            for fold, curve in zip(folds, stack):
                ax.plot(centers_ref, curve, alpha=0.28, linewidth=1, label="Fold %s" % fold)
            ax.plot(centers_ref, stack.mean(axis=0), color="#C62828", linewidth=2.6, label="Five-Fold Mean")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set(title=title, xlabel="Credit Limit (credamt)")
            ax.grid(alpha=0.2); ax.legend(fontsize=8)
        axes[0].set_ylabel("ALE")
        fig.suptitle("Five-Fold ALE Review: %s" % label)
        _save_figure(fig, os.path.join(out_dir, "part2_five_fold_ale_%s.png" % key))

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(os.path.join(out_dir, "part2_ale_detail.csv"), index=False, encoding="utf-8-sig")
    mean_table = detail.groupby(["target", "probability", "bin"], as_index=False).agg(
        credit_center=("credit_center", "mean"), ale_five_fold_mean=("ale", "mean")
    )
    mean_table.to_csv(os.path.join(out_dir, "part2_ale_five_fold_mean.csv"), index=False, encoding="utf-8-sig")
    return detail


def _curve_stats(matrix, tolerance, chunk_size=512):
    # 候选额度网格可能很宽，逐块统计，避免 np.diff 整张矩阵造成数 GB 内存峰值。
    non_decreasing_count = 0
    up = down = flat = total = 0
    start_end_parts = []
    for start in range(0, matrix.shape[0], chunk_size):
        end = min(start + chunk_size, matrix.shape[0])
        block = np.asarray(matrix[start:end], dtype=np.float32)
        diff = np.diff(block, axis=1)
        non_decreasing_count += int(np.sum(np.all(diff >= -tolerance, axis=1)))
        up += int(np.sum(diff > tolerance))
        down += int(np.sum(diff < -tolerance))
        flat += int(np.sum(np.abs(diff) <= tolerance))
        total += diff.size
        start_end_parts.append(block[:, -1] - block[:, 0])
    start_end = np.concatenate(start_end_parts)
    return {
        "customer_count": matrix.shape[0],
        "monotonic_non_decreasing_customer_ratio": float(non_decreasing_count / matrix.shape[0]),
        "step_up_ratio": float(up / total),
        "step_down_ratio": float(down / total),
        "step_flat_ratio": float(flat / total),
        "start_end_change_mean": float(np.mean(start_end)),
        "start_end_change_median": float(np.median(start_end)),
        "start_end_change_p10": float(np.percentile(start_end, 10.0)),
        "start_end_change_p90": float(np.percentile(start_end, 90.0)),
    }


def make_curve_diagnostics(grid, ids, work, arrays, out_dir, tolerance=1e-7):
    summary = []
    rep_tables = []
    for key, (label, _, _, _) in TARGETS.items():
        for kind in ["raw", "calibrated"]:
            summary.append({"target": key, "probability": kind, **_curve_stats(arrays[key][kind], tolerance)})

        delta = np.asarray(arrays[key]["calibrated"][:, -1] - arrays[key]["calibrated"][:, 0])
        order = np.argsort(delta)
        positions = np.unique(np.linspace(0, len(order) - 1, 5).round().astype(int))
        chosen = order[positions]
        fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 2.35 * len(chosen)), sharex=True)
        axes = np.atleast_1d(axes)
        for rank, (ax, row) in enumerate(zip(axes, chosen), 1):
            ax.plot(grid, arrays[key]["raw"][row], label="Raw", linewidth=1.2)
            ax.plot(grid, arrays[key]["calibrated"][row], label="Calibrated", linewidth=1.6)
            ax.set_ylabel("Probability")
            ax.set_title("Customer %s | Original Limit %.0f | End-to-End Change %.5f" % (
                ids[row], float(work.iloc[row]["credamt"]), delta[row]
            ), fontsize=9)
            ax.grid(alpha=0.2); ax.legend(fontsize=8)
            rep_tables.append(pd.DataFrame({
                "target": key, "representative_rank": rank, "row_index": int(row),
                "customer_id": ids[row], "original_credit_limit": float(work.iloc[row]["credamt"]),
                "candidate_credit_limit": grid,
                "raw_probability": np.asarray(arrays[key]["raw"][row]),
                "calibrated_probability": np.asarray(arrays[key]["calibrated"][row]),
            }))
        axes[-1].set_xlabel("Candidate Credit Limit")
        fig.suptitle("OOF Candidate-Limit Probability Paths: %s" % label, y=1.005)
        _save_figure(fig, os.path.join(out_dir, "part3_representative_curves_%s.png" % key))

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, "part3_curve_shape_summary.csv"), index=False, encoding="utf-8-sig")
    pd.concat(rep_tables, ignore_index=True).to_csv(
        os.path.join(out_dir, "part3_representative_curve_data.csv"), index=False, encoding="utf-8-sig"
    )
    return summary_df


def make_budget_comparison(calculator, talent_levels, df_results, out_dir):
    """比较逐客独立最优与联合组合约束下的正式解。"""
    levels = np.asarray(talent_levels).reshape(-1).astype(int)
    independent = np.asarray(calculator._best_limits_for_lambda_chunked(levels, 0.0), dtype=float)
    budgeted = np.asarray(calculator.credit_limits, dtype=float)
    if len(independent) != len(budgeted):
        raise ValueError("逐客独立解和联合组合解长度不一致")
    ids = calculator.data_info["df_aggregated"]["cst_id"].astype(str).to_numpy()
    budget = float(calculator.config.total_budget)

    detail = pd.DataFrame({
        "customer_id": ids, "talent_level": levels,
        "tier_label": [TIER_LABEL.get(v, str(v)) for v in levels],
        "limit_independent": independent, "limit_portfolio_constrained": budgeted,
    })
    detail["limit_change_due_to_portfolio_constraints"] = budgeted - independent
    detail["portfolio_constraint_effect"] = np.select(
        [detail["limit_change_due_to_portfolio_constraints"] > 1e-9, detail["limit_change_due_to_portfolio_constraints"] < -1e-9],
        ["Increase", "Decrease"], default="Unchanged"
    )
    detail.to_csv(os.path.join(out_dir, "portfolio_constraint_customer_detail.csv"), index=False, encoding="utf-8-sig")

    rows = []
    for name, limits in [("Independent Customer Optimum", independent), ("Joint Portfolio Constraints", budgeted)]:
        rows.append({
            "scenario": name, "customer_count": len(limits), "total_limit": float(limits.sum()),
            "mean_limit": float(limits.mean()), "median_limit": float(np.median(limits)),
            "budget": budget, "budget_usage_ratio": float(limits.sum() / budget),
            "risk_adjusted_objective": float(calculator._calculate_profit_vectorized(limits, levels)),
            "lgd_adjusted_expected_loss": float(calculator._calculate_expected_loss(limits, levels)),
            "increase_count_vs_no_budget": int(np.sum(limits > independent + 1e-9)),
            "decrease_count_vs_no_budget": int(np.sum(limits < independent - 1e-9)),
            "unchanged_count_vs_no_budget": int(np.sum(np.isclose(limits, independent))),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "portfolio_constraint_summary.csv"), index=False, encoding="utf-8-sig")

    tier = detail.groupby(["tier_label"], as_index=False).agg(
        customer_count=("customer_id", "size"),
        total_independent=("limit_independent", "sum"),
        total_portfolio_constrained=("limit_portfolio_constrained", "sum"),
        mean_independent=("limit_independent", "mean"),
        mean_portfolio_constrained=("limit_portfolio_constrained", "mean"),
    )
    tier["total_change"] = tier["total_portfolio_constrained"] - tier["total_independent"]
    tier["_order"] = tier["tier_label"].map({v: i for i, v in enumerate(TIER_ORDER)})
    tier = tier.sort_values("_order").drop(columns="_order")
    tier.to_csv(os.path.join(out_dir, "portfolio_constraint_by_tier.csv"), index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    axes[0].scatter(independent, budgeted, s=9, alpha=0.3)
    line_max = float(max(independent.max(), budgeted.max()))
    axes[0].plot([0, line_max], [0, line_max], "--", color="black", linewidth=1)
    axes[0].set(xlabel="Independent Customer Optimum", ylabel="Joint Portfolio Solution", title="Customer-Level Constraint Comparison")
    axes[0].grid(alpha=0.2)
    x = np.arange(len(tier)); width = 0.38
    axes[1].bar(x - width / 2, tier["mean_independent"], width, label="Independent")
    axes[1].bar(x + width / 2, tier["mean_portfolio_constrained"], width, label="Portfolio Constraints")
    # 兼容旧版 Matplotlib，避免将 Series 误解释为 minor 参数。
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tier["tier_label"].astype(str).tolist())
    axes[1].set(xlabel="Talent Tier", ylabel="Mean Final Credit Limit", title="Portfolio Constraint Comparison by Tier")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.2)
    _save_figure(fig, os.path.join(out_dir, "portfolio_constraint_comparison.png"))
    return summary, detail


def main(namespace=None):
    ns = globals() if namespace is None else namespace
    grid_dir = ns.get("OOF_ANALYSIS_GRID_DIR_OVERRIDE", ns.get("PROB_GRID_DIR", "probability_grid_large") + "_crossfit_calibrated")
    out_dir = ns.get("OOF_ANALYSIS_OUT_DIR_OVERRIDE", os.path.join(ns.get("REPORTS_DIR", "reports"), "oof_diagnostics"))
    n_bins = int(ns.get("OOF_BIN_COUNT_OVERRIDE", 10))
    ale_bins = int(ns.get("ALE_BIN_COUNT_OVERRIDE", 20))
    tolerance = float(ns.get("CURVE_TOLERANCE_OVERRIDE", 1e-7))
    os.makedirs(out_dir, exist_ok=True)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("读取五折 OOF 保存结果:", os.path.abspath(grid_dir))
    grid, ids, work, manifest, arrays = _load_inputs(grid_dir)
    print("[1/4] 全量五折 OOF 分箱图")
    part1 = make_oof_bin_plots(grid, work, arrays, out_dir, n_bins=n_bins)
    print("[2/4] 第一折快速 ALE + 五折 ALE")
    part2 = make_ale_plots(grid, work, manifest, arrays, out_dir, n_bins=ale_bins)
    print("[3/4] OOF 候选额度概率曲线诊断")
    part3 = make_curve_diagnostics(grid, ids, work, arrays, out_dir, tolerance=tolerance)

    if "calculator" not in ns or "talent_levels" not in ns or "df_results" not in ns:
        raise RuntimeError("第四部分需要 Cell 6 恢复的 calculator、talent_levels、df_results；请先运行 Cell 6")
    print("[4/4] 同一收益矩阵的无预算 / 有预算最终额度对照")
    part4, detail = make_budget_comparison(
        ns["calculator"], ns["talent_levels"], ns["df_results"], out_dir
    )
    print("✓ 四部分分析完成，输出目录:", os.path.abspath(out_dir))
    print("  分箱汇总行数:", len(part1), "ALE明细行数:", len(part2))
    print(part3.to_string(index=False))
    print(part4.to_string(index=False))
    return {"part1_bins": part1, "part2_ale": part2, "part3_curves": part3,
            "part4_budget": part4, "part4_customer_detail": detail}


if __name__ == "__main__":
    oof_diagnostic_results = main(globals())
