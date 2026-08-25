#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大规模最优额度计算：预计算概率网格 + 可切换组合优化后端。

适用规模示例：
    4万客户 × 1999个额度网格点（1000~1000000，步长500）

当前主流程将候选额度对应的风险调整收益预先计算，并联合处理总额度预算、
风险预算和人才等级平均额度单调约束。Python 3.6 环境默认使用明确标记的
可行解启发式算法；具备 SciPy milp 时仍可显式切换到整数规划后端。

推荐概率网格文件目录格式：
    probability_grid_large/
        customer_id.npy   shape=(n,)
        grid.npy          shape=(m,)
        p_usage.npy       shape=(n, m), float32 推荐
        p_default.npy     shape=(n, m), float32 推荐

也支持单个 .npz 文件：
    customer_id, grid, p_usage, p_default
但 .npz 一般不能像 .npy 一样稳定内存映射，不如目录格式适合大规模数据。
"""

import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

OPTIMIZER_STATE_VERSION = "runtime_minimum_limit_policy_v15"
DEFAULT_CUSTOMER_MIN_LIMIT_DECREASE = 50_000.0
DEFAULT_TIER_MIN_LIMITS = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.0,
    6: 0.0,
    7: 0.0,
    8: 0.0,
}

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
    drop_post_label_cols,
    feature_engineering_potential,
)
from optimal_credit_limit import (
    OptimalCreditLimitConfig,
    OptimalCreditLimitCalculator,
    build_dual_labels_train_threshold,
)
from portfolio_milp_optimizer import (
    build_portfolio_candidate_table,
    concentration_shares,
    distribution_summary,
    risk_adjusted_value,
    solve_discrete_portfolio_heuristic,
    solve_discrete_portfolio_lagrangian,
    solve_discrete_portfolio_milp,
)
def audit_runtime_minimum_limit_policy(
    result_df,
    expected_tier_min_limits,
    expected_customer_min_limit_decrease,
):
    """核验结果表实际使用的等级下限、客户降幅和逐客户硬下限。"""
    required = {
        "talent_level",
        "original_credit_limit",
        "tier_minimum_limit",
        "customer_minimum_limit",
        "customer_min_limit_decrease",
    }
    missing = sorted(required - set(result_df.columns))
    if missing:
        raise ValueError("额度结果缺少下限政策审计字段: %s" % missing)
    levels = pd.to_numeric(result_df["talent_level"], errors="raise").to_numpy(dtype=int)
    original = pd.to_numeric(
        result_df["original_credit_limit"], errors="raise"
    ).to_numpy(dtype=float)
    actual_tier = pd.to_numeric(
        result_df["tier_minimum_limit"], errors="raise"
    ).to_numpy(dtype=float)
    actual_customer = pd.to_numeric(
        result_df["customer_minimum_limit"], errors="raise"
    ).to_numpy(dtype=float)
    actual_decrease = pd.to_numeric(
        result_df["customer_min_limit_decrease"], errors="raise"
    ).to_numpy(dtype=float)
    expected_decrease = float(expected_customer_min_limit_decrease)
    expected_tier = np.asarray([
        float(expected_tier_min_limits.get(
            int(level), expected_tier_min_limits.get(str(int(level)), 0.0)
        ))
        for level in levels
    ])
    expected_customer = np.maximum(expected_tier, original - expected_decrease)
    if not np.allclose(actual_tier, expected_tier, rtol=0.0, atol=1e-9):
        raise ValueError("结果中的等级最低额度与本次 TIER_MIN_LIMIT_POLICY 不一致")
    if not np.allclose(actual_decrease, expected_decrease, rtol=0.0, atol=1e-9):
        raise ValueError("结果中的 CUSTOMER_MIN_LIMIT_DECREASE 与本次配置不一致")
    if not np.allclose(actual_customer, expected_customer, rtol=0.0, atol=1e-9):
        raise ValueError("结果中的逐客户额度硬下限计算错误")
    summary = result_df.assign(
        _tier_minimum_limit=actual_tier,
        _customer_minimum_limit=actual_customer,
    ).groupby("talent_level", as_index=False).agg(
        customer_count=("talent_level", "size"),
        tier_minimum_limit=("_tier_minimum_limit", "first"),
        resolved_customer_minimum_min=("_customer_minimum_limit", "min"),
        resolved_customer_minimum_max=("_customer_minimum_limit", "max"),
    )
    print("✓ 结果级额度下限政策校验通过")
    print(summary.to_string(index=False))
    return summary


class LargePrecomputedGridConfig(OptimalCreditLimitConfig):
    """大规模预计算概率网格配置。"""

    def __init__(self):
        super().__init__()
        self.prob_grid_path = globals().get(
            "PROB_GRID_DIR_OPT_OVERRIDE",
            "probability_grid_large_crossfit_calibrated"
        )
        self.limit_step = 500.0
        self.chunk_size = 2000
        self.lambda_iterations = 70
        self.repair_batch_size = 20000
        self.cleaned_file = "data_cleaned.csv"
        self.csv_encoding = "utf-8-sig"
        self.snapshot_date = "2026-01-31"
        self.apply_maturity_filter = False
        self.maturity_cutoff = "2026-07-21"
        self.apply_eff_date_filter = True
        self.eff_date_lower = "2025-01-01"
        self.eff_date_upper = "2026-03-31"
        self.dedup_cst_loan = False
        self.y_freq_mode = "bout_gt0_and_curr_p80"
        # 人才等级只进入额度区间及组均额度约束，不再作为收益乘数。
        self.lgd_coefficient = 0.45
        self.linear_cost = 0.005
        self.quadratic_cost = 1e-9
        self.risk_budget = None
        self.risk_tolerance = 1.05
        self.enforce_group_mean_monotonic = True
        self.group_mean_min_ratio = {2: 1.0, 3: 1.0, 4: 0.8, 5: 0.8}
        self.solver_backend = "lagrangian"
        self.heuristic_max_rounds = 30
        self.lagrangian_iterations = 40
        self.lagrangian_time_limit_seconds = 60.0
        self.lagrangian_step_size = 0.5
        self.lagrangian_multiplier_cap = 1000.0
        self.local_search_enabled = True
        self.local_search_max_passes = 3
        self.local_search_time_limit_seconds = 30.0
        self.local_search_pair_candidate_pool = 80
        self.milp_max_variables = 400000
        self.milp_candidates_per_customer = 16
        self.milp_time_limit_seconds = 600.0
        self.milp_relative_gap = 0.01
        self.optimization_scope = "all"
        self.optimization_sample_enabled = False
        self.optimization_sample_size = None
        self.optimization_sample_random_state = 42
        self.report_dir = "reports"
        self.utilization_lookup_file = "utilization_customer_lookup.csv"
        self.utilization_column = "utilization_used"
        self.gross_interest_rate = 0.03
        self.ftp_rate = 0.02
        self.linear_cost_mode = "manual"
        self.customer_min_limit_enabled = True
        self.customer_min_limit_decrease = DEFAULT_CUSTOMER_MIN_LIMIT_DECREASE
        self.min_limit.update(DEFAULT_TIER_MIN_LIMITS)


class LargePrecomputedGridCalculator(OptimalCreditLimitCalculator):
    """读取预计算概率网格并执行配置的组合优化后端。"""

    def __init__(self, config: LargePrecomputedGridConfig):
        super().__init__(config)
        self.prob_grid = {}

    def load_data(self):
        """加载潜力客户数据，对齐 generate_probability_grid / Cell 3。"""
        print("加载数据（load_kechuang_potential_data）...")
        cleaned_file = str(getattr(self.config, "cleaned_file", "data_cleaned.csv"))
        csv_encoding = str(getattr(self.config, "csv_encoding", "utf-8-sig"))

        if os.path.isfile(cleaned_file):
            print(f"  读取清洗后数据: {cleaned_file}")
            work = pd.read_csv(cleaned_file, encoding=csv_encoding)
            if "y_freq" not in work.columns or "y_dq_risk" not in work.columns:
                raise ValueError(
                    f"{cleaned_file} 缺少 y_freq 或 y_dq_risk，请先运行 Cell 3 重新清洗"
                )
        else:
            excel_path = str(self.config.excel_path)
            print(f"  未找到 {cleaned_file}，从原始文件重跑预处理: {excel_path}")
            df0 = read_data(excel_path, csv_encoding=csv_encoding)
            df0 = rename_kechuang_cols(df0)
            df0 = report_cst_loan_duplicates(df0)
            if bool(getattr(self.config, "dedup_cst_loan", False)):
                df0 = deduplicate_exact_cst_loan(df0)
            df0 = cast_cat_cols(df0)
            df0 = filter_by_maturity(
                df0,
                apply_filter=bool(getattr(self.config, "apply_maturity_filter", False)),
                maturity_cutoff=str(getattr(self.config, "maturity_cutoff", "2026-07-21")),
            )
            if bool(getattr(self.config, "apply_eff_date_filter", True)):
                df0 = filter_by_eff_date(
                    df0,
                    eff_date_lower=str(getattr(self.config, "eff_date_lower", "2025-01-01")),
                    eff_date_upper=str(getattr(self.config, "eff_date_upper", "2026-03-31")),
                )
            else:
                print("  apply_eff_date_filter=False，跳过生效日筛选")
            df0 = filter_dq_start_customers(df0)
            df_agg = aggregate_by_customer_potential(df0)
            df_clean = clean_data_potential(
                df_agg,
                snapshot_date=str(getattr(self.config, "snapshot_date", "2026-01-31")),
                target="y_dq_risk",
            )
            y_freq_mode = str(getattr(self.config, "y_freq_mode", "bout_gt0_and_curr_p80"))
            df_labeled, _, _ = build_dual_labels_train_threshold(
                df_clean,
                y_freq_mode=y_freq_mode,
                train_ratio=float(getattr(
                    self.config, "label_threshold_train_ratio", 0.60
                )),
            )
            work = drop_post_label_cols(df_labeled, target="y_dq_risk")

        if "credamt" in work.columns and "授信额度" not in work.columns:
            work = work.copy()
            work["授信额度"] = work["credamt"]

        X_usage, y_usage, feature_names_usage, _ = feature_engineering_potential(
            work, target="y_freq",
            add_quota_sq=False, add_quota_cube=False, add_quota_log=False,
        )
        X_default, y_default, feature_names_default, _ = feature_engineering_potential(
            work, target="y_dq_risk",
            add_quota_sq=False, add_quota_cube=False, add_quota_log=False,
        )

        self.data_info = {
            "df_raw": work,
            "df_aggregated": work,
            "df_cleaned": work,
            "X_usage": X_usage,
            "y_usage": y_usage,
            "X_default": X_default,
            "y_default": y_default,
            "feature_names_usage": feature_names_usage,
            "feature_names_default": feature_names_default,
            "thresholds_usage": {},
            "thresholds_default": {},
        }

        utilization_path = str(getattr(
            self.config, "utilization_lookup_file", "utilization_customer_lookup.csv"
        ))
        if not os.path.isfile(utilization_path):
            raise FileNotFoundError(
                "未找到客户支用率表: %s；请先运行 notebook 的清洗/支用率统计 Cell。"
                % utilization_path
            )
        utilization = pd.read_csv(
            utilization_path, encoding=csv_encoding, dtype={"cst_id": str}
        )
        utilization_col = str(getattr(self.config, "utilization_column", "utilization_used"))
        required_utilization = {
            "cst_id", "utilization_method", "utilization_raw", utilization_col
        }
        missing_utilization = sorted(required_utilization - set(utilization.columns))
        if missing_utilization:
            raise ValueError("客户支用率表缺少字段: %s" % missing_utilization)
        if utilization["cst_id"].duplicated().any():
            raise ValueError("客户支用率表 cst_id 存在重复")
        utilization = utilization.set_index("cst_id")
        work_ids = work["cst_id"].astype(str)
        missing_ids = work_ids[~work_ids.isin(utilization.index)]
        if len(missing_ids):
            raise ValueError("客户支用率表缺少当前客户，示例: %s" % missing_ids.head(10).tolist())
        aligned_utilization = utilization.loc[work_ids].reset_index(drop=True)
        for col in ("utilization_raw", utilization_col):
            self.data_info["df_aggregated"][col] = pd.to_numeric(
                aligned_utilization[col], errors="coerce"
            ).to_numpy(dtype=float)
        utilization_methods = aligned_utilization["utilization_method"].dropna().astype(str).unique()
        if len(utilization_methods) != 1:
            raise ValueError("客户支用率表中的 utilization_method 必须全表一致")
        self.data_info["utilization_method"] = str(utilization_methods[0])
        if not np.isfinite(self.data_info["df_aggregated"][utilization_col]).all():
            raise ValueError("客户支用率存在空值或非有限值")
        print(f"数据加载完成，样本数: {len(X_usage)}")
        # 大规模网格：不调用父类 _auto_configure（避免旧版 seaborn 画图报错，且 grid 上下限由 main 配置）
        df = self.data_info.get("df_aggregated")
        cred_col = "credamt" if df is not None and "credamt" in df.columns else "授信额度"
        if df is not None and cred_col in df.columns:
            vals = pd.to_numeric(df[cred_col], errors="coerce").dropna()
            if len(vals) > 0:
                print(f"  原始额度总额: {float(vals.sum()):,.0f}  （样本 {len(vals):,}）")
        return self.data_info

    def train_models(self):
        """不训练模型，只读取预计算概率网格，并保存原始额度对应概率。"""
        print("读取大规模预计算概率网格，不训练模型...")
        self.load_probability_grid()

        # 频繁支用概率网格不再进入目标函数。保留原网格文件以兼容既有目录，
        # 运行时用客户历史支用率构造零拷贝的静态二维视图。
        utilization_col = str(getattr(self.config, "utilization_column", "utilization_used"))
        utilization = pd.to_numeric(
            self.data_info["df_aggregated"][utilization_col], errors="coerce"
        ).to_numpy(dtype=np.float32)
        grid_width = int(self.prob_grid["p_default"].shape[1])
        utilization_grid = np.broadcast_to(utilization[:, None], (len(utilization), grid_width))
        self.prob_grid["p_usage"] = utilization_grid
        self.prob_grid["p_usage_raw"] = utilization_grid
        print("  目标函数支用项已切换为客户历史支用率；频繁支用概率网格未使用")

        # 字段名兼容：聚合后为 credamt；旧版可能仍叫 授信额度
        _df_agg = self.data_info["df_aggregated"]
        _cred_col = "credamt" if "credamt" in _df_agg.columns else "授信额度"
        base_limits = pd.to_numeric(
            _df_agg[_cred_col], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        p_usage_base, p_default_base = self._lookup_probabilities(base_limits)
        self.models = {
            "y_proba_usage": p_usage_base,
            "y_proba_default": p_default_base,
        }
        print("概率网格读取完成")
        return self.models

    def _subset_data_rows_only(self, idx, label: str):
        """在概率矩阵载入前，仅把客户表/特征裁剪到网格客户并保持同序。"""
        idx = np.asarray(idx, dtype=int).reshape(-1)
        n_before = len(self.data_info["df_aggregated"])
        if not len(idx) or len(np.unique(idx)) != len(idx):
            raise ValueError(f"{label} 的客户索引为空或存在重复。")
        if np.any(idx < 0) or np.any(idx >= n_before):
            raise IndexError(f"{label} 的客户索引超出 0~{n_before - 1}。")
        for key in (
            "df_raw", "df_aggregated", "df_cleaned",
            "X_usage", "y_usage", "X_default", "y_default",
        ):
            value = self.data_info.get(key)
            if value is None:
                continue
            if len(value) != n_before:
                raise ValueError(
                    f"{label} 前 data_info[{key}] 长度={len(value)}，期望={n_before}。"
                )
            self.data_info[key] = value.iloc[idx].reset_index(drop=True)
        self.source_row_indices = idx.copy()
        self.scope_row_indices = idx.copy()
        print(f"[{label}] 客户数: {n_before:,} -> {len(idx):,}")

    def _subset_aligned_rows(self, idx, label: str):
        """同步裁剪客户表、特征/标签和四套概率矩阵，防止样本边界错位。"""
        idx = np.asarray(idx, dtype=int).reshape(-1)
        n_before = len(self.data_info["df_aggregated"])
        if not len(idx):
            raise ValueError(f"{label} 的客户索引为空。")
        if np.any(idx < 0) or np.any(idx >= n_before):
            raise IndexError(f"{label} 的客户索引超出当前样本范围 0~{n_before - 1}。")
        if len(np.unique(idx)) != len(idx):
            raise ValueError(f"{label} 的客户索引存在重复。")

        row_keys = (
            "df_raw", "df_aggregated", "df_cleaned",
            "X_usage", "y_usage", "X_default", "y_default",
        )
        for key in row_keys:
            value = self.data_info.get(key)
            if value is None:
                continue
            if len(value) != n_before:
                raise ValueError(
                    f"{label} 前 data_info[{key}] 长度={len(value)}，"
                    f"与客户数={n_before} 不一致。"
                )
            selected = value.iloc[idx]
            self.data_info[key] = selected.reset_index(drop=True)

        for key in ("p_usage", "p_default", "p_usage_raw", "p_default_raw"):
            matrix = self.prob_grid.get(key)
            if matrix is None or matrix.shape[0] != n_before:
                actual = None if matrix is None else matrix.shape[0]
                raise ValueError(
                    f"{label} 前概率矩阵 {key} 客户数={actual}，期望={n_before}。"
                )
            self.prob_grid[key] = np.asarray(matrix[idx], dtype=np.float32)

        source = getattr(self, "source_row_indices", np.arange(n_before, dtype=int))
        if len(source) != n_before:
            raise ValueError("source_row_indices 与当前客户数不一致。")
        self.source_row_indices = np.asarray(source[idx], dtype=int)
        self.scope_row_indices = self.source_row_indices.copy()

        df_agg = self.data_info["df_aggregated"]
        cred_col = "credamt" if "credamt" in df_agg.columns else "授信额度"
        base_limits = pd.to_numeric(
            df_agg[cred_col], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        p_usage_base, p_default_base = self._lookup_probabilities(base_limits)
        self.models = {
            "y_proba_usage": p_usage_base,
            "y_proba_default": p_default_base,
        }
        print(f"[{label}] 客户数: {n_before:,} -> {len(idx):,}")

    def apply_optimization_scope(self, scope: str):
        """将同一概率网格限定为开发拟合客户、最终测试客户或全部客户。"""
        scope = str(scope or "all").strip().lower()
        n_current = len(self.data_info["df_aggregated"])
        if not hasattr(self, "source_row_indices") or len(self.source_row_indices) != n_current:
            self.source_row_indices = np.arange(n_current, dtype=int)
        if scope in ("all", "full", "crossfit_all"):
            self.config.optimization_scope = "all"
            self.scope_row_indices = self.source_row_indices.copy()
            scope_label = "概率网格固定抽样客户" if getattr(
                self, "_probability_grid_pre_sampled", False
            ) else "全部客户"
            print("[优化范围] %s: %d" % (scope_label, n_current))
            return
        scope_files = {
            "test": "test_idx.npy",
            "final_test": "test_idx.npy",
            "fit": "fit_idx.npy",
            "development": "fit_idx.npy",
        }
        if scope not in scope_files:
            raise ValueError("optimization_scope 仅支持 all、fit/development 或 test")

        grid_dir = str(self.config.prob_grid_path)
        index_path = os.path.join(grid_dir, scope_files[scope])
        if not os.path.isfile(index_path):
            raise FileNotFoundError("优化范围索引不存在: %s" % index_path)
        idx = np.asarray(np.load(index_path), dtype=int)
        if len(idx) == 0:
            raise ValueError("优化范围索引为空: %s" % index_path)
        sample_enabled = bool(
            getattr(self.config, "optimization_sample_enabled", False)
        )
        sample_size = getattr(self.config, "optimization_sample_size", None)
        if sample_enabled and getattr(self, "_probability_grid_pre_sampled", False):
            self._optimization_sample_already_applied = True
            self._subset_aligned_rows(idx, "优化范围（复用概率网格固定抽样名单）")
        elif sample_enabled:
            if sample_size is None or int(sample_size) <= 0:
                raise ValueError(
                    "OPTIMIZATION_SAMPLE_ENABLED=True 时，"
                    "OPTIMIZATION_SAMPLE_SIZE 必须为正整数。"
                )
            if int(sample_size) < len(idx):
                all_levels = self._extract_talent_levels()
                relative_idx = self._stratified_sample_positions(
                    all_levels[idx],
                    sample_size=int(sample_size),
                    random_state=int(getattr(
                        self.config, "optimization_sample_random_state", 42
                    )),
                )
                idx = idx[relative_idx]
                self._optimization_sample_already_applied = True
                self._subset_aligned_rows(idx, "优化范围+抽样")
            else:
                self._subset_aligned_rows(idx, "优化范围")
        else:
            self._subset_aligned_rows(idx, "优化范围")
        normalized_scope = "test" if scope in ("test", "final_test") else "fit"
        self.config.optimization_scope = normalized_scope
        scope_name = "最终测试客户" if normalized_scope == "test" else "开发拟合客户"
        print("[优化范围] 当前范围: %s" % scope_name)

    @staticmethod
    def _stratified_sample_positions(strata, sample_size: int, random_state: int):
        """按人才等级近似等比例抽样，保证样本量允许时每个现有等级至少一人。"""
        strata = np.asarray(strata).reshape(-1)
        sample_size = int(sample_size)
        levels, counts = np.unique(strata, return_counts=True)
        if sample_size < len(levels):
            raise ValueError(
                f"抽样数 {sample_size} 小于当前人才等级数 {len(levels)}，"
                "无法保证每个等级至少保留一名客户。"
            )
        ideal = counts.astype(float) * sample_size / len(strata)
        allocations = np.minimum(counts, np.maximum(1, np.floor(ideal).astype(int)))
        while int(allocations.sum()) > sample_size:
            candidates = np.flatnonzero(allocations > 1)
            if not len(candidates):
                raise RuntimeError("无法将人才等级抽样配额缩减到目标样本数。")
            chosen = int(candidates[np.argmax(allocations[candidates] - ideal[candidates])])
            allocations[chosen] -= 1
        while int(allocations.sum()) < sample_size:
            candidates = np.flatnonzero(allocations < counts)
            if not len(candidates):
                raise RuntimeError("无法将人才等级抽样配额扩充到目标样本数。")
            chosen = int(candidates[np.argmax(ideal[candidates] - allocations[candidates])])
            allocations[chosen] += 1

        rng = np.random.RandomState(int(random_state))
        selected = []
        for level, allocation in zip(levels, allocations):
            positions = np.flatnonzero(strata == level)
            rng.shuffle(positions)
            selected.extend(positions[:int(allocation)].tolist())
        return np.sort(np.asarray(selected, dtype=int))

    def apply_optimization_sample(
        self,
        enabled: bool,
        sample_size: Optional[int],
        random_state: int = 42,
    ):
        """在 optimization_scope 内按人才等级分层抽样，并同步裁剪所有输入。"""
        self.config.optimization_sample_enabled = bool(enabled)
        self.config.optimization_sample_size = None if sample_size is None else int(sample_size)
        self.config.optimization_sample_random_state = int(random_state)
        n_current = len(self.data_info["df_aggregated"])
        if enabled and getattr(self, "_probability_grid_pre_sampled", False):
            self._optimization_sample_already_applied = True
            print(
                "[优化抽样] 已复用概率网格生成阶段的固定名单，当前客户数: %d"
                % n_current
            )
            return
        if not enabled:
            print("[优化抽样] 关闭，使用当前范围全部客户: %d" % n_current)
            return
        if getattr(self, "_optimization_sample_already_applied", False):
            print("[优化抽样] 已与 optimization_scope 合并执行，当前客户数: %d" % n_current)
            return
        if sample_size is None or int(sample_size) <= 0:
            raise ValueError("OPTIMIZATION_SAMPLE_ENABLED=True 时，OPTIMIZATION_SAMPLE_SIZE 必须为正整数。")
        if int(sample_size) >= n_current:
            print(
                "[优化抽样] 目标样本数 %d >= 当前客户数 %d，使用全部客户。"
                % (int(sample_size), n_current)
            )
            return
        talent_levels = self._extract_talent_levels()
        idx = self._stratified_sample_positions(
            talent_levels,
            sample_size=int(sample_size),
            random_state=int(random_state),
        )
        self._subset_aligned_rows(idx, "优化抽样")
        sampled_levels = self._extract_talent_levels()
        level_counts = dict(zip(*np.unique(sampled_levels, return_counts=True)))
        print(f"[优化抽样] random_state={int(random_state)}，等级分布={level_counts}")

    def load_probability_grid(self):
        """读取 .npy 目录或 .npz 概率网格，并按当前客户顺序对齐。"""
        path = str(getattr(self.config, "prob_grid_path", "probability_grid_large"))
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到概率网格路径: {path}")

        if os.path.isdir(path):
            metadata_path = os.path.join(path, "grid_metadata.json")
            if os.path.isfile(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                    grid_metadata = json.load(metadata_file)
            else:
                grid_metadata = {"probability_grid_sampled": False}
            customer_id = np.load(os.path.join(path, "customer_id.npy"), allow_pickle=True)
            grid = np.load(os.path.join(path, "grid.npy"), mmap_mode="r")
            p_usage_raw = np.load(os.path.join(path, "p_usage.npy"), mmap_mode="r")
            p_default_raw = np.load(os.path.join(path, "p_default.npy"), mmap_mode="r")
            raw_usage_path = os.path.join(path, "p_usage_raw.npy")
            raw_default_path = os.path.join(path, "p_default_raw.npy")
            if not os.path.isfile(raw_usage_path) or not os.path.isfile(raw_default_path):
                raise FileNotFoundError(
                    "交叉拟合目录缺少校准前概率网格 p_usage_raw.npy / "
                    "p_default_raw.npy；请用新版 Cell 5 重新生成。"
                )
            p_usage_before_cal_raw = np.load(raw_usage_path, mmap_mode="r")
            p_default_before_cal_raw = np.load(raw_default_path, mmap_mode="r")
        elif path.lower().endswith(".npz"):
            grid_metadata = {"probability_grid_sampled": False}
            arr = np.load(path, allow_pickle=True)
            required = {"customer_id", "grid", "p_usage", "p_default", "p_usage_raw", "p_default_raw"}
            missing = required - set(arr.files)
            if missing:
                raise ValueError(f"NPZ 缺少字段: {sorted(missing)}")
            customer_id = arr["customer_id"]
            grid = arr["grid"]
            p_usage_raw = arr["p_usage"]
            p_default_raw = arr["p_default"]
            p_usage_before_cal_raw = arr["p_usage_raw"]
            p_default_before_cal_raw = arr["p_default_raw"]
            print("[提示] 大规模场景更推荐 .npy 目录格式；.npz 可能会一次性载入内存。")
        else:
            raise ValueError("大规模版本仅支持 .npy 目录或 .npz 文件，不支持 CSV/XLSX 长表。")

        current_ids = self.data_info["df_aggregated"]["cst_id"].astype(str).to_numpy()
        file_ids = np.asarray(customer_id).astype(str)
        if len(np.unique(current_ids)) != len(current_ids):
            raise ValueError("当前清洗数据的 cst_id 存在重复，无法可靠对齐概率网格。")
        if len(np.unique(file_ids)) != len(file_ids):
            raise ValueError("概率网格 customer_id.npy 存在重复客户。")

        grid_is_sampled = bool(grid_metadata.get("probability_grid_sampled", False))
        configured_sampled = bool(
            getattr(self.config, "optimization_sample_enabled", False)
        )
        if grid_is_sampled != configured_sampled:
            raise ValueError(
                "概率网格抽样口径与 OPTIMIZATION_SAMPLE_ENABLED 不一致: "
                f"grid={grid_is_sampled}, config={configured_sampled}。请使用对应目录。"
            )
        if grid_is_sampled:
            saved_size = grid_metadata.get("sample_size")
            configured_size = getattr(self.config, "optimization_sample_size", None)
            saved_seed = int(grid_metadata.get("sample_random_state", -1))
            configured_seed = int(
                getattr(self.config, "optimization_sample_random_state", 42)
            )
            if int(saved_size) != int(configured_size) or saved_seed != configured_seed:
                raise ValueError(
                    "抽样概率网格的 sample_size/random_state 与当前优化配置不一致。"
                )

        if len(file_ids) < len(current_ids):
            if not grid_is_sampled:
                raise ValueError("概率网格客户少于清洗数据，但缺少抽样元数据。")
            current_pos = {cid: i for i, cid in enumerate(current_ids)}
            missing = [cid for cid in file_ids if cid not in current_pos]
            if missing:
                raise ValueError(f"抽样概率网格客户不在当前清洗数据中，示例: {missing[:10]}")
            data_order = np.asarray([current_pos[cid] for cid in file_ids], dtype=int)
            self._subset_data_rows_only(data_order, "概率网格固定抽样名单")
            current_ids = self.data_info["df_aggregated"]["cst_id"].astype(str).to_numpy()
            order = np.arange(len(file_ids), dtype=int)
            self._probability_grid_pre_sampled = True
            self.grid_metadata = grid_metadata
        else:
            if len(file_ids) != len(current_ids):
                raise ValueError(
                    f"概率网格客户数异常: grid={len(file_ids)}, data={len(current_ids)}"
                )
            order = self._align_customer_order(file_ids, current_ids)
            self._probability_grid_pre_sampled = grid_is_sampled
            self.grid_metadata = grid_metadata

        if np.array_equal(order, np.arange(len(order))):
            p_usage = p_usage_raw
            p_default = p_default_raw
            p_usage_before_cal = p_usage_before_cal_raw
            p_default_before_cal = p_default_before_cal_raw
        else:
            # 非同序时只能重排，会产生新矩阵；大规模生产建议生成概率网格时即按 cst_id 对齐。
            print("[提示] 概率网格客户顺序与当前数据不一致，正在重排；这会增加内存占用。")
            p_usage = np.asarray(p_usage_raw)[order]
            p_default = np.asarray(p_default_raw)[order]
            p_usage_before_cal = np.asarray(p_usage_before_cal_raw)[order]
            p_default_before_cal = np.asarray(p_default_before_cal_raw)[order]

        if p_usage.shape != p_default.shape:
            raise ValueError("p_usage 与 p_default 形状不一致。")
        if p_usage_before_cal.shape != p_usage.shape or p_default_before_cal.shape != p_default.shape:
            raise ValueError("校准前后概率网格形状不一致。")
        if p_usage.shape[0] != len(current_ids):
            raise ValueError(f"概率网格客户数不一致: grid={p_usage.shape[0]}, data={len(current_ids)}")
        if p_usage.shape[1] != len(grid):
            raise ValueError(f"概率矩阵列数与 grid 长度不一致: matrix={p_usage.shape[1]}, grid={len(grid)}")

        self.prob_grid = {
            "grid": np.asarray(grid, dtype=np.float64),
            "p_usage": p_usage,
            "p_default": p_default,
            "p_usage_raw": p_usage_before_cal,
            "p_default_raw": p_default_before_cal,
        }
        print(
            f"[概率网格] customers={p_usage.shape[0]:,}, grid_points={p_usage.shape[1]:,}, "
            f"range={float(np.min(grid)):,.0f}~{float(np.max(grid)):,.0f}, "
            f"p_usage_dtype={p_usage.dtype}, p_default_dtype={p_default.dtype}"
        )

    @staticmethod
    def _align_customer_order(file_ids: np.ndarray, current_ids: np.ndarray) -> np.ndarray:
        """返回把概率矩阵重排到当前客户顺序的下标。"""
        pos = {cid: i for i, cid in enumerate(file_ids)}
        missing = [cid for cid in current_ids if cid not in pos]
        if missing:
            raise ValueError(f"概率网格缺少当前客户，示例: {missing[:10]}")
        return np.asarray([pos[cid] for cid in current_ids], dtype=int)

    def _grid_indices_for_limits(self, L_array: np.ndarray) -> np.ndarray:
        """把额度映射到最近的网格下标。"""
        grid = self.prob_grid["grid"]
        L = np.asarray(L_array, dtype=float).reshape(-1)
        idx = np.searchsorted(grid, L)
        idx = np.clip(idx, 0, len(grid) - 1)
        left = np.clip(idx - 1, 0, len(grid) - 1)
        choose_left = np.abs(grid[left] - L) <= np.abs(grid[idx] - L)
        return np.where(choose_left, left, idx).astype(int)

    def _lookup_probabilities(self, L_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """按额度数组从预计算网格取最近点概率。"""
        idx = self._grid_indices_for_limits(L_array)
        rows = np.arange(len(idx))
        p_usage = np.asarray(self.prob_grid["p_usage"][rows, idx], dtype=float)
        p_default = np.asarray(self.prob_grid["p_default"][rows, idx], dtype=float)
        return p_usage, p_default

    def _lookup_raw_probabilities(self, L_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Lookup pre-isotonic out-of-fold probabilities at the requested limits."""
        idx = self._grid_indices_for_limits(L_array)
        rows = np.arange(len(idx))
        return (
            np.asarray(self.prob_grid["p_usage_raw"][rows, idx], dtype=float),
            np.asarray(self.prob_grid["p_default_raw"][rows, idx], dtype=float),
        )

    def _effective_default_prob(self, p_default):
        """返回校准违约概率；风险折算由 LGD 系数承担，不再缩放概率。"""
        return np.asarray(p_default, dtype=float)

    def _level_arrays(self, talent_levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回等级相关的上下限、LGD 系数和利率数组。"""
        levels = np.asarray(talent_levels).reshape(-1).astype(int)
        L_min = np.asarray([self.config.min_limit.get(level, 0.0) for level in levels], dtype=float)
        L_max = np.asarray([self.config.max_limit.get(level, float("inf")) for level in levels], dtype=float)
        lgd = np.full(len(levels), float(getattr(self.config, "lgd_coefficient", 0.45)), dtype=float)
        rates = np.asarray([self.config.interest_rates.get(level, 0.05) for level in levels], dtype=float)
        return L_min, L_max, lgd, rates

    @staticmethod
    def _tier_label(level: int) -> str:
        return {8: "A", 7: "B", 6: "C", 5: "D", 4: "E", 3: "F1", 2: "F2", 1: "F3"}.get(
            int(level), str(level)
        )

    def _print_limit_distribution_diagnostics(
        self,
        limits: np.ndarray,
        talent_levels: np.ndarray,
        title: str,
    ):
        """打印总体/分等级 0 额度占比，以及每个等级选择了哪些优化额度。"""
        limits = np.asarray(limits, dtype=float)
        levels = np.asarray(talent_levels).reshape(-1).astype(int)
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print("总体所有人才 0 额度占比: %.4f%% (%d/%d)" % (
            float(np.mean(limits == 0.0) * 100.0),
            int(np.sum(limits == 0.0)),
            len(limits),
        ))

        summary_rows = []
        dist_rows = []
        for level in sorted(np.unique(levels), reverse=True):
            mask = levels == level
            sub = limits[mask]
            tier = self._tier_label(level)
            summary_rows.append({
                "talent_level": int(level),
                "tier": tier,
                "n": int(mask.sum()),
                "zero_share": float(np.mean(sub == 0.0)),
                "mean": float(np.mean(sub)),
                "median": float(np.median(sub)),
                "min": float(np.min(sub)),
                "p75": float(np.percentile(sub, 75)),
                "p90": float(np.percentile(sub, 90)),
                "p95": float(np.percentile(sub, 95)),
                "max": float(np.max(sub)),
            })
            vals, cnts = np.unique(sub, return_counts=True)
            for v, c in zip(vals, cnts):
                dist_rows.append({
                    "talent_level": int(level),
                    "tier": tier,
                    "credit_limit": float(v),
                    "n": int(c),
                    "share": float(c / len(sub)),
                })
            print("\n等级 %s(level=%d) 0额度占比: %.4f%% (%d/%d)" % (
                tier, level, np.mean(sub == 0.0) * 100.0, int(np.sum(sub == 0.0)), len(sub)
            ))
            print("  mean=%.2f median=%.2f min=%.2f p75=%.2f p90=%.2f p95=%.2f max=%.2f" % (
                np.mean(sub), np.median(sub), np.min(sub),
                np.percentile(sub, 75), np.percentile(sub, 90),
                np.percentile(sub, 95), np.max(sub),
            ))
            dist_df = pd.DataFrame({"credit_limit": vals, "n": cnts})
            dist_df["share"] = dist_df["n"] / len(sub)
            print("  优化额度分布（该等级全部额度取值）:")
            print(dist_df.to_string(index=False))

        report_dir = globals().get("REPORTS_DIR_OPT_OVERRIDE", globals().get("REPORTS_DIR", "reports"))
        os.makedirs(report_dir, exist_ok=True)
        safe_title = (
            title.replace(" ", "_")
            .replace("/", "_")
            .replace("：", "_")
            .replace(":", "_")
        )
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(report_dir, f"{safe_title}_limit_summary_by_tier.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(dist_rows).to_csv(
            os.path.join(report_dir, f"{safe_title}_limit_distribution_by_tier.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    def _print_value_per_limit_diagnostics(
        self,
        talent_levels: np.ndarray,
        lam: float,
    ):
        """检查目标函数单位利润是否整体偏负。"""
        grid = self.prob_grid["grid"]
        p_usage = self.prob_grid["p_usage"]
        p_default = self.prob_grid["p_default"]
        n = p_usage.shape[0]
        levels = np.asarray(talent_levels).reshape(-1).astype(int)
        c1 = float(getattr(self.config, "linear_cost", 0.0))
        c2 = float(getattr(self.config, "quadratic_cost", 0.0))
        chunk_size = int(getattr(self.config, "chunk_size", 2000))

        unit_min = np.inf
        unit_max = -np.inf
        unit_sum = 0.0
        unit_neg = 0
        value_min = np.inf
        value_max = -np.inf
        value_sum = 0.0
        value_neg = 0
        total_count = 0

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            _, _, lgd, rates = self._level_arrays(levels[start:end])
            pu = np.asarray(p_usage[start:end], dtype=np.float32)
            pd_ = self._effective_default_prob(np.asarray(p_default[start:end], dtype=np.float32))
            unit_profit_no_lambda = (
                rates[:, None] * pu
                - lgd[:, None] * pd_ * pu
                - c1
                - c2 * grid.reshape(1, -1)
            )
            value_per_limit = unit_profit_no_lambda - float(lam)

            unit_min = min(unit_min, float(unit_profit_no_lambda.min()))
            unit_max = max(unit_max, float(unit_profit_no_lambda.max()))
            unit_sum += float(unit_profit_no_lambda.sum())
            unit_neg += int(np.sum(unit_profit_no_lambda < 0.0))
            value_min = min(value_min, float(value_per_limit.min()))
            value_max = max(value_max, float(value_per_limit.max()))
            value_sum += float(value_per_limit.sum())
            value_neg += int(np.sum(value_per_limit < 0.0))
            total_count += int(value_per_limit.size)

        print("\n" + "=" * 80)
        print("目标函数单位值诊断 = rate*u - lambda_LGD*p_default*u - c1 - c2*L - lambda")
        print("=" * 80)
        print("诊断范围: 全量客户 %d, grid点数: %d, lambda=%.12f" % (n, len(grid), float(lam)))
        print("不含 lambda 的单位利润 unit_profit_no_lambda:")
        print(float(unit_min))
        print(float(unit_sum / max(total_count, 1)))
        print(float(unit_max))
        print("含 lambda 的 value_per_limit:")
        print(float(value_min))
        print(float(value_sum / max(total_count, 1)))
        print(float(value_max))
        print("value_per_limit < 0 占比: %.4f%%" % (value_neg / max(total_count, 1) * 100.0))
        print("unit_profit_no_lambda < 0 占比: %.4f%%" % (unit_neg / max(total_count, 1) * 100.0))

    def _debug_sample_profit_paths(
        self,
        talent_levels: np.ndarray,
        best_limits: np.ndarray,
        lam: float,
        base_idx: int = 100,
        n_random: int = 4,
    ):
        """随机抽客户打印 grid、概率、score、best_limit，并列出 10w~100w 利润。"""
        grid = self.prob_grid["grid"]
        p_usage = self.prob_grid["p_usage"]
        p_default = self.prob_grid["p_default"]
        n = p_usage.shape[0]
        if n == 0:
            return
        rng = np.random.RandomState(42)
        sample = [min(max(int(base_idx), 0), n - 1)]
        candidates = np.setdiff1d(np.arange(n), np.asarray(sample), assume_unique=False)
        if len(candidates) > 0 and n_random > 0:
            sample.extend(rng.choice(candidates, size=min(n_random, len(candidates)), replace=False).tolist())

        levels = np.asarray(talent_levels).reshape(-1).astype(int)
        c1 = float(getattr(self.config, "linear_cost", 0.0))
        c2 = float(getattr(self.config, "quadratic_cost", 0.0))
        target_limits = np.arange(100000.0, 1000000.0 + 1.0, 100000.0)

        print("\n" + "=" * 80)
        print("随机客户利润曲线诊断：打印 grid / p_usage[idx] / p_default[idx] / score[idx] / best_limit[idx]")
        print("=" * 80)
        print("grid:")
        print(grid)

        for idx in sample:
            level = int(levels[idx])
            rate = float(self.config.interest_rates.get(level, 0.05))
            lgd = float(getattr(self.config, "lgd_coefficient", 0.45))
            pu = np.asarray(p_usage[idx], dtype=float)
            pd_eff = self._effective_default_prob(np.asarray(p_default[idx], dtype=float))
            score_no_lambda = risk_adjusted_value(
                grid, pu, pd_eff, np.full(len(grid), rate), np.full(len(grid), lgd), c1, c2
            )
            unit_profit = np.divide(score_no_lambda, grid, out=np.zeros_like(score_no_lambda), where=grid > 0)
            score = score_no_lambda - float(lam) * grid

            print("\nidx=%d, talent_level=%d(%s), best_limit=%.2f, lambda=%.12f" % (
                idx, level, self._tier_label(level), float(best_limits[idx]), float(lam)
            ))
            print("p_usage[idx]:")
            print(pu)
            print("p_default[idx]:")
            print(np.asarray(p_default[idx], dtype=float))
            print("score[idx] (含 lambda):")
            print(score)
            print("best_limit[idx]:")
            print(float(best_limits[idx]))

            rows = []
            for L in target_limits:
                gi = int(np.argmin(np.abs(grid - L)))
                rows.append({
                    "limit": float(grid[gi]),
                    "p_usage": float(pu[gi]),
                    "p_default_eff": float(pd_eff[gi]),
                    "unit_profit_no_lambda": float(unit_profit[gi]),
                    "profit_no_lambda": float(score_no_lambda[gi]),
                    "score_with_lambda": float(score[gi]),
                })
            print("10w/20w/.../100w 对应利润:")
            print(pd.DataFrame(rows).to_string(index=False))

    def _best_limits_for_lambda_chunked(
        self,
        talent_levels: np.ndarray,
        lam: float,
    ) -> np.ndarray:
        """给定 lambda，分块求每个客户的 argmax_L f_i(L)-lambda*L。"""
        grid = self.prob_grid["grid"]
        p_usage = self.prob_grid["p_usage"]
        p_default = self.prob_grid["p_default"]
        n = p_usage.shape[0]
        chunk_size = int(getattr(self.config, "chunk_size", 2000))
        c1 = float(getattr(self.config, "linear_cost", 0.0))
        c2 = float(getattr(self.config, "quadratic_cost", 0.0))
        L_min, L_max, lgd, rates = self._level_arrays(talent_levels)

        best_L = np.empty(n, dtype=np.float64)
        grid_row = grid.reshape(1, -1)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            pu = np.asarray(p_usage[start:end], dtype=np.float32)
            pd_ = self._effective_default_prob(
                np.asarray(p_default[start:end], dtype=np.float32)
            )

            score = risk_adjusted_value(
                grid_row,
                pu,
                pd_,
                rates[start:end, None],
                lgd[start:end, None],
                c1,
                c2,
            )
            score = score - float(lam) * grid_row

            feasible = (
                (grid_row >= L_min[start:end, None] - 1e-9)
                & (grid_row <= L_max[start:end, None] + 1e-9)
            )
            score = np.where(feasible, score, -np.inf)
            best_idx = np.argmax(score, axis=1)
            best_L[start:end] = grid[best_idx]

        return best_L

    def _one_step_delta(
        self,
        idx_rows: np.ndarray,
        idx_grid_from: np.ndarray,
        idx_grid_to: np.ndarray,
        talent_levels: np.ndarray,
    ) -> np.ndarray:
        """计算一批客户从一个网格点移动到另一个网格点的利润变化。"""
        grid = self.prob_grid["grid"]
        levels = np.asarray(talent_levels).reshape(-1).astype(int)
        rates = np.asarray([self.config.interest_rates.get(level, 0.05) for level in levels[idx_rows]], dtype=float)
        lgd = np.full(len(idx_rows), float(getattr(self.config, "lgd_coefficient", 0.45)), dtype=float)
        c1 = float(getattr(self.config, "linear_cost", 0.0))
        c2 = float(getattr(self.config, "quadratic_cost", 0.0))

        pu_from = np.asarray(self.prob_grid["p_usage"][idx_rows, idx_grid_from], dtype=float)
        pd_from = self._effective_default_prob(np.asarray(self.prob_grid["p_default"][idx_rows, idx_grid_from], dtype=float))
        pu_to = np.asarray(self.prob_grid["p_usage"][idx_rows, idx_grid_to], dtype=float)
        pd_to = self._effective_default_prob(np.asarray(self.prob_grid["p_default"][idx_rows, idx_grid_to], dtype=float))

        profit_from = risk_adjusted_value(
            grid[idx_grid_from], pu_from, pd_from, rates, lgd, c1, c2
        )
        profit_to = risk_adjusted_value(
            grid[idx_grid_to], pu_to, pd_to, rates, lgd, c1, c2
        )
        return profit_to - profit_from

    def _repair_to_budget(self, L: np.ndarray, talent_levels: np.ndarray, budget: float) -> np.ndarray:
        """按 500 元网格做批量预算修补。"""
        grid = self.prob_grid["grid"]
        L = np.asarray(L, dtype=float).copy()
        _repair_input_mean = float(np.mean(L))
        _repair_input_sum = float(np.sum(L))
        print("[repair-check] before repair mean=%.2f, sum=%.2f, budget=%.2f" % (
            _repair_input_mean, _repair_input_sum, float(budget)
        ))
        idx = self._grid_indices_for_limits(L)
        n = len(idx)
        step = float(getattr(self.config, "limit_step", 500.0))
        L_min, L_max, _, _ = self._level_arrays(talent_levels)
        idx_min = self._grid_indices_for_limits(L_min)
        idx_max = self._grid_indices_for_limits(L_max)

        # 超预算：每批选择“降低一档利润损失最小”的客户。
        while float(grid[idx].sum()) > budget + 1e-9:
            excess = float(grid[idx].sum() - budget)
            need_steps = int(np.ceil(excess / step))
            can_cut = np.where(idx > idx_min)[0]
            if len(can_cut) == 0:
                break
            idx_to = idx[can_cut] - 1
            gains = self._one_step_delta(can_cut, idx[can_cut], idx_to, talent_levels)
            losses = -gains
            batch = min(need_steps, len(can_cut), int(getattr(self.config, "repair_batch_size", 20000)))
            choose = can_cut[np.argpartition(losses, batch - 1)[:batch]]
            idx[choose] -= 1

        # 未用满：每批选择“提高一档利润增量最大”的客户；若最大增益为负则停止。
        while float(grid[idx].sum()) + step <= budget + 1e-9:
            room = float(budget - grid[idx].sum())
            add_steps = int(np.floor(room / step))
            can_add = np.where(idx < idx_max)[0]
            if len(can_add) == 0:
                break
            idx_to = idx[can_add] + 1
            gains = self._one_step_delta(can_add, idx[can_add], idx_to, talent_levels)
            positive = gains > 1e-12
            if not positive.any():
                break
            can_add = can_add[positive]
            gains = gains[positive]
            batch = min(add_steps, len(can_add), int(getattr(self.config, "repair_batch_size", 20000)))
            choose = can_add[np.argpartition(-gains, batch - 1)[:batch]]
            idx[choose] += 1

        repaired = grid[idx]
        print("[repair-check] after repair mean=%.2f, sum=%.2f, budget=%.2f" % (
            float(np.mean(repaired)), float(np.sum(repaired)), float(budget)
        ))
        return repaired

    def _lagrangian_grid_optimize(self, talent_levels: np.ndarray) -> np.ndarray:
        """旧版单预算诊断方法；正式额度解不再调用此函数。"""
        grid = self.prob_grid["grid"]
        budget = float(self.config.total_budget)
        L_min, L_max, _, _ = self._level_arrays(talent_levels)
        min_idx = self._grid_indices_for_limits(L_min)
        max_idx = self._grid_indices_for_limits(L_max)
        min_total = float(grid[min_idx].sum())
        max_total = float(grid[max_idx].sum())
        print(
            "[lagrangian-check] budget=%.2f, min_total=%.2f, max_total=%.2f, "
            "grid_min=%.2f, grid_max=%.2f"
            % (budget, min_total, max_total, float(grid.min()), float(grid.max()))
        )

        if budget <= min_total:
            print("[大规模网格] 预算不足以覆盖最低额度，返回最低额度并修补。")
            return self._repair_to_budget(grid[min_idx], talent_levels, budget)
        # 预算是上限约束（sum L_i <= budget），允许不用完。
        independent_L = self._best_limits_for_lambda_chunked(talent_levels, 0.0)
        independent_total = float(independent_L.sum())
        if independent_total <= budget + 1e-9:
            self._last_best_lambda = 0.0
            self._last_repair_before_limits = np.asarray(independent_L, dtype=float).copy()
            self._last_repair_after_limits = np.asarray(independent_L, dtype=float).copy()
            print(
                "[lagrangian-check] 无预算独立最优已满足预算上限；"
                "lambda=0，允许预算剩余。allocated=%.2f, budget=%.2f"
                % (independent_total, budget)
            )
            return np.asarray(independent_L, dtype=float)

        # 用一个较宽的单位利润边界初始化 lambda bracket。
        _, _, lgd, rates = self._level_arrays(talent_levels)
        max_unit = float(np.max(rates) + np.max(lgd) + 1.0)
        # 上限约束的 KKT 乘子必须非负，不得用负lambda补贴亏损额度。
        lam_low = 0.0
        lam_high = max_unit

        for _ in range(50):
            if self._best_limits_for_lambda_chunked(talent_levels, lam_high).sum() <= budget:
                break
            lam_high *= 2.0
        print("[lagrangian-check] lambda bracket: low=%.12f, high=%.12f" % (lam_low, lam_high))

        best_L = None
        best_gap = np.inf
        best_lam = None
        n_iter = int(getattr(self.config, "lambda_iterations", 70))
        for it in range(n_iter):
            lam_mid = 0.5 * (lam_low + lam_high)
            L_mid = self._best_limits_for_lambda_chunked(talent_levels, lam_mid)
            total_mid = float(L_mid.sum())
            gap = abs(total_mid - budget)
            if gap < best_gap:
                best_gap = gap
                best_L = L_mid.copy()
                best_lam = float(lam_mid)
            if total_mid > budget:
                lam_low = lam_mid
            else:
                lam_high = lam_mid
            if (it + 1) % 10 == 0:
                print(f"[lambda] iter={it + 1}, total={total_mid:,.0f}, gap={gap:,.0f}")

        if best_L is None:
            best_L = self._best_limits_for_lambda_chunked(talent_levels, lam_high)
            best_lam = float(lam_high)

        print("[lagrangian-check] best_lambda=%.12f, best_gap=%.2f" % (float(best_lam), float(best_gap)))
        self._last_best_lambda = float(best_lam)
        self._last_repair_before_limits = np.asarray(best_L, dtype=float).copy()
        self._print_value_per_limit_diagnostics(talent_levels, lam=float(best_lam))
        self._print_limit_distribution_diagnostics(
            best_L, talent_levels, "budget_repair_before_candidate_limits_NOT_FINAL"
        )
        print("repair之前 best_limits.mean():")
        print(float(np.mean(best_L)))
        repaired = self._repair_to_budget(best_L, talent_levels, budget)
        self._last_repair_after_limits = np.asarray(repaired, dtype=float).copy()
        print("repair之后 best_limits.mean():")
        print(float(np.mean(repaired)))
        self._print_limit_distribution_diagnostics(
            repaired, talent_levels, "budget_repair_after_final_limits"
        )
        self._debug_sample_profit_paths(
            talent_levels, repaired, lam=float(best_lam), base_idx=100, n_random=4
        )
        print(
            f"[大规模网格] budget={budget:,.0f}, allocated={repaired.sum():,.0f}, "
            f"usage={repaired.sum() / budget:.2%}"
        )
        return repaired

    def _calculate_profit_vectorized(self, L_list, talent_levels):
        """用查表概率计算风险调整后的模型目标函数总值。"""
        L = np.asarray(L_list, dtype=float)
        p_usage, p_default = self._lookup_probabilities(L)
        levels = np.asarray(talent_levels).reshape(-1).astype(int)
        rates = np.asarray([self.config.interest_rates.get(level, 0.05) for level in levels], dtype=float)
        lgd = np.full(len(levels), float(getattr(self.config, "lgd_coefficient", 0.45)), dtype=float)
        values = risk_adjusted_value(
            L,
            p_usage,
            p_default,
            rates,
            lgd,
            float(getattr(self.config, "linear_cost", 0.0)),
            float(getattr(self.config, "quadratic_cost", 0.0)),
        )
        return float(values.sum())

    def _calculate_expected_loss(self, L_list, talent_levels):
        """按 lambda_LGD * L * p_default * utilization 计算预期损失。"""
        L = np.asarray(L_list, dtype=float)
        utilization, p_default = self._lookup_probabilities(L)
        lgd = float(getattr(self.config, "lgd_coefficient", 0.45))
        return float((lgd * p_default * utilization * L).sum())

    def _calculate_weighted_default_risk(self, L_list):
        """计算额度加权模型预测违约风险（不乘 LGD）。"""
        L = np.asarray(L_list, dtype=float)
        _, p_default = self._lookup_probabilities(L)
        return float((p_default * L).sum())

    def _resolve_customer_minimum_limits(self, base_limits, talent_levels):
        """返回逐客户硬下限：max(0, 原始授信额度 - 最大允许降幅)。"""
        base_limits = np.asarray(base_limits, dtype=float).reshape(-1)
        levels = np.asarray(talent_levels, dtype=int).reshape(-1)
        if len(base_limits) != len(levels):
            raise ValueError("原额度与人才等级长度必须一致")
        tier_min_limits = np.asarray(
            [self.config.min_limit.get(int(level), 0.0) for level in levels],
            dtype=float,
        )
        if not bool(getattr(self.config, "customer_min_limit_enabled", True)):
            return tier_min_limits
        return np.maximum(
            tier_min_limits,
            base_limits - float(getattr(
                self.config,
                "customer_min_limit_decrease",
                DEFAULT_CUSTOMER_MIN_LIMIT_DECREASE,
            )),
        )

    def calculate_optimal_limits(
        self,
        talent_levels,
        candidate_reference_quadratic_costs=None,
        candidate_index_table=None,
    ):
        """计算建议额度，并生成与原脚本兼容的结果 DataFrame。"""
        solver_backend = str(
            getattr(self.config, "solver_backend", "lagrangian")
        ).strip().lower()
        print("计算建议信用额度（优化后端: %s）..." % solver_backend)
        n_customers = len(self.data_info["df_aggregated"])
        if len(talent_levels) != n_customers:
            raise ValueError(f"talent_levels长度必须与样本数一致，期望: {n_customers}, 实际: {len(talent_levels)}")

        df_agg = self.data_info["df_aggregated"]
        _cred_col2 = "credamt" if "credamt" in df_agg.columns else "授信额度"
        base_limits_raw = pd.to_numeric(df_agg[_cred_col2], errors="coerce").fillna(0.0).to_numpy()
        levels = np.asarray(talent_levels, dtype=int)
        L_min, L_max, lgd, rates = self._level_arrays(levels)
        base_limits = base_limits_raw.astype(float).copy()
        customer_min_limits = self._resolve_customer_minimum_limits(
            base_limits, levels
        )
        self.customer_min_limits = customer_min_limits.copy()
        tier_minimum_limits = np.asarray(
            [self.config.min_limit.get(int(level), 0.0) for level in levels],
            dtype=float,
        )
        _tier_label = {8: "A", 7: "B", 6: "C", 5: "D", 4: "E", 3: "F1", 2: "F2", 1: "F3"}
        _minimum_policy_rows = []
        for _level in sorted(np.unique(levels).tolist()):
            _mask = levels == int(_level)
            _minimum_policy_rows.append({
                "talent_level": int(_level),
                "tier": _tier_label.get(int(_level), str(_level)),
                "customer_count": int(_mask.sum()),
                "tier_minimum_limit": float(self.config.min_limit.get(int(_level), 0.0)),
                "customer_min_limit_decrease": float(self.config.customer_min_limit_decrease),
                "resolved_customer_minimum_min": float(customer_min_limits[_mask].min()),
                "resolved_customer_minimum_max": float(customer_min_limits[_mask].max()),
            })
        print("实际生效的客户额度下限政策:")
        print(pd.DataFrame(_minimum_policy_rows).to_string(index=False))

        self.data_info["base_limits_raw"] = base_limits_raw.astype(float).copy()
        self.data_info["base_limits"] = base_limits.copy()
        self.base_total_limit = float(base_limits_raw.sum())
        configured_budget = getattr(self.config, "total_budget_override", None)
        self.config.total_budget = (
            self.base_total_limit if configured_budget is None else float(configured_budget)
        )
        self.base_profit = float(self._calculate_profit_vectorized(base_limits, talent_levels))
        self.base_expected_loss = float(self._calculate_expected_loss(base_limits, talent_levels))
        historical_weighted_risk = float(self._calculate_weighted_default_risk(base_limits))
        configured_risk_budget = getattr(self.config, "risk_budget", None)
        self.config.risk_budget = (
            float(getattr(self.config, "risk_tolerance", 1.05)) * historical_weighted_risk
            if configured_risk_budget is None else float(configured_risk_budget)
        )
        self.historical_weighted_default_risk = historical_weighted_risk

        common_solver_kwargs = dict(
            grid=self.prob_grid["grid"],
            p_usage=self.prob_grid["p_usage"],
            p_default=self.prob_grid["p_default"],
            levels=levels,
            min_limits=customer_min_limits,
            max_limits=self.config.max_limit,
            interest_rates=rates,
            lgd_coefficients=lgd,
            linear_cost=float(getattr(self.config, "linear_cost", 0.0)),
            quadratic_cost=float(getattr(self.config, "quadratic_cost", 0.0)),
            total_budget=float(self.config.total_budget),
            risk_budget=float(self.config.risk_budget),
            base_limits=base_limits,
            enforce_group_mean_monotonic=bool(
                getattr(self.config, "enforce_group_mean_monotonic", True)
            ),
            group_mean_min_ratio=dict(getattr(self.config, "group_mean_min_ratio", {})),
            max_variables=int(getattr(self.config, "milp_max_variables", 400000)),
            candidates_per_customer=int(
                getattr(self.config, "milp_candidates_per_customer", 16)
            ),
            candidate_reference_quadratic_costs=(
                None
                if candidate_reference_quadratic_costs is None
                else tuple(float(v) for v in candidate_reference_quadratic_costs)
            ),
            candidate_index_table=candidate_index_table,
        )
        if solver_backend == "heuristic":
            optimization_result = solve_discrete_portfolio_heuristic(
                max_rounds=int(getattr(self.config, "heuristic_max_rounds", 30)),
                **common_solver_kwargs
            )
        elif solver_backend == "lagrangian":
            optimization_result = solve_discrete_portfolio_lagrangian(
                max_rounds=int(getattr(self.config, "heuristic_max_rounds", 30)),
                lagrangian_iterations=int(
                    getattr(self.config, "lagrangian_iterations", 40)
                ),
                lagrangian_time_limit_seconds=float(
                    getattr(self.config, "lagrangian_time_limit_seconds", 60.0)
                ),
                lagrangian_step_size=float(
                    getattr(self.config, "lagrangian_step_size", 0.5)
                ),
                lagrangian_multiplier_cap=float(
                    getattr(self.config, "lagrangian_multiplier_cap", 1000.0)
                ),
                local_search_enabled=bool(
                    getattr(self.config, "local_search_enabled", True)
                ),
                local_search_max_passes=int(
                    getattr(self.config, "local_search_max_passes", 3)
                ),
                local_search_time_limit_seconds=float(
                    getattr(self.config, "local_search_time_limit_seconds", 30.0)
                ),
                local_search_pair_candidate_pool=int(
                    getattr(self.config, "local_search_pair_candidate_pool", 80)
                ),
                **common_solver_kwargs
            )
        elif solver_backend == "scipy_milp":
            optimization_result = solve_discrete_portfolio_milp(
                time_limit_seconds=float(
                    getattr(self.config, "milp_time_limit_seconds", 600.0)
                ),
                mip_relative_gap=float(
                    getattr(self.config, "milp_relative_gap", 0.01)
                ),
                **common_solver_kwargs
            )
        else:
            raise ValueError(
                "不支持的 SOLVER_BACKEND=%s；可选 lagrangian / "
                "heuristic / scipy_milp"
                % solver_backend
            )
        optimal_L = np.asarray(optimization_result.limits, dtype=float)
        self.optimization_result = optimization_result
        # 保留旧属性名，避免下游报告代码失效。
        self.milp_result = optimization_result
        p_usage_opt, p_default_opt = self._lookup_probabilities(optimal_L)
        self.models["y_proba_usage"] = p_usage_opt
        self.models["y_proba_default"] = p_default_opt

        self.credit_limits = optimal_L
        self.total_allocated = float(optimal_L.sum())
        self.total_profit = float(self._calculate_profit_vectorized(optimal_L, talent_levels))
        self.opt_expected_loss = float(self._calculate_expected_loss(optimal_L, talent_levels))
        self.opt_weighted_default_risk = float(self._calculate_weighted_default_risk(optimal_L))

        print(f"历史模型目标函数值: {self.base_profit:.2f}")
        print(f"优化模型目标函数值: {self.total_profit:.2f}")
        print(f"模型目标函数变化: {self.total_profit - self.base_profit:.2f}")
        print(f"总分配额度: {self.total_allocated:.2f}")
        print(f"预算使用比例: {self.total_allocated / self.config.total_budget:.2%}")
        _risk_usage = (
            self.opt_weighted_default_risk / self.config.risk_budget
            if self.config.risk_budget > 0 else np.nan
        )
        print(f"风险预算使用比例: {_risk_usage:.2%}")
        self._print_limit_distribution_diagnostics(
            optimal_L, talent_levels, "final_optimized_limits"
        )

        raw_u_orig, raw_d_orig = self._lookup_raw_probabilities(base_limits)
        cal_u_orig, cal_d_orig = self._lookup_probabilities(base_limits)
        raw_u_opt, raw_d_opt = self._lookup_raw_probabilities(optimal_L)
        cal_u_opt, cal_d_opt = self._lookup_probabilities(optimal_L)
        original_objective = risk_adjusted_value(
            base_limits, cal_u_orig, cal_d_orig, rates, lgd,
            float(getattr(self.config, "linear_cost", 0.0)),
            float(getattr(self.config, "quadratic_cost", 0.0)),
        )
        optimized_objective = risk_adjusted_value(
            optimal_L, cal_u_opt, cal_d_opt, rates, lgd,
            float(getattr(self.config, "linear_cost", 0.0)),
            float(getattr(self.config, "quadratic_cost", 0.0)),
        )
        df_results = pd.DataFrame({
            "customer_id": df_agg["cst_id"].astype(str).to_numpy(),
            "talent_level": levels,
            "original_credit_limit": base_limits_raw.astype(float),
            "credit_limit_continuous": optimal_L,
            "credit_limit": optimal_L,
            "original_usage_prob_raw": raw_u_orig,
            "original_usage_prob_calibrated": cal_u_orig,
            "original_default_prob_raw": raw_d_orig,
            "original_default_prob_calibrated": cal_d_orig,
            "optimized_usage_prob_raw": raw_u_opt,
            "optimized_usage_prob_calibrated": cal_u_opt,
            "optimized_default_prob_raw": raw_d_opt,
            "optimized_default_prob_calibrated": cal_d_opt,
            "original_objective_value": original_objective,
            "optimized_objective_value": optimized_objective,
            "objective_change": optimized_objective - original_objective,
            "limit_change": optimal_L - base_limits_raw,
            "original_weighted_default_risk": base_limits_raw * cal_d_orig,
            "optimized_weighted_default_risk": optimal_L * cal_d_opt,
            "interest_rate": rates,
            "objective_interest_rate": rates,
            "lgd_coefficient": lgd,
            "linear_cost": float(getattr(self.config, "linear_cost", 0.0)),
            "quadratic_cost": float(getattr(self.config, "quadratic_cost", 0.0)),
            "objective_parameter_source": "optimizer_runtime_config",
            "linear_cost_mode": str(getattr(self.config, "linear_cost_mode", "manual")),
            "tier_minimum_limit": tier_minimum_limits,
            "customer_minimum_limit": customer_min_limits,
            "customer_min_limit_decrease": float(getattr(
                self.config,
                "customer_min_limit_decrease",
                DEFAULT_CUSTOMER_MIN_LIMIT_DECREASE,
            )),
            "utilization_raw": pd.to_numeric(
                df_agg["utilization_raw"], errors="coerce"
            ).to_numpy(dtype=float),
            "utilization_used": pd.to_numeric(
                df_agg[str(getattr(self.config, "utilization_column", "utilization_used"))],
                errors="coerce",
            ).to_numpy(dtype=float),
            "utilization_method": str(self.data_info.get("utilization_method", "unknown")),
            "gross_interest_rate": float(getattr(self.config, "gross_interest_rate", 0.03)),
            "ftp_rate": float(getattr(self.config, "ftp_rate", 0.02)),
        })
        # Compatibility aliases used by reports/viz: calibrated probabilities at optimized limits.
        df_results["predicted_usage_prob"] = df_results["optimized_usage_prob_calibrated"]
        df_results["predicted_default_prob"] = df_results["optimized_default_prob_calibrated"]
        df_results["original_unit_profit"] = np.divide(
            original_objective,
            base_limits,
            out=np.zeros_like(original_objective),
            where=base_limits > 0,
        )
        df_results["unit_profit"] = np.divide(
            optimized_objective,
            optimal_L,
            out=np.zeros_like(optimized_objective),
            where=optimal_L > 0,
        )
        df_results["profit_change"] = df_results["objective_change"]
        self.data_info["credit_limit_results"] = df_results
        return df_results

    def select_quadratic_cost(
        self,
        talent_levels,
        candidates,
        output_dir,
        max_zero_rate=None,
        max_upper_hit_rate=None,
        objective_close_tolerance=0.01,
    ):
        """在开发客户上逐档求解c2，并按业务可接受性、目标值和小惩罚优先选择。"""
        candidates = sorted({float(v) for v in candidates})
        if not candidates or candidates[0] < 0:
            raise ValueError("c2候选集合必须为非空的非负数")
        os.makedirs(output_dir, exist_ok=True)
        rows = []
        levels = np.asarray(talent_levels, dtype=int)
        upper = np.asarray([self.config.max_limit.get(int(g), np.inf) for g in levels], dtype=float)
        df_agg = self.data_info["df_aggregated"]
        credit_column = "credamt" if "credamt" in df_agg.columns else "授信额度"
        base_limits = pd.to_numeric(
            df_agg[credit_column], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        customer_min_limits = self._resolve_customer_minimum_limits(
            base_limits, levels
        )
        _, _, lgd, rates = self._level_arrays(levels)
        common_candidate_table = build_portfolio_candidate_table(
            grid=self.prob_grid["grid"],
            p_usage=self.prob_grid["p_usage"],
            p_default=self.prob_grid["p_default"],
            levels=levels,
            min_limits=customer_min_limits,
            max_limits=self.config.max_limit,
            interest_rates=rates,
            lgd_coefficients=lgd,
            linear_cost=float(getattr(self.config, "linear_cost", 0.0)),
            quadratic_cost=float(candidates[0]),
            base_limits=base_limits,
            max_variables=int(getattr(self.config, "milp_max_variables", 400000)),
            candidates_per_customer=int(
                getattr(self.config, "milp_candidates_per_customer", 16)
            ),
            candidate_reference_quadratic_costs=candidates,
        )
        print(
            "c2 公共候选集已固定: variables=%d, reduced=%s, references=%d"
            % (
                len(common_candidate_table["grid_index"]),
                bool(common_candidate_table["candidate_reduced"]),
                len(candidates),
            )
        )
        for c2 in candidates:
            print("\n" + "=" * 80)
            print("c2敏感性分析: %.12g" % c2)
            print("=" * 80)
            self.config.quadratic_cost = float(c2)
            result_df = self.calculate_optimal_limits(
                levels,
                candidate_reference_quadratic_costs=candidates,
                candidate_index_table=common_candidate_table,
            )
            limits = result_df["credit_limit"].to_numpy(dtype=float)
            zero_rate = float(np.mean(np.isclose(limits, 0.0)))
            upper_hit_rate = float(np.mean(np.isclose(limits, upper)))
            acceptable = True
            reasons = []
            if max_zero_rate is not None and zero_rate > float(max_zero_rate):
                acceptable = False
                reasons.append("zero_rate")
            if max_upper_hit_rate is not None and upper_hit_rate > float(max_upper_hit_rate):
                acceptable = False
                reasons.append("upper_hit_rate")
            change = result_df["limit_change"].to_numpy(dtype=float)
            objective_total = float(result_df["optimized_objective_value"].sum())
            quadratic_component = float(c2 * np.square(limits).sum())
            rows.append({
                "c2": c2,
                "model_estimated_net_benefit": objective_total,
                "estimated_net_benefit_before_quadratic_cost": (
                    objective_total + quadratic_component
                ),
                "quadratic_cost_component": quadratic_component,
                "objective_change_vs_history": float(result_df["objective_change"].sum()),
                "zero_limit_rate": zero_rate,
                "upper_limit_hit_rate": upper_hit_rate,
                "weighted_default_risk": float(result_df["optimized_weighted_default_risk"].sum()),
                "risk_budget": float(self.config.risk_budget),
                "risk_budget_usage_rate": float(
                    result_df["optimized_weighted_default_risk"].sum() / self.config.risk_budget
                ) if self.config.risk_budget > 0 else np.nan,
                "total_limit": float(limits.sum()),
                "mean_limit_change": float(np.mean(change)),
                "median_limit_change": float(np.median(change)),
                "p05_limit_change": float(np.percentile(change, 5.0)),
                "p95_limit_change": float(np.percentile(change, 95.0)),
                "business_acceptable": acceptable,
                "rejection_reasons": ",".join(reasons),
                "optimizer_variable_count": int(
                    self.optimization_result.variable_count
                ),
                "candidate_reduced": bool(
                    self.optimization_result.candidate_reduced
                ),
                "common_candidate_reference_count": len(candidates),
            })

        summary = pd.DataFrame(rows).sort_values("c2").reset_index(drop=True)
        eligible = summary[summary["business_acceptable"]].copy()
        if eligible.empty:
            raise ValueError(
                "全部c2候选均超过预设业务阈值；请查看c2_sensitivity_summary.csv，"
                "调整候选或由业务重新确认阈值，不能自动绕过。"
            )
        best_objective = float(eligible["model_estimated_net_benefit"].max())
        tolerance = abs(best_objective) * float(objective_close_tolerance)
        close = eligible[
            eligible["model_estimated_net_benefit"] >= best_objective - tolerance - 1e-9
        ]
        selected_c2 = float(close["c2"].min())
        summary["selected"] = np.isclose(summary["c2"], selected_c2)
        summary["selection_rule"] = (
            "排除业务指标不可接受方案；剩余方案按模型预计净收益最高选择；"
            "在预设接近容差内选择较小c2"
        )
        summary.to_csv(
            os.path.join(output_dir, "c2_sensitivity_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        self.config.quadratic_cost = selected_c2
        print("\n✓ 已选择 c2=%.12g；在开发客户上重新生成最终参数选择解" % selected_c2)
        selected_df = self.calculate_optimal_limits(
            levels,
            candidate_reference_quadratic_costs=candidates,
            candidate_index_table=common_candidate_table,
        )
        self.c2_sensitivity_summary = summary
        self.selected_quadratic_cost = selected_c2
        return selected_c2, summary, selected_df

    def generate_report(self, df_results: pd.DataFrame):
        """输出模型目标、调整分布、预算、等级、概率和集中度离线评价。"""
        report_dir = str(getattr(self.config, "report_dir", "reports"))
        os.makedirs(report_dir, exist_ok=True)
        TIER_LABEL = {8: "A", 7: "B", 6: "C", 5: "D", 4: "E", 3: "F1", 2: "F2", 1: "F3"}
        work = df_results.copy()
        work["tier"] = work["talent_level"].map(lambda x: TIER_LABEL.get(int(x), str(x)))
        work["is_increase"] = work["limit_change"] > 1e-9
        work["is_decrease"] = work["limit_change"] < -1e-9
        work["is_unchanged"] = ~(work["is_increase"] | work["is_decrease"])
        work["usage_probability_change"] = (
            work["optimized_usage_prob_calibrated"] - work["original_usage_prob_calibrated"]
        )
        work["default_probability_change"] = (
            work["optimized_default_prob_calibrated"] - work["original_default_prob_calibrated"]
        )
        work.to_csv(
            os.path.join(report_dir, "credit_limit_large_grid_results.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        adjustment = distribution_summary(work["limit_change"], "limit_change")
        adjustment.to_csv(
            os.path.join(report_dir, "limit_adjustment_distribution.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        probability_change = pd.concat([
            distribution_summary(work["usage_probability_change"], "usage_probability_change"),
            distribution_summary(work["default_probability_change"], "default_probability_change"),
        ], ignore_index=True)
        probability_change.to_csv(
            os.path.join(report_dir, "probability_change_distribution.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        tier_summary = work.groupby(["talent_level", "tier"], as_index=False).agg(
            customer_count=("customer_id", "size"),
            historical_mean_limit=("original_credit_limit", "mean"),
            optimized_mean_limit=("credit_limit", "mean"),
            optimized_median_limit=("credit_limit", "median"),
            increase_rate=("is_increase", "mean"),
            decrease_rate=("is_decrease", "mean"),
            unchanged_rate=("is_unchanged", "mean"),
        ).sort_values("talent_level")
        ratio_policy = dict(getattr(self.config, "group_mean_min_ratio", {}))
        optimized_means = dict(zip(tier_summary["talent_level"], tier_summary["optimized_mean_limit"]))
        historical_means = dict(zip(tier_summary["talent_level"], tier_summary["historical_mean_limit"]))
        required_ratios = []
        optimized_checks = []
        historical_checks = []
        for level in tier_summary["talent_level"].astype(int):
            ratio = float(ratio_policy.get(level, np.nan))
            has_pair = level - 1 in optimized_means and np.isfinite(ratio)
            required_ratios.append(ratio if has_pair else np.nan)
            optimized_checks.append(
                True if not has_pair
                else optimized_means[level] + 1e-6 >= ratio * optimized_means[level - 1]
            )
            historical_checks.append(
                True if not has_pair
                else historical_means[level] + 1e-6 >= ratio * historical_means[level - 1]
            )
        tier_summary["required_ratio_to_previous_tier"] = required_ratios
        tier_summary["mean_constraint_satisfied"] = optimized_checks
        tier_summary["historical_mean_constraint_satisfied"] = historical_checks
        tier_summary.to_csv(
            os.path.join(report_dir, "talent_level_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        levels = work["talent_level"].to_numpy(dtype=int)
        lower = work["customer_minimum_limit"].to_numpy(dtype=float)
        upper = np.asarray([self.config.max_limit.get(int(g), np.inf) for g in levels])
        opt = work["credit_limit"].to_numpy(dtype=float)
        boundary = pd.DataFrame([{
            "lower_bound_rate": float(np.mean(np.isclose(opt, lower))),
            "upper_bound_rate": float(np.mean(np.isclose(opt, upper))),
            "historical_outside_tier_interval_rate": float(np.mean(
                (work["original_credit_limit"].to_numpy(dtype=float) < lower - 1e-9)
                | (work["original_credit_limit"].to_numpy(dtype=float) > upper + 1e-9)
            )),
        }])
        boundary.to_csv(
            os.path.join(report_dir, "boundary_check.csv"), index=False, encoding="utf-8-sig"
        )
        minimum_policy = work.groupby(
            ["talent_level", "tier"], as_index=False
        ).agg(
            customer_count=("customer_id", "size"),
            tier_minimum_limit=("tier_minimum_limit", "first"),
            customer_min_limit_decrease=("customer_min_limit_decrease", "first"),
            resolved_customer_minimum_min=("customer_minimum_limit", "min"),
            resolved_customer_minimum_max=("customer_minimum_limit", "max"),
        ).sort_values("talent_level")
        minimum_policy.to_csv(
            os.path.join(report_dir, "minimum_limit_policy_by_tier.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        concentration_shares(opt).to_csv(
            os.path.join(report_dir, "limit_concentration.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        hist_obj = float(work["original_objective_value"].sum())
        opt_obj = float(work["optimized_objective_value"].sum())
        total_budget = float(self.config.total_budget)
        risk_budget = float(self.config.risk_budget)
        summary = pd.DataFrame([{
            "optimization_scope": str(getattr(self.config, "optimization_scope", "all")),
            "customer_count": len(work),
            "historical_objective": hist_obj,
            "optimized_objective": opt_obj,
            "objective_change": opt_obj - hist_obj,
            "historical_total_limit": float(work["original_credit_limit"].sum()),
            "optimized_total_limit": float(work["credit_limit"].sum()),
            "total_budget": total_budget,
            "total_budget_usage_rate": float(work["credit_limit"].sum() / total_budget) if total_budget > 0 else np.nan,
            "historical_weighted_default_risk": float(work["original_weighted_default_risk"].sum()),
            "optimized_weighted_default_risk": float(work["optimized_weighted_default_risk"].sum()),
            "risk_budget": risk_budget,
            "risk_budget_usage_rate": float(work["optimized_weighted_default_risk"].sum() / risk_budget) if risk_budget > 0 else np.nan,
            "increase_rate": float(work["is_increase"].mean()),
            "decrease_rate": float(work["is_decrease"].mean()),
            "unchanged_rate": float(work["is_unchanged"].mean()),
            "group_mean_constraint_satisfied": bool(tier_summary["mean_constraint_satisfied"].all()),
            "historical_group_mean_constraint_satisfied": bool(
                tier_summary["historical_mean_constraint_satisfied"].all()
            ),
        }])
        summary.to_csv(
            os.path.join(report_dir, "portfolio_summary.csv"), index=False, encoding="utf-8-sig"
        )

        result = self.milp_result
        pd.DataFrame([{
            "solver": result.solver_backend,
            "status": result.solver_status,
            "success": result.solver_success,
            "message": result.solver_message,
            "mip_gap": result.mip_gap,
            "global_optimality_proven": bool(
                result.solver_backend == "scipy_milp"
                and np.isfinite(result.mip_gap)
                and result.mip_gap <= 1e-9
            ),
            "constraint_feasible": result.constraint_feasible,
            "max_normalized_constraint_violation": result.max_constraint_violation,
            "iterations_or_nodes": result.iterations,
            "dual_upper_bound_reduced_candidates": result.dual_upper_bound,
            "heuristic_dual_bound_gap_reduced_candidates": result.heuristic_dual_bound_gap,
            "feasible_start_count": result.feasible_start_count,
            "local_search_passes": result.local_search_passes,
            "local_search_single_moves": result.local_search_single_moves,
            "local_search_pair_exchanges": result.local_search_pair_exchanges,
            "local_search_objective_gain": result.local_search_objective_gain,
            "selected_variable_count": result.variable_count,
            "full_feasible_candidate_count": result.feasible_candidate_count,
            "candidate_reduced": result.candidate_reduced,
        }]).to_csv(
            os.path.join(report_dir, "solver_summary.csv"), index=False, encoding="utf-8-sig"
        )

        with open(os.path.join(report_dir, "offline_evaluation_notes.txt"), "w", encoding="utf-8") as f:
            f.write("本报告为模型内离线评价。历史数据没有建议额度下的反事实结果，\n")
            f.write("因此 objective_change 仅表示当前概率模型、目标函数与约束下的模型目标变化，\n")
            f.write("不能直接解释为实际利润提升。若历史额度违反新增人才政策约束，历史方案不属于同一可行域。\n")
            if result.solver_backend in ("heuristic", "lagrangian"):
                f.write("本次使用 Python 3.6 近似算法；结果已通过硬约束审计，但不保证全局最优，\n")
                f.write("也不存在可报告的 MIP gap。不得将该结果表述为整数规划最优解。\n")
            if result.solver_backend == "lagrangian":
                f.write("heuristic_dual_bound_gap 仅针对压缩后的候选集合，不是完整网格最优性差距。\n")
                f.write("拉格朗日双起点后运行限时双向局部搜索，允许单客户回退及两客户交换。\n")
        print("离线评价报告已保存到:", report_dir)
        return tier_summary


def _viz_all(*args, **kwargs):
    """可视化已拆分到 viz_credit_limit.py，请在 notebook 中单独运行该文件。"""
    print("  [可视化] 请运行 viz_credit_limit.py 生成图表（已从本文件中独立出去）。")




def _scorecard_all(df_results, calculator, tl, tier_labels,
                   orig_L, opt_L, p_u, p_d, out_dir, DPI, TIER_LABEL):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import warnings; warnings.filterwarnings("ignore")
    import lightgbm as lgb, shap
    from sklearn.preprocessing import LabelEncoder
    import sys; sys.path.insert(0, os.getcwd())

    MODEL_DIR = "saved_models"
    if not (os.path.exists(os.path.join(MODEL_DIR,"booster_usage.txt")) and
            os.path.exists(os.path.join(MODEL_DIR,"booster_default.txt"))):
        print("  [跳过评分卡] 未找到 saved_models/，请先运行 shap_analysis.py 保存模型")
        return

    print("  [评分卡] 加载模型 & 计算 SHAP...")
    LEAK = {"cst_id","loanacctno","ac_curr_bal","time_of_dq","y_freq","y_dq_risk",
            "latent_group",
            "credamt_sq","credamt_cube","credamt_log",
            "授信额度_sq","授信额度_cube","授信额度_log"}

    # 从模型文件里读出训练时的特征列名，确保完全对齐
    import lightgbm as lgb
    bu = lgb.Booster(model_file=os.path.join(MODEL_DIR,"booster_usage.txt"))
    bd = lgb.Booster(model_file=os.path.join(MODEL_DIR,"booster_default.txt"))
    model_feat_names = bu.feature_name()   # 训练时的129个特征名
    print(f"  模型特征数: {len(model_feat_names)}")

    # 用完整流程重建 work（clean + build_labels），保证特征工程一致
    df_agg_raw = calculator.data_info["df_aggregated"].copy()
    work = drop_post_label_cols(df_agg_raw, target="y_dq_risk")

    def _fe(df):
        drop = LEAK - {"y_freq"}
        X = df.drop(columns=[c for c in drop if c in df.columns]).copy()
        X = X.drop(columns=["y_freq","y_dq_risk"], errors="ignore")
        dt = list(X.select_dtypes(include=["datetime64[ns]","datetime64[ns, UTC]"]).columns)
        if dt: X = X.drop(columns=dt)
        if "档位" in X.columns:
            to = {"D":5,"E":4,"F1":3,"F2":2,"F3":1}
            X["档位"] = X["档位"].astype(str).map(to).fillna(0).astype(int)
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder(); X[col] = le.fit_transform(X[col].astype(str))
        X = X.select_dtypes(include=[np.number]).copy()
        # 严格对齐到模型训练时的特征列，缺失列补0
        for c in model_feat_names:
            if c not in X.columns:
                X[c] = 0.0
        return X[model_feat_names].copy()

    X_feat = _fe(work)
    feat_names = model_feat_names
    print(f"  特征矩阵 shape: {X_feat.shape}")
    def _shap(booster, X):
        try:
            sv = shap.TreeExplainer(booster).shap_values(X)
            if isinstance(sv, list): sv = sv[1]
            return np.array(sv)
        except Exception:
            c = booster.predict(X, pred_contrib=True)
            return c[:, :-1].astype(np.float32)

    sv_u = _shap(bu, X_feat); sv_d = _shap(bd, X_feat)
    q_u = np.percentile(p_u, [10, 25, 50, 75, 90])
    q_d = np.percentile(p_d, [10, 25, 50, 75, 90])
    qlbls = ["P10以下","P10-P25","P25-P50","P50-P75","P75-P90","P90以上"]
    def _ql(prob, q):
        for i, t in enumerate(list(q)+[1.1]):
            if prob <= t: return qlbls[i]
        return qlbls[-1]
    def _top5(sv, idx):
        v = sv[idx]; top = np.argsort(np.abs(v))[::-1][:5]
        return [(feat_names[i], float(v[i])) for i in top]

    delta = opt_L - orig_L
    hi_idx = int(np.argmax(p_d))
    gm = (p_d < np.median(p_d)) & (p_u > np.percentile(p_u, 75)) & (delta > 0)
    up_idx = int(np.where(gm)[0][np.argmax(delta[gm])]) if gm.any() else int(np.argmax(delta))
    dn_idx = int(np.argmin(delta))

    for lbl, idx in [("高风险",hi_idx),("额度提升",up_idx),("额度下降",dn_idx)]:
        tier = TIER_LABEL.get(int(tl[idx]), str(tl[idx]))
        pu_v = float(p_u[idx]); pd_v = float(p_d[idx])
        ov = float(orig_L[idx]); nv = float(opt_L[idx]); dv = nv - ov
        dc = "#1a7a3c" if dv >= 0 else "#c0392b"; ds = "+" if dv >= 0 else ""
        cid = str(df_results.iloc[idx].get("customer_id", f"#{idx}"))
        t5u = _top5(sv_u, idx); t5d = _top5(sv_d, idx)
        mu = max(abs(s) for _,s in t5u) or 1.0
        md = max(abs(s) for _,s in t5d) or 1.0

        # ── 新布局：左列=支用概率+SHAP，右列=违约概率+SHAP，底栏=额度 ──
        fig = plt.figure(figsize=(16, 10), dpi=DPI)
        fig.patch.set_facecolor("#f5f7fa")

        # 标题栏
        ax_t = fig.add_axes([0.0, 0.93, 1.0, 0.07])
        ax_t.set_facecolor("#1a3a5c"); ax_t.axis("off")
        ax_t.text(0.5, 0.5, f"客户评分卡  [{lbl}]  ·  客编：{cid}  ·  人才等级：{tier}",
                  ha="center", va="center", fontsize=13, color="white", fontweight="bold")

        # ── 左列：支用概率 ──────────────────────────────────────────
        def _draw_prob_shap(fig, x0, prob_val, prob_lbl, q_arr, color,
                            shap_title, ldir, top5_list, max_abs):
            """画单列：概率数值+分位+进度条+SHAP Top5"""
            # 概率区（上部25%高度）
            ax_prob = fig.add_axes([x0, 0.72, 0.46, 0.19])
            ax_prob.set_facecolor("white"); ax_prob.axis("off")
            ax_prob.text(0.5, 0.95, prob_lbl, ha="center", va="top",
                         fontsize=11, fontweight="bold", color="#1a3a5c")
            ax_prob.text(0.5, 0.60, f"{prob_val:.4f}", ha="center", va="top",
                         fontsize=26, color=color, fontweight="bold")
            ax_prob.text(0.5, 0.08, _ql(prob_val, q_arr),
                         ha="center", va="bottom", fontsize=9, color="#888")
            # 进度条
            bax = fig.add_axes([x0 + 0.04, 0.735, 0.38, 0.018])
            bax.set_xlim(0, 1); bax.set_ylim(0, 1); bax.axis("off")
            bax.barh(0, 1, color="#e8e8e8", height=1)
            bax.barh(0, min(prob_val, 1.0), color=color, height=1, alpha=0.8)

            # SHAP区（下部60%高度）
            ax_s = fig.add_axes([x0, 0.10, 0.46, 0.60])
            ax_s.set_facecolor("white"); ax_s.axis("off")
            ax_s.text(0.5, 0.97, shap_title, ha="center", va="top",
                      fontsize=10, fontweight="bold", color="#1a3a5c")
            ax_s.text(0.02, 0.88, "特征名",  va="top", fontsize=8, color="#888")
            ax_s.text(0.55, 0.88, "SHAP值",  va="top", fontsize=8, color="#888", ha="center")
            ax_s.text(0.97, 0.88, "方向",    va="top", fontsize=8, color="#888", ha="right")
            ax_s.axhline(0.85, color="#ddd", linewidth=0.8, xmin=0.01, xmax=0.99)

            for rank, (fn, sv_val) in enumerate(top5_list):
                ry   = 0.76 - rank * 0.155
                col  = "#c0392b" if sv_val > 0 else "#2166ac"
                sign = "+" if sv_val > 0 else ""
                direc = ldir[0] if sv_val > 0 else ldir[1]
                bw   = abs(sv_val) / max_abs * 0.35
                fns  = fn if len(fn) <= 18 else fn[:16] + "…"
                ax_s.text(0.02, ry, f"{rank+1}. {fns}", va="center", fontsize=8.5, color="#333")
                ax_s.add_patch(mpatches.FancyBboxPatch(
                    (0.53, ry - 0.045), bw, 0.09,
                    boxstyle="round,pad=0.005", facecolor=col, alpha=0.7,
                    transform=ax_s.transAxes))
                ax_s.text(0.53 + bw + 0.01, ry, f"{sign}{sv_val:.4f}",
                          va="center", fontsize=8, color=col, fontweight="bold")
                ax_s.text(0.97, ry, direc, va="center", ha="right", fontsize=8, color=col)
                if rank < 4:
                    ax_s.axhline(ry - 0.075, color="#f0f0f0",
                                 linewidth=0.6, xmin=0.01, xmax=0.99)

        # 左列：支用概率
        _draw_prob_shap(fig,
            x0=0.02,
            prob_val=pu_v, prob_lbl="支用概率", q_arr=q_u, color="#1a7a3c",
            shap_title="支用概率 Top5 特征（SHAP）",
            ldir=("↑支用", "↓支用"), top5_list=t5u, max_abs=mu)

        # 中间分隔线
        ax_div = fig.add_axes([0.499, 0.10, 0.002, 0.82])
        ax_div.set_facecolor("#e0e0e0"); ax_div.axis("off")

        # 右列：违约概率
        _draw_prob_shap(fig,
            x0=0.52,
            prob_val=pd_v, prob_lbl="违约概率", q_arr=q_d, color="#c0392b",
            shap_title="违约概率 Top5 特征（SHAP）",
            ldir=("↑违约", "↓违约"), top5_list=t5d, max_abs=md)

        # 底栏：额度优化结果
        ax_bot = fig.add_axes([0.02, 0.01, 0.96, 0.08])
        ax_bot.set_facecolor("#eaf0fb"); ax_bot.axis("off")
        ax_bot.text(0.18, 0.65, "优化前额度",  ha="center", va="center", fontsize=9,  color="#555")
        ax_bot.text(0.18, 0.25, f"¥{ov:,.0f}", ha="center", va="center",
                    fontsize=13, color="#333", fontweight="bold")
        ax_bot.text(0.38, 0.45, "→", ha="center", va="center", fontsize=18, color="#aaa")
        ax_bot.text(0.58, 0.65, "优化后额度",  ha="center", va="center", fontsize=9,  color="#555")
        ax_bot.text(0.58, 0.25, f"¥{nv:,.0f}", ha="center", va="center",
                    fontsize=13, color=dc, fontweight="bold")
        ax_bot.text(0.80, 0.65, "变动金额",    ha="center", va="center", fontsize=9,  color="#555")
        ax_bot.text(0.80, 0.25, f"{ds}¥{abs(dv):,.0f}", ha="center", va="center",
                    fontsize=13, color=dc, fontweight="bold")
        ax_bot.text(0.5, -0.15, "注：SHAP正值推高概率，负值拉低；分位数基于全样本分布。",
                    ha="center", va="top", fontsize=7.5, color="#aaa")

        out = os.path.join(out_dir, f"scorecard_{lbl}.png")
        plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(); print(f"    已保存: scorecard_{lbl}.png")


def main():
    """运行大规模预计算概率网格版本。
    
    所有参数优先从 notebook 传入的 _OVERRIDE 全局变量读取；
    仅当 notebook 未设置时才使用括号内的默认值。
    直接 python 运行时默认值同样生效。
    """
    print("=" * 80)
    print("建议额度计算：校准概率网格 + 可切换组合优化后端")
    print("=" * 80)

    # ── 从 notebook Cell 1 读取覆盖参数（%run 时全局变量已注入）──────────
    _g = globals()
    _excel_path    = _g.get("EXCEL_PATH_OPT_OVERRIDE",   _g.get("DATA_FILE", "kechuang_merged0722.csv"))
    _prob_grid_dir = _g.get("PROB_GRID_DIR_OPT_OVERRIDE", _g.get("PROB_GRID_DIR", "probability_grid_large"))
    _reports_dir   = _g.get("REPORTS_DIR_OPT_OVERRIDE",   _g.get("REPORTS_DIR",  "reports"))
    _interest_rate = float(_g.get("INTEREST_RATE_OPT_OVERRIDE", _g.get("INTEREST_RATE", 0.03)))
    _ftp_rate       = float(_g.get("FTP_RATE_OPT_OVERRIDE", _g.get("FTP_RATE", 0.02)))
    _lgd            = float(_g.get("LGD_COEFFICIENT_OPT_OVERRIDE", _g.get("LGD_COEFFICIENT", 0.45)))
    _linear_cost    = float(_g.get("LINEAR_COST_OPT_OVERRIDE", _g.get("LINEAR_COST", 0.005)))
    _linear_cost_mode = str(_g.get(
        "LINEAR_COST_MODE_OPT_OVERRIDE", _g.get("LINEAR_COST_MODE", "manual")
    )).strip().lower()
    _quadratic_cost = float(_g.get("QUADRATIC_COST_OPT_OVERRIDE", _g.get("QUADRATIC_COST", 1e-9)))
    _scope          = str(_g.get("OPTIMIZATION_SCOPE_OVERRIDE", "all"))
    _total_budget_override = _g.get("TOTAL_BUDGET_OPT_OVERRIDE", None)
    _risk_budget_override = _g.get("RISK_BUDGET_OPT_OVERRIDE", None)
    _risk_tolerance = float(_g.get("RISK_TOLERANCE_OPT_OVERRIDE", 1.05))
    _group_mean_min_ratio = dict(_g.get(
        "GROUP_MEAN_MIN_RATIO_OPT_OVERRIDE", {2: 1.0, 3: 1.0, 4: 0.8, 5: 0.8}
    ))
    _customer_min_limit_enabled = bool(_g.get(
        "CUSTOMER_MIN_LIMIT_ENABLED_OPT_OVERRIDE",
        _g.get("CUSTOMER_MIN_LIMIT_ENABLED", True),
    ))
    _customer_min_limit_decrease = float(_g.get(
        "CUSTOMER_MIN_LIMIT_DECREASE_OPT_OVERRIDE",
        _g.get("CUSTOMER_MIN_LIMIT_DECREASE", DEFAULT_CUSTOMER_MIN_LIMIT_DECREASE),
    ))
    _c2_selection_mode = bool(_g.get("C2_SELECTION_MODE_OVERRIDE", False))
    _c2_candidates = tuple(_g.get("C2_CANDIDATES_OPT_OVERRIDE", (_quadratic_cost,)))
    _c2_max_zero_rate = _g.get("C2_MAX_ZERO_RATE_OPT_OVERRIDE", None)
    _c2_max_upper_hit_rate = _g.get("C2_MAX_UPPER_HIT_RATE_OPT_OVERRIDE", None)
    _c2_objective_close_tolerance = float(
        _g.get("C2_OBJECTIVE_CLOSE_TOLERANCE_OPT_OVERRIDE", 0.01)
    )
    _grid_min      = float(_g.get("GRID_MIN_OPT_OVERRIDE",  _g.get("GRID_MIN",  1000.0)))
    _grid_max      = float(_g.get("GRID_MAX_OPT_OVERRIDE",  _g.get("GRID_MAX",  1000000.0)))
    _grid_step     = float(_g.get("GRID_STEP_OPT_OVERRIDE", _g.get("GRID_STEP", 500.0)))
    _cleaned_file  = _g.get("CLEANED_FILE_OPT_OVERRIDE",  _g.get("CLEANED_FILE", "data_cleaned.csv"))
    _utilization_lookup_file = _g.get(
        "UTILIZATION_LOOKUP_FILE_OPT_OVERRIDE",
        _g.get("UTILIZATION_LOOKUP_FILE", "utilization_customer_lookup.csv"),
    )
    _tier_min       = _g.get(
        "TIER_MIN_LIMITS_OPT_OVERRIDE",
        _g.get("TIER_MIN_LIMITS", DEFAULT_TIER_MIN_LIMITS),
    )
    _tier_max       = _g.get("TIER_MAX_LIMITS_OPT_OVERRIDE", _g.get("TIER_MAX_LIMITS", {}))
    _reuse_optimization = bool(_g.get("REUSE_OPTIMIZATION_OVERRIDE", _g.get("REUSE_OPTIMIZATION", False)))
    _sample_enabled = bool(_g.get("OPTIMIZATION_SAMPLE_ENABLED_OPT_OVERRIDE", False))
    _sample_size = _g.get("OPTIMIZATION_SAMPLE_SIZE_OPT_OVERRIDE", None)
    _sample_random_state = int(_g.get("OPTIMIZATION_SAMPLE_RANDOM_STATE_OPT_OVERRIDE", 42))
    _solver_backend = str(_g.get(
        "SOLVER_BACKEND_OPT_OVERRIDE", _g.get("SOLVER_BACKEND", "lagrangian")
    )).strip().lower()
    _heuristic_max_rounds = int(_g.get(
        "HEURISTIC_MAX_ROUNDS_OPT_OVERRIDE", _g.get("HEURISTIC_MAX_ROUNDS", 30)
    ))
    _lagrangian_iterations = int(_g.get(
        "LAGRANGIAN_ITERATIONS_OPT_OVERRIDE", _g.get("LAGRANGIAN_ITERATIONS", 40)
    ))
    _lagrangian_time_limit_seconds = float(_g.get(
        "LAGRANGIAN_TIME_LIMIT_SECONDS_OPT_OVERRIDE",
        _g.get("LAGRANGIAN_TIME_LIMIT_SECONDS", 60.0),
    ))
    _lagrangian_step_size = float(_g.get(
        "LAGRANGIAN_STEP_SIZE_OPT_OVERRIDE", _g.get("LAGRANGIAN_STEP_SIZE", 0.5)
    ))
    _lagrangian_multiplier_cap = float(_g.get(
        "LAGRANGIAN_MULTIPLIER_CAP_OPT_OVERRIDE",
        _g.get("LAGRANGIAN_MULTIPLIER_CAP", 1000.0),
    ))
    _local_search_enabled = bool(_g.get(
        "LOCAL_SEARCH_ENABLED_OPT_OVERRIDE", _g.get("LOCAL_SEARCH_ENABLED", True)
    ))
    _local_search_max_passes = int(_g.get(
        "LOCAL_SEARCH_MAX_PASSES_OPT_OVERRIDE",
        _g.get("LOCAL_SEARCH_MAX_PASSES", 3),
    ))
    _local_search_time_limit_seconds = float(_g.get(
        "LOCAL_SEARCH_TIME_LIMIT_SECONDS_OPT_OVERRIDE",
        _g.get("LOCAL_SEARCH_TIME_LIMIT_SECONDS", 30.0),
    ))
    _local_search_pair_candidate_pool = int(_g.get(
        "LOCAL_SEARCH_PAIR_CANDIDATE_POOL_OPT_OVERRIDE",
        _g.get("LOCAL_SEARCH_PAIR_CANDIDATE_POOL", 80),
    ))

    print(f"  数据文件        : {_excel_path}")
    print(f"  清洗数据文件    : {_cleaned_file}")
    print(f"  概率网格目录    : {_prob_grid_dir}")
    print(f"  结果目录        : {_reports_dir}")
    print(f"  interest_rate   : {_interest_rate}")
    print(f"  FTP rate (configured): {_ftp_rate}")
    print(f"  lambda_LGD      : {_lgd}")
    print(f"  linear cost     : {_linear_cost}")
    print(f"  linear cost mode: {_linear_cost_mode}")
    print(f"  quadratic cost  : {_quadratic_cost}")
    print(f"  optimization scope: {_scope}")
    print(f"  risk tolerance   : {_risk_tolerance}")
    print(f"  group mean ratios: {_group_mean_min_ratio}")
    _customer_lower_bound_label = (
        f"max(0, original limit - {_customer_min_limit_decrease:,.0f}, tier minimum)"
        if _customer_min_limit_enabled else "tier lower bound"
    )
    print(f"  customer lower bound: {_customer_lower_bound_label}")
    print("  resolved tier minimums:", {
        int(level): float(_tier_min.get(
            level,
            _tier_min.get(str(level), DEFAULT_TIER_MIN_LIMITS.get(level, 0.0)),
        ))
        for level in range(1, 9)
    })
    print(f"  solver backend    : {_solver_backend}")
    if _solver_backend == "lagrangian":
        print(
            "  lagrangian limits : iterations=%d, time_limit=%.1fs"
            % (_lagrangian_iterations, _lagrangian_time_limit_seconds)
        )
        print(
            "  local search      : enabled=%s, passes=%d, time_limit=%.1fs, pair_pool=%d"
            % (
                _local_search_enabled,
                _local_search_max_passes,
                _local_search_time_limit_seconds,
                _local_search_pair_candidate_pool,
            )
        )
    print(f"  额度区间        : [{_grid_min:,.0f}, {_grid_max:,.0f}]  步长={_grid_step:.0f}")
    print(
        "  优化抽样        : %s"
        % (
            "开启，样本数=%s，random_state=%d"
            % (_sample_size, _sample_random_state)
            if _sample_enabled else "关闭"
        )
    )

    config = LargePrecomputedGridConfig()
    config.limit_step       = _grid_step
    config.chunk_size       = 2000
    config.lambda_iterations = 70
    config.prob_grid_path   = _prob_grid_dir
    config.excel_path       = _excel_path
    config.cleaned_file     = _cleaned_file
    config.utilization_lookup_file = _utilization_lookup_file
    config.utilization_column = "utilization_used"
    config.gross_interest_rate = _interest_rate
    config.ftp_rate = _ftp_rate
    config.linear_cost_mode = _linear_cost_mode
    config.csv_encoding     = _g.get("CSV_ENCODING_OPT_OVERRIDE", _g.get("CSV_ENCODING", "utf-8-sig"))
    config.snapshot_date    = _g.get("SNAPSHOT_DATE_OPT_OVERRIDE", _g.get("SNAPSHOT_DATE", "2026-01-31"))
    config.apply_maturity_filter = bool(_g.get("APPLY_MATURITY_FILTER_OPT_OVERRIDE", _g.get("APPLY_MATURITY_FILTER", False)))
    config.maturity_cutoff  = _g.get("MATURITY_CUTOFF_OPT_OVERRIDE", _g.get("MATURITY_CUTOFF", "2026-07-21"))
    config.apply_eff_date_filter = bool(_g.get("APPLY_EFF_DATE_FILTER_OPT_OVERRIDE", _g.get("APPLY_EFF_DATE_FILTER", True)))
    config.dedup_cst_loan = bool(_g.get("DEDUP_CST_LOAN_OPT_OVERRIDE", _g.get("DEDUP_CST_LOAN", False)))
    config.eff_date_lower   = _g.get("EFF_DATE_LOWER_OPT_OVERRIDE", _g.get("EFF_DATE_LOWER", "2025-01-01"))
    config.eff_date_upper   = _g.get("EFF_DATE_UPPER_OPT_OVERRIDE", _g.get("EFF_DATE_UPPER", "2026-03-31"))
    config.y_freq_mode      = _g.get("Y_FREQ_MODE_OPT_OVERRIDE", _g.get("Y_FREQ_MODE", "bout_gt0_and_curr_p80"))
    config.lgd_coefficient  = _lgd
    if _linear_cost_mode == "ftp_utilization":
        # r*L*u - FTP*L*u 等价于 (r-FTP)*L*u。
        _objective_interest_rate = _interest_rate - _ftp_rate
        config.linear_cost = 0.0
    elif _linear_cost_mode == "manual":
        _objective_interest_rate = _interest_rate
        config.linear_cost = _linear_cost
    else:
        raise ValueError("LINEAR_COST_MODE 只能为 'manual' 或 'ftp_utilization'")
    config.quadratic_cost   = _quadratic_cost
    config.total_budget_override = None if _total_budget_override is None else float(_total_budget_override)
    config.risk_budget      = None if _risk_budget_override is None else float(_risk_budget_override)
    config.risk_tolerance   = _risk_tolerance
    config.group_mean_min_ratio = {int(k): float(v) for k, v in _group_mean_min_ratio.items()}
    config.customer_min_limit_enabled = _customer_min_limit_enabled
    config.customer_min_limit_decrease = _customer_min_limit_decrease
    config.optimization_scope = _scope
    config.optimization_sample_enabled = _sample_enabled
    config.optimization_sample_size = None if _sample_size is None else int(_sample_size)
    config.optimization_sample_random_state = _sample_random_state
    config.report_dir       = _reports_dir
    config.enforce_group_mean_monotonic = bool(
        _g.get("ENFORCE_GROUP_MEAN_MONOTONIC_OPT_OVERRIDE", True)
    )
    config.solver_backend = _solver_backend
    config.heuristic_max_rounds = _heuristic_max_rounds
    config.lagrangian_iterations = _lagrangian_iterations
    config.lagrangian_time_limit_seconds = _lagrangian_time_limit_seconds
    config.lagrangian_step_size = _lagrangian_step_size
    config.lagrangian_multiplier_cap = _lagrangian_multiplier_cap
    config.local_search_enabled = _local_search_enabled
    config.local_search_max_passes = _local_search_max_passes
    config.local_search_time_limit_seconds = _local_search_time_limit_seconds
    config.local_search_pair_candidate_pool = _local_search_pair_candidate_pool
    config.milp_max_variables = int(_g.get(
        "OPTIMIZER_MAX_VARIABLES_OPT_OVERRIDE",
        _g.get("MILP_MAX_VARIABLES_OPT_OVERRIDE", 400000),
    ))
    config.milp_candidates_per_customer = int(
        _g.get(
            "OPTIMIZER_CANDIDATES_PER_CUSTOMER_OPT_OVERRIDE",
            _g.get("MILP_CANDIDATES_PER_CUSTOMER_OPT_OVERRIDE", 16),
        )
    )
    config.milp_time_limit_seconds = float(
        _g.get("MILP_TIME_LIMIT_SECONDS_OPT_OVERRIDE", 600.0)
    )
    config.milp_relative_gap = float(_g.get("MILP_RELATIVE_GAP_OPT_OVERRIDE", 0.01))
    for level in [1, 2, 3, 4, 5, 6, 7, 8]:
        config.min_limit[level] = float(_tier_min.get(
            level,
            _tier_min.get(str(level), DEFAULT_TIER_MIN_LIMITS.get(level, 0.0)),
        ))
        config.max_limit[level] = float(_tier_max.get(level, _grid_max))
    for level in [1, 2, 3, 4, 5, 6, 7, 8]:
        config.interest_rates[level] = _objective_interest_rate

    print(f"[DEBUG] prob_grid_path = {config.prob_grid_path}")
    print(
        f"[DEBUG] 目标函数参数: gross_r={_interest_rate}, FTP={_ftp_rate}, "
        f"objective_rate={_objective_interest_rate}, lambda_LGD={_lgd}, "
        f"c1={config.linear_cost}, c2={_quadratic_cost}"
    )

    calculator = LargePrecomputedGridCalculator(config)

    print("\n步骤1：加载客户数据...")
    calculator.load_data()

    print("\n步骤2：读取概率网格...")
    calculator.train_models()
    calculator.apply_optimization_scope(_scope)
    calculator.apply_optimization_sample(
        enabled=_sample_enabled,
        sample_size=_sample_size,
        random_state=_sample_random_state,
    )

    print("\n步骤3：获取人才等级...")
    talent_levels = calculator._extract_talent_levels()
    unique, counts = np.unique(talent_levels, return_counts=True)
    print("人才等级分布（数值等级 -> 人数）:", dict(zip(unique.tolist(), counts.tolist())))

    if _reuse_optimization and not _c2_selection_mode:
        csv_path = os.path.join(_reports_dir, "credit_limit_large_grid_results.csv")
        state_path = os.path.join(_reports_dir, "optimization_state.npz")
        missing = [p for p in [csv_path, state_path] if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "REUSE_OPTIMIZATION=True but saved outputs are incomplete: %s. "
                "Run Cell 6 once with REUSE_OPTIMIZATION=False." % missing
            )
        df_results = pd.read_csv(csv_path, encoding="utf-8-sig")
        state = np.load(state_path, allow_pickle=True)
        if "optimizer_version" not in state.files or str(np.asarray(state["optimizer_version"]).reshape(-1)[0]) != OPTIMIZER_STATE_VERSION:
            raise ValueError(
                "Saved optimization state was produced by an older objective or solver. "
                "Set REUSE_OPTIMIZATION=False and rerun Cell 6 once."
            )
        expected_ids = calculator.data_info["df_aggregated"]["cst_id"].astype(str).to_numpy()
        saved_ids = np.asarray(state["customer_id"]).astype(str)
        if not np.array_equal(expected_ids, saved_ids):
            raise ValueError("Saved optimization state customer order does not match current cleaned data.")
        for key, current in [
            ("lgd_coefficient", config.lgd_coefficient),
            ("linear_cost", config.linear_cost),
            ("quadratic_cost", config.quadratic_cost),
            ("risk_tolerance", config.risk_tolerance),
            ("gross_interest_rate", config.gross_interest_rate),
            ("ftp_rate", config.ftp_rate),
        ]:
            saved = float(np.asarray(state[key]).reshape(-1)[0])
            if not np.isclose(saved, float(current)):
                raise ValueError("Saved %s=%s differs from current %s; rerun optimization." % (key, saved, current))
        saved_interest_rates = json.loads(str(
            np.asarray(state["interest_rates"]).reshape(-1)[0]
        ))
        saved_interest_rates = {
            int(k): float(v) for k, v in saved_interest_rates.items()
        }
        current_interest_rates = {
            int(k): float(v) for k, v in config.interest_rates.items()
        }
        if saved_interest_rates != current_interest_rates:
            raise ValueError("Saved objective interest rates differ; rerun optimization.")
        saved_linear_cost_mode = str(
            np.asarray(state["linear_cost_mode"]).reshape(-1)[0]
        )
        if saved_linear_cost_mode != str(config.linear_cost_mode):
            raise ValueError("Saved linear_cost_mode differs; rerun optimization.")
        saved_ratio = json.loads(str(np.asarray(state["group_mean_min_ratio"]).reshape(-1)[0]))
        saved_ratio = {int(k): float(v) for k, v in saved_ratio.items()}
        if saved_ratio != config.group_mean_min_ratio:
            raise ValueError("Saved group_mean_min_ratio differs from current policy; rerun optimization.")
        saved_tier_min = json.loads(str(np.asarray(state["tier_min_limits"]).reshape(-1)[0]))
        saved_tier_min = {int(k): float(v) for k, v in saved_tier_min.items()}
        current_tier_min = {int(k): float(v) for k, v in config.min_limit.items()}
        if saved_tier_min != current_tier_min:
            raise ValueError("Saved tier_min_limits differs from current policy; rerun optimization.")
        saved_customer_min_enabled = bool(
            np.asarray(state["customer_min_limit_enabled"]).reshape(-1)[0]
        )
        saved_customer_min_decrease = float(
            np.asarray(state["customer_min_limit_decrease"]).reshape(-1)[0]
        )
        if (
            saved_customer_min_enabled != bool(config.customer_min_limit_enabled)
            or not np.isclose(
                saved_customer_min_decrease,
                float(config.customer_min_limit_decrease),
            )
        ):
            raise ValueError("Saved customer minimum-limit policy differs; rerun optimization.")
        saved_scope = str(np.asarray(state["optimization_scope"]).reshape(-1)[0])
        if saved_scope != config.optimization_scope:
            raise ValueError("Saved optimization_scope differs from current scope; rerun optimization.")
        saved_backend = str(np.asarray(state["solver_backend"]).reshape(-1)[0])
        if saved_backend != config.solver_backend:
            raise ValueError("Saved solver_backend differs from current backend; rerun optimization.")
        for key, current in [
            ("optimizer_max_variables", config.milp_max_variables),
            ("optimizer_candidates_per_customer", config.milp_candidates_per_customer),
            ("heuristic_max_rounds", config.heuristic_max_rounds),
            ("lagrangian_iterations", config.lagrangian_iterations),
            ("lagrangian_time_limit_seconds", config.lagrangian_time_limit_seconds),
            ("lagrangian_step_size", config.lagrangian_step_size),
            ("lagrangian_multiplier_cap", config.lagrangian_multiplier_cap),
            ("local_search_enabled", config.local_search_enabled),
            ("local_search_max_passes", config.local_search_max_passes),
            ("local_search_time_limit_seconds", config.local_search_time_limit_seconds),
            ("local_search_pair_candidate_pool", config.local_search_pair_candidate_pool),
        ]:
            saved = float(np.asarray(state[key]).reshape(-1)[0])
            if not np.isclose(saved, float(current)):
                raise ValueError(
                    "Saved %s=%s differs from current %s; rerun optimization."
                    % (key, saved, current)
                )
        saved_sample_enabled = bool(
            np.asarray(state["optimization_sample_enabled"]).reshape(-1)[0]
        )
        saved_sample_size = int(
            np.asarray(state["optimization_sample_size"]).reshape(-1)[0]
        )
        saved_sample_seed = int(
            np.asarray(state["optimization_sample_random_state"]).reshape(-1)[0]
        )
        current_sample_size = -1 if config.optimization_sample_size is None else int(
            config.optimization_sample_size
        )
        if (
            saved_sample_enabled != bool(config.optimization_sample_enabled)
            or saved_sample_size != current_sample_size
            or saved_sample_seed != int(config.optimization_sample_random_state)
        ):
            raise ValueError("Saved optimization sampling configuration differs; rerun optimization.")
        calculator.data_info["base_limits"] = np.asarray(state["base_limits"], dtype=float)
        calculator.data_info["base_limits_raw"] = np.asarray(state["base_limits_raw"], dtype=float)
        calculator.credit_limits = np.asarray(state["final_limits"], dtype=float)
        calculator._last_repair_before_limits = calculator.credit_limits.copy()
        calculator._last_repair_after_limits = calculator.credit_limits.copy()
        calculator._last_best_lambda = np.nan
        calculator.base_total_limit = float(np.sum(calculator.data_info["base_limits_raw"]))
        calculator.config.total_budget = float(np.asarray(state["total_budget"]).reshape(-1)[0])
        calculator.config.risk_budget = float(np.asarray(state["risk_budget"]).reshape(-1)[0])
        calculator.total_allocated = float(np.sum(calculator.credit_limits))
        calculator.base_profit = calculator._calculate_profit_vectorized(calculator.data_info["base_limits"], talent_levels)
        calculator.total_profit = calculator._calculate_profit_vectorized(calculator.credit_limits, talent_levels)
        calculator.base_expected_loss = calculator._calculate_expected_loss(calculator.data_info["base_limits"], talent_levels)
        calculator.opt_expected_loss = calculator._calculate_expected_loss(calculator.credit_limits, talent_levels)
        calculator.opt_weighted_default_risk = calculator._calculate_weighted_default_risk(calculator.credit_limits)
        calculator.data_info["credit_limit_results"] = df_results
        print("✓ 已复用优化结果并恢复诊断状态:", state_path)
        return calculator, np.asarray(talent_levels), df_results

    print("\n步骤4：计算建议额度...")
    if _c2_selection_mode:
        selected_c2, c2_sensitivity_summary, df_results = calculator.select_quadratic_cost(
            talent_levels=talent_levels,
            candidates=_c2_candidates,
            output_dir=_reports_dir,
            max_zero_rate=_c2_max_zero_rate,
            max_upper_hit_rate=_c2_max_upper_hit_rate,
            objective_close_tolerance=_c2_objective_close_tolerance,
        )
        globals()["SELECTED_QUADRATIC_COST"] = float(selected_c2)
        globals()["C2_SENSITIVITY_SUMMARY"] = c2_sensitivity_summary
    else:
        df_results = calculator.calculate_optimal_limits(talent_levels)

    print("\n步骤5：保存结果与报告...")
    os.makedirs(_reports_dir, exist_ok=True)
    output_path = os.path.join(_reports_dir, "credit_limit_large_grid_results.xlsx")
    csv_path    = os.path.join(_reports_dir, "credit_limit_large_grid_results.csv")
    df_results.to_excel(output_path, index=False)
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"详细结果已保存到: {output_path}")
    print(f"CSV 版本已保存到: {csv_path}")
    state_path = os.path.join(_reports_dir, "optimization_state.npz")
    np.savez(
        state_path,
        customer_id=calculator.data_info["df_aggregated"]["cst_id"].astype(str).to_numpy(),
        base_limits=np.asarray(calculator.data_info["base_limits"], dtype=float),
        base_limits_raw=np.asarray(calculator.data_info["base_limits_raw"], dtype=float),
        final_limits=np.asarray(calculator.credit_limits, dtype=float),
        total_budget=np.asarray([calculator.config.total_budget], dtype=float),
        risk_budget=np.asarray([calculator.config.risk_budget], dtype=float),
        lgd_coefficient=np.asarray([calculator.config.lgd_coefficient], dtype=float),
        linear_cost=np.asarray([calculator.config.linear_cost], dtype=float),
        quadratic_cost=np.asarray([calculator.config.quadratic_cost], dtype=float),
        gross_interest_rate=np.asarray([
            calculator.config.gross_interest_rate
        ], dtype=float),
        ftp_rate=np.asarray([calculator.config.ftp_rate], dtype=float),
        linear_cost_mode=np.asarray([calculator.config.linear_cost_mode]),
        interest_rates=np.asarray([
            json.dumps(calculator.config.interest_rates, sort_keys=True)
        ]),
        risk_tolerance=np.asarray([calculator.config.risk_tolerance], dtype=float),
        group_mean_min_ratio=np.asarray([json.dumps(calculator.config.group_mean_min_ratio, sort_keys=True)]),
        tier_min_limits=np.asarray([json.dumps(calculator.config.min_limit, sort_keys=True)]),
        customer_min_limit_enabled=np.asarray([calculator.config.customer_min_limit_enabled]),
        customer_min_limit_decrease=np.asarray([
            calculator.config.customer_min_limit_decrease
        ], dtype=float),
        optimization_scope=np.asarray([calculator.config.optimization_scope]),
        solver_backend=np.asarray([calculator.config.solver_backend]),
        optimizer_max_variables=np.asarray([calculator.config.milp_max_variables]),
        optimizer_candidates_per_customer=np.asarray([
            calculator.config.milp_candidates_per_customer
        ]),
        heuristic_max_rounds=np.asarray([calculator.config.heuristic_max_rounds]),
        lagrangian_iterations=np.asarray([calculator.config.lagrangian_iterations]),
        lagrangian_time_limit_seconds=np.asarray([
            calculator.config.lagrangian_time_limit_seconds
        ]),
        lagrangian_step_size=np.asarray([calculator.config.lagrangian_step_size]),
        lagrangian_multiplier_cap=np.asarray([
            calculator.config.lagrangian_multiplier_cap
        ]),
        local_search_enabled=np.asarray([calculator.config.local_search_enabled]),
        local_search_max_passes=np.asarray([
            calculator.config.local_search_max_passes
        ]),
        local_search_time_limit_seconds=np.asarray([
            calculator.config.local_search_time_limit_seconds
        ]),
        local_search_pair_candidate_pool=np.asarray([
            calculator.config.local_search_pair_candidate_pool
        ]),
        optimization_sample_enabled=np.asarray([calculator.config.optimization_sample_enabled]),
        optimization_sample_size=np.asarray([
            -1 if calculator.config.optimization_sample_size is None
            else calculator.config.optimization_sample_size
        ]),
        optimization_sample_random_state=np.asarray([
            calculator.config.optimization_sample_random_state
        ]),
        source_row_indices=np.asarray(calculator.scope_row_indices, dtype=int),
        optimizer_version=np.asarray([OPTIMIZER_STATE_VERSION]),
    )
    print(f"优化诊断状态已保存到: {state_path}")

    # 保存 talent_levels 供可视化 & 评分卡使用
    np.save(os.path.join(_reports_dir, "talent_levels.npy"), np.asarray(talent_levels))
    print(f"talent_levels.npy 已保存到 {_reports_dir}/")

    calculator.generate_report(df_results)

    print("\n" + "=" * 80)
    print("关键指标")
    print("=" * 80)
    print(f"总授信人数: {len(df_results)}")
    print(f"总预算: {config.total_budget:.2f}")
    print(f"总分配额度: {calculator.total_allocated:.2f}")
    print(f"预算使用比例: {calculator.total_allocated / config.total_budget:.2%}")
    print(f"历史模型目标函数值: {calculator.base_profit:.2f}")
    print(f"优化模型目标函数值: {calculator.total_profit:.2f}")
    print(f"风险预算: {config.risk_budget:.2f}")
    print(f"优化额度加权违约风险: {calculator.opt_weighted_default_risk:.2f}")

    print("\n步骤6：可视化图表请运行 viz_credit_limit.py")
    return calculator, np.asarray(talent_levels), df_results


if __name__ == "__main__":
    # Keep the exact optimization state available to the following notebook cells.
    calculator, talent_levels, df_results = main()
