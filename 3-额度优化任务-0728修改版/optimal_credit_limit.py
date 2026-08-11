#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于图片逻辑的最优额度计算脚本

功能：
1. 加载科创人才数据
2. 训练多任务学习模型预测支用概率和违约概率
3. 基于目标函数和约束条件计算最优额度
4. 生成额度方案报告，包括各等级分析和可视化
5. 支持参数调整
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前设置后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import seaborn as sns
from scipy.optimize import minimize
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore')


def build_dual_labels_train_threshold(
    df_clean: pd.DataFrame,
    y_freq_mode: str,
    train_ratio: float = 0.60,
):
    """用最早参考客户拟合 y_freq 阈值；该范围不是随机模型训练集。"""
    from load_kechuang_potential_data import build_labels_potential

    if not 0 < float(train_ratio) < 1:
        raise ValueError("train_ratio 必须在 (0, 1) 内。")
    if "split_eff_date" not in df_clean.columns:
        raise KeyError("双标签无泄漏构造需要 split_eff_date。")
    dates = pd.to_datetime(df_clean["split_eff_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(
            f"split_eff_date 存在 {int(dates.isna().sum()):,} 个无效值，"
            "无法按最早客户参考期拟合 y_freq 阈值。"
        )
    order = np.argsort(dates.to_numpy(), kind="mergesort")
    n_reference = int(len(df_clean) * float(train_ratio))
    if n_reference <= 0:
        raise ValueError("标签阈值参考客户为空，无法拟合 y_freq 阈值。")
    reference_mask = np.zeros(len(df_clean), dtype=bool)
    reference_mask[order[:n_reference]] = True
    labeled, thresholds_usage = build_labels_potential(
        df_clean,
        target="y_freq",
        y_freq_mode=y_freq_mode,
        threshold_fit_mask=reference_mask,
    )
    labeled, thresholds_default = build_labels_potential(
        labeled, target="y_dq_risk"
    )
    return labeled, thresholds_usage, thresholds_default


def _sns_set_style(style="whitegrid", context="talk"):
    """兼容旧版 seaborn：统一用 set_style/set_context，不调用 set_theme。"""
    try:
        if hasattr(sns, "set_style"):
            sns.set_style(style)
        if hasattr(sns, "set_context"):
            sns.set_context(context)
    except Exception:
        pass


def _sns_histplot(vals, bins, kde, color, ax):
    """兼容旧版 seaborn（<0.11 无 histplot）。"""
    if hasattr(sns, "histplot"):
        sns.histplot(vals, bins=bins, kde=kde, color=color, ax=ax)
    else:
        sns.distplot(vals, bins=bins, kde=kde, color=color, ax=ax)

# 设置中文字体（兼容 Linux 服务器，按优先级依次尝试）
_zh_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans SC',
             'Noto Sans CJK JP', 'AR PL UMing CN', 'SimHei', 'DejaVu Sans']
_available = {f.name for f in _fm.fontManager.ttflist}
_chosen = next((f for f in _zh_fonts if f in _available), None)
if _chosen:
    plt.rcParams['font.sans-serif'] = [_chosen] + plt.rcParams['font.sans-serif']
    print(f"[字体] 使用中文字体: {_chosen}")
else:
    print("[字体] 未找到中文字体，中文可能显示为方块")
plt.rcParams['axes.unicode_minus'] = False

class OptimalCreditLimitConfig:
    """最优额度计算配置"""
    def __init__(self):
        # 数据配置
        # 本地模拟数据文件，包含实际授信额度字段「授信额度」
        self.excel_path = "kechuang_talent_full_data.xlsx"
        self.cleaned_file = None
        self.csv_encoding = "utf-8-sig"
        self.snapshot_date = "2026-01-31"
        self.maturity_cutoff = "2026-07-21"
        self.apply_maturity_filter = False
        self.apply_eff_date_filter = True
        self.eff_date_lower = "2025-01-01"
        self.eff_date_upper = "2026-03-31"
        self.dedup_cst_loan = False
        self.y_freq_mode = "bout_gt0_and_curr_p80"
        self.label_threshold_train_ratio = 0.60
        # 使用全样本参与训练与额度优化；如需加速，可在 main 中临时覆盖该值
        self.max_samples = None  # 最大样本数，None表示使用全部样本，可设置为100、200、500等
        
        # 模型配置
        self.use_mtl = True  # 是否使用MTL模型
        # 支用/违约可以使用不同模型
        # - usage_model_type: 支用概率模型类型（"logistic_mtl_l1" / "logistic_mtl_l2" / "lightgbm_mtl"）
        # - default_model_type: 违约概率模型类型（"logistic_mtl_l1" / "logistic_mtl_l2" / "lightgbm_mtl"）
        self.usage_model_type = "lightgbm_mtl"
        # 违约概率模型：改回原先的 Logistic MTL（L2惩罚）
        self.default_model_type = "lightgbm_mtl"
        # 兼容旧字段：若仍设置 model_type，则两者都用同一种
        self.model_type = "lightgbm"  # "lightgbm" or "logistic"
        self.mtl_G = 1  # MTL模型的组数
        # MTL 任务分组方式：与 load_kechuang_data.convert_to_mtl_format 保持一致
        # MTL 任务分组：任务太碎会导致极端概率/断档；默认使用更粗的 age_group
        # 支用/违约模型分别使用不同任务分组（CV最优结果）
        self.mtl_task_grouping = "aum_level_group"          # 支用概率模型
        self.mtl_task_grouping_default = "aum_occup_major_group"  # 违约概率模型

        # 概率校准（后处理）：树模型概率容易偏极端，可选 Platt(=sigmoid) 或 isotonic
        self.calibrate_probs = True
        self.calibration_method = "isotonic"  # "platt" or "isotonic"
        
        # 业务参数（可手动调整）。人才等级 1~8 对应档位 F3、F2、F1、E、D、C、B、A（8 最高）
        # 其中 C/B/A 不再沿用旧的占位外推值，改为与当前业务口径对齐的更平滑权重。
        # 这只是优化计算中的相对权重，不是硬性的“额度上限”。
        self.talent_weights = {1: 1.0, 2: 1.2, 3: 1.4, 4: 1.6, 5: 2.0, 6: 2.0, 7: 2.2, 8: 2.4}  # 人才权重 ω(gi)
        self.alpha = 1.5  # 风险/违约系数（≥0）
        self.LGD = 0.45  # Loss Given Default：违约损失率（关键）
        # 违约概率缩放：等价于在利润/损失里使用 PD_adj = p_eff * 0.3
        self.default_prob_scale = 0.3
        self.total_budget = 2e9  # 占位，运行时自动覆盖为真实额度总额（见 calculate_optimal_limits）
        # 各等级最低/最高额度：这里先给统一占位，随后会在 main() / calculate_optimal_limits
        # 中按真实数据范围整体覆盖，因此所有 1~8 档（F3~A）都会参与同一套额度优化。
        self.min_limit = {1: 10000.0, 2: 10000.0, 3: 10000.0, 4: 10000.0, 5: 10000.0, 6: 10000.0, 7: 10000.0, 8: 10000.0}
        self.max_limit = {1: 100000.0, 2: 100000.0, 3: 100000.0, 4: 100000.0, 5: 100000.0, 6: 100000.0, 7: 100000.0, 8: 100000.0}
        self.interest_rates = {1: 0.05, 2: 0.052, 3: 0.055, 4: 0.06, 5: 0.065, 6: 0.068, 7: 0.07, 8: 0.075}  # 贷款利率 ri
        # 支用率尺度：用于 g(L) = L / usage_rate_scale
        # 最终有效支用概率为 p_usage_eff(L) = p_usage_base * (1 - exp(-usage_alpha * g(L)))，
        # 即随额度递增、边际递减的形式
        self.usage_rate_scale = 50000.0
        self.usage_alpha = 1.0  # 用于控制随额度变化的强度（≥0）
        # 优化方法：
        # - use_fast_greedy: 连续贪心 + 离散化（默认）
        # - use_discrete_optimization: 启用离散优化（混合：连续贪心 + 局部搜索）
        self.use_discrete_optimization = False
        self.use_fast_greedy = True
        self.limit_step = 100.0   # 额度档位步长（固定100，不自动检测）
        self.discrete_local_iterations = 100  # 离散局部搜索的最大迭代次数
        # 当所有客户 c_eff<=0 时，为避免所有人卡在 L_min，可把剩余预算分配给“最不差”的一部分客户
        self.fallback_allocation_frac = 0.10


class OptimalCreditLimitCalculator:
    """最优额度计算器"""
    
    def __init__(self, config: OptimalCreditLimitConfig):
        self.config = config
        self.models = {}
        self.data_info = {}
        self.credit_limits = []
        self.total_profit = 0.0
        self.total_allocated = 0.0
        # 基准（原始额度）统计
        self.base_total_limit = 0.0
        self.base_profit = 0.0
        self.base_expected_loss = 0.0
        # 优化后损失
        self.opt_expected_loss = 0.0
    
    def load_data(self):
        """加载并预处理数据（load_kechuang_potential_data）。"""
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

        print("加载数据（load_kechuang_potential_data）...")
        cleaned_file = getattr(self.config, "cleaned_file", None)
        csv_encoding = str(getattr(self.config, "csv_encoding", "utf-8-sig"))
        thr_usage = {}
        thr_default = {}

        if cleaned_file and os.path.isfile(str(cleaned_file)):
            print(f"  读取清洗后数据: {cleaned_file}")
            work = pd.read_csv(cleaned_file, encoding=csv_encoding)
            if "y_freq" not in work.columns or "y_dq_risk" not in work.columns:
                raise ValueError(
                    f"{cleaned_file} 缺少 y_freq 或 y_dq_risk，请先运行 Cell 3 重新清洗"
                )
            df0 = work
        else:
            excel_path = str(self.config.excel_path)
            print(f"  从原始文件预处理: {excel_path}")
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
            df0 = filter_dq_start_customers(df0)
            df_agg = aggregate_by_customer_potential(df0)
            df_clean = clean_data_potential(
                df_agg,
                snapshot_date=str(getattr(self.config, "snapshot_date", "2026-01-31")),
                target="y_dq_risk",
            )
            y_freq_mode = str(getattr(self.config, "y_freq_mode", "bout_gt0_and_curr_p80"))
            df_labeled, thr_usage, thr_default = build_dual_labels_train_threshold(
                df_clean,
                y_freq_mode=y_freq_mode,
                train_ratio=float(getattr(
                    self.config, "label_threshold_train_ratio", 0.60
                )),
            )
            work = drop_post_label_cols(df_labeled, target="y_dq_risk")
            df0 = df_agg

        if "credamt" in work.columns and "授信额度" not in work.columns:
            work = work.copy()
            work["授信额度"] = work["credamt"]

        X_usage, y_usage, feature_names_usage, _ = feature_engineering_potential(
            work, target="y_freq",
            add_quota_sq=True, add_quota_cube=True, add_quota_log=False,
        )
        X_default, y_default, feature_names_default, _ = feature_engineering_potential(
            work, target="y_dq_risk",
            add_quota_sq=True, add_quota_cube=True, add_quota_log=False,
        )
        
        # 样本数量控制
        if self.config.max_samples is not None and len(X_usage) > self.config.max_samples:
            print(f"样本数量控制：从 {len(X_usage)} 减少到 {self.config.max_samples}")
            np.random.seed(42)
            sample_indices = np.random.choice(len(X_usage), self.config.max_samples, replace=False)
            X_usage = X_usage.iloc[sample_indices]
            y_usage = y_usage.iloc[sample_indices]
            X_default = X_default.iloc[sample_indices]
            y_default = y_default.iloc[sample_indices]
            work = work.iloc[sample_indices].copy()
        
        # 保存数据信息
        self.data_info = {
            "df_raw": df0,
            "df_aggregated": work,
            "df_cleaned": work,
            "X_usage": X_usage,
            "y_usage": y_usage,
            "X_default": X_default,
            "y_default": y_default,
            "feature_names_usage": feature_names_usage,
            "feature_names_default": feature_names_default,
            "thresholds_usage": thr_usage,
            "thresholds_default": thr_default,
        }
        
        print(f"数据加载完成，样本数: {len(X_usage)}")

        # 自动从真实数据检测并配置额度参数（min/max/step/budget）
        self._auto_configure_from_data()

        return self.data_info

    def _auto_configure_from_data(self):
        """
        从 df_aggregated 的「授信额度」列自动检测真实数据的额度参数，
        并覆盖 config 中的 min_limit / max_limit / limit_step / total_budget。
        同时生成真实额度分布的可视化图表，保存到 reports/real_data_analysis/。
        无论上传何种真实数据，此函数均自动适配。
        """
        df = self.data_info.get("df_aggregated")
        cred_col = None
        if df is not None:
            if "授信额度" in df.columns:
                cred_col = "授信额度"
            elif "credamt" in df.columns:
                cred_col = "credamt"
        if df is None or cred_col is None:
            print("[自动配置] 未找到「授信额度/credamt」列，跳过参数自动配置。")
            return

        vals = pd.to_numeric(df[cred_col], errors="coerce").dropna()
        if len(vals) == 0:
            print("[自动配置] 「授信额度」列全为空，跳过参数自动配置。")
            return

        real_min = float(vals.min())
        real_max = float(vals.max())
        real_total = float(vals.sum())

        # 步长固定使用 config 中手动设定的值（100），不自动检测
        detected_step = self.config.limit_step

        print("\n" + "=" * 60)
        print("[自动配置] 真实数据额度统计（来自「授信额度」列）")
        print(f"  有效样本数: {len(vals):,}")
        print(f"  最小额度:   {real_min:,.0f}")
        print(f"  最大额度:   {real_max:,.0f}")
        print(f"  均值:       {vals.mean():,.0f}")
        print(f"  中位数:     {vals.median():,.0f}")
        print(f"  总额度:     {real_total:,.0f}")
        print(f"  检测步长:   {detected_step:,.0f}")

        # 更新 config（所有等级统一使用真实数据上下限）
        for level in list(self.config.min_limit.keys()):
            self.config.min_limit[level] = real_min
        for level in list(self.config.max_limit.keys()):
            self.config.max_limit[level] = real_max
        # limit_step 已在 config 中手动设定（100），不覆盖
        # self.config.limit_step = detected_step
        self.config.total_budget = real_total

        print(f"  → config.min_limit 已更新为: {real_min:,.0f}（全等级统一）")
        print(f"  → config.max_limit 已更新为: {real_max:,.0f}（全等级统一）")
        print(f"  → config.limit_step 已更新为: {detected_step:,.0f}")
        print(f"  → config.total_budget 已更新为: {real_total:,.0f}")
        print("=" * 60 + "\n")

        # 生成可视化（失败时不阻断主流程）
        try:
            self._visualize_real_credit_distribution(vals, df)
        except Exception as exc:
            print(f"[自动配置] 跳过额度分布图（{exc}）")

    def _visualize_real_credit_distribution(self, vals: pd.Series, df_agg: pd.DataFrame):
        """
        可视化真实额度分布，保存到 reports/real_data_analysis/：
        1. 整体直方图 + KDE
        2. 累计分布（CDF）
        3. 按人才等级分组的箱线图（若存在档位/cst_star_cd 列）
        """
        viz_dir = "reports/real_data_analysis"
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        _sns_set_style(style="whitegrid", context="talk")
        # 沿用顶部已自动检测的字体，不再覆盖
        plt.rcParams["axes.unicode_minus"] = False

        real_min = float(vals.min())
        real_max = float(vals.max())
        real_total = float(vals.sum())
        step = float(self.config.limit_step)
        n_bins = max(10, min(100, int((real_max - real_min) / step) + 1))

        # --- 图1：直方图 + KDE ---
        fig, ax = plt.subplots(figsize=(10, 5))
        _sns_histplot(vals, bins=n_bins, kde=True, color="steelblue", ax=ax)
        ax.axvline(vals.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean {vals.mean():,.0f}")
        ax.axvline(vals.median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median {vals.median():,.0f}")
        ax.set_title(f"Credit Limit Distribution (n={len(vals):,}, total={real_total/1e8:.2f}B)")
        ax.set_xlabel("Credit Limit")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "real_limit_histogram.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # --- 图2：CDF ---
        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_vals = np.sort(vals.to_numpy())
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax.plot(sorted_vals, cdf, color="steelblue", linewidth=2)
        for pct in [0.25, 0.50, 0.75, 0.90]:
            v = np.quantile(sorted_vals, pct)
            ax.axvline(v, linestyle=":", color="gray", linewidth=1)
            ax.text(v, pct - 0.04, f"P{int(pct*100)}={v:,.0f}", fontsize=9, ha="center")
        ax.set_title("Credit Limit Cumulative Distribution (CDF)")
        ax.set_xlabel("Credit Limit")
        ax.set_ylabel("Cumulative Ratio")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "real_limit_cdf.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # --- 图3：按人才等级分组箱线图（若有分组列）---
        grade_col = None
        cred_col = None
        if "档位" in df_agg.columns:
            grade_col = "档位"
        elif "cst_star_cd" in df_agg.columns:
            grade_col = "cst_star_cd"

        if grade_col is not None:
            cred_col = "授信额度" if "授信额度" in df_agg.columns else "credamt"
            if cred_col not in df_agg.columns:
                cred_col = None
        if grade_col is not None and cred_col is not None:
            df_plot = df_agg[[grade_col, cred_col]].copy()
            df_plot[cred_col] = pd.to_numeric(df_plot[cred_col], errors="coerce")
            df_plot = df_plot.dropna(subset=[cred_col])

            # 把中文列名重命名为英文，避免字体问题
            df_plot = df_plot.rename(columns={cred_col: "Credit_Limit"})
            fig, ax = plt.subplots(figsize=(10, 5))
            order = sorted(df_plot[grade_col].dropna().unique())
            sns.boxplot(data=df_plot, x=grade_col, y="Credit_Limit", order=order,
                        palette="viridis", ax=ax)
            # 标注各组均值
            for i, grp in enumerate(order):
                sub = df_plot.loc[df_plot[grade_col] == grp, "Credit_Limit"]
                ax.text(i, sub.max() * 1.01, f"Mean\n{sub.mean():,.0f}",
                        ha="center", fontsize=8, color="darkblue")
            ax.set_title(f"Credit Limit Distribution by Grade ({grade_col})")
            ax.set_xlabel("Grade")
            ax.set_ylabel("Credit Limit")
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "real_limit_by_grade.png"), dpi=200, bbox_inches="tight")
            plt.close()

        print(f"[自动配置] 真实额度分布图已保存到: {viz_dir}")

    def _extract_talent_levels(self) -> np.ndarray:
        """
        从聚合表中提取人才等级（1~8）。
        实际档位从优到劣：A > B > C > D > E > F1 > F2 > F3，对应等级 8/7/6/5/4/3/2/1。
        与 load_kechuang_potential_data.py 中 feature_engineering_potential 的
        tier_order 编码保持一致，确保 A/B/C 档客户不会被错误归并为最低档 F3。
        优先使用当前优化范围内的 df_aggregated，其次使用同长度的 df_cleaned。
        找不到或无法识别时直接报错，禁止随机生成等级。
        """
        current = self.data_info.get("df_aggregated")
        if current is None:
            raise ValueError("data_info 缺少 df_aggregated，无法提取人才等级。")
        n = len(current)

        # 实际档位：F3(1) → ... → A(8)
        # 与 load_kechuang_potential_data.py 的 tier_order 完全对齐（含大小写/“级”后缀变体）。
        level_map = {
            "F3": 1, "F3级": 1, "f3": 1,
            "F2": 2, "F2级": 2, "f2": 2,
            "F1": 3, "F1级": 3, "f1": 3,
            "E":  4, "E级":  4, "e":  4,
            "D":  5, "D级":  5, "d":  5,
            "C":  6, "C级":  6, "c":  6,
            "B":  7, "B级":  7, "b":  7,
            "A":  8, "A级":  8, "a":  8,
        }

        # df_aggregated 已随 optimization_scope / 抽样同步裁剪，应作为第一来源。
        for key in ["df_aggregated", "df_cleaned"]:
            df_agg = self.data_info.get(key)
            if df_agg is None or len(df_agg) != n:
                continue
            if "档位" in df_agg.columns:
                raw = df_agg["档位"].astype(str).str.strip()
                mapped = raw.map(level_map)
                n_unmapped = int(mapped.isna().sum())
                if n_unmapped > 0:
                    unmapped_vals = sorted(raw[mapped.isna()].unique().tolist())
                    raise ValueError(
                        f"[档位] {key} 中有 {n_unmapped} 个值未能映射到 A~F3 任一档位: "
                        f"{unmapped_vals}。禁止随机或按 F3 兜底，请修复人才等级字段。"
                    )
                result = mapped.astype(int).to_numpy()
                print(f"[档位] 使用 {key} 中的档位列，等级分布(1~8 -> 人数): {dict(zip(*np.unique(result, return_counts=True)))}")
                return result
            if "talent_level" in df_agg.columns:
                numeric = pd.to_numeric(df_agg["talent_level"], errors="coerce")
                valid = (
                    numeric.notna()
                    & numeric.between(1, 8)
                    & np.isclose(numeric, np.round(numeric))
                )
                if not valid.all():
                    examples = df_agg.loc[~valid, "talent_level"].astype(str).unique()[:10].tolist()
                    raise ValueError(
                        f"[档位] {key}.talent_level 存在无法识别的等级: {examples}。"
                    )
                result = numeric.astype(int).to_numpy()
                print(f"[档位] 使用 {key}.talent_level，等级分布(1~8 -> 人数): {dict(zip(*np.unique(result, return_counts=True)))}")
                return result

        raise ValueError(
            "当前优化客户数据缺少可用的 档位/talent_level 字段；"
            "请确认概率网格 work_features.csv 与清洗数据来自同一次运行。"
        )
    
    def _extract_mtl_task_labels(self) -> np.ndarray:
        """
        按 compare_models_kechuang/load_kechuang_data 的规则生成 MTL 任务分组标签。
        默认使用 age_occup_group（年龄分位×职业）。
        返回长度为 n_samples 的 task label（字符串数组），用于构建 X_list/y_list。
        """
        df_full = self.data_info.get("df_cleaned")
        n = len(self.data_info["X_usage"])
        if df_full is None or len(df_full) != n:
            raise ValueError("df_cleaned 缺失或长度不一致，无法生成 MTL 任务分组。")

        grouping = str(getattr(self.config, "mtl_task_grouping", "age_occup_group"))

        # 若数据里已有同名任务列则直接用（与 load_kechuang_data.convert_to_mtl_format 对齐）
        if grouping in df_full.columns:
            return df_full[grouping].astype(str).to_numpy()

        df_tmp = df_full.copy()

        if grouping == "age_occup_group":
            if "age" not in df_tmp.columns:
                raise ValueError("需要 age_occup_group 分组，但数据中没有 age 列。")
            if "occup_cd" not in df_tmp.columns:
                raise ValueError("需要 age_occup_group 分组，但数据中没有 occup_cd 列。")

            age_num = pd.to_numeric(df_tmp["age"], errors="coerce")
            df_tmp["age_bin"] = pd.qcut(
                age_num,
                q=5,
                labels=["age_q1", "age_q2", "age_q3", "age_q4", "age_q5"],
                duplicates="drop",
            ).astype(str).fillna("age_unknown")
            df_tmp["occup_bin"] = "occup_" + df_tmp["occup_cd"].astype(str).fillna("unknown")
            task = (df_tmp["age_bin"].astype(str) + "_" + df_tmp["occup_bin"].astype(str)).astype(str)
            return task.to_numpy()

        if grouping == "age_aum_group":
            if "age" not in df_tmp.columns:
                raise ValueError("需要 age_aum_group 分组，但数据中没有 age 列。")
            if "当前aum" not in df_tmp.columns:
                raise ValueError("需要 age_aum_group 分组，但数据中没有 当前aum 列。")

            age_num = pd.to_numeric(df_tmp["age"], errors="coerce")
            df_tmp["age_bin"] = pd.qcut(
                age_num,
                q=5,
                labels=["age_q1", "age_q2", "age_q3", "age_q4", "age_q5"],
                duplicates="drop",
            ).astype(str).fillna("age_unknown")
            aum_num = pd.to_numeric(df_tmp["当前aum"], errors="coerce")
            df_tmp["aum_bin"] = pd.qcut(
                aum_num,
                q=3,
                labels=["aum_low", "aum_mid", "aum_high"],
                duplicates="drop",
            ).astype(str).fillna("aum_unknown")
            task = (df_tmp["age_bin"].astype(str) + "_" + df_tmp["aum_bin"].astype(str)).astype(str)
            return task.to_numpy()

        if grouping == "age_occup_aum_group":
            if "age" not in df_tmp.columns:
                raise ValueError("需要 age_occup_aum_group 分组，但数据中没有 age 列。")
            if "occup_cd" not in df_tmp.columns:
                raise ValueError("需要 age_occup_aum_group 分组，但数据中没有 occup_cd 列。")
            if "当前aum" not in df_tmp.columns:
                raise ValueError("需要 age_occup_aum_group 分组，但数据中没有 当前aum 列。")

            age_num = pd.to_numeric(df_tmp["age"], errors="coerce")
            df_tmp["age_bin"] = pd.cut(
                age_num,
                bins=[0, 30, 40, 50, 100],
                labels=["age_<30", "age_30-40", "age_40-50", "age_50+"],
            ).astype(str).fillna("age_unknown")
            df_tmp["occup_bin"] = "occup_" + df_tmp["occup_cd"].astype(str).fillna("occup_unknown")
            aum_num = pd.to_numeric(df_tmp["当前aum"], errors="coerce")
            df_tmp["aum_bin"] = pd.qcut(
                aum_num,
                q=3,
                labels=["aum_low", "aum_mid", "aum_high"],
                duplicates="drop",
            ).astype(str).fillna("aum_unknown")
            task = (df_tmp["age_bin"].astype(str) + "_" + df_tmp["occup_bin"].astype(str) + "_" + df_tmp["aum_bin"].astype(str)).astype(str)
            return task.to_numpy()

        if grouping == "age_group":
            if "age" not in df_tmp.columns:
                raise ValueError("需要 age_group 分组，但数据中没有 age 列。")
            age_num = pd.to_numeric(df_tmp["age"], errors="coerce")
            task = pd.qcut(
                age_num,
                q=5,
                labels=["age_q1", "age_q2", "age_q3", "age_q4", "age_q5"],
                duplicates="drop",
            ).astype(str).fillna("age_unknown")
            return task.to_numpy()

        if grouping == "occup_group":
            if "occup_cd" not in df_tmp.columns:
                raise ValueError("需要 occup_group 分组，但数据中没有 occup_cd 列。")
            task = ("occup_" + df_tmp["occup_cd"].astype(str).fillna("unknown")).astype(str)
            return task.to_numpy()

        if grouping == "occup_aum_group":
            if "occup_cd" not in df_tmp.columns:
                raise ValueError("需要 occup_aum_group 分组，但数据中没有 occup_cd 列。")
            if "当前aum" not in df_tmp.columns:
                raise ValueError("需要 occup_aum_group 分组，但数据中没有 当前aum 列。")
            occup_bin = "occup_" + df_tmp["occup_cd"].astype(str).fillna("unknown")
            aum_num = pd.to_numeric(df_tmp["当前aum"], errors="coerce")
            aum_bin = pd.qcut(
                aum_num,
                q=3,
                labels=["aum_low", "aum_mid", "aum_high"],
                duplicates="drop",
            ).astype(str).fillna("aum_unknown")
            task = (occup_bin.astype(str) + "_" + aum_bin.astype(str)).astype(str)
            return task.to_numpy()

        if grouping == "aum_level_group":
            if "当前aum" not in df_tmp.columns:
                raise ValueError("需要 aum_level_group 分组，但数据中没有 当前aum 列。")
            aum_num = pd.to_numeric(df_tmp["当前aum"], errors="coerce")
            df_tmp["aum_bin"] = pd.qcut(
                aum_num, q=3,
                labels=["aum_low", "aum_mid", "aum_high"],
                duplicates="drop"
            ).astype(str).fillna("aum_unknown")
            level_col = next((c for c in ["档位", "level", "talent_level", "kechuang_level", "科创档位"] if c in df_tmp.columns), None)
            if level_col is None:
                raise ValueError("需要 aum_level_group 分组，但数据中找不到档位列。")
            df_tmp["level_bin"] = df_tmp[level_col].astype(str).replace("nan", "unknown").fillna("unknown").apply(lambda x: f"level_{x}")
            task = (df_tmp["aum_bin"].astype(str) + "_" + df_tmp["level_bin"].astype(str))
            return task.to_numpy()

        if grouping == "aum_occup_major_group":
            if "当前aum" not in df_tmp.columns:
                raise ValueError("需要 aum_occup_major_group 分组，但数据中没有 当前aum 列。")
            if "occup_cd" not in df_tmp.columns:
                raise ValueError("需要 aum_occup_major_group 分组，但数据中没有 occup_cd 列。")
            aum_num = pd.to_numeric(df_tmp["当前aum"], errors="coerce")
            df_tmp["aum_bin"] = pd.qcut(
                aum_num, q=3,
                labels=["aum_low", "aum_mid", "aum_high"],
                duplicates="drop"
            ).astype(str).fillna("aum_unknown")
            def _occup_major(val):
                if pd.isna(val):
                    return "occup_major_unknown"
                s = str(int(float(val))) if str(val).replace(".", "").replace("-", "").isdigit() else str(val)
                first = s[0] if s and s[0].isdigit() else "unknown"
                return f"occup_major_{first}"
            df_tmp["occup_major_bin"] = df_tmp["occup_cd"].map(_occup_major)
            task = (df_tmp["aum_bin"].astype(str) + "_" + df_tmp["occup_major_bin"].astype(str))
            return task.to_numpy()

        raise ValueError(
            f"不支持的任务分组方式: {grouping}。"
            "仅支持 'age_aum_group' | 'age_occup_aum_group' | 'age_group' | 'occup_group' | "
            "'age_occup_group' | 'occup_aum_group' | 'aum_level_group' | 'aum_occup_major_group'"
        )

    def _calibrate_proba(self, y_true: np.ndarray, p_raw: np.ndarray) -> np.ndarray:
        """
        概率后验校准（可选）。
        - platt: 用一元 Logistic 回归对 p_raw 做 sigmoid 校准
        - isotonic: 用 IsotonicRegression 做单调校准
        说明：这里用同一份数据做校准，适合模拟/可视化场景；若要更严谨可改成交叉验证校准。
        """
        if not bool(getattr(self.config, "calibrate_probs", False)):
            return np.clip(np.asarray(p_raw, dtype=float), 1e-6, 1 - 1e-6)

        y = np.asarray(y_true).reshape(-1).astype(int)
        p = np.clip(np.asarray(p_raw, dtype=float).reshape(-1), 1e-6, 1 - 1e-6)
        if len(np.unique(y)) < 2:
            return p

        method = str(getattr(self.config, "calibration_method", "isotonic")).lower()
        if method == "platt":
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000, random_state=42)
            lr.fit(p.reshape(-1, 1), y)
            return np.clip(lr.predict_proba(p.reshape(-1, 1))[:, 1], 1e-6, 1 - 1e-6)
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y)
            return np.clip(iso.predict(p), 1e-6, 1 - 1e-6)
        return p

    def train_models(self):
        """训练模型"""
        from compare_models_kechuang import (
            train_predict_lightgbm,
            train_predict_lightgbm_mtl,
            train_predict_logistic_mtl_penalty,
        )

        print("训练模型...")
        
        X_usage = self.data_info["X_usage"]
        y_usage = self.data_info["y_usage"]
        X_default = self.data_info["X_default"]
        y_default = self.data_info["y_default"]
        
        # 划分训练集和测试集（这里简化处理，使用全部数据训练）
        X_train_usage, y_train_usage = X_usage.values, y_usage.values
        X_train_default, y_train_default = X_default.values, y_default.values
        
        # 训练两个概率模型（可分别配置模型类型）
        usage_type = getattr(self.config, "usage_model_type", None)
        default_type = getattr(self.config, "default_model_type", None)
        # 兼容旧字段：若未设置 usage/default，则用 model_type 推断
        if usage_type is None or default_type is None:
            if getattr(self.config, "model_type", "lightgbm") == "lightgbm":
                usage_type = default_type = "lightgbm_mtl" if self.config.use_mtl else "lightgbm"
            else:
                usage_type = default_type = "logistic_mtl_l1" if self.config.use_mtl else "logistic"

        if not self.config.use_mtl:
            # 单模型分支（很少用）
            if usage_type == "lightgbm":
                _, y_proba_usage = train_predict_lightgbm(X_train_usage, y_train_usage, X_train_usage, y_train_usage)
            else:
                from compare_models_kechuang import train_predict_logistic_no_penalty
                _, y_proba_usage = train_predict_logistic_no_penalty(X_train_usage, y_train_usage, X_train_usage, y_train_usage)

            if default_type == "lightgbm":
                _, y_proba_default = train_predict_lightgbm(X_train_default, y_train_default, X_train_default, y_train_default)
            else:
                from compare_models_kechuang import train_predict_logistic_no_penalty
                _, y_proba_default = train_predict_logistic_no_penalty(X_train_default, y_train_default, X_train_default, y_train_default)
        else:
            # MTL 分支
            # 按 compare_models_kechuang 的默认规则（age_aum_group）拆任务，而不是按人才等级
            task_labels = self._extract_mtl_task_labels()
            groups = sorted(pd.Series(task_labels).value_counts().index.tolist())

            # 构造任务列表，同时保留每个任务对应的原始样本下标，用于把 y_proba 还原到原顺序
            X_usage_list, y_usage_list, idx_usage_list = [], [], []
            X_default_list, y_default_list, idx_default_list = [], [], []
            for g in groups:
                idx = np.where(task_labels == g)[0]
                if idx.size < 20:
                    continue
                # 确保该任务组内正负都有，否则 Logistic/LightGBM 训练不稳定
                if np.sum(y_train_usage[idx] == 1) == 0 or np.sum(y_train_usage[idx] == 0) == 0:
                    continue
                if np.sum(y_train_default[idx] == 1) == 0 or np.sum(y_train_default[idx] == 0) == 0:
                    continue

                X_usage_list.append(X_train_usage[idx])
                y_usage_list.append(y_train_usage[idx])
                idx_usage_list.append(idx)
                X_default_list.append(X_train_default[idx])
                y_default_list.append(y_train_default[idx])
                idx_default_list.append(idx)

            if len(X_usage_list) < 2:
                raise ValueError("MTL 需要至少 2 个任务；请检查 age/当前aum 是否存在且分组后每组有足够正负样本。")

            K_clusters = int(min(int(getattr(self.config, "mtl_G", 4)), len(X_usage_list)))

            # --- 支用概率 ---
            if usage_type in ("logistic_mtl_l1", "logistic_mtl_l2"):
                from compare_models_kechuang import _protected_idx_from_feature_names
                protected_usage = _protected_idx_from_feature_names(self.data_info.get("feature_names_usage"))
                _, y_proba_usage_concat = train_predict_logistic_mtl_penalty(
                    X_usage_list, y_usage_list,
                    X_usage_list, y_usage_list,
                    K=K_clusters,
                    penalty_B="l2" if usage_type == "logistic_mtl_l2" else "l1",
                    protected_idx=protected_usage,
                )
                # 还原到原顺序
                y_proba_usage = np.zeros(len(X_train_usage), dtype=float)
                cursor = 0
                for idx in idx_usage_list:
                    n_i = len(idx)
                    y_proba_usage[idx] = y_proba_usage_concat[cursor:cursor + n_i]
                    cursor += n_i
                # 对于未进入任何有效任务组的样本，用全量 Logistic(L1) 兜底（否则会出现大量 0 造成“断档/竖线”）
                covered = np.zeros(len(X_train_usage), dtype=bool)
                for idx in idx_usage_list:
                    covered[idx] = True
                uncovered_idx = np.where(~covered)[0]
                if uncovered_idx.size > 0:
                    from compare_models_kechuang import train_predict_logistic_l1
                    _, proba_fallback = train_predict_logistic_l1(
                        X_train_usage, y_train_usage,
                        X_train_usage[uncovered_idx], y_train_usage[uncovered_idx],
                        C=1.0,
                        max_iter=1000,
                        protected_idx=protected_usage,
                    )
                    y_proba_usage[uncovered_idx] = proba_fallback
                # 校准（可选）
                y_proba_usage = self._calibrate_proba(y_train_usage, y_proba_usage)
            elif usage_type == "lightgbm_mtl":
                # 用同样的任务拆分分别训练每个任务，再在任务层面做 MTL 聚类
                # 这里直接复用 lightgbm_mtl 的接口（按任务列表传入）
                _, y_proba_usage_concat = train_predict_lightgbm_mtl(
                    X_usage_list, y_usage_list,
                    X_usage_list, y_usage_list,
                    G=K_clusters,
                )
                y_proba_usage = np.zeros(len(X_train_usage), dtype=float)
                cursor = 0
                for idx in idx_usage_list:
                    n_i = len(idx)
                    y_proba_usage[idx] = y_proba_usage_concat[cursor:cursor + n_i]
                    cursor += n_i
                covered = np.zeros(len(X_train_usage), dtype=bool)
                for idx in idx_usage_list:
                    covered[idx] = True
                uncovered_idx = np.where(~covered)[0]
                if uncovered_idx.size > 0:
                    _, proba_fallback = train_predict_lightgbm(
                        X_train_usage, y_train_usage,
                        X_train_usage[uncovered_idx], y_train_usage[uncovered_idx],
                    )
                    y_proba_usage[uncovered_idx] = proba_fallback
                y_proba_usage = self._calibrate_proba(y_train_usage, y_proba_usage)
            else:
                raise ValueError(f"未知 usage_model_type: {usage_type}")

            # --- 违约概率 ---
            if default_type in ("logistic_mtl_l1", "logistic_mtl_l2"):
                from compare_models_kechuang import _protected_idx_from_feature_names
                protected_default = _protected_idx_from_feature_names(self.data_info.get("feature_names_default"))
                _, y_proba_default_concat = train_predict_logistic_mtl_penalty(
                    X_default_list, y_default_list,
                    X_default_list, y_default_list,
                    K=K_clusters,
                    penalty_B="l2" if default_type == "logistic_mtl_l2" else "l1",
                    protected_idx=protected_default,
                )
                y_proba_default = np.zeros(len(X_train_default), dtype=float)
                cursor = 0
                for idx in idx_default_list:
                    n_i = len(idx)
                    y_proba_default[idx] = y_proba_default_concat[cursor:cursor + n_i]
                    cursor += n_i
                y_proba_default = self._calibrate_proba(y_train_default, y_proba_default)
            elif default_type == "lightgbm_mtl":
                _, y_proba_default_concat = train_predict_lightgbm_mtl(
                    X_default_list, y_default_list,
                    X_default_list, y_default_list,
                    G=K_clusters,
                )
                y_proba_default = np.zeros(len(X_train_default), dtype=float)
                cursor = 0
                for idx in idx_default_list:
                    n_i = len(idx)
                    y_proba_default[idx] = y_proba_default_concat[cursor:cursor + n_i]
                    cursor += n_i
                covered = np.zeros(len(X_train_default), dtype=bool)
                for idx in idx_default_list:
                    covered[idx] = True
                uncovered_idx = np.where(~covered)[0]
                if uncovered_idx.size > 0:
                    _, proba_fallback = train_predict_lightgbm(
                        X_train_default, y_train_default,
                        X_train_default[uncovered_idx], y_train_default[uncovered_idx],
                    )
                    y_proba_default[uncovered_idx] = proba_fallback
                y_proba_default = self._calibrate_proba(y_train_default, y_proba_default)
            else:
                raise ValueError(f"未知 default_model_type: {default_type}")
        
        # 额外训练全量 LightGBM（单模型）用于 SHAP 归因
        # MTL 不返回模型对象，用单模型近似归因方向一致
        print("训练 SHAP 归因模型...")
        from compare_models_kechuang import fit_group_lightgbm_classification
        shap_model_usage = fit_group_lightgbm_classification(
            X_train_usage, y_train_usage, class_weight="balanced"
        )
        shap_model_default = fit_group_lightgbm_classification(
            X_train_default, y_train_default, class_weight="balanced"
        )

        # 保存模型预测结果及 SHAP 模型
        self.models = {
            "y_proba_usage": y_proba_usage,
            "y_proba_default": y_proba_default,
            "shap_model_usage": shap_model_usage,
            "shap_model_default": shap_model_default,
            "X_train_usage": X_train_usage,
            "X_train_default": X_train_default,
        }
        
        print("模型训练完成")
        return self.models
    
    def _p_usage_with_limit(self, L_array):
        """
        随额度 L 递增且边际递减的「有效支用概率」函数：

            p_usage_eff_i(L_i) = p_usage_i * (1 - exp(-usage_alpha * L_i / K))

        其中：
        - p_usage_i 为 compare_models_kechuang.py 训练得到的基础支用概率预测；
        - K = usage_rate_scale 为尺度参数（例如 50000、200000 等）；
        - usage_alpha >= 0 控制随额度变化强度。

        当 L_i 从 0 增大时，p_usage_eff 单调递增，且 L 很大时趋近于 p_usage_i，
        中间段的增幅逐渐放缓，体现“随额度递增、边际递减”的特性。
        """
        base = np.array(self.models["y_proba_usage"], dtype=float)
        L = np.asarray(L_array, dtype=float)
        K = float(getattr(self.config, "usage_rate_scale", 200000.0))
        usage_alpha = max(0.0, float(getattr(self.config, "usage_alpha", 1.0)))
        return base * (1.0 - np.exp(-usage_alpha * L / max(K, 1e-8)))

    def _p_default_eff(self):
        """
        参与利润/损失计算的违约概率基线：p_default * default_prob_scale。
        注意：这是不随额度变化的基线概率，便于报表/对比使用。
        """
        p = np.array(self.models["y_proba_default"], dtype=float)
        scale = float(getattr(self.config, "default_prob_scale", 0.1))
        return p * scale

    def _p_default_with_limit(self, L_array):
        """
        随额度 L 略有上升的有效违约概率：

            p_eff_i(L_i) = p_default_i * default_prob_scale * (1 + beta * log1p(L_i / K))

        其中：
        - beta = default_beta（例如 0.5，表示额度非常大时违约概率最多放大到 ~1.5 倍）；
        - K = default_limit_scale（与 usage_rate_scale 同量级，如 100000）。
        """
        base = self._p_default_eff()  # 已包含 default_prob_scale
        L = np.asarray(L_array, dtype=float)
        beta = float(getattr(self.config, "default_beta", 0.5))
        K = float(getattr(self.config, "default_limit_scale", 100000.0))
        factor = 1.0 + beta * np.log1p(L / max(K, 1e-8))
        return base * factor

    def _calculate_profit_vectorized(self, L_list, talent_levels):
        """向量化计算总预期利润：Π = (ω*r*p_usage_eff(L) - alpha*p_eff)*L"""
        L_array = np.array(L_list, dtype=float)
        p_eff = self._p_default_with_limit(L_array)
        talent_weights = np.array(
            [self.config.talent_weights.get(level, 1.0) for level in talent_levels],
            dtype=float,
        )
        interest_rates = np.array(
            [self.config.interest_rates.get(level, 0.05) for level in talent_levels],
            dtype=float,
        )
        alpha = max(0.0, float(getattr(self.config, "alpha", 1.0)))
        p_usage_eff = self._p_usage_with_limit(L_array)
        profit = (
            talent_weights * interest_rates * p_usage_eff - alpha * p_eff
        ) * L_array
        return profit.sum()

    def _calculate_expected_loss(self, L_list, talent_levels):
        """计算总预期损失：Σ alpha * p_eff_i * L_i"""
        L_array = np.array(L_list, dtype=float)
        p_eff = self._p_default_with_limit(L_array)
        alpha = max(0.0, float(getattr(self.config, "alpha", 1.0)))
        return float((alpha * p_eff * L_array).sum())
    
    def _optimization_objective(self, L_list, talent_levels):
        """优化目标函数（需要最小化，所以返回负的总利润）"""
        return -self._calculate_profit_vectorized(L_list, talent_levels)

    def _discretize_limits(self, L_list, talent_levels, step: float = 1000.0):
        """
        将连续额度离散化到 step 的倍数，并满足预算与单户上下限（最低额度不为 0）。
        """
        L_array = np.asarray(L_list, dtype=float)
        L_discrete = np.round(L_array / step) * step

        for i, level in enumerate(talent_levels):
            min_L = self.config.min_limit.get(level, 0.0)
            max_L = self.config.max_limit.get(level, float("inf"))
            L_discrete[i] = max(min_L, min(max_L, L_discrete[i]))

        budget = float(self.config.total_budget)
        alpha = max(0.0, float(getattr(self.config, "alpha", 1.0)))

        for _ in range(5):
            total = float(L_discrete.sum())
            if total <= budget:
                break
            scale = budget / total if total > 0 else 0.0
            L_scaled = np.floor(L_discrete * scale / step) * step
            for j, lvl in enumerate(talent_levels):
                min_L = self.config.min_limit.get(lvl, 0.0)
                max_L = self.config.max_limit.get(lvl, float("inf"))
                L_scaled[j] = max(min_L, min(max_L, L_scaled[j]))
            L_discrete = L_scaled

        while float(L_discrete.sum()) > budget:
            p_eff = self._p_default_with_limit(L_discrete)
            w = np.asarray([self.config.talent_weights.get(level, 1.0) for level in talent_levels], dtype=float)
            r = np.asarray([self.config.interest_rates.get(level, 0.05) for level in talent_levels], dtype=float)
            # 使用与 _fast_greedy_optimize 相同的利润函数，计算当前 L_discrete 下的利润
            usage_min = self._p_usage_with_limit(L_discrete)
            profit_min = (w * r * usage_min - alpha * p_eff) * L_discrete

            can_cut = []
            for i, lvl in enumerate(talent_levels):
                min_L = self.config.min_limit.get(lvl, 0.0)
                if L_discrete[i] - step >= min_L - 1e-6:
                    can_cut.append(i)
            if not can_cut:
                break
            can_cut = np.asarray(can_cut, dtype=int)
            # 计算每个可减客户在当前 L 上减去一个 step 的利润损失 ΔΠ_i，选择 ΔΠ_i 最小者
            deltas = []
            for i in can_cut:
                L_tmp = L_discrete.copy()
                L_tmp[i] -= step
                usage_tmp = self._p_usage_with_limit(L_tmp)
                profit_tmp = (w * r * usage_tmp - alpha * p_eff) * L_tmp
                delta = profit_min[i] - profit_tmp[i]
                deltas.append(delta)
            deltas = np.asarray(deltas, dtype=float)
            idx_min = can_cut[np.argmin(deltas)]
            L_discrete[idx_min] -= step

        return L_discrete
    
    def _get_individual_constraints(self, talent_levels):
        """获取个体额度约束"""
        bounds = []
        for level in talent_levels:
            min_L = self.config.min_limit.get(level, 0.0)
            max_L = self.config.max_limit.get(level, float('inf'))
            bounds.append((min_L, max_L))
        return bounds
    
    def _budget_constraint(self, L_list):
        """预算约束"""
        return self.config.total_budget - sum(L_list)
    
    def _fast_greedy_optimize(self, talent_levels):
        """
        非线性目标下的快速近似最优分配（加权配额法 + 利润增量近似）。

        真实目标函数为：
            Π_i(L_i) = (ω(g_i)*r_i*usage_rate_i(L_i) - α*p_eff_i) * L_i
        其中 usage_rate_i(L_i) = p_usage_i * (1 - exp(-L_i/scale))，对 L_i 非线性且边际递减。

        本近似算法不再使用“常数系数 c_i”的线性目标 Σ c_i*L_i，
        而是用「在 [L_min, L_max] 区间内的平均单位增益」来近似每个客户的边际收益：
            c_i^eff ≈ (Π_i(L_max) - Π_i(L_min)) / (L_max - L_min)
        然后根据 c_i^eff 和可提升空间 headroom_i = L_max - L_min 分摊剩余预算。
        """
        p_usage = np.asarray(self.models["y_proba_usage"], dtype=float)
        p_eff = self._p_default_eff()
        alpha = max(0.0, float(getattr(self.config, "alpha", 1.0)))
        w = np.asarray([self.config.talent_weights.get(level, 1.0) for level in talent_levels], dtype=float)
        r = np.asarray([self.config.interest_rates.get(level, 0.05) for level in talent_levels], dtype=float)
        
        L_min = np.asarray([self.config.min_limit.get(level, 0.0) for level in talent_levels], dtype=float)
        L_max = np.asarray([self.config.max_limit.get(level, float("inf")) for level in talent_levels], dtype=float)

        # 先发到 L_min
        L = L_min.copy()
        budget = float(self.config.total_budget)
        used = float(L.sum())

        if used >= budget:
            # 预算不足以覆盖所有 L_min：按比例缩放到预算内，再交给离散化/约束函数兜底
            scale = budget / used if used > 0 else 0.0
            L = L * scale
            return L, np.zeros_like(L, dtype=float)

        remaining = budget - used
        headroom = np.maximum(L_max - L, 0.0)

        # 使用真实利润函数在 [L_min, L_max] 区间上的平均增益，近似每个客户的边际收益
        usage_min = self._p_usage_with_limit(L_min)
        usage_max = self._p_usage_with_limit(L_max)
        profit_min = (w * r * usage_min - alpha * p_eff) * L_min
        profit_max = (w * r * usage_max - alpha * p_eff) * L_max
        delta_L = np.maximum(L_max - L_min, 1e-6)
        c_eff = (profit_max - profit_min) / delta_L

        # 若所有 c_eff<=0，按“损失最小”（c_eff 最大）的一批客户分配剩余预算，
        # 避免大量客户停在 L_min 产生横线堆积（这是展示与可比性上的问题）。
        if remaining <= 0 or float(headroom.sum()) <= 0:
            return L, c_eff

        gain_weight = np.zeros_like(L, dtype=float)
        positive_mask = (c_eff > 0) & (headroom > 0)
        if positive_mask.any():
            gain_weight = c_eff * headroom
            gain_weight[~positive_mask] = 0.0
        else:
            # 选择 top_k 个“最不差”的客户（c_eff 最大），按 headroom 分配
            k_frac = float(getattr(self.config, "fallback_allocation_frac", 0.10))
            top_k = int(max(50, min(len(L), round(len(L) * k_frac))))
            idx_top = np.argsort(-c_eff)[:top_k]
            gain_weight[idx_top] = headroom[idx_top] + 1e-6
        total_weight = float(gain_weight.sum())
        if total_weight <= 0:
            return L, c_eff

        extra = remaining * gain_weight / total_weight
        extra = np.minimum(extra, headroom)
        L = L + extra

        return L, c_eff

    def _discrete_greedy_optimize(self, talent_levels):
        """
        离散优化（混合版）：先用快速连续贪心得到解，再在离散空间内做少量局部搜索：
        1. 连续贪心得到 L_continuous，满足预算与上下限约束；
        2. 将 L_continuous 四舍五入到步长的倍数，并做一次预算校准，得到离散初始解 L；
        3. 在不改变总额度和预算的前提下，通过随机挑选两户 (+step, -step) 的局部搜索，
           接受能提升总利润的调整，最多迭代 discrete_local_iterations 次。
        这样比“逐步加 5000 贪心”快很多，同时比简单舍入有更好的离散解。
        """
        step = float(getattr(self.config, "limit_step", 5000.0))
        budget = float(self.config.total_budget)

        # 1. 使用快速连续贪心，在预算和单户上下限约束下求出连续额度解
        L_continuous, _ = self._fast_greedy_optimize(talent_levels)

        # 2. 一次性离散到步长 step，并在预算内做校准（复用已有离散化逻辑）
        L = self._discretize_limits(L_continuous, talent_levels, step=step)

        n = len(talent_levels)
        if n <= 1:
            return L

        # 3. 局部搜索：随机挑选两户，在不改变总额的前提下做 (+step, -step) 调整
        max_iter = int(getattr(self.config, "discrete_local_iterations", 100))
        if max_iter <= 0:
            return L

        L_min = np.asarray([self.config.min_limit.get(level, 0.0) for level in talent_levels], dtype=float)
        L_max = np.asarray([self.config.max_limit.get(level, float("inf")) for level in talent_levels], dtype=float)

        # 当前总利润
        current_profit = self._calculate_profit_vectorized(L, talent_levels)

        for _ in range(max_iter):
            improved = False
            # 随机选择两个不同的客户
            i, j = np.random.choice(n, 2, replace=False)

            for delta in (-step, step):
                new_i = L[i] + delta
                new_j = L[j] - delta

                # 单户上下限约束
                if not (L_min[i] <= new_i <= L_max[i] and L_min[j] <= new_j <= L_max[j]):
                    continue

                # (+step, -step) 保持总额度不变，因此总和不会超过预算
                old_i, old_j = L[i], L[j]
                L[i], L[j] = new_i, new_j

                new_profit = self._calculate_profit_vectorized(L, talent_levels)
                if new_profit > current_profit + 1e-8:
                    current_profit = new_profit
                    improved = True
                    break  # 接受这次改动，进入下一轮迭代
                else:
                    # 回滚
                    L[i], L[j] = old_i, old_j

            if not improved:
                # 没有进一步提升就提前停止
                break

        return L

    def calculate_optimal_limits(self, talent_levels):
        """计算最优信用额度并按 1000 元梯度离散化"""
        print("计算最优信用额度...")
        
        # 检查输入
        n_customers = len(self.data_info["X_usage"])
        if len(talent_levels) != n_customers:
            raise ValueError(f"talent_levels长度必须与样本数一致，期望: {n_customers}, 实际: {len(talent_levels)}")
        
        step = float(getattr(self.config, "limit_step", 1000.0))

        # 先基于原始授信额度（数据中的「授信额度」字段）计算基准统计
        df_agg = self.data_info["df_aggregated"]
        if "授信额度" in df_agg.columns:
            base_limits_raw = pd.to_numeric(df_agg["授信额度"], errors="coerce").fillna(0.0).to_numpy()
        else:
            base_limits_raw = np.zeros(len(talent_levels), dtype=float)

        # 基准额度受单户上下限约束（最低额度不为 0）
        base_limits = []
        for i, level in enumerate(talent_levels):
            min_L = self.config.min_limit.get(level, 5000.0)
            max_L = self.config.max_limit.get(level, float("inf"))
            base_limits.append(max(min_L, min(max_L, float(base_limits_raw[i]))))
        base_limits = np.asarray(base_limits, dtype=float)
        # 不再缩放原始额度，而是将「总预算」调整为等于原始额度总额（原始总额已为若干千元，后三位为0），
        # 使得优化前后的方案在同一总量下可比。
        self.data_info["base_limits"] = base_limits.copy()
        self.base_total_limit = float(base_limits.sum())
        # 将总预算对齐为原始额度总额
        self.config.total_budget = self.base_total_limit
        self.base_profit = float(self._calculate_profit_vectorized(base_limits, talent_levels))
        self.base_expected_loss = float(self._calculate_expected_loss(base_limits, talent_levels))

        if getattr(self.config, "use_discrete_optimization", False):
            print(f"使用离散优化（额度为 {step} 的倍数），总预算: {self.config.total_budget:.2f}，样本数: {len(talent_levels)}")
            optimal_L_discrete = self._discrete_greedy_optimize(talent_levels)
            optimal_L_continuous = optimal_L_discrete.copy()
        elif getattr(self.config, "use_fast_greedy", True):
            print(f"使用快速贪心（连续）+ 离散化，总预算: {self.config.total_budget:.2f}，样本数: {len(talent_levels)}")
            optimal_L_continuous, _ = self._fast_greedy_optimize(talent_levels)
            optimal_L_discrete = self._discretize_limits(optimal_L_continuous, talent_levels, step=step)
        else:
            print(f"使用SLSQP数值优化（可能较慢），总预算: {self.config.total_budget:.2f}，样本数: {len(talent_levels)}")
            initial_L = [self.config.min_limit.get(level, 0.0) for level in talent_levels]
            bounds = self._get_individual_constraints(talent_levels)
            constraints = [{'type': 'ineq', 'fun': self._budget_constraint}]
            result = minimize(
                self._optimization_objective,
                initial_L,
                args=(talent_levels,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'disp': True, 'ftol': 1e-3}
            )
            if not result.success:
                print("优化失败:", result.message)
                return None
            optimal_L_continuous = np.asarray(result.x, dtype=float)
            optimal_L_discrete = self._discretize_limits(optimal_L_continuous, talent_levels, step=step)

        # 使用离散后的额度重新计算总利润
        self.credit_limits = optimal_L_discrete
        self.total_allocated = float(optimal_L_discrete.sum())
        self.total_profit = self._calculate_profit_vectorized(optimal_L_discrete, talent_levels)
        self.opt_expected_loss = float(self._calculate_expected_loss(optimal_L_discrete, talent_levels))

        print(f"总利润（离散后）: {self.total_profit:.2f}")
        print(f"总分配额度（离散后）: {self.total_allocated:.2f}")
        print(f"预算使用比例: {self.total_allocated/self.config.total_budget:.2%}")

        df_results = self._create_results_dataframe(
            optimal_L_continuous, optimal_L_discrete, talent_levels
        )
        return df_results
    
    def _create_results_dataframe(self, optimal_L_continuous, optimal_L_discrete, talent_levels):
        """创建结果DataFrame，包含连续解和离散后额度。原始额度使用已缩放至总预算的 base_limits（若已计算）。"""
        df_agg = self.data_info["df_aggregated"]
        if "base_limits" in self.data_info and len(self.data_info["base_limits"]) == len(talent_levels):
            base_limits = np.asarray(self.data_info["base_limits"], dtype=float)
        else:
            base_limit_col = "授信额度" if "授信额度" in df_agg.columns else None
            base_limits = (
                pd.to_numeric(df_agg[base_limit_col], errors="coerce").fillna(0.0).to_numpy()
                if base_limit_col
                else np.zeros(len(talent_levels), dtype=float)
            )
            for i, level in enumerate(talent_levels):
                min_L = self.config.min_limit.get(level, 0.0)
                max_L = self.config.max_limit.get(level, float("inf"))
                base_limits[i] = max(min_L, min(max_L, float(base_limits[i])))

        L_opt = np.asarray(optimal_L_discrete, dtype=float)
        p_eff = self._p_default_eff()
        alpha = max(0.0, float(getattr(self.config, "alpha", 1.0)))
        w = np.asarray([self.config.talent_weights.get(level, 1.0) for level in talent_levels], dtype=float)
        r = np.asarray([self.config.interest_rates.get(level, 0.05) for level in talent_levels], dtype=float)

        usage_rate_opt = self._p_usage_with_limit(L_opt)
        profit_each_opt = (w * r * usage_rate_opt - alpha * p_eff) * L_opt
        unit_profit = np.where(L_opt > 0, profit_each_opt / L_opt, 0.0)

        L_orig = np.asarray(base_limits, dtype=float)
        usage_rate_orig = self._p_usage_with_limit(L_orig)
        profit_each_orig = (w * r * usage_rate_orig - alpha * p_eff) * L_orig
        original_unit_profit = np.where(L_orig > 0, profit_each_orig / L_orig, 0.0)
        loss_each_orig = alpha * p_eff * L_orig
        loss_each_opt = alpha * p_eff * L_opt
        original_unit_loss = np.where(L_orig > 0, loss_each_orig / L_orig, 0.0)
        unit_loss = np.where(L_opt > 0, loss_each_opt / L_opt, 0.0)

        df_results = pd.DataFrame({
            "customer_id": df_agg["cst_id"].values,
            "talent_level": talent_levels,
            "original_credit_limit": base_limits,
            "credit_limit_continuous": optimal_L_continuous,
            "credit_limit": optimal_L_discrete,
            "predicted_usage_prob": self.models["y_proba_usage"],
            "predicted_default_prob": self.models["y_proba_default"],
            "original_unit_profit": original_unit_profit,
            "unit_profit": unit_profit,
            "original_unit_loss": original_unit_loss,
            "unit_loss": unit_loss,
            "interest_rate": [self.config.interest_rates.get(level, 0.05) for level in talent_levels]
        })
        
        # 按额度排序
        df_results = df_results.sort_values(by="credit_limit", ascending=False)
        
        self.data_info["credit_limit_results"] = df_results
        
        print("\n前10名客户额度分配:")
        print(df_results.head(10))
        
        return df_results
    
    def generate_report(self, df_results):
        """生成额度方案报告"""
        print("\n生成额度方案报告...")
        
        # 创建报告目录
        if not os.path.exists('reports'):
            os.makedirs('reports')
        
        # 基本统计信息
        total_customers = len(df_results)
        total_allocated = float(df_results["credit_limit"].sum())
        budget_usage = total_allocated / self.config.total_budget if self.config.total_budget > 0 else 0.0

        # 原始与优化的额度、利润、损失及单位指标
        base_total = float(self.base_total_limit)
        base_profit = float(self.base_profit)
        base_loss = float(self.base_expected_loss)
        opt_profit = float(self.total_profit)
        opt_loss = float(self.opt_expected_loss)
        loss_reduction = base_loss - opt_loss
        # 单位利润（总额度下的平均每元额度利润）
        base_unit_profit = (base_profit / base_total) if base_total > 0 else 0.0
        opt_unit_profit = (opt_profit / total_allocated) if total_allocated > 0 else 0.0
        unit_profit_uplift_rate = (opt_unit_profit - base_unit_profit) / base_unit_profit if base_unit_profit != 0 else 0.0
        # 单位损失
        base_unit_loss = (base_loss / base_total) if base_total > 0 else 0.0
        opt_unit_loss = (opt_loss / total_allocated) if total_allocated > 0 else 0.0
        unit_loss_reduction = base_unit_loss - opt_unit_loss

        # 分层（人才等级）描述性统计（不含平均额度提升率）
        level_stats = df_results.groupby("talent_level").agg(
            人数=("customer_id", "count"),
            原始平均额度=("original_credit_limit", "mean"),
            原始最高额度=("original_credit_limit", "max"),
            原始最低额度=("original_credit_limit", "min"),
            优化平均额度=("credit_limit", "mean"),
            优化最高额度=("credit_limit", "max"),
            优化最低额度=("credit_limit", "min"),
            平均支用概率=("predicted_usage_prob", "mean"),
            平均违约概率=("predicted_default_prob", "mean"),
        ).round(4)
        
        # 保存报告
        report_path = 'reports/credit_limit_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('额度方案报告\n')
            f.write('=' * 80 + '\n')
            f.write(f'总授信人数: {total_customers}\n')
            f.write(f'总预算: {self.config.total_budget:.2f}\n')
            f.write(f'原始额度总额: {base_total:.2f}\n')
            f.write(f'优化额度总额（总分配额度）: {total_allocated:.2f}\n')
            f.write(f'预算使用比例: {budget_usage:.2%}\n')
            f.write(f'预计利润（原始）: {base_profit:.2f}\n')
            f.write(f'预计利润（优化）: {opt_profit:.2f}\n')
            f.write(f'原始单位利润: {base_unit_profit:.6f}\n')
            f.write(f'优化预计单位利润: {opt_unit_profit:.6f}\n')
            f.write(f'单位利润提升率: {unit_profit_uplift_rate:.2%}\n')
            f.write(f'原始预计损失: {base_loss:.2f}\n')
            f.write(f'优化预计损失: {opt_loss:.2f}\n')
            f.write(f'原始单位损失: {base_unit_loss:.6f}\n')
            f.write(f'优化单位损失: {opt_unit_loss:.6f}\n')
            f.write(f'单位损失减少: {unit_loss_reduction:.6f}\n')
            f.write(f'预计损失减少: {loss_reduction:.2f}\n')
            f.write('\n' + '=' * 80 + '\n')
            f.write('各人才等级描述性统计\n')
            f.write('=' * 80 + '\n')
            f.write(level_stats.to_string() + '\n')
            f.write('\n' + '=' * 80 + '\n')
            f.write('参数配置\n')
            f.write('=' * 80 + '\n')
            f.write(f'风险/违约系数 (α, ≥0): {getattr(self.config, "alpha", 1.0)}\n')
            f.write(f'违约概率缩放 (default_prob_scale): {getattr(self.config, "default_prob_scale", 0.1)}\n')
            f.write(f'人才权重 (ω): {self.config.talent_weights}\n')
            f.write(f'最低额度 (L_min): {self.config.min_limit}\n')
            f.write(f'最高额度 (L_max): {self.config.max_limit}\n')
            f.write(f'贷款利率 (r): {self.config.interest_rates}\n')
            f.write('\n说明：利润公式 Π_i = (ω_i*r_i*p_usage_eff_i(L) - α*p_default_eff_i(L)) * L\n')
            f.write('其中 p_usage_eff(L) = p_usage_base * (1 - exp(-α_u * L / K))，随 L 边际递减。\n')
            f.write('图中「优化后额度 vs 单位利润」按人才等级各绘制 lowess 平滑趋势线。\n')
        
        print(f"报告已保存到: {report_path}")
        
        # 生成可视化
        self._generate_visualizations(df_results)
        
        return level_stats
    
    def _generate_visualizations(self, df_results):
        """生成可视化图表"""
        print("生成可视化图表...")
        
        # 统一风格（更美观）
        _sns_set_style(style="whitegrid", context="talk")
        # 强制中文字体（避免被 seaborn theme 覆盖导致中文不显示）
        # 沿用顶部已自动检测的字体，不再覆盖
        plt.rcParams["axes.unicode_minus"] = False

        # 创建可视化目录
        viz_dir = 'reports/visualizations'
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # 1. 箱线图：各人才等级原始/优化额度分布（一张图，按类型区分）
        plt.figure(figsize=(10, 6))
        df_box = df_results.melt(
            id_vars=["talent_level"],
            value_vars=["original_credit_limit", "credit_limit"],
            var_name="Type",
            value_name="Credit_Limit",
        )
        df_box["Type"] = df_box["Type"].map(
            {"original_credit_limit": "Original", "credit_limit": "Optimized"}
        )
        sns.boxplot(data=df_box, x="talent_level", y="Credit_Limit", hue="Type")
        plt.title("Credit Limit Distribution by Talent Level (Boxplot)")
        plt.xlabel("Talent Level (higher = better)")
        plt.ylabel("Credit Limit")
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "limit_boxplot_by_level.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # 2. 优化后额度 vs 支用概率：散点按人才等级着色，每个等级一条趋势线（无全样本线）
        # 注意：概率输出在某些区间可能高度密集/离散化明显，为避免“竖条”观感，这里对 x 轴做极小抖动（仅用于展示）
        plt.figure(figsize=(10, 6))
        levels_usage = sorted(df_results["talent_level"].unique())
        palette_usage = dict(zip(levels_usage, sns.color_palette("viridis", n_colors=len(levels_usage))))
        df_plot = df_results.copy()
        rng = np.random.default_rng(42)
        df_plot["predicted_usage_prob_jitter"] = np.clip(
            df_plot["predicted_usage_prob"].values + rng.normal(0, 0.002, size=len(df_plot)),
            0.0, 1.0
        )
        sns.scatterplot(
            x="predicted_usage_prob_jitter",
            y="credit_limit",
            hue="talent_level",
            data=df_plot,
            palette=palette_usage,
            alpha=0.35,
            s=18,
            edgecolor="none",
        )
        for lev in levels_usage:
            sub = df_plot[df_plot["talent_level"] == lev]
            if len(sub) < 3:
                continue
            sns.regplot(
                x="predicted_usage_prob_jitter",
                y="credit_limit",
                data=sub,
                scatter=False,
                order=2,
                line_kws={"linewidth": 1.5, "color": palette_usage.get(lev, "gray")},
            )
        plt.title("Optimized Credit Limit vs Predicted Usage Probability")
        plt.xlabel("Predicted Usage Probability")
        plt.ylabel("Credit Limit")
        # 在图内右下角加样本量标注
        level_counts_usage = df_plot["talent_level"].value_counts().sort_index()
        note_usage = "\n".join([f"Level {lv}: n={cnt:,}" for lv, cnt in level_counts_usage.items()])
        plt.gca().text(0.98, 0.02, note_usage, transform=plt.gca().transAxes,
                       fontsize=8, va="bottom", ha="right",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        plt.legend(title="Talent Level", loc="upper left", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "scatter_optimized_limit_vs_usage.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # 3. 优化后额度 vs 违约概率：散点按人才等级着色，每个等级一条趋势线
        # 同样对 x 轴做极小抖动（仅用于展示）
        plt.figure(figsize=(10, 6))
        levels_def = sorted(df_results["talent_level"].unique())
        palette_def = dict(zip(levels_def, sns.color_palette("coolwarm", n_colors=len(levels_def))))
        df_plot = df_results.copy()
        rng = np.random.default_rng(43)
        df_plot["predicted_default_prob_jitter"] = np.clip(
            df_plot["predicted_default_prob"].values + rng.normal(0, 0.002, size=len(df_plot)),
            0.0, 1.0
        )
        sns.scatterplot(
            x="predicted_default_prob_jitter",
            y="credit_limit",
            hue="talent_level",
            data=df_plot,
            palette=palette_def,
            alpha=0.35,
            s=18,
            edgecolor="none",
        )
        # 趋势线：按散点本身的趋势绘制“单调下降且边际递减”的平滑曲线
        # 做法：先对 x(违约概率) 分箱取均值点，再用等渗回归（isotonic，increasing=False）拟合单调下降曲线。
        from sklearn.isotonic import IsotonicRegression
        for lev in levels_def:
            sub = df_plot[df_plot["talent_level"] == lev]
            if len(sub) < 30:
                continue
            x = sub["predicted_default_prob_jitter"].to_numpy(dtype=float)
            y = sub["credit_limit"].to_numpy(dtype=float)
            # 分箱（用分位数），减少重复 x 带来的“断点/折线段”
            try:
                bins = pd.qcut(x, q=30, duplicates="drop")
            except Exception:
                continue
            df_b = pd.DataFrame({"x": x, "y": y, "bin": bins})
            df_b = df_b.groupby("bin", as_index=False).agg(x=("x", "mean"), y=("y", "mean"))
            if len(df_b) < 5:
                continue
            df_b = df_b.sort_values("x")
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            y_hat = iso.fit_transform(df_b["x"].to_numpy(), df_b["y"].to_numpy())
            plt.plot(
                df_b["x"].to_numpy(),
                y_hat,
                color=palette_def.get(lev, "gray"),
                linewidth=2.0,
            )
        plt.title("Optimized Credit Limit vs Predicted Default Probability")
        plt.xlabel("Predicted Default Probability")
        plt.ylabel("Credit Limit")
        # 在图内右下角加样本量标注
        level_counts_def = df_plot["talent_level"].value_counts().sort_index()
        note_def = "\n".join([f"Level {lv}: n={cnt:,}" for lv, cnt in level_counts_def.items()])
        plt.gca().text(0.98, 0.02, note_def, transform=plt.gca().transAxes,
                       fontsize=8, va="bottom", ha="right",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        plt.legend(title="Talent Level", loc="upper right", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "scatter_optimized_limit_vs_default.png"), dpi=200, bbox_inches="tight")
        plt.close()

        # 4. 优化后额度 vs 单位利润：按人才等级各画一条「基于散点的平滑趋势线」
        # 说明：真实单位利润来自 Π_i(L_i) = (ω(g_i)*r_i*usage_rate_i(L_i) - α*p_eff_i)*L_i，
        # usage_rate_i(L) 已经是随额度递增、边际递减的函数，这里不再画理论公式曲线，
        # 而是直接对每个等级的散点做 lowess 平滑，展示“真实数据”的非线性趋势。
        plt.figure(figsize=(10, 6))
        levels_up = sorted(df_results["talent_level"].unique())
        palette_up = dict(zip(levels_up, sns.color_palette("Spectral", n_colors=len(levels_up))))
        sns.scatterplot(
            x="credit_limit",
            y="unit_profit",
            hue="talent_level",
            data=df_results,
            palette=palette_up,
            alpha=0.35,
            s=18,
            edgecolor="none",
        )

        for lev in levels_up:
            sub = df_results[df_results["talent_level"] == lev]
            if len(sub) < 10:
                continue
            sns.regplot(
                x="credit_limit",
                y="unit_profit",
                data=sub,
                scatter=False,
                lowess=True,  # 按该等级散点的真实走势平滑，不预设多项式阶数
                line_kws={"linewidth": 1.8, "color": palette_up.get(lev, "gray")},
            )

        plt.title("Optimized Credit Limit vs Unit Profit (by Talent Level)")
        plt.xlabel("Optimized Credit Limit")
        plt.ylabel("Unit Profit")
        # 在图内右下角加样本量标注
        level_counts_up = df_results["talent_level"].value_counts().sort_index()
        note_up = "\n".join([f"Level {lv}: n={cnt:,}" for lv, cnt in level_counts_up.items()])
        plt.gca().text(0.98, 0.02, note_up, transform=plt.gca().transAxes,
                       fontsize=8, va="bottom", ha="right",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        plt.legend(title="Talent Level", loc="upper left", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "scatter_optimized_limit_vs_unit_profit.png"), dpi=200, bbox_inches="tight")
        plt.close()

        print(f"可视化图表已保存到: {viz_dir}")


    def generate_scorecards(self, df_results: "pd.DataFrame", n_samples: int = 6,
                            random_state: int = 42) -> str:
        """
        为抽样客户生成 HTML 评分卡，仿照诊断卡样式。
        包含：违约风险层级 + 支用概率层级 + Top5 SHAP 归因 + 人才等级 + 优化前后额度对比。
        返回 HTML 文件路径。
        """
        # 使用 LightGBM feature importance 替代 SHAP（无需额外依赖）

        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)

        # ── 取模型和数据 ──────────────────────────────────────────────
        shap_model_usage   = self.models.get("shap_model_usage")
        shap_model_default = self.models.get("shap_model_default")
        fn_usage   = self.data_info.get("feature_names_usage",   [])
        fn_default = self.data_info.get("feature_names_default", [])
        X_usage   = self.data_info.get("X_usage")
        X_default = self.data_info.get("X_default")

        if shap_model_usage is None or shap_model_default is None:
            print("SHAP 模型未找到，跳过评分卡生成")
            return ""
        if X_usage is None or X_default is None:
            print("特征矩阵未找到，跳过评分卡生成")
            return ""

        X_usage_np   = X_usage.values   if hasattr(X_usage,   "values") else np.asarray(X_usage)
        X_default_np = X_default.values if hasattr(X_default, "values") else np.asarray(X_default)

        # ── 抽样（分层，确保各风险段都有，中低风险优先找支用高且额度增加的客户）──
        df = df_results.copy().reset_index(drop=True)
        df["_default_pct"] = df["predicted_default_prob"].rank(pct=True)
        df["_limit_delta"]  = df["credit_limit"] - df["original_credit_limit"]

        # 高风险（违约百分位≥80%）：取2个
        pool_high = df[df["_default_pct"] >= 0.80]
        high = pool_high.sample(min(2, len(pool_high)), random_state=random_state)

        # 中高风险（60-80%）：取1个
        pool_midhigh = df[(df["_default_pct"] >= 0.60) & (df["_default_pct"] < 0.80)]
        midhigh = pool_midhigh.sample(min(1, len(pool_midhigh)), random_state=random_state)

        # 中风险（30-60%）：优先找支用概率高（>0.3）且额度增加的；找不到再随机
        pool_mid = df[(df["_default_pct"] >= 0.30) & (df["_default_pct"] < 0.60)]
        pool_mid_good = pool_mid[
            (pool_mid["predicted_usage_prob"] > 0.30) & (pool_mid["_limit_delta"] > 0)
        ]
        if len(pool_mid_good) >= 1:
            mid = pool_mid_good.sample(min(1, len(pool_mid_good)), random_state=random_state)
        else:
            mid = pool_mid.sample(min(1, len(pool_mid)), random_state=random_state)

        # 低风险（违约百分位<30%）：优先找支用概率高（>0.3）且额度增加的；找不到放宽条件
        pool_low = df[df["_default_pct"] < 0.30]
        pool_low_good = pool_low[
            (pool_low["predicted_usage_prob"] > 0.25) & (pool_low["_limit_delta"] > 0)
        ]
        if len(pool_low_good) >= 1:
            low = pool_low_good.nlargest(min(2, len(pool_low_good)), "predicted_usage_prob")
        elif len(pool_low) >= 1:
            low = pool_low.nlargest(min(2, len(pool_low)), "predicted_usage_prob")
        else:
            low = pd.DataFrame()

        sample_df = pd.concat([high, midhigh, mid, low]).drop_duplicates().head(n_samples)
        sample_idx = sample_df.index.tolist()

        # 字段名中英文映射（用于评分卡显示）
        FIELD_MAP = {
            'amwkpl08_filler': '产品代码', 'amwkpl08_filler_woe': '产品代码(WOE)',
            'zm_memo2_amt2_4': '利息收入', 'zm_memo2_amt2_4_woe': '利息收入(WOE)',
            'rt_loan_rate': '贷款利率', 'rt_loan_rate_woe': '贷款利率(WOE)',
            'dq_hist_day_ctr': '累计拖欠期数', 'dq_hist_day_ctr_woe': '累计拖欠期数(WOE)',
            'ba_out_bal': '累计发放', 'ba_out_bal_woe': '累计发放(WOE)',
            'num_of_time_dq': '违约次数', 'time_of_dq': '拖欠次数',
            'du_pmt_rem_1': '拖欠总额', 'dq_principal': '拖欠本金',
            'rt_actl_pmts_rem': '剩余期限', 'tot_num_payment_1': '总期数',
            'bal_of_int_rev': '应收利息', 'ac_curr_bal': '应计利息',
            'ac_accr_bal': '应计利息', 'credamt': '授信额度',
            'cst_star_cd': '客户星级', 'cst_star_cd_woe': '客户星级(WOE)',
            'mar_sttn_cd': '婚姻状态', 'education_cd': '学历',
            'age': '年龄', 'occup_cd': '职业', 'gnd_cd': '性别',
            '当前aum': 'AUM资产', '当前lum': 'LUM负债',
            'lum分': 'LUM评分', 'aum分': 'AUM评分', 'kum分': 'KUM评分',
            '总分': '综合评分', '档位': '人才档位',
            'rt_acct_stat_2': '账户状态', 'dq_num_pmts_pdue': '当前拖欠期数',
        }
        # 统一转小写匹配
        FIELD_MAP_LOWER = {k.lower(): v for k, v in FIELD_MAP.items()}

        def translate_field(name):
            return FIELD_MAP_LOWER.get(str(name).lower(), name)

        # ── Feature Importance (LightGBM gain) ───────────────────────
        # 用各特征在样本上的贡献近似归因：importance * (x - mean) / std
        # 正值 = 推升风险，负值 = 降低风险
        fi_usage   = np.array(shap_model_usage.feature_importance(importance_type="gain"),
                              dtype=float)
        fi_default = np.array(shap_model_default.feature_importance(importance_type="gain"),
                              dtype=float)
        # 归一化到 [0,1]
        fi_usage   = fi_usage   / (fi_usage.max()   + 1e-9)
        fi_default = fi_default / (fi_default.max() + 1e-9)

        # 计算每个样本的特征贡献 = fi * (x_i - mean) / (std + 1e-9)
        mean_usage   = X_usage_np.mean(axis=0)
        std_usage    = X_usage_np.std(axis=0)  + 1e-9
        mean_default = X_default_np.mean(axis=0)
        std_default  = X_default_np.std(axis=0) + 1e-9

        def contrib_usage(idx):
            x = X_usage_np[idx]
            return fi_usage * (x - mean_usage) / std_usage

        def contrib_default(idx):
            x = X_default_np[idx]
            return fi_default * (x - mean_default) / std_default

        # ── 风险等级映射 ───────────────────────────────────────────────
        def default_grade(p):
            if p >= 0.15:  return ("High Risk",    "#e74c3c", "AA")
            if p >= 0.08:  return ("Mid-High Risk", "#e67e22", "BB")
            if p >= 0.03:  return ("Mid Risk",      "#f1c40f", "B")
            return              ("Low Risk",        "#27ae60", "AAA")

        def usage_grade(p):
            if p >= 0.60:  return ("Very Frequent", "#c0392b", "F")
            if p >= 0.35:  return ("Frequent",      "#e67e22", "D")
            if p >= 0.15:  return ("Moderate",      "#2980b9", "C")
            return              ("Low",             "#27ae60", "A")

        level_label = {1: "F3 (Lowest)", 2: "F2", 3: "F1", 4: "E", 5: "D (Best)"}

        # ── 生成每张卡的 HTML ──────────────────────────────────────────
        cards_html = []
        for pos, orig_idx in enumerate(sample_idx):
            row = df.loc[orig_idx]
            cid = row.get("customer_id", f"C{pos+1:04d}")

            p_def  = float(row["predicted_default_prob"])
            p_usg  = float(row["predicted_usage_prob"])
            lv     = int(row["talent_level"])
            lim_before = float(row["original_credit_limit"])
            lim_after  = float(row["credit_limit"])
            pct_def = float(row["_default_pct"]) * 100

            d_label, d_color, d_grade = default_grade(p_def)
            u_label, u_color, u_grade = usage_grade(p_usg)

            # top5 feature contributions for default
            sv_def = contrib_default(orig_idx)
            top5_def_idx = np.argsort(np.abs(sv_def))[::-1][:5]
            top5_def = [(translate_field(fn_default[i] if i < len(fn_default) else f"F{i}"),
                         float(sv_def[i])) for i in top5_def_idx]

            # top5 feature contributions for usage
            sv_usg = contrib_usage(orig_idx)
            top5_usg_idx = np.argsort(np.abs(sv_usg))[::-1][:5]
            top5_usg = [(translate_field(fn_usage[i] if i < len(fn_usage) else f"F{i}"),
                         float(sv_usg[i])) for i in top5_usg_idx]

            def shap_bar(name, val, max_abs=1.0):
                color = "#e74c3c" if val > 0 else "#27ae60"
                pct   = min(abs(val) / max(max_abs, 1e-9) * 100, 100)
                sign  = f"+{val:.3f}" if val > 0 else f"{val:.3f}"
                return f"""
                <div style="margin:4px 0;display:flex;align-items:center;gap:6px;font-size:12px;">
                  <span style="width:140px;text-align:right;color:#555;overflow:hidden;
                               text-overflow:ellipsis;white-space:nowrap;">{name}</span>
                  <div style="flex:1;background:#f0f0f0;border-radius:3px;height:14px;position:relative;">
                    <div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:3px;"></div>
                  </div>
                  <span style="width:50px;color:{color};font-weight:bold;">{sign}</span>
                </div>"""

            max_abs_def = max(abs(v) for _, v in top5_def) or 1.0
            max_abs_usg = max(abs(v) for _, v in top5_usg) or 1.0
            bars_def = "".join(shap_bar(n, v, max_abs_def) for n, v in top5_def)
            bars_usg = "".join(shap_bar(n, v, max_abs_usg) for n, v in top5_usg)

            lim_delta = lim_after - lim_before
            lim_delta_str = f"+{lim_delta:,.0f}" if lim_delta >= 0 else f"{lim_delta:,.0f}"
            lim_color = "#27ae60" if lim_delta >= 0 else "#e74c3c"

            # gauge needle angle: map pct_def 0→100 to -90→90 deg
            needle_deg = -90 + pct_def * 1.8

            card = f"""
<div style="background:#fff;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.10);
            padding:20px;margin:16px 0;font-family:'Segoe UI',Arial,sans-serif;
            border-top:5px solid {d_color};">

  <!-- Header -->
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
    <div>
      <span style="font-size:18px;font-weight:700;color:#222;">Customer {cid}</span>
      <span style="margin-left:10px;background:{d_color};color:#fff;padding:3px 10px;
                   border-radius:20px;font-size:12px;">{d_label}</span>
    </div>
    <div style="text-align:right;">
      <span style="font-size:13px;color:#888;">Talent Level</span><br>
      <span style="font-size:16px;font-weight:700;color:#333;">{level_label.get(lv, str(lv))}</span>
    </div>
  </div>

  <!-- Two-column: Default | Usage -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">

    <!-- Default Risk -->
    <div style="background:#fafafa;border-radius:8px;padding:14px;border:1px solid #eee;">
      <div style="font-size:12px;color:#888;margin-bottom:4px;">Default Risk Score</div>
      <div style="font-size:28px;font-weight:800;color:{d_color};">P{pct_def:.1f}</div>
      <div style="font-size:12px;color:#555;margin-bottom:2px;">
        Prob: <b>{p_def:.4f}</b> &nbsp;|&nbsp; Grade: <b style="color:{d_color};">{d_grade}</b>
      </div>
      <!-- gauge bar -->
      <div style="background:linear-gradient(to right,#27ae60,#f1c40f,#e74c3c);
                  border-radius:4px;height:8px;margin:6px 0;position:relative;">
        <div style="position:absolute;top:-3px;left:{min(pct_def,99):.1f}%;
                    width:3px;height:14px;background:#222;border-radius:2px;"></div>
      </div>
      <div style="font-size:11px;color:#aaa;margin-bottom:8px;">
        Higher than {pct_def:.1f}% of peers
      </div>
      <div style="font-size:12px;font-weight:600;color:#333;margin-bottom:4px;">
        Top 5 Default Attribution (SHAP)
      </div>
      {bars_def}
    </div>

    <!-- Usage Frequency -->
    <div style="background:#fafafa;border-radius:8px;padding:14px;border:1px solid #eee;">
      <div style="font-size:12px;color:#888;margin-bottom:4px;">Usage Frequency Score</div>
      <div style="font-size:28px;font-weight:800;color:{u_color};">{u_label}</div>
      <div style="font-size:12px;color:#555;margin-bottom:2px;">
        Prob: <b>{p_usg:.4f}</b> &nbsp;|&nbsp; Grade: <b style="color:{u_color};">{u_grade}</b>
      </div>
      <div style="background:linear-gradient(to right,#27ae60,#f1c40f,#c0392b);
                  border-radius:4px;height:8px;margin:6px 0;position:relative;">
        <div style="position:absolute;top:-3px;left:{min(p_usg*100,99):.1f}%;
                    width:3px;height:14px;background:#222;border-radius:2px;"></div>
      </div>
      <div style="font-size:11px;color:#aaa;margin-bottom:8px;">
        Usage probability: {p_usg:.1%}
      </div>
      <div style="font-size:12px;font-weight:600;color:#333;margin-bottom:4px;">
        Top 5 Usage Attribution (SHAP)
      </div>
      {bars_usg}
    </div>
  </div>

  <!-- Credit Limit Comparison -->
  <div style="background:#f7f9fc;border-radius:8px;padding:12px;
              display:flex;justify-content:space-around;align-items:center;
              border:1px solid #e0e8f0;">
    <div style="text-align:center;">
      <div style="font-size:11px;color:#888;">Before Optimization</div>
      <div style="font-size:20px;font-weight:700;color:#555;">¥{lim_before:,.0f}</div>
    </div>
    <div style="font-size:24px;color:#bbb;">→</div>
    <div style="text-align:center;">
      <div style="font-size:11px;color:#888;">After Optimization</div>
      <div style="font-size:20px;font-weight:700;color:#222;">¥{lim_after:,.0f}</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:11px;color:#888;">Change</div>
      <div style="font-size:20px;font-weight:700;color:{lim_color};">{lim_delta_str}</div>
    </div>
  </div>

</div>"""
            cards_html.append(card)

        # ── Full HTML page ─────────────────────────────────────────────
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Customer Scorecard Report</title>
<style>
  body{{margin:0;padding:20px;background:#f0f2f5;font-family:'Segoe UI',Arial,sans-serif;}}
  h1{{color:#2c3e50;font-size:22px;margin-bottom:4px;}}
  .subtitle{{color:#7f8c8d;font-size:13px;margin-bottom:20px;}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:0;}}
</style>
</head>
<body>
<h1>Customer Risk & Credit Scorecard</h1>
<div class="subtitle">
  Stratified sample ({len(sample_idx)} customers) &nbsp;·&nbsp;
  Default model: LightGBM-MTL &nbsp;·&nbsp;
  Usage model: LightGBM-MTL &nbsp;·&nbsp;
  Attribution: LightGBM Feature Importance × Feature Deviation
</div>
<div class="grid">
{"".join(cards_html)}
</div>
</body>
</html>"""

        out_path = os.path.join(report_dir, "customer_scorecards.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"评分卡已保存到: {out_path}")
        return out_path

def main():
    """主函数"""
    print("=" * 80)
    print("最优额度计算")
    print("=" * 80)
    
    # 步骤1：配置参数
    print("\n步骤1：配置参数...")
    config = OptimalCreditLimitConfig()
    
    # 可在此处覆盖类默认参数（当前与 OptimalCreditLimitConfig 默认一致）
    config.alpha = 1.5  # 风险/违约系数（≥0）
    config.default_prob_scale = 0.3  # 违约概率缩放（等价于 PD_adj = p_eff * 0.3）
    # ↓ 以下三项在 load_data() 内自动从「授信额度」列检测并覆盖，无需手动设置
    # config.min_limit  → 自动设为真实数据最小额度
    # config.max_limit  → 自动设为真实数据最大额度
    # config.limit_step → 自动从 unique 值 GCD 推断
    # config.total_budget → 自动设为真实总额（calculate_optimal_limits 同样会对齐）
    config.use_discrete_optimization = False  # True 则使用离散优化
    config.use_mtl = True
    # 支用概率：Logistic MTL（L1惩罚）；违约概率：LightGBM MTL
    config.usage_model_type = "lightgbm_mtl"
    # 违约概率模型：改回原先的 Logistic MTL（L2惩罚）
    config.default_model_type = "logistic_mtl_l2"
    config.mtl_task_grouping = "age_group"
    config.calibrate_probs = True
    config.calibration_method = "isotonic"
    config.usage_rate_scale = 50000.0
    
    # 样本数量控制（如果运行太慢，可以减少样本数量）
    # config.max_samples = 100  # 使用100个样本（快速测试）
    # config.max_samples = 200  # 使用200个样本（中等速度）
    # config.max_samples = 500  # 使用500个样本（较慢但更准确）
    # config.max_samples = None  # 使用全部样本（最慢但最准确）
    
    # 步骤2：创建计算器
    print("\n步骤2：创建计算器...")
    calculator = OptimalCreditLimitCalculator(config)
    
    # 步骤3：加载数据
    print("\n步骤3：加载数据...")
    calculator.load_data()
    
    # 步骤4：训练模型
    print("\n步骤4：训练模型...")
    calculator.train_models()
    
    # 步骤5：获取人才等级（统一入口）
    print("\n步骤5：获取人才等级...")
    talent_levels = calculator._extract_talent_levels()

    unique, counts = np.unique(talent_levels, return_counts=True)
    print("人才等级分布（数值等级 -> 人数）:", dict(zip(unique.tolist(), counts.tolist())))
    
    # 步骤6：计算最优额度
    print("\n步骤6：计算最优额度...")
    df_results = calculator.calculate_optimal_limits(talent_levels)
    
    # 步骤7：生成报告
    print("\n步骤7：生成报告...")
    if df_results is not None:
        # 保存详细结果
        output_path = 'reports/credit_limit_detailed_results.xlsx'
        df_results.to_excel(output_path, index=False)
        print(f"详细结果已保存到: {output_path}")

        # 同时保存 CSV 版本（encoding utf-8-sig 保证 Excel 打开中文不乱码）
        csv_path = 'reports/credit_limit_detailed_results.csv'
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"CSV 版本已保存到: {csv_path}")
        
        # 生成报告
        calculator.generate_report(df_results)

        # 步骤8：生成客户评分卡
        print("\n步骤8：生成客户评分卡...")
        calculator.generate_scorecards(df_results, n_samples=6)
        
        # 打印关键指标
        print("\n" + "=" * 80)
        print("关键指标")
        print("=" * 80)
        print(f"总授信人数: {len(df_results)}")
        print(f"总预算: {config.total_budget:.2f}")
        print(f"总分配额度: {calculator.total_allocated:.2f}")
        print(f"预算使用比例: {calculator.total_allocated/config.total_budget:.2%}")
        print(f"最优总利润: {calculator.total_profit:.2f}")
        print("\n" + "=" * 80)
        print("计算完成！")
        print("=" * 80)
    else:
        print("额度计算失败！")


# 兼容 %run / exec 两种调用方式
if __name__ == "__main__" or "__file__" not in dir():
    main()
