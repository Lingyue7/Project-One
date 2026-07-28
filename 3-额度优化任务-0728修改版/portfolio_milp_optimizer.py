#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离散额度组合优化与离线评价。

目标函数（对客户 i、候选额度 L）为：

    pi_i(L) = r_i * L * p_usage_i(L)
              - lgd_i * L * p_default_i(L)
              - c1 * L
              - c2 * L ** 2

人才等级不进入收益乘数，仅通过分等级额度区间和等级平均额度单调约束
进入可行域。候选额度概率和目标系数预先计算后，问题使用 0-1 线性
整数规划求解，同时施加总额度预算和额度加权违约风险预算。
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class PortfolioMilpResult:
    limits: np.ndarray
    selected_grid_indices: np.ndarray
    objective_value: float
    weighted_default_risk: float
    total_limit: float
    solver_status: int
    solver_message: str
    solver_success: bool
    mip_gap: float
    candidate_reduced: bool
    variable_count: int
    feasible_candidate_count: int


def risk_adjusted_value(
    limits,
    p_usage,
    p_default,
    interest_rates,
    lgd_coefficients,
    linear_cost: float,
    quadratic_cost: float,
):
    """逐客户计算风险调整后的模型目标函数值。"""
    L = np.asarray(limits, dtype=float)
    pu = np.asarray(p_usage, dtype=float)
    pd_ = np.asarray(p_default, dtype=float)
    r = np.asarray(interest_rates, dtype=float)
    lgd = np.asarray(lgd_coefficients, dtype=float)
    return r * L * pu - lgd * L * pd_ - float(linear_cost) * L - float(quadratic_cost) * L ** 2


def _nearest_grid_indices(values, grid):
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    idx = np.searchsorted(grid, values)
    idx = np.clip(idx, 0, len(grid) - 1)
    left = np.clip(idx - 1, 0, len(grid) - 1)
    return np.where(np.abs(grid[left] - values) <= np.abs(grid[idx] - values), left, idx).astype(int)


def _candidate_index_table(
    grid,
    p_usage,
    p_default,
    levels,
    min_limits: Dict[int, float],
    max_limits: Dict[int, float],
    rates,
    lgd,
    linear_cost,
    quadratic_cost,
    base_limits,
    max_variables,
    candidates_per_customer,
):
    """构造每位客户的候选下标；超大问题仅保留有代表性的高价值档位。"""
    grid = np.asarray(grid, dtype=float)
    levels = np.asarray(levels, dtype=int)
    n = len(levels)
    lo = np.asarray([min_limits.get(int(g), float(grid[0])) for g in levels], dtype=float)
    hi = np.asarray([max_limits.get(int(g), float(grid[-1])) for g in levels], dtype=float)
    lo_idx = np.searchsorted(grid, lo - 1e-9, side="left")
    hi_idx = np.searchsorted(grid, hi + 1e-9, side="right") - 1
    lo_idx = np.clip(lo_idx, 0, len(grid) - 1)
    hi_idx = np.clip(hi_idx, 0, len(grid) - 1)
    if np.any(lo_idx > hi_idx):
        bad = np.where(lo_idx > hi_idx)[0][:10]
        raise ValueError("部分客户的人才等级额度区间与候选网格没有交集，示例行: %s" % bad.tolist())

    feasible_count = hi_idx - lo_idx + 1
    feasible_total = int(feasible_count.sum())
    reduce_candidates = feasible_total > int(max_variables)
    base_idx = _nearest_grid_indices(base_limits, grid)
    per_customer = max(4, int(candidates_per_customer))
    if reduce_candidates:
        per_customer = max(4, min(per_customer, int(max_variables) // max(n, 1)))

    customer_parts = []
    grid_parts = []
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        feasible = np.arange(lo_idx[i], hi_idx[i] + 1, dtype=int)
        if not reduce_candidates or len(feasible) <= per_customer:
            chosen = feasible
        else:
            L = grid[feasible]
            values = risk_adjusted_value(
                L,
                np.asarray(p_usage[i, feasible], dtype=float),
                np.asarray(p_default[i, feasible], dtype=float),
                np.full(len(feasible), rates[i], dtype=float),
                np.full(len(feasible), lgd[i], dtype=float),
                linear_cost,
                quadratic_cost,
            )
            mandatory = {
                int(feasible[0]),
                int(feasible[-1]),
                int(np.clip(base_idx[i], feasible[0], feasible[-1])),
                int(feasible[int(np.argmax(values))]),
            }
            # 保留目标值最高的档位，并加入均匀分布档位以保留预算/均值约束的调节空间。
            top_n = max(1, per_customer - len(mandatory) - 3)
            top_local = np.argpartition(-values, min(top_n, len(values)) - 1)[:top_n]
            mandatory.update(feasible[top_local].tolist())
            quantile_local = np.linspace(0, len(feasible) - 1, num=5, dtype=int)
            mandatory.update(feasible[quantile_local].tolist())
            chosen = np.asarray(sorted(mandatory), dtype=int)
            if len(chosen) > per_customer:
                must_keep = {int(feasible[0]), int(feasible[-1]), int(np.clip(base_idx[i], feasible[0], feasible[-1]))}
                optional = [v for v in chosen if int(v) not in must_keep]
                optional = sorted(optional, key=lambda v: float(values[int(v - feasible[0])]), reverse=True)
                chosen = np.asarray(sorted(list(must_keep) + optional[:per_customer - len(must_keep)]), dtype=int)

        customer_parts.append(np.full(len(chosen), i, dtype=np.int32))
        grid_parts.append(chosen.astype(np.int32))
        offsets[i + 1] = offsets[i] + len(chosen)

    return {
        "customer": np.concatenate(customer_parts),
        "grid_index": np.concatenate(grid_parts),
        "offsets": offsets,
        "lo_idx": lo_idx,
        "hi_idx": hi_idx,
        "candidate_reduced": bool(reduce_candidates),
        "feasible_total": feasible_total,
    }


def solve_discrete_portfolio_milp(
    grid,
    p_usage,
    p_default,
    levels,
    min_limits: Dict[int, float],
    max_limits: Dict[int, float],
    interest_rates,
    lgd_coefficients,
    linear_cost: float,
    quadratic_cost: float,
    total_budget: float,
    risk_budget: float,
    base_limits,
    enforce_group_mean_monotonic: bool = True,
    group_mean_min_ratio: Optional[Dict[int, float]] = None,
    max_variables: int = 400_000,
    candidates_per_customer: int = 16,
    time_limit_seconds: float = 600.0,
    mip_relative_gap: float = 0.01,
):
    """求解多选 0-1 线性规划；目标系数由离散概率网格预先计算。"""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix, vstack
    except ImportError as exc:
        raise ImportError(
            "离散组合优化需要 scipy>=1.9（使用 scipy.optimize.milp/HiGHS），请先安装或升级 scipy。"
        ) from exc

    grid = np.asarray(grid, dtype=float)
    levels = np.asarray(levels, dtype=int)
    rates = np.asarray(interest_rates, dtype=float)
    lgd = np.asarray(lgd_coefficients, dtype=float)
    base_limits = np.asarray(base_limits, dtype=float)
    n = len(levels)
    if not (len(rates) == len(lgd) == len(base_limits) == n):
        raise ValueError("levels、利率、LGD系数和历史额度长度必须一致")
    if total_budget < 0 or risk_budget < 0:
        raise ValueError("总额度预算和风险预算必须为非负数")

    cand = _candidate_index_table(
        grid, p_usage, p_default, levels, min_limits, max_limits,
        rates, lgd, linear_cost, quadratic_cost, base_limits,
        max_variables, candidates_per_customer,
    )
    customers = cand["customer"]
    grid_idx = cand["grid_index"]
    offsets = cand["offsets"]
    m = len(grid_idx)
    cols = np.arange(m, dtype=np.int64)
    limits = grid[grid_idx]
    pu = np.asarray(p_usage[customers, grid_idx], dtype=float)
    pd_ = np.asarray(p_default[customers, grid_idx], dtype=float)
    objective = risk_adjusted_value(
        limits, pu, pd_, rates[customers], lgd[customers], linear_cost, quadratic_cost
    )
    weighted_risk = limits * pd_

    min_total = 0.0
    min_risk = 0.0
    for i in range(n):
        start, end = int(offsets[i]), int(offsets[i + 1])
        min_total += float(np.min(limits[start:end]))
        min_risk += float(np.min(weighted_risk[start:end]))
    if min_total > float(total_budget) + 1e-6:
        raise ValueError(
            "总额度预算低于当前候选集合可达到的最低额度总和: budget=%.2f, min_total=%.2f"
            % (float(total_budget), min_total)
        )
    if min_risk > float(risk_budget) + 1e-6:
        raise ValueError(
            "风险预算低于当前候选集合可达到的逐客最低风险之和: budget=%.6f, min_risk=%.6f"
            % (float(risk_budget), min_risk)
        )

    # 每位客户恰选一个档位。
    select_one = coo_matrix((np.ones(m), (customers, cols)), shape=(n, m)).tocsr()
    matrices = [select_one]
    lower = [np.ones(n)]
    upper = [np.ones(n)]

    # 总额度预算与额度加权违约风险预算。
    matrices.append(coo_matrix((limits, (np.zeros(m, dtype=int), cols)), shape=(1, m)).tocsr())
    lower.append(np.asarray([-np.inf]))
    upper.append(np.asarray([float(total_budget)]))
    matrices.append(coo_matrix((weighted_risk, (np.zeros(m, dtype=int), cols)), shape=(1, m)).tocsr())
    lower.append(np.asarray([-np.inf]))
    upper.append(np.asarray([float(risk_budget)]))

    # 只对名义上相邻且均存在的等级施加约束。ratio[高等级]=q 表示
    # mean(high) >= q * mean(low)；F3/F2/F1用1，E/D可用0.8。
    observed_levels = set(np.unique(levels).tolist())
    if group_mean_min_ratio is None:
        group_mean_min_ratio = {g: 1.0 for g in range(2, 9)}
    pairs = [
        (int(high) - 1, int(high), float(ratio))
        for high, ratio in sorted(group_mean_min_ratio.items())
        if int(high) - 1 in observed_levels and int(high) in observed_levels
    ]
    if enforce_group_mean_monotonic and pairs:
        group_rows = []
        group_cols = []
        group_data = []
        for row, (g_low, g_high, ratio) in enumerate(pairs):
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("等级平均额度最低比例必须位于[0,1]: high=%s ratio=%s" % (g_high, ratio))
            n_low = int(np.sum(levels == g_low))
            n_high = int(np.sum(levels == g_high))
            mask_low = levels[customers] == g_low
            mask_high = levels[customers] == g_high
            group_rows.extend(np.full(int(mask_low.sum()), row, dtype=int).tolist())
            group_cols.extend(cols[mask_low].tolist())
            group_data.extend((ratio * limits[mask_low] / n_low).tolist())
            group_rows.extend(np.full(int(mask_high.sum()), row, dtype=int).tolist())
            group_cols.extend(cols[mask_high].tolist())
            group_data.extend((-limits[mask_high] / n_high).tolist())
        group_matrix = coo_matrix(
            (np.asarray(group_data), (np.asarray(group_rows), np.asarray(group_cols))),
            shape=(len(pairs), m),
        ).tocsr()
        matrices.append(group_matrix)
        lower.append(np.full(len(pairs), -np.inf))
        upper.append(np.zeros(len(pairs)))

    A = vstack(matrices, format="csr")
    lb = np.concatenate(lower)
    ub = np.concatenate(upper)
    result = milp(
        c=-objective,
        integrality=np.ones(m, dtype=np.int8),
        bounds=Bounds(np.zeros(m), np.ones(m)),
        constraints=LinearConstraint(A, lb, ub),
        options={
            "presolve": True,
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": float(mip_relative_gap),
        },
    )
    if result.x is None:
        raise RuntimeError("整数规划未返回可行解: status=%s, message=%s" % (result.status, result.message))

    selected_var = np.empty(n, dtype=np.int64)
    for i in range(n):
        start, end = int(offsets[i]), int(offsets[i + 1])
        selected_var[i] = start + int(np.argmax(result.x[start:end]))
    selected_grid = grid_idx[selected_var]
    selected_limits = grid[selected_grid]
    selected_pd = np.asarray(p_default[np.arange(n), selected_grid], dtype=float)
    selected_objective = objective[selected_var]

    return PortfolioMilpResult(
        limits=selected_limits,
        selected_grid_indices=selected_grid,
        objective_value=float(selected_objective.sum()),
        weighted_default_risk=float(np.sum(selected_limits * selected_pd)),
        total_limit=float(selected_limits.sum()),
        solver_status=int(result.status),
        solver_message=str(result.message),
        solver_success=bool(result.success),
        mip_gap=float(getattr(result, "mip_gap", np.nan)),
        candidate_reduced=bool(cand["candidate_reduced"]),
        variable_count=int(m),
        feasible_candidate_count=int(cand["feasible_total"]),
    )


def distribution_summary(values, metric: str):
    values = np.asarray(values, dtype=float)
    stats = {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "p05": np.quantile(values, 0.05),
        "p25": np.quantile(values, 0.25),
        "median": np.median(values),
        "p75": np.quantile(values, 0.75),
        "p95": np.quantile(values, 0.95),
        "max": np.max(values),
    }
    return pd.DataFrame({"metric": metric, "statistic": list(stats), "value": list(stats.values())})


def concentration_shares(limits, percentages: Iterable[float] = (0.01, 0.05, 0.10)):
    limits = np.asarray(limits, dtype=float)
    total = float(limits.sum())
    ordered = np.sort(limits)[::-1]
    rows = []
    for pct in percentages:
        count = max(1, int(np.ceil(len(ordered) * float(pct))))
        rows.append({
            "top_customer_fraction": float(pct),
            "customer_count": count,
            "limit_share": float(ordered[:count].sum() / total) if total > 0 else np.nan,
        })
    return pd.DataFrame(rows)
