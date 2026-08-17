#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离散额度组合优化与离线评价。

目标函数（对客户 i、候选额度 L）为：

    pi_i(L) = r_i * L * p_usage_i(L)
              - lgd_i * L * p_default_i(L)
              - c1 * L
              - c2 * L ** 2

人才等级不进入收益乘数，仅通过分等级额度区间和等级平均额度单调约束
进入可行域。候选额度概率和目标系数预先计算后，可使用 SciPy 0-1 线性
整数规划，或使用兼容 Python 3.6 的近似可行解后端；两条路径施加相同的
总额度预算和额度加权违约风险预算。
"""

import time
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


class PortfolioMilpResult(object):
    """统一的组合优化结果对象；不用 dataclasses，兼容 Python 3.6。"""

    def __init__(
        self,
        limits,
        selected_grid_indices,
        objective_value,
        weighted_default_risk,
        total_limit,
        solver_status,
        solver_message,
        solver_success,
        mip_gap,
        candidate_reduced,
        variable_count,
        feasible_candidate_count,
        solver_backend="scipy_milp",
        constraint_feasible=True,
        max_constraint_violation=0.0,
        iterations=0,
        dual_upper_bound=np.nan,
        heuristic_dual_bound_gap=np.nan,
        feasible_start_count=1,
        local_search_passes=0,
        local_search_single_moves=0,
        local_search_pair_exchanges=0,
        local_search_objective_gain=0.0,
    ):
        self.limits = np.asarray(limits, dtype=float)
        self.selected_grid_indices = np.asarray(selected_grid_indices, dtype=int)
        self.objective_value = float(objective_value)
        self.weighted_default_risk = float(weighted_default_risk)
        self.total_limit = float(total_limit)
        self.solver_status = int(solver_status)
        self.solver_message = str(solver_message)
        self.solver_success = bool(solver_success)
        self.mip_gap = float(mip_gap)
        self.candidate_reduced = bool(candidate_reduced)
        self.variable_count = int(variable_count)
        self.feasible_candidate_count = int(feasible_candidate_count)
        self.solver_backend = str(solver_backend)
        self.constraint_feasible = bool(constraint_feasible)
        self.max_constraint_violation = float(max_constraint_violation)
        self.iterations = int(iterations)
        self.dual_upper_bound = float(dual_upper_bound)
        self.heuristic_dual_bound_gap = float(heuristic_dual_bound_gap)
        self.feasible_start_count = int(feasible_start_count)
        self.local_search_passes = int(local_search_passes)
        self.local_search_single_moves = int(local_search_single_moves)
        self.local_search_pair_exchanges = int(local_search_pair_exchanges)
        self.local_search_objective_gain = float(local_search_objective_gain)


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
    candidate_reference_quadratic_costs=None,
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
            if candidate_reference_quadratic_costs is None:
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
                top_local = np.argpartition(
                    -values, min(top_n, len(values)) - 1
                )[:top_n]
                mandatory.update(feasible[top_local].tolist())
                quantile_local = np.linspace(
                    0, len(feasible) - 1, num=5, dtype=int
                )
                mandatory.update(feasible[quantile_local].tolist())
                chosen = np.asarray(sorted(mandatory), dtype=int)
                if len(chosen) > per_customer:
                    must_keep = {
                        int(feasible[0]),
                        int(feasible[-1]),
                        int(np.clip(base_idx[i], feasible[0], feasible[-1])),
                    }
                    optional = [v for v in chosen if int(v) not in must_keep]
                    optional = sorted(
                        optional,
                        key=lambda v: float(values[int(v - feasible[0])]),
                        reverse=True,
                    )
                    chosen = np.asarray(
                        sorted(
                            list(must_keep)
                            + optional[:per_customer - len(must_keep)]
                        ),
                        dtype=int,
                    )
            else:
                reference_costs = np.asarray(
                    candidate_reference_quadratic_costs, dtype=float
                ).reshape(-1)
                if (
                    len(reference_costs) == 0
                    or not np.isfinite(reference_costs).all()
                    or np.any(reference_costs < 0)
                ):
                    raise ValueError(
                        "candidate_reference_quadratic_costs "
                        "必须为非空、有限、非负数列"
                    )
                reference_costs = np.unique(reference_costs)
                reference_values = []
                for reference_c2 in reference_costs:
                    reference_values.append(risk_adjusted_value(
                        L,
                        np.asarray(p_usage[i, feasible], dtype=float),
                        np.asarray(p_default[i, feasible], dtype=float),
                        np.full(len(feasible), rates[i], dtype=float),
                        np.full(len(feasible), lgd[i], dtype=float),
                        linear_cost,
                        float(reference_c2),
                    ))
                # 各 c2 共用一份候选集；只取每个目标的前若干名，
                # 避免对每位客户、每个 c2 执行完整排序。
                scaled_reference_values = []
                ranked_local_pool = set()
                for reference_value in reference_values:
                    span = float(np.max(reference_value) - np.min(reference_value))
                    if span > 1e-15:
                        scaled_value = (
                            reference_value - float(np.min(reference_value))
                        ) / span
                    else:
                        scaled_value = np.zeros(len(feasible), dtype=float)
                    scaled_reference_values.append(scaled_value)
                    top_count = min(len(feasible), max(per_customer, 4))
                    if top_count == len(feasible):
                        top_local = np.arange(len(feasible), dtype=int)
                    else:
                        top_local = np.argpartition(
                            -reference_value, top_count - 1
                        )[:top_count]
                    ranked_local_pool.update(top_local.tolist())
                scaled_matrix = np.vstack(scaled_reference_values)
                ranked_local = sorted(
                    ranked_local_pool,
                    key=lambda local: (
                        -float(np.max(scaled_matrix[:, local])),
                        -float(np.mean(scaled_matrix[:, local])),
                        int(feasible[local]),
                    ),
                )
                reference_order = list(range(len(reference_values)))
                if len(reference_order) > 1:
                    reference_order = (
                        [0, len(reference_order) - 1]
                        + reference_order[1:-1]
                    )
                preferred = [
                    int(feasible[int(np.argmax(reference_values[idx]))])
                    for idx in reference_order
                ]
                preferred.extend(
                    feasible[np.linspace(
                        0, len(feasible) - 1, num=5, dtype=int
                    )].tolist()
                )
                preferred.extend(feasible[ranked_local].tolist())
                chosen_set = {
                    int(feasible[0]),
                    int(feasible[-1]),
                    int(np.clip(base_idx[i], feasible[0], feasible[-1])),
                }
                for grid_value in preferred:
                    if len(chosen_set) >= per_customer:
                        break
                    chosen_set.add(int(grid_value))
                chosen = np.asarray(sorted(chosen_set), dtype=int)

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


def build_portfolio_candidate_table(
    grid,
    p_usage,
    p_default,
    levels,
    min_limits,
    max_limits,
    interest_rates,
    lgd_coefficients,
    linear_cost,
    quadratic_cost,
    base_limits,
    max_variables=400_000,
    candidates_per_customer=16,
    candidate_reference_quadratic_costs=None,
):
    """预先构造可复用候选表；用于 c2 敏感性分析保持候选口径一致。"""
    return _candidate_index_table(
        grid,
        p_usage,
        p_default,
        levels,
        min_limits,
        max_limits,
        np.asarray(interest_rates, dtype=float),
        np.asarray(lgd_coefficients, dtype=float),
        linear_cost,
        quadratic_cost,
        np.asarray(base_limits, dtype=float),
        max_variables,
        candidates_per_customer,
        candidate_reference_quadratic_costs,
    )


def _group_mean_pairs(levels, group_mean_min_ratio):
    observed_levels = set(np.unique(np.asarray(levels, dtype=int)).tolist())
    if group_mean_min_ratio is None:
        group_mean_min_ratio = {g: 1.0 for g in range(2, 9)}
    pairs = []
    for high, ratio in sorted(group_mean_min_ratio.items()):
        high = int(high)
        low = high - 1
        ratio = float(ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                "等级平均额度最低比例必须位于[0,1]: high=%s ratio=%s"
                % (high, ratio)
            )
        if low in observed_levels and high in observed_levels:
            pairs.append((low, high, ratio))
    return pairs


def audit_portfolio_constraints(
    limits,
    weighted_risk,
    levels,
    min_limits,
    max_limits,
    total_budget,
    risk_budget,
    enforce_group_mean_monotonic=True,
    group_mean_min_ratio=None,
):
    """按统一口径审计个体区间、总预算、风险预算和等级均值约束。"""
    limits = np.asarray(limits, dtype=float)
    weighted_risk = np.asarray(weighted_risk, dtype=float)
    levels = np.asarray(levels, dtype=int)
    if not (len(limits) == len(weighted_risk) == len(levels)):
        raise ValueError("额度、风险和人才等级长度必须一致")

    lower = np.asarray([min_limits.get(int(g), -np.inf) for g in levels], dtype=float)
    upper = np.asarray([max_limits.get(int(g), np.inf) for g in levels], dtype=float)
    scale_limit = max(float(np.max(np.abs(upper[np.isfinite(upper)]))) if np.isfinite(upper).any() else 0.0, 1.0)
    budget_scale = max(abs(float(total_budget)), 1.0)
    risk_scale = max(abs(float(risk_budget)), 1.0)
    bound_violation = max(
        float(np.max(np.maximum(lower - limits, 0.0))) if len(limits) else 0.0,
        float(np.max(np.maximum(limits - upper, 0.0))) if len(limits) else 0.0,
    )
    total_limit = float(np.sum(limits))
    total_risk = float(np.sum(weighted_risk))
    budget_violation = max(total_limit - float(total_budget), 0.0)
    risk_violation = max(total_risk - float(risk_budget), 0.0)

    group_rows = []
    group_max_violation = 0.0
    if enforce_group_mean_monotonic:
        for low, high, ratio in _group_mean_pairs(levels, group_mean_min_ratio):
            low_mean = float(np.mean(limits[levels == low]))
            high_mean = float(np.mean(limits[levels == high]))
            violation = max(ratio * low_mean - high_mean, 0.0)
            group_max_violation = max(group_max_violation, violation)
            group_rows.append({
                "low_level": low,
                "high_level": high,
                "minimum_ratio": ratio,
                "low_mean": low_mean,
                "high_mean": high_mean,
                "violation": violation,
            })

    normalized = [
        bound_violation / scale_limit,
        budget_violation / budget_scale,
        risk_violation / risk_scale,
        group_max_violation / scale_limit,
    ]
    max_normalized_violation = float(max(normalized))
    feasible = bool(max_normalized_violation <= 1e-9)
    return {
        "feasible": feasible,
        "max_normalized_violation": max_normalized_violation,
        "bound_violation": bound_violation,
        "budget_violation": budget_violation,
        "risk_violation": risk_violation,
        "group_max_violation": group_max_violation,
        "total_limit": total_limit,
        "weighted_default_risk": total_risk,
        "group_constraints": group_rows,
    }


def _prepare_heuristic_problem(
    grid,
    p_usage,
    p_default,
    levels,
    min_limits,
    max_limits,
    interest_rates,
    lgd_coefficients,
    linear_cost,
    quadratic_cost,
    base_limits,
    max_variables,
    candidates_per_customer,
    candidate_reference_quadratic_costs=None,
    candidate_index_table=None,
):
    """构造贪心与拉格朗日后端共享的压缩候选问题。"""
    grid = np.asarray(grid, dtype=float)
    levels = np.asarray(levels, dtype=int)
    rates = np.asarray(interest_rates, dtype=float)
    lgd = np.asarray(lgd_coefficients, dtype=float)
    base_limits = np.asarray(base_limits, dtype=float)
    n = len(levels)
    if not (len(rates) == len(lgd) == len(base_limits) == n):
        raise ValueError("levels、利率、LGD系数和历史额度长度必须一致")

    cand = (
        _candidate_index_table(
            grid, p_usage, p_default, levels, min_limits, max_limits,
            rates, lgd, linear_cost, quadratic_cost, base_limits,
            max_variables, candidates_per_customer,
            candidate_reference_quadratic_costs,
        )
        if candidate_index_table is None else candidate_index_table
    )
    required_candidate_keys = {
        "customer", "grid_index", "offsets", "candidate_reduced",
        "feasible_total",
    }
    if not required_candidate_keys.issubset(set(cand)):
        raise ValueError("candidate_index_table 缺少必要字段")
    customers = cand["customer"]
    grid_idx = cand["grid_index"]
    if (
        len(np.asarray(cand["offsets"]).reshape(-1)) != n + 1
        or len(customers) != len(grid_idx)
        or int(np.asarray(cand["offsets"]).reshape(-1)[-1]) != len(grid_idx)
        or np.any(np.asarray(grid_idx, dtype=int) < 0)
        or np.any(np.asarray(grid_idx, dtype=int) >= len(grid))
    ):
        raise ValueError("candidate_index_table 与当前客户或额度网格不匹配")
    candidate_limits = grid[grid_idx]
    candidate_pd = np.asarray(p_default[customers, grid_idx], dtype=float)
    candidate_risk = candidate_limits * candidate_pd
    candidate_objective = risk_adjusted_value(
        candidate_limits,
        np.asarray(p_usage[customers, grid_idx], dtype=float),
        candidate_pd,
        rates[customers],
        lgd[customers],
        linear_cost,
        quadratic_cost,
    )
    if not (
        np.isfinite(candidate_limits).all()
        and np.isfinite(candidate_risk).all()
        and np.isfinite(candidate_objective).all()
    ):
        raise ValueError("候选额度、风险或目标函数存在 NaN/Inf")
    return {
        "grid": grid,
        "levels": levels,
        "cand": cand,
        "customers": customers,
        "grid_idx": grid_idx,
        "offsets": cand["offsets"],
        "candidate_limits": candidate_limits,
        "candidate_risk": candidate_risk,
        "candidate_objective": candidate_objective,
    }


def _feasible_growth_from_minimum(
    problem,
    min_limits,
    max_limits,
    total_budget,
    risk_budget,
    enforce_group_mean_monotonic,
    group_mean_min_ratio,
    max_rounds,
    priority_candidate_value=None,
):
    """从最低额度构造可行解；priority 可用拉格朗日调整价值引导铺路移动。"""
    if int(max_rounds) <= 0:
        raise ValueError("max_rounds 必须大于0")
    levels = problem["levels"]
    grid = problem["grid"]
    offsets = problem["offsets"]
    candidate_limits = problem["candidate_limits"]
    candidate_risk = problem["candidate_risk"]
    candidate_objective = problem["candidate_objective"]
    priority = (
        candidate_objective
        if priority_candidate_value is None
        else np.asarray(priority_candidate_value, dtype=float)
    )
    if len(priority) != len(candidate_objective) or not np.isfinite(priority).all():
        raise ValueError("priority_candidate_value 必须与候选变量等长且全部有限")

    n = len(levels)
    selected_var = offsets[:-1].astype(np.int64).copy()
    selected_limits = candidate_limits[selected_var]
    selected_risk = candidate_risk[selected_var]
    baseline_audit = audit_portfolio_constraints(
        selected_limits,
        selected_risk,
        levels,
        min_limits,
        max_limits,
        total_budget,
        risk_budget,
        enforce_group_mean_monotonic,
        group_mean_min_ratio,
    )
    if not baseline_audit["feasible"]:
        raise RuntimeError(
            "最低额度基线不可行；无法在不突破硬约束的前提下开始优化。"
            " budget_violation=%.6f, risk_violation=%.6f, group_violation=%.6f"
            % (
                baseline_audit["budget_violation"],
                baseline_audit["risk_violation"],
                baseline_audit["group_max_violation"],
            )
        )

    observed = np.unique(levels)
    group_counts = {int(g): int(np.sum(levels == g)) for g in observed}
    group_sums = {int(g): float(np.sum(selected_limits[levels == g])) for g in observed}
    pairs = (
        _group_mean_pairs(levels, group_mean_min_ratio)
        if enforce_group_mean_monotonic else []
    )
    total_limit = float(np.sum(selected_limits))
    total_risk = float(np.sum(selected_risk))
    raw_objective = float(np.sum(candidate_objective[selected_var]))
    best_objective = raw_objective
    best_selected_var = selected_var.copy()
    budget_tol = max(1e-6, abs(float(total_budget)) * 1e-10)
    risk_tol = max(1e-6, abs(float(risk_budget)) * 1e-10)
    group_tol = max(1e-6, float(np.max(np.abs(grid))) * 1e-10)

    def transition_is_feasible(customer, new_var):
        current_var = int(selected_var[customer])
        delta_limit = float(candidate_limits[new_var] - candidate_limits[current_var])
        delta_risk = float(candidate_risk[new_var] - candidate_risk[current_var])
        if total_limit + delta_limit > float(total_budget) + budget_tol:
            return False
        if total_risk + delta_risk > float(risk_budget) + risk_tol:
            return False
        customer_level = int(levels[customer])
        for low, high, ratio in pairs:
            low_sum = group_sums[low] + (delta_limit if customer_level == low else 0.0)
            high_sum = group_sums[high] + (delta_limit if customer_level == high else 0.0)
            low_mean = low_sum / float(group_counts[low])
            high_mean = high_sum / float(group_counts[high])
            if ratio * low_mean - high_mean > group_tol:
                return False
        return True

    rounds_run = 0
    accepted_total = 0
    for round_idx in range(int(max_rounds)):
        suggestions = []
        for customer in range(n):
            current_var = int(selected_var[customer])
            start = current_var + 1
            end = int(offsets[customer + 1])
            if start >= end:
                continue
            best_var = None
            best_score = -np.inf
            current_priority = float(priority[current_var])
            current_limit = float(candidate_limits[current_var])
            current_risk = float(candidate_risk[current_var])
            for new_var in range(start, end):
                priority_gain = float(priority[new_var] - current_priority)
                if priority_gain <= 1e-12 or not transition_is_feasible(customer, new_var):
                    continue
                delta_limit = float(candidate_limits[new_var] - current_limit)
                delta_risk = float(candidate_risk[new_var] - current_risk)
                budget_fraction = max(delta_limit, 0.0) / max(float(total_budget), 1.0)
                risk_fraction = max(delta_risk, 0.0) / max(float(risk_budget), 1.0)
                resource = max(budget_fraction + risk_fraction, 1e-15)
                score = priority_gain / resource
                if score > best_score:
                    best_score = score
                    best_var = int(new_var)
            if best_var is not None:
                suggestions.append((best_score, customer, best_var))

        if not suggestions:
            break
        suggestions.sort(key=lambda item: item[0], reverse=True)
        accepted_this_round = 0
        for _, customer, new_var in suggestions:
            if not transition_is_feasible(customer, new_var):
                continue
            current_var = int(selected_var[customer])
            priority_gain = float(priority[new_var] - priority[current_var])
            if priority_gain <= 1e-12:
                continue
            delta_limit = float(candidate_limits[new_var] - candidate_limits[current_var])
            delta_risk = float(candidate_risk[new_var] - candidate_risk[current_var])
            delta_objective = float(
                candidate_objective[new_var] - candidate_objective[current_var]
            )
            selected_var[customer] = int(new_var)
            total_limit += delta_limit
            total_risk += delta_risk
            raw_objective += delta_objective
            group_sums[int(levels[customer])] += delta_limit
            accepted_this_round += 1
        rounds_run = round_idx + 1
        accepted_total += accepted_this_round
        if raw_objective > best_objective + 1e-9:
            best_objective = raw_objective
            best_selected_var = selected_var.copy()
        if accepted_this_round == 0:
            break

    return {
        "selected_var": best_selected_var,
        "objective_value": float(best_objective),
        "rounds": int(rounds_run),
        "accepted_moves": int(accepted_total),
    }


def _build_result_from_growth(
    problem,
    growth,
    min_limits,
    max_limits,
    total_budget,
    risk_budget,
    enforce_group_mean_monotonic,
    group_mean_min_ratio,
    solver_backend,
    solver_message,
    iterations,
    dual_upper_bound=np.nan,
    heuristic_dual_bound_gap=np.nan,
    feasible_start_count=1,
    local_search_passes=0,
    local_search_single_moves=0,
    local_search_pair_exchanges=0,
    local_search_objective_gain=0.0,
):
    selected_var = np.asarray(growth["selected_var"], dtype=np.int64)
    selected_grid = problem["grid_idx"][selected_var]
    final_limits = problem["candidate_limits"][selected_var]
    final_risk = problem["candidate_risk"][selected_var]
    final_objective = problem["candidate_objective"][selected_var]
    final_audit = audit_portfolio_constraints(
        final_limits,
        final_risk,
        problem["levels"],
        min_limits,
        max_limits,
        total_budget,
        risk_budget,
        enforce_group_mean_monotonic,
        group_mean_min_ratio,
    )
    if not final_audit["feasible"]:
        raise RuntimeError("启发式返回结果未通过硬约束审计: %s" % final_audit)
    cand = problem["cand"]
    return PortfolioMilpResult(
        limits=final_limits,
        selected_grid_indices=selected_grid,
        objective_value=float(np.sum(final_objective)),
        weighted_default_risk=float(np.sum(final_risk)),
        total_limit=float(np.sum(final_limits)),
        solver_status=0,
        solver_message=solver_message,
        solver_success=True,
        mip_gap=np.nan,
        candidate_reduced=bool(cand["candidate_reduced"]),
        variable_count=int(len(problem["grid_idx"])),
        feasible_candidate_count=int(cand["feasible_total"]),
        solver_backend=solver_backend,
        constraint_feasible=True,
        max_constraint_violation=float(final_audit["max_normalized_violation"]),
        iterations=int(iterations),
        dual_upper_bound=dual_upper_bound,
        heuristic_dual_bound_gap=heuristic_dual_bound_gap,
        feasible_start_count=feasible_start_count,
        local_search_passes=local_search_passes,
        local_search_single_moves=local_search_single_moves,
        local_search_pair_exchanges=local_search_pair_exchanges,
        local_search_objective_gain=local_search_objective_gain,
    )


def _candidate_constraint_system(
    problem,
    total_budget,
    risk_budget,
    enforce_group_mean_monotonic,
    group_mean_min_ratio,
):
    """返回候选变量的标准化耦合约束系数 A 和右端 b（A x <= b）。"""
    levels = problem["levels"]
    customers = problem["customers"]
    limits = problem["candidate_limits"]
    risks = problem["candidate_risk"]
    budget_scale = max(abs(float(total_budget)), 1.0)
    risk_scale = max(abs(float(risk_budget)), 1.0)
    limit_scale = max(float(np.max(np.abs(problem["grid"]))), 1.0)
    rows = [limits / budget_scale, risks / risk_scale]
    rhs = [float(total_budget) / budget_scale, float(risk_budget) / risk_scale]
    pairs = (
        _group_mean_pairs(levels, group_mean_min_ratio)
        if enforce_group_mean_monotonic else []
    )
    for low, high, ratio in pairs:
        customer_factor = np.zeros(len(levels), dtype=float)
        low_mask = levels == low
        high_mask = levels == high
        customer_factor[low_mask] = (
            float(ratio) / (float(np.sum(low_mask)) * limit_scale)
        )
        customer_factor[high_mask] = (
            -1.0 / (float(np.sum(high_mask)) * limit_scale)
        )
        rows.append(limits * customer_factor[customers])
        rhs.append(0.0)
    return np.vstack(rows), np.asarray(rhs, dtype=float)


def _top_indices(values, mask, count):
    indices = np.flatnonzero(mask)
    if len(indices) <= int(count):
        return indices
    local_values = np.asarray(values[indices], dtype=float)
    keep = np.argpartition(-local_values, int(count) - 1)[:int(count)]
    return indices[keep]


def _bidirectional_local_search(
    problem,
    growth,
    total_budget,
    risk_budget,
    enforce_group_mean_monotonic,
    group_mean_min_ratio,
    max_passes=3,
    time_limit_seconds=30.0,
    pair_candidate_pool=80,
):
    """限时 1-opt + 2-opt：允许降额、回退和两客户协调交换。"""
    if int(max_passes) <= 0 or float(time_limit_seconds) <= 0:
        return dict(growth, local_search_passes=0, single_moves=0,
                    pair_exchanges=0, local_search_objective_gain=0.0)
    if int(pair_candidate_pool) <= 0:
        raise ValueError("pair_candidate_pool 必须大于0")

    started_at = time.time()
    selected_var = np.asarray(growth["selected_var"], dtype=np.int64).copy()
    customers = problem["customers"]
    offsets = problem["offsets"]
    objective = problem["candidate_objective"]
    m = len(objective)
    variable_index = np.arange(m, dtype=np.int64)
    constraint_coeff, constraint_rhs = _candidate_constraint_system(
        problem,
        total_budget,
        risk_budget,
        enforce_group_mean_monotonic,
        group_mean_min_ratio,
    )
    tolerance = 1e-9
    initial_objective = float(np.sum(objective[selected_var]))
    current_objective = initial_objective
    passes_run = 0
    single_moves_total = 0
    pair_exchanges_total = 0

    for pass_idx in range(int(max_passes)):
        if time.time() - started_at >= float(time_limit_seconds):
            break
        current_coeff_by_customer = constraint_coeff[:, selected_var]
        current_violation = np.sum(current_coeff_by_customer, axis=1) - constraint_rhs
        move_delta = (
            constraint_coeff - current_coeff_by_customer[:, customers]
        )
        objective_gain = objective - objective[selected_var][customers]
        alternative = variable_index != selected_var[customers]
        individually_feasible = np.all(
            current_violation[:, None] + move_delta <= tolerance,
            axis=0,
        )
        valid_single = alternative & individually_feasible & (objective_gain > 1e-12)
        single_score = np.where(valid_single, objective_gain, -np.inf)
        best_gain_by_customer = np.maximum.reduceat(single_score, offsets[:-1])
        winner_mask = (
            valid_single
            & (single_score >= best_gain_by_customer[customers] - 1e-12)
        )
        winner_candidates = np.where(winner_mask, variable_index, m)
        best_single_var = np.minimum.reduceat(winner_candidates, offsets[:-1])
        single_suggestions = [
            (float(objective_gain[new_var]), customer, int(new_var))
            for customer, new_var in enumerate(best_single_var)
            if int(new_var) < m
        ]
        single_suggestions.sort(key=lambda item: item[0], reverse=True)
        accepted_customers = set()
        accepted_single = 0
        for gain, customer, new_var in single_suggestions:
            if customer in accepted_customers:
                continue
            current_var = int(selected_var[customer])
            delta = constraint_coeff[:, new_var] - constraint_coeff[:, current_var]
            if gain <= 1e-12 or np.any(current_violation + delta > tolerance):
                continue
            selected_var[customer] = int(new_var)
            current_violation += delta
            current_objective += float(objective[new_var] - objective[current_var])
            accepted_customers.add(customer)
            accepted_single += 1
        if accepted_single:
            single_moves_total += accepted_single
            passes_run = pass_idx + 1
            continue

        # 没有单客户可行改进时，寻找“释放约束 + 使用约束”的两客户联合交换。
        positive_gain = alternative & (objective_gain > 1e-12)
        infeasible_alone = ~individually_feasible
        receiver_idx = _top_indices(
            objective_gain,
            positive_gain & infeasible_alone,
            int(pair_candidate_pool),
        )
        donor_parts = []
        for constraint_idx in range(move_delta.shape[0]):
            release = -move_delta[constraint_idx]
            donor_mask = alternative & (release > tolerance)
            donor_score = np.divide(
                objective_gain,
                release,
                out=np.full(m, -np.inf, dtype=float),
                where=donor_mask,
            )
            donor_parts.append(_top_indices(
                donor_score,
                donor_mask,
                int(pair_candidate_pool),
            ))
        donor_idx = (
            np.unique(np.concatenate(donor_parts))
            if donor_parts else np.asarray([], dtype=int)
        )
        pair_suggestions = []
        for receiver_var in receiver_idx:
            receiver_customer = int(customers[receiver_var])
            different_customer = customers[donor_idx] != receiver_customer
            combined_gain = objective_gain[receiver_var] + objective_gain[donor_idx]
            combined_delta = move_delta[:, receiver_var, None] + move_delta[:, donor_idx]
            feasible_pair = np.all(
                current_violation[:, None] + combined_delta <= tolerance,
                axis=0,
            )
            valid = different_customer & feasible_pair & (combined_gain > 1e-12)
            if not valid.any():
                continue
            valid_positions = np.flatnonzero(valid)
            best_position = valid_positions[
                int(np.argmax(combined_gain[valid_positions]))
            ]
            donor_var = int(donor_idx[best_position])
            pair_suggestions.append((
                float(combined_gain[best_position]),
                receiver_customer,
                int(receiver_var),
                int(customers[donor_var]),
                donor_var,
            ))
        pair_suggestions.sort(key=lambda item: item[0], reverse=True)
        accepted_pair_customers = set()
        accepted_pairs = 0
        for _, receiver_customer, receiver_var, donor_customer, donor_var in pair_suggestions:
            if (
                receiver_customer in accepted_pair_customers
                or donor_customer in accepted_pair_customers
                or receiver_customer == donor_customer
            ):
                continue
            receiver_current = int(selected_var[receiver_customer])
            donor_current = int(selected_var[donor_customer])
            receiver_gain = float(objective[receiver_var] - objective[receiver_current])
            donor_gain = float(objective[donor_var] - objective[donor_current])
            combined_gain = receiver_gain + donor_gain
            combined_delta = (
                constraint_coeff[:, receiver_var]
                - constraint_coeff[:, receiver_current]
                + constraint_coeff[:, donor_var]
                - constraint_coeff[:, donor_current]
            )
            if combined_gain <= 1e-12 or np.any(
                current_violation + combined_delta > tolerance
            ):
                continue
            selected_var[receiver_customer] = int(receiver_var)
            selected_var[donor_customer] = int(donor_var)
            current_violation += combined_delta
            current_objective += combined_gain
            accepted_pair_customers.add(receiver_customer)
            accepted_pair_customers.add(donor_customer)
            accepted_pairs += 1
        passes_run = pass_idx + 1
        pair_exchanges_total += accepted_pairs
        if accepted_pairs == 0:
            break

    return {
        "selected_var": selected_var,
        "objective_value": float(current_objective),
        "rounds": int(growth.get("rounds", 0)),
        "accepted_moves": int(growth.get("accepted_moves", 0)),
        "local_search_passes": int(passes_run),
        "single_moves": int(single_moves_total),
        "pair_exchanges": int(pair_exchanges_total),
        "local_search_objective_gain": float(current_objective - initial_objective),
    }


def solve_discrete_portfolio_heuristic(
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
    max_rounds: int = 30,
    candidate_reference_quadratic_costs=None,
    candidate_index_table=None,
):
    """保留原有 Python 3.6 单起点可行贪心入口。"""
    if total_budget < 0 or risk_budget < 0:
        raise ValueError("总额度预算和风险预算必须为非负数")
    problem = _prepare_heuristic_problem(
        grid, p_usage, p_default, levels, min_limits, max_limits,
        interest_rates, lgd_coefficients, linear_cost, quadratic_cost,
        base_limits, max_variables, candidates_per_customer,
        candidate_reference_quadratic_costs,
        candidate_index_table,
    )
    growth = _feasible_growth_from_minimum(
        problem, min_limits, max_limits, total_budget, risk_budget,
        enforce_group_mean_monotonic, group_mean_min_ratio, max_rounds,
    )
    message = (
        "Feasible greedy solution; no global-optimality guarantee and no MIP gap. "
        "rounds=%d, accepted_moves=%d"
        % (growth["rounds"], growth["accepted_moves"])
    )
    return _build_result_from_growth(
        problem, growth, min_limits, max_limits, total_budget, risk_budget,
        enforce_group_mean_monotonic, group_mean_min_ratio,
        "heuristic", message, growth["rounds"],
    )


def solve_discrete_portfolio_lagrangian(
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
    max_rounds: int = 30,
    lagrangian_iterations: int = 40,
    lagrangian_time_limit_seconds: float = 60.0,
    lagrangian_step_size: float = 0.5,
    lagrangian_multiplier_cap: float = 1000.0,
    local_search_enabled: bool = True,
    local_search_max_passes: int = 3,
    local_search_time_limit_seconds: float = 30.0,
    local_search_pair_candidate_pool: int = 80,
    candidate_reference_quadratic_costs=None,
    candidate_index_table=None,
):
    """向量化拉格朗日定价 + 双起点 + 限时双向局部搜索。"""
    if total_budget < 0 or risk_budget < 0:
        raise ValueError("总额度预算和风险预算必须为非负数")
    if int(lagrangian_iterations) <= 0:
        raise ValueError("lagrangian_iterations 必须大于0")
    if float(lagrangian_time_limit_seconds) <= 0:
        raise ValueError("lagrangian_time_limit_seconds 必须大于0")
    if float(lagrangian_step_size) <= 0 or float(lagrangian_multiplier_cap) <= 0:
        raise ValueError("拉格朗日步长和乘子上限必须大于0")
    if int(local_search_max_passes) < 0:
        raise ValueError("local_search_max_passes 不能小于0")
    if float(local_search_time_limit_seconds) < 0:
        raise ValueError("local_search_time_limit_seconds 不能小于0")
    if int(local_search_pair_candidate_pool) <= 0:
        raise ValueError("local_search_pair_candidate_pool 必须大于0")

    started_at = time.time()
    problem = _prepare_heuristic_problem(
        grid, p_usage, p_default, levels, min_limits, max_limits,
        interest_rates, lgd_coefficients, linear_cost, quadratic_cost,
        base_limits, max_variables, candidates_per_customer,
        candidate_reference_quadratic_costs,
        candidate_index_table,
    )
    greedy_growth = _feasible_growth_from_minimum(
        problem, min_limits, max_limits, total_budget, risk_budget,
        enforce_group_mean_monotonic, group_mean_min_ratio, max_rounds,
    )

    levels_array = problem["levels"]
    customers = problem["customers"]
    offsets = problem["offsets"]
    candidate_limits = problem["candidate_limits"]
    candidate_risk = problem["candidate_risk"]
    candidate_objective = problem["candidate_objective"]
    n = len(levels_array)
    m = len(candidate_objective)
    variable_index = np.arange(m, dtype=np.int64)
    objective_scale = max(float(np.max(np.abs(candidate_objective))), 1.0)
    objective_scaled = candidate_objective / objective_scale
    budget_scale = max(abs(float(total_budget)), 1.0)
    risk_scale = max(abs(float(risk_budget)), 1.0)
    budget_coeff = candidate_limits / budget_scale
    risk_coeff = candidate_risk / risk_scale
    budget_rhs = float(total_budget) / budget_scale
    risk_rhs = float(risk_budget) / risk_scale
    limit_scale = max(float(np.max(np.abs(problem["grid"]))), 1.0)
    pairs = (
        _group_mean_pairs(levels_array, group_mean_min_ratio)
        if enforce_group_mean_monotonic else []
    )
    pair_factors = np.zeros((len(pairs), n), dtype=float)
    for pair_idx, (low, high, ratio) in enumerate(pairs):
        low_mask = levels_array == low
        high_mask = levels_array == high
        pair_factors[pair_idx, low_mask] = (
            float(ratio) / (float(np.sum(low_mask)) * limit_scale)
        )
        pair_factors[pair_idx, high_mask] = (
            -1.0 / (float(np.sum(high_mask)) * limit_scale)
        )

    lambda_budget = 0.0
    lambda_risk = 0.0
    lambda_group = np.zeros(len(pairs), dtype=float)
    best_dual_bound_scaled = np.inf
    best_multipliers = (lambda_budget, lambda_risk, lambda_group.copy())
    iterations_run = 0
    dual_started_at = time.time()
    for iteration in range(int(lagrangian_iterations)):
        group_customer_penalty = (
            np.dot(lambda_group, pair_factors)
            if len(pairs) else np.zeros(n, dtype=float)
        )
        adjusted_score = (
            objective_scaled
            - lambda_budget * budget_coeff
            - lambda_risk * risk_coeff
            - candidate_limits * group_customer_penalty[customers]
        )
        maximum_by_customer = np.maximum.reduceat(adjusted_score, offsets[:-1])
        winner_candidates = np.where(
            adjusted_score >= maximum_by_customer[customers] - 1e-12,
            variable_index,
            m,
        )
        selected_var = np.minimum.reduceat(winner_candidates, offsets[:-1])
        selected_limits = candidate_limits[selected_var]
        selected_risk = candidate_risk[selected_var]
        budget_violation = float(np.sum(selected_limits) / budget_scale - budget_rhs)
        risk_violation = float(np.sum(selected_risk) / risk_scale - risk_rhs)
        group_violations = np.asarray([
            float(
                ratio * np.mean(selected_limits[levels_array == low])
                - np.mean(selected_limits[levels_array == high])
            ) / limit_scale
            for low, high, ratio in pairs
        ], dtype=float)
        dual_bound_scaled = float(
            np.sum(objective_scaled[selected_var])
            - lambda_budget * budget_violation
            - lambda_risk * risk_violation
            - np.dot(lambda_group, group_violations)
        )
        if dual_bound_scaled < best_dual_bound_scaled:
            best_dual_bound_scaled = dual_bound_scaled
            best_multipliers = (
                float(lambda_budget), float(lambda_risk), lambda_group.copy()
            )

        step = float(lagrangian_step_size) / np.sqrt(float(iteration + 1))
        lambda_budget = float(np.clip(
            lambda_budget + step * budget_violation,
            0.0,
            float(lagrangian_multiplier_cap),
        ))
        lambda_risk = float(np.clip(
            lambda_risk + step * risk_violation,
            0.0,
            float(lagrangian_multiplier_cap),
        ))
        lambda_group = np.clip(
            lambda_group + step * group_violations,
            0.0,
            float(lagrangian_multiplier_cap),
        )
        iterations_run = iteration + 1
        if time.time() - dual_started_at >= float(lagrangian_time_limit_seconds):
            break

    best_lambda_budget, best_lambda_risk, best_lambda_group = best_multipliers
    best_group_customer_penalty = (
        np.dot(best_lambda_group, pair_factors)
        if len(pairs) else np.zeros(n, dtype=float)
    )
    guided_priority = objective_scale * (
        objective_scaled
        - best_lambda_budget * budget_coeff
        - best_lambda_risk * risk_coeff
        - candidate_limits * best_group_customer_penalty[customers]
    )
    guided_growth = _feasible_growth_from_minimum(
        problem, min_limits, max_limits, total_budget, risk_budget,
        enforce_group_mean_monotonic, group_mean_min_ratio, max_rounds,
        priority_candidate_value=guided_priority,
    )
    best_growth = (
        guided_growth
        if guided_growth["objective_value"] > greedy_growth["objective_value"] + 1e-9
        else greedy_growth
    )
    selected_start = "lagrangian_guided" if best_growth is guided_growth else "greedy"
    local_search_started_at = time.time()
    if bool(local_search_enabled):
        improved_growth = _bidirectional_local_search(
            problem,
            best_growth,
            total_budget,
            risk_budget,
            enforce_group_mean_monotonic,
            group_mean_min_ratio,
            max_passes=int(local_search_max_passes),
            time_limit_seconds=float(local_search_time_limit_seconds),
            pair_candidate_pool=int(local_search_pair_candidate_pool),
        )
        if (
            improved_growth["objective_value"]
            >= best_growth["objective_value"] - 1e-9
        ):
            best_growth = improved_growth
    local_search_elapsed = time.time() - local_search_started_at
    dual_upper_bound = max(
        float(best_dual_bound_scaled * objective_scale),
        float(best_growth["objective_value"]),
    )
    heuristic_gap = max(
        dual_upper_bound - float(best_growth["objective_value"]), 0.0
    ) / max(abs(dual_upper_bound), 1.0)
    elapsed = time.time() - started_at
    dual_elapsed = time.time() - dual_started_at
    message = (
        "Feasible Lagrangian-guided solution; reduced-candidate dual bound only, "
        "not a MIP gap. iterations=%d, dual_seconds=%.3f, total_seconds=%.3f, "
        "selected_start=%s, local_search_seconds=%.3f, local_search_passes=%d, "
        "single_moves=%d, pair_exchanges=%d"
        % (
            iterations_run,
            dual_elapsed,
            elapsed,
            selected_start,
            local_search_elapsed,
            int(best_growth.get("local_search_passes", 0)),
            int(best_growth.get("single_moves", 0)),
            int(best_growth.get("pair_exchanges", 0)),
        )
    )
    return _build_result_from_growth(
        problem, best_growth, min_limits, max_limits, total_budget, risk_budget,
        enforce_group_mean_monotonic, group_mean_min_ratio,
        "lagrangian", message, iterations_run,
        dual_upper_bound=dual_upper_bound,
        heuristic_dual_bound_gap=heuristic_gap,
        feasible_start_count=2,
        local_search_passes=int(best_growth.get("local_search_passes", 0)),
        local_search_single_moves=int(best_growth.get("single_moves", 0)),
        local_search_pair_exchanges=int(best_growth.get("pair_exchanges", 0)),
        local_search_objective_gain=float(
            best_growth.get("local_search_objective_gain", 0.0)
        ),
    )


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
    candidate_reference_quadratic_costs=None,
    candidate_index_table=None,
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

    cand = (
        _candidate_index_table(
            grid, p_usage, p_default, levels, min_limits, max_limits,
            rates, lgd, linear_cost, quadratic_cost, base_limits,
            max_variables, candidates_per_customer,
            candidate_reference_quadratic_costs,
        )
        if candidate_index_table is None else candidate_index_table
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
    selected_risk = selected_limits * selected_pd
    final_audit = audit_portfolio_constraints(
        selected_limits,
        selected_risk,
        levels,
        min_limits,
        max_limits,
        total_budget,
        risk_budget,
        enforce_group_mean_monotonic,
        group_mean_min_ratio,
    )
    node_count = getattr(result, "mip_node_count", 0)
    if node_count is None or not np.isfinite(node_count):
        node_count = 0
    mip_gap = getattr(result, "mip_gap", np.nan)
    if mip_gap is None:
        mip_gap = np.nan

    return PortfolioMilpResult(
        limits=selected_limits,
        selected_grid_indices=selected_grid,
        objective_value=float(selected_objective.sum()),
        weighted_default_risk=float(np.sum(selected_risk)),
        total_limit=float(selected_limits.sum()),
        solver_status=int(result.status),
        solver_message=str(result.message),
        solver_success=bool(result.success),
        mip_gap=float(mip_gap),
        candidate_reduced=bool(cand["candidate_reduced"]),
        variable_count=int(m),
        feasible_candidate_count=int(cand["feasible_total"]),
        solver_backend="scipy_milp",
        constraint_feasible=bool(final_audit["feasible"]),
        max_constraint_violation=float(final_audit["max_normalized_violation"]),
        iterations=int(node_count),
    )


def distribution_summary(values, metric: str):
    values = np.asarray(values, dtype=float)
    stats = {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "p05": np.percentile(values, 5.0),
        "p25": np.percentile(values, 25.0),
        "median": np.median(values),
        "p75": np.percentile(values, 75.0),
        "p95": np.percentile(values, 95.0),
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
