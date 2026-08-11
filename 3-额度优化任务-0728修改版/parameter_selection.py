#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""额度组合优化参数的数据估计与预设情景构造。"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TIER_TO_LEVEL = {
    "F3": 1, "F3级": 1, "f3": 1,
    "F2": 2, "F2级": 2, "f2": 2,
    "F1": 3, "F1级": 3, "f1": 3,
    "E": 4, "E级": 4, "e": 4,
    "D": 5, "D级": 5, "d": 5,
}
LEVEL_TO_TIER = {1: "F3", 2: "F2", 3: "F1", 4: "E", 5: "D"}


@dataclass
class ParameterSelectionResult:
    interest_rate: float
    lgd_coefficient: float
    lgd_source: str
    average_utilization: float
    ftp_rate: float
    linear_cost: float
    c2_reference_limit: float
    c2_deltas: Tuple[float, ...]
    c2_candidates: Tuple[float, ...]
    tier_min_limits: Dict[int, float]
    tier_max_limits: Dict[int, float]
    group_mean_min_ratio: Dict[int, float]
    risk_tolerance: float
    development_customer_count: int


def read_table(path: str, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """读取参数估计所需的 CSV/Excel 明细。"""
    suffix = os.path.splitext(str(path))[1].lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path, encoding=encoding)
    raise ValueError("仅支持 CSV/XLS/XLSX 参数明细: %s" % path)


def _numeric(series):
    return pd.to_numeric(series, errors="coerce")


def _normalize_status(series: pd.Series) -> pd.Series:
    """统一数值/字符状态编码，避免CSV把“3”读取成“3.0”后匹配失败。"""
    text = series.astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    integer_like = numeric.notna() & np.isclose(numeric, np.round(numeric))
    text.loc[integer_like] = numeric.loc[integer_like].round().astype("Int64").astype(str)
    return text


def _normalize_talent_levels(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mapped = series.astype(str).str.strip().map(TIER_TO_LEVEL)
    result = numeric.fillna(mapped)
    bad = result.isna()
    if bad.any():
        raise ValueError("无法识别人才等级，示例: %s" % series.loc[bad].astype(str).unique()[:10].tolist())
    result = result.astype(int)
    unsupported = sorted(set(result.unique()) - set(LEVEL_TO_TIER))
    if unsupported:
        raise ValueError("技术方案仅定义 F3/F2/F1/E/D 五档，发现未定义等级: %s" % unsupported)
    return result


def estimate_average_utilization(
    account_df: pd.DataFrame,
    customer_col: str,
    credit_limit_col: str,
    utilized_balance_col: str,
    winsor_quantiles: Tuple[float, float] = (0.01, 0.99),
):
    """先按客户汇总实际支用余额与额度，再估计客户平均支用率。"""
    required = [customer_col, credit_limit_col, utilized_balance_col]
    missing = [c for c in required if c not in account_df.columns]
    if missing:
        raise ValueError("平均支用率估计缺少字段: %s" % missing)
    work = account_df[required].copy()
    work[credit_limit_col] = _numeric(work[credit_limit_col])
    work[utilized_balance_col] = _numeric(work[utilized_balance_col])
    work = work.dropna(subset=required)
    work = work[(work[credit_limit_col] > 0) & (work[utilized_balance_col] >= 0)]
    if work.empty:
        raise ValueError("没有可用于平均支用率估计的有效账户记录")

    customer = work.groupby(customer_col, as_index=False).agg(
        total_credit_limit=(credit_limit_col, "sum"),
        utilized_balance=(utilized_balance_col, "sum"),
    )
    customer["utilization_raw"] = customer["utilized_balance"] / customer["total_credit_limit"]
    finite = np.isfinite(customer["utilization_raw"].to_numpy(dtype=float))
    customer = customer.loc[finite].copy()
    if customer.empty:
        raise ValueError("客户支用率均为无效值")

    # 支用率定义在[0,1]；同时记录越界数量，不能静默把异常当作正常数据。
    customer["below_zero"] = customer["utilization_raw"] < 0
    customer["above_one"] = customer["utilization_raw"] > 1
    bounded = customer["utilization_raw"].clip(0.0, 1.0)
    q_low, q_high = winsor_quantiles
    lower = float(bounded.quantile(q_low))
    upper = float(bounded.quantile(q_high))
    customer["utilization_used"] = bounded.clip(lower, upper)
    average = float(customer["utilization_used"].mean())
    summary = pd.DataFrame([{
        "customer_count": len(customer),
        "raw_mean": float(customer["utilization_raw"].mean()),
        "winsorized_mean": average,
        "winsor_lower_quantile": float(q_low),
        "winsor_upper_quantile": float(q_high),
        "winsor_lower_value": lower,
        "winsor_upper_value": upper,
        "below_zero_count": int(customer["below_zero"].sum()),
        "above_one_count": int(customer["above_one"].sum()),
    }])
    return average, summary, customer


def estimate_historical_lgd(
    recovery_df: pd.DataFrame,
    loan_id_col: str,
    observation_date_col: str,
    account_status_col: str,
    outstanding_principal_col: str,
    cumulative_principal_paid_col: str,
    default_status_values: Sequence[str] = ("3", "4", "7", "9"),
    close_date_col: Optional[str] = None,
    closed_status_values: Sequence[str] = ("5",),
    chargeoff_status_values: Sequence[str] = ("7",),
    cutoff_date: Optional[str] = None,
    recovery_months: int = 12,
):
    """按首次违约EAD与12个月累计本金回收差额估计组合LGD。"""
    required = [
        loan_id_col, observation_date_col, account_status_col,
        outstanding_principal_col, cumulative_principal_paid_col,
    ]
    missing = [c for c in required if c not in recovery_df.columns]
    if missing:
        raise ValueError("历史回收法LGD缺少字段: %s" % missing)

    work = recovery_df.copy()
    work[observation_date_col] = pd.to_datetime(work[observation_date_col], errors="coerce")
    work[outstanding_principal_col] = _numeric(work[outstanding_principal_col])
    work[cumulative_principal_paid_col] = _numeric(work[cumulative_principal_paid_col])
    work[account_status_col] = _normalize_status(work[account_status_col])
    work = work.dropna(subset=[loan_id_col, observation_date_col]).sort_values(
        [loan_id_col, observation_date_col]
    )
    if work.empty:
        raise ValueError("历史回收明细没有有效贷款日期记录")
    cutoff = pd.Timestamp(cutoff_date) if cutoff_date else pd.Timestamp(work[observation_date_col].max())
    default_values = set(_normalize_status(pd.Series(default_status_values)).tolist())
    closed_values = set(_normalize_status(pd.Series(closed_status_values)).tolist())
    chargeoff_values = set(_normalize_status(pd.Series(chargeoff_status_values)).tolist())

    rows = []
    for loan_id, loan in work.groupby(loan_id_col, sort=False):
        default_rows = loan[loan[account_status_col].isin(default_values)]
        if default_rows.empty:
            continue
        first_default = default_rows.iloc[0]
        default_date = pd.Timestamp(first_default[observation_date_col])
        ead = float(first_default[outstanding_principal_col])
        paid_at_default = float(first_default[cumulative_principal_paid_col])
        if not np.isfinite(ead) or ead <= 0 or not np.isfinite(paid_at_default):
            rows.append({"loan_id": loan_id, "valid": False, "invalid_reason": "invalid_ead_or_paid"})
            continue

        twelve_month_end = default_date + pd.DateOffset(months=int(recovery_months))
        close_date = pd.NaT
        if close_date_col and close_date_col in loan.columns:
            close_candidates = pd.to_datetime(loan[close_date_col], errors="coerce").dropna()
            close_candidates = close_candidates[close_candidates >= default_date]
            if len(close_candidates):
                close_date = close_candidates.min()
        status_close_rows = loan[
            (loan[observation_date_col] >= default_date)
            & loan[account_status_col].isin(closed_values)
        ]
        if not status_close_rows.empty:
            status_close_date = pd.Timestamp(status_close_rows.iloc[0][observation_date_col])
            close_date = status_close_date if pd.isna(close_date) else min(close_date, status_close_date)

        planned_end = min(twelve_month_end, cutoff)
        if not pd.isna(close_date):
            planned_end = min(planned_end, close_date)
        observed = loan[
            (loan[observation_date_col] >= default_date)
            & (loan[observation_date_col] <= planned_end)
        ]
        if observed.empty:
            rows.append({"loan_id": loan_id, "valid": False, "invalid_reason": "no_post_default_record"})
            continue
        endpoint = observed.iloc[-1]
        endpoint_date = pd.Timestamp(endpoint[observation_date_col])
        paid_at_end = float(endpoint[cumulative_principal_paid_col])
        recovery = paid_at_end - paid_at_default
        negative_recovery = (not np.isfinite(recovery)) or recovery < -1e-9
        # 月末快照通常不会与“违约日+12个月”完全同日。只要统一截止日及
        # 该贷款的观测覆盖已经跨过12个月，即使用终点前最后一期快照并视为完成。
        completed_12m = bool(
            cutoff >= twelve_month_end
            and pd.Timestamp(loan[observation_date_col].max()) >= twelve_month_end
        )
        closed_within_window = (not pd.isna(close_date)) and close_date <= twelve_month_end
        completed = bool(completed_12m or closed_within_window)
        post_default_status = loan.loc[loan[observation_date_col] >= default_date, account_status_col]
        rows.append({
            "loan_id": loan_id,
            "valid": not negative_recovery,
            "invalid_reason": "negative_or_invalid_recovery" if negative_recovery else "",
            "default_date": default_date,
            "endpoint_date": endpoint_date,
            "observation_months": float((endpoint_date - default_date).days / 30.4375),
            "ead": ead,
            "recovery_principal": recovery,
            "completed": completed,
            "closed": bool(closed_within_window),
            "charged_off": bool(post_default_status.isin(chargeoff_values).any()),
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("未识别到与违约标签口径一致的历史违约贷款")
    valid = detail[detail.get("valid", False) == True].copy()  # noqa: E712
    completed = valid[valid["completed"]].copy()
    if completed.empty:
        raise ValueError("没有满足12个月观察期或已结清条件的有效LGD完成样本")

    base_lgd = 1.0 - float(completed["recovery_principal"].sum() / completed["ead"].sum())
    current_lgd = 1.0 - float(valid["recovery_principal"].sum() / valid["ead"].sum())
    if not 0.0 <= base_lgd <= 1.0:
        raise ValueError("历史回收法基准LGD超出[0,1]，请核查EAD、累计还本、核销或账户迁移字段")
    summary = pd.DataFrame([{
        "lgd_base": base_lgd,
        "lgd_current": current_lgd,
        "completed_count": len(completed),
        "incomplete_count": int(len(valid) - len(completed)),
        "invalid_count": int(len(detail) - len(valid)),
        "average_observation_months": float(valid["observation_months"].mean()),
        "closed_count": int(valid["closed"].sum()),
        "charged_off_count": int(valid["charged_off"].sum()),
        "total_ead_completed": float(completed["ead"].sum()),
        "total_recovery_completed": float(completed["recovery_principal"].sum()),
        "total_ead_all_valid": float(valid["ead"].sum()),
        "total_recovery_all_valid": float(valid["recovery_principal"].sum()),
        "cutoff_date": str(cutoff.date()),
    }])
    return base_lgd, summary, detail


def resolve_lgd(
    recovery_df: Optional[pd.DataFrame],
    lgd_kwargs: Optional[dict],
    scenario_values: Dict[str, float],
    selected_scenario: str,
):
    """优先使用历史回收法；无可靠明细时显式使用预设情景。"""
    if recovery_df is not None:
        value, summary, detail = estimate_historical_lgd(recovery_df, **(lgd_kwargs or {}))
        return float(value), "historical_recovery_12m", summary, detail
    if selected_scenario not in scenario_values:
        raise ValueError("LGD情景必须是 %s 之一" % sorted(scenario_values))
    value = float(scenario_values[selected_scenario])
    if not 0.0 <= value <= 1.0:
        raise ValueError("LGD情景值必须位于[0,1]")
    summary = pd.DataFrame([{
        "lgd_base": value,
        "lgd_current": np.nan,
        "source": "scenario_%s" % selected_scenario,
        "note": "缺少可靠首次违约/EAD/本金回收明细；该值为情景假设，不是本行历史经验LGD。",
    }])
    return value, "scenario_%s" % selected_scenario, summary, pd.DataFrame()


def derive_tier_limit_policy(
    customer_df: pd.DataFrame,
    talent_col: str,
    credit_limit_col: str,
    shrink_k: float = 500.0,
    overall_quantile: float = 0.99,
    grid_step: float = 500.0,
    grid_max: Optional[float] = None,
):
    """按技术文档的总体P99、历史上限收缩与E/D放缩计算F3~D上限。"""
    if talent_col not in customer_df or credit_limit_col not in customer_df:
        raise ValueError("等级上限估计缺少字段: %s" % [talent_col, credit_limit_col])
    work = customer_df[[talent_col, credit_limit_col]].copy()
    work["talent_level"] = _normalize_talent_levels(work[talent_col])
    work["credit_limit"] = _numeric(work[credit_limit_col])
    work = work[np.isfinite(work["credit_limit"]) & (work["credit_limit"] >= 0)].copy()
    if work.empty:
        raise ValueError("没有可用于等级额度上限估计的历史客户")
    if float(grid_step) <= 0:
        raise ValueError("grid_step 必须大于0")
    u0 = float(work["credit_limit"].quantile(float(overall_quantile)))
    if u0 <= 0:
        raise ValueError("历史额度P99必须大于0")

    rows = []
    raw_caps = {}
    previous = 0.0
    for level in (1, 2, 3):
        sub = work.loc[work["talent_level"] == level, "credit_limit"]
        n = int(len(sub))
        hist_max = float(sub.max()) if n else u0
        weight = float(n / (n + float(shrink_k))) if n else 0.0
        shrink_cap = (1.0 - weight) * u0 + weight * hist_max
        cap = shrink_cap if level == 1 else max(previous, shrink_cap)
        raw_caps[level] = cap
        previous = cap
        rows.append({
            "talent_level": level,
            "tier": LEVEL_TO_TIER[level],
            "sample_count": n,
            "historical_max": hist_max,
            "shrink_weight": weight,
            "overall_p99": u0,
            "raw_upper_limit": cap,
            "method": "sample_size_shrinkage",
        })
    raw_caps[4] = max(raw_caps[3], 1.1 * u0)
    raw_caps[5] = max(raw_caps[4], 1.2 * u0)
    for level, factor in ((4, 1.1), (5, 1.2)):
        sub = work.loc[work["talent_level"] == level, "credit_limit"]
        rows.append({
            "talent_level": level,
            "tier": LEVEL_TO_TIER[level],
            "sample_count": int(len(sub)),
            "historical_max": float(sub.max()) if len(sub) else np.nan,
            "shrink_weight": np.nan,
            "overall_p99": u0,
            "raw_upper_limit": raw_caps[level],
            "method": "max(previous_tier, %.1f*overall_p99)" % factor,
        })

    # 按技术文档仅要求非递减（允许相等）；向下取整到业务额度网格。
    rounded = {}
    previous = 0.0
    for level in range(1, 6):
        cap = np.floor(raw_caps[level] / float(grid_step)) * float(grid_step)
        if grid_max is not None:
            cap = min(cap, float(grid_max))
        cap = max(previous, cap, 0.0)
        rounded[level] = float(cap)
        previous = cap

    summary = pd.DataFrame(rows)
    summary["rounded_upper_limit"] = summary["talent_level"].map(rounded)
    min_limits = {level: 0.0 for level in range(1, 6)}
    return min_limits, rounded, summary


def build_c2_candidates(reference_limit: float, deltas: Iterable[float]):
    reference_limit = float(reference_limit)
    if reference_limit <= 0:
        raise ValueError("c2参考额度必须大于0")
    values = tuple(float(delta) / reference_limit for delta in deltas)
    if any(v < 0 for v in values):
        raise ValueError("c2候选必须非负")
    return values


def save_parameter_selection(
    result: ParameterSelectionResult,
    output_dir: str,
    utilization_summary: pd.DataFrame,
    utilization_detail: pd.DataFrame,
    lgd_summary: pd.DataFrame,
    lgd_detail: pd.DataFrame,
    tier_summary: pd.DataFrame,
):
    os.makedirs(output_dir, exist_ok=True)
    payload = asdict(result)
    payload["tier_min_limits"] = {str(k): v for k, v in result.tier_min_limits.items()}
    payload["tier_max_limits"] = {str(k): v for k, v in result.tier_max_limits.items()}
    payload["group_mean_min_ratio"] = {str(k): v for k, v in result.group_mean_min_ratio.items()}
    with open(os.path.join(output_dir, "derived_parameters.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    utilization_summary.to_csv(
        os.path.join(output_dir, "utilization_parameter_summary.csv"), index=False, encoding="utf-8-sig"
    )
    utilization_detail.to_csv(
        os.path.join(output_dir, "utilization_customer_detail.csv"), index=False, encoding="utf-8-sig"
    )
    lgd_summary.to_csv(
        os.path.join(output_dir, "lgd_parameter_summary.csv"), index=False, encoding="utf-8-sig"
    )
    if not lgd_detail.empty:
        lgd_detail.to_csv(
            os.path.join(output_dir, "lgd_loan_detail.csv"), index=False, encoding="utf-8-sig"
        )
    tier_summary.to_csv(
        os.path.join(output_dir, "tier_limit_parameter_summary.csv"), index=False, encoding="utf-8-sig"
    )
