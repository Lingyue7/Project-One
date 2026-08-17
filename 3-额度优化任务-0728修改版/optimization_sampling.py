#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""额度优化客户抽样清单。

模型训练、特征预处理和概率校准仍使用完整开发数据；本模块只决定哪些客户
需要生成候选额度概率网格并进入后续组合优化。
"""


from typing import Dict, Iterable

import numpy as np
import pandas as pd


TIER_TO_LEVEL = {
    "F3": 1, "F3级": 1, "F3档": 1,
    "F2": 2, "F2级": 2, "F2档": 2,
    "F1": 3, "F1级": 3, "F1档": 3,
    "E": 4, "E级": 4, "E档": 4, "E类": 4,
    "D": 5, "D级": 5, "D档": 5, "D类": 5,
}


def extract_talent_levels(
    customer_df: pd.DataFrame,
    candidates: Iterable[str] = ("档位", "talent_level"),
) -> np.ndarray:
    """从客户表提取技术文档定义的 F3/F2/F1/E/D 数值等级。"""
    column = next((name for name in candidates if name in customer_df.columns), None)
    if column is None:
        raise ValueError("优化抽样缺少人才等级字段（档位/talent_level）")

    raw = customer_df[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    # 档位字段在历史数据中可能出现小写 f3/f2/f1/e/d；统一大小写后再映射，
    # 与额度优化主程序既有的人才等级识别逻辑保持一致。
    text = raw.astype(str).str.strip().str.upper()
    mapped = text.map(TIER_TO_LEVEL)
    levels = numeric.where(numeric.notna(), mapped)
    bad = levels.isna() | ~np.isclose(levels.fillna(0), np.round(levels.fillna(0)))
    if bad.any():
        examples = raw.loc[bad].astype(str).unique()[:10].tolist()
        raise ValueError("无法识别人才等级，示例: %s" % examples)
    result = levels.round().astype(int).to_numpy()
    unsupported = sorted(set(result.tolist()) - set(range(1, 6)))
    if unsupported:
        raise ValueError("技术方案仅定义 F3/F2/F1/E/D 五档，发现: %s" % unsupported)
    return result


def stratified_sample_positions(
    strata,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """按人才等级近似等比例抽样，并在样本量允许时保留每个现有等级。"""
    strata = np.asarray(strata).reshape(-1)
    sample_size = int(sample_size)
    if sample_size <= 0:
        raise ValueError("sample_size 必须为正整数")
    if sample_size >= len(strata):
        return np.arange(len(strata), dtype=int)

    levels, counts = np.unique(strata, return_counts=True)
    if sample_size < len(levels):
        raise ValueError(
            "抽样数 %d 小于当前人才等级数 %d，无法保证每档至少一名客户。"
            % (sample_size, len(levels))
        )
    ideal = counts.astype(float) * sample_size / len(strata)
    allocations = np.minimum(counts, np.maximum(1, np.floor(ideal).astype(int)))
    while int(allocations.sum()) > sample_size:
        candidates = np.flatnonzero(allocations > 1)
        chosen = int(candidates[np.argmax(allocations[candidates] - ideal[candidates])])
        allocations[chosen] -= 1
    while int(allocations.sum()) < sample_size:
        candidates = np.flatnonzero(allocations < counts)
        chosen = int(candidates[np.argmax(ideal[candidates] - allocations[candidates])])
        allocations[chosen] += 1

    rng = np.random.RandomState(int(random_state))
    selected = []
    for level, allocation in zip(levels, allocations):
        positions = np.flatnonzero(strata == level)
        rng.shuffle(positions)
        selected.extend(positions[:int(allocation)].tolist())
    return np.sort(np.asarray(selected, dtype=int))


def sample_source_indices(
    source_indices,
    talent_levels,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """在给定原始行索引范围内分层抽样，返回原始行索引。"""
    source_indices = np.asarray(source_indices, dtype=int).reshape(-1)
    talent_levels = np.asarray(talent_levels, dtype=int).reshape(-1)
    if np.any(source_indices < 0) or np.any(source_indices >= len(talent_levels)):
        raise IndexError("优化抽样范围索引超出客户表边界")
    relative = stratified_sample_positions(
        talent_levels[source_indices], sample_size=sample_size, random_state=random_state
    )
    return np.sort(source_indices[relative])


def build_optimization_sample_plan(
    customer_df: pd.DataFrame,
    sample_size: int,
    random_state: int,
    fit_indices=None,
    test_indices=None,
) -> Dict[str, np.ndarray]:
    """创建 all 以及可选 fit/test 的固定抽样名单。"""
    levels = extract_talent_levels(customer_df)
    plan = {
        "all": sample_source_indices(
            np.arange(len(customer_df), dtype=int),
            levels,
            sample_size=sample_size,
            random_state=random_state,
        )
    }
    if fit_indices is not None:
        plan["fit"] = sample_source_indices(
            fit_indices, levels, sample_size=sample_size, random_state=random_state
        )
    if test_indices is not None:
        plan["test"] = sample_source_indices(
            test_indices, levels, sample_size=sample_size, random_state=random_state
        )
    return plan
