#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科创潜在客户数据预处理模块
==============================
适配新的 SAS 取数结构（科创人才 ∩ 消贷账户），与原 load_kechuang_data.py 流程对齐。

字段变化说明：
  - SAS 取数中科创人才字段已改为英文别名，读取后第一步恢复为中文列名
  - 支持两种因变量（通过 target 参数切换）：
      y_freq    : 频率活跃度标签（多种构造模式，详见 build_labels_potential）
      y_dq_risk : 违约风险标签（rt_acct_stat_2_end IN ('3','4','7','9') → 1）
  - y_dq_risk 模式特有流程：
      ① 聚合前过滤：rt_acct_stat_2 IN ('3','4','7','9') 的贷款对应的整个 cst_id 剔除
      ② 聚合后对 rt_acct_stat_2_end 取最差值，IN ('3','4','7','9') → 1，否则 → 0
      ③ 构造完 y_dq_risk 后删除 rt_acct_stat_2_end（防泄露）
  - y_freq 模式：在 Part1 清洗时直接删除 rt_acct_stat_2_end（不需要此字段）

泄露字段（drop_post_label_cols 中删除）：
  ac_curr_bal_diff、ac_accr_bal_diff、ba_out_bal_diff（构造因变量后删除）
  rt_acct_stat_2、time_of_dq、dq_hist_day_ctr、dq_num_pmts_pdue、
  du_pmt_rem_1、rt_actl_pmts_rem、bal_of_int_rev、rt_acct_eff_date_1、
  ac_curr_bal、ac_accr_bal、ba_out_bal（贷款起点原始字段，除 rt_loan_rate /
  tot_num_payment_1 / zm_memo2_amt2_4 外均删除）
  rt_acct_stat_2_end（y_dq_risk 构造后删除；y_freq 模式在 Part1 删除）

保留的贷款起点字段（有建模意义）：
  rt_loan_rate、tot_num_payment_1、zm_memo2_amt2_4

主要流程：
  1.   read_data                        —— 读取 CSV 或 Excel
  2.   rename_kechuang_cols             —— 英文科创字段名恢复为中文
  3.   filter_by_maturity               —— [可选] 按到期日过滤
  3.5  filter_by_eff_date               —— [可选] 按贷款生效日过滤
  3.7  filter_dq_start_customers        —— [仅 y_dq_risk] 剔除起点已违约客户
  4.   aggregate_by_customer_potential  —— 一客多贷聚合
  5.   clean_data_potential             —— 字段重命名、删除冗余/ID/日期字段
  6.   build_labels_potential           —— 因变量构造（y_freq 或 y_dq_risk）
  7.   drop_post_label_cols             —— 删除泄露字段/diff字段
  8.   get_feature_stats                —— 特征缺失率统计（供 notebook 展示）
  9.   feature_engineering_potential    —— 缺失删列/填充、LabelEncoder、档位有序编码、授信额度多项式
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def _mode_or_nan(x: pd.Series):
    m = x.mode()
    return m.iloc[0] if len(m) > 0 else np.nan


def _worst_status(x: pd.Series):
    """
    一客多贷账户状态聚合：取最差状态码。
    优先级：9 > 7 > 4 > 3 > 其他
    只要有任意一笔是不良状态（3/4/7/9），客户就被标记为不良。

    注意：字段从 Excel/CSV 读入后可能是浮点型（如 3.0），
    此处先转 float 再转 int 再转 str，确保 "3.0" → "3"。
    """
    def _to_str(v):
        try:
            return str(int(float(v)))
        except (ValueError, TypeError):
            return str(v).strip()

    vals = x.map(_to_str)
    for bad in ["9", "7", "4", "3"]:
        if bad in vals.values:
            return bad
    m = vals.mode()
    return m.iloc[0] if len(m) > 0 else np.nan


# ─────────────────────────────────────────────
# Step 0：读取文件（CSV / Excel 均支持）
# ─────────────────────────────────────────────

def read_data(file_path: str, csv_encoding: str = "utf-8-sig") -> pd.DataFrame:
    """
    支持 .csv / .xlsx / .xls 三种格式。

    参数
    ----
    file_path    : 文件路径
    csv_encoding : CSV 编码，默认 utf-8-sig（兼容 Excel 另存的 CSV 中文）
                   若乱码可改为 "gbk" 或 "gb18030"

    返回
    ----
    df : 原始 DataFrame
    """
    ext = file_path.lower().rsplit(".", 1)[-1]
    if ext == "csv":
        df = pd.read_csv(file_path, encoding=csv_encoding)
        print(f"  已读取 CSV：{file_path}  编码={csv_encoding}")
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(file_path)
        print(f"  已读取 Excel：{file_path}")
    else:
        raise ValueError(f"不支持的文件格式：{ext}，仅支持 csv / xlsx / xls")

    print(f"  原始数据：{df.shape[0]:,} 行，{df.shape[1]:,} 列")
    return df


# ─────────────────────────────────────────────
# Step 0.05：按 (cst_id, loanacctno) 去重
# ─────────────────────────────────────────────

def dedup_by_cst_loan(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 (cst_id, loanacctno) 组合去重，保留第一条。
    必须在所有筛选和聚合之前、列名小写化之后执行。

    逻辑：
      同一客户同一贷款账户出现多行，是拉链表在 SQL 层面未完全去净的残留。
      此处在 Python 层面做最终兜底，确保后续聚合的行级数据干净。
    """
    df = df.copy()

    loan_col = next(
        (c for c in ["loanacctno", "loan_acct_no"] if c in df.columns), None
    )

    n_before = len(df)

    if loan_col is None or "cst_id" not in df.columns:
        print("  ⚠️  未找到 cst_id 或 loanacctno 字段，跳过去重")
        return df

    df = df.drop_duplicates(subset=["cst_id", loan_col], keep="first")
    n_after = len(df)

    print(f"\n  ── (cst_id, loanacctno) 去重 ──")
    print(f"     去重前行数 : {n_before:,}")
    print(f"     去重后行数 : {n_after:,}  （删除 {n_before - n_after:,} 条重复行）")
    print(f"     唯一客户数 : {df['cst_id'].nunique():,}")
    print(f"     唯一账户数 : {df[loan_col].nunique():,}")
    return df


# ─────────────────────────────────────────────
# Step 0.08：类别型字段强制转换（聚合前，必须在 filter_dq_start_customers 之前）
# ─────────────────────────────────────────────

def cast_cat_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    将从 Excel/CSV 读入后被误识别为浮点型的类别字段，
    统一转换为整数字符串（如 3.0 → "3"）。

    必须在聚合和起点违约剔除之前执行：
      - filter_dq_start_customers 用 rt_acct_stat_2 做字符串匹配，
        若此时字段仍为 "3.0" 则匹配不上 "3"，导致起点违约客户漏判，
        全部错误保留进样本，正样本虚高。
      - _worst_status 在聚合时同样依赖字符串比较，虽然内部有兜底，
        但提前转换更安全。
    """
    df = df.copy()

    cat_int_cols = [
        "mar_sttn_cd",
        "cst_star_cd",
        "age",
        "occup_cd",
        "gnd_cd",
        "rt_acct_stat_2_end",
        "rt_acct_stat_2",
    ]

    def _float_to_intstr(v):
        try:
            return str(int(float(v)))
        except (ValueError, TypeError):
            return np.nan

    converted = []
    for col in cat_int_cols:
        if col in df.columns:
            df[col] = df[col].map(_float_to_intstr)
            converted.append(col)

    if converted:
        print(f"  已将类别型字段转为整数字符串（共 {len(converted)} 个）: {converted}")
    return df


# ─────────────────────────────────────────────
# Step 0.1：英文科创字段名恢复为中文
# ─────────────────────────────────────────────

def rename_kechuang_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    SAS 取数中科创人才字段已改为英文别名，此处恢复为中文列名，
    避免后续字段处理出错（聚合规则、特征重要性展示等均依赖中文列名）。

    映射关系（与 SAS SQL 中 AS 对应）：
      cur_aum          → 当前aum
      cur_lum          → 当前lum
      tech_talent_score → 科技人才对应得分
      kum_score        → kum分
      lum_score        → lum分
      aum_score        → aum分
      total_score      → 总分
      tier             → 档位
    """
    df = df.copy()
    # 统一列名小写后再做映射
    df.columns = [c.lower() for c in df.columns]

    rename_map = {
        "cur_aum":           "当前aum",
        "cur_lum":           "当前lum",
        "tech_talent_score": "科技人才对应得分",
        "kum_score":         "kum分",
        "lum_score":         "lum分",
        "aum_score":         "aum分",
        "total_score":       "总分",
        "tier":              "档位",
    }
    actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    if actual_rename:
        df = df.rename(columns=actual_rename)
        print(f"  科创字段已恢复中文列名: {list(actual_rename.values())}")
    else:
        print("  ⚠️  未找到英文科创字段，跳过重命名（可能已是中文列名）")
    return df


# ─────────────────────────────────────────────
# Step 0.5：到期日筛选（可选）
# ─────────────────────────────────────────────

def filter_by_maturity(
    df_raw: pd.DataFrame,
    apply_filter: bool = True,
    maturity_cutoff: str = "2026-05-31",
) -> pd.DataFrame:
    """
    按贷款到期日（rt_curr_matur_date_1）筛选账户行。

    逻辑：
      - 保留 rt_curr_matur_date_1 >= maturity_cutoff 的账户行
      - 筛选前后分别汇报 总客户数（cst_id）、总贷款账号数（loanacctno）

    参数
    ----
    df_raw         : 原始行级 DataFrame（每行一个贷款账户，列名已小写）
    apply_filter   : True = 执行筛选；False = 跳过，原样返回
    maturity_cutoff: 到期日阈值（含），格式 "YYYY-MM-DD"

    返回
    ----
    df : 筛选后的 DataFrame
    """
    df = df_raw.copy()

    n_rows_before = len(df)
    n_cust_before = df["cst_id"].nunique() if "cst_id" in df.columns else None
    _loan_col = next((c for c in ["loanacctno", "loan_acct_no", "loanacct_no"]
                      if c in df.columns), None)
    n_loan_before = df[_loan_col].nunique() if _loan_col else None

    def _fmt(v):
        return f"{v:,}" if isinstance(v, (int, float)) else "N/A（列不存在）"

    print(f"\n  ── 到期日筛选前 ──")
    print(f"     账户行数  : {n_rows_before:,}")
    print(f"     客户数    : {_fmt(n_cust_before)}")
    print(f"     贷款账号数: {_fmt(n_loan_before)}")

    if not apply_filter:
        print(f"  apply_filter=False，跳过到期日筛选，原样返回")
        return df

    if "rt_curr_matur_date_1" not in df.columns:
        print(f"  ⚠️  未找到字段 rt_curr_matur_date_1，跳过到期日筛选")
        return df

    # 转换到期日字段：兼容 "31MAY2026"（SAS 格式）和标准日期格式
    df["rt_curr_matur_date_1"] = pd.to_datetime(
        df["rt_curr_matur_date_1"], format="%d%b%Y", errors="coerce"
    )
    mask_failed = df["rt_curr_matur_date_1"].isna()
    if mask_failed.any():
        df.loc[mask_failed, "rt_curr_matur_date_1"] = pd.to_datetime(
            df_raw.copy().rename(columns=str.lower)["rt_curr_matur_date_1"][mask_failed],
            errors="coerce"
        )

    cutoff_dt = pd.Timestamp(maturity_cutoff)
    df = df[df["rt_curr_matur_date_1"] >= cutoff_dt].copy()

    n_rows_after = len(df)
    n_cust_after = df["cst_id"].nunique() if "cst_id" in df.columns else None
    n_loan_after = df[_loan_col].nunique() if _loan_col else None

    def _diff(a, b):
        return f"{a - b:,}" if isinstance(a, int) and isinstance(b, int) else "N/A"

    print(f"\n  ── 到期日筛选后（rt_curr_matur_date_1 >= {maturity_cutoff}）──")
    print(f"     账户行数  : {n_rows_after:,}  （减少 {n_rows_before - n_rows_after:,} 行）")
    print(f"     客户数    : {_fmt(n_cust_after)}  （减少 {_diff(n_cust_before, n_cust_after)} 位）")
    print(f"     贷款账号数: {_fmt(n_loan_after)}  （减少 {_diff(n_loan_before, n_loan_after)} 个）")
    return df


# ─────────────────────────────────────────────
# Step 0.6：贷款生效日筛选（新增）
# ─────────────────────────────────────────────

def filter_by_eff_date(
    df: pd.DataFrame,
    eff_date_lower: str = "2024-10-01",
    eff_date_upper: str = "2026-03-31",
) -> pd.DataFrame:
    """
    按贷款生效日（rt_acct_eff_date_1）筛选行级数据，必须在聚合之前执行。

    逻辑：
      保留 eff_date_lower <= rt_acct_eff_date_1 <= eff_date_upper 的贷款行。

    目的：
      手机银行表和消费偏好表的历史快照有时间覆盖限制，
      X_SNAP_DT（贷款生效月上个月月底）若早于这些表的最早快照，
      特征将全部缺失。通过限制生效日范围可消除系统性缺失。

    参数
    ----
    df             : 行级 DataFrame（每行一笔贷款，列名已小写）
    eff_date_lower : 生效日下限（含），默认 "2024-10-01"
                     对应消费偏好表最早快照 2024-09-30（X_SNAP_DT>=2024-09-30）
    eff_date_upper : 生效日上限（含），默认 "2026-03-31"
                     对应消费偏好表最晚快照 2026-02-28（X_SNAP_DT<=2026-02-28）
    """
    df = df.copy()

    if "rt_acct_eff_date_1" not in df.columns:
        print("  ⚠️  未找到字段 rt_acct_eff_date_1，跳过生效日筛选")
        return df

    n_rows_before = len(df)
    n_cust_before = df["cst_id"].nunique() if "cst_id" in df.columns else None
    _loan_col = next((c for c in ["loanacctno", "loan_acct_no"] if c in df.columns), None)
    n_loan_before = df[_loan_col].nunique() if _loan_col else None

    def _fmt(v):
        return f"{v:,}" if isinstance(v, (int, float)) else "N/A"

    print(f"\n  ── 贷款生效日筛选前 ──")
    print(f"     账户行数  : {n_rows_before:,}")
    print(f"     客户数    : {_fmt(n_cust_before)}")
    print(f"     贷款账号数: {_fmt(n_loan_before)}")

    # 转换生效日（兼容 SAS 格式和标准格式）
    eff_col = df["rt_acct_eff_date_1"].copy()
    eff_parsed = pd.to_datetime(eff_col, format="%d%b%Y", errors="coerce")
    mask_failed = eff_parsed.isna()
    if mask_failed.any():
        eff_parsed[mask_failed] = pd.to_datetime(eff_col[mask_failed], errors="coerce")
    df["rt_acct_eff_date_1"] = eff_parsed

    lower_dt = pd.Timestamp(eff_date_lower)
    upper_dt = pd.Timestamp(eff_date_upper)
    df = df[
        (df["rt_acct_eff_date_1"] >= lower_dt) &
        (df["rt_acct_eff_date_1"] <= upper_dt)
    ].copy()

    n_rows_after = len(df)
    n_cust_after = df["cst_id"].nunique() if "cst_id" in df.columns else None
    n_loan_after = df[_loan_col].nunique() if _loan_col else None

    print(f"\n  ── 贷款生效日筛选后（{eff_date_lower} ~ {eff_date_upper}）──")
    print(f"     账户行数  : {n_rows_after:,}  （减少 {n_rows_before - n_rows_after:,} 行）")
    print(f"     客户数    : {_fmt(n_cust_after)}")
    print(f"     贷款账号数: {_fmt(n_loan_after)}")
    return df


# ─────────────────────────────────────────────
# Step 0.7：剔除起点已违约客户（仅 y_dq_risk 模式）
# ─────────────────────────────────────────────

def filter_dq_start_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    仅在 target='y_dq_risk' 时调用，必须在聚合之前执行。

    逻辑：
      rt_acct_stat_2 为贷款生效日时的账户状态（起点快照）。
      若某笔贷款的 rt_acct_stat_2 IN ('3','4','7','9')，
      则该贷款对应的 cst_id 整体从数据中删除（包含该客户所有贷款行）。
      一客多贷场景下只要有任意一笔起点违约，整个客户移除。

    参数
    ----
    df : 行级 DataFrame（列名已小写，每行一笔贷款），需含 cst_id、rt_acct_stat_2

    返回
    ----
    df : 剔除起点违约客户后的 DataFrame
    """
    df = df.copy()

    if "rt_acct_stat_2" not in df.columns:
        print("  ⚠️  未找到字段 rt_acct_stat_2，跳过起点违约客户剔除")
        return df

    DQ_STATUS = {"3", "4", "7", "9"}
    stat_str = df["rt_acct_stat_2"].astype(str).str.strip()
    bad_cst_ids = set(df.loc[stat_str.isin(DQ_STATUS), "cst_id"].unique())

    n_cust_before = df["cst_id"].nunique()
    n_rows_before = len(df)
    df = df[~df["cst_id"].isin(bad_cst_ids)].copy()
    n_cust_after = df["cst_id"].nunique()
    n_rows_after = len(df)

    print(f"\n  ── 起点违约客户剔除（y_dq_risk 模式）──")
    print(f"     剔除前客户数 : {n_cust_before:,}  行数: {n_rows_before:,}")
    print(f"     起点违约客户 : {len(bad_cst_ids):,} 位（rt_acct_stat_2 ∈ {{3,4,7,9}}）")
    print(f"     剔除后客户数 : {n_cust_after:,}  行数: {n_rows_after:,}")
    return df


# ─────────────────────────────────────────────
# Step 1：一客多贷聚合
# ─────────────────────────────────────────────

def aggregate_by_customer_potential(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    按 cst_id 对一客多贷数据做聚合，压成一行。

    字段说明：
      - ac_curr_bal_diff / ba_out_bal_diff / ac_accr_bal_diff：差值字段 → sum 聚合
      - num_of_time_dq_diff / dq_principal_diff：sum 聚合（聚合后在 clean 阶段删除）
      - rt_acct_stat_2：取最差状态码
      - credamt：排除状态码 5/8 的行后求和
      - 科创人才字段（中文列名）：first 聚合
    """
    df = df_raw.copy()

    # ── 开户日期副本（供概率校准的时间切分使用，不参与建模） ──
    # 原始 rt_acct_eff_date_1 会在后续 clean/drop 阶段被删除（防泄露），
    # 这里额外复制一列 split_eff_date 并解析为 datetime，一客多贷取最早一笔的开户日。
    # 该列为 datetime 类型，在 feature_engineering_potential 中会被显式删除，
    # 因此不会进入模型，仅保留在 work/data_cleaned.csv 中供切分排序。
    if "rt_acct_eff_date_1" in df.columns:
        _eff = pd.to_datetime(df["rt_acct_eff_date_1"], format="%d%b%Y", errors="coerce")
        _failed = _eff.isna()
        if _failed.any():
            _eff[_failed] = pd.to_datetime(df["rt_acct_eff_date_1"][_failed], errors="coerce")
        df["split_eff_date"] = _eff

    # ── 授信额度：排除状态码 5/8 的行后再求和 ──
    CREDAMT_EXCLUDE_STATUS = {"5", "8"}
    if "credamt" in df.columns and "rt_acct_stat_2" in df.columns:
        df["credamt"] = df.apply(
            lambda row: np.nan
            if str(row["rt_acct_stat_2"]).strip() in CREDAMT_EXCLUDE_STATUS
            else row["credamt"],
            axis=1,
        )

    # ── 各字段聚合规则 ──

    # 差值字段：求和（含 num_of_time_dq_diff / dq_principal_diff，聚合后统一删除）
    bal_diff_sum_cols = [
        "ac_curr_bal_diff", "ba_out_bal_diff", "ac_accr_bal_diff",
        "num_of_time_dq_diff", "dq_principal_diff",
    ]

    # 账户状态：取最差
    loan_special_agg = {
        "rt_acct_stat_2":     _worst_status,
        "rt_acct_stat_2_end": _worst_status,  # 终点账户状态，用于构造 y_dq_risk
    }

    # 授信台账
    credit_sum_cols   = ["credamt"]
    credit_first_cols = ["busikind", "acctbegindate"]

    # 保留的贷款起点字段（rt_loan_rate / tot_num_payment_1 / zm_memo2_amt2_4 / rt_actl_pmts_rem 已移除）
    loan_keep_first_cols = []

    # 客户静态信息
    cust_cols_mode  = ["cst_blng_insid", "gnd_cd", "mar_sttn_cd",
                       "education_cd", "occup_cd", "cst_star_cd"]
    cust_cols_first = ["age", "bank_cust_become_date"]

    # 科创人才字段（中文列名，rename 后已恢复）
    kechuang_first_cols = [
        "当前aum", "当前lum", "科技人才对应得分",
        "kum分", "lum分", "aum分", "总分", "档位",
    ]

    # 消费偏好：rt12* 均值
    pref_cols = [c for c in df.columns if str(c).lower().startswith("rt12")]

    # 手机银行统计：first（客户级）
    mpb_first_cols = [
        "lblclbrmpbactvcst_ind",
        "yr_acm_mpb_land_cnt", "lmth_acm_mpb_land_cnt",
        "acgmocrr3mampblandcnt", "acgmocr12mampblandcnt",
        "mo_acm_mpb_land_dys", "acgmocrr12mampblmonum",
        "yr_acm_mpb_land_monum", "lastyracmmpblandmonum",
        "moacm_mpb_fncltx_dnum", "mo_acm_mpb_fncltx_amt",
        "acgmocrr3mampblanddys", "yr_acm_mpb_land_dys",
        "yracm_mpb_fncltx_dnum", "yr_acm_mpb_fncltx_amt",
        "acgmocrr3mampbftxdnum", "acgmoclrr3mampbftxamt",
        "acgmocrr6mampbftxdnum", "acgmoclrr6mampbftxamt",
    ]

    agg_dict: Dict[str, object] = {}

    for c in bal_diff_sum_cols:
        if c in df.columns:
            agg_dict[c] = "sum"

    for c, func in loan_special_agg.items():
        if c in df.columns:
            agg_dict[c] = func

    for c in credit_sum_cols:
        if c in df.columns:
            agg_dict[c] = "sum"

    for c in credit_first_cols + loan_keep_first_cols:
        if c in df.columns:
            agg_dict[c] = "first"

    # 开户日期副本：一客多贷取最早开户日（min），供时间切分排序
    if "split_eff_date" in df.columns:
        agg_dict["split_eff_date"] = "min"

    for c in cust_cols_mode:
        if c in df.columns:
            agg_dict[c] = _mode_or_nan

    for c in cust_cols_first:
        if c in df.columns:
            agg_dict[c] = "first"

    for c in kechuang_first_cols:
        if c in df.columns:
            agg_dict[c] = "first"

    for c in pref_cols:
        if c in df.columns:
            agg_dict[c] = "mean"

    for c in mpb_first_cols:
        if c in df.columns:
            agg_dict[c] = "first"

    # 其余字段默认 first（排除 cst_id 与明确不需要聚合的字段）
    # 注意：loanacctno / acct_eff_dt / matur_dt / x_snap_dt 等 ID/日期字段
    # 在 clean_data_potential 中删除，此处 first 聚合后仍会进入下一步被删除
    skip_cols = {"cst_id"}
    other_cols = [
        c for c in df.columns
        if c not in agg_dict
        and c not in skip_cols
    ]
    for c in other_cols:
        agg_dict[c] = "first"

    df_agg = df.groupby("cst_id", as_index=False).agg(agg_dict)

    print(f"  聚合完成：{len(df_agg):,} 位客户，{df_agg.shape[1]} 个字段")
    return df_agg


# ─────────────────────────────────────────────
# Step 2a：清洗 Part1（因变量构造前）
# ─────────────────────────────────────────────

def clean_data_potential(
    df: pd.DataFrame,
    snapshot_date: str = "2026-01-31",
    target: str = "y_freq",
) -> pd.DataFrame:
    """
    清洗 Part1（在因变量构造之前执行）：
      - cst_id 转字符串
      - 日期列转 datetime，构造 days_* 衍生特征后删除原始日期列
      - 删除所有 ID / 日期 / 无建模意义字段（含新取数结构新增的字段）
      - y_freq 模式：直接删除 rt_acct_stat_2_end（不需要此字段）
      - y_dq_risk 模式：保留 rt_acct_stat_2_end（供 build_labels_potential 使用）
      - 不做缺失值填充（由 feature_engineering 处理）
    """
    df = df.copy()

    # ── cst_id 统一为字符串 ──
    df["cst_id"] = df["cst_id"].astype(str)

    # ── 数值型读入但实为类别型的字段：去掉 .0 后转字符串 ──
    # 这些字段从 Excel/CSV 读入后为 float（如 3.0），需还原为整数字符串（"3"）
    cat_int_cols = [
        "mar_sttn_cd", "cst_star_cd", "age",
        "occup_cd", "gnd_cd", "rt_acct_stat_2_end",
        "rt_acct_stat_2",
    ]
    for col in cat_int_cols:
        if col in df.columns:
            def _float_to_intstr(v):
                try:
                    return str(int(float(v)))
                except (ValueError, TypeError):
                    return np.nan
            df[col] = df[col].map(_float_to_intstr)
    print(f"  已将类别型字段转为整数字符串: "
          f"{[c for c in cat_int_cols if c in df.columns]}")

    # ── 日期列转 datetime ──
    date_cols = ["acctbegindate", "bank_cust_become_date", "acct_eff_dt", "matur_dt"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── days_* 衍生特征 ──
    snap = pd.Timestamp(snapshot_date)
    if "bank_cust_become_date" in df.columns:
        df["days_since_become_cust"] = (snap - df["bank_cust_become_date"]).dt.days
        print("  已生成 days_since_become_cust")

    # ── 删除所有不进入建模的字段 ──
    pre_drop_cols = [
        # ── 新取数结构新增的 ID / 时间戳字段 ──
        "loanacctno",           # 贷款账号（行级 ID）
        "acct_eff_dt",          # 贷款开始日期
        "matur_dt",             # 贷款到期日期
        "x_snap_dt",            # X 特征快照时点
        # ── 旧结构 ID 字段（兼容保留） ──
        "cst_id0",
        "rt_ctl3",
        "rt_ctl4",
        "rt_acct_num",
        # ── 日期字段（已转为 days_* 或无建模意义） ──
        "bank_cust_become_date",
        "acctbegindate",
        "rt_curr_matur_date_1",
        # ── 产品代码 / 无建模意义 ──
        "amwkpl08_filler",
        "busikind",
        # ── 消费偏好表带入的冗余字段 ──
        "data_dt",
        "multi_tenancy_id",
        # ── 账户状态（不构造 y_dq_risk，此字段无需保留） ──
        "rt_acct_stat_2",
    ]
    # y_freq 模式下 rt_acct_stat_2_end 无用，在 Part1 直接删除
    if target == "y_freq":
        pre_drop_cols.append("rt_acct_stat_2_end")
    existing_pre = [c for c in pre_drop_cols if c in df.columns]
    if existing_pre:
        df = df.drop(columns=existing_pre)
        print(f"  Part1 已删除 ID/日期/无意义字段（共 {len(existing_pre)} 个）: {existing_pre}")

    print(f"  清洗Part1完成：{len(df):,} 行，{df.shape[1]} 列")
    return df


# ─────────────────────────────────────────────
# Step 2b：清洗 Part2（因变量构造后）
# ─────────────────────────────────────────────

def drop_post_label_cols(df: pd.DataFrame, target: str = "y_freq") -> pd.DataFrame:
    """
    清洗 Part2（在因变量构造之后执行）：
    删除以下字段：
      - 因变量来源 diff 字段（构造完标签后必须删除）：
          ac_curr_bal_diff、ac_accr_bal_diff、ba_out_bal_diff
      - 无意义的 diff 字段：
          num_of_time_dq_diff、dq_principal_diff
      - 起终点校验值（仅供人工校验，不进入建模）：
          ac_curr_bal_start、ac_curr_bal_end、ba_out_bal_start、ba_out_bal_end
      - 贷款起点原始字段中无建模意义的字段（保留 rt_loan_rate / tot_num_payment_1 / zm_memo2_amt2_4）
      - y_dq_risk 模式额外删除 rt_acct_stat_2_end（构造标签后防泄露）
    """
    df = df.copy()

    post_drop_cols = [
        # 因变量来源 diff（构造 y_freq 后删除）
        "ac_curr_bal_diff",
        "ac_accr_bal_diff",
        "ba_out_bal_diff",
        # 无意义 diff 字段
        "num_of_time_dq_diff",
        "dq_principal_diff",
        # 起终点校验值
        "ac_curr_bal_start",
        "ac_curr_bal_end",
        "ba_out_bal_start",
        "ba_out_bal_end",
        # 贷款起点原始字段（rt_loan_rate / tot_num_payment_1 / zm_memo2_amt2_4 / rt_actl_pmts_rem 一并删除）
        "rt_acct_eff_date_1",
        "rt_loan_rate",
        "tot_num_payment_1",
        "zm_memo2_amt2_4",
        "rt_actl_pmts_rem",
        "ac_curr_bal",
        "ac_accr_bal",
        "time_of_dq",
        "dq_principal",
        "num_of_time_dq",
        "du_pmt_rem_1",
        "dq_hist_day_ctr",
        "dq_num_pmts_pdue",
        "ba_out_bal",
        "bal_of_int_rev",
        # 兜底：其他泄露字段
        "ba_avail_bal",
        "ac_amt_pd_itd",
    ]
    # y_dq_risk 模式：构造标签后删除 rt_acct_stat_2_end（防泄露）
    if target == "y_dq_risk":
        post_drop_cols.append("rt_acct_stat_2_end")
    existing_post = [c for c in post_drop_cols if c in df.columns]
    if existing_post:
        df = df.drop(columns=existing_post)
        print(f"  Part2 已删除泄露/diff/无意义字段（共 {len(existing_post)} 个）: {existing_post}")

    print(f"  清洗Part2完成：{len(df):,} 行，{df.shape[1]} 列")
    return df


# ─────────────────────────────────────────────
# Step 3：因变量构造 + 分布统计
# ─────────────────────────────────────────────

def build_labels_potential(
    df: pd.DataFrame,
    target: str = "y_freq",
    y_freq_mode: str = "bout_gt0_and_curr_p80",
    threshold_fit_mask=None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    构造因变量，根据 target 参数分支：

    target='y_freq'：支持四种构造模式（y_freq_mode 参数）：
      bout_gt0_and_curr_p80  : ba_out_bal_diff>0 且 ac_curr_bal_diff>=P80（默认）
      bout_p80_and_accr_p80  : ba_out_bal_diff>=P80 且 ac_accr_bal_diff>=P80
      curr_p80_only          : ac_curr_bal_diff>=P80（单条件）
      curr_p80_and_bout_p80  : ac_curr_bal_diff>=P80 且 ba_out_bal_diff>=P80

    target='y_dq_risk'：
      rt_acct_stat_2_end（聚合后最差终点账户状态）IN ('3','4','7','9') → 1，否则 → 0
    """
    work = df.copy()
    n_total = len(work)
    if threshold_fit_mask is None:
        threshold_source = work
        threshold_scope = "all rows"
    else:
        fit_mask = np.asarray(threshold_fit_mask, dtype=bool)
        if len(fit_mask) != len(work):
            raise ValueError("threshold_fit_mask length must equal len(df)")
        if not fit_mask.any():
            raise ValueError("threshold_fit_mask selects zero rows")
        threshold_source = work.loc[fit_mask]
        threshold_scope = "train only (%d rows)" % len(threshold_source)

    thresholds: Dict[str, float] = {
        "target":      target,
        "y_freq_mode": y_freq_mode,
        "thr_bout":    float("nan"),
        "thr_curr":    float("nan"),
        "thr_accr":    float("nan"),
    }

    # ════════════════════════════════════════
    # y_dq_risk 分支
    # ════════════════════════════════════════
    if target == "y_dq_risk":
        if "rt_acct_stat_2_end" not in work.columns:
            raise ValueError("y_dq_risk 模式需要字段 rt_acct_stat_2_end，但该字段不存在。"
                             "请确认 SAS 取数已包含 RT_ACCT_STAT_2_END 字段。")
        DQ_STATUS = {"3", "4", "7", "9"}
        # 与 _worst_status 保持一致：float→int→str，防止 "3.0" 匹配不上 "3"
        def _to_clean_str_label(v):
            try:
                return str(int(float(v)))
            except (ValueError, TypeError):
                return str(v).strip()
        stat_str = work["rt_acct_stat_2_end"].map(_to_clean_str_label)
        work["y_dq_risk"] = stat_str.isin(DQ_STATUS).astype(int)
        n_pos = int(work["y_dq_risk"].sum())
        print(f"\n  {'─'*55}")
        print(f"  y_dq_risk 构造统计")
        print(f"  {'─'*55}")
        print(f"  rt_acct_stat_2_end ∈ {{3,4,7,9}} → 1 : {n_pos:>6,} 人  ({n_pos/n_total:.2%})")
        print(f"  {'─'*55}")
        return work, thresholds

    # ════════════════════════════════════════
    # y_freq 分支
    # ════════════════════════════════════════
    VALID_MODES = {
        "bout_gt0_and_curr_p80",
        "bout_p80_and_accr_p80",
        "curr_p80_only",
        "curr_p80_and_bout_p80",
    }
    if y_freq_mode not in VALID_MODES:
        raise ValueError(f"y_freq_mode 仅支持 {VALID_MODES}，当前传入: '{y_freq_mode}'")

    thr_bout = float("nan")
    thr_curr = float("nan")
    thr_accr = float("nan")

    print(f"\n  {'─'*55}")
    print(f"  y_freq 构造统计（模式: {y_freq_mode}）")
    print(f"  P80 阈值拟合范围: {threshold_scope}")
    print(f"  {'─'*55}")

    if y_freq_mode == "bout_gt0_and_curr_p80":
        required = ["ba_out_bal_diff", "ac_curr_bal_diff"]
        missing = [c for c in required if c not in work.columns]
        if missing:
            raise ValueError(f"模式 '{y_freq_mode}' 缺少字段: {missing}")
        thr_curr  = float(threshold_source["ac_curr_bal_diff"].quantile(0.80))
        cond_bout = work["ba_out_bal_diff"] > 0
        cond_curr = work["ac_curr_bal_diff"] >= thr_curr
        work["y_freq"] = (cond_bout & cond_curr).astype(int)
        print(f"  条件①  ba_out_bal_diff > 0                          : "
              f"{int(cond_bout.sum()):>6,} 人  ({cond_bout.mean():.2%})")
        print(f"  条件②  ac_curr_bal_diff >= {thr_curr:.2f}（P80）: "
              f"{int(cond_curr.sum()):>6,} 人  ({cond_curr.mean():.2%})")

    elif y_freq_mode == "curr_p80_only":
        required = ["ac_curr_bal_diff"]
        missing = [c for c in required if c not in work.columns]
        if missing:
            raise ValueError(f"模式 '{y_freq_mode}' 缺少字段: {missing}")
        thr_curr  = float(threshold_source["ac_curr_bal_diff"].quantile(0.80))
        cond_curr = work["ac_curr_bal_diff"] >= thr_curr
        work["y_freq"] = cond_curr.astype(int)
        print(f"  条件    ac_curr_bal_diff >= {thr_curr:.2f}（P80）: "
              f"{int(cond_curr.sum()):>6,} 人  ({cond_curr.mean():.2%})")

    elif y_freq_mode == "bout_p80_and_accr_p80":
        required = ["ba_out_bal_diff", "ac_accr_bal_diff"]
        missing = [c for c in required if c not in work.columns]
        if missing:
            raise ValueError(f"模式 '{y_freq_mode}' 缺少字段: {missing}")
        thr_bout  = float(threshold_source["ba_out_bal_diff"].quantile(0.80))
        thr_accr  = float(threshold_source["ac_accr_bal_diff"].quantile(0.80))
        cond_bout = work["ba_out_bal_diff"] >= thr_bout
        cond_accr = work["ac_accr_bal_diff"] >= thr_accr
        work["y_freq"] = (cond_bout & cond_accr).astype(int)
        print(f"  条件①  ba_out_bal_diff >= {thr_bout:.2f}（P80）: "
              f"{int(cond_bout.sum()):>6,} 人  ({cond_bout.mean():.2%})")
        print(f"  条件②  ac_accr_bal_diff >= {thr_accr:.2f}（P80）: "
              f"{int(cond_accr.sum()):>6,} 人  ({cond_accr.mean():.2%})")

    else:  # curr_p80_and_bout_p80
        required = ["ac_curr_bal_diff", "ba_out_bal_diff"]
        missing = [c for c in required if c not in work.columns]
        if missing:
            raise ValueError(f"模式 '{y_freq_mode}' 缺少字段: {missing}")
        thr_curr  = float(threshold_source["ac_curr_bal_diff"].quantile(0.80))
        thr_bout  = float(threshold_source["ba_out_bal_diff"].quantile(0.80))
        cond_curr = work["ac_curr_bal_diff"] >= thr_curr
        cond_bout = work["ba_out_bal_diff"]  >= thr_bout
        work["y_freq"] = (cond_curr & cond_bout).astype(int)
        print(f"  条件①  ac_curr_bal_diff >= {thr_curr:.2f}（P80）: "
              f"{int(cond_curr.sum()):>6,} 人  ({cond_curr.mean():.2%})")
        print(f"  条件②  ba_out_bal_diff  >= {thr_bout:.2f}（P80）: "
              f"{int(cond_bout.sum()):>6,} 人  ({cond_bout.mean():.2%})")

    n_freq = int(work["y_freq"].sum())
    print(f"  y_freq=1（两者交集）                                 : "
          f"{n_freq:>6,} 人  ({n_freq/n_total:.2%})")
    print(f"  {'─'*55}")

    thresholds.update({"thr_bout": thr_bout, "thr_curr": thr_curr, "thr_accr": thr_accr})
    return work, thresholds


# ─────────────────────────────────────────────
# Step 4：特征统计（供 notebook 展示）
# ─────────────────────────────────────────────

def get_feature_stats(
    df: pd.DataFrame,
    target: str = "y_freq",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回两个 DataFrame：
      1. feature_missing_df  —— 每个特征 X 的缺失率（排除 cst_id、y 列及泄露列）
      2. label_stats_df      —— 因变量 y_freq 的缺失率与正样本率
    """
    exclude_cols = {
        "cst_id", "y_freq", "y_dq_risk",
        "rt_acct_stat_2_end",
        "ba_out_bal_diff", "ac_curr_bal_diff", "ac_accr_bal_diff",
        "num_of_time_dq_diff", "dq_principal_diff",
        "ac_curr_bal_start", "ac_curr_bal_end", "ba_out_bal_start", "ba_out_bal_end",
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    miss_rates = df[feature_cols].isnull().mean()
    feature_missing_df = (
        pd.DataFrame({"feature": miss_rates.index, "missing_rate": miss_rates.values})
        .sort_values("missing_rate", ascending=False)
        .reset_index(drop=True)
    )
    feature_missing_df["missing_rate_pct"] = (
        feature_missing_df["missing_rate"] * 100
    ).round(2).astype(str) + "%"

    # 因变量统计（支持 y_freq / y_dq_risk）
    label_rows = []
    for col in [target]:
        if col not in df.columns:
            label_rows.append({
                "label": col, "missing_rate": "N/A",
                "positive_rate": "N/A", "n_positive": "N/A", "n_total": len(df)
            })
            continue
        miss = df[col].isnull().mean()
        pos_rate = df[col].mean() if df[col].notnull().any() else None
        n_pos = int(df[col].sum()) if df[col].notnull().any() else None
        label_rows.append({
            "label": col,
            "missing_rate": f"{miss*100:.2f}%",
            "positive_rate": f"{pos_rate*100:.4f}%" if pos_rate is not None else "N/A",
            "n_positive": n_pos,
            "n_total": len(df),
        })
    label_stats_df = pd.DataFrame(label_rows)

    return feature_missing_df, label_stats_df


# ─────────────────────────────────────────────
# Step 5：特征工程（删列/填充、LabelEncoder）
# ─────────────────────────────────────────────

def feature_engineering_potential(
    work: pd.DataFrame,
    target: str = "y_freq",
    add_quota_sq: bool = True,
    add_quota_cube: bool = True,
    add_quota_log: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, LabelEncoder]]:
    """
    特征工程：
      1. 提取 y，去掉 cst_id / 标签列 / 兜底泄露列
      2. 删除任何残留 datetime 列
      3. 授信额度多项式特征（可选）
      4. 档位有序编码（F3=1 最低 → A=8 最高）
      5. 数值型缺失处理：
           缺失率 <40%  → 中位数填充
           缺失率 >=40% → 删列，并输出被删列名
      6. 类别型缺失填充：众数，无众数则 "未知"
      7. LabelEncoder（直接 fit_transform）
      8. 只保留数值型列，返回纯数值特征矩阵 X
    """
    if target not in ("y_freq", "y_dq_risk"):
        raise ValueError("target 仅支持 'y_freq' 或 'y_dq_risk'")

    y = work[target].astype(int)

    # ── 去掉所有不进入建模的列（兜底） ──
    # split_eff_date 是开户日期副本（供概率校准时间切分），绝不能进入模型：
    # 从 CSV 读回时它会变成字符串（非 datetime），无法被下方 datetime 过滤捕获，
    # 因此必须在此显式删除，防止泄露。
    drop_cols = (
        {"cst_id", "cst_id0"} |
        {"y_freq", "y_dq_risk", "rt_acct_stat_2_end"} |
        {"split_eff_date"}
    )
    X = work.drop(columns=[c for c in drop_cols if c in work.columns]).copy()

    # ── 删除剩余 datetime 列 ──
    dt_cols = list(X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)
    if dt_cols:
        X = X.drop(columns=dt_cols)
        print(f"  已删除剩余日期列: {dt_cols}")

    # ── 授信额度多项式特征 ──
    if "credamt" in X.columns:
        cred = X["credamt"].astype(float)
        cred_norm = cred / 1_000_000
        if add_quota_sq:
            X["credamt_sq"] = cred_norm ** 2
        if add_quota_cube:
            X["credamt_cube"] = cred_norm ** 3
        if add_quota_log:
            X["credamt_log"] = np.log(np.maximum(cred.fillna(0), 1.0))

    # ── 档位有序编码（F3=1 最低 → A=8 最高） ──
    if "档位" in X.columns:
        tier_order = {
            "F3": 1, "F3级": 1, "f3": 1,
            "F2": 2, "F2级": 2, "f2": 2,
            "F1": 3, "F1级": 3, "f1": 3,
            "E":  4, "E级":  4, "e":  4,
            "D":  5, "D级":  5, "d":  5,
            "C":  6, "C级":  6, "c":  6,
            "B":  7, "B级":  7, "b":  7,
            "A":  8, "A级":  8, "a":  8,
        }
        raw_vals = X["档位"].astype(str).str.strip()
        mapped = raw_vals.map(tier_order)
        n_unmapped = mapped.isna().sum()
        if n_unmapped > 0:
            unmapped_vals = raw_vals[mapped.isna()].unique().tolist()
            print(f"  ⚠️  档位字段有 {n_unmapped} 个值未能映射: {unmapped_vals}，填充为 0")
        X["档位"] = mapped.fillna(0).astype(int)

    # ── 数值型缺失处理：>=40% 删列，<40% 中位数填充 ──
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    cols_to_drop_high_miss = []
    n_filled_num = 0
    for col in numeric_cols:
        if X[col].isnull().sum() > 0:
            miss_ratio = X[col].isnull().mean()
            if miss_ratio >= 0.4:
                cols_to_drop_high_miss.append(col)
            else:
                X[col] = X[col].fillna(X[col].median())
                n_filled_num += 1

    if cols_to_drop_high_miss:
        X = X.drop(columns=cols_to_drop_high_miss)
        print(f"\n  ⚠️  因缺失率 >=40% 被删除的列（共 {len(cols_to_drop_high_miss)} 个）：")
        for c in cols_to_drop_high_miss:
            print(f"       {c}")

    if n_filled_num > 0:
        print(f"  数值型缺失填充：{n_filled_num} 个字段（缺失率<40%，用中位数填充）")

    # ── 类别型缺失填充 ──
    categorical_cols = [c for c in X.select_dtypes(include=["object"]).columns if c != "档位"]
    n_filled_cat = 0
    for col in categorical_cols:
        if X[col].isnull().sum() > 0:
            mode_val = X[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "未知"
            X[col] = X[col].fillna(fill_val)
            n_filled_cat += 1
    if n_filled_cat > 0:
        print(f"  类别型缺失填充：{n_filled_cat} 个字段（众数，无众数填'未知'）")

    # ── LabelEncoder ──
    cat_features = [c for c in X.select_dtypes(include=["object"]).columns if c != "档位"]
    label_encoders: Dict[str, LabelEncoder] = {}
    for col in cat_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    if cat_features:
        print(f"  LabelEncoder：{len(cat_features)} 个类别特征已编码")

    # ── 只保留数值型 ──
    X = X.select_dtypes(include=[np.number]).copy()
    feature_names = X.columns.tolist()

    print(f"  特征工程完成：{len(feature_names)} 个特征，{len(y)} 个样本")
    return X, y, feature_names, label_encoders


# ─────────────────────────────────────────────
# 端到端入口函数
# ─────────────────────────────────────────────

def load_kechuang_potential(
    file_path: str,
    snapshot_date: str = "2026-01-31",
    target: str = "y_freq",
    csv_encoding: str = "utf-8-sig",
    add_quota_sq: bool = True,
    add_quota_cube: bool = True,
    add_quota_log: bool = False,
    apply_maturity_filter: bool = True,
    maturity_cutoff: str = "2026-05-31",
    y_freq_mode: str = "bout_gt0_and_curr_p80",
    apply_eff_date_filter: bool = True,
    eff_date_lower: str = "2024-10-01",
    eff_date_upper: str = "2026-03-31",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, List[str], Dict[str, float]]:
    """
    端到端入口：读文件 → 科创字段重命名 → [可选]到期日筛选 → 聚合 → 清洗 → 构造标签 → 统计 → 特征工程

    参数
    ----
    file_path              : Excel（.xlsx/.xls）或 CSV（.csv）文件路径
    snapshot_date          : 快照日期（用于 days_* 衍生特征），默认 2026-01-31
    target                 : 建模目标，可选：
                               "y_freq"    → 频率活跃度标签（默认）
                               "y_dq_risk" → 违约风险标签
    csv_encoding           : CSV 编码，默认 utf-8-sig；中文乱码时改为 "gbk"
    add_quota_sq           : 是否添加授信额度二次方项
    add_quota_cube         : 是否添加授信额度三次方项
    add_quota_log          : 是否添加 log(授信额度) 项
    apply_maturity_filter  : 是否执行到期日筛选（True=筛选，False=跳过）
    maturity_cutoff        : 到期日阈值（含），格式 "YYYY-MM-DD"，默认 "2026-05-31"
    y_freq_mode            : 仅 target='y_freq' 时有效，可选：
                               "bout_gt0_and_curr_p80"  → ba_out_bal_diff>0 且 ac_curr_bal_diff>=P80（默认）
                               "bout_p80_and_accr_p80"  → ba_out_bal_diff>=P80 且 ac_accr_bal_diff>=P80
                               "curr_p80_only"          → ac_curr_bal_diff>=P80（单条件）
                               "curr_p80_and_bout_p80"  → ac_curr_bal_diff>=P80 且 ba_out_bal_diff>=P80
    apply_eff_date_filter  : 是否执行生效日筛选（True=筛选，False=跳过），默认 True
    eff_date_lower         : 生效日下限（含），默认 "2024-10-01"（对应消费偏好表最早快照）
    eff_date_upper         : 生效日上限（含），默认 "2026-03-31"（对应消费偏好表最晚快照）

    返回
    ----
    df_clean          : 清洗 + 标签构造后的完整 DataFrame（含 cst_id、y_freq 列）
    feature_missing_df: 特征缺失率 DataFrame
    label_stats_df    : 因变量统计 DataFrame
    X                 : 特征矩阵（pd.DataFrame，缺失已处理）
    y                 : 标签向量（pd.Series，y_freq）
    feature_names     : 特征名列表
    thresholds        : y_freq 阈值字典 {y_freq_mode, thr_bout, thr_curr, thr_accr}
    """
    if target not in ("y_freq", "y_dq_risk"):
        raise ValueError("target 仅支持 'y_freq' 或 'y_dq_risk'")

    # ── y_dq_risk 模式下的违约追踪辅助函数 ──
    DQ_SET = {"3", "4", "7", "9"}

    def _clean_stat(v):
        try:
            return str(int(float(v)))
        except (ValueError, TypeError):
            return str(v).strip()

    def _count_dq_rows(df, label):
        """行级数据：统计 rt_acct_stat_2_end 为违约状态的行数和客户数，并展示各状态码明细"""
        if target != "y_dq_risk" or "rt_acct_stat_2_end" not in df.columns:
            return
        s = df["rt_acct_stat_2_end"].map(_clean_stat)
        dq_mask  = s.isin(DQ_SET)
        dq_rows  = int(dq_mask.sum())
        dq_cst   = int(df.loc[dq_mask, "cst_id"].nunique()) if "cst_id" in df.columns else -1
        null_cnt = int(df["rt_acct_stat_2_end"].isna().sum())
        total_rows = len(df)
        total_cst  = df["cst_id"].nunique() if "cst_id" in df.columns else -1
        print(f"  ┌─ [违约追踪] {label}")
        print(f"  │  总行数={total_rows:,}  总客户={total_cst:,}")
        print(f"  │  终点违约行(∈3/4/7/9)={dq_rows:,}  对应客户数={dq_cst:,}")
        # 各状态码明细（仅针对违约行）
        loan_col = next((c for c in ["loanacctno", "loan_acct_no"] if c in df.columns), None)
        for code in ["3", "4", "7", "9"]:
            mask_c = s == code
            n_rows_c = int(mask_c.sum())
            n_cst_c  = int(df.loc[mask_c, "cst_id"].nunique()) if "cst_id" in df.columns else -1
            n_loan_c = int(df.loc[mask_c, loan_col].nunique()) if loan_col else -1
            if n_rows_c > 0:
                pct_of_dq = n_rows_c / dq_rows * 100 if dq_rows > 0 else 0
                print(f"  │    状态{code}: 行数={n_rows_c:,}({pct_of_dq:.1f}%)  客户数={n_cst_c:,}  贷款账户数={n_loan_c:,}")
        print(f"  └─ 终点状态为空={null_cnt:,}")

    def _count_dq_agg(df, label):
        """聚合后数据（一客一行）：统计 rt_acct_stat_2_end 为违约状态的客户数"""
        if target != "y_dq_risk" or "rt_acct_stat_2_end" not in df.columns:
            return
        s = df["rt_acct_stat_2_end"].map(_clean_stat)
        dq_cst   = int(s.isin(DQ_SET).sum())
        null_cnt = int(df["rt_acct_stat_2_end"].isna().sum())
        total    = len(df)
        print(f"  ┌─ [违约追踪] {label}")
        print(f"  │  总客户={total:,}")
        print(f"  │  终点违约客户(∈3/4/7/9)={dq_cst:,}  ({dq_cst/total:.2%})")
        print(f"  └─ 终点状态为空={null_cnt:,}")
    # ─────────────────────────────────────────

    print(f"[1/7] 读取文件: {file_path}")
    df0 = read_data(file_path, csv_encoding=csv_encoding)

    print(f"[2/7] 科创字段英文别名恢复为中文...")
    df0 = rename_kechuang_cols(df0)

    print(f"[2.5/8] (cst_id, loanacctno) 去重...")
    df0 = dedup_by_cst_loan(df0)
    _count_dq_rows(df0, "去重后（行级）")

    print(f"[2.8/8] 类别型字段浮点→整数字符串转换（必须在起点违约剔除之前）...")
    df0 = cast_cat_cols(df0)
    _count_dq_rows(df0, "类型转换后（行级）")

    print(f"[3/8] 到期日筛选（apply={apply_maturity_filter}, cutoff={maturity_cutoff}）...")
    df0 = filter_by_maturity(df0, apply_filter=apply_maturity_filter,
                             maturity_cutoff=maturity_cutoff)
    _count_dq_rows(df0, "到期日筛选后（行级）")

    print(f"[3.5/8] 贷款生效日筛选（apply={apply_eff_date_filter}, "
          f"{eff_date_lower} ~ {eff_date_upper}）...")
    _count_dq_rows(df0, "生效日筛选前（行级）")   # ← 筛选前快照
    if apply_eff_date_filter:
        df0 = filter_by_eff_date(df0,
                                  eff_date_lower=eff_date_lower,
                                  eff_date_upper=eff_date_upper)
    else:
        print("  apply_eff_date_filter=False，跳过生效日筛选")
    _count_dq_rows(df0, "生效日筛选后（行级）")

    if target == "y_dq_risk":
        print("[3.7/8] 剔除起点已违约客户（仅 y_dq_risk 模式）...")
        df0 = filter_dq_start_customers(df0)
        _count_dq_rows(df0, "起点违约剔除后（行级）")

    print("[4/8] 一客多贷聚合...")
    df1 = aggregate_by_customer_potential(df0)
    _count_dq_agg(df1, "聚合后（客户级，一客一行）")

    print("[5/8] 数据清洗 Part1（因变量构造前：删ID/日期/无意义字段）...")
    df2 = clean_data_potential(df1, snapshot_date=snapshot_date, target=target)
    _count_dq_agg(df2, "清洗Part1后（客户级）")

    label_desc = f"y_freq（模式={y_freq_mode}）" if target == "y_freq" else "y_dq_risk"
    print(f"[6/8] 因变量构造（{label_desc}）...")
    df3, thresholds = build_labels_potential(df2, target=target, y_freq_mode=y_freq_mode)
    if target == "y_dq_risk" and "y_dq_risk" in df3.columns:
        n_pos = int(df3["y_dq_risk"].sum())
        total  = len(df3)
        print(f"  ┌─ [违约追踪] 标签构造后")
        print(f"  │  总客户={total:,}  y_dq_risk=1={n_pos:,}  ({n_pos/total:.2%})")
        print(f"  └─ y_dq_risk=0={total - n_pos:,}")

    print("[6.5/8] 数据清洗 Part2（因变量构造后：删泄露/diff/起点无意义字段）...")
    df_clean = drop_post_label_cols(df3, target=target)
    if target == "y_dq_risk" and "y_dq_risk" in df_clean.columns:
        n_pos = int(df_clean["y_dq_risk"].sum())
        total  = len(df_clean)
        print(f"  ┌─ [违约追踪] 清洗Part2后（最终建模样本）")
        print(f"  │  总客户={total:,}  y_dq_risk=1={n_pos:,}  ({n_pos/total:.2%})")
        print(f"  └─ y_dq_risk=0={total - n_pos:,}")

    print("[7/8] 特征统计 + 特征工程...")
    feature_missing_df, label_stats_df = get_feature_stats(df_clean, target=target)

    X, y, feature_names, _ = feature_engineering_potential(
        df_clean, target=target,
        add_quota_sq=add_quota_sq,
        add_quota_cube=add_quota_cube,
        add_quota_log=add_quota_log,
    )

    # ── 最终汇报 ──
    print(f"\n{'━'*55}")
    print(f"✅  预处理完成")
    print(f"   客户数          : {len(df_clean):,}")
    print(f"   特征数          : {len(feature_names):,}")
    print(f"   建模目标        : {target}")
    if target == "y_freq":
        print(f"   y_freq 模式     : {y_freq_mode}")
    print(f"   正样本率        : {y.mean():.4%}  ({y.sum()} / {len(y)})")
    if target == "y_dq_risk":
        print(f"   最终 y_dq_risk=1: {int(y.sum()):,} 人")
    print(f"{'━'*55}")

    return df_clean, feature_missing_df, label_stats_df, X, y, feature_names, thresholds
