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
  ac_curr_bal、ac_accr_bal、ba_out_bal、rt_loan_rate、tot_num_payment_1、
  zm_memo2_amt2_4（贷款起点/表现辅助字段均不进入最终模型）
  rt_acct_stat_2_end（y_dq_risk 构造后删除；y_freq 模式在 Part1 删除）

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

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def _mode_or_nan(x: pd.Series):
    m = x.mode()
    return m.iloc[0] if len(m) > 0 else np.nan


def _sum_min_count_one(x: pd.Series):
    """求和但保留“整组均缺失”为 NaN，避免把未知结果伪装成 0。"""
    return pd.to_numeric(x, errors="coerce").sum(min_count=1)


def _canonicalize_category_code(value):
    """规范化数值类别代码，同时保留合法的文本类别。"""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return np.nan
    try:
        numeric = float(text)
    except (ValueError, TypeError):
        return text
    if not np.isfinite(numeric):
        return np.nan
    return str(int(numeric)) if numeric.is_integer() else text


def get_consumption_field_cn_map() -> Dict[str, str]:
    """消费行为字段中英文映射；键统一为预处理后的英文小写字段名。"""
    return {
        "prvtcstrt12mocns1pamt": "对私客户近十二个月消费金额",
        "prvtcstrt12mo_csmdnum": "对私客户近十二个月消费笔数",
        "prvtcstrt12m1cns1pamt": "对私客户近十二个月境外消费金额",
        "prvtcstrt12moacsmdnum": "对私客户近十二个月境外消费笔数",
        "prvtcstr12mghccnmpamt": "对私客户近十二个月宾馆类消费金额",
        "prvtcstr12mghccsmdnum": "对私客户近十二个月宾馆类消费笔数",
        "prvtcstr12mcccns1pamt": "对私客户近十二个月餐饮类消费金额",
        "prvtcstrt12mcccs1dnum": "对私客户近十二个月餐饮类消费笔数",
        "prvtcstr12mjtccnmpamt": "对私客户近十二个月珠宝工艺类消费金额",
        "prvtcstr12mjtccsmdnum": "对私客户近十二个月珠宝工艺类消费笔数",
        "prvtcstr12meccnsmpamt": "对私客户近十二个月娱乐类消费金额",
        "prvtcstrt12meccsmdnum": "对私客户近十二个月娱乐类消费笔数",
        "prvtcstr12mreccnmpamt": "对私客户近十二个月房地产类消费金额",
        "prvtcstr12mreccsmdnum": "对私客户近十二个月房地产类消费笔数",
        "prvtcstr12mcsccn1pamt": "对私客户近十二个月汽车销售类消费金额",
        "prvtcstr12mcsccs1dnum": "对私客户近十二个月汽车销售类消费笔数",
        "prvtcstr12mwccnsmpamt": "对私客户近十二个月批发类消费金额",
        # 源表中同时存在以 0 结尾的字段版本，保留独立映射以保证 SHAP 中文展示。
        "prvtcstr12mwccnsmpamt0": "对私客户近十二个月批发类消费金额",
        "prvtcstrt12mwccsmdnum": "对私客户近十二个月批发类消费笔数",
        "prvtcstrt12mwccsmdnum0": "对私客户近十二个月批发类消费笔数",
        "prvtcstr12matccnmpamt": "对私客户近十二个月航空售票类消费金额",
        "prvtcstr12matccsmdnum": "对私客户近十二个月航空售票类消费笔数",
        "prvtcstr12mgccns1pamt": "对私客户近十二个月加油类消费金额",
        "prvtcstrt12mgccsmdnum": "对私客户近十二个月加油类消费笔数",
        "prvtcstr12msccns1pamt": "对私客户近十二个月超市类消费金额",
        "prvtcstr12msccnsmpamt": "对私客户近十二个月超市类消费金额",
        "prvtcstrt12msccsmdnum": "对私客户近十二个月超市类消费笔数",
        "prvtcstr12mrccns1pamt": "对私客户近十二个月铁路客运类消费金额",
        "prvtcstrt12mrccs1dnum": "对私客户近十二个月铁路客运类消费笔数",
        "prvtcstr12mlamccnpamt": "对私客户近十二个月大型家电专卖类消费金额",
        "prvtcstr12mlamccmdnum": "对私客户近十二个月大型家电专卖类消费笔数",
        "prvtcstr12mphccnmpamt": "对私客户近十二个月公立医院类消费金额",
        "prvtcstr12mphccsmdnum": "对私客户近十二个月公立医院类消费笔数",
        "prvtcstr12mpsccnmpamt": "对私客户近十二个月公共学校类消费金额",
        "prvtcstr12mpsccsmdnum": "对私客户近十二个月公共学校类消费笔数",
        "prvtcstr12mcwssccpamt": "对私客户近十二个月慈善与社会服务类消费金额",
        "prvtcstr12mcwssccdnum": "对私客户近十二个月慈善与社会服务类消费笔数",
        "prvtcstr12mgsccnmpamt": "对私客户近十二个月政府服务类消费金额",
        "prvtcstr12mgsccsmdnum": "对私客户近十二个月政府服务类消费笔数",
        "prvtcstr12mgmccnmpamt": "对私客户近十二个月百货类消费金额",
        "prvtcstr12mgmccsmdnum": "对私客户近十二个月百货类消费笔数",
        "prvtcstr12mcccnsmpamt": "对私客户近十二个月一般类消费金额",
        "prvtcstrt12mcccsmdnum": "对私客户近十二个月一般类消费笔数",
        "prvtcstr12mdccnsmpamt": "对私客户近十二个月服饰类消费金额",
        "prvtcstrt12mdccsmdnum": "对私客户近十二个月服饰类消费笔数",
        "prvtcstr12mgccnsmpamt": "对私客户近十二个月高尔夫类消费金额",
        "prvtcstrt12mgccs1dnum": "对私客户近十二个月高尔夫类消费笔数",
        "prvtcstr12miccnsmpamt": "对私客户近十二个月保险类消费金额",
        "prvtcstrt12miccsmdnum": "对私客户近十二个月保险类消费笔数",
        "prvtcstr12mlepccnpamt": "对私客户近十二个月大型企业采购类消费金额",
        "prvtcstr12mlepccmdnum": "对私客户近十二个月大型企业采购类消费笔数",
        "prvtcstr12mhcbccnpamt": "对私客户近十二个月保健美容类消费金额",
        "prvtcstr12mhcbccmdnum": "对私客户近十二个月保健美容类消费笔数",
        "prvtcstr12mtsccnmpamt": "对私客户近十二个月电信服务类消费金额",
        "prvtcstr12mtsccsmdnum": "对私客户近十二个月电信服务类消费笔数",
        "prvtcstr12mwvtccnpamt": "对私客户近十二个月车船运输类消费金额",
        "prvtcstr12mwvtccmdnum": "对私客户近十二个月车船运输类消费笔数",
        "prvtcstr12mtccns1pamt": "对私客户近十二个月旅行类消费金额",
        "prvtcstr12mtccnsmpamt": "对私客户近十二个月旅行类消费金额",
        "prvtcstrt12mtccsmdnum": "对私客户近十二个月旅行类消费笔数",
        "prvtcstr12mtpccnmpamt": "对私客户近十二个月纳税类消费金额",
        "prvtcstr12mtpccsmdnum": "对私客户近十二个月纳税类消费笔数",
        "prvtcstr12mfsccnmpamt": "对私客户近十二个月金融服务类消费金额",
        "prvtcstr12mfsccsmdnum": "对私客户近十二个月金融服务类消费笔数",
        "prvtcstr12masccnmpamt": "对私客户近十二个月机场服务类消费金额",
        "prvtcstr12masccsmdnum": "对私客户近十二个月机场服务类消费笔数",
        "prvtcstr12mpatccnpamt": "对私客户近十二个月典当拍卖信托类消费金额",
        "prvtcstr12mpatccmdnum": "对私客户近十二个月典当拍卖信托类消费笔数",
        "prvtcstr12mrccnsmpamt": "对私客户近十二个月零售类消费金额",
        "prvtcstrt12mrccsmdnum": "对私客户近十二个月零售类消费笔数",
        "prvtcstr12mcsccnmpamt": "对私客户近十二个月汽车服务类消费金额",
        "prvtcstr12mcsccsmdnum": "对私客户近十二个月汽车服务类消费笔数",
        "prvtcstr12mfccnsmpamt": "对私客户近十二个月食品类消费金额",
        "prvtcstrt12mfccsmdnum": "对私客户近十二个月食品类消费笔数",
        "prvtcstr12mvsccnmpamt": "对私客户近十二个月兽医服务类消费金额",
        "prvtcstr12mvsccsmdnum": "对私客户近十二个月兽医服务类消费笔数",
        "prvtcstr12mlpfccnpamt": "对私客户近十二个月水电煤缴费类消费金额",
        "prvtcstr12mlpfccmdnum": "对私客户近十二个月水电煤缴费类消费笔数",
        "prvtcstr12mpmccnmpamt": "对私客户近十二个月物业管理类消费金额",
        "prvtcstr12mpmccsmdnum": "对私客户近十二个月物业管理类消费笔数",
        "prvtcstr12mcpccnmpamt": "对私客户近十二个月县乡优惠类消费金额",
        "prvtcstr12mcpccsmdnum": "对私客户近十二个月县乡优惠类消费笔数",
        "prvtcstr12mcruccnpamt": "对私客户近十二个月租车用车类消费金额",
        "prvtcstr12mcruccmdnum": "对私客户近十二个月租车用车类消费笔数",
        "prvtcstr12mosccnmpamt": "对私客户近十二个月其他服务类消费金额",
        "prvtcstr12mosccsmdnum": "对私客户近十二个月其他服务类消费笔数",
        "prvtcstrt12mocnsmpamt": "对私客户近十二个月其他消费金额",
        "prvtcstrt12morcsmdnum": "对私客户近十二个月其他消费笔数",
        "prvtcstrt12mvcnsmpamt": "对私客户近十二个月有效消费金额",
        "prvtcstrt12mvdcsmdnum": "对私客户近十二个月有效消费笔数",
    }


def _worst_status(x: pd.Series):
    """
    一客多贷账户状态聚合：取最差状态码。
    优先级：9 > 7 > 4 > 3 > 其他
    只要有任意一笔是不良状态（3/4/7/9），客户就被标记为不良。

    注意：字段从 Excel/CSV 读入后可能是浮点型（如 3.0），
    此处先转 float 再转 int 再转 str，确保 "3.0" → "3"。
    """
    def _to_str(v):
        if pd.isna(v):
            return np.nan
        try:
            return str(int(float(v)))
        except (ValueError, TypeError):
            return str(v).strip()

    vals = x.map(_to_str).dropna()
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
# Step 0.05：按 (cst_id, loanacctno) 检查重复（可选去重）
# ─────────────────────────────────────────────

def report_cst_loan_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    统计 (cst_id, loanacctno) 重复组合及字段差异，但不删除任何行。
    """
    df = df.copy()

    loan_col = next(
        (c for c in ["loanacctno", "loan_acct_no"] if c in df.columns), None
    )

    if loan_col is None or "cst_id" not in df.columns:
        print("  [警告] 未找到 cst_id 或 loanacctno 字段，跳过重复检查")
        return df

    key_cols = ["cst_id", loan_col]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    dup_detail = df.loc[dup_mask].sort_values(key_cols)
    dup_summary = (
        dup_detail.groupby(key_cols, dropna=False).size()
        .reset_index(name="重复行数").sort_values("重复行数", ascending=False)
    )
    print(f"\n  -- (cst_id, {loan_col}) 重复检查（不去重）--")
    print("重复组合数：", len(dup_summary))
    print("重复涉及行数：", int(dup_mask.sum()))
    if dup_summary.empty:
        print("重复行数分布：无重复组合")
        print("差异字段数分布：无重复组合")
        return df
    print("重复行数分布：")
    print(dup_summary["重复行数"].value_counts().sort_index())

    value_cols = [c for c in df.columns if c not in key_cols]
    diff_field_count = (
        dup_detail.groupby(key_cols, dropna=False)[value_cols]
        .nunique(dropna=False).ne(1).sum(axis=1)
        .rename("存在差异字段数").reset_index()
    )
    diff_field_summary = (
        diff_field_count.groupby("存在差异字段数", dropna=False).size()
        .reset_index(name="cst_id+loanacctno组合数")
        .sort_values("存在差异字段数")
    )
    print("差异字段数分布（存在该差异字段数的 cst_id+loanacctno 组合数）：")
    print(diff_field_summary.to_string(index=False))
    return df


def deduplicate_exact_cst_loan(df: pd.DataFrame) -> pd.DataFrame:
    """仅删除字段完全一致的重复账户；冲突重复必须先制定业务合并规则。"""
    loan_col = next(
        (c for c in ["loanacctno", "loan_acct_no", "loanacct_no"] if c in df.columns),
        None,
    )
    if "cst_id" not in df.columns or loan_col is None:
        raise KeyError("账户去重需要 cst_id 和 loanacctno。")
    for key in ("cst_id", loan_col):
        normalized = df[key].astype("string").str.strip()
        invalid = normalized.isna() | normalized.eq("")
        if invalid.any():
            raise ValueError(
                f"账户去重键 {key} 存在 {int(invalid.sum()):,} 个空值，"
                "不能安全执行去重。"
            )

    key_cols = ["cst_id", loan_col]
    duplicate_rows = df[df.duplicated(key_cols, keep=False)]
    if not duplicate_rows.empty:
        value_cols = [c for c in df.columns if c not in key_cols]
        conflicting = (
            duplicate_rows.groupby(key_cols, dropna=False)[value_cols]
            .nunique(dropna=False).gt(1).any(axis=1)
        )
        if conflicting.any():
            raise ValueError(
                f"发现 {int(conflicting.sum()):,} 组同一客户-贷款账号存在字段冲突；"
                "不能用 keep='first' 任意去重，请先制定业务合并规则。"
            )
    return df.drop_duplicates(subset=key_cols, keep="first").copy()


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
        "occup_cd",
        "gnd_cd",
        "rt_acct_stat_2_end",
        "rt_acct_stat_2",
    ]

    converted = []
    for col in cat_int_cols:
        if col in df.columns:
            df[col] = df[col].map(_canonicalize_category_code)
            converted.append(col)

    if converted:
        print(f"  已将类别型字段转为整数字符串（共 {len(converted)} 个）: {converted}")
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
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
        # SQL输出名统一到本模块内部使用的标准名
        "acct_eff_dt":         "rt_acct_eff_date_1",
        "matur_dt":            "rt_curr_matur_date_1",
        "rt_acct_stat_2_start": "rt_acct_stat_2",
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
        print("  [警告] 未找到英文科创字段，跳过重命名（可能已是中文列名）")
    return df


def get_field_date_coverage(
    df: pd.DataFrame,
    eff_date_col: str = "rt_acct_eff_date_1",
    stable_months: int = 3,
    stable_threshold: float = 0.90,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按贷款生效月份统计每个字段的非空覆盖率及稳定覆盖起始月。"""
    work = df.copy()
    if eff_date_col not in work.columns:
        raise KeyError(f"字段覆盖统计缺少生效日字段 {eff_date_col}")

    loan_col = next(
        (c for c in ["loanacctno", "loan_acct_no", "loanacct_no"] if c in work.columns),
        None,
    )
    if loan_col is None:
        work["__coverage_loan_id__"] = np.arange(len(work)).astype(str)
        loan_col = "__coverage_loan_id__"

    raw_eff = work[eff_date_col]
    eff = pd.to_datetime(raw_eff, format="%d%b%Y", errors="coerce")
    failed = eff.isna()
    if failed.any():
        eff.loc[failed] = pd.to_datetime(raw_eff.loc[failed], errors="coerce")
    work["__eff_date__"] = eff
    work = work[work["__eff_date__"].notna()].copy()
    work["生效月份"] = work["__eff_date__"].dt.to_period("M").astype(str)

    field_cols = [
        c for c in df.columns
        if c not in {"__coverage_loan_id__", "__eff_date__", "生效月份"}
    ]
    total_loans = int(work[loan_col].nunique())
    month_totals = work.groupby("生效月份")[loan_col].nunique().sort_index()
    all_months = month_totals.index.tolist()

    summary_rows = []
    monthly_rows = []
    for field in field_cols:
        valid = work[field].notna()
        if pd.api.types.is_object_dtype(work[field]) or pd.api.types.is_string_dtype(work[field]):
            valid &= work[field].astype("string").str.strip().ne("")

        nonempty_loans = int(work.loc[valid, loan_col].nunique())
        first_date = work.loc[valid, "__eff_date__"].min() if valid.any() else pd.NaT
        last_date = work.loc[valid, "__eff_date__"].max() if valid.any() else pd.NaT

        month_nonempty = (
            work.loc[valid].groupby("生效月份")[loan_col].nunique()
            .reindex(all_months, fill_value=0)
        )
        coverage = month_nonempty.div(month_totals).fillna(0.0)

        stable_start = pd.NaT
        if len(all_months) >= stable_months:
            periods = pd.PeriodIndex(all_months, freq="M")
            for i in range(len(all_months) - stable_months + 1):
                window_periods = periods[i:i + stable_months]
                is_consecutive = all(
                    window_periods[j] == window_periods[0] + j
                    for j in range(stable_months)
                )
                if is_consecutive and (coverage.iloc[i:i + stable_months] >= stable_threshold).all():
                    stable_start = window_periods[0].to_timestamp()
                    break

        summary_rows.append({
            "字段": field,
            "最早非空生效日": first_date,
            "最晚非空生效日": last_date,
            "非空贷款数": nonempty_loans,
            "总贷款数": total_loans,
            "总体覆盖率": nonempty_loans / total_loans if total_loans else np.nan,
            "稳定覆盖起始月": stable_start,
        })
        for month in all_months:
            monthly_rows.append({
                "字段": field,
                "生效月份": month,
                "贷款数": int(month_totals.loc[month]),
                "非空数": int(month_nonempty.loc[month]),
                "覆盖率": float(coverage.loc[month]),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(monthly_rows)


def get_feature_descriptive_stats(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """返回数值型字段统计，以及类别型字段众数和各类别比例。"""
    known_categorical = {
        "gnd_cd", "mar_sttn_cd", "education_cd", "occup_cd",
        "cst_star_cd", "档位",
    }
    cols = [c for c in feature_cols if c in df.columns]
    categorical_cols = [
        c for c in cols
        if c in known_categorical
        or not pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric_cols = [c for c in cols if c not in categorical_cols]

    numeric_rows = []
    for field in numeric_cols:
        values = pd.to_numeric(df[field], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        numeric_rows.append({
            "字段": field,
            "最小值": values.min(),
            "最大值": values.max(),
            "平均值": values.mean(),
        })

    category_rows = []
    for field in categorical_cols:
        values = df[field].astype("string").str.strip()
        values = values.mask(
            values.isna() | values.str.lower().isin({"", "nan", "none", "<na>"})
        ).fillna("<缺失>")
        counts = values.value_counts(dropna=False)
        mode_value = counts.index[0] if len(counts) else np.nan
        total = len(values)
        for category, count in counts.items():
            category_rows.append({
                "字段": field,
                "众数": mode_value,
                "类别": category,
                "数量": int(count),
                "类别比例": count / total if total else np.nan,
            })

    return pd.DataFrame(numeric_rows), pd.DataFrame(category_rows)


# ─────────────────────────────────────────────
# Step 0.5：到期日筛选（可选）
# ─────────────────────────────────────────────

def filter_by_maturity(
    df_raw: pd.DataFrame,
    apply_filter: bool = True,
    maturity_cutoff: str = "2026-07-21",
) -> pd.DataFrame:
    """
    按贷款到期日（rt_curr_matur_date_1）筛选观察日前已经到期的账户行。

    逻辑：
      - 保留 rt_curr_matur_date_1 <= maturity_cutoff 的账户行
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

    print(f"\n  -- 到期日筛选前 --")
    print(f"     账户行数  : {n_rows_before:,}")
    print(f"     客户数    : {_fmt(n_cust_before)}")
    print(f"     贷款账号数: {_fmt(n_loan_before)}")

    if not apply_filter:
        print(f"  apply_filter=False，跳过到期日筛选，原样返回")
        return df

    if "rt_curr_matur_date_1" not in df.columns:
        raise KeyError("已启用到期日筛选，但数据缺少 rt_curr_matur_date_1。")

    # 转换到期日字段：兼容 "31MAY2026"（SAS 格式）和标准日期格式
    raw_maturity = df["rt_curr_matur_date_1"].copy()
    parsed_maturity = pd.to_datetime(raw_maturity, format="%d%b%Y", errors="coerce")
    mask_failed = parsed_maturity.isna()
    if mask_failed.any():
        parsed_maturity.loc[mask_failed] = pd.to_datetime(
            raw_maturity.loc[mask_failed], errors="coerce"
        )
    invalid_dates = int(parsed_maturity.isna().sum())
    if invalid_dates:
        print(f"  [警告] 到期日缺失/无法解析 {invalid_dates:,} 行，这些行不会通过到期日筛选。")
    df["rt_curr_matur_date_1"] = parsed_maturity

    cutoff_dt = pd.Timestamp(maturity_cutoff)
    df = df[df["rt_curr_matur_date_1"] <= cutoff_dt].copy()
    if df.empty:
        raise ValueError("到期日筛选后没有剩余样本，请核对日期格式与 maturity_cutoff。")

    n_rows_after = len(df)
    n_cust_after = df["cst_id"].nunique() if "cst_id" in df.columns else None
    n_loan_after = df[_loan_col].nunique() if _loan_col else None

    def _diff(a, b):
        return f"{a - b:,}" if isinstance(a, int) and isinstance(b, int) else "N/A"

    print(f"\n  -- 到期日筛选后（rt_curr_matur_date_1 <= {maturity_cutoff}）--")
    print(f"     账户行数  : {n_rows_after:,}  （减少 {n_rows_before - n_rows_after:,} 行）")
    print(f"     客户数    : {_fmt(n_cust_after)}  （减少 {_diff(n_cust_before, n_cust_after)} 位）")
    print(f"     贷款账号数: {_fmt(n_loan_after)}  （减少 {_diff(n_loan_before, n_loan_after)} 个）")
    return df


# ─────────────────────────────────────────────
# Step 0.6：贷款生效日筛选（新增）
# ─────────────────────────────────────────────

def filter_by_eff_date(
    df: pd.DataFrame,
    eff_date_lower: str = "2025-01-01",
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
    eff_date_lower : 生效日下限（含），默认 "2025-01-01"
                     与两项建模任务的统一样本窗口保持一致
    eff_date_upper : 生效日上限（含），默认 "2026-03-31"
                     对应消费偏好表最晚快照 2026-02-28（X_SNAP_DT<=2026-02-28）
    """
    df = df.copy()

    if "rt_acct_eff_date_1" not in df.columns:
        raise KeyError("已启用生效日筛选，但数据缺少 rt_acct_eff_date_1。")

    n_rows_before = len(df)
    n_cust_before = df["cst_id"].nunique() if "cst_id" in df.columns else None
    _loan_col = next((c for c in ["loanacctno", "loan_acct_no"] if c in df.columns), None)
    n_loan_before = df[_loan_col].nunique() if _loan_col else None

    def _fmt(v):
        return f"{v:,}" if isinstance(v, (int, float)) else "N/A"

    print(f"\n  -- 贷款生效日筛选前 --")
    print(f"     账户行数  : {n_rows_before:,}")
    print(f"     客户数    : {_fmt(n_cust_before)}")
    print(f"     贷款账号数: {_fmt(n_loan_before)}")

    # 转换生效日（兼容 SAS 格式和标准格式）
    eff_col = df["rt_acct_eff_date_1"].copy()
    eff_parsed = pd.to_datetime(eff_col, format="%d%b%Y", errors="coerce")
    mask_failed = eff_parsed.isna()
    if mask_failed.any():
        eff_parsed[mask_failed] = pd.to_datetime(eff_col[mask_failed], errors="coerce")
    invalid_dates = int(eff_parsed.isna().sum())
    if invalid_dates:
        print(f"  [警告] 生效日缺失/无法解析 {invalid_dates:,} 行，这些行不会通过生效日筛选。")
    df["rt_acct_eff_date_1"] = eff_parsed

    lower_dt = pd.Timestamp(eff_date_lower)
    upper_dt = pd.Timestamp(eff_date_upper)
    if lower_dt > upper_dt:
        raise ValueError("eff_date_lower 不能晚于 eff_date_upper。")
    df = df[
        (df["rt_acct_eff_date_1"] >= lower_dt) &
        (df["rt_acct_eff_date_1"] <= upper_dt)
    ].copy()
    if df.empty:
        raise ValueError("生效日筛选后没有剩余样本，请核对日期格式和筛选区间。")

    n_rows_after = len(df)
    n_cust_after = df["cst_id"].nunique() if "cst_id" in df.columns else None
    n_loan_after = df[_loan_col].nunique() if _loan_col else None

    print(f"\n  -- 贷款生效日筛选后（{eff_date_lower} ~ {eff_date_upper}）--")
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
        raise KeyError("y_dq_risk 模式需要 rt_acct_stat_2 以剔除起点已违约客户。")
    if "cst_id" not in df.columns:
        raise KeyError("起点违约客户剔除缺少 cst_id。")

    DQ_STATUS = {"3", "4", "7", "9"}
    stat_str = df["rt_acct_stat_2"].astype(str).str.strip()
    bad_cst_ids = set(df.loc[stat_str.isin(DQ_STATUS), "cst_id"].unique())

    n_cust_before = df["cst_id"].nunique()
    n_rows_before = len(df)
    df = df[~df["cst_id"].isin(bad_cst_ids)].copy()
    n_cust_after = df["cst_id"].nunique()
    n_rows_after = len(df)

    print(f"\n  -- 起点违约客户剔除（y_dq_risk 模式）--")
    print(f"     剔除前客户数 : {n_cust_before:,}  行数: {n_rows_before:,}")
    print(f"     起点违约客户 : {len(bad_cst_ids):,} 位（rt_acct_stat_2 in {{3,4,7,9}}）")
    print(f"     剔除后客户数 : {n_cust_after:,}  行数: {n_rows_after:,}")
    return df


# ─────────────────────────────────────────────
# Step 1：一客多贷聚合
# ─────────────────────────────────────────────

def get_customer_static_conflicts(
    df: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """统计同一客户静态字段出现多个值的情况，供聚合前质量审计。"""
    result_columns = ["字段", "冲突客户数", "单客最大不同值数"]
    if "cst_id" not in df.columns:
        raise KeyError("客户静态字段冲突检查缺少 cst_id。")
    available = [col for col in columns if col in df.columns]
    multi_customer_rows = df[df.duplicated("cst_id", keep=False)]
    if not available or multi_customer_rows.empty:
        return pd.DataFrame(columns=result_columns)
    distinct_counts = (
        multi_customer_rows.groupby("cst_id", dropna=False)[available]
        .nunique(dropna=False)
    )
    rows = []
    for col in available:
        conflict = distinct_counts[col] > 1
        if conflict.any():
            rows.append({
                "字段": col,
                "冲突客户数": int(conflict.sum()),
                "单客最大不同值数": int(distinct_counts.loc[conflict, col].max()),
            })
    if not rows:
        return pd.DataFrame(columns=result_columns)
    return pd.DataFrame(rows).sort_values(
        ["冲突客户数", "字段"], ascending=[False, True]
    ).reset_index(drop=True)


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
    if "cst_id" not in df_raw.columns:
        raise KeyError("一客多贷聚合缺少 cst_id。")
    customer_ids = df_raw["cst_id"].astype("string").str.strip()
    invalid_id = customer_ids.isna() | customer_ids.eq("")
    if invalid_id.any():
        raise ValueError(
            f"发现 {int(invalid_id.sum()):,} 行 cst_id 为空；"
            "不能静默丢弃或把多个未知客户聚合到一起。"
        )
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
    cust_cols_mode  = ["gnd_cd", "mar_sttn_cd",
                       "education_cd", "occup_cd", "cst_star_cd"]
    cust_cols_first = ["age", "bank_cust_become_date"]

    # 科创人才字段（中文列名，rename 后已恢复）
    kechuang_first_cols = [
        "当前aum", "当前lum", "科技人才对应得分",
        "kum分", "lum分", "aum分", "总分", "档位",
    ]

    # 消费行为：PRVT* 字段均值
    pref_cols = [c for c in df.columns if str(c).lower().startswith("prvt")]

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

    static_audit_cols = (
        cust_cols_mode + cust_cols_first + kechuang_first_cols + mpb_first_cols
    )
    static_conflicts = get_customer_static_conflicts(df, static_audit_cols)
    if not static_conflicts.empty:
        print(
            "  [警告] 同一客户的静态字段存在多值；"
            "mode/first 聚合可能遮蔽上游冲突，请核对："
        )
        print(static_conflicts.to_string(index=False))

    agg_dict: Dict[str, object] = {}

    for c in bal_diff_sum_cols:
        if c in df.columns:
            agg_dict[c] = _sum_min_count_one

    for c, func in loan_special_agg.items():
        if c in df.columns:
            agg_dict[c] = func

    for c in credit_sum_cols:
        if c in df.columns:
            agg_dict[c] = _sum_min_count_one

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
    # 注意：loanacctno / rt_acct_eff_date_1 / rt_curr_matur_date_1 等 ID/日期字段
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
        "mar_sttn_cd", "cst_star_cd", "occup_cd", "gnd_cd",
        "rt_acct_stat_2_end", "rt_acct_stat_2",
    ]
    for col in cat_int_cols:
        if col in df.columns:
            df[col] = df[col].map(_canonicalize_category_code)
    print(f"  已将类别型字段转为整数字符串: "
          f"{[c for c in cat_int_cols if c in df.columns]}")

    # 年龄是真实连续变量，不能随代码字段一起转成 LabelEncoder 类别。
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # ── 日期列转 datetime ──
    date_cols = [
        "acctbegindate", "bank_cust_become_date",
        "rt_acct_eff_date_1", "rt_curr_matur_date_1", "end_snap_dt", "x_snap_dt",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── days_* 衍生特征 ──
    snap = pd.Timestamp(snapshot_date)
    if "bank_cust_become_date" in df.columns:
        df["days_since_become_cust"] = (snap - df["bank_cust_become_date"]).dt.days
        future_count = int((df["days_since_become_cust"] < 0).sum())
        if future_count:
            print(
                f"  [警告] bank_cust_become_date 晚于快照日的客户 "
                f"{future_count:,} 人，days_since_become_cust 已设为缺失。"
            )
            df.loc[df["days_since_become_cust"] < 0, "days_since_become_cust"] = np.nan
        print("  已生成 days_since_become_cust")

    # ── 删除所有不进入建模的字段 ──
    pre_drop_cols = [
        # ── 新取数结构新增的 ID / 时间戳字段 ──
        "loanacctno",           # 贷款账号（行级 ID）
        "acct_eff_dt",          # 兼容旧SQL：贷款开始日期
        "matur_dt",             # 兼容旧SQL：贷款到期日期
        "end_snap_dt",          # 贷款终点快照日（到期日前一天）
        "risk_snap_dt",         # 违约风险终点快照日，仅用于取数对齐，不进入清洗后数据或模型
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
        # 客户归属机构字段：按要求不进入模型
        "blng_insid",
        "cst_blng_insid",
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
        fit_mask = np.ones(len(work), dtype=bool)
    else:
        fit_mask = np.asarray(threshold_fit_mask, dtype=bool)
        if len(fit_mask) != len(work):
            raise ValueError("threshold_fit_mask length must equal len(df)")
        if not fit_mask.any():
            raise ValueError("threshold_fit_mask selects zero rows")

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
            if pd.isna(v):
                return np.nan
            try:
                return str(int(float(v)))
            except (ValueError, TypeError):
                return str(v).strip()
        stat_str = work["rt_acct_stat_2_end"].map(_to_clean_str_label)
        valid_status = stat_str.notna() & stat_str.str.fullmatch(r"\d+")
        n_missing_status = int((~valid_status).sum())
        if n_missing_status:
            print(
                f"  [警告] rt_acct_stat_2_end 缺失/无效 {n_missing_status:,} 人，"
                "标签无法定义，已从建模样本排除。"
            )
            work = work.loc[valid_status].copy()
            stat_str = stat_str.loc[valid_status]
        if work.empty:
            raise ValueError("rt_acct_stat_2_end 没有任何有效值，无法构造 y_dq_risk。")
        n_total = len(work)
        work["y_dq_risk"] = stat_str.isin(DQ_STATUS).astype(int)
        n_pos = int(work["y_dq_risk"].sum())
        print(f"\n  {'-'*55}")
        print(f"  y_dq_risk 构造统计")
        print(f"  {'-'*55}")
        print(f"  rt_acct_stat_2_end in {{3,4,7,9}} -> 1 : {n_pos:>6,} 人  ({n_pos/n_total:.2%})")
        print(f"  {'-'*55}")
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

    required_by_mode = {
        "bout_gt0_and_curr_p80": ["ba_out_bal_diff", "ac_curr_bal_diff"],
        "bout_p80_and_accr_p80": ["ba_out_bal_diff", "ac_accr_bal_diff"],
        "curr_p80_only": ["ac_curr_bal_diff"],
        "curr_p80_and_bout_p80": ["ac_curr_bal_diff", "ba_out_bal_diff"],
    }
    label_source_cols = required_by_mode[y_freq_mode]
    missing = [c for c in label_source_cols if c not in work.columns]
    if missing:
        raise ValueError(f"模式 '{y_freq_mode}' 缺少字段: {missing}")
    label_values = work[label_source_cols].apply(pd.to_numeric, errors="coerce")
    valid_label = np.isfinite(label_values.to_numpy(dtype=float)).all(axis=1)
    n_invalid_label = int((~valid_label).sum())
    if n_invalid_label:
        print(
            f"  [警告] 标签来源字段 {label_source_cols} 存在缺失的客户 "
            f"{n_invalid_label:,} 人，已从建模样本排除。"
        )
        work = work.loc[valid_label].copy()
        label_values = label_values.loc[valid_label]
        fit_mask = fit_mask[valid_label]
    if work.empty:
        raise ValueError("标签来源字段没有完整样本，无法构造 y_freq。")
    if not fit_mask.any():
        raise ValueError("排除标签缺失客户后，threshold_fit_mask 不再包含任何样本。")
    work.loc[:, label_source_cols] = label_values
    threshold_source = work.loc[fit_mask]
    threshold_scope = (
        "all valid rows" if threshold_fit_mask is None
        else "train only (%d valid rows)" % len(threshold_source)
    )
    n_total = len(work)

    thr_bout = float("nan")
    thr_curr = float("nan")
    thr_accr = float("nan")

    print(f"\n  {'-'*55}")
    print(f"  y_freq 构造统计（模式: {y_freq_mode}）")
    print(f"  P80 阈值拟合范围: {threshold_scope}")
    print(f"  {'-'*55}")

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
    print(f"  {'-'*55}")

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
    missing_masks = {}
    for col in feature_cols:
        values = df[col]
        if pd.api.types.is_numeric_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce")
            missing_masks[col] = numeric.isna() | ~np.isfinite(numeric)
        else:
            text_values = values.astype("string").str.strip()
            missing_masks[col] = (
                text_values.isna()
                | text_values.str.lower().isin({"", "nan", "none", "<na>"})
            )
    miss_rates = pd.DataFrame(missing_masks, index=df.index).mean()
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

class PotentialFeaturePreprocessor:
    """只在训练数据上拟合缺失处理与类别编码，再复用于其他数据集。"""

    _DROP_COLS = {
        "cst_id", "cst_id0", "blng_insid", "cst_blng_insid",
        "y_freq", "y_dq_risk", "rt_acct_stat_2_end", "split_eff_date",
    }
    _TIER_ORDER = {
        "F3": 1, "F3级": 1, "f3": 1,
        "F2": 2, "F2级": 2, "f2": 2,
        "F1": 3, "F1级": 3, "f1": 3,
        "E": 4, "E级": 4, "e": 4,
        "D": 5, "D级": 5, "d": 5,
        "C": 6, "C级": 6, "c": 6,
        "B": 7, "B级": 7, "b": 7,
        "A": 8, "A级": 8, "a": 8,
    }
    # 这些字段即使从 CSV/Excel 读成整数，也是无序代码而非连续数值。
    _KNOWN_CATEGORICAL = {
        "gnd_cd", "mar_sttn_cd", "education_cd", "occup_cd",
        "cst_star_cd", "busikind",
    }

    def __init__(
        self,
        target: str = "y_freq",
        add_quota_sq: bool = True,
        add_quota_cube: bool = True,
        add_quota_log: bool = False,
        missing_threshold: float = 0.4,
        categorical_features: Optional[Sequence[str]] = None,
    ):
        if target not in ("y_freq", "y_dq_risk"):
            raise ValueError("target 仅支持 'y_freq' 或 'y_dq_risk'")
        if not 0 < missing_threshold <= 1:
            raise ValueError("missing_threshold 必须在 (0, 1] 内。")
        self.target = target
        self.add_quota_sq = add_quota_sq
        self.add_quota_cube = add_quota_cube
        self.add_quota_log = add_quota_log
        self.missing_threshold = float(missing_threshold)
        self.explicit_categorical_features = set(categorical_features or [])
        invalid_categorical = self.explicit_categorical_features & {"age", "档位"}
        if invalid_categorical:
            raise ValueError(
                "age 是连续变量、档位是有序变量，不能作为无序类别特征: "
                f"{sorted(invalid_categorical)}"
            )

    def _prepare_features(self, work: pd.DataFrame) -> pd.DataFrame:
        X = work.drop(
            columns=[c for c in self._DROP_COLS if c in work.columns]
        ).copy()
        dt_cols = list(X.select_dtypes(include=["datetime", "datetimetz"]).columns)
        if dt_cols:
            X = X.drop(columns=dt_cols)

        if "credamt" in X.columns:
            cred = pd.to_numeric(X["credamt"], errors="coerce")
            cred_norm = cred / 1_000_000
            if self.add_quota_sq:
                X["credamt_sq"] = cred_norm ** 2
            if self.add_quota_cube:
                X["credamt_cube"] = cred_norm ** 3
            if self.add_quota_log:
                X["credamt_log"] = np.log(np.maximum(cred.fillna(0), 1.0))

        if "档位" in X.columns:
            raw_tier = X["档位"].astype("string").str.strip()
            X["档位"] = raw_tier.map(self._TIER_ORDER).fillna(0).astype(int)
        return X

    @staticmethod
    def _clean_category(values: pd.Series) -> pd.Series:
        cleaned = values.astype("string").str.strip()
        return cleaned.mask(
            cleaned.isna() | cleaned.str.lower().isin({"", "nan", "none", "<na>"})
        )

    def fit(self, work: pd.DataFrame):
        X = self._prepare_features(work)
        categorical_dtypes = ["object", "string", "category", "bool"]
        dtype_categorical = set(
            X.select_dtypes(include=categorical_dtypes).columns
        )
        categorical_cols = [
            col for col in X.columns
            if col in dtype_categorical
            or col in self._KNOWN_CATEGORICAL
            or col in self.explicit_categorical_features
        ]
        numeric_cols = [
            col for col in X.select_dtypes(include=[np.number]).columns
            if col not in categorical_cols
        ]
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        missing_rates = {
            col: float(X[col].isna().mean()) for col in numeric_cols
        }
        missing_rates.update({
            col: float(self._clean_category(X[col]).isna().mean())
            for col in categorical_cols
        })
        self.dropped_high_missing_ = [
            col for col in X.columns
            if col in missing_rates and missing_rates[col] >= self.missing_threshold
        ]
        numeric_cols = [c for c in numeric_cols if c not in self.dropped_high_missing_]
        categorical_cols = [
            c for c in categorical_cols if c not in self.dropped_high_missing_
        ]
        self.numeric_fill_values_ = {
            col: float(pd.to_numeric(X[col], errors="coerce").median())
            for col in numeric_cols
        }

        self.categorical_features_ = categorical_cols
        self.categorical_fill_values_ = {}
        self.label_encoders_: Dict[str, LabelEncoder] = {}
        for col in self.categorical_features_:
            cleaned = self._clean_category(X[col])
            non_missing = cleaned.dropna().astype(str)
            mode = non_missing.mode()
            fill_value = str(mode.iloc[0]) if len(mode) else "未知"
            values = cleaned.fillna(fill_value).astype(str)
            encoder = LabelEncoder().fit(values)
            self.categorical_fill_values_[col] = fill_value
            self.label_encoders_[col] = encoder

        valid_features = set(numeric_cols) | set(self.categorical_features_)
        self.numeric_features_ = numeric_cols
        self.feature_names_ = [c for c in X.columns if c in valid_features]
        if not self.feature_names_:
            raise ValueError("预处理后没有可用于建模的数值或类别特征。")
        self.is_fitted_ = True
        return self

    def transform(self, work: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("请先在训练数据上调用 fit。")
        X = self._prepare_features(work)
        transformed = pd.DataFrame(index=work.index)

        for col in self.numeric_features_:
            values = (
                pd.to_numeric(X[col], errors="coerce")
                if col in X.columns else pd.Series(np.nan, index=work.index)
            )
            values = values.replace([np.inf, -np.inf], np.nan)
            transformed[col] = values.fillna(self.numeric_fill_values_[col]).astype(float)

        for col in self.categorical_features_:
            if col in X.columns:
                values = self._clean_category(X[col])
            else:
                values = pd.Series(pd.NA, index=work.index, dtype="string")
            values = values.fillna(self.categorical_fill_values_[col]).astype(str)
            mapping = {
                value: code
                for code, value in enumerate(self.label_encoders_[col].classes_)
            }
            # 验证/测试中新类别不能反向改变训练词表；-1 交给 LightGBM 按缺失类别处理。
            transformed[col] = values.map(mapping).fillna(-1).astype(int)

        return transformed[self.feature_names_]

    def fit_transform(self, work: pd.DataFrame) -> pd.DataFrame:
        return self.fit(work).transform(work)


def feature_engineering_potential(
    work: pd.DataFrame,
    target: str = "y_freq",
    add_quota_sq: bool = True,
    add_quota_cube: bool = True,
    add_quota_log: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, LabelEncoder]]:
    """兼容旧接口；正式建模应在切分后单独 fit ``PotentialFeaturePreprocessor``。"""
    if target not in work.columns:
        raise KeyError(f"数据中不存在目标列: {target}")
    y = work[target].astype(int)
    preprocessor = PotentialFeaturePreprocessor(
        target=target,
        add_quota_sq=add_quota_sq,
        add_quota_cube=add_quota_cube,
        add_quota_log=add_quota_log,
    )
    X = preprocessor.fit_transform(work)
    if preprocessor.dropped_high_missing_:
        print(
            f"  [警告] 因缺失率 >=40% 被删除的列（训练数据口径，"
            f"共 {len(preprocessor.dropped_high_missing_)} 个）："
        )
        for col in preprocessor.dropped_high_missing_:
            print(f"       {col}")
    print(
        f"  特征工程完成：{len(preprocessor.feature_names_)} 个特征，{len(y)} 个样本；"
        f"类别特征 {len(preprocessor.categorical_features_)} 个"
    )
    return X, y, preprocessor.feature_names_, preprocessor.label_encoders_


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
    maturity_cutoff: str = "2026-07-21",
    y_freq_mode: str = "bout_gt0_and_curr_p80",
    apply_eff_date_filter: bool = False,
    eff_date_lower: str = "2025-01-01",
    eff_date_upper: str = "2026-03-31",
    dedup_cst_loan: bool = False,
    exclude_dq_start_customers: Optional[bool] = None,
    label_threshold_train_ratio: Optional[float] = None,
    require_dual_label_cohort: bool = False,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series,
    List[str], Dict[str, float], pd.DataFrame, pd.DataFrame,
]:
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
    maturity_cutoff        : 到期日阈值（含），格式 "YYYY-MM-DD"，默认 "2026-07-21"
    y_freq_mode            : 仅 target='y_freq' 时有效，可选：
                               "bout_gt0_and_curr_p80"  → ba_out_bal_diff>0 且 ac_curr_bal_diff>=P80（默认）
                               "bout_p80_and_accr_p80"  → ba_out_bal_diff>=P80 且 ac_accr_bal_diff>=P80
                               "curr_p80_only"          → ac_curr_bal_diff>=P80（单条件）
                               "curr_p80_and_bout_p80"  → ac_curr_bal_diff>=P80 且 ba_out_bal_diff>=P80
    apply_eff_date_filter  : 是否执行生效日筛选（True=筛选，False=跳过），默认 False
    eff_date_lower         : 生效日下限（含），默认 "2025-01-01"
    eff_date_upper         : 生效日上限（含），默认 "2026-03-31"（对应消费偏好表最晚快照）
    dedup_cst_loan         : True 时仅删除字段完全一致的重复账户；冲突重复会报错
    exclude_dq_start_customers : 是否剔除起点已违约客户；None 时仅 y_dq_risk 剔除
    label_threshold_train_ratio: y_freq 的 P80 仅用最早该比例客户拟合；None 表示全样本
    require_dual_label_cohort  : True 时同时要求 y_freq 来源和 y_dq 终点状态完整

    返回
    ----
    df_clean          : 清洗 + 标签构造后的完整 DataFrame（含 cst_id、y_freq 列）
    feature_missing_df: 特征缺失率 DataFrame
    label_stats_df    : 因变量统计 DataFrame
    X                 : 特征矩阵（pd.DataFrame，缺失已处理）
    y                 : 标签向量（pd.Series，y_freq）
    feature_names     : 特征名列表
    thresholds        : y_freq 阈值字典 {y_freq_mode, thr_bout, thr_curr, thr_accr}
    field_coverage_summary_df : 字段覆盖汇总表
    field_coverage_monthly_df : 字段按贷款生效月份覆盖明细表
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
        print(f"  +-- [违约追踪] {label}")
        print(f"  |  总行数={total_rows:,}  总客户={total_cst:,}")
        print(f"  |  终点违约行(in 3/4/7/9)={dq_rows:,}  对应客户数={dq_cst:,}")
        # 各状态码明细（仅针对违约行）
        loan_col = next((c for c in ["loanacctno", "loan_acct_no"] if c in df.columns), None)
        for code in ["3", "4", "7", "9"]:
            mask_c = s == code
            n_rows_c = int(mask_c.sum())
            n_cst_c  = int(df.loc[mask_c, "cst_id"].nunique()) if "cst_id" in df.columns else -1
            n_loan_c = int(df.loc[mask_c, loan_col].nunique()) if loan_col else -1
            if n_rows_c > 0:
                pct_of_dq = n_rows_c / dq_rows * 100 if dq_rows > 0 else 0
                print(f"  |    状态{code}: 行数={n_rows_c:,}({pct_of_dq:.1f}%)  客户数={n_cst_c:,}  贷款账户数={n_loan_c:,}")
        print(f"  +-- 终点状态为空={null_cnt:,}")

    def _count_dq_agg(df, label):
        """聚合后数据（一客一行）：统计 rt_acct_stat_2_end 为违约状态的客户数"""
        if target != "y_dq_risk" or "rt_acct_stat_2_end" not in df.columns:
            return
        s = df["rt_acct_stat_2_end"].map(_clean_stat)
        dq_cst   = int(s.isin(DQ_SET).sum())
        null_cnt = int(df["rt_acct_stat_2_end"].isna().sum())
        total    = len(df)
        print(f"  +-- [违约追踪] {label}")
        print(f"  |  总客户={total:,}")
        print(f"  |  终点违约客户(in 3/4/7/9)={dq_cst:,}  ({dq_cst/total:.2%})")
        print(f"  +-- 终点状态为空={null_cnt:,}")
    # ─────────────────────────────────────────

    print(f"[1/7] 读取文件: {file_path}")
    df0 = read_data(file_path, csv_encoding=csv_encoding)

    print(f"[2/7] 科创字段英文别名恢复为中文...")
    df0 = rename_kechuang_cols(df0)

    print("[2.2/8] 字段日期覆盖统计（列名统一后、其他筛选和聚合前）...")
    field_coverage_summary_df, field_coverage_monthly_df = get_field_date_coverage(df0)
    print(f"  覆盖汇总字段数: {len(field_coverage_summary_df):,}")
    print(f"  覆盖月度明细行数: {len(field_coverage_monthly_df):,}")

    print(f"[2.5/8] (cst_id, loanacctno) 重复检查（dedup={dedup_cst_loan}）...")
    df0 = report_cst_loan_duplicates(df0)
    if dedup_cst_loan:
        n_before = len(df0)
        df0 = deduplicate_exact_cst_loan(df0)
        print(
            f"  已删除字段完全一致的重复账户：{n_before:,} → {len(df0):,} 行，"
            f"删除 {n_before - len(df0):,} 行"
        )
    _count_dq_rows(df0, "重复检查后（行级）")

    print(f"[2.8/8] 类别型字段浮点->整数字符串转换（必须在起点违约剔除之前）...")
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

    exclude_dq_start = (
        target == "y_dq_risk"
        if exclude_dq_start_customers is None
        else bool(exclude_dq_start_customers)
    )
    if exclude_dq_start:
        print("[3.7/8] 剔除起点已违约客户（统一建模客户池）...")
        df0 = filter_dq_start_customers(df0)
        _count_dq_rows(df0, "起点违约剔除后（行级）")

    print("[4/8] 一客多贷聚合...")
    df1 = aggregate_by_customer_potential(df0)
    _count_dq_agg(df1, "聚合后（客户级，一客一行）")

    print("[5/8] 数据清洗 Part1（因变量构造前：删ID/日期/无意义字段）...")
    clean_target = "y_dq_risk" if require_dual_label_cohort else target
    df2 = clean_data_potential(
        df1, snapshot_date=snapshot_date, target=clean_target
    )
    _count_dq_agg(df2, "清洗Part1后（客户级）")

    label_desc = f"y_freq（模式={y_freq_mode}）" if target == "y_freq" else "y_dq_risk"
    print(f"[6/8] 因变量构造（{label_desc}）...")
    threshold_fit_mask = None
    if (target == "y_freq" or require_dual_label_cohort) and label_threshold_train_ratio is not None:
        ratio = float(label_threshold_train_ratio)
        if not 0 < ratio < 1:
            raise ValueError("label_threshold_train_ratio 必须在 (0, 1) 内。")
        if "split_eff_date" not in df2.columns:
            raise KeyError("按时间拟合 y_freq 阈值需要 split_eff_date。")
        label_dates = pd.to_datetime(df2["split_eff_date"], errors="coerce")
        if label_dates.isna().any():
            raise ValueError(
                f"split_eff_date 存在 {int(label_dates.isna().sum()):,} 个无效值，"
                "无法按时间拟合 y_freq 阈值。"
            )
        label_order = np.argsort(label_dates.to_numpy(), kind="mergesort")
        n_threshold_train = int(len(df2) * ratio)
        if n_threshold_train <= 0:
            raise ValueError("y_freq 阈值训练段为空。")
        threshold_fit_mask = np.zeros(len(df2), dtype=bool)
        threshold_fit_mask[label_order[:n_threshold_train]] = True
    if require_dual_label_cohort:
        df3, thresholds_freq = build_labels_potential(
            df2,
            target="y_freq",
            y_freq_mode=y_freq_mode,
            threshold_fit_mask=threshold_fit_mask,
        )
        df3, thresholds_dq = build_labels_potential(
            df3, target="y_dq_risk"
        )
        thresholds = thresholds_freq if target == "y_freq" else thresholds_dq
    else:
        df3, thresholds = build_labels_potential(
            df2,
            target=target,
            y_freq_mode=y_freq_mode,
            threshold_fit_mask=threshold_fit_mask,
        )
    if target == "y_dq_risk" and "y_dq_risk" in df3.columns:
        n_pos = int(df3["y_dq_risk"].sum())
        total  = len(df3)
        print(f"  +-- [违约追踪] 标签构造后")
        print(f"  |  总客户={total:,}  y_dq_risk=1={n_pos:,}  ({n_pos/total:.2%})")
        print(f"  +-- y_dq_risk=0={total - n_pos:,}")

    print("[6.5/8] 数据清洗 Part2（因变量构造后：删泄露/diff/起点无意义字段）...")
    drop_target = "y_dq_risk" if require_dual_label_cohort else target
    df_clean = drop_post_label_cols(df3, target=drop_target)
    if target == "y_dq_risk" and "y_dq_risk" in df_clean.columns:
        n_pos = int(df_clean["y_dq_risk"].sum())
        total  = len(df_clean)
        print(f"  +-- [违约追踪] 清洗Part2后（最终建模样本）")
        print(f"  |  总客户={total:,}  y_dq_risk=1={n_pos:,}  ({n_pos/total:.2%})")
        print(f"  +-- y_dq_risk=0={total - n_pos:,}")

    print("[7/8] 特征统计 + 特征工程...")
    feature_missing_df, label_stats_df = get_feature_stats(df_clean, target=target)

    X, y, feature_names, _ = feature_engineering_potential(
        df_clean, target=target,
        add_quota_sq=add_quota_sq,
        add_quota_cube=add_quota_cube,
        add_quota_log=add_quota_log,
    )

    # ── 最终汇报 ──
    print(f"\n{'='*55}")
    print(f"[完成] 预处理完成")
    print(f"   客户数          : {len(df_clean):,}")
    print(f"   特征数          : {len(feature_names):,}")
    print(f"   建模目标        : {target}")
    if target == "y_freq":
        print(f"   y_freq 模式     : {y_freq_mode}")
    print(f"   正样本率        : {y.mean():.4%}  ({y.sum()} / {len(y)})")
    if target == "y_dq_risk":
        print(f"   最终 y_dq_risk=1: {int(y.sum()):,} 人")
    print(f"{'='*55}")

    return (
        df_clean, feature_missing_df, label_stats_df, X, y, feature_names,
        thresholds, field_coverage_summary_df, field_coverage_monthly_df,
    )
