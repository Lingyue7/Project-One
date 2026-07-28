# 科创人才消费贷款分类预测

本目录用于构建频繁支用预测（`y_freq`）和违约风险预测（`y_dq_risk`）模型。建模粒度为客户：账户级数据先按 `cst_id` 聚合为一客一行，再执行 LightGBM 训练、采样方法比较和最终测试评价。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `kechuang_potential_preprocessing.ipynb` | 主分析 notebook：配置数据与任务，调用预处理、分层划分数据集、训练 LightGBM、比较采样方法、输出最终测试与解释结果。 |
| `load_kechuang_potential_data.py` | 数据预处理模块：读取数据、重复检查/可选去重、日期筛选、一客多贷聚合、标签构造、泄露字段删除和特征工程。 |
| `sampling_methods.py` | 过采样、欠采样、组合采样和集成采样方法。 |

## 数据准备

将基础数据、违约终点状态和 LGD 字段按 `cst_id + loanacctno` 合并后，导出 CSV 或 Excel，随后在 notebook 的配置区设置：

```python
FILE_PATH = 'kechuang_base_risk_lgd_merged.csv'
TARGET = 'y_freq'       # 或 'y_dq_risk'
SNAPSHOT_DATE = '2026-06-24'
```

### 贷款账户重复处理

加载阶段始终输出 `(cst_id, loanacctno)` 的重复组合数、重复涉及行数、重复行数分布，以及重复组中“存在差异字段数”的分布。

```python
DEDUP_CST_LOAN = False
```

- `False`：不删除重复记录，仅输出统计；
- `True`：输出统计后，按 `(cst_id, loanacctno)` 保留第一条。

关闭去重时，重复账户会参与客户级聚合，因此应先检查重复字段差异统计再决定是否开启。

## 标签定义

### 频繁支用 `y_freq`

通过 `Y_FREQ_MODE` 选择口径：

| 模式 | 正样本定义 |
| --- | --- |
| `bout_gt0_and_curr_p80` | `ba_out_bal_diff > 0` 且 `ac_curr_bal_diff >= P80`。 |
| `bout_p80_and_accr_p80` | `ba_out_bal_diff` 和 `ac_accr_bal_diff` 均不低于 P80。 |
| `curr_p80_only` | `ac_curr_bal_diff >= P80`。 |
| `curr_p80_and_bout_p80` | `ac_curr_bal_diff` 和 `ba_out_bal_diff` 均不低于 P80。 |

### 违约风险 `y_dq_risk`

终点账户状态 `rt_acct_stat_2_end` 属于 `3、4、7、9` 时标签为 1，否则为 0。一客多贷时取最差终点状态；任一账户起点已为不良状态的客户会被剔除。标签及贷后表现字段不会进入模型特征。

## 数据划分与阈值

数据不再按开户时间顺序切分，而是通过三次分层随机抽样划分为四个互斥集合：

| 集合 | 比例 | 用途 |
| --- | ---: | --- |
| 训练集 | 60% | 训练候选模型；只有此集合可以采样。 |
| 模型选择验证集 | 15% | 候选方法的 AUC 与 Precision 取舍、早停。 |
| 概率校准集 | 15% | 保留原始分布，供后续需要时增加概率校准。 |
| 最终测试集 | 10% | 方法确定后的最终评价。 |

每次切分均使用标签分层，因此各集合的正样本率尽量保持与总体一致。

不再配置 `KNOWN_POSITIVE_RATE` 或 `pi_k`。每次评估时，阈值自动取该评估集预测概率的：

```text
1 - 当前评估集实际正样本率
```

分位数；因此模型选择验证集按验证集自身正样本率选阈值，最终测试集按测试集自身正样本率报告相应指标。

## 采样方法和模型选择

`sampling_methods.py` 支持：

- 过采样：`random_over`、`smote`、`borderline_smote`、`adasyn`；
- 欠采样：`random_under`、`tomek`、`enn`；
- 组合采样：`smoteenn`、`smotetomek`；
- 集成采样：`balance_cascade`、`easy_ensemble`（`ensemble` 是别名）。

候选方法仅在训练集采样与训练。先按模型选择验证集 AUC 取 Top 3，再比较各方法在验证集自动阈值下的 Precision；选定方法后以训练集和验证集合并重训，最终测试集不参与候选选择。

## 运行方式

安装依赖：

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib seaborn shap openpyxl jupyter imbalanced-learn
```

打开并按顺序运行：

```bash
jupyter notebook kechuang_potential_preprocessing.ipynb
```

主要输出包括：样本与特征统计、采样方法选择表、最终测试指标及混淆矩阵、分位数 Precision/Recall、ROC/PR 曲线、累积增益图、Gain 特征重要性和 SHAP 图。
