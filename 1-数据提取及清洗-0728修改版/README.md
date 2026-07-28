# 数据提取、合并与清洗

本目录用于提取科创客户消费贷款数据、补充违约标签和 LGD 所需字段，并合并为后续分类模型可读取的数据文件。

## SQL 文件

| 文件 | 用途 |
| --- | --- |
| `data_extraction_kechuang.txt` | 基础特征与频繁支用任务所需字段。包含科创客户、消贷账户、起终点余额、授信台账、客户画像、消费行为和手机银行特征；不提取违约终点状态。 |
| `data_extraction_kechuang_risk_snapshot.txt` | 仅提取违约因变量：`cst_id`、`loanacctno`、`risk_snap_dt`、`rt_acct_stat_2_end`。风险快照日为 `min(贷款到期日, 观察日)-1天`。 |
| `loan_default_transition_repayment_observation.txt` | 提取 LGD/违约后还款观察所需字段。观察起点为首次“非违约→违约”状态变化日；观察终点为起点后 12 个月、贷款到期日、观察日三者较早者。 |

三个 SQL 均通过 `cst_id + loanacctno` 关联。运行 SQL 后，请将结果导出为 CSV、XLSX 或 XLS。

## 合并三个抽数结果

使用 [merge_kechuang_base_risk_lgd.ipynb](merge_kechuang_base_risk_lgd.ipynb)。在第一个代码单元配置：

```python
BASE_FILE = '基础数据.csv'
RISK_FILE = '违约因变量数据.csv'
LGD_FILE  = 'LGD观察数据.csv'
OUTPUT_FILE = 'kechuang_base_risk_lgd_merged.csv'
```

合并顺序为：

1. 基础数据左连接违约因变量数据；
2. 上述结果左连接 LGD 数据。

默认使用 `cst_id + loanacctno` 的一对一键验证。notebook 会输出每份数据的重复键统计和各步骤的匹配/未匹配行数。若业务确认存在一对多关系，可把 `MERGE_VALIDATE` 改为 `None`，并人工核对重复键和合并后行数。

## 清洗与特征工程

`load_kechuang_potential_data.py` 负责：

- 读取 CSV / Excel、标准化字段名；
- 输出 `(cst_id, loanacctno)` 重复组和字段差异统计；
- 可选按贷款键去重（保留第一条）；
- 到期日和生效日筛选；
- 一客多贷聚合、双目标标签构造、泄露字段删除；
- 缺失值处理、类别编码和额度衍生特征。

在分类 notebook 中使用 `DEDUP_CST_LOAN` 控制贷款键处理：

```python
DEDUP_CST_LOAN = False  # False：仅统计重复；True：统计后按键保留第一条
```

注意：若关闭去重，重复账户记录会参与后续一客多贷聚合；请结合重复字段差异统计决定是否开启该开关。
