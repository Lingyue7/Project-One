# 科创人才消费贷款额度组合优化

本目录提供一套面向科创人才历史客户的授信额度组合优化流程。系统先在不同候选额度下预测客户的支用概率和违约概率，再根据历史数据计算优化参数，并在额度预算、风险预算和人才等级政策约束下，为每名客户选择一个可执行的离散建议额度。

推荐从 `credit_limit_optimization.ipynb` 开始运行。Notebook 串联数据清洗、四段时间划分、概率模型与校准、优化参数计算、`c2` 候选比较、最终测试集离线评价、全量历史客户交叉拟合优化、诊断和可视化。

## 文件说明

| 文件 | 主要职责 |
| --- | --- |
| `credit_limit_optimization.ipynb` | 主流程 Notebook；集中配置数据、模型、参数计算、候选实测、优化和报告。 |
| `parameter_selection.py` | 计算平均支用率与 `c1`、历史回收法 LGD、人才等级额度上限，并保存参数审计结果。 |
| `load_kechuang_potential_data.py` | 将账户级数据聚合为客户级样本，构造双标签并处理泄露字段。 |
| `sampling_methods.py` | 类别不平衡采样工具；采样只允许作用于模型训练数据。 |
| `generate_probability_grid.py` | 执行时间划分、训练支用/违约模型并生成开发阶段和全量折外校准概率网格。 |
| `probability_calibration.py` | Isotonic 校准、Brier Score、ECE 和校准曲线评估工具。 |
| `portfolio_milp_optimizer.py` | 构造并求解离散 0-1 组合整数规划。 |
| `optimal_credit_limit_precomputed_grid_large.py` | 读取概率网格，运行 `c2` 敏感性分析或正式优化，输出组合评价报告。 |
| `analyze_oof_credit_limit.py` | 生成折外概率分箱、ALE、概率曲线和组合约束诊断。 |
| `viz_credit_limit.py` | 生成额度、概率、等级、风险和目标函数图表。 |
| `修改记录_0716到0723.md` | 记录从 0716 基线代码到 0723 当前代码的全部调整。 |

## 运行流程

```text
原始账户数据
    ↓
客户级聚合、双标签构造、泄露字段删除
    ↓
按时间划分 train / validation / calibration / final test
            60%          15%          15%          10%
    ↓
固定采样方法后，用 train+validation 重新训练开发模型
    ↓
仅用 calibration 拟合 Isotonic 校准器
    ↓
用 train+validation 客户和独立历史资料计算优化参数
    ↓
在开发拟合集独立求解四个 c2 候选并选定参数
    ↓
final test 一次性离线评价
    ↓
全量历史客户嵌套交叉拟合概率 + 联合组合优化
```

最终测试集不参与模型训练、概率校准、参数估计、`c2` 选择或约束调节。

## 数据划分与概率口径

历史客户按 `split_eff_date` 稳定排序，默认划分为：

| 数据集 | 比例 | 用途 |
| --- | ---: | --- |
| `train` | 60% | 分类模型和采样方法开发 |
| `validation` | 15% | 采样方法选择 |
| `calibration` | 15% | 只拟合概率校准器 |
| `final test` | 10% | 最终一次性离线评价 |

采样方法选定后，开发模型在 `train + validation` 上重新拟合。特征缺失率、数值填补、类别字典等数据相关规则也只在这个开发拟合集上拟合。校准器只读取校准集在历史实际额度点的原始预测与真实标签。

开发阶段概率网格默认保存在：

```text
probability_grid_large_dev_calibrated/
```

其中保存 `train_idx.npy`、`validation_idx.npy`、`fit_idx.npy`、`cal_idx.npy` 和 `test_idx.npy`，供后续步骤检查数据边界。

全量历史客户优化使用嵌套交叉拟合生成的样本外校准概率。每个外层目标折的标签不进入对应模型或校准器。目录默认是：

```text
probability_grid_large_crossfit_calibrated/
```

## 风险调整目标

对客户 `i` 和候选额度 `L`，优化目标为：

```text
pi_i(L) = r_i × L × p_usage_i(L)
          − LGD_i × L × p_default_i(L)
          − c1 × L
          − c2 × L²
```

人才等级不直接乘入收益，而是通过分档额度上下限和组均额度约束进入优化。

## 参数设置

### 单位额度收益率

`INTEREST_RATE` 是单位额度净收益率或综合收益系数，默认配置为 `0.03`。如银行已有统一业务口径，应在运行前替换。

### 平均支用率与线性成本

代码只使用 `train + validation` 开发拟合集计算历史平均支用率：

1. 按客户汇总历史额度和实际支用余额；
2. 计算客户支用率，并限制到 `[0, 1]`；
3. 默认按 1% 和 99% 分位进行 Winsorize；
4. 取处理后的客户平均支用率 `u_bar`；
5. 按 `c1 = 2% × u_bar` 计算线性成本。

对应配置为：

```python
FTP_RATE = 0.02
UTILIZED_BALANCE_COLUMN = "ac_curr_bal"
UTILIZATION_WINSOR_QUANTILES = (0.01, 0.99)
```

计算摘要与客户明细分别写入 `utilization_parameter_summary.csv` 和 `utilization_customer_detail.csv`。

### LGD

若有逐期历史回收明细，在 `LGD_HISTORY_FILE` 填入 CSV/XLSX 路径，并按实际字段修改 `LGD_COLUMN_MAP`。代码按以下口径计算：

- 违约日取贷款首次进入违约状态的日期；
- EAD 取违约日未偿本金；
- 本金回收额取观察终点累计已还本金减去违约日累计已还本金；
- 观察终点取违约后 12 个月、结清日和数据截止日中的最早日期；
- 基准 LGD 只使用已观察满 12 个月或已结清样本；
- 使用 EAD 加权汇总 LGD；
- 未完成样本的当前 LGD 单独报告，不混入基准值；
- 同时报告样本数、观察月数、结清、核销、EAD 和回收额。

若缺少可靠回收字段，代码不会把假设值写成历史经验值，而是明确使用情景：

```python
LGD_HISTORY_FILE = None
LGD_SELECTED_SCENARIO = "neutral"
LGD_SCENARIOS = {
    "optimistic": 0.30,
    "neutral": 0.45,
    "conservative": 0.60,
}
```

### 二次成本系数

参考额度为 110 万元，预设四个候选值：

```python
C2_REFERENCE_LIMIT = 1_100_000.0
C2_CANDIDATES = (0.0, 1e-9, 2.5e-9, 5e-9)
```

Notebook 会在开发拟合集上分别运行四次组合优化，并比较：

- 模型预计净收益；
- 相对历史方案的目标变化；
- 零额度率；
- 分档上限命中率；
- 组合额度加权违约风险；
- 风险预算使用率；
- 总额度、平均/中位数及 5%/95% 分位额度调整。

如业务已给出不可接受阈值，可填写：

```python
C2_MAX_ZERO_RATE = None
C2_MAX_UPPER_HIT_RATE = None
```

代码先排除超过阈值的候选，再选择模型预计净收益最高者；净收益在默认 1% 相对容忍范围内时选择较小 `c2`。完整结果保存在：

```text
reports/parameter_selection/c2_selection/c2_sensitivity_summary.csv
```

### 人才等级额度上下限

当前政策只定义 F3、F2、F1、E、D 五档，数字映射为：

| 数字等级 | 档位 |
| ---: | --- |
| 1 | F3 |
| 2 | F2 |
| 3 | F1 |
| 4 | E |
| 5 | D |

所有等级最低额度均为 0。最高额度只使用开发拟合集历史额度计算：

1. 计算全体客户历史额度 P99，记为 `U0`；
2. F3/F2/F1 档按 `w=n/(n+500)` 对该档历史最大额度和 `U0` 做样本量收缩；
3. 强制 F3、F2、F1 上限非递减；
4. E 档上限取 `max(F1上限, 1.1×U0)`；
5. D 档上限取 `max(E上限, 1.2×U0)`；
6. 上限向下对齐到业务额度网格，并不超过 `GRID_MAX`。

如果数据中出现未定义的 A/B/C 等级，参数计算会直接报错并终止整个流程。代码不会删除 A/B/C 客户后继续优化其余等级，也不会让 A/B/C 在缺少专门额度上限和组均约束的情况下进入优化。必须先补充并确认 A/B/C 的政策规则，才能将这三类客户纳入正式优化。

### 组均额度约束

对当前组合中实际存在的名义相邻等级，默认执行：

```text
mean(F2) ≥ mean(F3)
mean(F1) ≥ mean(F2)
mean(E)  ≥ 0.8 × mean(F1)
mean(D)  ≥ 0.8 × mean(E)
```

若其中某档在当前组合缺失，与该档有关的约束跳过，不跨档拼接新的约束。

### 总额度与风险预算

当 `TOTAL_BUDGET=None` 时，总额度预算等于当前评价组合的历史实际额度总和。

当 `RISK_BUDGET=None` 时，风险预算按下式由当前组合的历史额度和校准违约概率计算：

```text
R = 1.05 × Σ[历史额度_i × 校准违约概率_i(历史额度_i)]
```

风险预算不使用真实违约标签构造。最终测试阶段沿用同一 1.05 容忍系数。

## 求解方法

每名客户必须从其人才等级允许的离散额度网格中选择一个额度。候选额度的支用概率、违约概率、风险和目标系数均在求解前计算，随后由 `scipy.optimize.milp` 的 HiGHS 求解器执行一次联合 0-1 整数规划。

若完整网格变量数超过 `MILP_MAX_VARIABLES`，代码会保留目标较高、上下界、历史额度邻近点和均匀覆盖点等代表性候选。是否压缩、变量数、求解状态和 MIP gap 均写入 `solver_summary.csv`。

## 运行前检查

至少需要确认 Notebook 配置区的以下内容：

```python
DATA_FILE = "实际原始数据文件.csv"
SAMPLING_METHOD = None        # 填写任务2已经选定的方法；无采样则保留 None
LGD_HISTORY_FILE = None       # 有可靠回收明细时填写路径
LGD_SELECTED_SCENARIO = "neutral"
INTEREST_RATE = 0.03
```

如果填写 `LGD_HISTORY_FILE`，还必须把 `LGD_COLUMN_MAP` 改成文件中的真实列名。

运行前还应检查人才等级取值。如果数据中存在 A、B 或 C，当前流程会停止，不会自动过滤这些客户并继续运行。此时应先补充 A/B/C 的最高额度及平均额度约束规则。

首次运行或任何上游数据、模型、采样、网格、参数和政策发生变化时，应设置：

```python
REUSE_PROBABILITY_GRID = False
REUSE_OPTIMIZATION = False
```

安装依赖：

```bash
pip install numpy pandas scipy scikit-learn lightgbm matplotlib seaborn openpyxl jupyter
```

运行：

```bash
jupyter notebook credit_limit_optimization.ipynb
```

建议从上到下顺序执行全部 Cell，不要先运行最终测试或全量优化 Cell。

## 主要输出

参数审计目录：

```text
reports/parameter_selection/
```

主要包括：

- `derived_parameters.json`：最终传入优化器的参数及来源；
- `utilization_parameter_summary.csv`：平均支用率和异常处理摘要；
- `utilization_customer_detail.csv`：客户级支用率明细；
- `lgd_parameter_summary.csv`：LGD 结果、样本状态或情景说明；
- `lgd_loan_detail.csv`：逐贷款 LGD 明细（使用历史回收法时生成）；
- `tier_limit_parameter_summary.csv`：分档样本量、历史最大额、收缩权重和最终上限；
- `c2_selection/c2_sensitivity_summary.csv`：四个 `c2` 候选的比较和最终选择。

最终测试集离线评价目录：

```text
reports/test_offline_evaluation/
```

全量历史客户优化目录：

```text
reports/full_crossfit_optimization/
```

正式优化目录主要包含客户级结果、组合摘要、额度调整分布、人才等级摘要、概率变化、边界命中、集中度、求解器状态、复用状态和图表。

## 结果解释边界

- `objective_change` 是当前概率模型、参数和约束下的模型目标变化，不能直接解释为实际利润提升。
- 概率网格是模型情景预测，不是提高额度对支用或违约的因果效应。
- 历史额度可能不满足新的人才等级区间或组均约束，因此历史方案未必属于相同可行域。
- 0 额度表示允许拒贷或不授信，是否可用须由业务和风控政策确认。
- 未填写业务阈值时，`c2` 候选不会因零额度率或上限命中率自动淘汰，应由业务查看敏感性表确认。
- 输出额度只能用于决策支持；上线前仍需时间外验证、压力测试、风险审批、政策验证、合规审查和人工兜底。
