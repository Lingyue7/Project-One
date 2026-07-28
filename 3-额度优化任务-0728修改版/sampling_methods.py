"""
sampling_methods.py
════════════════════════════════════════════════════════════════════════
纯 NumPy / Pandas 实现的类别不平衡采样方法，无需安装 imbalanced-learn。

包含方法
--------
过采样
  1. SMOTE              — 少数类线性插值过采样 (Chawla et al. 2002)
  2. BorderlineSMOTE    — 仅对边界危险样本插值 (Han et al. 2005)
  3. ADASYN             — 自适应权重过采样     (He et al. 2008)

欠采样
  4. RandomUnderSampler — 随机丢弃多数类
  5. TomekLinks         — 删除 Tomek 对中的多数类样本
  6. ENN                — 编辑最近邻欠采样

组合
  7. SMOTEENN           — SMOTE + ENN 清洗     (Batista et al. 2004)
  8. SMOTETomek         — SMOTE + Tomek Links 清洗
  9. BalanceCascade     — 级联欠采样集成        (Liu et al. 2006 变体)
 10. EasyEnsemble       — 随机欠采样集成子集    (Liu et al. 2009)

使用示例
--------
    from sampling_methods import sampler_factory

    # 单次采样，返回 (X_res, y_res)
    X_res, y_res = sampler_factory(
        method="smote",
        sampling_strategy=0.1,
        random_state=42,
    ).fit_resample(X_train, y_train)

    # BalanceCascade / EasyEnsemble 返回子集列表
    subsets = sampler_factory("balance_cascade", n_estimators=10).fit_resample(X, y)
    # subsets: [(X_sub0, y_sub0), (X_sub1, y_sub1), ...]

参数说明
--------
sampling_strategy : float
    采样后 正样本数 / 负样本数 的目标比值。
    0.1  → 正样本补/删 到负样本的 10%（正样本率约 9%）
    0.2  → 正样本补/删 到负样本的 20%（正样本率约 17%）
    "auto" / 1.0 → 补到 1:1（一般不推荐，易过拟合）
k_neighbors : int
    SMOTE 类方法的近邻数，默认 5。
    正样本很少时会自动降低到 min(k, n_pos-1)。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────

def _to_numpy(X, y) -> Tuple[np.ndarray, np.ndarray]:
    """统一转成 numpy array，保留列顺序（DataFrame → ndarray）。"""
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    return X_arr.astype(float), y_arr.astype(int)


def _wrap_output(X_res: np.ndarray, y_res: np.ndarray,
                 original_X) -> Tuple:
    """如果原始输入是 DataFrame，输出也包成 DataFrame。"""
    if hasattr(original_X, "columns"):
        return pd.DataFrame(X_res, columns=original_X.columns), pd.Series(y_res)
    return X_res, y_res


def _knn_indices(X: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """
    暴力 KNN（欧氏距离）。
    返回 query 中每个样本在 X 中最近的 k 个索引（不含自身）。
    query 可以是 X 的子集，也可以完全一致（查自身时会跳过距离=0的点）。

    参数
    ----
    X     : (n, d) 参考集
    query : (m, d) 查询集
    k     : 近邻数

    返回
    ----
    indices : (m, k) int array
    """
    # 分块计算避免大矩阵 OOM；每块 512 行
    m = query.shape[0]
    indices = np.empty((m, k), dtype=int)
    block = 512
    for start in range(0, m, block):
        end = min(start + block, m)
        q = query[start:end]                          # (b, d)
        # 平方距离: ||q - X||^2 = ||q||^2 - 2*q@X^T + ||X||^2
        dist2 = (
            (q ** 2).sum(axis=1, keepdims=True)
            - 2 * q @ X.T
            + (X ** 2).sum(axis=1)
        )                                              # (b, n)
        # 排除自身（距离精确为 0 的点）
        dist2[dist2 < 1e-12] = np.inf
        # 取最近 k 个
        idx = np.argpartition(dist2, k, axis=1)[:, :k]
        # 精确排序（argpartition 不保证顺序）
        for i in range(end - start):
            idx[i] = idx[i][np.argsort(dist2[i + start - start, idx[i]])]
        indices[start:end] = idx
    return indices


def _resolve_strategy(strategy, n_pos: int, n_neg: int) -> int:
    """
    将 sampling_strategy 转换为"目标正样本数"。
    strategy=0.1 → 目标正样本数 = round(n_neg * 0.1)
    strategy="auto" 或 1.0 → 补到 1:1
    """
    if strategy == "auto":
        strategy = 1.0
    ratio = float(strategy)
    target_pos = int(round(n_neg * ratio))
    return max(target_pos, n_pos)   # 只增不减（过采样语义）


def _resolve_under_strategy(strategy, n_pos: int, n_neg: int) -> int:
    """
    将 sampling_strategy 转换为"目标负样本数"（欠采样语义）。
    strategy=0.1 → 目标负样本数 = round(n_pos / 0.1)
    """
    if strategy == "auto":
        strategy = 1.0
    ratio = float(strategy)
    target_neg = int(round(n_pos / ratio))
    return min(target_neg, n_neg)   # 只减不增


# ─────────────────────────────────────────────────────────────────────
# 过采样方法
# ─────────────────────────────────────────────────────────────────────

class SMOTE:
    """
    Synthetic Minority Over-sampling Technique
    参考：Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling
          Technique. Journal of Artificial Intelligence Research, 16, 321-357.

    原理：对每个少数类样本，在其 k 个最近邻（少数类内部）中随机选一个，
         在两者连线上随机插值生成合成样本。

    参数
    ----
    sampling_strategy : float 或 "auto"
        目标 正/负 比值。
    k_neighbors : int
        插值近邻数，默认 5。
    random_state : int 或 None
    """

    def __init__(self,
                 sampling_strategy: Union[float, str] = 0.1,
                 k_neighbors: int = 5,
                 random_state: Optional[int] = None):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X_arr, y_arr = _to_numpy(X, y)

        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)

        target_pos = _resolve_strategy(self.sampling_strategy, n_pos, n_neg)
        n_synthetic = target_pos - n_pos
        if n_synthetic <= 0:
            return _wrap_output(X_arr, y_arr, X)

        X_pos = X_arr[pos_idx]
        k = min(self.k_neighbors, n_pos - 1)
        if k < 1:
            # 正样本太少，直接随机复制
            chosen = rng.choice(n_pos, size=n_synthetic, replace=True)
            X_syn = X_pos[chosen]
        else:
            nn_idx = _knn_indices(X_pos, X_pos, k)        # (n_pos, k)
            chosen = rng.randint(0, n_pos, size=n_synthetic)
            neighbor = nn_idx[chosen, rng.randint(0, k, size=n_synthetic)]
            lam = rng.uniform(0, 1, size=(n_synthetic, 1))
            X_syn = X_pos[chosen] + lam * (X_pos[neighbor] - X_pos[chosen])

        X_res = np.vstack([X_arr, X_syn])
        y_res = np.concatenate([y_arr, np.ones(n_synthetic, dtype=int)])
        return _wrap_output(X_res, y_res, X)


class BorderlineSMOTE:
    """
    Borderline-SMOTE（变体1：仅用少数类近邻插值）
    参考：Han et al. (2005). Borderline-SMOTE: A New Over-Sampling Method
          in Imbalanced Data Sets Learning. ICIC 2005, LNCS 3644, 878-887.

    原理：
      1. 对每个少数类样本，在全体训练集中找 m 个最近邻
      2. 若多数类近邻占比 ∈ [0.5, 1.0) → 危险样本（DANGER），参与过采样
      3. 占比 = 1.0 → 噪声，跳过；< 0.5 → 安全，跳过
      4. 仅在危险样本的少数类近邻之间插值
    """

    def __init__(self,
                 sampling_strategy: Union[float, str] = 0.1,
                 k_neighbors: int = 5,
                 m_neighbors: int = 10,
                 random_state: Optional[int] = None):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X_arr, y_arr = _to_numpy(X, y)

        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)

        target_pos = _resolve_strategy(self.sampling_strategy, n_pos, n_neg)
        n_synthetic = target_pos - n_pos
        if n_synthetic <= 0:
            return _wrap_output(X_arr, y_arr, X)

        X_pos = X_arr[pos_idx]
        m = min(self.m_neighbors, len(X_arr) - 1)

        # 步骤1：在全体数据中找每个正样本的 m 近邻
        nn_all = _knn_indices(X_arr, X_pos, m)           # (n_pos, m)
        # 统计多数类近邻占比
        is_neg_neighbor = (y_arr[nn_all] == 0)           # (n_pos, m) bool
        danger_ratio = is_neg_neighbor.mean(axis=1)       # (n_pos,)

        # 危险样本：0.5 ≤ ratio < 1.0
        danger_mask = (danger_ratio >= 0.5) & (danger_ratio < 1.0)
        danger_local_idx = np.where(danger_mask)[0]       # 在 X_pos 中的索引

        if len(danger_local_idx) == 0:
            # 没有危险样本，回退到普通 SMOTE
            return SMOTE(self.sampling_strategy,
                         self.k_neighbors,
                         self.random_state).fit_resample(X, y)

        X_danger = X_pos[danger_local_idx]
        k = min(self.k_neighbors, n_pos - 1)

        # 步骤2：危险样本在少数类内部找 k 近邻
        nn_pos = _knn_indices(X_pos, X_danger, k)         # (n_danger, k)

        chosen = rng.randint(0, len(danger_local_idx), size=n_synthetic)
        neighbor = nn_pos[chosen, rng.randint(0, k, size=n_synthetic)]
        lam = rng.uniform(0, 1, size=(n_synthetic, 1))
        X_syn = X_danger[chosen] + lam * (X_pos[neighbor] - X_danger[chosen])

        X_res = np.vstack([X_arr, X_syn])
        y_res = np.concatenate([y_arr, np.ones(n_synthetic, dtype=int)])
        return _wrap_output(X_res, y_res, X)


class ADASYN:
    """
    Adaptive Synthetic Sampling
    参考：He et al. (2008). ADASYN: Adaptive Synthetic Sampling Approach
          for Imbalanced Learning. IJCNN 2008, 1322-1328.

    原理：根据每个少数类样本周围多数类的密度（学习难度）自适应分配合成数量，
         难度越高的样本生成越多合成样本。
    """

    def __init__(self,
                 sampling_strategy: Union[float, str] = 0.1,
                 k_neighbors: int = 5,
                 random_state: Optional[int] = None):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X_arr, y_arr = _to_numpy(X, y)

        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)

        target_pos = _resolve_strategy(self.sampling_strategy, n_pos, n_neg)
        G = target_pos - n_pos          # 总合成数
        if G <= 0:
            return _wrap_output(X_arr, y_arr, X)

        X_pos = X_arr[pos_idx]
        k = min(self.k_neighbors, len(X_arr) - 1)

        # 计算每个正样本的"学习难度"：k 近邻中负样本占比
        nn_all = _knn_indices(X_arr, X_pos, k)
        r = (y_arr[nn_all] == 0).mean(axis=1).astype(float)   # (n_pos,)

        r_sum = r.sum()
        if r_sum < 1e-12:
            # 所有正样本都是安全样本，回退到均匀 SMOTE
            return SMOTE(self.sampling_strategy,
                         self.k_neighbors,
                         self.random_state).fit_resample(X, y)

        r_hat = r / r_sum               # 归一化权重
        g_i = np.round(r_hat * G).astype(int)

        # 少数类内部近邻（用于插值）
        k_pos = min(self.k_neighbors, n_pos - 1)
        if k_pos < 1:
            nn_pos = None
        else:
            nn_pos = _knn_indices(X_pos, X_pos, k_pos)

        X_syn_list = []
        for i, gi in enumerate(g_i):
            if gi <= 0:
                continue
            if nn_pos is None or k_pos < 1:
                X_syn_list.append(np.tile(X_pos[i], (gi, 1)))
            else:
                neighbors = nn_pos[i]
                chosen_nb = rng.choice(neighbors, size=gi, replace=True)
                lam = rng.uniform(0, 1, size=(gi, 1))
                X_syn_list.append(
                    X_pos[i] + lam * (X_pos[chosen_nb] - X_pos[i])
                )

        if not X_syn_list:
            return _wrap_output(X_arr, y_arr, X)

        X_syn = np.vstack(X_syn_list)
        n_syn = len(X_syn)
        X_res = np.vstack([X_arr, X_syn])
        y_res = np.concatenate([y_arr, np.ones(n_syn, dtype=int)])
        return _wrap_output(X_res, y_res, X)


# ─────────────────────────────────────────────────────────────────────
# 欠采样方法
# ─────────────────────────────────────────────────────────────────────

class RandomUnderSampler:
    """
    随机欠采样：随机丢弃多数类样本。

    参数
    ----
    sampling_strategy : float
        目标 正/负 比值（欠采样降低分母）。
    """

    def __init__(self,
                 sampling_strategy: Union[float, str] = 0.1,
                 random_state: Optional[int] = None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X_arr, y_arr = _to_numpy(X, y)

        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)

        target_neg = _resolve_under_strategy(self.sampling_strategy, n_pos, n_neg)
        keep_neg = rng.choice(neg_idx, size=target_neg, replace=False)
        keep_idx = np.concatenate([pos_idx, keep_neg])
        rng.shuffle(keep_idx)

        return _wrap_output(X_arr[keep_idx], y_arr[keep_idx], X)


class TomekLinks:
    """
    Tomek Links 欠采样：删除互为最近邻的跨类别样本对中的多数类样本。
    参考：Tomek (1976). Two Modifications of CNN. IEEE Trans. SMC, 6(11).
         Batista et al. (2004). A Study of the Behavior of Several Methods
         for Balancing Machine Learning Training Data. SIGKDD Explor., 6(1).

    原理：若样本 a（少数类）和 b（多数类）互为彼此的最近邻，
         则 (a, b) 构成一个 Tomek Link，删除 b（多数类侧）。
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def fit_resample(self, X, y):
        X_arr, y_arr = _to_numpy(X, y)
        n = len(X_arr)

        # 找每个样本在全体中的最近邻（1-NN，不含自身）
        nn1 = _knn_indices(X_arr, X_arr, 1).ravel()   # (n,)

        remove = set()
        for i in range(n):
            j = nn1[i]
            # i 和 j 互为最近邻，且类别不同
            if nn1[j] == i and y_arr[i] != y_arr[j]:
                # 删除多数类（y==0）
                if y_arr[i] == 0:
                    remove.add(i)
                else:
                    remove.add(j)

        keep = np.array([i for i in range(n) if i not in remove])
        return _wrap_output(X_arr[keep], y_arr[keep], X)


class ENN:
    """
    Edited Nearest Neighbors 欠采样
    参考：Wilson (1972). Asymptotic Properties of Nearest Neighbor Rules
          Using Edited Data. IEEE Trans. SMC, 2(3), 408-421.

    原理：对每个样本找 k 个近邻；若多数近邻与自身类别不同，则删除该样本。
          默认只删除多数类样本（保护少数类）。
    """

    def __init__(self,
                 k_neighbors: int = 3,
                 kind_sel: str = "majority",
                 random_state: Optional[int] = None):
        """
        kind_sel : "majority" → 只删多数类误分样本（推荐）
                   "all"      → 删除所有被误分样本（会也删少数类噪声）
        """
        self.k_neighbors = k_neighbors
        self.kind_sel = kind_sel
        self.random_state = random_state

    def fit_resample(self, X, y):
        X_arr, y_arr = _to_numpy(X, y)
        n = len(X_arr)
        k = min(self.k_neighbors, n - 1)

        nn_idx = _knn_indices(X_arr, X_arr, k)
        # 多数投票：近邻中占多数的类别
        nn_labels = y_arr[nn_idx]           # (n, k)
        voted = (nn_labels.mean(axis=1) >= 0.5).astype(int)  # 近邻多数类

        mismatch = (voted != y_arr)
        if self.kind_sel == "majority":
            # 只删多数类（y==0）中被误判的
            remove_mask = mismatch & (y_arr == 0)
        else:
            remove_mask = mismatch

        keep = np.where(~remove_mask)[0]
        return _wrap_output(X_arr[keep], y_arr[keep], X)


# ─────────────────────────────────────────────────────────────────────
# 组合方法
# ─────────────────────────────────────────────────────────────────────

class SMOTEENN:
    """
    SMOTE + ENN 组合采样
    参考：Batista et al. (2004). A Study of the Behavior of Several Methods
          for Balancing Machine Learning Training Data. SIGKDD Explor., 6(1).

    两步：
      1. SMOTE 过采样（按 sampling_strategy 生成合成正样本）
      2. ENN 清洗（删除边界附近被误判的多数类样本）
    """

    def __init__(self,
                 sampling_strategy: Union[float, str] = 0.1,
                 k_neighbors: int = 5,
                 enn_k: int = 3,
                 random_state: Optional[int] = None):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.enn_k = enn_k
        self.random_state = random_state

    def fit_resample(self, X, y):
        # Step 1: SMOTE
        X_res, y_res = SMOTE(
            self.sampling_strategy, self.k_neighbors, self.random_state
        ).fit_resample(X, y)

        # Step 2: ENN
        X_res, y_res = ENN(
            k_neighbors=self.enn_k, kind_sel="majority"
        ).fit_resample(X_res, y_res)

        return X_res, y_res


class SMOTETomek:
    """
    SMOTE + Tomek Links 组合采样
    参考：Batista et al. (2004).

    两步：
      1. SMOTE 过采样
      2. Tomek Links 清洗（删除边界处的多数类样本）
    """

    def __init__(self,
                 sampling_strategy: Union[float, str] = 0.1,
                 k_neighbors: int = 5,
                 random_state: Optional[int] = None):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        # Step 1: SMOTE
        X_res, y_res = SMOTE(
            self.sampling_strategy, self.k_neighbors, self.random_state
        ).fit_resample(X, y)

        # Step 2: Tomek
        X_res, y_res = TomekLinks().fit_resample(X_res, y_res)

        return X_res, y_res


class BalanceCascade:
    """
    Balance Cascade（级联欠采样集成）
    参考：Liu et al. (2009). Exploratory Undersampling for Class-Imbalance
          Learning. IEEE Trans. SMC-B, 39(2), 539-550.

    原理：
      - 将多数类分成 T 轮，每轮：
          1. 随机抽取负样本子集（与正样本 1:1 或按 ratio）
          2. 构成一个平衡子集 (X_sub, y_sub)
          3. 可选：训练一个简单分类器，把已被正确识别的负样本移除（级联）
      - 本实现的简化版：纯随机级联（不依赖分类器），每轮抽一个不重叠子集

    返回
    ----
    list of (X_sub, y_sub)：n_estimators 个子集，供上层做集成训练。

    用法示例
    --------
        subsets = BalanceCascade(n_estimators=10).fit_resample(X_train, y_train)
        models  = []
        for X_sub, y_sub in subsets:
            m = LGBMClassifier().fit(X_sub, y_sub)
            models.append(m)
        # 预测时对所有模型取概率均值
    """

    def __init__(self,
                 n_estimators: int = 10,
                 ratio: float = 1.0,
                 random_state: Optional[int] = None):
        """
        n_estimators : 子集数量（基分类器数量）
        ratio        : 每个子集中 负/正 的比值（默认 1:1）
        """
        self.n_estimators = n_estimators
        self.ratio = ratio
        self.random_state = random_state

    def fit_resample(self, X, y) -> List[Tuple]:
        rng = np.random.RandomState(self.random_state)
        X_arr, y_arr = _to_numpy(X, y)

        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        n_pos = len(pos_idx)

        neg_per_sub = min(int(round(n_pos * self.ratio)), len(neg_idx))
        # 如果负样本不够分 T 轮，允许重复
        rng.shuffle(neg_idx)

        subsets = []
        remaining_neg = neg_idx.copy()

        for t in range(self.n_estimators):
            if len(remaining_neg) < neg_per_sub:
                # 不够了，重新从所有负样本中采
                sampled_neg = rng.choice(neg_idx, size=neg_per_sub, replace=False)
            else:
                sampled_neg = remaining_neg[:neg_per_sub]
                remaining_neg = remaining_neg[neg_per_sub:]

            sub_idx = np.concatenate([pos_idx, sampled_neg])
            rng.shuffle(sub_idx)
            sub_X, sub_y = _wrap_output(X_arr[sub_idx], y_arr[sub_idx], X)
            subsets.append((sub_X, sub_y))

        return subsets


class EasyEnsemble:
    """
    EasyEnsemble（随机欠采样集成）
    参考：Liu et al. (2009). Exploratory Undersampling for Class-Imbalance
          Learning. IEEE Trans. SMC-B, 39(2), 539-550.

    原理：从多数类中随机独立地（有放回）抽取 T 个子集，
         每个子集与全部少数类合并构成平衡训练集，分别训练 T 个分类器，
         最终对 T 个分类器的输出取平均（AdaBoost 或简单均值）。

    本实现：返回 T 个 (X_sub, y_sub) 子集列表，分类器训练由调用方完成。

    与 BalanceCascade 的区别：
      - BalanceCascade：级联，每轮剔除已正确识别的负样本（有记忆）
      - EasyEnsemble  ：独立随机采样，每轮相互独立（无记忆，更简单）
    """

    def __init__(self,
                 n_estimators: int = 10,
                 ratio: float = 1.0,
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.ratio = ratio
        self.random_state = random_state

    def fit_resample(self, X, y) -> List[Tuple]:
        rng = np.random.RandomState(self.random_state)
        X_arr, y_arr = _to_numpy(X, y)

        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        n_pos = len(pos_idx)
        neg_per_sub = min(int(round(n_pos * self.ratio)), len(neg_idx))

        subsets = []
        for t in range(self.n_estimators):
            seed_t = rng.randint(0, 2**31)
            rng_t = np.random.RandomState(seed_t)
            sampled_neg = rng_t.choice(neg_idx, size=neg_per_sub, replace=False)
            sub_idx = np.concatenate([pos_idx, sampled_neg])
            rng_t.shuffle(sub_idx)
            sub_X, sub_y = _wrap_output(X_arr[sub_idx], y_arr[sub_idx], X)
            subsets.append((sub_X, sub_y))

        return subsets


# ─────────────────────────────────────────────────────────────────────
# 工厂函数（统一入口）
# ─────────────────────────────────────────────────────────────────────

def sampler_factory(method: str,
                    sampling_strategy: Union[float, str] = 0.1,
                    k_neighbors: int = 5,
                    random_state: Optional[int] = None,
                    **kwargs):
    """
    采样器工厂：根据 method 字符串返回对应的采样器实例。

    支持的 method（不区分大小写）
    --------------------------------
    过采样：
      "smote"            → SMOTE
      "borderline_smote" → BorderlineSMOTE
      "adasyn"           → ADASYN

    欠采样：
      "random_under"     → RandomUnderSampler
      "tomek"            → TomekLinks（strategy/k_neighbors 无效）
      "enn"              → ENN

    组合：
      "smoteenn"         → SMOTEENN
      "smotetomek"       → SMOTETomek
      "balance_cascade"  → BalanceCascade（返回子集列表）
      "easy_ensemble"    → EasyEnsemble（返回子集列表）

    示例
    ----
        sampler = sampler_factory("smote", sampling_strategy=0.1, random_state=42)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    """
    m = method.lower().replace("-", "_").replace(" ", "_")

    if m == "smote":
        return SMOTE(sampling_strategy, k_neighbors, random_state)
    elif m in ("borderline_smote", "borderlinesmote"):
        return BorderlineSMOTE(sampling_strategy, k_neighbors,
                               kwargs.get("m_neighbors", 10), random_state)
    elif m == "adasyn":
        return ADASYN(sampling_strategy, k_neighbors, random_state)
    elif m in ("random_under", "randomundersampler", "undersample"):
        return RandomUnderSampler(sampling_strategy, random_state)
    elif m in ("tomek", "tomeklinks"):
        return TomekLinks(random_state)
    elif m == "enn":
        return ENN(k_neighbors, kwargs.get("kind_sel", "majority"), random_state)
    elif m in ("smoteenn", "smote_enn"):
        return SMOTEENN(sampling_strategy, k_neighbors,
                        kwargs.get("enn_k", 3), random_state)
    elif m in ("smotetomek", "smote_tomek"):
        return SMOTETomek(sampling_strategy, k_neighbors, random_state)
    elif m in ("balance_cascade", "balancecascade"):
        return BalanceCascade(
            n_estimators=kwargs.get("n_estimators", 10),
            ratio=kwargs.get("ratio", 1.0),
            random_state=random_state,
        )
    elif m in ("easy_ensemble", "easyensemble", "ensemble"):
        return EasyEnsemble(
            n_estimators=kwargs.get("n_estimators", 10),
            ratio=kwargs.get("ratio", 1.0),
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"未知采样方法: '{method}'。\n"
            "支持: smote / borderline_smote / adasyn / random_under / "
            "tomek / enn / smoteenn / smotetomek / balance_cascade / easy_ensemble / ensemble"
        )


# ─────────────────────────────────────────────────────────────────────
# 诊断打印工具
# ─────────────────────────────────────────────────────────────────────

def print_sampling_summary(y_before, y_after, method_name: str):
    """打印采样前后的正样本数量/比例变化。"""
    n_before = len(y_before)
    n_after  = len(y_after) if not isinstance(y_after, list) else sum(len(yy) for _, yy in y_after)

    if isinstance(y_after, list):
        print(f"✅ {method_name} 完成：生成 {len(y_after)} 个子集")
        for i, (_, ys) in enumerate(y_after):
            ys_arr = ys.values if hasattr(ys, "values") else np.asarray(ys)
            print(f"   子集 {i:02d}: {len(ys_arr)} 样本 | 正样本率: {ys_arr.mean():.2%}")
        return

    yb = y_before.values if hasattr(y_before, "values") else np.asarray(y_before)
    ya = y_after.values  if hasattr(y_after,  "values") else np.asarray(y_after)

    print(f"✅ {method_name} 完成")
    print(f"   样本数  : {n_before:,} → {len(ya):,}  (Δ {len(ya)-n_before:+,})")
    print(f"   正样本率: {yb.mean():.2%} → {ya.mean():.2%}")
    print(f"   正样本数: {int(yb.sum())} → {int(ya.sum())}")
    print(f"   负样本数: {int((yb==0).sum())} → {int((ya==0).sum())}")
