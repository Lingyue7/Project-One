"""仅依赖 NumPy/Pandas/scikit-learn 的类别不平衡采样方法。

实现原则：
1. 距离型方法只在传入的训练集上拟合 RobustScaler；
2. SMOTENC 按连续特征尺度加权 one-hot 距离，合成类别取 k 个近邻的众数；
3. SMOTE+ENN/Tomek 按标准组合顺序执行；
4. Balance Cascade 按目标假阳性率逐轮保留难分多数类，并由实际子模型驱动级联。
"""


from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, RobustScaler


ArrayLike = Union[np.ndarray, pd.DataFrame]


def _frame(X: ArrayLike) -> Tuple[pd.DataFrame, bool]:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True).copy(), True
    a = np.asarray(X)
    return pd.DataFrame(a, columns=[f"x{i}" for i in range(a.shape[1])]), False


def _series(y) -> pd.Series:
    return pd.Series(np.asarray(y)).reset_index(drop=True)


def _restore(X: pd.DataFrame, y, was_frame: bool):
    X = X.reset_index(drop=True)
    ys = _series(y)
    return (X, ys) if was_frame else (X.to_numpy(), ys.to_numpy())


def _cat_names(X: pd.DataFrame, categorical_features: Optional[Sequence]) -> List[str]:
    values = list(categorical_features or [])
    if not values:
        return []
    if all(isinstance(v, (bool, np.bool_)) for v in values):
        if len(values) != X.shape[1]:
            raise ValueError("类别特征布尔掩码长度必须等于特征数。")
        return [c for c, flag in zip(X.columns, values) if flag]
    if all(isinstance(v, (int, np.integer)) for v in values):
        indices = [int(v) for v in values]
        invalid = [v for v in indices if v < 0 or v >= X.shape[1]]
        if invalid:
            raise ValueError(f"类别特征索引越界: {invalid}")
        names = [X.columns[v] for v in indices]
        if len(names) != len(set(names)):
            raise ValueError("类别特征不能重复指定。")
        return names
    missing = sorted(set(values) - set(X.columns))
    if missing:
        raise ValueError(f"类别特征不在 X 中: {missing}")
    if len(values) != len(set(values)):
        raise ValueError("类别特征不能重复指定。")
    return values


class _MetricSpace:
    """训练集内拟合的稳健缩放 + 加权 one-hot 距离空间。"""

    def __init__(self, X: pd.DataFrame, categorical_features=None):
        self.columns = list(X.columns)
        self.cat = _cat_names(X, categorical_features)
        self.cont = [c for c in self.columns if c not in self.cat]
        self.scaler = RobustScaler().fit(X[self.cont]) if self.cont else None
        if self.cat:
            try:
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[self.cat])
            except TypeError:  # scikit-learn < 1.2
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(X[self.cat])
        else:
            self.encoder = None

    def transform(self, X: pd.DataFrame, categorical_scale: float = 1.0) -> np.ndarray:
        parts = []
        if self.cont:
            parts.append(self.scaler.transform(X[self.cont]))
        if self.cat:
            parts.append(self.encoder.transform(X[self.cat]) * float(categorical_scale))
        if not parts:
            raise ValueError("X 至少需要一个特征。")
        return np.hstack(parts).astype(float)

    def numeric_scaled(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X[self.cont]) if self.cont else np.empty((len(X), 0))

    def categorical_scale(self, X_reference: pd.DataFrame) -> float:
        """返回 SMOTENC 的 one-hot 非零值，使一次类别不匹配等于连续特征典型标准差。"""
        if not self.cat:
            return 1.0
        if not self.cont:
            # 纯类别数据不是标准 SMOTENC 的适用范围；保留普通 one-hot 欧氏距离。
            return 1.0
        numeric = self.numeric_scaled(X_reference)
        median_std = float(np.median(np.std(numeric, axis=0, ddof=0)))
        if not np.isfinite(median_std) or median_std <= np.finfo(float).eps:
            # 连续特征全为常数时不能把类别距离缩成 0；令一次类别不匹配距离为 1。
            return 1.0 / np.sqrt(2.0)
        # 与 SMOTENC 一致：一个类别不匹配会在 one-hot 中产生两个非零差值。
        return median_std / np.sqrt(2.0)

    def make_rows(self, numeric_scaled: np.ndarray, categorical_rows: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=np.arange(len(numeric_scaled)))
        if self.cont:
            out[self.cont] = self.scaler.inverse_transform(numeric_scaled)
        for c in self.cat:
            out[c] = categorical_rows[c].to_numpy()
        return out[self.columns]


def _nearest_neighbors_excluding_self(
    fitted_rows: np.ndarray,
    query_rows: np.ndarray,
    self_indices: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """逐行按真实索引剔除自身，避免重复坐标时误删另一个零距离样本。"""
    if n_neighbors < 1 or n_neighbors >= len(fitted_rows):
        raise ValueError("非自身近邻数必须在 [1, 样本数-1] 内。")
    raw = NearestNeighbors(
        n_neighbors=min(n_neighbors + 1, len(fitted_rows))
    ).fit(fitted_rows).kneighbors(query_rows, return_distance=False)
    result = []
    for row, self_index in zip(raw, np.asarray(self_indices, dtype=int)):
        non_self = row[row != self_index][:n_neighbors]
        if len(non_self) != n_neighbors:
            raise RuntimeError("未能取得足够的非自身近邻。")
        result.append(non_self)
    return np.vstack(result)


def _classes(y: pd.Series):
    counts = y.value_counts()
    if len(counts) != 2:
        raise ValueError("当前 float sampling_strategy 仅支持二分类。")
    if counts.iloc[0] == counts.iloc[1]:
        # 平衡数据没有天然多数/少数类，但两个返回标签仍必须不同。
        return counts.index[1], counts.index[0], int(counts.iloc[1]), int(counts.iloc[0])
    return counts.idxmin(), counts.idxmax(), int(counts.min()), int(counts.max())


def _over_target(strategy, n_min, n_maj):
    ratio = 1.0 if strategy == "auto" else float(strategy)
    if not 0 < ratio <= 1:
        raise ValueError("过采样 sampling_strategy 必须在 (0, 1] 或为 'auto'。")
    target = int(round(n_maj * ratio))
    if target < n_min:
        raise ValueError("目标比例小于当前少数/多数比例，过采样无法减少少数类。")
    return target


def _under_target(strategy, n_min, n_maj):
    ratio = 1.0 if strategy == "auto" else float(strategy)
    if not 0 < ratio <= 1:
        raise ValueError("欠采样 sampling_strategy 必须在 (0, 1] 或为 'auto'。")
    target = int(round(n_min / ratio))
    if target > n_maj:
        raise ValueError("目标比例小于当前少数/多数比例，欠采样无法增加多数类。")
    return target


class RandomOverSampler:
    def __init__(self, sampling_strategy=0.1, random_state=None, **_):
        self.sampling_strategy, self.random_state = sampling_strategy, random_state

    def fit_resample(self, X, y):
        Xf, wf = _frame(X); ys = _series(y)
        minority, _, nmin, nmaj = _classes(ys)
        add = _over_target(self.sampling_strategy, nmin, nmaj) - nmin
        if add == 0:
            return _restore(Xf, ys, wf)
        rng = np.random.RandomState(self.random_state)
        extra = rng.choice(np.flatnonzero(ys.to_numpy() == minority), add, replace=True)
        idx = np.r_[np.arange(len(ys)), extra]; rng.shuffle(idx)
        return _restore(Xf.iloc[idx], ys.iloc[idx], wf)


class RandomUnderSampler:
    def __init__(self, sampling_strategy=0.1, random_state=None, **_):
        self.sampling_strategy, self.random_state = sampling_strategy, random_state

    def fit_resample(self, X, y):
        Xf, wf = _frame(X); ys = _series(y)
        minority, majority, nmin, nmaj = _classes(ys)
        target = _under_target(self.sampling_strategy, nmin, nmaj)
        if target == nmaj:
            return _restore(Xf, ys, wf)
        rng = np.random.RandomState(self.random_state)
        min_idx = np.flatnonzero(ys.to_numpy() == minority)
        maj_idx = rng.choice(np.flatnonzero(ys.to_numpy() == majority), target, replace=False)
        idx = np.r_[min_idx, maj_idx]; rng.shuffle(idx)
        return _restore(Xf.iloc[idx], ys.iloc[idx], wf)


class MixedSMOTE:
    """SMOTENC，以及 Borderline-SMOTE/ADASYN 的混合特征扩展。"""

    def __init__(self, variant="smote", sampling_strategy=0.1, k_neighbors=5,
                 m_neighbors=10, random_state=None, categorical_features=None):
        self.variant, self.sampling_strategy = variant, sampling_strategy
        self.k_neighbors, self.m_neighbors = k_neighbors, m_neighbors
        self.random_state, self.categorical_features = random_state, categorical_features

    def fit_resample(self, X, y):
        Xf, wf = _frame(X); ys = _series(y)
        minority, majority, nmin, nmaj = _classes(ys)
        n_new = _over_target(self.sampling_strategy, nmin, nmaj) - nmin
        if n_new == 0:
            return _restore(Xf, ys, wf)
        min_idx = np.flatnonzero(ys.to_numpy() == minority)
        if len(min_idx) < 2:
            return RandomOverSampler(self.sampling_strategy, self.random_state).fit_resample(X, y)

        metric = _MetricSpace(Xf, self.categorical_features)
        Xmin = Xf.iloc[min_idx]
        categorical_scale = metric.categorical_scale(Xmin)
        Z = metric.transform(Xf, categorical_scale); Zmin = Z[min_idx]
        k = min(self.k_neighbors, len(min_idx) - 1)
        nn_min = _nearest_neighbors_excluding_self(
            Zmin, Zmin, np.arange(len(Zmin)), k
        )
        candidates = np.arange(len(min_idx))
        weights = np.ones(len(min_idx), dtype=float)

        if self.variant in ("borderline", "adasyn"):
            m = min(self.m_neighbors if self.variant == "borderline" else self.k_neighbors,
                    len(Xf) - 1)
            nn_all = _nearest_neighbors_excluding_self(Z, Zmin, min_idx, m)
            majority_ratio = (ys.to_numpy()[nn_all] == majority).mean(axis=1)
            if self.variant == "borderline":
                danger = (majority_ratio >= 0.5) & (majority_ratio < 1.0)
                if not danger.any():
                    return _restore(Xf, ys, wf)
                candidates = np.flatnonzero(danger)
                weights = np.ones(len(candidates))
            else:
                if majority_ratio.sum() == 0:
                    raise ValueError("ADASYN 未发现含多数类近邻的少数类样本，无法进行自适应采样。")
                weights = majority_ratio

        weights = weights / weights.sum()
        rng = np.random.RandomState(self.random_state)
        chosen = rng.choice(candidates, n_new, replace=True, p=weights)
        neighbor_local = np.array([rng.choice(nn_min[i]) for i in chosen])
        lam = rng.uniform(size=(n_new, 1))
        num_min = metric.numeric_scaled(Xmin)
        num_syn = num_min[chosen] + lam * (num_min[neighbor_local] - num_min[chosen])

        # SMOTENC：每个类别特征取所选种子的 k 个少数类近邻的众数；并列时随机打破。
        cat_syn = pd.DataFrame(index=np.arange(n_new))
        for c in metric.cat:
            neighbor_values = Xmin[c].to_numpy()[nn_min[chosen]]
            modes = []
            for row in neighbor_values:
                counts = pd.Series(row).value_counts(dropna=False)
                tied_modes = counts.index[counts == counts.iloc[0]].to_numpy()
                modes.append(rng.choice(tied_modes))
            cat_syn[c] = modes
        Xsyn = metric.make_rows(num_syn, cat_syn)
        Xr = pd.concat([Xf, Xsyn], ignore_index=True)
        yr = pd.concat([ys, pd.Series([minority] * n_new)], ignore_index=True)
        return _restore(Xr, yr, wf)


class TomekLinks:
    def __init__(self, sampling_strategy="auto", categorical_features=None, **_):
        self.sampling_strategy, self.categorical_features = sampling_strategy, categorical_features

    def fit_resample(self, X, y):
        Xf, wf = _frame(X); ys = _series(y); minority, majority, _, _ = _classes(ys)
        metric = _MetricSpace(Xf, self.categorical_features)
        Z = metric.transform(Xf, metric.categorical_scale(Xf))
        nn = _nearest_neighbors_excluding_self(
            Z, Z, np.arange(len(Z)), 1
        )[:, 0]
        remove = set()
        for i, j in enumerate(nn):
            if nn[j] == i and ys.iloc[i] != ys.iloc[j]:
                if self.sampling_strategy == "all":
                    remove.update((i, j))
                else:
                    remove.add(i if ys.iloc[i] == majority else j)
        keep = np.array([i for i in range(len(ys)) if i not in remove], dtype=int)
        self.sample_indices_ = keep
        return _restore(Xf.iloc[keep], ys.iloc[keep], wf)


class EditedNearestNeighbours:
    def __init__(self, sampling_strategy="auto", n_neighbors=3, kind_sel="all",
                 categorical_features=None, **_):
        if kind_sel not in ("all", "mode"):
            raise ValueError("ENN kind_sel 仅支持 'all' 或 'mode'。")
        self.sampling_strategy, self.n_neighbors = sampling_strategy, n_neighbors
        self.kind_sel, self.categorical_features = kind_sel, categorical_features

    def fit_resample(self, X, y):
        Xf, wf = _frame(X); ys = _series(y); minority, majority, _, _ = _classes(ys)
        metric = _MetricSpace(Xf, self.categorical_features)
        Z = metric.transform(Xf, metric.categorical_scale(Xf))
        k = min(self.n_neighbors, len(ys) - 1)
        nn = _nearest_neighbors_excluding_self(
            Z, Z, np.arange(len(Z)), k
        )
        labels = ys.to_numpy(); neigh = labels[nn]
        if self.kind_sel == "all":
            acceptable = (neigh == labels[:, None]).all(axis=1)
        else:
            acceptable = np.array([(pd.Series(row).mode().iloc[0] == labels[i]) for i, row in enumerate(neigh)])
        targeted = np.ones(len(ys), dtype=bool) if self.sampling_strategy == "all" else (labels == majority)
        keep = np.flatnonzero(~targeted | acceptable)
        self.sample_indices_ = keep
        return _restore(Xf.iloc[keep], ys.iloc[keep], wf)


class CombinedSampler:
    def __init__(self, cleaner, sampling_strategy=0.1, k_neighbors=5,
                 random_state=None, categorical_features=None):
        self.cleaner, self.sampling_strategy = cleaner, sampling_strategy
        self.k_neighbors, self.random_state = k_neighbors, random_state
        self.categorical_features = categorical_features

    def fit_resample(self, X, y):
        Xo, yo = MixedSMOTE("smote", self.sampling_strategy, self.k_neighbors,
                            random_state=self.random_state,
                            categorical_features=self.categorical_features).fit_resample(X, y)
        return self.cleaner.fit_resample(Xo, yo)


class EasyEnsemble:
    def __init__(self, n_estimators=10, ratio=1.0, random_state=None):
        self.n_estimators, self.ratio, self.random_state = n_estimators, ratio, random_state

    def fit_resample(self, X, y) -> List[Tuple]:
        if int(self.n_estimators) < 1:
            raise ValueError("Easy Ensemble 的 n_estimators 必须 >= 1。")
        if float(self.ratio) < 1:
            raise ValueError("Easy Ensemble 的多数/少数比例 ratio 必须 >= 1。")
        rng = np.random.RandomState(self.random_state)
        return [RandomUnderSampler(1.0 / self.ratio, int(rng.randint(2**31 - 1))).fit_resample(X, y)
                for _ in range(self.n_estimators)]


class BalanceCascade:
    """遵循论文核心采样/阈值流程的二分类 Balance Cascade。

    ``initialize`` / ``next_subset`` / ``update`` 允许调用方用真正要集成的模型逐轮驱动
    采样。``fit_resample`` 保留便捷接口，并使用 ``estimator`` 生成同样的子集。
    ``ratio=1`` 时对应每轮多数类子集大小等于少数类大小；其他值是扩展用法。
    """

    def __init__(self, n_estimators=10, ratio=1.0, random_state=None, estimator=None):
        self.n_estimators, self.ratio, self.random_state = n_estimators, ratio, random_state
        self.estimator = estimator if estimator is not None else GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.08, random_state=random_state
        )

    def initialize(self, X, y):
        if int(self.n_estimators) < 1:
            raise ValueError("Balance Cascade 的 n_estimators 必须 >= 1。")
        if float(self.ratio) <= 0:
            raise ValueError("Balance Cascade 的 ratio 必须 > 0。")

        self._X, self._was_frame = _frame(X)
        self._y = _series(y)
        minority, majority, n_min, n_maj = _classes(self._y)
        self.minority_class_, self.majority_class_ = minority, majority
        self._minority_indices = np.flatnonzero(self._y.to_numpy() == minority)
        self._remaining_indices = np.flatnonzero(self._y.to_numpy() == majority)
        self._target_majority_count = max(1, int(round(n_min * float(self.ratio))))
        self.effective_n_estimators_ = (
            1 if n_maj <= self._target_majority_count else int(self.n_estimators)
        )
        if self.effective_n_estimators_ == 1:
            self.target_fpr_ = 1.0
        else:
            self.target_fpr_ = (
                self._target_majority_count / n_maj
            ) ** (1.0 / (self.effective_n_estimators_ - 1))
        self._rng = np.random.RandomState(self.random_state)
        self._stage = 0
        self._awaiting_update = False
        self.subsets_: List[Tuple] = []
        self.estimators_: List = []
        self.thresholds_: List[float] = []
        self.remaining_majority_counts_ = [len(self._remaining_indices)]
        return self

    def has_next_subset(self) -> bool:
        return (
            self._stage < self.effective_n_estimators_
            and len(self._remaining_indices) > 0
        )

    def next_subset(self):
        if self._awaiting_update:
            raise RuntimeError("必须先用本轮模型调用 update，再取得下一子集。")
        if not self.has_next_subset():
            raise StopIteration("Balance Cascade 已完成。")
        take = min(self._target_majority_count, len(self._remaining_indices))
        majority_indices = self._rng.choice(
            self._remaining_indices, take, replace=False
        )
        indices = np.r_[self._minority_indices, majority_indices]
        self._rng.shuffle(indices)
        X_sub = self._X.iloc[indices].reset_index(drop=True)
        y_sub = self._y.iloc[indices].reset_index(drop=True)
        restored = _restore(X_sub, y_sub, self._was_frame)
        self.subsets_.append(restored)
        self._awaiting_update = True
        return restored

    def remaining_X(self):
        X_remaining = self._X.iloc[self._remaining_indices]
        return X_remaining if self._was_frame else X_remaining.to_numpy()

    def update(self, scores, score_label=1):
        """按本轮模型分数保留目标 FPR 比例的难分多数类。

        ``scores`` 必须对应 ``remaining_X()``；值越大越倾向 ``score_label``。
        对常见的二分类概率，传入正类概率并保持 ``score_label=1`` 即可。
        """
        if not self._awaiting_update:
            raise RuntimeError("当前没有等待 update 的 Balance Cascade 子集。")
        minority_scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(minority_scores) != len(self._remaining_indices):
            raise ValueError("scores 长度必须等于当前剩余多数类样本数。")
        if score_label != self.minority_class_:
            minority_scores = 1.0 - minority_scores

        # T 个子模型之间只有 T-1 次样本池转移；最后一轮模型训练完成后不再
        # 进行无后续用途的多数类淘汰。这也与 target_fpr_ 的 (T-1) 次方定义一致。
        has_following_stage = self._stage + 1 < self.effective_n_estimators_
        threshold = np.nan
        if has_following_stage:
            keep_count = min(
                len(self._remaining_indices),
                max(1, int(round(len(self._remaining_indices) * self.target_fpr_))),
            )
            order = np.argsort(minority_scores, kind="mergesort")[::-1]
            kept_local = order[:keep_count]
            threshold = float(minority_scores[kept_local[-1]])
            self._remaining_indices = self._remaining_indices[kept_local]
            self.thresholds_.append(threshold)
            self.remaining_majority_counts_.append(len(self._remaining_indices))
        self._stage += 1
        self._awaiting_update = False
        return {
            "stage": self._stage,
            "threshold": threshold,
            "target_fpr": self.target_fpr_,
            "remaining_majority": len(self._remaining_indices),
            "pruned_for_next_stage": has_following_stage,
        }

    def fit_resample(self, X, y) -> List[Tuple]:
        self.initialize(X, y)
        while self.has_next_subset():
            X_sub, y_sub = self.next_subset()
            model = clone(self.estimator).fit(X_sub, y_sub)
            self.estimators_.append(model)
            X_remaining = self.remaining_X()
            if hasattr(model, "predict_proba"):
                class_index = list(model.classes_).index(self.minority_class_)
                scores = model.predict_proba(X_remaining)[:, class_index]
                score_label = self.minority_class_
            else:
                scores = (model.predict(X_remaining) == self.minority_class_).astype(float)
                score_label = self.minority_class_
            self.update(scores, score_label=score_label)
        return self.subsets_


def sampler_factory(method: str, sampling_strategy: Union[float, str] = 0.1,
                    k_neighbors: int = 5, random_state: Optional[int] = None,
                    categorical_features=None, **kwargs):
    m = method.lower().replace("-", "_").replace(" ", "_")
    if m in ("random_over", "randomoversampler", "oversample"):
        return RandomOverSampler(sampling_strategy, random_state)
    if m == "smote":
        return MixedSMOTE("smote", sampling_strategy, k_neighbors,
                          random_state=random_state, categorical_features=categorical_features)
    if m in ("borderline_smote", "borderlinesmote"):
        return MixedSMOTE("borderline", sampling_strategy, k_neighbors,
                          kwargs.get("m_neighbors", 10), random_state, categorical_features)
    if m == "adasyn":
        return MixedSMOTE("adasyn", sampling_strategy, k_neighbors,
                          random_state=random_state, categorical_features=categorical_features)
    if m in ("random_under", "randomundersampler", "undersample"):
        return RandomUnderSampler(sampling_strategy, random_state)
    if m in ("tomek", "tomeklinks"):
        return TomekLinks("auto", categorical_features)
    if m == "enn":
        return EditedNearestNeighbours("auto", k_neighbors, kwargs.get("kind_sel", "all"),
                                       categorical_features)
    if m in ("smoteenn", "smote_enn"):
        cleaner = EditedNearestNeighbours("all", kwargs.get("enn_k", 3), "all", categorical_features)
        return CombinedSampler(cleaner, sampling_strategy, k_neighbors, random_state, categorical_features)
    if m in ("smotetomek", "smote_tomek"):
        cleaner = TomekLinks("all", categorical_features)
        return CombinedSampler(cleaner, sampling_strategy, k_neighbors, random_state, categorical_features)
    if m in ("balance_cascade", "balancecascade"):
        return BalanceCascade(kwargs.get("n_estimators", 10), kwargs.get("ratio", 1.0),
                              random_state, kwargs.get("estimator"))
    if m in ("easy_ensemble", "easyensemble", "ensemble"):
        return EasyEnsemble(kwargs.get("n_estimators", 10), kwargs.get("ratio", 1.0), random_state)
    raise ValueError("未知采样方法: " + method)


def print_sampling_summary(y_before, y_after, method_name: str):
    yb, ya = np.asarray(y_before), np.asarray(y_after)
    print(f"[完成] {method_name}: {len(yb):,} -> {len(ya):,}; 正样本率 {yb.mean():.2%} -> {ya.mean():.2%}")
