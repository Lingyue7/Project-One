"""仅依赖 NumPy/Pandas/scikit-learn 的类别不平衡采样方法。

实现原则：
1. 距离型方法只在传入的训练集上拟合 RobustScaler；
2. 类别列用 one-hot 参与近邻距离，但生成时直接选择合法类别，不做线性插值；
3. SMOTE+ENN/Tomek 按标准组合顺序执行；
4. Balance Cascade 每轮训练分类器并移除已正确识别的多数类。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
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
        return [X.columns[int(v)] for v in values]
    missing = sorted(set(values) - set(X.columns))
    if missing:
        raise ValueError(f"类别特征不在 X 中: {missing}")
    return values


class _MetricSpace:
    """训练集内拟合的稳健缩放 + one-hot 距离空间。"""

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

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.cont:
            parts.append(self.scaler.transform(X[self.cont]))
        if self.cat:
            parts.append(self.encoder.transform(X[self.cat]))
        return np.hstack(parts).astype(float)

    def numeric_scaled(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X[self.cont]) if self.cont else np.empty((len(X), 0))

    def make_rows(self, numeric_scaled: np.ndarray, categorical_rows: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=np.arange(len(numeric_scaled)))
        if self.cont:
            out[self.cont] = self.scaler.inverse_transform(numeric_scaled)
        for c in self.cat:
            out[c] = categorical_rows[c].to_numpy()
        return out[self.columns]


def _classes(y: pd.Series):
    counts = y.value_counts()
    if len(counts) != 2:
        raise ValueError("当前 float sampling_strategy 仅支持二分类。")
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
        rng = np.random.RandomState(self.random_state)
        min_idx = np.flatnonzero(ys.to_numpy() == minority)
        maj_idx = rng.choice(np.flatnonzero(ys.to_numpy() == majority), target, replace=False)
        idx = np.r_[min_idx, maj_idx]; rng.shuffle(idx)
        return _restore(Xf.iloc[idx], ys.iloc[idx], wf)


class MixedSMOTE:
    """SMOTE/Borderline-SMOTE/ADASYN 的混合特征安全实现。"""

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
        Z = metric.transform(Xf); Zmin = Z[min_idx]
        k = min(self.k_neighbors, len(min_idx) - 1)
        nn_min = NearestNeighbors(n_neighbors=k + 1).fit(Zmin).kneighbors(return_distance=False)[:, 1:]
        candidates = np.arange(len(min_idx))
        weights = np.ones(len(min_idx), dtype=float)

        if self.variant in ("borderline", "adasyn"):
            m = min(self.m_neighbors if self.variant == "borderline" else self.k_neighbors,
                    len(Xf) - 1)
            nn_all = NearestNeighbors(n_neighbors=m + 1).fit(Z).kneighbors(Zmin, return_distance=False)[:, 1:]
            majority_ratio = (ys.to_numpy()[nn_all] == majority).mean(axis=1)
            if self.variant == "borderline":
                danger = (majority_ratio >= 0.5) & (majority_ratio < 1.0)
                if danger.any():
                    candidates = np.flatnonzero(danger)
                    weights = np.ones(len(candidates))
            else:
                if majority_ratio.sum() > 0:
                    weights = majority_ratio

        weights = weights / weights.sum()
        rng = np.random.RandomState(self.random_state)
        chosen = rng.choice(candidates, n_new, replace=True, p=weights)
        neighbor_local = np.array([rng.choice(nn_min[i]) for i in chosen])
        lam = rng.uniform(size=(n_new, 1))
        num_min = metric.numeric_scaled(Xf.iloc[min_idx])
        num_syn = num_min[chosen] + lam * (num_min[neighbor_local] - num_min[chosen])

        # SMOTENC 思路：类别直接从种子或少数类近邻选择，永不对编码做小数插值。
        cat_syn = pd.DataFrame(index=np.arange(n_new))
        for c in metric.cat:
            base_values = Xf.iloc[min_idx[chosen]][c].to_numpy()
            neighbor_values = Xf.iloc[min_idx[neighbor_local]][c].to_numpy()
            cat_syn[c] = np.where(rng.uniform(size=n_new) < 0.5, base_values, neighbor_values)
        Xsyn = metric.make_rows(num_syn, cat_syn)
        Xr = pd.concat([Xf, Xsyn], ignore_index=True)
        yr = pd.concat([ys, pd.Series([minority] * n_new)], ignore_index=True)
        return _restore(Xr, yr, wf)


class TomekLinks:
    def __init__(self, sampling_strategy="auto", categorical_features=None, **_):
        self.sampling_strategy, self.categorical_features = sampling_strategy, categorical_features

    def fit_resample(self, X, y):
        Xf, wf = _frame(X); ys = _series(y); minority, majority, _, _ = _classes(ys)
        Z = _MetricSpace(Xf, self.categorical_features).transform(Xf)
        nn = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(return_distance=False)[:, 1]
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
        Z = _MetricSpace(Xf, self.categorical_features).transform(Xf)
        k = min(self.n_neighbors, len(ys) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(return_distance=False)[:, 1:]
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
        rng = np.random.RandomState(self.random_state)
        return [RandomUnderSampler(1.0 / self.ratio, int(rng.randint(2**31 - 1))).fit_resample(X, y)
                for _ in range(self.n_estimators)]


class BalanceCascade:
    def __init__(self, n_estimators=10, ratio=1.0, random_state=None, estimator=None):
        self.n_estimators, self.ratio, self.random_state = n_estimators, ratio, random_state
        self.estimator = estimator or HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.08, random_state=random_state
        )

    def fit_resample(self, X, y) -> List[Tuple]:
        Xf, wf = _frame(X); ys = _series(y); minority, majority, _, _ = _classes(ys)
        min_idx = np.flatnonzero(ys.to_numpy() == minority)
        remaining = np.flatnonzero(ys.to_numpy() == majority)
        rng = np.random.RandomState(self.random_state); subsets = []
        for _ in range(self.n_estimators):
            if len(remaining) == 0:
                break
            take = min(int(round(len(min_idx) * self.ratio)), len(remaining))
            maj = rng.choice(remaining, take, replace=False)
            idx = np.r_[min_idx, maj]; rng.shuffle(idx)
            Xsub, ysub = Xf.iloc[idx].reset_index(drop=True), ys.iloc[idx].reset_index(drop=True)
            subsets.append(_restore(Xsub, ysub, wf))
            model = clone(self.estimator).fit(Xsub, ysub)
            remaining = remaining[model.predict(Xf.iloc[remaining]) != majority]
        return subsets


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
    print(f"✅ {method_name}: {len(yb):,} → {len(ya):,}; 正样本率 {yb.mean():.2%} → {ya.mean():.2%}")
