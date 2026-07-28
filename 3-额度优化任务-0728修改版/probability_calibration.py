#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probability_calibration.py
==========================
概率校准模块：基于等距回归（Isotonic Regression）对LightGBM预测概率进行校准

方法原理：
- 使用PAVA (Pool Adjacent Violators Algorithm) 方法
- 学习一个单调非递减函数将原始预测概率映射到校准概率
- 确保校准过程只压缩或拉伸概率分布，不改变样本相对排序

评估指标：
1. Brier Score：预测概率与真实标签的平方误差
2. ECE (Expected Calibration Error)：预期校准误差
3. 校准曲线：可视化预测概率与真实概率的关系
4. 业务指标：平均单位利润和总体利润
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class ProbabilityCalibrator:
    """概率校准器"""

    def __init__(self, method='isotonic'):
        """
        初始化校准器

        Parameters:
        -----------
        method : str, default='isotonic'
            校准方法，支持 'isotonic' (等距回归) 或 'platt' (Platt缩放)
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False

    def fit(self, y_true, y_pred):
        """
        在校准集上拟合校准函数

        Parameters:
        -----------
        y_true : array-like
            真实标签 (0 或 1)
        y_pred : array-like
            原始预测概率

        Returns:
        --------
        self
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true和y_pred长度不一致: {len(y_true)} vs {len(y_pred)}")

        if len(np.unique(y_true)) < 2:
            warnings.warn("校准集中只有一个类别，无法进行校准")
            self.calibrator = None
            self.is_fitted = False
            return self

        # 使用等距回归进行校准
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred, y_true)
        elif self.method in ('platt', 'sigmoid'):
            from sklearn.linear_model import LogisticRegression
            eps = 1e-6
            p = np.clip(y_pred, eps, 1.0 - eps)
            x = np.log(p / (1.0 - p)).reshape(-1, 1)
            # Large C approximates unregularized Platt scaling and remains
            # compatible with older sklearn versions that lack penalty='none'.
            self.calibrator = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000)
            self.calibrator.fit(x, y_true)
        else:
            raise ValueError(f"不支持的校准方法: {self.method}")

        self.is_fitted = True
        return self

    def predict(self, y_pred):
        """
        对新样本进行概率校准

        Parameters:
        -----------
        y_pred : array-like
            原始预测概率

        Returns:
        --------
        y_calibrated : np.ndarray
            校准后的概率
        """
        if not self.is_fitted or self.calibrator is None:
            warnings.warn("校准器未拟合，返回原始概率")
            return np.clip(y_pred, 0.0, 1.0)

        y_pred = np.asarray(y_pred).ravel()

        if self.method == 'isotonic':
            y_calibrated = self.calibrator.predict(y_pred)
        elif self.method in ('platt', 'sigmoid'):
            eps = 1e-6
            p = np.clip(y_pred, eps, 1.0 - eps)
            x = np.log(p / (1.0 - p)).reshape(-1, 1)
            y_calibrated = self.calibrator.predict_proba(x)[:, 1]

        # 确保概率在[0,1]范围内
        return np.clip(y_calibrated, 0.0, 1.0)


def calculate_ece(y_true, y_pred, n_bins=10):
    """
    计算预期校准误差 (Expected Calibration Error)

    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测概率
    n_bins : int
        分桶数量

    Returns:
    --------
    ece : float
        预期校准误差
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # 等频分桶
    bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-8  # 确保最大值被包含

    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i+1])
        if mask.sum() == 0:
            continue

        bin_size = mask.sum()
        bin_pred_mean = y_pred[mask].mean()
        bin_true_mean = y_true[mask].mean()

        ece += bin_size / len(y_true) * np.abs(bin_pred_mean - bin_true_mean)

    return ece


def plot_calibration_curve(y_true, y_pred_before, y_pred_after,
                          n_bins=10, save_path=None):
    """
    绘制校准曲线

    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred_before : array-like
        校准前预测概率
    y_pred_after : array-like
        校准后预测概率
    n_bins : int
        分桶数量
    save_path : str, optional
        保存路径
    """
    import matplotlib
    matplotlib.use('Agg')

    y_true = np.asarray(y_true).ravel()
    y_pred_before = np.asarray(y_pred_before).ravel()
    y_pred_after = np.asarray(y_pred_after).ravel()

    def get_calibration_curve(y_true, y_pred, n_bins):
        """计算校准曲线数据点"""
        bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-8

        pred_means = []
        true_means = []

        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i+1])
            if mask.sum() > 0:
                pred_means.append(y_pred[mask].mean())
                true_means.append(y_true[mask].mean())

        return np.array(pred_means), np.array(true_means)

    # 计算校准前后的曲线
    pred_before, true_before = get_calibration_curve(y_true, y_pred_before, n_bins)
    pred_after, true_after = get_calibration_curve(y_true, y_pred_after, n_bins)

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))

    # 对角线（完美校准）
    ax.plot([0, 1], [0, 1], 'k--', label='完美校准', linewidth=2)

    # 校准前
    ax.plot(pred_before, true_before, 'o-', label='校准前',
            linewidth=2, markersize=8, color='#e74c3c')

    # 校准后
    ax.plot(pred_after, true_after, 's-', label='校准后',
            linewidth=2, markersize=8, color='#27ae60')

    ax.set_xlabel('平均预测概率', fontsize=12)
    ax.set_ylabel('真实正样本率', fontsize=12)
    ax.set_title('概率校准曲线', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  校准曲线已保存: {save_path}")

    plt.close()


def evaluate_calibration(y_true, y_pred_before, y_pred_after,
                         task_name='', save_dir=None):
    """
    综合评估概率校准效果

    Parameters:
    -----------
    y_true : array-like
        真实标签
    y_pred_before : array-like
        校准前预测概率
    y_pred_after : array-like
        校准后预测概率
    task_name : str
        任务名称（如'支用概率'或'违约概率'）
    save_dir : str, optional
        报告保存目录

    Returns:
    --------
    metrics : dict
        评估指标字典
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_before = np.asarray(y_pred_before).ravel()
    y_pred_after = np.asarray(y_pred_after).ravel()

    # 计算Brier Score
    brier_before = brier_score_loss(y_true, y_pred_before)
    brier_after = brier_score_loss(y_true, y_pred_after)
    brier_improve = (brier_before - brier_after) / brier_before * 100

    # 计算ECE
    ece_before = calculate_ece(y_true, y_pred_before, n_bins=10)
    ece_after = calculate_ece(y_true, y_pred_after, n_bins=10)
    ece_improve = (ece_before - ece_after) / ece_before * 100 if ece_before > 0 else 0.0

    # 计算平均预测概率 vs 真实正样本率
    mean_pred_before = y_pred_before.mean()
    mean_pred_after = y_pred_after.mean()
    mean_true = y_true.mean()

    metrics = {
        'task_name': task_name,
        'brier_score_before': brier_before,
        'brier_score_after': brier_after,
        'brier_improvement': brier_improve,
        'ece_before': ece_before,
        'ece_after': ece_after,
        'ece_improvement': ece_improve,
        'mean_pred_before': mean_pred_before,
        'mean_pred_after': mean_pred_after,
        'mean_true': mean_true,
    }

    # 打印评估报告
    print(f"\n{'='*60}")
    print(f"  {task_name}概率校准效果评估")
    print(f"{'='*60}")
    print(f"  【Brier Score】")
    print(f"    校准前: {brier_before:.6f}")
    print(f"    校准后: {brier_after:.6f}")
    print(f"    改善率: {brier_improve:+.2f}%")
    print(f"\n  【ECE (预期校准误差)】")
    print(f"    校准前: {ece_before:.6f}")
    print(f"    校准后: {ece_after:.6f}")
    print(f"    改善率: {ece_improve:+.2f}%")
    print(f"\n  【平均概率 vs 真实率】")
    print(f"    真实正样本率:   {mean_true:.4f} ({mean_true*100:.2f}%)")
    print(f"    校准前平均概率: {mean_pred_before:.4f} ({mean_pred_before*100:.2f}%)")
    print(f"    校准后平均概率: {mean_pred_after:.4f} ({mean_pred_after*100:.2f}%)")
    print(f"    校准前偏差: {(mean_pred_before - mean_true)*100:+.2f}pp")
    print(f"    校准后偏差: {(mean_pred_after - mean_true)*100:+.2f}pp")
    print(f"{'='*60}\n")

    # 绘制校准曲线
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f'calibration_curve_{task_name}.png')
        plot_calibration_curve(y_true, y_pred_before, y_pred_after,
                             n_bins=10, save_path=plot_path)

    return metrics


def calibrate_probabilities(y_true_train, y_pred_train,
                            y_true_cal, y_pred_cal,
                            y_true_test, y_pred_test,
                            task_name='', method='isotonic', save_dir=None):
    """
    完整的概率校准流程

    Parameters:
    -----------
    y_true_train, y_pred_train : array-like
        训练集的真实标签和预测概率（用于报告）
    y_true_cal, y_pred_cal : array-like
        校准集的真实标签和预测概率（用于拟合校准函数）
    y_true_test, y_pred_test : array-like
        测试集的真实标签和预测概率（用于评估）
    task_name : str
        任务名称
    method : str
        校准方法
    save_dir : str, optional
        保存目录

    Returns:
    --------
    calibrator : ProbabilityCalibrator
        拟合后的校准器
    y_pred_test_calibrated : np.ndarray
        测试集校准后的概率
    metrics : dict
        评估指标
    """
    print(f"\n{'='*60}")
    print(f"  {task_name}概率校准流程")
    print(f"{'='*60}")
    print(f"  训练集样本数: {len(y_true_train):,}  (正例率: {y_true_train.mean():.2%})")
    print(f"  校准集样本数: {len(y_true_cal):,}  (正例率: {y_true_cal.mean():.2%})")
    print(f"  测试集样本数: {len(y_true_test):,}  (正例率: {y_true_test.mean():.2%})")
    print(f"  校准方法: {method}")

    # 步骤1：在校准集上拟合校准函数
    print(f"\n  步骤1: 在校准集上拟合校准函数...")
    calibrator = ProbabilityCalibrator(method=method)
    calibrator.fit(y_true_cal, y_pred_cal)

    # 步骤2：对测试集进行校准
    print(f"  步骤2: 对测试集进行概率校准...")
    y_pred_test_calibrated = calibrator.predict(y_pred_test)

    # 步骤3：评估校准效果
    print(f"  步骤3: 评估校准效果...")
    metrics = evaluate_calibration(
        y_true_test, y_pred_test, y_pred_test_calibrated,
        task_name=task_name, save_dir=save_dir
    )

    return calibrator, y_pred_test_calibrated, metrics
