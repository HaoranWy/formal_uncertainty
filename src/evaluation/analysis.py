import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

class Evaluator:
    def __init__(self):
        # 定义指标方向性：ascending=True 表示值越小越正确(不确定性)，False 表示值越大越正确(置信度)
        self.metric_config = {
            'Grammar Entropy': {'ascending': True},
            'Perplexity': {'ascending': True},
            'KL Divergence': {'ascending': True},
            'NSUI': {'ascending': True},
            'Spectral Radius': {'ascending': True},
            'Rule Dist Kurtosis': {'ascending': True},
            'Self Consistency Text': {'ascending': False},
            'Self Consistency SMT': {'ascending': False},
            'Ensemble_Prob': {'ascending': False},
            'Ensemble ML': {'ascending': False},
            'Ensemble Average': {'ascending': False},
            'Ensemble Weighted': {'ascending': False},
            'Ensemble Simple': {'ascending': False}
        }

    def evaluate_metric(self, df, metric_name, target_col):
        """计算单个指标的统计数据"""
        if metric_name not in df.columns: return None
        
        y_true = df[target_col].values
        raw_scores = df[metric_name].values
        
        # 去除 NaN
        mask = ~np.isnan(raw_scores)
        y_true = y_true[mask]
        raw_scores = raw_scores[mask]
        
        if len(y_true) == 0: return None

        # 统一方向：将不确定性指标取反，变为置信度方向
        config = self.metric_config.get(metric_name, {'ascending': True})
        if config['ascending']:
            y_score = -raw_scores
        else:
            y_score = raw_scores

        # 计算 AUROC
        try:
            auroc = roc_auc_score(y_true, y_score)
        except ValueError:
            auroc = 0.5

        # 计算 AURC (Area Under Risk-Coverage)
        # 简单的风险覆盖率积分
        sorted_idx = np.argsort(y_score)[::-1] # 从高置信度到低
        y_sorted = y_true[sorted_idx]
        n = len(y_sorted)
        # 风险 = 累积错误数 / 当前覆盖样本数
        risk = np.cumsum(1 - y_sorted) / np.arange(1, n + 1)
        coverage = np.arange(1, n + 1) / n
        aurc = np.trapz(risk, coverage)

        # 计算 ECE (需归一化到 0-1 概率空间)
        y_prob = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)
        ece = self._compute_ece(y_true, y_prob)

        return {
            "Metric": metric_name,
            "AUROC": auroc,
            "ECE": ece,
            "AURC": aurc
        }

    def _compute_ece(self, y_true, y_prob, n_bins=10):
        """辅助函数：计算 ECE"""
        bin_edges = np.linspace(0., 1., n_bins + 1)
        binids = np.digitize(y_prob, bin_edges) - 1
        
        ece = 0.0
        total = len(y_true)
        for i in range(n_bins):
            idx = binids == i
            if np.sum(idx) > 0:
                acc = np.mean(y_true[idx])
                conf = np.mean(y_prob[idx])
                weight = np.sum(idx) / total
                ece += weight * np.abs(acc - conf)
        return ece

    def plot_roc_curves(self, df, metrics, target_col, title="ROC Curves"):
        """
        绘制 ROC 曲线并保存到本地
        """
        # 确保输出目录存在
        save_dir = "outputs/plots"
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        y_true = df[target_col].values
        
        # 颜色循环
        colors = plt.cm.get_cmap('tab10')
        
        for i, metric in enumerate(metrics):
            if metric not in df.columns: continue
            
            raw_scores = df[metric].values
            # 处理 NaN
            mask = ~np.isnan(raw_scores)
            y_masked = y_true[mask]
            scores_masked = raw_scores[mask]
            
            if len(y_masked) < 2: continue

            # 统一方向
            config = self.metric_config.get(metric, {'ascending': True})
            if config['ascending']:
                scores_masked = -scores_masked
                
            fpr, tpr, _ = roc_curve(y_masked, scores_masked)
            auc = roc_auc_score(y_masked, scores_masked)
            
            plt.plot(fpr, tpr, lw=2, label=f'{metric} (AUC={auc:.4f})')

        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        # 保存图片
        filename = f"{save_dir}/{title.replace(' ', '_').replace('/', '-')}.png"
        plt.savefig(filename, dpi=300)
        print(f"🖼️  ROC Plot saved to: {filename}")
        plt.close()