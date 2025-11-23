import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

class Evaluator:
    def __init__(self):
        # 定义指标与其方向性
        # ascending=True 表示：指标越小，结果越正确 (不确定性指标)
        # ascending=False 表示：指标越大，结果越正确 (置信度指标)
        self.metric_config = {
            # Uncertainty Metrics (Lower is better for Correctness)
            'Grammar Entropy': {'ascending': True},
            'Perplexity': {'ascending': True},
            'KL Divergence': {'ascending': True},
            'NSUI': {'ascending': True},
            'Spectral Radius': {'ascending': True},
            'Rule Dist Kurtosis': {'ascending': True}, # 论文发现 Kurtosis 异常通常意味着错误
            
            # Confidence Metrics (Higher is better for Correctness)
            'Self Consistency Text': {'ascending': False},
            'Self Consistency SMT': {'ascending': False},
            'Ensemble_Prob': {'ascending': False}  # 集成模型的预测概率
        }

    def compute_ece(self, y_true, y_prob, n_bins=10):
        """
        计算 Expected Calibration Error (ECE)
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        
        # 计算每个 bin 的样本数权重
        bin_edges = np.linspace(0., 1., n_bins + 1)
        binids = np.digitize(y_prob, bin_edges) - 1
        
        bin_sums = np.bincount(binids, minlength=n_bins)
        total_samples = len(y_prob)
        
        ece = 0.0
        # calibration_curve 返回的只是非空 bin 的点，我们需要加权
        # 这里为了简单，使用手动循环计算加权平均
        for i in range(n_bins):
            idx = binids == i
            if np.sum(idx) > 0:
                acc = np.mean(y_true[idx])
                conf = np.mean(y_prob[idx])
                weight = np.sum(idx) / total_samples
                ece += weight * np.abs(acc - conf)
        return ece

    def compute_aurc(self, y_true, y_conf):
        """
        计算 Area Under Risk-Coverage Curve (AURC)。
        衡量选择性预测（拒答）的能力。
        
        y_conf: 置信度分数 (越高越确信)
        """
        # 1. 按照置信度从高到低排序
        sorted_indices = np.argsort(y_conf)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # 2. 计算覆盖率 (Coverage) 和 风险 (Risk/Error Rate)
        n = len(y_true)
        coverages = np.arange(1, n + 1) / n
        
        # 累积错误数
        cumulative_errors = np.cumsum(1 - y_true_sorted)
        # 当前覆盖率下的错误率 (Risk)
        risks = cumulative_errors / np.arange(1, n + 1)
        
        # 3. 计算曲线下面积 (黎曼和)
        # 论文通常计算 risk 对 coverage 的积分
        aurc = np.trapz(risks, coverages)
        return aurc

    def evaluate_metric(self, df, metric_name, target_col):
        """
        评估单个指标的性能 (对应 Table 3/4 的一行)
        """
        y_true = df[target_col].values
        raw_scores = df[metric_name].values
        
        # 处理 NaN
        mask = ~np.isnan(raw_scores)
        y_true = y_true[mask]
        raw_scores = raw_scores[mask]
        
        if len(y_true) == 0:
            return None

        config = self.metric_config.get(metric_name, {'ascending': True})
        
        # 转换为“置信度”分数 (用于 AUROC/AURC 计算)
        # 如果指标是 Entropy (Ascending=True)，则分数取反，变成 -Entropy (越大越好)
        if config['ascending']:
            y_score = -raw_scores
            # 为了 ECE/Brier 计算，我们需要将 Raw Score 归一化到 [0,1] 概率
            # 简单的 Min-Max 归一化 (仅用于校准类指标估算，实际论文可能用了 Isotonic Regression)
            # 这里为了展示，仅做线性缩放
            y_prob = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)
        else:
            y_score = raw_scores
            if metric_name == 'Ensemble_Prob':
                y_prob = raw_scores
            else:
                y_prob = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-9)

        # 1. AUROC
        try:
            auroc = roc_auc_score(y_true, y_score)
        except ValueError:
            auroc = 0.5

        # 2. ECE
        ece = self.compute_ece(y_true, y_prob)

        # 3. Brier Score
        brier = brier_score_loss(y_true, y_prob)

        # 4. AURC
        aurc = self.compute_aurc(y_true, y_score)

        return {
            "Metric": metric_name,
            "AUROC": auroc,
            "ECE": ece,
            "Brier": brier,
            "AURC": aurc
        }

    def plot_roc_curves(self, df, metrics, target_col, title="ROC Curves"):
        """
        绘制 ROC 曲线对比图
        """
        plt.figure(figsize=(10, 8))
        
        y_true = df[target_col].values
        
        for metric in metrics:
            if metric not in df.columns: continue
            
            raw_scores = df[metric].values
            # 处理 NaN
            mask = ~np.isnan(raw_scores)
            y_masked = y_true[mask]
            scores_masked = raw_scores[mask]
            
            # 统一方向
            config = self.metric_config.get(metric, {'ascending': True})
            if config['ascending']:
                scores_masked = -scores_masked
                
            fpr, tpr, _ = roc_curve(y_masked, scores_masked)
            auc = roc_auc_score(y_masked, scores_masked)
            
            plt.plot(fpr, tpr, label=f'{metric} (AUC={auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(f"/data/newmodel/uncertainty/formal_uncertainty/output/generations/{title.replace(' ', '_')}.png")
        print(f"Plot saved to outputs/{title.replace(' ', '_')}.png")
        # plt.show()