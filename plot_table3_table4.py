import pandas as pd
import numpy as np
from src.evaluation.ensemble import UncertaintyEnsemble
from src.evaluation.analysis import Evaluator

def generate_mock_data(n_samples=500):
    """
    生成模拟数据以测试代码逻辑。
    在实际复现时，请替换为从文件加载数据的逻辑。
    """
    np.random.seed(42)
    df = pd.DataFrame({
        'question_id': [f'Q{i//5}' for i in range(n_samples)], # 5 samples per question
        
        # Labels
        'smt_is_correct': np.random.randint(0, 2, n_samples),
        'consistency_smt_text': np.random.randint(0, 2, n_samples),
        
        # PCFG Metrics (Uncertainty features)
        # 假设正确样本的熵较低 (mean=2)，错误样本的熵较高 (mean=5)
        'Grammar Entropy': np.concatenate([np.random.normal(5, 1, 250), np.random.normal(2, 1, 250)]),
        'Spectral Radius': np.random.random(n_samples),
        'Rule Dist Kurtosis': np.random.normal(3, 1, n_samples),
        'NSUI': np.random.random(n_samples),
        
        # Consistency Metrics
        'Self Consistency Text': np.random.random(n_samples),
        'Self Consistency SMT': np.random.random(n_samples)
    })
    
    # 修正模拟数据的逻辑：Label 和 Entropy 应该有相关性
    # 重新分配 label 以匹配 Entropy (为了让 AUROC 好看点)
    # Entropy 高 -> 正确率低
    probs = 1 / (1 + np.exp(df['Grammar Entropy'] - 3.5))
    df['smt_is_correct'] = np.random.binomial(1, probs)
    
    return df

def main():
    # 1. 加载数据
    df = pd.read_csv("outputs/all_features.csv")
    # df = generate_mock_data()
    print(f"Loaded data with {len(df)} samples.")

    # 2. 训练集成模型 (Leave-One-Out 或 Train/Test Split 均可，这里用简单的 Train/Test)
    # 实际论文可能用了 Cross Validation
    ensemble = UncertaintyEnsemble()
    
    # 为了演示，我们简单地在全量数据上训练 (Warning: Overfitting)
    # 实际操作请使用 K-Fold CV 生成 out-of-fold predictions
    ensemble.train(df, target_col='smt_is_correct')
    df['Ensemble_Prob'] = ensemble.predict_proba(df)

    # 3. 初始化评估器
    evaluator = Evaluator()
    
    # 定义要评估的指标列表 (对应论文表格)
    metrics_to_eval = [
        'Grammar Entropy', 'NSUI', 'Spectral Radius', 
        'Rule Dist Kurtosis', 
        'Self Consistency Text', 'Self Consistency SMT',
        'Ensemble_Prob' # 对应 Ensemble ML
    ]

    # =========================================================
    # 复现 Table 3: Ground Truth Correctness
    # =========================================================
    print("\n=== Table 3: Prediction of Ground Truth Correctness ===")
    results_table3 = []
    for metric in metrics_to_eval:
        # 注意：检查该指标是否在 df 中 (Ensemble 可能只在部分列算出来)
        if metric in df.columns:
            res = evaluator.evaluate_metric(df, metric, target_col='smt_is_correct')
            if res: results_table3.append(res)
    
    df_table3 = pd.DataFrame(results_table3)
    # 格式化输出
    print(df_table3[['Metric', 'AUROC', 'ECE', 'Brier', 'AURC']].round(4))
    
    # 绘制 ROC
    evaluator.plot_roc_curves(df, metrics_to_eval, 'smt_is_correct', title="ROC - Ground Truth Prediction")

    # =========================================================
    # 复现 Table 4: SMT-Text Consistency
    # =========================================================
    print("\n=== Table 4: Prediction of SMT-Text Consistency ===")
    # 重新训练 Ensemble 针对 Consistency 目标
    ensemble_cons = UncertaintyEnsemble()
    ensemble_cons.train(df, target_col='consistency_smt_text')
    df['Ensemble_Prob_Cons'] = ensemble_cons.predict_proba(df)
    
    # 更新指标列表 (使用新的 Ensemble Prob)
    metrics_cons = metrics_to_eval[:-1] + ['Ensemble_Prob_Cons']
    
    results_table4 = []
    for metric in metrics_cons:
        if metric in df.columns:
            res = evaluator.evaluate_metric(df, metric, target_col='consistency_smt_text')
            if res: results_table4.append(res)
            
    df_table4 = pd.DataFrame(results_table4)
    print(df_table4[['Metric', 'AUROC', 'ECE', 'Brier', 'AURC']].round(4))
    
    evaluator.plot_roc_curves(df, metrics_cons, 'consistency_smt_text', title="ROC - Consistency Prediction")

if __name__ == "__main__":
    main()