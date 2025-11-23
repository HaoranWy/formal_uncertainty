import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# 添加 src 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.ensemble import UncertaintyEnsemble
from src.evaluation.analysis import Evaluator

def evaluate_dataset(dataset_name, csv_path, evaluator):
    print(f"\n{'='*20} Evaluating {dataset_name} {'='*20}")
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 简单的清洗：去掉全 NaN 的列或行
    df = df.dropna(subset=['smt_is_correct'])
    
    if len(df) < 10:
        print("⚠️ Too few samples to evaluate.")
        return

    print(f"Loaded {len(df)} samples.")

    # --- 训练集成模型 ---
    # 注意：这里使用简单的 Train/Test 逻辑，严谨复现建议使用 5-Fold CV
    ensemble = UncertaintyEnsemble()
    try:
        ensemble.train(df, target_col='smt_is_correct')
        df['Ensemble_Prob'] = ensemble.predict_proba(df)
    except Exception as e:
        print(f"⚠️ Ensemble training failed (maybe class imbalance?): {e}")
        df['Ensemble_Prob'] = 0.5

    # --- 定义要评估的指标 ---
    metrics = [
        'Grammar Entropy', 'NSUI', 'Spectral Radius', 
        'Rule Dist Kurtosis', 
        'Self Consistency Text', 'Self Consistency SMT',
        'Ensemble_Prob'
    ]

    # --- 1. Ground Truth Prediction (Table 3) ---
    print("\n--- Task: Prediction of Ground Truth Correctness ---")
    results = []
    for metric in metrics:
        if metric in df.columns:
            res = evaluator.evaluate_metric(df, metric, target_col='smt_is_correct')
            if res: results.append(res)
    
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print(res_df[['Metric', 'AUROC', 'AURC']].sort_values('AUROC', ascending=False).to_string(index=False))
    
    # 画图
    evaluator.plot_roc_curves(df, metrics, 'smt_is_correct', title=f"ROC - {dataset_name} (Ground Truth)")

    # --- 2. Consistency Prediction (Table 4) ---
    # 只有当存在不一致的情况时才评估
    if df['consistency_smt_text'].nunique() > 1:
        print("\n--- Task: Prediction of SMT-Text Consistency ---")
        # 重新训练集成模型用于一致性
        ensemble_cons = UncertaintyEnsemble()
        ensemble_cons.train(df, target_col='consistency_smt_text')
        df['Ensemble_Prob_Cons'] = ensemble_cons.predict_proba(df)
        
        metrics_cons = metrics[:-1] + ['Ensemble_Prob_Cons']
        results_cons = []
        for metric in metrics_cons:
            if metric in df.columns:
                res = evaluator.evaluate_metric(df, metric, target_col='consistency_smt_text')
                if res: results_cons.append(res)
        
        res_df_cons = pd.DataFrame(results_cons)
        if not res_df_cons.empty:
            print(res_df_cons[['Metric', 'AUROC', 'AURC']].sort_values('AUROC', ascending=False).to_string(index=False))
            
        evaluator.plot_roc_curves(df, metrics_cons, 'consistency_smt_text', title=f"ROC - {dataset_name} (Consistency)")
    else:
        print("\n⚠️ SMT and Text are 100% consistent (or data error), skipping Table 4.")

def main():
    evaluator = Evaluator()
    
    datasets = [
        ("StrategyQA", "/data/newmodel/uncertainty/formal_uncertainty/output/generations/strategyqa_features.csv"),
        ("ProofWriter", "/data/newmodel/uncertainty/formal_uncertainty/output/generations/proofwriter_features.csv"),
        ("FOLIO", "/data/newmodel/uncertainty/formal_uncertainty/output/generations/folio_features.csv"),
        ("ProntoQA", "/data/newmodel/uncertainty/formal_uncertainty/output/generations/prontoqa_features.csv"),
    ]
    
    for name, path in datasets:
        evaluate_dataset(name, path, evaluator)

if __name__ == "__main__":
    main()