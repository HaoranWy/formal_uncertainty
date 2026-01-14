import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, GroupKFold

# 添加 src 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.ensemble import UncertaintyEnsemble
from src.evaluation.analysis import Evaluator

# ==============================================================================
# 定义 25 个指标
# ==============================================================================
METRICS_PCFG = [
    'Grammar Entropy', 'Perplexity', 'KL Divergence', 'NSUI',
    'Renyi Ent (2)', 'Renyi Ent (0.5)', 'Max Ent', 'Ent Ratio',
    'Spectral Factor', 'Spectral Radius', '# Nonterminals', '# Rules',
    'Avg Rules / NT', 'Avg RHS Len', 'Max Branch Factor',
    'Rule Dist Mean', 'Rule Dist StdDev', 'Rule Dist Skew', 'Rule Dist Kurtosis'
]

METRICS_CONSISTENCY = [
    'Self Consistency Text', 'Self Consistency SMT'
]

METRICS_ENSEMBLE = [
    'Ensemble Average', 'Ensemble Weighted', 'Ensemble ML', 'Ensemble Simple'
]

ALL_METRICS = METRICS_PCFG + METRICS_CONSISTENCY + METRICS_ENSEMBLE

def calculate_heuristic_ensembles(df):
    """计算启发式集成指标"""
    df = df.copy()
    
    # 提取特征并填充 NaN
    uncertainty_cols = [c for c in METRICS_PCFG if c in df.columns]
    confidence_cols = [c for c in METRICS_CONSISTENCY if c in df.columns]
    
    if not uncertainty_cols: return df # 数据不足
    
    # 标准化
    scaler = StandardScaler()
    features = df[uncertainty_cols + confidence_cols].fillna(0)
    scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=df.index)
    
    # 统一方向：将不确定性指标取反 (-x)
    for col in uncertainty_cols:
        scaled[col] = -scaled[col]
            
    # 1. Ensemble Average
    df['Ensemble Average'] = scaled.mean(axis=1)
    
    # 2. Ensemble Simple (Grammar Entropy + Text Consistency)
    simple_cols = ['Grammar Entropy', 'Self Consistency Text']
    available_simple = [c for c in simple_cols if c in scaled]
    if available_simple:
        df['Ensemble Simple'] = scaled[available_simple].mean(axis=1)
    else:
        df['Ensemble Simple'] = 0
        
    # 3. Ensemble Weighted (启发式权重)
    w_score = np.zeros(len(df))
    if 'Self Consistency Text' in scaled:
        w_score += 0.6 * scaled['Self Consistency Text']
    if 'Grammar Entropy' in scaled:
        w_score += 0.2 * scaled['Grammar Entropy']
    if 'Spectral Radius' in scaled:
        w_score += 0.2 * scaled['Spectral Radius']
    df['Ensemble Weighted'] = w_score

    return df

def evaluate_dataset(dataset_name, csv_path, evaluator):
    print(f"\n{'='*40}\n Evaluating {dataset_name} \n{'='*40}")
    
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['smt_is_correct'])
    
    if len(df) < 10:
        print("⚠️ Too few samples to evaluate.")
        return

    print(f"Loaded {len(df)} samples.")

    # ---------------------------------------------------------
    # 1. 训练集成模型 (Ensemble ML) - 使用 Group K-Fold
    # ---------------------------------------------------------
    print("Training Ensemble ML with 5-Fold CV (Grouped by Question)...")
    
    ensemble_ml = UncertaintyEnsemble()
    
    # 准备数据 X, y
    # 注意：这里我们需要手动调用 ensemble_ml 内部的 prepare_data 来获取处理好的 X
    # 但为了简单，我们直接利用 sklearn 的 cross_val_predict 结合 pipeline
    
    X, y = ensemble_ml.prepare_data(df, target_col='smt_is_correct')
    groups = df['question_id'] # 确保按问题分组划分，防止泄漏
    
    try:
        # 使用 GroupKFold 保证同一个问题的 100 个样本要么全在训练集，要么全在测试集
        gkf = GroupKFold(n_splits=5)
        
        # cross_val_predict 会返回每个样本在作为 Test set 时的预测概率
        # method='predict_proba' 返回 (n_samples, 2)，我们取第二列 (Positive class)
        probs = cross_val_predict(
            ensemble_ml.model, 
            X, 
            y, 
            cv=gkf, 
            groups=groups, 
            method='predict_proba'
        )[:, 1]
        
        df['Ensemble ML'] = probs
        
    except Exception as e:
        print(f"⚠️ Ensemble ML CV failed: {e}")
        # 如果 CV 失败（例如样本太少），回退到简单训练
        ensemble_ml.train(df, target_col='smt_is_correct')
        df['Ensemble ML'] = ensemble_ml.predict_proba(df)


    # --- 2. 计算其他集成指标 ---
    df = calculate_heuristic_ensembles(df)

    # --- 3. 更新 Evaluator 的指标方向 ---
    for m in METRICS_ENSEMBLE:
        evaluator.metric_config[m] = {'ascending': False}

    # --- 4. 评估 Ground Truth Prediction ---
    print("\n>>> Task: Prediction of Ground Truth Correctness (Table 3)")
    results = []
    
    for metric in ALL_METRICS:
        if metric in df.columns:
            res = evaluator.evaluate_metric(df, metric, target_col='smt_is_correct')
            if res: results.append(res)
    
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        # 打印完整榜单
        print(res_df[['Metric', 'AUROC', 'ECE', 'AURC']].sort_values('AUROC', ascending=False).to_string(index=False))
        
        # **画图**：选出 AUROC 最高的 Top 7 指标进行绘制
        top_metrics = res_df.sort_values('AUROC', ascending=False).head(25)['Metric'].tolist()
        evaluator.plot_roc_curves(df, top_metrics, 'smt_is_correct', title=f"{dataset_name} - Ground Truth Prediction")

    # --- 5. 评估 Consistency Prediction ---
    if 'consistency_smt_text' in df.columns and df['consistency_smt_text'].nunique() > 1:
        print("\n>>> Task: Prediction of SMT-Text Consistency (Table 4)")
        
        # 针对 Consistency 重新训练 ML
        try:
            ensemble_ml_cons = UncertaintyEnsemble()
            ensemble_ml_cons.train(df, target_col='consistency_smt_text')
            df['Ensemble ML'] = ensemble_ml_cons.predict_proba(df)
        except: pass
        
        results_cons = []
        for metric in ALL_METRICS:
            if metric in df.columns:
                res = evaluator.evaluate_metric(df, metric, target_col='consistency_smt_text')
                if res: results_cons.append(res)
        
        res_df_cons = pd.DataFrame(results_cons)
        if not res_df_cons.empty:
            print(res_df_cons[['Metric', 'AUROC', 'ECE', 'AURC']].sort_values('AUROC', ascending=False).to_string(index=False))
            
            # **画图**
            top_metrics_cons = res_df_cons.sort_values('AUROC', ascending=False).head(25)['Metric'].tolist()
            evaluator.plot_roc_curves(df, top_metrics_cons, 'consistency_smt_text', title=f"{dataset_name} - Consistency Prediction")
    else:
        print("\n⚠️ Skipping Consistency evaluation (Data is 100% consistent or error).")

def main():
    evaluator = Evaluator()
    
    # 你的路径配置
    base_path = "/data/newmodel/uncertainty/formal_uncertainty/output/generations"
    # 如果路径不存在，尝试 fallback
    if not os.path.exists(base_path):
        base_path = "/data/newmodel/uncertainty/formal_uncertainty/outputs"

    datasets = [
        ("StrategyQA", f"{base_path}/strategyqa_features.csv"),
        ("ProofWriter", f"{base_path}/proofwriter_features.csv"),
        ("FOLIO", f"{base_path}/folio_features.csv"),
        ("ProntoQA", f"{base_path}/prontoqa_features.csv"),
    ]
    
    for name, path in datasets:
        evaluate_dataset(name, path, evaluator)

if __name__ == "__main__":
    main()