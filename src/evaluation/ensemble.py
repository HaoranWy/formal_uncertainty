import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

class UncertaintyEnsemble:
    """
    Ensemble ML Predictor based on 'Grammars of Formal Uncertainty'.
    Uses Logistic Regression to fuse 19 PCFG-derived metrics.
    """
    
    def __init__(self):
        # 论文中定义的 19 个核心 PCFG 指标
        self.feature_columns = [
            # --- Information Theoretic ---
            'Grammar Entropy', 'Perplexity', 'KL Divergence', 'NSUI',
            'Renyi Ent (2)', 'Renyi Ent (0.5)', 'Max Ent', 'Ent Ratio',
            
            # --- Spectral & Structural ---
            'Spectral Radius', 'Spectral Factor', 
            '# Nonterminals', '# Rules', 'Avg Rules / NT', 
            'Avg RHS Len', 'Max Branch Factor',
            
            # --- Rule Distribution ---
            'Rule Dist Mean', 'Rule Dist StdDev', 'Rule Dist Skew', 'Rule Dist Kurtosis',
            'Self Consistency Text',
            'Self Consistency SMT' 
        ]
        
        # 构建处理流水线：
        # 1. Imputer: 处理可能出现的 NaN (例如标准差计算时只有一个样本)
        # 2. Scaler: 标准化 (非常重要！因为 Entropy 和 Spectral Radius 的量级完全不同)
        # 3. Model: 逻辑回归 (class_weight='balanced' 处理正负样本不平衡)
        self.model = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')), 
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                class_weight='balanced', 
                max_iter=10000,  # 论文提及：trained for up to 10,000 iterations
                solver='lbfgs',
                random_state=42
            ))
        ])
        
        self.is_fitted = False

    def prepare_data(self, df: pd.DataFrame, target_col: str):
        """
        从 DataFrame 中提取特征矩阵 X 和 目标向量 y
        """
        # 1. 检查特征列是否存在，缺失的填 0
        for col in self.feature_columns:
            if col not in df.columns:
                print(f"Warning: Feature '{col}' missing in data. Filling with 0.")
                df[col] = 0.0
        
        # 2. 提取 X
        X = df[self.feature_columns].copy()
        
        # 3. 处理 Inf / -Inf (对数计算可能会产生)
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # 4. 提取 y (如果用于推理，y 可以是 None)
        y = None
        if target_col in df.columns:
            # 确保 y 是整数 (0/1)
            y = df[target_col].astype(int).values
            
        return X, y

    def train(self, df: pd.DataFrame, target_col='smt_is_correct'):
        """
        训练模型
        :param df: 包含特征和标签的 DataFrame
        :param target_col: 预测目标 
                           - 'smt_is_correct': 预测 SMT 是否正确 (Ground Truth Prediction)
                           - 'consistency_smt_text': 预测 SMT 是否与 Text 一致
        """
        print(f"Training Ensemble ML for target: {target_col}...")
        
        X, y = self.prepare_data(df, target_col)
        
        if y is None:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
            
        # 拟合流水线
        self.model.fit(X, y)
        self.is_fitted = True
        
        # 打印特征权重 (可选，用于分析哪个指标最重要)
        self._print_feature_importance()

    def predict_proba(self, df: pd.DataFrame):
        """
        返回预测为 Positive (Class 1, e.g., Correct) 的概率
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call train() first.")
        
        X, _ = self.prepare_data(df, target_col="") # y not needed for predict
        
        # predict_proba 返回 [P(0), P(1)]
        return self.model.predict_proba(X)[:, 1]

    def _print_feature_importance(self):
        """
        打印逻辑回归的系数，系数绝对值越大说明该特征越重要
        """
        classifier = self.model.named_steps['classifier']
        coeffs = classifier.coef_[0]
        
        feature_importance = pd.DataFrame({
            'Feature': self.feature_columns,
            'Coefficient': coeffs,
            'Abs_Coeff': np.abs(coeffs)
        }).sort_values(by='Abs_Coeff', ascending=False)
        
        print("\n=== Feature Importance (Top 5) ===")
        print(feature_importance.head(5))
        print("==================================\n")

# ==============================================================================
# 模拟运行 (用于验证代码逻辑)
# ==============================================================================
if __name__ == "__main__":
    # 1. 模拟一批数据 (100个样本)
    np.random.seed(42)
    mock_data = {
        'Grammar Entropy': np.random.normal(5, 1, 100),
        'Spectral Radius': np.random.normal(0.8, 0.2, 100),
        'Rule Dist Kurtosis': np.random.normal(3, 1, 100),
        # ... 其他特征省略，prepare_data 会自动补0
        'smt_is_correct': np.random.randint(0, 2, 100)  # 0 或 1
    }
    df_mock = pd.DataFrame(mock_data)
    
    # 2. 初始化
    ensemble = UncertaintyEnsemble()
    
    # 3. 划分训练集/测试集 (模拟实际使用中的 Cross-Validation)
    df_train, df_test = train_test_split(df_mock, test_size=0.2, random_state=42)
    
    # 4. 训练
    ensemble.train(df_train, target_col='smt_is_correct')
    
    # 5. 预测
    probs = ensemble.predict_proba(df_test)
    y_test = df_test['smt_is_correct'].values
    
    # 6. 简单评估
    try:
        auc = roc_auc_score(y_test, probs)
        print(f"Test AUROC: {auc:.4f}")
    except ValueError:
        print("样本太少，无法计算 AUROC")
    
    print(f"Predicted Probabilities (First 5): {probs[:5]}")