import pandas as pd

def add_self_consistency_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 Self-Consistency Text 和 Self-Consistency SMT 指标。
    
    原理：
    对于某个问题 (question_id)，如果 100 个样本中有 80 个预测 'True'，20 个预测 'False'。
    - 预测 'True' 的样本，其 Self Consistency 分数为 0.8。
    - 预测 'False' 的样本，其 Self Consistency 分数为 0.2。
    
    :param df: 包含所有样本的 DataFrame，必须包含列:
               ['question_id', 'text_pred_bool', 'smt_pred_bool']
    :return: 新增了两列 ['Self Consistency Text', 'Self Consistency SMT'] 的 DataFrame
    """
    
    # 确保数据按问题分组
    # transform 方法可以将聚合结果直接广播回每一行，保持 DataFrame 形状不变
    
    # --- 1. 计算 Text Consistency ---
    # 注意：NaN (解析失败) 会被视为一种独立的类别，或者在计算中被忽略。
    # 这里我们将 NaN 视为一种答案状态（即 '无法回答' 也是一种回答）
    
    def calculate_frequency(series):
        # value_counts(normalize=True) 返回频率 (0.0 - 1.0)
        # map 将频率映射回原始 Series 中的每个值
        counts = series.value_counts(normalize=True, dropna=False)
        return series.map(counts)

    # 对每个问题分组，分别计算 Text 预测分布
    df['Self Consistency Text'] = df.groupby('question_id')['text_pred_bool']\
        .transform(calculate_frequency)
    
    # --- 2. 计算 SMT Consistency ---
    # 同样对 SMT 预测结果 (True/False/None) 计算分布
    df['Self Consistency SMT'] = df.groupby('question_id')['smt_pred_bool']\
        .transform(calculate_frequency)
    
    # 填充可能的 NaN (极少数情况)
    df['Self Consistency Text'] = df['Self Consistency Text'].fillna(0.0)
    df['Self Consistency SMT'] = df['Self Consistency SMT'].fillna(0.0)
    
    return df

# ==============================================================================
# 单元测试
# ==============================================================================
if __name__ == "__main__":
    # 模拟数据：2个问题，每个问题 5 个样本
    data = {
        'question_id': ['Q1']*5 + ['Q2']*5,
        
        # Q1: 4个True, 1个False -> True得分0.8, False得分0.2
        # Q2: 3个True, 2个None (Error) -> True得分0.6, None得分0.4
        'text_pred_bool': [True, True, True, True, False,  
                           True, True, True, None, None],
        
        'smt_pred_bool':  [True, False, True, False, True, 
                           False, False, False, False, False] 
    }
    
    df = pd.DataFrame(data)
    print("=== 原始数据 ===")
    print(df)
    
    df_processed = add_self_consistency_scores(df)
    
    print("\n=== 计算一致性后 ===")
    print(df_processed[['question_id', 'text_pred_bool', 'Self Consistency Text']])
    
    # 验证 Q1 的逻辑
    # Row 0 (True): 应该 0.8
    # Row 4 (False): 应该 0.2