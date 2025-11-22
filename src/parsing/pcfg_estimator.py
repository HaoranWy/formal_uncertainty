from collections import defaultdict

class PCFGEstimator:
    """
    将原始规则计数转换为概率分布 (Probabilistic Context-Free Grammar)。
    
    """
    
    def __init__(self, rules_counts: dict, alpha: float = 1.0):
        """
        :param rules_counts: 来自 pcfg_builder.py 的计数结果 
                             { "LHS": {"RHS1": 10, "RHS2": 5} }
        :param alpha: Laplace Smoothing factor (default=1.0)
        """
        self.raw_counts = rules_counts
        self.alpha = alpha
        self.pcfg_probs = defaultdict(dict)  # 存储最终概率
        self.lhs_stats = {}  # 存储每个 LHS 的统计信息 (总数, 规则种类数)
        
        self._estimate()

    def _estimate(self):
        """
        执行 MLE + Laplace Smoothing 计算
        """
        for lhs, rhs_map in self.raw_counts.items():
            # 1. 计算该 LHS 的原始总计数 (Total raw count)
            total_raw_count = sum(rhs_map.values())
            
            # 2. 计算该 LHS 下有多少种不同的规则 (Vocabulary size of productions for A)
            # 注意：这里仅基于观察到的规则集合。
            # 严格的 SMT-LIB 全集很难获取，论文通常指观察到的集合。
            unique_rules_count = len(rhs_map)
            
            # 3. 计算平滑后的分母
            # Denominator = N + alpha * |V|
            smoothed_denominator = total_raw_count + (self.alpha * unique_rules_count)
            
            # 记录统计信息供后续分析 (例如计算熵)
            self.lhs_stats[lhs] = {
                "total_raw": total_raw_count,
                "unique_rules": unique_rules_count,
                "denominator": smoothed_denominator
            }
            
            # 4. 计算每个 RHS 的条件概率
            for rhs, count in rhs_map.items():
                # Numerator = count + alpha
                smoothed_count = count + self.alpha
                probability = smoothed_count / smoothed_denominator
                
                self.pcfg_probs[lhs][rhs] = probability

    def get_probability(self, lhs: str, rhs: str) -> float:
        """
        查询特定规则的概率 P(LHS -> RHS)
        如果规则从未出现过（在训练集中不存在），返回平滑后的最小概率。
        """
        if lhs in self.pcfg_probs and rhs in self.pcfg_probs[lhs]:
            return self.pcfg_probs[lhs][rhs]
        
        # 处理 Unseen Rule (虽然在基于自身样本集诱导 PCFG 时不会发生，但为了代码健壮性)
        if lhs in self.lhs_stats:
            denom = self.lhs_stats[lhs]["denominator"]
            return self.alpha / denom # 仅剩平滑项
        
        return 0.0

    def export_grammar(self):
        """
        导出概率字典，格式: { LHS: { RHS: prob } }
        """
        return dict(self.pcfg_probs)

# ==============================================================================
# 单元测试
# ==============================================================================
if __name__ == "__main__":
    # 模拟 Step 3.1 的输出数据
    # 假设 Non-terminal 'term' 出现了 10 次：
    #   7 次变成 'x'
    #   3 次变成 '0'
    mock_counts = {
        "term": {
            "x": 7,
            "0": 3
        },
        "command": {
            "assert term": 5
        }
    }
    
    print("原始计数:", mock_counts)
    
    # 1. 不使用平滑 (alpha=0) -> 纯 MLE
    estimator_mle = PCFGEstimator(mock_counts, alpha=0.0)
    print("\n--- MLE (alpha=0) ---")
    print(f"P(term -> x) = {estimator_mle.get_probability('term', 'x'):.4f} (Expected: 0.7)")
    print(f"P(term -> 0) = {estimator_mle.get_probability('term', '0'):.4f} (Expected: 0.3)")

    # 2. 使用 Laplace 平滑 (alpha=1)
    # term -> x: (7 + 1) / (10 + 1*2) = 8 / 12 = 0.666...
    # term -> 0: (3 + 1) / (10 + 1*2) = 4 / 12 = 0.333...
    estimator_laplace = PCFGEstimator(mock_counts, alpha=1.0)
    print("\n--- Laplace (alpha=1) ---")
    print(f"P(term -> x) = {estimator_laplace.get_probability('term', 'x'):.4f} (Expected: 0.6667)")
    print(f"P(term -> 0) = {estimator_laplace.get_probability('term', '0'):.4f} (Expected: 0.3333)")
    
    # 3. command 只有 1 种规则
    # command -> assert term: (5 + 1) / (5 + 1*1) = 6 / 6 = 1.0
    print(f"P(command -> assert term) = {estimator_laplace.get_probability('command', 'assert term'):.4f}")