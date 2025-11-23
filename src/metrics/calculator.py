import math
import numpy as np
from scipy.stats import entropy, skew, kurtosis
from collections import defaultdict

class PCFGMetricCalculator:
    def __init__(self, pcfg_probs: dict, start_symbol="script"):
        """
        :param pcfg_probs: 来自 Step 3.2 的概率字典 { LHS: { RHS: prob } }
        :param start_symbol: SMT-LIB 文法的起始符号，通常是 'script' 或 'start'
        """
        self.pcfg = pcfg_probs
        self.start_symbol = start_symbol
        
        # 预处理：提取所有非终结符 (V) 和规则 (R)
        self.non_terminals = sorted(list(self.pcfg.keys()))
        self.nt_to_idx = {nt: i for i, nt in enumerate(self.non_terminals)}
        self.n_nt = len(self.non_terminals)
        
        # 展平所有概率值用于统计
        self.all_probs = []
        for lhs in self.pcfg:
            for prob in self.pcfg[lhs].values():
                self.all_probs.append(prob)

    # ==========================================================================
    # 1. 辅助计算：期望频率与矩阵
    # ==========================================================================
    def _compute_stationary_distribution(self, max_iter=50):
        """
        估算每个非终结符在推导过程中出现的期望频率 (Stationary Distribution pi(A))。
        通过模拟随机游走或迭代传播计算。
        """
        # 初始化：Start Symbol 概率为 1.0，其他为 0
        pi = defaultdict(float)
        pi[self.start_symbol] = 1.0
        
        # 简单迭代传播 (Power Iteration-like approach)
        # 注意：对于包含递归的语法，这里计算的是 Expected Counts
        for _ in range(max_iter):
            new_pi = defaultdict(float)
            # 总是假设从 start 开始会有新的推导（保持流）
            new_pi[self.start_symbol] += 1.0 
            
            changes = 0.0
            for lhs, freq in pi.items():
                if freq < 1e-6: continue
                if lhs not in self.pcfg: continue
                
                # 该 LHS 产生的规则分布
                for rhs, prob in self.pcfg[lhs].items():
                    # 解析 RHS 中的非终结符
                    # 假设 RHS 是空格分隔的字符串 "ParOpen assert term ParClose"
                    symbols = rhs.split()
                    for sym in symbols:
                        if sym in self.pcfg: # 如果是 Non-terminal
                            add_val = freq * prob
                            new_pi[sym] += add_val
            
            # 检查收敛 (这里简化处理，仅迭代固定次数或直到变化很小)
            # 在严格定义中，Spectral Radius >= 1 时期望次数是无穷大
            # 这里我们取归一化后的分布作为权重
            pi = new_pi

        # 归一化，使其和为 1，作为加权系数
        total = sum(pi.values())
        if total == 0: return pi
        for k in pi:
            pi[k] /= total
        return pi

    def _build_expectation_matrix(self):
        """
        构建期望矩阵 (Jacobian) B。
        B[i, j] = 期望 NT_j 在 NT_i 的一次推导中出现的次数。
        """
        matrix = np.zeros((self.n_nt, self.n_nt))
        
        for i, lhs in enumerate(self.non_terminals):
            if lhs not in self.pcfg: continue
            
            for rhs, prob in self.pcfg[lhs].items():
                symbols = rhs.split()
                for sym in symbols:
                    if sym in self.nt_to_idx:
                        j = self.nt_to_idx[sym]
                        # 期望次数 = 概率 * 出现次数 (这里每次出现算1)
                        matrix[i, j] += prob * 1.0
        return matrix

    # ==========================================================================
    # 2. 核心指标计算函数
    # ==========================================================================
    
    def compute_all(self):
        metrics = {}
        
        # --- A. 基础统计 (Structural) ---
        metrics['# Nonterminals'] = self.n_nt
        metrics['# Rules'] = sum(len(rhs_map) for rhs_map in self.pcfg.values())
        metrics['Avg Rules / NT'] = metrics['# Rules'] / self.n_nt if self.n_nt > 0 else 0
        
        lens = [len(rhs.split()) for rhs_map in self.pcfg.values() for rhs in rhs_map]
        metrics['Avg RHS Len'] = np.mean(lens) if lens else 0
        metrics['Max Branch Factor'] = max([len(rhs_map) for rhs_map in self.pcfg.values()]) if self.pcfg else 0

        # --- B. 规则概率分布统计 (Statistical) ---
        probs_array = np.array(self.all_probs)
        metrics['Rule Dist Mean'] = np.mean(probs_array)
        metrics['Rule Dist StdDev'] = np.std(probs_array)
        metrics['Rule Dist Skew'] = skew(probs_array) if len(probs_array) > 1 else 0
        metrics['Rule Dist Kurtosis'] = kurtosis(probs_array) if len(probs_array) > 1 else 0

        # --- C. 谱分析 (Spectral) ---
        B = self._build_expectation_matrix()
        try:
            # 计算特征值，取模最大的那个 (Spectral Radius)
            eigenvalues = np.linalg.eigvals(B)
            rho = max(abs(eigenvalues))
        except:
            rho = 0.0
        
        metrics['Spectral Radius'] = rho
        metrics['Spectral Factor'] = rho / (1 + rho)

        # --- D. 信息论指标 (Information-Theoretic) ---
        # 1. 计算每个 NT 的局部熵
        local_entropy = {}     # Shannon
        local_renyi_2 = {}     # Collision
        local_renyi_05 = {}    # alpha=0.5
        local_max_ent = {}     # log2(|Rules|)
        local_kl_uniform = {}  # KL(P || U) = log|R| - H(P)
        
        for lhs in self.non_terminals:
            probs = list(self.pcfg[lhs].values())
            
            # Shannon Entropy (base 2)
            h = entropy(probs, base=2)
            local_entropy[lhs] = h
            
            # Max Entropy
            n_rules = len(probs)
            h_max = math.log2(n_rules) if n_rules > 0 else 0
            local_max_ent[lhs] = h_max
            
            # KL Divergence from Uniform
            local_kl_uniform[lhs] = h_max - h
            
            # Renyi Entropy (alpha != 1)
            # H_alpha(X) = 1/(1-alpha) * log2(sum(p^alpha))
            p_arr = np.array(probs)
            
            # Renyi alpha=2
            sum_sq = np.sum(p_arr ** 2)
            local_renyi_2[lhs] = -math.log2(sum_sq) if sum_sq > 0 else 0
            
            # Renyi alpha=0.5
            sum_sqrt = np.sum(np.sqrt(p_arr))
            local_renyi_05[lhs] = 2 * math.log2(sum_sqrt) if sum_sqrt > 0 else 0

        # 2. 加权聚合 (Global Metrics)
        pi = self._compute_stationary_distribution()
        
        def weighted_avg(local_metric_dict):
            val = 0.0
            weight_sum = 0.0
            for lhs, w in pi.items():
                if lhs in local_metric_dict:
                    val += w * local_metric_dict[lhs]
                    weight_sum += w
            return val # 权重已归一化，直接求和
        
        grammar_entropy = weighted_avg(local_entropy)
        metrics['Grammar Entropy'] = grammar_entropy
        metrics['Perplexity'] = 2 ** grammar_entropy
        metrics['KL Divergence'] = weighted_avg(local_kl_uniform)
        metrics['Renyi Ent (2)'] = weighted_avg(local_renyi_2)
        metrics['Renyi Ent (0.5)'] = weighted_avg(local_renyi_05)
        
        global_max_ent = weighted_avg(local_max_ent)
        metrics['Max Ent'] = global_max_ent
        
        # Ent Ratio
        metrics['Ent Ratio'] = grammar_entropy / global_max_ent if global_max_ent > 1e-6 else 0
        
        # NSUI = Normalized Entropy * Spectral Factor
        # Paper: "NSUI... combines normalized grammar entropy with a factor reflecting recursive structure"
        # NSUI = E_ratio * S_factor
        metrics['NSUI'] = metrics['Ent Ratio'] * metrics['Spectral Factor']

        return metrics

# ==============================================================================
# 单元测试
# ==============================================================================
if __name__ == "__main__":
    # 模拟一个简单的概率字典
    # S -> A B (1.0)
    # A -> "x" (0.5) | A "x" (0.5)  <-- 递归
    # B -> "y" (1.0)
    mock_probs = {
        "script": {"command": 1.0},
        "command": {"assert term": 1.0},
        "term": {"x": 0.4, "func term": 0.6}, # 递归
        "func": {"f": 1.0}
    }
    
    calc = PCFGMetricCalculator(mock_probs, start_symbol="script")
    metrics = calc.compute_all()
    
    print(f"{'Metric Name':<25} | {'Value':<10}")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k:<25} | {v:.4f}")