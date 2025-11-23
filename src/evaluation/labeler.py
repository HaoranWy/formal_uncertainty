import re
import sys
import os

# 确保能导入之前的模块
# 假设项目根目录在 PYTHONPATH 中，或者通过相对路径导入
try:
    from src.solver.z3_executor import Z3Executor
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from src.solver.z3_executor import Z3Executor

class correctness_labeler:
    def __init__(self):
        self.executor = Z3Executor(timeout_ms=5000)

    def normalize_ground_truth(self, gt_str: str) -> bool:
        """
        将各种格式的 Ground Truth (e.g., "TRUE", "True", "yes", True) 统一为布尔值。
        """
        if isinstance(gt_str, bool):
            return gt_str
        
        s = str(gt_str).strip().lower()
        if s in ["true", "yes", "1", "t"]:
            return True
        if s in ["false", "no", "0", "f"]:
            return False
        return None # 无法识别

    def evaluate_smt(self, smt_code: str, ground_truth: bool) -> dict:
        """
        执行 SMT 代码并判定正确性。
        逻辑基于反证法：UNSAT = TRUE, SAT = FALSE
        """
        # 1. 执行 Z3
        status, result_str, error_msg = self.executor.execute(smt_code)
        
        prediction = None
        is_correct = False
        
        # 2. 映射结果 (Proof by Contradiction)
        if status == "unsat":
            # 否定不可满足 -> 原命题必然为真
            prediction = True
        elif status == "sat":
            # 否定可满足 -> 存在反例 -> 原命题为假
            prediction = False
        else:
            # unknown, timeout, error
            prediction = None

        # 3. 比对正确性
        if prediction is not None and ground_truth is not None:
            is_correct = (prediction == ground_truth)
        
        return {
            "smt_status": status,       # sat/unsat/unknown/error
            "smt_prediction": prediction, # True/False/None
            "is_correct": is_correct,   # Boolean
            "error_msg": error_msg
        }

    def evaluate_text(self, text_output: str, ground_truth: bool) -> dict:
        """
        使用启发式规则从 LLM 的自然语言回答中提取 Yes/No，并判定正确性。
        """
        prediction = None
        is_correct = False
        
        # 简单的文本清洗
        text = text_output.strip().lower()
        
        # 启发式提取：查找最后的有效关键词
        # 很多 CoT 会说 "blah blah... so the answer is False."
        # 我们搜索 "true", "false", "yes", "no"
        
        # 优先级：精确匹配 -> 句尾匹配 -> 包含匹配
        if text == "true" or text == "yes":
            prediction = True
        elif text == "false" or text == "no":
            prediction = False
        else:
            # 使用正则寻找答案模式，例如 "answer is yes"
            # 这是一个简化版，实际工程可能需要更复杂的 Regex
            matches = re.findall(r"\b(yes|no|true|false)\b", text)
            if matches:
                last_match = matches[-1] # 取最后一个词通常是最可靠的结论
                if last_match in ["true", "yes"]:
                    prediction = True
                else:
                    prediction = False
        
        # 比对
        if prediction is not None and ground_truth is not None:
            is_correct = (prediction == ground_truth)

        return {
            "text_prediction": prediction,
            "is_correct": is_correct
        }

    def process_sample(self, sample: dict) -> dict:
        """
        处理单个数据样本
        sample 结构:
        {
            "id": "...",
            "ground_truth": "True",
            "text_output": "...",
            "smt_code": "..."
        }
        """
        gt = self.normalize_ground_truth(sample.get("ground_truth"))
        
        # 1. 评估 SMT
        smt_res = self.evaluate_smt(sample.get("smt_code", ""), gt)
        
        # 2. 评估 Text
        text_res = self.evaluate_text(sample.get("text_output", ""), gt)
        
        # 3. 合并结果
        result = {
            "id": sample.get("id"),
            "ground_truth_bool": gt,
            # SMT 结果
            "smt_executed_status": smt_res["smt_status"],
            "smt_pred_bool": smt_res["smt_prediction"],
            "smt_is_correct": smt_res["is_correct"],
            # Text 结果
            "text_pred_bool": text_res["text_prediction"],
            "text_is_correct": text_res["is_correct"],
            # 一致性 (Consistency)
            "consistency_smt_text": (smt_res["smt_prediction"] == text_res["text_prediction"]) 
                                     if (smt_res["smt_prediction"] is not None and text_res["text_prediction"] is not None) 
                                     else False
        }
        return result

# ==============================================================================
# 单元测试
# ==============================================================================
if __name__ == "__main__":
    labeler = correctness_labeler()
    
    print("=== Test 1: SMT Correctness (Proof by Contradiction) ===")
    # 假设 Ground Truth 是 TRUE
    # 场景 A: LLM 生成了正确的反证代码，Z3 发现否定无法满足 -> unsat -> 预测为 True -> Correct
    code_correct = "(declare-const x Int) (assert (> x 5)) (assert (not (> x 5))) (check-sat)" # 矛盾 -> unsat
    res = labeler.evaluate_smt(code_correct, ground_truth=True)
    print(f"Case A (Expected: Correct): {res['is_correct']} (Status: {res['smt_status']})")
    
    # 场景 B: LLM 生成了错误代码，Z3 发现否定是可以满足的 -> sat -> 预测为 False -> Incorrect
    code_wrong = "(declare-const x Int) (assert (> x 5)) (assert (not (> x 100))) (check-sat)" # x=6 满足 -> sat
    res = labeler.evaluate_smt(code_wrong, ground_truth=True)
    print(f"Case B (Expected: Incorrect): {res['is_correct']} (Status: {res['smt_status']})")

    print("\n=== Test 2: Text Extraction ===")
    text_1 = "Therefore, the answer is True."
    print(f"Text: '{text_1}' -> {labeler.evaluate_text(text_1, True)}")
    
    text_2 = "I am not sure, but looking at the facts, no."
    print(f"Text: '{text_2}' -> {labeler.evaluate_text(text_2, False)}")