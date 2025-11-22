import z3
import re
import signal

class Z3Executor:
    def __init__(self, timeout_ms=5000):
        """
        初始化执行器
        :param timeout_ms: 超时时间，单位毫秒 (默认5秒)
        """
        self.timeout_ms = timeout_ms

    def clean_smt_code(self, raw_text: str) -> str:
        """
        清洗 LLM 输出，提取代码块，去除 Markdown 标记
        """
        # 1. 尝试提取 ```smt2 或 ```smt 或 ``` 代码块中的内容
        code_block_pattern = r"```(?:smt2?|lisp)?\s*(.*?)```"
        match = re.search(code_block_pattern, raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 2. 如果没有代码块，尝试去除行首行尾的无关文本（简单启发式）
        # 这一步视情况而定，如果 LLM 直接输出代码则不需要
        return raw_text.strip()

    def execute(self, smt_code: str):
        """
        执行 SMT-LIB 代码并返回结果
        :return: (status, result_str, error_msg)
                 status: 'success' | 'unsat' | 'sat' | 'unknown' | 'timeout' | 'error'
        """
        cleaned_code = self.clean_smt_code(smt_code)
        
        # 重置 Z3 上下文，防止之前的定义污染当前执行
        # 虽然 Z3 Py API 主要基于对象，但彻底清理是一个好习惯
        z3.main_ctx().context = None 
        
        try:
            # 1. 解析 SMT 代码字符串为 Z3 的表达式对象
            # parse_smt2_string 会返回一个 AstVector（包含所有的 assert）
            assertions = z3.parse_smt2_string(cleaned_code)
            
            # 2. 创建求解器
            s = z3.Solver()
            
            # 3. 设置硬性参数：超时时间
            s.set("timeout", self.timeout_ms)
            
            # 4. 将解析出的断言加入求解器
            s.add(assertions)
            
            # 5. 执行检查 (check-sat)
            result = s.check()
            
            if result == z3.sat:
                return "sat", "sat", None
            elif result == z3.unsat:
                return "unsat", "unsat", None
            elif result == z3.unknown:
                # 可能是超时导致 unknown，也可能是逻辑太复杂
                reason = s.reason_unknown()
                if "timeout" in reason or "canceled" in reason:
                     return "timeout", "unknown", reason
                return "unknown", "unknown", reason
                
        except z3.Z3Exception as e:
            # 捕获语法错误 (Syntax Error) 或解析错误
            return "error", None, f"Z3 Parsing Error: {str(e)}"
        except Exception as e:
            # 捕获其他 Python 层面错误
            return "error", None, f"Runtime Error: {str(e)}"

        return "error", None, "Unexpected execution path"

# --- 单元测试部分 ---
if __name__ == "__main__":
    executor = Z3Executor(timeout_ms=2000) # 设置2秒超时测试
    
    # 案例 1: 正常的 SAT 逻辑
    print("Test 1 (SAT):", executor.execute("""
    (declare-const x Int)
    (assert (> x 10))
    """))
    
    # 案例 2: 正常的 UNSAT 逻辑
    print("Test 2 (UNSAT):", executor.execute("""
    (declare-const x Int)
    (assert (> x 10))
    (assert (< x 5))
    """))
    
    # 案例 3: 带 Markdown 的脏数据
    print("Test 3 (Markdown):", executor.execute("""
    Sure, here is the code:
    ```smt
    (declare-const a Bool)
    (assert (= a (not a)))
    ```
    """))

    # 案例 4: 语法错误
    print("Test 4 (Syntax Error):", executor.execute("""
    (declare-const x Int)
    (assert (> x ))  <-- 缺少操作数
    """))
    
    # 案例 5: 模拟超时 (构造一个超难的非线性整数算术问题，或者海量约束)
    # 注意：构造短时间内必超时的 Z3 代码其实不容易，因为 Z3 很强。
    # 这里用一个著名的 Ackermann 函数递归定义（Z3 处理递归很慢）
    ackermann_code = """
    (define-fun-rec ack ((m Int) (n Int)) Int
      (ite (<= m 0) (+ n 1)
        (ite (<= n 0) (ack (- m 1) 1)
          (ack (- m 1) (ack m (- n 1))))))
    (assert (> (ack 4 1) 100)) 
    """
    print("Test 5 (Possible Timeout):", executor.execute(ackermann_code))