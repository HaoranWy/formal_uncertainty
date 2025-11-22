import sys
from collections import defaultdict, Counter
from antlr4 import *
from antlr4.tree.Tree import TerminalNode
from antlr4.error.ErrorListener import ErrorListener

# 注意：根据你的运行方式，这里可能需要去掉 '.' 使用绝对导入
# 如果作为脚本直接运行，请去掉前面的点；如果作为模块运行，保留点
try:
    from .SMTLIBv2Lexer import SMTLIBv2Lexer
    from .SMTLIBv2Parser import SMTLIBv2Parser
    from .SMTLIBv2Listener import SMTLIBv2Listener
except ImportError:
    # Fallback for running directly as script
    from SMTLIBv2Lexer import SMTLIBv2Lexer
    from SMTLIBv2Parser import SMTLIBv2Parser
    from SMTLIBv2Listener import SMTLIBv2Listener

class PCFGTracker(SMTLIBv2Listener):
    """
    核心监听器：用于遍历 Parse Tree 并统计产生式规则 (Production Rules)。
    """
    def __init__(self, parser):
        self.parser = parser
        # 记录 V (Non-terminals) 和 R (Rules) 的计数
        # 结构: LHS -> { RHS_string: count }
        self.rules_counter = defaultdict(Counter)
        self.total_rule_usage = 0

    def enterEveryRule(self, ctx):
        """
        ANTLR 的钩子函数，每进入一个非终结符节点都会触发。
        """
        # 1. 获取左侧非终结符 (LHS) 的名字
        try:
            lhs_name = self.parser.ruleNames[ctx.getRuleIndex()]
        except IndexError:
            lhs_name = "UNKNOWN_RULE"
        
        # 2. 构建右侧产生式 (RHS)
        rhs_components = []
        
        if ctx.children:
            for child in ctx.children:
                if isinstance(child, TerminalNode):
                    # --- FIX START: 修复 AttributeError: no attribute 'vocabulary' ---
                    token_type_id = child.getSymbol().type
                    
                    # 尝试从 symbolicNames 列表中获取 Token 名称 (例如 'ParOpen', 'Symbol')
                    token_name = None
                    if hasattr(self.parser, 'symbolicNames'):
                        try:
                            if 0 <= token_type_id < len(self.parser.symbolicNames):
                                token_name = self.parser.symbolicNames[token_type_id]
                        except IndexError:
                            pass
                    
                    # 如果 symbolicNames 里是 None (或者是字面量)，或者越界，回退到文本内容
                    if not token_name:
                         # 尝试用 DisplayName 或直接用文本
                         token_name = str(child)
                    
                    rhs_components.append(token_name)
                    # --- FIX END ---
                
                else:
                    # 如果是非终结符 (RuleContext)，获取其规则名称
                    try:
                        rule_index = child.getRuleIndex()
                        rule_name = self.parser.ruleNames[rule_index]
                        rhs_components.append(rule_name)
                    except:
                        rhs_components.append("UNKNOWN_NONTERM")
        else:
            # 处理空规则 (Epsilon)
            rhs_components.append("<EPSILON>")

        # 3. 组合 RHS
        rhs_string = " ".join(str(x) for x in rhs_components)
        
        # 4. 记录计数
        self.rules_counter[lhs_name][rhs_string] += 1
        self.total_rule_usage += 1


class SyntaxErrorCatcher(ErrorListener):
    """
    用于捕获解析错误的监听器。
    """
    def __init__(self):
        self.has_error = False
        self.error_msg = ""

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        self.has_error = True
        self.error_msg = f"Line {line}:{column} {msg}"


def parse_and_count(smt_codes: list):
    """
    批处理函数：接收多个 SMT 代码字符串，返回聚合后的 PCFG 统计信息。
    """
    
    # 全局计数器 (聚合所有样本)
    aggregated_counter = defaultdict(Counter)
    valid_samples = 0
    
    for code in smt_codes:
        if not code or not code.strip():
            continue

        try:
            # 1. 基础组件初始化
            input_stream = InputStream(code)
            lexer = SMTLIBv2Lexer(input_stream)
            token_stream = CommonTokenStream(lexer)
            parser = SMTLIBv2Parser(token_stream)
            
            # 2. 错误处理配置
            parser.removeErrorListeners()
            error_listener = SyntaxErrorCatcher()
            parser.addErrorListener(error_listener)
            
            # 3. 开始解析
            # 检查文法的根节点名称 (通常是 script 或 start)
            if hasattr(parser, 'script'):
                tree = parser.script()
            elif hasattr(parser, 'start'):
                tree = parser.start()
            elif hasattr(parser, 'parse'):
                tree = parser.parse()
            else:
                # 尝试根据 ruleNames 猜第一个规则
                first_rule = parser.ruleNames[0]
                if hasattr(parser, first_rule):
                    tree = getattr(parser, first_rule)()
                else:
                    raise ValueError(f"Cannot find root rule. Available: {parser.ruleNames}")
            
            # 4. 检查是否有语法错误
            if error_listener.has_error:
                # print(f"Skipping invalid sample: {error_listener.error_msg}")
                continue
            
            # 5. 遍历树并计数
            tracker = PCFGTracker(parser)
            walker = ParseTreeWalker()
            walker.walk(tracker, tree)
            
            # 6. 聚合结果
            for lhs, rhs_counts in tracker.rules_counter.items():
                aggregated_counter[lhs].update(rhs_counts)
            
            valid_samples += 1
            
        except Exception as e:
            print(f"Exception during parsing: {e}")
            # import traceback
            # traceback.print_exc()
            continue

    return aggregated_counter, valid_samples

# ==============================================================================
# 单元测试
# ==============================================================================
if __name__ == "__main__":
    # 模拟样本
    sample_1 = """
    (assert (> x 10))
    (check-sat)
    """
    
    sample_2 = """
    (declare-const y Int)
    (assert (< y 5))
    """
    
    # 错误样本
    sample_bad = """
    (assert (> x ))
    """
    
    samples = [sample_1, sample_2, sample_bad]
    
    print("开始解析样本...")
    rules, valid_n = parse_and_count(samples)
    
    print(f"\n有效样本数: {valid_n}/{len(samples)}")
    print("-" * 40)
    print("提取到的规则 (部分):")
    
    # 打印前几个结果看看
    count = 0
    for lhs, counts in rules.items():
        if count > 5: break
        print(f"\nLHS: <{lhs}>")
        total = sum(counts.values())
        for rhs, c in counts.items():
            prob = c / total
            print(f"  -> {rhs:<35} (Count: {c}, Prob: {prob:.2f})")
        count += 1