import sys
from antlr4 import *

# 确保 Python 能找到生成的 Parser 文件
sys.path.append('./src/parsing') 

from src.parsing.SMTLIBv2Lexer import SMTLIBv2Lexer
from src.parsing.SMTLIBv2Parser import SMTLIBv2Parser

def test_parsing():
    # 这是一个最简单的 SMT-LIB 示例代码
    smt_code = """
    (set-logic QF_LIA)
    (declare-const x Int)
    (assert (> x 0))
    (check-sat)
    """
    
    # 1. 创建输入流
    input_stream = InputStream(smt_code)
    
    # 2. 词法分析 (Lexer)
    lexer = SMTLIBv2Lexer(input_stream)
    stream = CommonTokenStream(lexer)
    
    # 3. 语法分析 (Parser)
    parser = SMTLIBv2Parser(stream)
    
    # 4. 构建语法树 (Parse Tree)
    # 注意：'start' 是文法的起始规则名，在 SMTLIBv2.g4 中可能是 'script' 或 'start'
    # 请打开 .g4 文件确认第一条规则的名字，如果是 script，这里就调 parser.script()
    try:
        tree = parser.script() 
        print("✅ 解析成功！生成的 Parse Tree 对象:", tree)
        print("LISP 风格树结构:", tree.toStringTree(recog=parser))
    except Exception as e:
        print("❌ 解析失败:", e)

if __name__ == "__main__":
    test_parsing()