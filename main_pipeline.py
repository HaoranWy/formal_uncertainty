import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
import warnings

# 引入我们之前写好的模块
from src.llm.prompts import PromptGenerator  # (Step 2.1)
from src.parsing.pcfg_builder import parse_and_count # (Step 3.1)
from src.parsing.pcfg_estimator import PCFGEstimator # (Step 3.2)
from src.metrics.calculator import PCFGMetricCalculator # (Step 3.3)
from src.evaluation.labeler import correctness_labeler # (Step 4.1)
from src.metrics.consistency import add_self_consistency_scores # (Step 4.x)
import time
# 忽略一些科学计算的 RuntimeWarning
warnings.filterwarnings('ignore')

class Pipeline:
    def __init__(self, input_file, output_file, dataset_name, sample_size=100):
        self.input_file = input_file
        self.output_file = output_file
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        
        # 初始化核心组件
        self.labeler = correctness_labeler()
        
        # 检查点管理：读取已处理的问题ID
        self.processed_ids = set()
        if os.path.exists(output_file):
            try:
                existing_df = pd.read_csv(output_file)
                if 'question_id' in existing_df.columns:
                    self.processed_ids = set(existing_df['question_id'].unique().astype(str))
                print(f"🔄 发现已存在的输出文件，已跳过 {len(self.processed_ids)} 个问题。")
            except Exception as e:
                print(f"⚠️ 读取检查点失败，将覆盖或重新开始: {e}")

    def load_data(self):
        """
        加载输入数据 (JSONL格式)
        假设每行是一个问题，包含字段: id, question, ground_truth
        以及预生成的 samples: [{'smt_code':..., 'text_output':...}, ...]
        """
        data = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 转换为字符串以匹配 processed_ids
                if str(item['id']) not in self.processed_ids:
                    data.append(item)
        return data

    def process_single_question(self, item):
        """
        核心处理逻辑：处理单个问题及其 N 个样本
        """
        q_id = str(item['id'])
        ground_truth = item['ground_truth']
        samples = item.get('samples', [])  # 预生成的 samples 列表
        
        # 如果样本不够，跳过 (或者在这里调用 LLM API 实时生成)
        if not samples:
            return []

        # === Phase 1: Row-Level Processing (Z3 & Labeling) ===
        processed_rows = []
        valid_smt_codes_for_pcfg = [] # 用于构建 PCFG 的有效代码
        
        for i, sample in enumerate(samples):
            smt_code = sample.get('smt_code', "")
            text_output = sample.get('text_output', "")
            
            # 调用 Labeler 评估正确性
            # 构造 labeler 需要的格式
            label_input = {
                "id": f"{q_id}_{i}",
                "ground_truth": ground_truth,
                "text_output": text_output,
                "smt_code": smt_code
            }
            
            res = self.labeler.process_sample(label_input)
            
            # 保存该样本的基础信息
            row = {
                "question_id": q_id,
                "sample_idx": i,
                "ground_truth": ground_truth,
                "smt_code": smt_code, # 可选：为了 CSV 瘦身可以不存代码
                "text_output": text_output,
                
                # Labeler 结果
                "smt_status": res["smt_executed_status"],
                "smt_pred_bool": res["smt_pred_bool"],
                "smt_is_correct": res["smt_is_correct"],
                "text_pred_bool": res["text_pred_bool"],
                "text_is_correct": res["text_is_correct"],
                "consistency_smt_text": res["consistency_smt_text"]
            }
            processed_rows.append(row)
            
            # 收集用于 PCFG 的代码 (仅收集非空代码，Labeler 内部不做语法检查，Parser 会做)
            if smt_code and smt_code.strip():
                valid_smt_codes_for_pcfg.append(smt_code)

        # === Phase 2: Question-Level Processing (PCFG Construction) ===
        # 使用 N 个样本构建 1 个 PCFG
        # 1. Parse & Count
        rules_counter, valid_parse_count = parse_and_count(valid_smt_codes_for_pcfg)
        
        # 2. Estimate Probabilities (MLE + Laplace)
        estimator = PCFGEstimator(rules_counter, alpha=1.0)
        
        # 3. Calculate PCFG Metrics
        # 注意：如果所有样本都解析失败，metrics 将由 Calculator 返回默认零值
        pcfg_calc = PCFGMetricCalculator(estimator.pcfg_probs, start_symbol="script") # 视你的文法而定
        pcfg_metrics = pcfg_calc.compute_all()
        
        # 添加解析成功率作为额外的 meta-feature
        pcfg_metrics['parse_success_rate'] = valid_parse_count / len(samples) if samples else 0
        
        # === Phase 3: Broadcasting & Merging ===
        # 将 PCFG 指标广播给该问题的每一行
        final_rows = []
        for row in processed_rows:
            # 合并两个字典
            merged_row = {**row, **pcfg_metrics}
            final_rows.append(merged_row)
            
        return final_rows

    def save_results(self, rows):
        """
        增量写入 CSV
        """
        if not rows:
            return
            
        df = pd.DataFrame(rows)
        
        # 如果文件不存在，写入 header；如果存在，追加模式 (mode='a') 不写 header
        need_header = not os.path.exists(self.output_file)
        
        df.to_csv(self.output_file, mode='a', header=need_header, index=False, encoding='utf-8')

    def run(self):
        """
        主执行循环
        """
        print(f"🚀 Starting pipeline for dataset: {self.dataset_name}")
        data_to_process = self.load_data()
        print(f"📝 Loaded {len(data_to_process)} questions to process.")
        
        if len(data_to_process) == 0:
            print("🎉 No new data to process. All done!")
            return

        # 使用 tqdm 显示进度条
        pbar = tqdm(data_to_process, desc="Processing Questions")
        
        batch_buffer = [] # 可以在内存里攒几个问题再写，这里为了安全每题必写
        
        for item in pbar:
            try:
                # 处理单个问题
                rows = self.process_single_question(item)
                
                if rows:
                    # 暂时转换成 DataFrame 处理一致性 (虽然 Consistency 需要 GroupBy，
                    # 但这里只针对单题 N 个样本计算 Consistency 也是可以的，
                    # 因为 GroupBy('question_id') 在单题数据下就是它自己)
                    
                    # 计算 Self-Consistency (Step 4.x)
                    df_temp = pd.DataFrame(rows)
                    df_temp = add_self_consistency_scores(df_temp)
                    
                    # 存盘
                    self.save_results(df_temp.to_dict('records'))
                    
            except Exception as e:
                error_id = item.get('id', 'unknown')
                print(f"\n❌ Error processing question {error_id}: {str(e)}")
                # 可以选择记录错误日志，而不是中断程序
                with open(f"output/logs/error_log_{time.time()}.txt", "a") as ef:
                    ef.write(f"{error_id}: {str(e)}\n")
                continue

        print(f"\n✅ Pipeline finished. Results saved to {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grammars of Uncertainty - Main Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with generations")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("--dataset", type=str, default="strategyqa", help="Dataset name")
    
    args = parser.parse_args()
    
    pipeline = Pipeline(
        input_file=args.input,
        output_file=args.output,
        dataset_name=args.dataset
    )
    
    pipeline.run()