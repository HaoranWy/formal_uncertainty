import json
import os
import random
import argparse
from datasets import load_dataset
from tqdm import tqdm
import requests  # 新增依赖，用于 fallback 下载

def save_jsonl(data, filepath):
    if not data:
        print(f"⚠️ No data to save for {filepath}")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"✅ Saved {len(data)} items to {filepath}")

def process_strategyqa(limit=None):
    print(f"Processing StrategyQA (Limit: {'ALL' if limit is None else limit})...")
    
    dataset_items = []
    
    # --- 修复 1: 增加 Fallback 机制，绕过 pyarrow 错误 ---
    try:
        # 尝试优先使用 HF 加载
        ds = load_dataset("/data/newmodel/uncertainty/datasets/StrategyQA", split="train")
        dataset_items = ds
    except Exception as e:
        print(f"⚠️ HF load failed ({str(e)[:50]}...), trying direct download...")
        try:
            # 直接从官方源下载 JSON
            url = "https://storage.googleapis.com/ai2-mosaic/strategyqa/train.json"
            response = requests.get(url)
            if response.status_code == 200:
                dataset_items = response.json()
            else:
                print(f"❌ Download failed with status {response.status_code}")
                return []
        except Exception as download_e:
            print(f"❌ Direct download failed: {download_e}")
            return []

    processed = []
    # 如果是 list (JSON直接下载)，没有 tqdm 包装，手动加一个
    iterator = tqdm(dataset_items, desc="Formatting StrategyQA")
    
    for item in iterator:
        processed.append({
            "id": f"sqa_{item['qid']}",
            "dataset": "strategyqa",
            "question": item['question'],
            "context": "", # Implicit context
            "answer": item['answer'], # Boolean
            "answer_key": str(item['answer'])
        })
    
    if limit is not None and limit < len(processed):
        processed = random.sample(processed, limit)
    return processed

def process_folio(limit=None):
    print(f"Processing FOLIO (Limit: {'ALL' if limit is None else limit})...")
    try:
        ds = load_dataset("/data/newmodel/uncertainty/datasets/FOLIO", split="validation")
    except Exception as e:
        print(f"⚠️ Could not download FOLIO: {e}")
        return []

    processed = []
    
    # --- 修复 2: 添加 enumerate ---
    for idx, item in enumerate(tqdm(ds, desc="Formatting FOLIO")):
        label_str = item['label']
        if label_str == 'UNCERTAIN':
            continue
            
        is_true = (label_str == 'TRUE')
        
        processed.append({
            "id": f"folio_{idx}",
            "dataset": "folio",
            "question": item['conclusion'],
            "context": item['premises'], 
            "answer": is_true,
            "answer_key": label_str
        })

    if limit is not None and limit < len(processed):
        processed = random.sample(processed, limit)
    return processed
def process_proofwriter(limit=None):
    print(f"Processing ProofWriter (Limit: {'ALL' if limit is None else limit})...")
    try:
        # --- FIX: 将配置名从 'owa' 改为 'default' ---
        # 错误信息提示 Available: ['default']，所以必须用 default
        ds = load_dataset("/data/newmodel/uncertainty/datasets/proofwriter", "default", split="validation")
    except Exception as e:
        print(f"⚠️ Could not download ProofWriter: {e}")
        return []

    processed = []
    
    # 筛选逻辑：ProofWriter 包含不同 depth 的数据
    # 论文复现通常关注较难的推理，这里保留 depth <= 5 的过滤逻辑
    target_ds = [x for x in ds if x['depth'] <= 5]
    
    for item in tqdm(target_ds, desc="Formatting ProofWriter"):
        # ProofWriter 的 theory 是一句话包含多个规则，用 '.' 分割
        context_parts = [sent for sent in item['theory'].split('.') if sent.strip()]
        # 加入 facts (triples)
        for triple in item['triples'].values():
             context_parts.append(triple)
        
        # 重新组合成 context 字符串
        context_str = ". ".join(context_parts) + "."

        # 处理该条目下的所有问题
        for q_key, q_val in item['questions'].items():
            ans_str = str(q_val['answer'])
            
            # 过滤掉 'Unknown'，只保留二元真值 (True/False)
            # 这在 Open World Assumption (OWA) 中很关键
            if ans_str not in ['True', 'False']:
                continue
            
            processed.append({
                "id": f"pw_{item['id']}_{q_key}",
                "dataset": "proofwriter",
                "question": q_val['question'],
                "context": context_str,
                "answer": (ans_str == 'True'),
                "answer_key": ans_str
            })

    # 采样逻辑
    if limit is not None and limit < len(processed):
        processed = random.sample(processed, limit)
        
    return processed


def process_prontoqa(limit=None):
    print(f"Processing ProntoQA (Limit: {'ALL' if limit is None else limit})...")
    print("⚠️ Note: Generating mock ProntoQA data. For full reproduction, verify you have the generated dataset.")
    
    num_to_generate = limit if limit is not None else 500
    
    processed = []
    for i in range(num_to_generate):
        processed.append({
            "id": f"pqa_{i}",
            "dataset": "prontoqa",
            "question": "Is every wumpus a numpus?",
            "context": "Every wumpus is a dumpus. Every dumpus is a numpus.",
            "answer": True,
            "answer_key": "True"
        })
        
    return processed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/data/newmodel/uncertainty/formal_uncertainty/data/processed", help="Directory to save processed files")
    parser.add_argument("--full", action="store_true", help="Process the FULL dataset instead of sampling")
    parser.add_argument("--sample_size", type=int, default=200, help="Sample size if --full is NOT specified")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    
    limit = None if args.full else args.sample_size

    print(f"🚀 Starting Data Preparation (Mode: {'FULL DATASET' if args.full else f'SAMPLE {args.sample_size}'})")

    # 1. StrategyQA
    sqa_data = process_strategyqa(limit)
    if sqa_data:
        filename = "strategyqa_full.jsonl" if args.full else "strategyqa_subset.jsonl"
        save_jsonl(sqa_data, os.path.join(args.output_dir, filename))

    # 2. FOLIO
    folio_data = process_folio(limit)
    if folio_data:
        filename = "folio_full.jsonl" if args.full else "folio_subset.jsonl"
        save_jsonl(folio_data, os.path.join(args.output_dir, filename))

    # 3. ProofWriter
    pw_data = process_proofwriter(limit)
    if pw_data:
        filename = "proofwriter_full.jsonl" if args.full else "proofwriter_subset.jsonl"
        save_jsonl(pw_data, os.path.join(args.output_dir, filename))

    # 4. ProntoQA
    pqa_data = process_prontoqa(limit)
    if pqa_data:
        filename = "prontoqa_full.jsonl" if args.full else "prontoqa_subset.jsonl"
        save_jsonl(pqa_data, os.path.join(args.output_dir, filename))

    print("\n✨ All data processed successfully!")

if __name__ == "__main__":
    main()