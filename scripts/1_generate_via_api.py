import sys
import os
import json
import asyncio
import argparse
from tqdm.asyncio import tqdm_asyncio

# 将项目根目录加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm.generator import APIInterface 
from src.llm.prompts import PromptGenerator

async def process_batch(interface, prompt_gen, items, dataset_name, n_samples, sem):
    async def process_one(item):
        async with sem:
            q_id = item['id']
            question = item['question']
            context = item.get('context', '')
            
            result = {
                "id": q_id,
                "question": question,
                "context": context,
                "dataset": dataset_name,
                "ground_truth": item.get('answer'),
                "samples": []
            }
            
            try:
                # 1. Text Baseline
                text_msgs = prompt_gen.get_prompt(dataset_name, question, context, use_cot=True)
                text_res = await interface.generate(messages=text_msgs, n=1, temperature=0.0, max_tokens=1024)
                text_output = text_res[0] if text_res else ""

                # 2. SMT Samples
                smt_msgs = prompt_gen.get_prompt(dataset_name, question, context, use_cot=False)
                smt_res_list = await interface.generate(messages=smt_msgs, n=n_samples, temperature=1.0, max_tokens=1024)
                
                for idx, code in enumerate(smt_res_list):
                    result["samples"].append({
                        "sample_idx": idx,
                        "text_output": text_output,
                        "smt_code": code
                    })
                return result

            except Exception as e:
                # 打印详细错误，如果是 401 认证错误，这会帮助定位
                print(f"❌ Error ID {q_id}: {str(e)}")
                return None

    tasks = [process_one(item) for item in items]
    results = await tqdm_asyncio.gather(*tasks)
    return [r for r in results if r is not None]

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=50)
    # --- FIX: 新增 api_key 参数 ---
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API Key for vLLM") 
    args = parser.parse_args()

    base_url = args.url.rstrip('/')

    print(f"🚀 Connecting to vLLM at {base_url}...")
    print(f"📋 Model: {args.model} | API Key: {args.api_key}")

    # --- FIX: 使用传入的 api_key ---
    interface = APIInterface(
        model_name=args.model,
        api_key=args.api_key, 
        base_url=base_url
    )
    prompt_gen = PromptGenerator()

    with open(args.input, 'r') as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} items.")

    sem = asyncio.Semaphore(args.concurrency)
    
    results = await process_batch(interface, prompt_gen, data, args.dataset, args.n_samples, sem)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    
    print(f"✅ Saved {len(results)} generations to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())