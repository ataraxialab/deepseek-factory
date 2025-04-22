import os
import re
import json
import sys
import yaml

from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. "
    "In the answer, only output the calculated number and yes or no, without including the process or any other explanations."
)

def get_list_with_fields(dataset_path, fields):
    try:
        with open(dataset_path, "r", encoding="utf8") as f:
            infs = json.load(f)
    except:
        return None  # File error or invalid JSON

    if not isinstance(infs, list):
        return None
    
    for item in infs:
        if not isinstance(item, dict) or not fields.issubset(item.keys()):
            return None 
    return infs

def run_inference():
    import argparse
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", type=str)
    args = parser.parse_args()

    if not args.config: 
        print(f"Usage: python {sys.argv[0]} --config <config_file>")
        return 1
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_dataset = config.get("eval_dataset", "")
    output_dir = config.get("output_dir", "")
    model_name_or_path = config.get("model_name_or_path", "")
    system_prompt = config.get("system_prompt", "")
    if not system_prompt:
        system_prompt = SYSTEM_PROMPT

    print(f"===== system prompt: {system_prompt} ==")

    if not eval_dataset or not output_dir or not model_name_or_path:
        print("eval_dataset or output_dir or model_name_or_path is empty")
        return 1

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 4096,
        seed = 3407
    )

    if os.path.isdir(eval_dataset):
        test_data = load_dataset(eval_dataset)
    else:
        test_data = get_list_with_fields(eval_dataset, {"question", "answer"})

    if not test_data:
        print(f"Invalid dataset: {eval_dataset}")
        return 1

    llm = LLM(model=model_name_or_path, gpu_memory_utilization=0.7, enforce_eager=True)

    correct = []
    for i in tqdm(test_data):
        #import torch
        #torch.cuda.empty_cache()
        content = i["question"]
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': content}, 
        ]
        gt = i["answer"]

        res = llm.generate(llm.get_tokenizer().apply_chat_template(messages, tokenize=False, continue_final_message=False), sampling_params)
        res = res[0].outputs[0].text

        answer_content = res
        match = re.search(r'<answer>(.*?)</answer>', res)
        if match:
            answer_content = match.group(1).strip()  # 提取并去除前后空格
        correct.append({"question":content, "gt":gt, "answer": answer_content})
        
    inffile = os.path.join(output_dir, "inference.json")
    with open(inffile, "w", encoding="utf8") as f:
        json.dump(correct, f, ensure_ascii=False, indent=4)
      
    print(f"-- output: {inffile}")

if __name__ == "__main__":
    run_inference()

