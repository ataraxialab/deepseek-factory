from openai import OpenAI
import json
import os
import yaml
import sys
import re

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. "
    "In the answer, only output the calculated number and yes or no, without including the process or any other explanations."
)

def run_distill():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", type=str)
    args = parser.parse_args()

    if not args.config: 
        print(f"Usage: python {sys.argv[0]} --config <config_file>")
        return 1
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_src_path = config.get("dataset_src_path", "")
    dataset_dst_path = config.get("dataset_dst_path", "")
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "")
    model = config.get("model", "")
    system_prompt = config.get("system_prompt", "")
    if not system_prompt:
        system_prompt = SYSTEM_PROMPT

    if not dataset_src_path or not dataset_dst_path or not api_key or not base_url or not model:
        print("need dataset_src_path, dataset_dst_path, api_key, base_url, model")
        return 1

    if not os.path.exists(dataset_dst_path):
        os.mkdir(dataset_dst_path)

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"system prompt: {system_prompt}")
    with open(dataset_src_path, "r", encoding="utf8") as f:
        data = json.load(f)
    name, ext = os.path.splitext(dataset_src_path)
    out_file = f"{name}_distill{ext}"

    outs = []
    for i in range(len(data)):
        print(i)
        question = data[i]["question"]
        gt = data[i]["answer"]
        
        print("question:", question)
        print("answer:", gt)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
       
        reasoning_content = ""
        answer_content = ""
        for chunk in response:
            if getattr(chunk.choices[0].delta, "reasoning_content", None):
                reasoning_chunk = chunk.choices[0].delta.reasoning_content
                if reasoning_chunk:
                    print(reasoning_chunk,end="",flush=True)
                    reasoning_content += reasoning_chunk
            if getattr(chunk.choices[0].delta, "content", None):
                answer_chunk = chunk.choices[0].delta.content
                if answer_chunk:
                    print(answer_chunk,end="",flush=True)
                    answer_content += answer_chunk

        m = re.search(r'<answer>(.*?)</answer>', answer_content)
        if m:
            answer_content = m.group(1).strip()
        tmp = {"question": question, "gt": gt, "thought": reasoning_content, "answer": answer_content, "index": i}
        outs.append(tmp)

    with open(out_file, "w", encoding="utf8") as f:
        json.dump(outs, f)
        f.write("\n")

    print(f"output file: {out_file}")

    import time
    time.sleep(5)
    return 0

# Add this block to make it executable as a script or importable as a module
if __name__ == "__main__":
    run_distill()
