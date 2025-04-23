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

def filter(in_file: str, dataset_dst_path: str):
    true_total = 0
    correct = []
    error = []
    with open(in_file, "r", encoding="utf8") as f:
        infs = f.readlines()
    for i in range(len(infs)):
        t = json.loads(infs[i])
        #index = t["index"]
        res = t["answer"]
        thought = t["thought"]
        question = t["question"]
        gt = t["gt"]
        
        match = re.search(r'<answer>(.*?)</answer>', res)
        reward = 0
        answer = ""
        if match:
            answer = match.group(1).strip()  # 提取并去除前后空格
            if gt == "yes" or gt == "no":
                answer = answer.split(" ")
                for j in answer:
                    if j.lower() == gt:
                        reward = 1.0
                        break
            else:
                if len(answer.split(" ")) >= 5:
                    reward = 0.0
                else:
                    answer = answer.replace("million", "").replace("billion", "").replace(",", "")
                    try:
                        if answer== gt:
                            reward = 1.0
                        elif "%" in answer:
                            answer = float(answer.replace("%", ""))
                            answer = answer / 100
                            if abs(float(gt) - answer) < 0.001:
                                reward = 1.0
                        elif "$" in answer:
                            answer = answer.replace("$", "")
                            if abs(float(answer) - float(gt)) < 0.001:
                                reward = 1.0
                        elif abs(float(answer) - float(gt)) < 0.001:
                            reward = 1.0
                    except:
                        pass
        if reward == 1.0:
            true_total += 1
            correct.append({"question":question, "gt":gt, "answer": answer, "thought":thought, "index": i})
        else:
            error.append({"question":question, "gt":gt, "answer": answer, "thought":thought, "index": i})
        
    print("accuracy:", true_total / len(infs))

    errfile = os.path.join(dataset_dst_path, "error.json")
    correctfile = os.path.join(dataset_dst_path, "correct.json")
    with open(errfile, "w", encoding="utf8") as f:
        json.dump(error, f, ensure_ascii=False, indent=4)

    with open(correctfile, "w", encoding="utf8") as f:
        json.dump(correct, f, ensure_ascii=False, indent=4)

    print(f"-- output: {correctfile}, {errfile}")

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
    out_file = os.path.join(dataset_dst_path, "_distill.json")
    # remove out_file
    if os.path.exists(out_file):
        try:
            os.remove(out_file)
        except:
            pass

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
            # 获取思考过程
            if not hasattr(chunk.choices[0].delta, "reasoning_content"):
                print(f"---- no reasoning content: {chunk.choices[0]} ",end="",flush=True)
                break 
            reasoning_chunk = chunk.choices[0].delta.reasoning_content
            # 获取回复
            answer_chunk = chunk.choices[0].delta.content
            # 如果思考过程不为空，则打印思考过程
            if reasoning_chunk != "" and reasoning_chunk != None:
                print(reasoning_chunk,end="",flush=True)
                reasoning_content += reasoning_chunk
            # 如果回复不为空，则打印回复。回复一般会在思考过程结束后返回
            elif answer_chunk != "" and answer_chunk != None:
                print(answer_chunk,end="",flush=True)
                answer_content += answer_chunk
                
        tmp = {"question": question, "gt": gt, "thought": reasoning_content, "answer": answer_content, "index": i}
        with open(out_file, "a", encoding="utf8") as f:
            json.dump(tmp, f)
            f.write("\n")

    filter(out_file, dataset_dst_path)

    import time
    time.sleep(5)
    return 0

# Add this block to make it executable as a script or importable as a module
if __name__ == "__main__":
    run_distill()
