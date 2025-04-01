import os
import re
import json
import sys
import yaml

from vllm import LLM, SamplingParams
from tqdm import tqdm

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. "
    "In the answer, only output the calculated number and yes or no, without including the process or any other explanations."
)

def run_inference():
    import argparse
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

    if not eval_dataset or not output_dir or not model_name_or_path:
        print("eval_dataset or output_dir or model_name_or_path is empty")
        return 1

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name_or_path,
        max_seq_length = 4096,
        load_in_4bit = False, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = 32,
        gpu_memory_utilization = 0.7, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
            model,
            r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ], # Remove QKVO if out of memory
            lora_alpha = 32,
            use_gradient_checkpointing = "unsloth", # Enable long context finetuning
            random_state = 3407,
        )

    # SYSTEM_PROMPT = (
    #     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    #     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    #     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    #     "<think> reasoning process here </think><answer> answer here </answer>"
    # )

    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 4096,
        seed = 3407
    )


    with open(eval_dataset, "r", encoding="utf8") as f:
        test_data = json.load(f)

    true_total = 0
    error = []
    correct = []
    for i in tqdm(test_data):
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': i["question"]},  # q1
        ]
        gt = i["answer"]

        text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        
        # res = model.fast_generate([text], sampling_params = sampling_params,
        #                                 lora_request = None )
        res = model.fast_generate(text, sampling_params = sampling_params,
                                        lora_request = None )
        res = res[0].outputs[0].text

        match = re.search(r'<answer>(.*?)</answer>', res)
        reward = 0
        if match:
            answer_content = match.group(1).strip()  # 提取并去除前后空格
            if gt == "yes" or gt == "no":
                answer = answer_content.split(" ")
                for j in answer:
                    if j.lower() == gt:
                        reward = 1.0
                        break
            else:
                if len(answer_content.split(" ")) >= 5:
                    reward = 0.0
                else:
                    answer_content = answer_content.replace("million", "").replace("billion", "").replace(",", "")
                    try:
                        if answer_content == gt:
                            reward = 1.0
                        elif "%" in answer_content:
                            answer = float(answer_content.replace("%", ""))
                            answer = answer / 100
                            if abs(float(gt) - answer) < 0.001:
                                reward = 1.0
                        elif "$" in answer_content:
                            answer = answer_content.replace("$", "")
                            if abs(float(answer) - float(gt)) < 0.001:
                                reward = 1.0
                        elif abs(float(answer_content) - float(gt)) < 0.001:
                            reward = 1.0
                    except:
                        pass
        if reward == 1.0:
            true_total += 1
            correct.append({"q":i["question"], "gt":gt, "answer": res})
        else:
            error.append({"q":i["question"], "gt":gt, "answer": res})
        
    print("accuracy:", true_total / len(test_data))

    errfile = os.path.join(output_dir, "error.json")
    correctfile = os.path.join(output_dir, "correct.json")
    with open(errfile, "w", encoding="utf8") as f:
        json.dump(error, f, ensure_ascii=False, indent=4)

    with open(correctfile, "w", encoding="utf8") as f:
        json.dump(correct, f, ensure_ascii=False, indent=4)
      
    print(f"-- output: {correctfile}, {errfile}")

if __name__ == "__main__":
    run_inference()
