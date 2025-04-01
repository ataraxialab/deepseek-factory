# import os
# from unsloth import FastLanguageModel, PatchFastRL
# PatchFastRL("GRPO", FastLanguageModel)
import torch
from peft import PeftModel
import os
from transformers import AutoModelForCausalLM
from safetensors import safe_open
from transformers import AutoTokenizer
import argparse

def run_merge(model_path_ori, save_path, lora_path):
    print(f" merged with {model_path_ori} and {lora_path} to {save_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path_ori)
    model_lora = PeftModel.from_pretrained(model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path_ori)

    res = model_lora.merge_and_unload()
    res.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--model_path_ori", default="/unsloth/hudong/unsloth_train/sft_train/output_ckpt_res", type=str)
#    parser.add_argument("--save_path", default="/unsloth/hudong/unsloth_train/grpo_train/ckpt_sft_rl", type=str)
#    parser.add_argument("--lora_path", default="/unsloth/hudong/unsloth_train/grpo_train/output", type=str)
#    args = parser.parse_args()
#    run_merge(args.model_path_ori, args.save_path, args.lora_path)

