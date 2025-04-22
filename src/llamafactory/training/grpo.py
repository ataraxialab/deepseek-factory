# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from llamafactory.training.merge import run_merge

from .open_r1.configs import GRPOConfig, GRPOScriptArguments
from .open_r1.rewards import get_reward_funcs
from .open_r1.utils import get_tokenizer
from .open_r1.utils.callbacks import get_callbacks
from .open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config
from dataclasses import dataclass
from datasets import Dataset
import json

global system_prompt
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. "
    "In the answer, only output the calculated number and yes or no, without including the process or any other explanations."
)
system_prompt = SYSTEM_PROMPT

logger = logging.getLogger(__name__)

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

def make_conversation_file(example):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
        ],
        'solution': example["answer"],
    }

def get_dataset(dataset_path):
    infs = get_list_with_fields(dataset_path, {"question", "answer"})
    if not infs:
        return None

    data = Dataset.from_list(infs)
    data = data.map(make_conversation_file)
    
    return data  # Returns Dataset on success

print(f"num_visible_gpus: {num_visible_gpus}")

def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    # Load the dataset
    custom_dataset = False
    if os.path.isdir(script_args.dataset_name):
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        dataset = dataset.map(make_conversation) 
        print(f"using local hf dataset {script_args.dataset_name}")
    else:
        dataset = get_dataset(script_args.dataset_name)
        if not dataset:
            dataset = load_dataset("json", name=script_args.dataset_name)
            if dataset:
                dataset = dataset.map(make_conversation)
            print(f"using json dataset {script_args.dataset_name}")
        else:
            custom_dataset = True
            print(f"using distilled dataset {script_args.dataset_name}")

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    #dataset = dataset.map(make_conversation_file)

    if not custom_dataset:
        for split in dataset:
            if "messages" in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    if custom_dataset:
        eval_dataset = dataset
    else:
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset if custom_dataset else dataset[script_args.dataset_train_split],
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

def do_merge(model_name_or_path: str):
    adapter_file = os.path.join(model_name_or_path, "adapter_config.json")
    merge_dir = os.path.join(model_name_or_path, "merged")

    if os.path.isfile(adapter_file):
        print(f"Found {adapter_file}.")
        with open(adapter_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            model_path_ori = config.get("base_model_name_or_path", "")

        if model_path_ori and not os.path.isdir(merge_dir):
            print(f"Found model_path_ori of {model_path_ori} and {merge_dir} does not exist, running merge")
            run_merge(model_path_ori, merge_dir, model_name_or_path)
            print(f"Change input from {model_name_or_path} to {merge_dir} after merged")
            model_name_or_path = merge_dir
        elif model_path_ori and os.path.isdir(merge_dir):
            print(f"Change input from {model_name_or_path} to {merge_dir}")
            model_name_or_path = merge_dir
        else:
            print(f"{model_name_or_path} no changed")

    return model_name_or_path
def run_grpo():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    global system_prompt
    if training_args.system_prompt:
        system_prompt = training_args.system_prompt

    model_args.model_name_or_path = do_merge(model_args.model_name_or_path)
    #if script_args.dataset_name is not None and not os.path.isdir(script_args.dataset_name):
    #    from .grpo_file import grpo_function
    #    grpo_function(model_args, script_args, training_args)
    #else:
    #    main(script_args, training_args, model_args)
    main(script_args, training_args, model_args)

if __name__ == "__main__":
    run_grpo()

