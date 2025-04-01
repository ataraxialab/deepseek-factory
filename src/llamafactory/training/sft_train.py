
import os
from dataclasses import dataclass


from unsloth import FastLanguageModel, PatchFastRL
from trl import SFTConfig, SFTTrainer, ModelConfig, TrlParser

from transformers.trainer_utils import get_last_checkpoint
import json
from datasets import Dataset

global system_prompt
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. "
    "In the answer, only output the calculated number and yes or no, without including the process or any other explanations."
)
system_prompt = SYSTEM_PROMPT

@dataclass
class ModelScriptArguments():
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    lora_rank: int = 32
    # lora_alpha: int = 32
    # gpu_memory_utilization: float = 0.7
    random_state: int = 3047
    dataset: str = "/mnt/AINAS1/Models/Meta-Llama-3.1-8B-Instruct"
    system_prompt: str = SYSTEM_PROMPT


########################
# # Setup logging
# ########################
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# handler.setFormatter(
#     logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# )
# logger.addHandler(handler)

def make_conversation(example):
    return {
        "messages": [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': example["question"]}, # q1
            {'role': 'assistant', 'content': "<think>" + example["thought"] + "</think>"+ "<answer>"+example["answer"]+"</answer>"}
        ]
    }

def get_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf8") as f:
        infs = json.load(f)
    
    data = Dataset.from_dict({
        "question": [item["question"] for item in infs],
        "thought": [item["thought"] for item in infs],
        "answer": [item["gt"] for item in infs]
    })
    
    data = data.map(make_conversation)
    
    return data
    
    
    

def get_checkpoint(training_args: SFTConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def main(script_args, training_args, model_args):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_args.model_name_or_path,
        max_seq_length = training_args.max_seq_length,
        load_in_4bit = False, #
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = script_args.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = model_args.lora_alpha,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = script_args.random_state,
        max_seq_length = training_args.max_seq_length,
        use_rslora =  False,
        loftq_config = None
    )

    # logger.info(f"Model parameters {model_args}")
    # logger.info(f"Training/evaluation parameters {training_args}")

    train_data = get_dataset(script_args.dataset)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        args = training_args,
        train_dataset = train_data,
        
    )
    
    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    # last_checkpoint = get_checkpoint(training_args)
    # if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #     logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # # Train the model
    # logger.info(
    #     f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    # )
    # train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    train_result = trainer.train()
    
    # model.save_lora("sft_llama_3500")
    
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_data)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # logger.info("*** Training complete ***")

    # ##################################
    # # Save model and create model card
    # ##################################

    # logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    # logger.info(f"Model saved to {training_args.output_dir}")
    # # training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    # tokenizer.save_pretrained(training_args.output_dir)
    # logger.info(f"Tokenizer saved to {training_args.output_dir}")

    
def run_sft():
    parser = TrlParser((ModelScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if script_args.system_prompt:
        system_prompt = script_args.system_prompt
    print(f"system prompt: {system_prompt}")
    main(script_args, training_args, model_args)

if __name__ == "__main__":
    run_sft()
    
