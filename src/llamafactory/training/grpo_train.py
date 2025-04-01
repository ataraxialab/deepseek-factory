
import os
import sys
from dataclasses import dataclass
from llamafactory.training import prepare_data
from llamafactory.training.prepare_data import *
from llamafactory.training.reward import *
from llamafactory.training.merge import run_merge

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

# import logging
from transformers.trainer_utils import get_last_checkpoint

@dataclass
class ModelScriptArguments():
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    max_seq_length: int = 512
    lora_rank: int = 32
    gpu_memory_utilization: float = 0.7
    random_state: int = 3047
    dataset: str = "/mnt/AINAS1/Models/Meta-Llama-3.1-8B-Instruct"
    system_prompt: str = prepare_data.SYSTEM_PROMPT


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

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def main(script_args, training_args, model_args):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_args.model_name_or_path,
        max_seq_length = script_args.max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = script_args.lora_rank,
        gpu_memory_utilization = script_args.gpu_memory_utilization, # Reduce if out of memory
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
    )

    # logger.info(f"Model parameters {model_args}")
    # logger.info(f"Training/evaluation parameters {training_args}")
    training_args.max_completion_length = script_args.max_seq_length - training_args.max_prompt_length

    train_data = get_finqa(script_args.dataset)
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            format_reward,
            accuracy_reward
        ],
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
    
    # model.save_lora("grpo_llama_lora_0220")
    
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

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial", "philschmid"]})
    # push to hub if needed
    # if training_args.push_to_hub is True:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub()

    # logger.info("*** Training complete! ***")
    
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

def dorungrpo():

    parser = TrlParser((ModelScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if script_args.system_prompt:
        prepare_data.system_prompt = script_args.system_prompt
    print(f"system prompt: {prepare_data.system_prompt}")
    model_args.model_name_or_path = do_merge(model_args.model_name_or_path)
    main(script_args, training_args, model_args)
    
def run_grpo():
    dorungrpo()

if __name__ == "__main__":
   run_grpo() 
