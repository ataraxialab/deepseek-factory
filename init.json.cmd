{
  "data_mount_dir": "/openr1_data",
  "system_prompt": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\nThe assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nThe reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.\n<think> reasoning process here </think><answer> answer here </answer>.\nIn the answer, only output the calculated number and yes or no, without including the process or any other explanations.",
  "models": {
    "Llama/Meta-Llama-3.1-8B-Instruct": "/openr1_data/Llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen2.5-1.5B-Instruct": "/openr1_data/Qwen/Qwen2.5-1.5B-Instruct"
  },
  "dataprocess": {
    "cmd": "python src/llamafactory/training/distill.py",
    "hp": {
      "dataset_src_path": "",
      "dataset_dst_path": "",
      "api_key": "",
      "base_url": "",
      "model": "",
      "system_prompt": "",
      "output_dir": ""
    }
  },
  "eval": {
    "cmd": "python src/llamafactory/training/inference.py",
    "hp": {
      "model_name_or_path": "",
      "preprocessing_num_workers": 16,
      "finetuning_type": "lora",
      "quantization_method": "bitsandbytes",
      "template": "qwen",
      "flash_attn": "auto",
      "eval_dataset": "",
      "cutoff_len": 1024,
      "max_samples": 100000,
      "per_device_eval_batch_size": 83,
      "predict_with_generate": true,
      "max_new_tokens": 512,
      "top_p": 0.7,
      "temperature": 0.95,
      "output_dir": "",
      "trust_remote_code": true,
      "do_predict": true
    }
  },
  "sft": {
    "cmd": "python src/llamafactory/training/sft_train.py",
    "hp": {
      "max_seq_length": 4096,
      "lora_rank": 32,
      "lora_alpha": 32,
      "random_state": 3047,
      "dataset": "/data/FINQA_distill/distill_correct.json",
      "model_name_or_path": "/models/Meta-Llama-3.1-8B-Instruct",
      "learning_rate": 1e-5,
      "weight_decay": 0.0001,
      "warmup_ratio": 0.1,
      "bf16": true,
      "fp16": false,
      "logging_strategy": "steps",
      "logging_steps": 1,
      "per_device_train_batch_size": 8,
      "gradient_accumulation_steps": 4,
      "num_train_epochs": 3,
      "save_steps": 100,
      "seed": 3407,
      "max_grad_norm": 0.1,
      "report_to": "tensorboard",
      "output_dir": "output",
      "save_strategy": "steps"
    }
  },
  "rl": {
    "cmd": "python src/llamafactory/training/grpo_train.py",
    "hp": {
      "max_seq_length": 4096,
      "lora_rank": 32,
      "lora_alpha": 32,
      "gpu_memory_utilization": 0.7,
      "random_state": 3407,
      "dataset": "/data/FINQA_json/train.json",
      "model_name_or_path": "/data/sft_train/output_ckpt_res",
      "use_vllm": true,
      "learning_rate": 1e-6,
      "adam_beta1": 0.9,
      "adam_beta2": 0.99,
      "weight_decay": 0.1,
      "warmup_ratio": 0.1,
      "bf16": true,
      "fp16": false,
      "lr_scheduler_type": "cosine",
      "optim": "paged_adamw_8bit",
      "logging_strategy": "steps",
      "logging_steps": 1,
      "per_device_train_batch_size": 1,
      "gradient_accumulation_steps": 8,
      "num_generations": 3,
      "max_prompt_length": 2048,
      "num_train_epochs": 1,
      "save_steps": 50,
      "max_grad_norm": 0.1,
      "output_dir": "output",
      "save_strategy": "steps"
    }
  }
}
