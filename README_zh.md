<h3 align="center">
    使用零代码Web UI进行数据处理和微调大模型
</h3>

## 目录

- [项目特色](#项目特色)
- [更新日志](#更新日志)
- [如何使用](#如何使用)

## 项目特色

- **多种模型**：目前支持Deepseek，Qwen，Llama，可通过配置逐步添加其他模型。
- **集成方法**：基于Unsloth的（全量）预训练、GRPO强化学习训练，后续逐步添加其他集成方法。
- **简单易用**：开放基础的训练配置参数，其他参数通过后台配置载入，可提前适配不同硬件形态。
- **极速推理**：基于 vLLM 的 OpenAI 风格 API、浏览器界面和命令行接口。
- **国产GPU卡支持**：支持沐曦C500/C550/C280单卡、2卡、4卡、8卡一体机一键式部署训练。

## 更新日志

[25/03/11] 
支持数据分割、基于（Deepseek）的数据蒸馏、基于Unsloth全量Finetune，GRPO强化学习训练和推理
支持动态添加独立的训练或推理Gradio页面，可配置后端执行python脚本或命令
支持根据训练类型或其他条件配置训练参数，并缺省加载预训练权重
支持国产沐曦卡一体机一键式部署训练

## 如何使用

### 安装 DeepseekFactory

https://github.com/ataraxialab/deepseek-factory.git
git checkout dev
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
（对于沐曦卡用户，镜像提供Torch等安装包，使用pip install -r requirements.txt.metax）
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple

```bash
GRADIO_SERVER_PORT=7860 deepseekfactory-cli webui
```

### 构建 Docker

CUDA用户：
```bash
cd docker/docker-cuda/
docker compose up -d
docker exec -it deepseekfactory bash
```

沐曦用户：
```bash
cd docker/docker-metax/
docker compose up -d
docker exec -it deepseekfactory bash
```

### 配置初始化json文件init.json
{
  "data_mount_dir": "/openr1_data",
  "system_prompt": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\nThe assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\nThe reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e.\n<think> reasoning process here </think><answer> answer here </answer>.\nIn the answer, only output the calculated number and yes or no, without including the process or any other explanations.",
  "models": {
    "Llama/Meta-Llama-3.1-8B-Instruct": "/openr1_data/Llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen2.5-1.5B-Instruct": "/openr1_data/Qwen/Qwen2.5-1.5B-Instruct"
  },
  "dataprocess": {
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
除了前三项作为共同参数外，下面每个block基本是针对一种训练类型提供超参数（HP）配置。可根据不同的硬件配置，灵活修改参数类型。比如，如果是针对
一体机，在单卡情况下运行GRPO设置一组参数，2卡，4卡，8卡分别设置另外的参数，则可以按如下的结构来添加，并在代码中使用类似
      args.get("rl", {}).get("2", {}).get("hp", {})的形式来获取所定义的超参配置。
{
    "rl": {
        "1" {
            "hp" :{
            }
        },
        "2" {
            "hp" :{
            }
        },
        "4" {
            "hp" :{
            }
        },
        "8" {
            "hp" :{
            }
        }
    }
}
各字段解释如下：
1） "data_mount_dir": "/openr1_data": 在选择存储或数据集时，缺省的搜索目录。改目录是在docker容器内的目录，而非宿主机的目录。
2） "system_prompt": ""：如果SFT或GRPO或Inference训练使用到prompt，则该promppt内容会缺省填充在界面上，可供修改。
3) 配置已有的模型名称及具体路径。注意，/openr1_data是挂载进到容器的路径。在界面上使用模型的地方，模型名称可供选择，选择后，模型路径会自动填充在界面上。
  "models": {
    "Llama/Meta-Llama-3.1-8B-Instruct": "/openr1_data/Llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen2.5-1.5B-Instruct": "/openr1_data/Qwen/Qwen2.5-1.5B-Instruct"
  },
4) "dataprocess": 数据蒸馏的缺省配置，如果配置，则缺省配置会自动填充在界面上。
5) "eval": 评估的缺省配置，如果配置，则缺省配置会自动填充在界面上。
6) "sft": SFT的缺省配置，如果配置，则缺省配置会自动填充在界面上。
7) "rl": GRPO  reinforcement learning的缺省配置，如果配置，则缺省配置会自动填充在界面上。
需要注意的是，在每种训练类型中，如果添加了cmd字段（参考init.json.cmd），则在点击运行时，会挑选配置的命令执行，这样就提供了一定的灵活性，
比如，配置的命令可以是shell命令，不一定是python执行命令。否则，采用内置命令，由deepseekfactory-cli xxx来启动。这样做是保持一定的简洁性和
兼容性，后续添加其他训练类型时，可以采用相同的方式启动。
