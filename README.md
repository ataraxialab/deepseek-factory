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

TODO:
- [ ] 除了金融领域，支持其他领域的数据蒸馏支持
- [ ] 添加DPO，KTO，PPO等集成方法
- [ ] 添加其他国产GPU卡支持


## 如何使用

### 安装 DeepseekFactory

```bash
git clone https://github.com/ataraxialab/deepseek-factory.git
git checkout main
pip install -r requirements.txt
（对于沐曦卡用户，镜像提供Torch等安装包，使用pip install -r requirements.txt.metax）
pip install . 
```

启动Web UI：
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
```bash
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
```

除了前三项作为共同参数外，下面每个block基本是针对一种训练类型提供超参数（HP）配置。可根据不同的硬件配置，灵活修改参数类型。比如，如果是针对
一体机，在单卡情况下运行GRPO设置一组参数，2卡，4卡，8卡分别设置另外的参数，则可以按如下的结构来添加，并在代码中使用类似
      args.get("rl", {}).get("2", {}).get("hp", {})的形式来获取所定义的超参配置。
```bash
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
```

各字段解释如下：

1） "data_mount_dir": "/openr1_data": 在选择存储或数据集时，缺省的搜索目录。改目录是在docker容器内的目录，而非宿主机的目录。  

2） "system_prompt": ""：如果SFT或GRPO或Inference训练使用到prompt，则该promppt内容会缺省填充在界面上，可供修改。  

3) 配置已有的模型名称及具体路径。注意，/openr1_data是挂载进到容器的路径。在界面上使用模型的地方，模型名称可供选择，选择后，模型路径会自动填充在界面上。  

```bash
  "models": {  
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "/openr1_data/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "Llama/Meta-Llama-3.1-8B-Instruct": "/openr1_data/Llama/Meta-Llama-3.1-8B-Instruct",   
    "Qwen2.5-1.5B-Instruct": "/openr1_data/Qwen/Qwen2.5-1.5B-Instruct"  
  },  
```

4) "dataprocess": 数据蒸馏的缺省配置，如果配置，则缺省配置会自动填充在界面上。  

5) "eval": 评估的缺省配置，如果配置，则缺省配置会自动填充在界面上。  

6) "sft": SFT的缺省配置，如果配置，则缺省配置会自动填充在界面上。  

7) "rl": GRPO  reinforcement learning的缺省配置，如果配置，则缺省配置会自动填充在界面上。  

需要注意的是，在每种训练类型中，如果添加了cmd字段（参考init.json.cmd），则在点击运行时，会挑选配置的命令执行，这样就提供了一定的灵活性，比如，配置的命令可以是shell命令，不一定是python执行命令。否则，采用内置命令，由deepseekfactory-cli xxx来启动。这样做是保持一定的简洁性和兼容性，后续添加其他训练类型时，可以采用相同的方式启动。  

### 数据处理

目前，数据处理支持数据分割和数据蒸馏。

数据分割支持json数组格式，只要输入是json数组即可，对内部格式没有特定要求。

而数据蒸馏依赖于预训练模型的推理能力，需要逐条发送待蒸馏数据给大模型，然后根据特定规则，选择合适的数据进行下一步的训练。

目前蒸馏支持基于金融问答类的数据输入，对于其他类型的数据，后续逐步添加相应的处理逻辑。金融类问答可以基于FINQA_tes数据集，格式如下


```bash
{
    "question": "Please answer the given financial question based on the context.\nContext: table of contents celanese purchases of its equity securities information regarding repurchases of our common stock during the three months ended december 31 , 2017 is as follows : period number of shares purchased ( 1 ) average price paid per share total number of shares purchased as part of publicly announced program approximate dollar value of shares remaining that may be purchased under the program ( 2 ) .\n|period|totalnumberof sharespurchased ( 1 )|averageprice paidper share|total numberof sharespurchased aspart of publiclyannounced program|approximatedollarvalue of sharesremaining thatmay bepurchased underthe program ( 2 )|\n|october 1 - 31 2017|10676|$ 104.10|2014|$ 1531000000|\n|november 1 - 30 2017|924|$ 104.02|2014|$ 1531000000|\n|december 1 - 31 2017|38605|$ 106.36|2014|$ 1531000000|\n|total|50205||2014||\n___________________________ ( 1 ) represents shares withheld from employees to cover their statutory minimum withholding requirements for personal income taxes related to the vesting of restricted stock units . ( 2 ) our board of directors has authorized the aggregate repurchase of $ 3.9 billion of our common stock since february 2008 , including an increase of $ 1.5 billion on july 17 , 2017 . see note 17 - stockholders' equity in the accompanying consolidated financial statements for further information. .\nQuestion: what is the total authorized the aggregate repurchase of common stock since february 2008 including the additional amount authorized in 2017 in billions",
    "answer": "5.4"
},
```
送往大模型蒸馏后，返回的数据格式如下：
```bash
{
    "question": "Please answer the given financial question based on the context.\nContext: interest rate to a variable interest rate based on the three-month libor plus 2.05% ( 2.05 % ) ( 2.34% ( 2.34 % ) as of october 31 , 2009 ) . if libor changes by 100 basis points , our annual interest expense would change by $ 3.8 million . foreign currency exposure as more fully described in note 2i . in the notes to consolidated financial statements contained in item 8 of this annual report on form 10-k , we regularly hedge our non-u.s . dollar-based exposures by entering into forward foreign currency exchange contracts . the terms of these contracts are for periods matching the duration of the underlying exposure and generally range from one month to twelve months . currently , our largest foreign currency exposure is the euro , primarily because our european operations have the highest proportion of our local currency denominated expenses . relative to foreign currency exposures existing at october 31 , 2009 and november 1 , 2008 , a 10% ( 10 % ) unfavorable movement in foreign currency exchange rates over the course of the year would not expose us to significant losses in earnings or cash flows because we hedge a high proportion of our year-end exposures against fluctuations in foreign currency exchange rates . the market risk associated with our derivative instruments results from currency exchange rate or interest rate movements that are expected to offset the market risk of the underlying transactions , assets and liabilities being hedged . the counterparties to the agreements relating to our foreign exchange instruments consist of a number of major international financial institutions with high credit ratings . we do not believe that there is significant risk of nonperformance by these counterparties because we continually monitor the credit ratings of such counterparties . while the contract or notional amounts of derivative financial instruments provide one measure of the volume of these transactions , they do not represent the amount of our exposure to credit risk . the amounts potentially subject to credit risk ( arising from the possible inability of counterparties to meet the terms of their contracts ) are generally limited to the amounts , if any , by which the counterparties 2019 obligations under the contracts exceed our obligations to the counterparties . the following table illustrates the effect that a 10% ( 10 % ) unfavorable or favorable movement in foreign currency exchange rates , relative to the u.s . dollar , would have on the fair value of our forward exchange contracts as of october 31 , 2009 and november 1 , 2008: .\n||october 31 2009|november 1 2008|\n|fair value of forward exchange contracts asset ( liability )|$ 6427|$ -23158 ( 23158 )|\n|fair value of forward exchange contracts after a 10% ( 10 % ) unfavorable movement in foreign currency exchange rates asset ( liability )|$ 20132|$ -9457 ( 9457 )|\n|fair value of forward exchange contracts after a 10% ( 10 % ) favorable movement in foreign currency exchange rates liability|$ -6781 ( 6781 )|$ -38294 ( 38294 )|\nfair value of forward exchange contracts after a 10% ( 10 % ) unfavorable movement in foreign currency exchange rates asset ( liability ) . . . . . . . . . $ 20132 $ ( 9457 ) fair value of forward exchange contracts after a 10% ( 10 % ) favorable movement in foreign currency exchange rates liability . . . . . . . . . . . . . . . . . . . . . . $ ( 6781 ) $ ( 38294 ) the calculation assumes that each exchange rate would change in the same direction relative to the u.s . dollar . in addition to the direct effects of changes in exchange rates , such changes typically affect the volume of sales or the foreign currency sales price as competitors 2019 products become more or less attractive . our sensitivity analysis of the effects of changes in foreign currency exchange rates does not factor in a potential change in sales levels or local currency selling prices. .\nQuestion: what is the the interest expense in 2009?",
    "gt": "3.8",
    "answer": "",
    "thought": "Okay, let's tackle this question. The user is asking for the interest expense in 2009 based on the provided context. Let me start by scanning through the context to find relevant information.\n\nThe context mentions that if LIBOR changes by 100 basis points, the annual interest expense would change by $3.8 million. But the question is about the actual interest expense in 2009, not the change. Wait, is there any direct figure given for the interest expense?\n\nLooking back, the context states the interest rate is tied to the three-month LIBOR plus 2.05%, which was 2.34% as of October 31, 2009. However, the problem here is that the provided data talks about the sensitivity of interest expense to LIBOR changes, not the actual expense. The $3.8 million is the change per 100 basis points movement. Since the question is asking for the interest expense itself, but there's no explicit number given for the 2009 interest expense in the context provided, I might be missing something.\n\nWait, maybe the answer is derived from the given data. If a 100 basis point change leads to a $3.8 million change in annual interest expense, but the actual LIBOR rate in 2009 is 2.34% (which includes the 2.05% spread plus the LIBOR). However, without knowing the base LIBOR rate before the spread, or the principal amount, it's impossible to calculate the exact interest expense. The context doesn't provide the principal or the exact LIBOR rate at that time beyond the spread. The 2.34% mentioned is the total rate (LIBOR plus spread) as of October 31, 2009. But even that, without the principal, we can't compute the expense.\n\nWait, perhaps the question is a trick. Since the user is asking for the interest expense in 2009, but the context only provides the sensitivity (change of $3.8M per 100bps), and the current rate is 2.34% as of October 31, 2009. However, without the principal or the actual LIBOR component, there's no way to calculate the exact figure. The context doesn't give the actual interest expense number. So maybe the answer is that the information isn't provided, but the user might expect a number based on the given data. Wait, but I need to check again.\n\nWait, the 2.34% is the rate as of October 31, 2009, which is the three-month LIBOR plus 2.05%. So if LIBOR was 0.29% (since 0.29% + 2.05% = 2.34%). But again, without knowing the principal amount, the interest expense can't be calculated. The $3.8 million change is for a 100 basis point move, which is 1% change. But the actual expense would require the principal multiplied by the rate. Since the principal isn't given, the answer can't be determined from the context provided. Therefore, the answer is that there's insufficient data to determine the interest expense in 2009. However, the user's instructions say to provide a number and yes/no. But the context doesn't state the actual interest expense. So maybe the answer is not available here. But the user might expect a specific number. Wait, perhaps I'm missing something. Let me check again.\n\nThe context says \"if LIBOR changes by 100 basis points, our annual interest expense would change by $3.8 million.\" But that's the sensitivity, not the actual expense. So the actual expense isn't provided. The answer would be that the interest expense for 2009 isn't stated in the context. But the user might expect me to answer with the information given. Wait, perhaps the question is a trick. Maybe the answer is $3.8 million, but that's the change, not the actual expense. Alternatively, maybe the answer is $3.8 million divided by 1 (since 100 basis points is 1%), so the principal is $380 million. Then the interest expense would be $380M * 2.34% = $8.892 million. But that's making assumptions. However, the problem is that the sensitivity is $3.8M per 1% change. So the delta is $3.8M = principal * 1% (100 basis points). Therefore, principal = $3.8M / 0.01 = $380 million. Then, the interest expense in 2009 would be $380M * 2.34% = $8,892,000. So that's $8.892 million. But the context doesn't mention the principal; this is inferred from the sensitivity. However, maybe that's the approach expected here. The answer would be 8.892 million. Let me check the math again: 380 million * 2.34% = 380,000,000 * 0.0234 = 8,892,000. So $8,892,000. But the problem is that the user might want the answer in millions. So perhaps $8.89 million. But the question is whether this is correct. However, the context doesn't explicitly state the principal, so this is an assumption. But according to financial sensitivity analysis, the dollar impact of a rate change is principal * delta rate. So given that delta rate is 1% (100 bps) leads to $3.8M change, then principal is $380M. Then, the actual interest expense would be principal * current rate. The current rate as of Oct 31, 2009, is 2.34%. So yes, 380M * 2.34% = 8.892M. Therefore, the answer is $8.89 million. But the question is, is this the correct approach? Given the user's context, maybe this is the intended answer. But I'm not 100% sure because the context doesn't explicitly state the principal. However, given that the user wants the answer in the specified format, I think this is the way to go.\n",
    "index": 0
},
```
返回的数据可以送往SFT训练。

### 训练

#### SFT 训练

目前，例子提供了基于FINQA_test的SFT训练，也就是说，训练假设输入是上述蒸馏生成的格式，包含question/answer/thgouth等字段，并转换为如下格式:

```bash
{
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': example["question"]}, # q1
    {'role': 'assistant', 'content': "<think>" + example["thought"] + "</think>"+ "<answer>"+example["answer"]+"</answer>"}
}
```
其中，example就是json中的蒸馏结果。而所需的训练超参配置，参考上述init.json中sft.hp部分。

#### Reinforcement Learning 训练

Reinforcement learning训练可基于SFT训练的结果,并和lora adapter部分做merge合并之后，再对输入数据进行训练。

输入数据仍然需要包含question/answer字段，但是不一定需要thought字段。

训练过程中，会根据输入，转换为如下格式的输入给训练过程：

```bash
{
    "prompt": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["question"]},
    ],
    'solution': example["answer"],
}
```

其中，example就是json中的蒸馏结果。而所需的训练超参配置，参考上述init.json中rl.hp部分。

### 添加新训练类型

目前已经支持了SFT和GRPO（RL），添加新的训练类型，请遵守如下方式：

1）在webui/interface.py中，添加相应的tab，并设置component id，例如

```bash
    with gr.Tab("训练1"):
        engine.manager.add_elems("train1", create_train1_tab(engine))
```

2）在webui/components/下面，添加新的Gradio页面，例如webui/components/train1.py，并暴露create_train1_tab()函数

3) 在上述文件中，把需要暴露的参数存放在input_elements (params)中，这些参数会被写入到training_args.xml中，作为训练参数执行

4）在cli.py中，添加新的处理命令，比如
```bash
    elif command == "train1":
        from .training.train1 import run_train1
        run_train1()
```

其中，run_train1()是新的训练处理文件暴露的入口，参考sft_train.py

5）在training目录下，添加新的训练处理文件，比如train1.py，并暴露run_train1()入口，供cli.py调用。

6）在init.json中，添加相应的训练超参和执行命令

添加超参
```bash
{
  "train1": {
    "cmd": "xxxx",
    "hp": {
      "max_seq_length": 4096,
      "lora_rank": 32,
  }
}
```

如果添加了cmd，那么在运行时，会执行cmd命令，否则，会执行上述的run_train1()命令执行训练

当然，在init.json中添加了超参时，记得在train1.py中预加载这些超参，具体使用请参考webui/components/train.py。



