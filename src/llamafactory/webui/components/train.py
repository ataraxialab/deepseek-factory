# Copyright 2025 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Dict, Any
from ...extras.constants import TRAINING_STAGES
from ...extras.packages import is_gradio_available
from ..control import list_files, updir, dump_cfg
from ..common import get_save_dir, get_time

import gradio as gr
if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine

def create_train_tab(engine: "Engine") -> Dict[str, "Component"]:
    initArgs = engine.ArgsManager.args
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    stages = list(TRAINING_STAGES.keys())[:2]
    sftcfg = initArgs.get("sft", {}).get("hp", {})
    rlcfg = initArgs.get("rl", {}).get("hp", {})

    with gr.Row():
        with gr.Column(scale=4, elem_classes=["dropdown-button-container"]):
            model_name_or_path = gr.Dropdown(multiselect=False, allow_custom_value=True, value=initArgs.get("data_mount_dir", "/openr1_data"), scale=9)
            modelupbtn = gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
            model_sft = gr.Dropdown(multiselect=False, allow_custom_value=True, value="", visible=False)

    with gr.Row():
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            dataset= gr.Dropdown(multiselect=False, allow_custom_value=True, value=initArgs.get("data_mount_dir", "/openr1_data"), scale=9)
            upbtn= gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)

    with gr.Row():
        training_stage = gr.Dropdown(choices=stages, value=stages[0])
        num_train_epochs = gr.Textbox(value=sftcfg.get("num_train_epochs", 3))
        save_steps = gr.Textbox(value=sftcfg.get("save_steps", 100))
        logging_steps = gr.Textbox(value=sftcfg.get("logging_steps", 1))

    with gr.Row():
        lora_rank= gr.Textbox(value=sftcfg.get("lora_rank", 32))
        lora_alpha= gr.Textbox(value=sftcfg.get("lora_alpha", 32))
        random_state= gr.Textbox(value=sftcfg.get("random_state", 3407))

    with gr.Row():
        learning_rate = gr.Textbox(value=sftcfg.get("learning_rate", 1.0e-5))
        warmup_ratio= gr.Textbox(value=sftcfg.get("warmup_ratio", 0.1))
        per_device_train_batch_size = gr.Textbox(value=sftcfg.get("per_device_train_batch_size", 8))

    with gr.Row():
        gradient_accumulation_steps = gr.Textbox(value=sftcfg.get("gradient_accumulation_steps", 4))
        max_seq_length = gr.Textbox(value=sftcfg.get("max_seq_length", 4096))
        max_prompt_length = gr.Textbox(visible=True, value=sftcfg.get("max_prompt_length", 1024))
    
    with gr.Row() as grpo_elems:
        gpu_memory_utilization= gr.Textbox(visible=False, value=rlcfg.get("gpu_memory_utilization", 0.7))
        num_generations = gr.Textbox(visible=False, value=rlcfg.get("num_generations", 3))
        max_grad_norm = gr.Textbox(visible=False, value=rlcfg.get("max_grad_norm", 0.1))

    with gr.Row():
        cmd_preview_btn = gr.Button()
        start_btn = gr.Button(variant="primary")
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                progress_bar = gr.Slider(visible=False, interactive=False)
            with gr.Row():
                with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
                    output_dir = gr.Dropdown(label="output_dir", multiselect=False, allow_custom_value=True, value=get_save_dir(f"train_{get_time()}"), scale=9)
                    output_dir_sft = gr.Dropdown(multiselect=False, allow_custom_value=True, value="", visible=False)
                    odirbtn= gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
            with gr.Row():
                system_prompt = gr.Textbox(label="Prompt", interactive=True, lines=5, value=initArgs.get("system_prompt", ""))
        with gr.Column(scale=1):
            loss_viewer = gr.Plot()
    with gr.Row():
        output_box = gr.Markdown(elem_classes="scroll-box")

    all_elements = {
        "num_generations": num_generations,
        "max_prompt_length": max_prompt_length,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_grad_norm": max_grad_norm,
        "model_name_or_path": model_name_or_path,
        "dataset": dataset,
        "training_stage": training_stage,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "per_device_train_batch_size": per_device_train_batch_size,
        "model_sft": model_sft,
        "modelupbtn": modelupbtn,
        "upbtn": upbtn,
        "num_train_epochs": num_train_epochs,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "max_seq_length": max_seq_length,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "random_state": random_state,
        "cmd_preview_btn": cmd_preview_btn,
        "start_btn": start_btn,
        "stop_btn": stop_btn,
        "progress_bar": progress_bar,
        "output_dir": output_dir,
        "output_dir_sft": output_dir_sft,
        "odirbtn": odirbtn,
        "system_prompt": system_prompt,
        "loss_viewer": loss_viewer,
        "output_box": output_box,
        }

    input_elems.update({training_stage, learning_rate, per_device_train_batch_size, model_name_or_path, dataset, num_train_epochs, 
                        num_generations, max_prompt_length, gpu_memory_utilization, max_grad_norm, max_seq_length, lora_rank, lora_alpha, random_state,
                        system_prompt, output_dir, gradient_accumulation_steps, logging_steps})
    input_elems_keys = {"training_stage", "learning_rate", "per_device_train_batch_size", "model_name_or_path", "dataset", "num_train_epochs", 
                        "num_generations", "max_prompt_length", "gpu_memory_utilization", "max_grad_norm", "max_seq_length", "lora_rank", "lora_alpha", "random_state",
                        "system_prompt", "output_dir", "gradient_accumulation_steps", "logging_steps"}
    elem_dict.update(all_elements)

    output_elems = [output_box, progress_bar, loss_viewer]

    with gr.Row():
        more_params = gr.Textbox(label="", lines=10, visible=False, value=dump_cfg(sftcfg, input_elems_keys))
    input_elems.update({more_params})
    elem_dict.update({"more_params": more_params})

    def _preview(*args):
        yield from engine.runner.preview("train", *args)
    cmd_preview_btn.click(_preview, input_elems, output_elems, concurrency_limit=None)

    def _run(*args):
        yield from engine.runner.run("train", *args)
    start_btn.click(_run, input_elems, output_elems)
    stop_btn.click(engine.runner.set_abort)

    model_name_or_path.focus(list_files, [model_name_or_path], [model_name_or_path], queue=False)
    modelupbtn.click(updir, inputs=[model_name_or_path], outputs=[model_name_or_path], concurrency_limit=None)

    dataset.focus(list_files, [dataset], [dataset], queue=False)
    upbtn.click(updir, inputs=[dataset], outputs=[dataset], concurrency_limit=None)

    output_dir.focus(list_files, [output_dir], [output_dir], queue=False)
    odirbtn.click(updir, inputs=[output_dir], outputs=[output_dir], concurrency_limit=None)

    def update_config(stage, ds, ds_sft, odir, odir_sft):
        s = TRAINING_STAGES[stage]
        cfg = sftcfg if s == "sfg" else rlcfg

        return gr.update(value=cfg.get("num_train_epochs", 3)), \
               gr.update(value=cfg.get("save_steps", 100)), \
               gr.update(value=cfg.get("logging_steps", 1)), \
               gr.update(value=cfg.get("lora_rank", 32)), \
               gr.update(value=cfg.get("lora_alpha", 32)), \
               gr.update(value=cfg.get("random_state", 3407)), \
               gr.update(value=cfg.get("learning_rate", 1.0e-5)), \
               gr.update(value=cfg.get("warmup_ratio", 0.1)), \
               gr.update(value=cfg.get("per_device_train_batch_size", 8)), \
               gr.update(value=cfg.get("gradient_accumulation_steps", 4)), \
               gr.update(value=cfg.get("max_seq_length", 4096)), \
               gr.update(value=cfg.get("max_prompt_length", 4096)), \
               gr.update(visible=s=="rl",value=cfg.get("gpu_memory_utilization", 0.7)), \
               gr.update(visible=s=="rl",value=cfg.get("num_generations", 3)), \
               gr.update(visible=s=="rl",value=cfg.get("max_grad_norm", 0.1)), \
               gr.update(value=odir) if s == "rl" else (gr.update(value=ds_sft) if ds_sft else gr.update(visible=True)), \
               gr.update(value=ds) if s == "rl" else gr.update(visible=False), \
               gr.update(value=f"{odir}_rl") if s == "rl" else (gr.update(value=odir_sft) if odir_sft else gr.update(visible=True)), \
               gr.update(value=f"{odir}") if s == "rl" else gr.update(visible=False), \
               gr.update(value=dump_cfg(sftcfg if s == "sfg" else rlcfg, input_elems_keys))

    training_stage.change(update_config, inputs=[training_stage, model_name_or_path, model_sft, output_dir, output_dir_sft],
            outputs=[num_train_epochs,
                    save_steps,
                    logging_steps,
                    lora_rank,
                    lora_alpha,
                    random_state,
                    learning_rate,
                    warmup_ratio,
                    per_device_train_batch_size,
                    gradient_accumulation_steps,
                    max_seq_length,
                    max_prompt_length,
                    gpu_memory_utilization, # rl
                    num_generations,        # rl
                    max_grad_norm,          # rl
                    model_name_or_path, model_sft, output_dir, output_dir_sft, more_params], 
            queue=False)
    return elem_dict

