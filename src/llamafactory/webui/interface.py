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

import os
import platform

from ..extras.misc import is_env_enabled
from ..extras.packages import is_gradio_available
from .common import save_config
from .components import (
    create_chat_box,
    create_eval_tab,
    create_export_tab,
    create_infer_tab,
    create_top,
    create_train_tab,
    create_distill_tab,
    create_train2_tab,
    create_eval2_tab,
    create_segment_tab,
    dataset_tab
)
from .css import CSS
from .engine import Engine


import gradio as gr
if is_gradio_available():
    import gradio as gr


def create_ui(demo_mode: bool = False) -> "gr.Blocks":
    engine = Engine(demo_mode=demo_mode, pure_chat=False)
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]

    with gr.Blocks(title=f"LLaMA Board ({hostname})", css=CSS) as demo:
        if demo_mode:
            gr.HTML("<h1><center>LLaMA Board: A One-stop Web UI for Getting Started with LLaMA Factory</center></h1>")
            gr.HTML(
                '<h3><center>Visit <a href="https://github.com/hiyouga/LLaMA-Factory" target="_blank">'
                "LLaMA Factory</a> for details.</center></h3>"
            )
            gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")

        with gr.Row(visible=False): 
            engine.manager.add_elems("top", create_top())
        lang: "gr.Dropdown" = engine.manager.get_elem_by_id("top.lang")

        with gr.Tab("数据拆分"):
            engine.manager.add_elems("segment", create_segment_tab(engine))

        def update_distill(v):
            if v is not None and os.path.exists(v):
                return gr.update(value=v)
            else:
                return gr.update(visible=True)
        with gr.Tab("数据蒸馏") as distill:
            engine.manager.add_elems("distill", create_distill_tab(engine))
            distill.select(update_distill, 
                    [engine.manager.get_elem_by_id("segment.dataset_out")],
                    [engine.manager.get_elem_by_id("distill.dataset")])

        def update_dataset(v):
            if v is not None and os.path.exists(os.path.join(v, "correct.json")):
                return gr.update(value=os.path.join(v, "correct.json"))
            else:
                return gr.update(visible=True)
        with gr.Tab("数据筛选") as datasets:
            engine.manager.add_elems("dataset", dataset_tab(engine))
            datasets.select(update_dataset, 
                    [engine.manager.get_elem_by_id("distill.output_dir")],
                    [engine.manager.get_elem_by_id("dataset.dataset")])

        def update_train(v):
            if v is not None and os.path.exists(v):
                return gr.update(value=v)
            else:
                return gr.update(visible=True)
        with gr.Tab("数据训练") as train2_tab:
            engine.manager.add_elems("train2", create_train2_tab(engine))
            train2_tab.select(update_train, 
                    [engine.manager.get_elem_by_id("dataset.dataset")],
                    [engine.manager.get_elem_by_id("train2.dataset")])

        def update_eval(v1, v2):
            rv1 = gr.update(visible=True) 
            rv2 = gr.update(visible=True) 
            if v1 is not None and os.path.exists(v1):
                rv1 = gr.update(value=v1)
            if v2 is not None and os.path.exists(v2):
                rv2 = gr.update(value=v2)
            return rv1, rv2

        with gr.Tab("效果验证") as eval_tab:
            engine.manager.add_elems("eval2", create_eval2_tab(engine))
            eval_tab.select(update_eval, 
                            [engine.manager.get_elem_by_id("segment.dataset_out"), engine.manager.get_elem_by_id("train2.output_dir")],
                            [engine.manager.get_elem_by_id("eval2.dataset"), engine.manager.get_elem_by_id("eval2.model")])

        #with gr.Tab("Train", visible=False):
        #    engine.manager.add_elems("train", create_train_tab(engine))

        #with gr.Tab("Evaluate & Predict", visible=False):
        #    engine.manager.add_elems("eval", create_eval_tab(engine))

        #with gr.Tab("Chat", visible=False):
        #    engine.manager.add_elems("infer", create_infer_tab(engine))

        #if not demo_mode:
        #    with gr.Tab("Export", visible=False):
        #        engine.manager.add_elems("export", create_export_tab(engine))

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        #lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        force_en= {
            "base_url",
            "system_prompt",
            "training_stage",
            "num_processes",
            "max_seq_length",
            "learning_rate",
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "logging_steps",
            "num_generations",
            "max_prompt_length",
            "max_completion_length",
            "save_steps",
            "gpu_memory_utilization",
            "warmup_ratio",
            "lora_rank",
            "lora_alpha",
            "random_state",
            "max_grad_norm",
        }

        lang.change(
                fn=lambda lang: engine.change_lang_ex(lang, force_en),
                inputs=[lang],
                outputs=engine.manager.get_elem_list(),
                queue=False
                )
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


def create_web_demo() -> "gr.Blocks":
    engine = Engine(pure_chat=True)

    with gr.Blocks(title="Web Demo", css=CSS) as demo:
        lang = gr.Dropdown(choices=["en", "ru", "zh", "ko", "ja"], scale=1)
        engine.manager.add_elems("top", dict(lang=lang))

        _, _, chat_elems = create_chat_box(engine, visible=True)
        engine.manager.add_elems("infer", chat_elems)

        demo.load(engine.resume, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        lang.change(engine.change_lang, [lang], engine.manager.get_elem_list(), queue=False)
        lang.input(save_config, inputs=[lang], queue=False)

    return demo


def run_web_ui() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)


def run_web_demo() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    create_web_demo().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)
