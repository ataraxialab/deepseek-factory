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
from .components import (
    create_dataprocess_tab,
    create_train_tab,
    create_eval_tab,
    create_upload_tab
)
from .css import CSS
from .engine import Engine


import gradio as gr
if is_gradio_available():
    import gradio as gr


def create_ui(demo_mode: bool = False) -> "gr.Blocks":
    engine = Engine(demo_mode=demo_mode, pure_chat=False)
    hostname = os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", platform.node())).split(".")[0]

    with gr.Blocks(title=f"Deepseek-Factory ({hostname})", css=CSS) as demo:
        lang = gr.Dropdown(choices=["en", "zh"], value="en", visible=False)
        template = gr.Dropdown(choices=["qwen", "baichuan"], value="qwen", visible=False)
        engine.manager.add_elems("global", dict(lang=lang, template=template))

        with gr.Tab("数据上传"):
            engine.manager.add_elems("upload", create_upload_tab(engine))

        with gr.Tab("数据处理"):
            engine.manager.add_elems("dataprocess", create_dataprocess_tab(engine))

        with gr.Tab("模型训练"):
            engine.manager.add_elems("train", create_train_tab(engine))

        with gr.Tab("推理评估"):
            engine.manager.add_elems("eval", create_eval_tab(engine))

        demo.load(engine.resume_x, outputs=engine.manager.get_elem_list(), concurrency_limit=None)
        force_en= {
            "base_url",
            "system_prompt",
            "training_stage",
            "num_processes",
            "learning_rate",
            "max_seq_length",
            "learning_rate",
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "logging_steps",
            "num_generations",
            "max_prompt_length",
            "max_completion_length",
            "top_p",
            "temperature"
        }

        lang.change(
                fn=lambda lang: engine.change_lang_ex(lang, force_en),
                inputs=[lang],
                outputs=engine.manager.get_elem_list(),
                queue=False
                )

    return demo

def run_web_ui() -> None:
    gradio_ipv6 = is_env_enabled("GRADIO_IPV6")
    gradio_share = is_env_enabled("GRADIO_SHARE")
    server_name = os.getenv("GRADIO_SERVER_NAME", "[::]" if gradio_ipv6 else "0.0.0.0")
    create_ui().queue().launch(share=gradio_share, server_name=server_name, inbrowser=True)

