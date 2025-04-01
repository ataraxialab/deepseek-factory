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

from typing import TYPE_CHECKING, Dict
from ...extras.packages import is_gradio_available
from ..control import list_files, updir
from ..common import get_save_dir, get_time

import gradio as gr
if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component
    from ..engine import Engine


def create_distill_tab(engine: "Engine") -> Dict[str, "Component"]:
    global gengine
    gengine = engine
    initArgs = engine.ArgsManager.args
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            dataset_src_path = gr.Dropdown(multiselect=False, allow_custom_value=True, value=initArgs.get("data_mount_dir", "/openr1_data"), scale=9)
            upbtn: gr.Button = gr.Button(elem_classes=["overlay-button"], value="..", scale=0, min_width=20)

    with gr.Row():
        base_url = gr.Textbox(label="url",value="", scale=1)

    with gr.Row():
        api_key = gr.Textbox(label="api key",value="", scale=1)
        model = gr.Textbox(label="模型",value="DeepSeek-R1", scale=1)
       
    input_elems.update({dataset_src_path, api_key, base_url, model})
    elem_dict.update(dict(dataset_src_path=dataset_src_path, upbtn=upbtn, api_key=api_key, base_url=base_url, model=model))

    with gr.Row():
        start_distill_btn = gr.Button()
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            sdir = get_save_dir(f"distill_{get_time()}")
            output_dir = gr.Dropdown(label="output_dir", multiselect=False, allow_custom_value=True, value=sdir, scale=9)
            odirbtn: gr.Button = gr.Button(elem_classes=["overlay-button"], value="..", scale=0, min_width=20)
            dataset_dst_path = gr.Textbox(label="dataset_dst_path", visible=False, value=sdir)
    with gr.Row():
        system_prompt = gr.Textbox(label="Prompt", visible=True, interactive=True, lines=10, scale=1, value=initArgs.get("system_prompt", ""))
    with gr.Row():
        output_box = gr.Markdown(elem_classes="scroll-box")

    input_elems.update({output_dir, dataset_dst_path, system_prompt})
    elem_dict.update(
        dict(
            output_dir=output_dir,
            dataset_dst_path=dataset_dst_path,
            odirbtn=odirbtn,
            start_distill_btn=start_distill_btn,
            system_prompt=system_prompt,
            output_box=output_box,
            stop_btn=stop_btn
        )
    )

    dataset_src_path.focus(list_files, [dataset_src_path], [dataset_src_path], queue=False)
    upbtn.click(updir, inputs=[dataset_src_path], outputs=[dataset_src_path], concurrency_limit=None)

    output_dir.focus(list_files, [output_dir], [output_dir], queue=False)
    output_dir.change(lambda x: gr.update(value=x), inputs=[output_dir], outputs=[dataset_dst_path])
    odirbtn.click(updir, inputs=[output_dir], outputs=[output_dir], concurrency_limit=None)

    def _run(*args):
        yield from engine.runner.run("distill", *args)
    start_distill_btn.click(_run, input_elems, [output_box])

    stop_btn.click(engine.runner.set_abort)

    return elem_dict

