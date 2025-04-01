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
from ..common import get_save_dir, get_time, WORKSPACE

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
        dataset = gr.Dropdown(multiselect=False, allow_custom_value=True, value=initArgs.get("data_mount_dir", WORKSPACE), scale=9)
        upbtn= gr.Button(variant="secondary", value="..", scale=1)

    with gr.Row():
        base_url = gr.Textbox(label="url",value="", scale=1)

    with gr.Row():
        api_key = gr.Textbox(label="api key",value="", scale=1)
        model = gr.Textbox(label="模型",value="DeepSeek-R1", scale=1)
       
    input_elems.update({dataset, upbtn, api_key, base_url,model})
    elem_dict.update(dict(dataset=dataset, upbtn=upbtn, api_key=api_key, base_url=base_url, model=model))

    with gr.Row():
        start_distill_btn = gr.Button()
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        output_dir = gr.Dropdown(label="output_dir", multiselect=False, allow_custom_value=True, 
                                 value=get_save_dir(f"distill_{get_time()}"), scale=9)
        odirbtn= gr.Button(variant="secondary", value="..", scale=1)
    with gr.Row():
        dataset_out = gr.Textbox(label="dataset_out", visible=False, interactive=False, scale=1, value="")
    with gr.Row():
        system_prompt = gr.Textbox(label="Prompt", visible=True, interactive=True, lines=10, scale=1, value=initArgs.get("system_prompt", ""))
    with gr.Row():
        output_box = gr.Markdown(elem_classes="scroll-box")

    input_elems.update({output_dir, odirbtn, start_distill_btn, dataset_out, system_prompt, output_box, stop_btn})
    elem_dict.update(
        dict(
            dataset_out=dataset_out,
            output_dir=output_dir,
            odirbtn=odirbtn,
            start_distill_btn=start_distill_btn,
            system_prompt=system_prompt,
            output_box=output_box,
            stop_btn=stop_btn
        )
    )

    dataset.focus(list_files, [dataset], [dataset], queue=False)
    upbtn.click(updir, inputs=[dataset], outputs=[dataset], concurrency_limit=None)

    output_dir.focus(list_files, [output_dir], [output_dir], queue=False)
    odirbtn.click(updir, inputs=[output_dir], outputs=[output_dir], concurrency_limit=None)

    start_distill_btn.click(lambda *args: next(engine.runner.run_x("distill", *args)),  
                            input_elems, [output_box])
    stop_btn.click(engine.runner.set_abort)
    return elem_dict

