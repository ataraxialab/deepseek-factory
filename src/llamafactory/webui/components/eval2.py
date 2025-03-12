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


def create_eval2_tab(engine: "Engine") -> Dict[str, "Component"]:
    initArgs = engine.ArgsManager.args
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        model = gr.Dropdown(multiselect=False, allow_custom_value=True, value=initArgs.get("data_mount_dir", "/openr1_data"), scale=9)
        modelupbtn= gr.Button(variant="secondary", value="..", scale=1)

    with gr.Row():
        dataset = gr.Dropdown(multiselect=False, allow_custom_value=True, value=initArgs.get("data_mount_dir", "/openr1_data"), scale=9)
        upbtn= gr.Button(variant="secondary", value="..", scale=1)

    input_elems.update({dataset, upbtn, model, modelupbtn})
    elem_dict.update(dict( dataset=dataset, upbtn=upbtn, model=model, modelupbtn=modelupbtn))
    
    with gr.Row():
        cmd_preview_btn = gr.Button()
        start_btn = gr.Button(variant="primary")
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        progress_bar = gr.Slider(interactive=False, visible=False)

    with gr.Row():
        dataset_out = gr.Textbox(label="dataset_out", visible=False, interactive=False, scale=1, value="")
        output_dir = gr.Dropdown(label="output_dir", multiselect=False, allow_custom_value=True, 
                                 value=get_save_dir(f"eval2_{get_time()}"), scale=9)
        odirbtn= gr.Button(variant="secondary", value="..", scale=1)
    with gr.Row():
        system_prompt = gr.Textbox(label="Prompt", visible=True, interactive=True, lines=10, scale=1, value=initArgs.get("system_prompt", ""))
    with gr.Row():
        output_box = gr.Markdown(elem_classes="scroll-box")
    input_elems.update({output_dir, dataset_out, odirbtn, modelupbtn, output_box, system_prompt})
    elem_dict.update(
        dict(
            cmd_preview_btn=cmd_preview_btn,
            start_btn=start_btn,
            stop_btn=stop_btn,
            progress_bar=progress_bar,
            dataset_out=dataset_out,
            output_dir=output_dir,
            odirbtn=odirbtn,
            system_prompt=system_prompt,
            output_box=output_box,
        )
    )
    output_elems = [output_box, progress_bar]

    dataset.focus(list_files, [dataset], [dataset], queue=False)
    upbtn.click(updir, inputs=[dataset], outputs=[dataset], concurrency_limit=None)

    output_dir.focus(list_files, [output_dir], [output_dir], queue=False)
    odirbtn.click(updir, inputs=[output_dir], outputs=[output_dir], concurrency_limit=None)

    model.focus(list_files, [model], [model], queue=False)
    modelupbtn.click(updir, inputs=[model], outputs=[model], concurrency_limit=None)

    cmd_preview_btn.click(engine.runner.preview_eval2, input_elems, output_elems, concurrency_limit=None)
    start_btn.click(engine.runner.run_eval2, input_elems, output_elems)
    stop_btn.click(engine.runner.set_abort)

    return elem_dict
