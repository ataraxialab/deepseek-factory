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
from ...extras.packages import is_gradio_available
from ..control import list_files, updir, dump_cfg
from ..common import get_save_dir, get_time

import gradio as gr
if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine


def create_eval_tab(engine: "Engine") -> Dict[str, "Component"]:
    initArgs = engine.ArgsManager.args
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    evalcfg = initArgs.get("eval", {}).get("hp", {})
    with gr.Row():
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            model_name_or_path = gr.Dropdown(multiselect=False, allow_custom_value=True, value=initArgs.get("data_mount_dir", "/openr1_data"), scale=9)
            modelupbtn = gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)

    with gr.Row():
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            eval_dataset = gr.Dropdown(multiselect=False, allow_custom_value=True, value=initArgs.get("data_mount_dir", "/openr1_data"), scale=9)
            upbtn = gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)

    input_elems.update({eval_dataset, model_name_or_path})
    elem_dict.update(dict( eval_dataset=eval_dataset, upbtn=upbtn, model_name_or_path=model_name_or_path, modelupbtn=modelupbtn))
    
    with gr.Row():
        cmd_preview_btn = gr.Button()
        start_btn = gr.Button(variant="primary")
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        progress_bar = gr.Slider(interactive=False, visible=False)

    with gr.Row():
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            output_dir = gr.Dropdown(label="output_dir", multiselect=False, allow_custom_value=True, 
                                     value=get_save_dir(f"eval_{get_time()}"), scale=9)
            odirbtn = gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
    with gr.Row():
        system_prompt = gr.Textbox(label="Prompt", visible=True, interactive=True, lines=10, scale=1, value=initArgs.get("system_prompt", ""))
    with gr.Row():
        output_box = gr.Markdown(elem_classes="scroll-box")
    input_elems.update({output_dir, system_prompt})
    input_elems_keys = {"eval_dataset", "model_name_or_path", "output_dir", "system_prompt"}
    elem_dict.update(
        dict(
            cmd_preview_btn=cmd_preview_btn,
            start_btn=start_btn,
            stop_btn=stop_btn,
            progress_bar=progress_bar,
            output_dir=output_dir,
            odirbtn=odirbtn,
            system_prompt=system_prompt,
            output_box=output_box,
        )
    )

    with gr.Row():
        more_params = gr.Textbox(visible=False, value=dump_cfg(evalcfg, input_elems_keys))
    input_elems.update({more_params})
    elem_dict.update({"more_params": more_params})

    output_elems = [output_box, progress_bar]

    eval_dataset.focus(list_files, [eval_dataset], [eval_dataset], queue=False)
    upbtn.click(updir, inputs=[eval_dataset], outputs=[eval_dataset], concurrency_limit=None)

    output_dir.focus(list_files, [output_dir], [output_dir], queue=False)
    odirbtn.click(updir, inputs=[output_dir], outputs=[output_dir], concurrency_limit=None)

    model_name_or_path.focus(list_files, [model_name_or_path], [model_name_or_path], queue=False)
    modelupbtn.click(updir, inputs=[model_name_or_path], outputs=[model_name_or_path], concurrency_limit=None)

    def _preview(*args):
        yield from engine.runner.preview("eval", *args)
    cmd_preview_btn.click(_preview, input_elems, output_elems, concurrency_limit=None)

    def _run(*args):
        yield from engine.runner.run("eval", *args)
    start_btn.click(_run, input_elems, output_elems)
    stop_btn.click(engine.runner.set_abort)

    return elem_dict
