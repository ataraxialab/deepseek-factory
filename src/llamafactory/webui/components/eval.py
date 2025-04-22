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

from yaml import safe_dump
from typing import TYPE_CHECKING, Dict, Optional, Any
from ...extras.packages import is_gradio_available
from ..control import list_files, updir
from ..common import get_save_dir, get_time, WORKSPACE, get_init_config
from .data2 import create_preview_box

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

    mlist: Dict[str, str] = initArgs.get("models", None)
    mk = next(iter(mlist)) if mlist else ""
    mv = mlist[mk] if mk else ""
    evalcfg = initArgs.get("eval", {}).get("hp", {})

    with gr.Row():
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            model = gr.Dropdown(choices=list(mlist.keys()) if mlist else [], label="Model Name", interactive=True, value=mk, multiselect=False, scale=3)
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            model_name_or_path = gr.Dropdown(multiselect=False, allow_custom_value=True, 
                                             value=mv if mv else initArgs.get("data_mount_dir", WORKSPACE), scale=7)
            mnbtn= gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)

    with gr.Row():
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            eval_dataset= gr.Dropdown(multiselect=False, allow_custom_value=True, 
                                    value=initArgs.get("data_mount_dir", WORKSPACE), scale=8)
            upbtn= gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
        with gr.Column(min_width=70):
            previewbtn: gr.Button = gr.Button(value="preview", min_width=70)
            editbtn: gr.Button = gr.Button(visible=False, value="edit", min_width=70)

    input_elems.update({model_name_or_path, eval_dataset})
    elem_dict.update(dict(model=model, model_name_or_path=model_name_or_path, mnbtn=mnbtn, 
                          eval_dataset=eval_dataset, upbtn=upbtn, previewbtn=previewbtn, editbtn=editbtn))

    with gr.Row():
        preview_elems = create_preview_box(eval_dataset, previewbtn, editbtn)

    def default_v(key: str, df: Any):
        if not evalcfg or not isinstance(evalcfg, dict):
            return df
        return evalcfg.get(key, df) 


    with gr.Group():
        with gr.Row():
            per_device_eval_batch_size = gr.Textbox(value=default_v("per_device_eval_batch_size", 1))
            top_p = gr.Textbox(value=default_v("top_p", 0.9))
            temperature = gr.Textbox(value=default_v("temperature", 0.8))

    input_elems.update({per_device_eval_batch_size, top_p, temperature})
    elem_dict.update(
        dict(
            per_device_eval_batch_size=per_device_eval_batch_size, 
            top_p=top_p,
            temperature=temperature
        )
    )

    params_keys = {"model_name_or_path", "eval_dataset", "per_device_eval_batch_size", "top_p", "temperature", "output_dir", "system_prompt"}

    def _load_init_args(d: Optional[Dict[str, Any]]):
        if not d:
            return ""
        filtered_data = {
                k: v for k, v in d.items() 
                if not k in elem_dict and not k in params_keys and  (v is not None and v is not False and v != "")
                }
        return safe_dump(filtered_data, default_style=None, allow_unicode=True, default_flow_style=False)
    with gr.Accordion(open=False, label="更多参数"):
        more_params = gr.Textbox(label="", lines=10, value=_load_init_args(initArgs.get(f"eval", {}).get("hp", {})))

    with gr.Row():
        cmd_preview_btn = gr.Button()
        start_btn = gr.Button(variant="primary")
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
                    config_path = gr.Dropdown(label="config_path", multiselect=False, allow_custom_value=True, 
                                             value=get_save_dir(f"eval_{get_time()}"), scale=9)
                    cfgpathbtn = gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
            with gr.Row():
                with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
                    output_dir = gr.Dropdown(label="output_dir", multiselect=False, allow_custom_value=True, 
                                             value=get_save_dir(f"eval_{get_time()}"), scale=9)
                    odirbtn= gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
        with gr.Column(scale=2):
            loss_viewer = gr.Plot()

    with gr.Row():
        progress_bar = gr.Slider(visible=False, interactive=False)
    with gr.Row():
        system_prompt = gr.Textbox(label="Prompt", interactive=True, lines=5, value=default_v("system_prompt", ""))
    with gr.Row():
        output_box = gr.Markdown(elem_classes="scroll-box")

    input_elems.update({more_params, config_path, output_dir, system_prompt})
    elem_dict.update(
        dict(
            more_params=more_params,
            cmd_preview_btn=cmd_preview_btn,
            start_btn=start_btn,
            stop_btn=stop_btn,
            output_dir=output_dir,
            odirbtn=odirbtn,
            config_path=config_path, 
            cfgpathbtn=odirbtn,
            progress_bar=progress_bar,
            loss_viewer=loss_viewer,
            system_prompt=system_prompt,
            output_box=output_box,
        )
    )

    output_elems = [output_box, progress_bar, loss_viewer]
    def _preview(*args):
        yield from engine.runner.preview("eval", *args)
    cmd_preview_btn.click(_preview, input_elems, output_elems, concurrency_limit=None)

    def _run(*args):
        yield from engine.runner.run("eval", *args)
    start_btn.click(_run, input_elems, output_elems)

    stop_btn.click(engine.runner.set_abort)

    model_name_or_path.focus(list_files, [model_name_or_path], [model_name_or_path], queue=False)
    mnbtn.click(updir, inputs=[model_name_or_path], outputs=[model_name_or_path], concurrency_limit=None)

    eval_dataset.focus(list_files, [eval_dataset], [eval_dataset], queue=False)
    upbtn.click(updir, inputs=[eval_dataset], outputs=[eval_dataset], concurrency_limit=None)

    output_dir.focus(list_files, [output_dir], [output_dir], queue=False)
    odirbtn.click(updir, inputs=[output_dir], outputs=[output_dir], concurrency_limit=None)
    config_path.focus(list_files, [config_path], [config_path], queue=False)
    cfgpathbtn.click(updir, inputs=[config_path], outputs=[config_path], concurrency_limit=None)

    return elem_dict

