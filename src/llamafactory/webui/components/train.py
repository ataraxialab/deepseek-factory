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
from typing import TYPE_CHECKING, Dict, Any, Optional
from ...extras.constants import TRAINING_STAGES
from ...extras.packages import is_gradio_available
from ..control import list_dirs, list_files, updir
from ..common import get_save_dir, get_time, WORKSPACE, get_init_config
from .data2 import create_preview_box

import gradio as gr
if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine

def create_train_tab(engine: "Engine") -> Dict[str, "Component"]:
    initArgs = engine.ArgsManager.args
    params = engine.manager.get_base_elems()
    elem_dict = dict()

    mlist: Dict[str, str] = initArgs.get("models", None)
    mk = next(iter(mlist)) if mlist else ""
    mv = mlist[mk] if mk else ""

    stages = list(TRAINING_STAGES.keys())
    s0 = list(TRAINING_STAGES.values())[0]
    s0cfg= initArgs.get(f"{s0}", {}).get("hp", {})

    with gr.Row():
        with gr.Column(scale=4, elem_classes=["dropdown-button-container"]):
            model = gr.Dropdown(choices=list(mlist.keys()) if mlist else [], label="Model Name", value=mk, 
                                multiselect=False, scale=3)
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            model_name_or_path = gr.Dropdown(multiselect=False, allow_custom_value=True, 
                                             value=mv if mv else initArgs.get("data_mount_dir", WORKSPACE), scale=7)
            mnbtn= gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)

    with gr.Row():
        training_stage = gr.Dropdown(choices=stages, value=stages[0], scale=0, min_width=200)
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            dataset = gr.Dropdown(multiselect=False, allow_custom_value=True, 
                                 value=initArgs.get("data_mount_dir", WORKSPACE), scale=8)
            upbtn= gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
        with gr.Column(min_width=70):
            previewbtn: gr.Button = gr.Button(value="preview", min_width=70)
            editbtn: gr.Button = gr.Button(value="edit", min_width=70)

    with gr.Row():
        preview_elems = create_preview_box(dataset, previewbtn, editbtn)

    def default_v(key: str, df: Any):
        if not s0cfg or not isinstance(s0cfg, dict):
            return df
        return s0cfg.get(key, df) 

    with gr.Group():
        with gr.Row():
            learning_rate = gr.Textbox(value=default_v("learning_rate", 1.0e-5))
            per_device_train_batch_size = gr.Textbox(value=default_v("per_device_train_batch_size", 2))
        with gr.Row():
            num_train_epochs = gr.Textbox(value=default_v("num_train_epochs", 3))
            gradient_accumulation_steps = gr.Textbox(value=default_v("gradient_accumulation_steps", 4))
            logging_steps = gr.Textbox(value=default_v("logging_steps", 1))
        
    params.update({training_stage, learning_rate, per_device_train_batch_size, model_name_or_path, dataset, num_train_epochs, 
                        gradient_accumulation_steps, logging_steps})
    params_keys = {"training_stage", "learning_rate", "per_device_train_batch_size", "model_name_or_path", "dataset", "num_train_epochs", 
                        "gradient_accumulation_steps", "logging_steps", "config_path", "output_dir", "system_prompt"}
    elem_dict.update(
            dict(training_stage=training_stage, 
                 learning_rate=learning_rate, 
                 per_device_train_batch_size=per_device_train_batch_size, 
                 model=model, 
                 model_name_or_path=model_name_or_path, 
                 dataset=dataset, 
                 upbtn=upbtn, 
                 previewbtn=previewbtn, 
                 editbtn=editbtn, 
                 num_train_epochs=num_train_epochs, 
                 gradient_accumulation_steps=gradient_accumulation_steps, 
                 logging_steps=logging_steps, 
                 ))

    def _load_init_args(d: Optional[Dict[str, Any]]):
        if not d:
            return ""
        filtered_data = {
                k: v for k, v in d.items() 
                if not k in elem_dict and not k in params_keys and  (v is not None and v is not False and v != "")
                }
        return safe_dump(filtered_data, default_style=None, allow_unicode=True, default_flow_style=False)

    with gr.Accordion(label="更多参数", open=False):
        more_params = gr.Textbox(label="", lines=10, value=_load_init_args(initArgs.get(f"{s0}", {}).get("hp", {})))

    with gr.Row():
        cmd_preview_btn = gr.Button()
        arg_save_btn = gr.Button(value="Save config", variant="secondary")
        arg_load_btn = gr.Button(value="Load config", variant="secondary")
        start_btn = gr.Button(variant="primary")
        stop_btn = gr.Button(variant="stop")

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
                    config_path = gr.Dropdown(label="config_path", multiselect=False, allow_custom_value=True, 
                                             value=get_save_dir(f"train_{get_time()}"), scale=9)
                    cfgpathbtn = gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
            with gr.Row():
                with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
                    output_dir = gr.Dropdown(label="output_dir", multiselect=False, allow_custom_value=True, 
                                             value=get_save_dir(f"train_{get_time()}"), scale=9)
                    odirbtn= gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
        with gr.Column(scale=2):
            loss_viewer = gr.Plot()

    with gr.Row():
        progress_bar = gr.Slider(visible=False, interactive=False)
    with gr.Row():
        system_prompt = gr.Textbox(label="Prompt", interactive=True, lines=5, value=default_v("system_prompt", ""))
    with gr.Row():
        output_box = gr.Markdown(elem_classes="scroll-box")

    params.update({more_params, config_path, system_prompt, output_dir})

    elem_dict.update(
            dict(training_stage=training_stage, 
                 more_params=more_params, 
                 cmd_preview_btn=cmd_preview_btn, 
                 arg_save_btn=arg_save_btn, 
                 arg_load_btn=arg_load_btn, 
                 start_btn=start_btn, 
                 stop_btn=stop_btn, 
                 output_dir=output_dir, 
                 odirbtn=odirbtn, 
                 config_path=config_path, 
                 cfgpathbtn=odirbtn, 
                 progress_bar=progress_bar, 
                 system_prompt=system_prompt,
                 output_box=output_box, 
                 loss_viewer=loss_viewer))

    output_elems = [output_box, progress_bar, loss_viewer]

    def _preview(*args):
        yield from engine.runner.preview("train", *args)
    cmd_preview_btn.click(_preview, params, output_elems, concurrency_limit=None)

    def _save_args(*args):
        return engine.runner.save_args("train", *args)
    arg_save_btn.click(_save_args, params, output_elems, concurrency_limit=None)

    def _load_args(*args):
        return engine.runner.load_args("train", *args)
    arg_load_btn.click(_load_args, params, list(params) + [output_box], concurrency_limit=None)

    def _run(*args):
        yield from engine.runner.run("train", *args)

    start_btn.click(_run, params, output_elems)

    stop_btn.click(engine.runner.set_abort)

    model_name_or_path.focus(list_dirs, [model_name_or_path], [model_name_or_path], queue=False)
    mnbtn.click(updir, inputs=[model_name_or_path], outputs=[model_name_or_path], concurrency_limit=None)

    dataset.focus(list_files, [dataset], [dataset], queue=False)
    upbtn.click(updir, inputs=[dataset], outputs=[dataset], concurrency_limit=None)

    def update_model(mn):
        v = mlist[mn] if mlist and mn in mlist else ""
        return gr.update(value=v) if v else gr.update(visible=False),  \
                gr.update(value=f"Selected model path: {v}")
    model.change(update_model, inputs=[model], outputs=[model_name_or_path, output_box])

    output_dir.focus(list_files, [output_dir], [output_dir], queue=False)
    odirbtn.click(updir, inputs=[output_dir], outputs=[output_dir], concurrency_limit=None)
    config_path.focus(list_files, [config_path], [config_path], queue=False)
    cfgpathbtn.click(updir, inputs=[config_path], outputs=[config_path], concurrency_limit=None)

    def update_config(stage):
        stage = TRAINING_STAGES[stage]  
        params = _load_init_args(initArgs.get(f"{stage}", {}).get("hp", {}))
        return gr.update(value=get_init_config(initArgs, f"{stage}.hp.learning_rate", 1.0e-5)), \
               gr.update(value=get_init_config(initArgs, f"{stage}.hp.num_train_epochs", 1)), \
               gr.update(value=get_init_config(initArgs, f"{stage}.hp.per_device_train_batch_size", 1)), \
               gr.update(value=get_init_config(initArgs, f"{stage}.hp.gradient_accumulation_steps", 1)), \
               gr.update(value=get_init_config(initArgs, f"{stage}.hp.logging_steps", 1)), \
               gr.update(value=params)

    training_stage.change(update_config, inputs=[training_stage],
            outputs=[learning_rate, 
                     num_train_epochs, 
                     per_device_train_batch_size, 
                     gradient_accumulation_steps, 
                     logging_steps, 
                     more_params], 
            queue=False)

    return elem_dict

