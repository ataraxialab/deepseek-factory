from typing import TYPE_CHECKING, Dict

from llamafactory.webui.locales import ALERTS
from ...extras.packages import is_gradio_available
from ...extras.packages import is_gradio_available
from ..control import list_dirs, list_files, updir
from .data2 import create_preview_box
from ..common import WORKSPACE
import random
import os
import json

import gradio as gr
if is_gradio_available():
    import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component
    from ..engine import Engine

def load_data(dataset: str):
    if not dataset:
        return []
    
    data_cache = []
    try:
        with open(dataset, "r", encoding="utf-8") as f:
            if dataset.endswith(".json"):
                data_cache = json.load(f)
            elif dataset.endswith(".jsonl"):
                data_cache = [json.loads(line) for line in f if line.strip()]
            else:
                data_cache = list(f)
    except Exception as e:
        print("加载失败:", e)
        data_cache = []

    print(f"Loading {dataset} with len  {len(data_cache)}")
    return data_cache

def segment_data(dataset, ratio, train_name, test_name):
    data = load_data(dataset)
    random.shuffle(data)
    split_index = int(len(data) * ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    print(f"Segmenting {dataset} into {train_name} and {test_name}")
    with open(train_name, 'w', encoding='utf-8') as f_train:
        json.dump(train_data, f_train, ensure_ascii=False, indent=4)
    with open(test_name, 'w', encoding='utf-8') as f_test:
        json.dump(test_data, f_test, ensure_ascii=False, indent=4)
    print(f"Segment done")

    finish_info = ALERTS["info_aborted"]["zh"]
    gr.Info(finish_info)
    return gr.update(value=f"生成训练集{train_name}和测试集{test_name}")

def create_dataprocess_tab(engine: "Engine") -> Dict[str, "Component"]:
    initArgs = engine.ArgsManager.args
    elem_dict = dict()
    input_elems = engine.manager.get_base_elems()
    lang = gr.Textbox(visible=False, value=initArgs.get("lang", "zh"))

    with gr.Row():
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            dataset_src_path: gr.Dropdown = gr.Dropdown(
                multiselect=False,
                allow_custom_value=True,
                value=initArgs.get("data_mount_dir", WORKSPACE),
                scale=10,
                label="Dataset"
            )
            upbtn: gr.Button = gr.Button(elem_classes=["overlay-button"], value="..", scale=0, min_width=20)
        with gr.Column(min_width=70):
            previewbtn: gr.Button = gr.Button(value="preview", min_width=50)
            editbtn: gr.Button = gr.Button(value="edit", min_width=50)
            #uploadbtn: gr.Button = gr.Button(visible=False, variant="secondary", value="upload")

    with gr.Row():
        preview_elems = create_preview_box(dataset_src_path, previewbtn, editbtn)

    input_elems.update({dataset_src_path})
    elem_dict.update(
            dict(
                dataset_src_path=dataset_src_path,
                upbtn=upbtn,
                previewbtn=previewbtn,
                editbtn=editbtn,
                **preview_elems
                )
            )

    with gr.Row():
        dataprocesstype = gr.Dropdown(label="DataProcessType", choices=["segment", "distill"], multiselect=False, 
                                  allow_custom_value=False, value="segment")

    with gr.Column() as segment_row:
        with gr.Row():
            with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
                train_name = gr.Dropdown(label="Traindir", multiselect=False, allow_custom_value=True, 
                                        value=initArgs.get("data_mount_dir", WORKSPACE), scale=9)
                train_name_btn= gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
            with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
                test_name = gr.Dropdown(label="TestDir", multiselect=False, allow_custom_value=True, 
                                       value=initArgs.get("data_mount_dir", WORKSPACE), scale=9)
                test_name_btn= gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
        with gr.Row():
            train_test_split_ratio = gr.Slider(label="Dataset Split", minimum=0, maximum=1, value=0.8, step=0.01)
        with gr.Row():
            cmd_segment_btn = gr.Button(variant="primary", value="Segment")

    with gr.Group(visible=False) as distill_row:
        with gr.Row():
            with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
                dataset_dst_path = gr.Dropdown(label="DistillDir", multiselect=False, allow_custom_value=True, 
                                          value=initArgs.get("data_mount_dir", WORKSPACE), scale=9)
                distill_dst_btn= gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
        with gr.Row():
            base_url = gr.Textbox(label="url",value="", scale=1)
        with gr.Row():
            api_key = gr.Textbox(label="api key",value="", scale=4)
            model = gr.Textbox(label="模型",value="DeepSeek-R1", scale=4)
        with gr.Row():
            start_btn = gr.Button(variant="primary", value="Start")
            stop_btn = gr.Button(variant="stop", value="Stop")

    with gr.Row():
        output_box = gr.Markdown(elem_classes="scroll-box", label="")
        output_dir = gr.Textbox(visible=False, label="")

    input_elems.update({
        dataprocesstype, train_name, test_name, train_test_split_ratio, dataset_dst_path,
        base_url, api_key, model, output_dir
        })
    elem_dict.update(
        dict(
            lang=lang,
            dataprocesstype=dataprocesstype,
            train_name=train_name,
            train_name_btn=train_name_btn,
            test_name=test_name,
            test_name_btn=test_name_btn,
            train_test_split_ratio=train_test_split_ratio,
            cmd_segment_btn=cmd_segment_btn,
            dataset_dst_path=dataset_dst_path,
            distill_dst_btn=distill_dst_btn,
            base_url=base_url,
            api_key=api_key,
            model=model,
            start_btn=start_btn,
            stop_btn=stop_btn,
            output_box=output_box,
            output_dir=output_dir,
        )
    )

    def update_segment_dst(src):
        name, ext = os.path.splitext(src)
        return gr.update(value=f"{name}_train{ext}"), gr.update(value=f"{name}_test{ext}"), \
                gr.update(value=os.path.dirname(src))
                
    dataset_src_path.change(update_segment_dst, inputs=[dataset_src_path], outputs=[train_name, test_name, output_dir], queue=False)

    dataset_src_path.focus(list_files, [dataset_src_path], [dataset_src_path], queue=False)
    upbtn.click(updir, inputs=[dataset_src_path], outputs=[dataset_src_path], concurrency_limit=None)

    train_name.focus(list_files, [train_name], [train_name], queue=False)
    train_name_btn.click(updir, inputs=[train_name], outputs=[train_name], concurrency_limit=None)
    test_name.focus(list_dirs, [test_name], [test_name], queue=False)
    test_name_btn.click(updir, inputs=[dataset_src_path], outputs=[test_name], concurrency_limit=None)
    
    dataset_dst_path.focus(list_dirs, [dataset_dst_path], [dataset_dst_path], queue=False)
    distill_dst_btn.click(updir, inputs=[dataset_src_path], outputs=[dataset_dst_path], concurrency_limit=None)

    train_test_split_ratio.change(None, inputs=train_test_split_ratio, outputs=train_test_split_ratio, queue=False)
    cmd_segment_btn.click(segment_data, inputs=[dataset_src_path, train_test_split_ratio, train_name, test_name], outputs=[output_box], concurrency_limit=None)

    def _run(*args):
        yield from engine.runner.run("dataprocess", *args)
    start_btn.click(_run,input_elems, [output_box])
    stop_btn.click(engine.runner.set_abort)

    def update_type(datatype, tn):
        seg = (datatype == "segment")
        dispath = os.path.dirname(tn)
        return gr.update(visible=seg), gr.update(visible=seg), gr.update(visible=seg), \
                gr.update(visible=not seg), gr.update(visible=not seg), gr.update(visible=not seg), \
                gr.update(value=tn) if not seg else gr.update(visible=True), \
                gr.update(value=f"{dispath}") if not seg else gr.update(visible=True)

    dataprocesstype.change(update_type, inputs=[dataprocesstype, train_name],
            outputs=[segment_row, train_test_split_ratio, cmd_segment_btn, 
                     distill_row, start_btn, stop_btn, dataset_src_path, dataset_dst_path],
            queue=False)

    return  elem_dict

