from typing import TYPE_CHECKING, Dict

from llamafactory.webui.locales import ALERTS
from ...extras.packages import is_gradio_available
from ...extras.packages import is_gradio_available
from ..control import list_files, updir
from .data2 import create_preview_box
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

def segment_data(dataset, ratio, lang):
    data = load_data(dataset)
    random.shuffle(data)
    split_index = int(len(data) * ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    name, ext = os.path.splitext(dataset)
    train_name, test_name = f"{name}_train{ext}", f"{name}_test{ext}"
    print(f"Segmenting {dataset} into {train_name} and {test_name}")
    with open(train_name, 'w', encoding='utf-8') as f_train:
        json.dump(train_data, f_train, ensure_ascii=False, indent=4)
    with open(test_name, 'w', encoding='utf-8') as f_test:
        json.dump(test_data, f_test, ensure_ascii=False, indent=4)
    print(f"Segment done")

    finish_info = ALERTS["info_aborted"][lang]
    gr.Info(finish_info)
    return gr.update(value=f"生成训练集{train_name}和测试集{test_name}"), gr.update(value=f"{name}_train{ext}")

def create_segment_tab(engine: "Engine") -> Dict[str, "Component"]:
    initArgs = engine.ArgsManager.args
    elem_dict = dict()
    lang = engine.manager.get_elem_by_id("top.lang")

    with gr.Row():
        dataset = gr.Dropdown(label="Dataset", info="", multiselect=False, allow_custom_value=True, value=initArgs.get("data_mount_dir", "/openr1_data"), scale=9)
        upbtn= gr.Button(variant="secondary", value="..", scale=1)
    with gr.Row():
        preview_elems = create_preview_box(dataset)

    elem_dict.update(
    dict(
        dataset=dataset,
        upbtn=upbtn,
        **preview_elems
    )
    )

    with gr.Row():
        train_test_split_ratio = gr.Slider(label="Dataset Split", minimum=0, maximum=1, value=0.8, step=0.01, scale=3)
    with gr.Row():
        cmd_segment_btn = gr.Button(variant="primary")

    with gr.Row():
        dataset_out = gr.Textbox(label="dataset_out", info=None, visible=False, interactive=False, scale=1, value="")
    with gr.Row():
        output_box = gr.Markdown(elem_classes="scroll-box", value="info")
    elem_dict.update(
        dict(
            output_box=output_box,
            dataset_out=dataset_out,
            train_test_split_ratio=train_test_split_ratio,
            cmd_segment_btn=cmd_segment_btn
        )
    )

    dataset.focus(list_files, [dataset], [dataset], queue=False)
    upbtn.click(updir, inputs=[dataset], outputs=[dataset], concurrency_limit=None)

    train_test_split_ratio.change(None, inputs=train_test_split_ratio, outputs=train_test_split_ratio, queue=False)
    cmd_segment_btn.click(segment_data, inputs=[dataset, train_test_split_ratio, lang], outputs=[output_box, dataset_out], concurrency_limit=None)

    return  elem_dict
