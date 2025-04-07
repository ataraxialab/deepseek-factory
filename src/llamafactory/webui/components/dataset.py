from typing import TYPE_CHECKING, Dict

from ...extras.packages import is_gradio_available
from ...extras.packages import is_gradio_available
from ..control import list_files, updir
from .data2 import create_preview_box
from .selector import create_path_selector

import gradio as gr
if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine

def dataset_tab(engine: "Engine") -> Dict[str, "Component"]:
    initArgs = engine.ArgsManager.args
    elem_dict = dict()

    with gr.Row():
        with gr.Column(scale=9, elem_classes=["dropdown-button-container"]):
            dataset = create_path_selector(base_path=initArgs.get("data_mount_dir", "/openr1_data"), label="Dataset")
            dataset_out = gr.Textbox(label="dataset_out", visible=False, interactive=False, scale=1, value="")
    with gr.Row():
        preview_elems = create_preview_box(dataset, True)
    elem_dict.update(
        dict(
            dataset=dataset,
            dataset_out=dataset_out,
            **preview_elems
        )
    )

    return  elem_dict

