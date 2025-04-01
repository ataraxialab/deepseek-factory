from typing import TYPE_CHECKING, Dict

from ...extras.packages import is_gradio_available
from ...extras.packages import is_gradio_available
from ..control import list_files, updir
from .data2 import create_preview_box

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
            dataset = gr.Dropdown(label="Dataset", multiselect=False, allow_custom_value=True, value=initArgs.get("data_mount_dir", "/openr1_data"), scale=9)
            upbtn = gr.Button(elem_classes=["overlay-button"], variant="secondary", value="..", scale=0, min_width=20)
            dataset_out = gr.Textbox(label="dataset_out", visible=False, interactive=False, scale=1, value="")
    with gr.Row():
        preview_elems = create_preview_box(dataset, True)
    elem_dict.update(
        dict(
            dataset=dataset,
            upbtn=upbtn,
            dataset_out=dataset_out,
            **preview_elems
        )
    )
    dataset.focus(list_files, [dataset], [dataset], queue=False)
    upbtn.click(updir, inputs=[dataset], outputs=[dataset], concurrency_limit=None)

    return  elem_dict

