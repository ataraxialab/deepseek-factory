from typing import TYPE_CHECKING, Dict, List
import os
import shutil   
import gradio as gr

from ..common import WORKSPACE
from ...extras.packages import is_gradio_available
if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from ..engine import Engine

global file_list
file_list = []
def sanitize_path(base_path, user_path):
    # 计算用户提供路径的绝对路径
    abs_path = os.path.abspath(os.path.join(base_path, user_path))
    # 确保路径在 base_path 内
    if not abs_path.startswith(base_path):
        raise ValueError("保存路径无效，不能超出指定目录范围。")
    return abs_path


def upload_files(files: List[gr.File], default_path, save_path):
    try:
        save_path = sanitize_path(default_path, save_path)
    except ValueError as e:
        gr.Info(str(e))
        return

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if not files:
        gr.Info("请选择文件")
        return 

    for file in files:
        dest_path = os.path.join(save_path, os.path.basename(file.name))
        shutil.copy(file.name, dest_path)
        if dest_path not in file_list:
            file_list.append(dest_path)

    gr.Success("文件上传成功")
    return file_list

def create_upload_tab(engine: "Engine") -> Dict[str, "Component"]:
    initArgs = engine.ArgsManager.args
    elem_dict = dict()
    with gr.Row():
        file_input = gr.File(label="拖动或选择文件", file_count="multiple")
    with gr.Row():
        with gr.Column():
            default_path = gr.Textbox(label="默认路径", value=initArgs.get("data_mount_dir", WORKSPACE), interactive=False)
        with gr.Column(scale=9):
            save_path = gr.Textbox(label="保存路径")
        with gr.Column():
            upload_button = gr.Button("上传文件")
    with gr.Row():
        uploaded_files_display = gr.Textbox(label="已上传文件", lines=5, interactive=False, value="\n".join(file_list))

    def handle_upload(files, default_path, save_path):
        uploaded_files = upload_files(files, default_path, save_path)
        return "\n".join(uploaded_files) if uploaded_files else ""

    upload_button.click(handle_upload, inputs=[file_input, default_path, save_path], outputs=[uploaded_files_display])
    return elem_dict

