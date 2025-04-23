import fnmatch
import os
import uuid
from ...extras.packages import is_gradio_available

if is_gradio_available():
    import gradio as gr

def generate_unique_id():
    return str(uuid.uuid4())

def render_item(full_path, item, is_dir):
    icon = "📁" if is_dir else "📄"
    return f'<li class="list-item" data-path="{full_path}"><span class="caret">{icon} {item}</span></li>\n'

def render_back_btn(original_path, not_found = False):
    parent_dir = os.path.dirname(original_path) if not not_found else original_path
    return render_item(parent_dir, "..", True) if parent_dir else ""

def render_filtered_tree(dir_path, keyword):
    html = ""
    for item in sorted(os.listdir(dir_path)):
        full_path = os.path.join(dir_path, item)
        if not (os.path.isfile(full_path) or os.path.isdir(full_path)):
            continue
        is_dir = os.path.isdir(full_path)
        if fnmatch.fnmatch(item.lower(), f"*{keyword.lower()}*"):
            html += render_item(full_path, item, is_dir)
    if not html and keyword:
        parent_dir = os.path.dirname(dir_path)
        if os.path.exists(parent_dir):
            html += render_back_btn(dir_path, True)
    return html

def render(dir_path, original_path):
    try:
        if not os.path.exists(dir_path):
            parent_dir = os.path.dirname(dir_path)
            if not os.path.exists(parent_dir):
                return ""

            keyword = os.path.basename(dir_path)
            return render_filtered_tree(parent_dir, keyword)

        html = render_back_btn(original_path)
        html += render_filtered_tree(dir_path, "")
        return html
    except Exception as e:
        print(f"Error rendering tree: {e}")
        return ""


def generate_directory_tree_html(path):
    parent_dir = os.path.dirname(path) if os.path.isfile(path) else path
    content = render(parent_dir, path)
    tree_html = f'<ul class="list-container">\n{content}</ul>'
    return tree_html if content else "", not os.path.isfile(path)


def on_change(path):
    html, is_dir = generate_directory_tree_html(path.strip())
    tree_update = gr.update(value=html, visible=True) if is_dir else gr.update(visible=False)
    button_update = gr.update(visible=True) if is_dir and html != "" else gr.update(visible=False)
    return tree_update, button_update

def on_show(path):
    html, _ = generate_directory_tree_html(path.strip())
    return gr.update(value=html, visible=True), gr.update(visible=True) if html != "" else gr.update(visible=False)

def hide_tree():
    return gr.update(visible=False), gr.update(visible=False)


def create_dir_selector(base_path: str = "", label: str = "") -> gr.Blocks:
    unique_id = generate_unique_id()
    with gr.Column():
        path_input = gr.Textbox(
            elem_id=f"path_input_{unique_id}",
            value=base_path,
            elem_classes="path_input",
            label=label
        )
        tree_output = gr.HTML(
            generate_directory_tree_html(base_path),
            elem_id=f"{unique_id}",
            elem_classes="tree",
            visible=False
        )
        confirmbtn = gr.Button(
            elem_classes=["overlay-button"],
            variant="secondary",
            value="选择",
            scale=0,
            min_width=20,
            visible=False
        )
        path_input.input(fn=on_change, inputs=[path_input], outputs=[tree_output, confirmbtn])
        path_input.focus(fn=on_show, inputs=[path_input], outputs=[tree_output, confirmbtn])
        confirmbtn.click(fn=hide_tree, inputs=[], outputs=[tree_output, confirmbtn])
    return path_input
