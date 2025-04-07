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

def render_back_btn(original_path):
    parent_dir = os.path.dirname(original_path)
    return render_item(parent_dir, "..", True) if parent_dir else ""

def render_filtered_tree(dir_path, keyword, show_dirs):
    html = ""
    for item in sorted(os.listdir(dir_path)):
        full_path = os.path.join(dir_path, item)
        if not (os.path.isfile(full_path) or os.path.isdir(full_path)):
            continue
        is_dir = os.path.isdir(full_path)
        if fnmatch.fnmatch(item.lower(), f"*{keyword.lower()}*") and (not show_dirs or is_dir):
            html += render_item(full_path, item, is_dir)
    return html

def render(dir_path, show_dirs, original_path):
    try:
        if not os.path.exists(dir_path):
            parent_dir = os.path.dirname(dir_path)
            if not os.path.exists(parent_dir):
                return ""

            keyword = os.path.basename(dir_path)
            return render_filtered_tree(parent_dir, keyword, show_dirs)

        html = render_back_btn(original_path)
        html += render_filtered_tree(dir_path, "", show_dirs)
        return html
    except Exception as e:
        print(f"Error rendering tree: {e}")
        return ""


def generate_directory_tree_html(path, show_dirs):
    parent_dir = os.path.dirname(path) if os.path.isfile(path) else path
    content = render(parent_dir, show_dirs, path)
    tree_html = f'<ul class="list-container">\n{content}</ul>'
    return tree_html if content else "", not os.path.isfile(path)


def on_change(path, show_dirs):
    html, is_dir = generate_directory_tree_html(path.strip(), show_dirs)
    tree_update = gr.update(value=html, visible=True) if is_dir else gr.update(visible=False)
    button_update = gr.update(visible=True) if is_dir and html != "" and show_dirs else gr.update(visible=False)
    return tree_update, button_update

def on_show(path, show_dirs):
    html, _ = generate_directory_tree_html(path.strip(), show_dirs)
    return gr.update(value=html, visible=True), gr.update(visible=True) if html != "" and show_dirs else gr.update(visible=False)

def hide_tree():
    return gr.update(visible=False), gr.update(visible=False)


def create_path_selector(base_path: str = "", label: str = "", showDirs: bool = False) -> gr.Blocks:
    unique_id = generate_unique_id()
    with gr.Column():
        path_input = gr.Textbox(
            elem_id=f"path_input_{unique_id}",
            value=base_path,
            elem_classes="path_input",
            label=label
        )
        tree_output = gr.HTML(
            generate_directory_tree_html(base_path, showDirs),
            elem_id=f"{unique_id}",
            elem_classes="tree",
            visible=False
        )
        show_dirs = gr.Checkbox(
            value=showDirs,
            elem_classes=["show_dirs"],
            visible=False
        )
        confirmbtn = gr.Button(
            elem_classes=["overlay-button"],
            variant="secondary",
            value="选择该路径",
            scale=0,
            min_width=20,
            visible=False
        )
        path_input.input(fn=on_change, inputs=[path_input, show_dirs], outputs=[tree_output, confirmbtn])
        path_input.focus(fn=on_show, inputs=[path_input, show_dirs], outputs=[tree_output, confirmbtn])
        confirmbtn.click(fn=hide_tree, inputs=[], outputs=[tree_output, confirmbtn])
    return path_input