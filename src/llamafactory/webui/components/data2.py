import json
import os
from typing import TYPE_CHECKING, Dict, Tuple

from ...extras.packages import is_gradio_available

import gradio as gr
if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component


PAGE_SIZE = 1
data_cache = []

def load_data(dataset: str) -> Tuple[int, list]:
    global data_cache
    if not dataset:
        return 0, []
    
    try:
        with open(dataset, "r", encoding="utf-8") as f:
            if dataset.endswith(".json"):
                data_cache = json.load(f)
            elif dataset.endswith(".jsonl"):
                data_cache = [json.loads(line) for line in f if line.strip()]
            else:
                data_cache = list(f)
    except Exception as e:
        print("加载失败2:", e)
        data_cache = []

    print(f"Loading {dataset} with len {len(data_cache)}")
    return len(data_cache), data_cache[:PAGE_SIZE]


def get_preview(page_index: int) -> Tuple[int, list]:
    """ 仅从 data_cache 获取数据 """
    start = page_index * PAGE_SIZE
    end = min(start + PAGE_SIZE, len(data_cache))
    return len(data_cache), data_cache[start:end]


def delete_current_page(page_index: int) -> Tuple[int, list]:
    """ 删除当前页数据，并更新页码 """
    global data_cache
    start = page_index * PAGE_SIZE
    end = min(start + PAGE_SIZE, len(data_cache))

    if start < len(data_cache):
        del data_cache[start:end]  # 直接从缓存删除

    new_page_index = min(page_index, max(0, (len(data_cache) - 1) // PAGE_SIZE))
    return new_page_index, get_preview(new_page_index)[1]


def save_data(dataset):
    """ 覆盖保存 data_cache 到原文件 """
    try:
        with open(dataset, "w", encoding="utf-8") as f:
            print(f"Saving {dataset} with len {len(data_cache)}")
            json.dump(data_cache, f, ensure_ascii=False, indent=4)
        gr.Success("保存成功")
    except Exception as e:
        gr.Error(f"保存失败: {e}")


def save_data_as(source_dataset: str, dest_dataset: str):
    dataset_dir = os.path.dirname(source_dataset)
    if not dest_dataset:
        gr.Error("请输入文件名")
        return
    filename = dest_dataset
    # 确保文件名以 .json 结尾
    if not filename.endswith(".json"):
        filename += ".json"

    data_path = os.path.join(dataset_dir, filename)
    try:
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data_cache, f, ensure_ascii=False, indent=4)
    except Exception as e:
        gr.Error(f"另存为失败: {e}")
        return

    dataset_info_path = os.path.join(dataset_dir, "dataset_info.json")
    if not os.path.exists(dataset_info_path):
        gr.Error("不存在 dataset_info.json")
        return None
    
    with open(dataset_info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    original_entry = dataset_info.get(source_dataset, {}).copy()
    original_entry["file_name"] = filename
    # 4. **按照相同格式添加新条目**
    dataset_info[dest_dataset] = original_entry

    # 5. **写回 dataset_info.json**
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    gr.Success(f"成功另存为 {filename}")

def change_dataset(dataset: "gr.Dropdown"):
    return gr.update(interactive=True, visible=True), gr.update(choices=(lambda x: x if isinstance(x, list) else [x])(dataset), value=(lambda x: x[0] if isinstance(x, list) else x)(dataset), multiselect=False)

def create_preview_box(dataset: "gr.Dropdown", do_edit: bool = False) -> Dict[str, "Component"]:
    data_preview_btn = gr.Button(value="Preview dataset", interactive=False, scale=1, visible=False)
    with gr.Column(visible=False, elem_classes="modal-box", min_width=800) as preview_box:
        with gr.Row():
            dataset_selector = gr.Dropdown(label="选择数据集")

        with gr.Row():
            preview_count = gr.Number(value=0, interactive=False, precision=0, elem_classes="hidden-border")
            page_index = gr.Number(value=0, interactive=True, precision=0)

        with gr.Row(visible=do_edit):
            delete_btn = gr.Button(value="删除本条数据", interactive=True)
            save_btn = gr.Button(value="保存", interactive=True)
            save_as_btn = gr.Button(value="另存为", interactive=True)
            save_as_input = gr.Textbox(value="", placeholder="输入新文件名", show_label=False, elem_classes="hidden-border")

        with gr.Row():
            preview_samples = gr.JSON(min_width=800, min_height=300)
        
        with gr.Row():
            prev_btn = gr.Button(value="Previous Page")
            next_btn = gr.Button(value="Next Page")
            close_btn = gr.Button(value="Close")

    dataset.change(change_dataset, [dataset], [data_preview_btn, dataset_selector], queue=False).then(
        lambda: 0, outputs=[page_index], queue=False
    )
    #dataset_selector.change(
    #    load_data, [dataset_selector], [preview_count, preview_samples], queue=False
    #)
    data_preview_btn.click(
        load_data, [dataset_selector], [preview_count, preview_samples], queue=False
    ).then(
        lambda: gr.update(visible=True), outputs=[preview_box], queue=False
    )

    prev_btn.click(
        lambda x: max(0, x - 1), [page_index], [page_index], queue=False
    ).then(
        get_preview, [page_index], [preview_count, preview_samples], queue=False
    )

    next_btn.click(
        lambda x, count: min(x + 1, (count - 1) // PAGE_SIZE), [page_index, preview_count], [page_index], queue=False
    ).then(
        get_preview, [page_index], [preview_count, preview_samples], queue=False
    )

    page_index.change(
        lambda x, count: min(max(0, int(x)), (count - 1) // PAGE_SIZE),
        [page_index, preview_count],
        [page_index],
        queue=False
    ).then(
        get_preview, [page_index], [preview_count, preview_samples], queue=False
    )

    close_btn.click(lambda: gr.update(visible=False), outputs=[preview_box], queue=False)

    delete_btn.click(
        delete_current_page, [page_index], [page_index, preview_samples], queue=False
    ).then(
        lambda: len(data_cache), [], [preview_count], queue=False
    )

    save_btn.click(
        save_data, [dataset_selector], concurrency_limit=None
    )

    save_as_btn.click(
        save_data_as, [dataset_selector, save_as_input], concurrency_limit=None
    )

    return dict(
        data_preview_btn=data_preview_btn,
        preview_count=preview_count,
        page_index=page_index,
        prev_btn=prev_btn,
        next_btn=next_btn,
        close_btn=close_btn,
        delete_btn=delete_btn,
        preview_samples=preview_samples,
        save_btn=save_btn,
        save_as_btn=save_as_btn,
        save_as_input=save_as_input,
    )
