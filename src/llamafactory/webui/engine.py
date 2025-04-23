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

import json
import os
from typing import TYPE_CHECKING, Any, Dict

from .common import create_ds_config, ArgsManager, get_init_config_path
from .locales import LOCALES
from .manager import Manager
from .runner import Runner


if TYPE_CHECKING:
    from gradio.components import Component

def get_args():
    file_path = get_init_config_path()
    if not os.path.exists(file_path):
        return {}
    with open(file_path, encoding="utf-8") as f:
        file_content = f.read()
    return json.loads(file_content)

class Engine:
    r"""
    A general engine to control the behaviors of Web UI.
    """

    def __init__(self, demo_mode: bool = False, pure_chat: bool = False) -> None:
        self.demo_mode = demo_mode
        self.pure_chat = pure_chat
        self.manager = Manager()
        args = ArgsManager(get_args())
        self.runner = Runner(self.manager, demo_mode, args)
        self.ArgsManager = args
        if not demo_mode:
            create_ds_config()

    def _update_component(self, input_dict: Dict[str, Dict[str, Any]]) -> Dict["Component", "Component"]:
        r"""
        Updates gradio components according to the (elem_id, properties) mapping.
        """
        output_dict: Dict["Component", "Component"] = {}
        for elem_id, elem_attr in input_dict.items():
            elem = self.manager.get_elem_by_id(elem_id)
            output_dict[elem] = elem.__class__(**elem_attr)

        return output_dict

    def resume_x(self):
        r"""
        Gets the initial value of gradio components and restores training status if necessary.
        """
        init_dict = {"global.lang": {"value": "zh"}} if self.manager.get_elem_by_id_safe("global.lang") else {}
        yield self._update_component(init_dict)

        if self.runner.running and not self.demo_mode and not self.pure_chat:
            yield {elem: elem.__class__(value=value) for elem, value in self.runner.running_data.items()}

    def change_lang(self, lang: str):
        r"""
        Updates the displayed language of gradio components.
        """
        return {
            elem: elem.__class__(**LOCALES[elem_name][lang])
            for elem_name, elem in self.manager.get_elem_iter()
            if elem_name in LOCALES
        }

    def change_lang_ex(self, lang: str, ex: set[str]):
        r"""
        Updates the displayed language of gradio components.
        """
        return {
            elem: elem.__class__(**{
                k: v for k, v in LOCALES[elem_name][("en" if elem_name in ex else lang)].items()
            })
            for elem_name, elem in self.manager.get_elem_iter()
            if elem_name in LOCALES
        }
