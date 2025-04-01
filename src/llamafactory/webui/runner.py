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
from copy import deepcopy
from subprocess import Popen, TimeoutExpired
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional

from ..extras.constants import LLAMABOARD_CONFIG, TRAINING_STAGES, RUNNING_LOG
from ..extras.misc import torch_gc
from ..extras.packages import is_gradio_available
from .common import (
    abort_process,
    _clean_cmd,
    ArgsManager,
    get_init_config,
    get_save_dir,
    load_args,
    save_cmd,
)
from .control import get_trainer_info, get_trainer_info_x
from .locales import ALERTS, LOCALES
from yaml import safe_dump


if is_gradio_available():
    import gradio as gr


if TYPE_CHECKING:
    from gradio.components import Component

    from .manager import Manager


class Runner:
    r"""
    A class to manage the running status of the trainers.
    """

    def __init__(self, manager: "Manager", demo_mode: bool = False, args: Optional[ArgsManager] = None) -> None:
        self.manager = manager
        self.args = args
        self.demo_mode = demo_mode
        """ Resume """
        self.trainer: Optional["Popen"] = None
        self.do_train = True
        self.kind = ""
        self.running_data: Dict["Component", Any] = None
        """ State """
        self.aborted = False
        self.running = False
        self._cli = "deepseekfactory-cli"
        #self.default_cmd = [self._cli, "train"]
        self.default_cmd = [self._cli]

    def set_abort(self) -> None:
        self.aborted = True
        if self.trainer is not None:
            abort_process(self.trainer.pid)

    def _finalize(self, lang: str, finish_info: str) -> str:
        r"""
        Cleans the cached memory and resets the runner.
        """
        finish_info = ALERTS["info_aborted"][lang] if self.aborted else finish_info
        gr.Info(finish_info)
        self.trainer = None
        self.aborted = False
        self.running = False
        self.running_data = None
        torch_gc()
        return finish_info

    def cast_value(self, value) -> Any:
        try:
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            elif value.isdigit():
                return int(value)
            return float(value)
        except Exception:
            return value

    def _build_config_dict(self, kind, data: Dict["Component", Any]) -> Dict[str, Any]:
        r"""
        Builds a dictionary containing the current training configuration.
        """
        config_dict = {}
        skip_ids = [f"{kind}.more_params", f"{kind}.config_path", f"{kind}.training_stage"]
        skip_ids.extend(self.manager.get_base_elems_list())
        if data:
            for elem, value in data.items():
                elem_id = self.manager.get_id_by_elem(elem)
                if elem_id not in skip_ids:
                    param = elem_id.split(".")[-1]
                    config_dict[param] = self.cast_value(value)

        more_params = self.manager.get_elem_by_id_safe(f"{kind}.more_params")
        if more_params:
            buf = data[more_params]
            if buf:
                for l in buf.split("\n"):
                    parts = l.strip().split(":")
                    if len(parts) == 2 and parts[0].strip() not in config_dict:
                        config_dict[parts[0].strip()] = self.cast_value(parts[1].strip())

        return _clean_cmd(config_dict)

    def preview(self, kind, data):
        yield from self._preview(data, kind)

    def _save_args(self, config_path: str, config_dict: Dict[str, Any]) -> None:
        r"""
        Saves the training configuration to config path.
        """
        with open(config_path, "w", encoding="utf-8") as f:
            #safe_dump(config_dict, f, default_style=None, allow_unicode=True, default_flow_style=False)
            safe_dump(config_dict, f, allow_unicode=True, default_flow_style=None, sort_keys=False)

    def save_args(self, kind: str, data: Dict["Component", Any]):
        r"""
        Saves the training configuration to config path.
        """
        output_box = self.manager.get_elem_by_id(f"{kind}.output_box")
        error = self._initialize(data, kind, from_preview=True)
        if error:
            gr.Warning(error)
            return {output_box: error}

        le = self.manager.get_elem_by_id_safe("global.lang")
        lang = data[le] if le and le in data else "zh"
        cp = self.manager.get_elem_by_id_safe(f"{kind}.config_path")
        if not cp:
            stre = f"Not {kind} config path defined."
            gr.Warning(stre)
            return {output_box: stre}
        config_path = data[cp]
        _, ext = os.path.splitext(config_path)
        if ext is None or ext == "":
            stre = "Should have a file extension."
            gr.Warning(stre)
            return {output_box: stre}
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
        except Exception as e:
            stre = str(e)
            gr.Warning(stre)
            return {output_box: stre}

        self._save_args(config_path, self._build_config_dict(kind, data))
        return {output_box: ALERTS["info_config_saved"][lang] + config_path}

    def load_args(self, kind: str, data: Dict["Component", Any]):
        r"""
        Loads the training configuration from config path.
        """
        output_box = self.manager.get_elem_by_id(f"{kind}.output_box")

        more_params = self.manager.get_elem_by_id_safe(f"{kind}.more_params")
        cp = self.manager.get_elem_by_id_safe(f"{kind}.config_path")
        if cp is None:
            gr.Warning(ALERTS["err_config_not_found"]["zh"])
            return {output_box: ALERTS["err_config_not_found"]["zh"]}

        config_path = data[cp]
        config_dict = load_args(config_path)
        if config_dict is None:
            gr.Warning(ALERTS["err_config_not_found"]["zh"])
            return {output_box: ALERTS["err_config_not_found"]["zh"]}

        output_dict: Dict["Component", Any] = {}
        lines: list[str] = []
        for elem_id, value in config_dict.items():
            e = self.manager.get_elem_by_id_safe(f"{kind}.{elem_id}")
            if e and e in data:
                output_dict[e] = value
            else:
                lines.append(f"{elem_id}: {value}")

        if more_params:
            output_dict[more_params] = "\n".join(lines)

        output_dict[output_box] = ALERTS["info_config_loaded"]["zh"]
        return output_dict

    def check_output_dir(self, lang: str, model_name: str, finetuning_type: str, output_dir: str):
        r"""
        Restore the training status if output_dir exists.
        """
        output_box = self.manager.get_elem_by_id("train.output_box")
        output_dict: Dict["Component", Any] = {output_box: LOCALES["output_box"][lang]["value"]}
        if model_name and output_dir and os.path.isdir(get_save_dir(model_name, finetuning_type, output_dir)):
            gr.Warning(ALERTS["warn_output_dir_exists"][lang])
            output_dict[output_box] = ALERTS["warn_output_dir_exists"][lang]

            output_dir = get_save_dir(model_name, finetuning_type, output_dir)
            config_dict = load_args(os.path.join(output_dir, LLAMABOARD_CONFIG))  # load llamaboard config
            for elem_id, value in config_dict.items():
                output_dict[self.manager.get_elem_by_id(elem_id)] = value

        return output_dict


    def get_cmd_func(self, kind, stage = None) -> list[str]:
        s = stage if stage is not None else kind
        cmd = get_init_config(self.args.args, f"{s}.cmd", None)
        return cmd.split() + ["--config"] if cmd else self.default_cmd + [f"{s}", "--config"]

    def get_trainer_info_func(self, kind):
        c = self.manager.get_elem_by_id_safe(f"{kind}.training_stage")
        stage = TRAINING_STAGES[self.running_data[c]] if c else None
        cmds = self.get_cmd_func(kind, stage)
        return get_trainer_info if (len(cmds) > 0 and cmds[0] == self._cli) else get_trainer_info_x

    def run(self, kind, data):
        yield from self._launch(data, kind)

    def gen_cmd(self, args: Dict[str, Any], kind: str, stage = None) -> str:
        r"""
        Generates CLI commands for previewing.
        """
        cmd_lines: list[str] = []
        ls = self.get_cmd_func(kind, stage)
        cmd_lines.extend(ls)
        for k, v in args.items():
            if isinstance(v, dict):
                cmd_lines.append(f"    --{k} {json.dumps(v, ensure_ascii=False)} ")
            elif isinstance(v, list):
                cmd_lines.append(f"    --{k} {' '.join(map(str, v))} ")
            else:
                cmd_lines.append(f"    --{k} {str(v)} ")

        cmd_text = "\\\n".join(cmd_lines)

        cmd_text = f"```bash\n{cmd_text}\n```"
        return cmd_text


    def _initialize(self, data: Dict["Component", Any], kind: str, from_preview: bool) -> str:
        if self.running:
            return ALERTS["err_conflict"]["zh"]

        if not self.manager.get_elem_by_id_safe(f"{kind}.output_dir"):
            return ALERTS["err_no_output_dir"]["zh"]

        return ""

    def _preview(self, data: Dict["Component", Any], kind: str):
        r"""
        Previews the training commands.
        """
        output_box = self.manager.get_elem_by_id(f"{kind}.output_box")
        error = self._initialize(data, kind, True)
        if error:
            gr.Warning(error)
            yield {output_box: error}
            return

        self.kind, self.running_data = kind, data
        odir = self.manager.get_elem_by_id_safe(f"{kind}.output_dir")
        if odir:
            os.makedirs(data[odir], exist_ok=True)
        ts = self.manager.get_elem_by_id_safe(f"{kind}.training_stage")
        stage = TRAINING_STAGES[data[ts]] if ts else None
        yield {output_box: self.gen_cmd(self._build_config_dict(kind, data), kind, stage)}

    def _launch(self, data: Dict["Component", Any], kind: str) -> Generator[Dict["Component", Any], None, None]:
        r"""
        Starts the training process.
        """
        output_box = self.manager.get_elem_by_id(f"{kind}.output_box")
        error = self._initialize(data, kind, False)
        if error:
            gr.Warning(error)
            yield {output_box: error}
            return

        self.kind, self.running_data = kind, data
        args = self._build_config_dict(kind, data)

        c = self.manager.get_elem_by_id_safe(f"{kind}.training_stage")
        stage = TRAINING_STAGES[data[c]] if c else None
        try:
            os.makedirs(args["output_dir"], exist_ok=True)
        except Exception as e:
            err = ALERTS["err_failed"]["zh"] + f" {e}"
            gr.Warning(err)
            yield {output_box: err}
            return

        cmds = self.get_cmd_func(kind, stage) + [save_cmd(_clean_cmd(args))]
        print(f"Running {kind} with cmd: {cmds}")

        env = deepcopy(os.environ)
        try:
            log_path = os.path.join(args["output_dir"], RUNNING_LOG)
            with open(log_path, "w", encoding="utf-8") as f:
                self.trainer = Popen(cmds, stdout=f, stderr=f, env=env)
        except Exception as e:
            err = ALERTS["err_failed"]["zh"] + f" {e}"
            gr.Warning(err)

        yield from self.monitor()

    def monitor(self):
        r"""
        Monitors the training progress and logs.
        """
        self.aborted = False
        self.running = True

        le = self.manager.get_elem_by_id_safe("global.lang")
        lang = self.running_data[le] if le and le in self.running_data else "zh"
        get = lambda elem_id: self.running_data[self.manager.get_elem_by_id(elem_id)]
        output_dir = get(f"{self.kind}.output_dir")
        output_path = get_save_dir(output_dir)

        output_box = self.manager.get_elem_by_id(f"{self.kind}.output_box")
        progress_bar = self.manager.get_elem_by_id_safe(f"{self.kind}.progress_bar")
        loss_viewer = self.manager.get_elem_by_id_safe(f"{self.kind}.loss_viewer")

        running_log = ""
        while self.trainer is not None:
            if self.aborted:
                return_dict = {
                    output_box: ALERTS["info_aborting"][lang],
                }
                if progress_bar is not None:
                    return_dict[progress_bar] = gr.Slider(visible=False)
                yield return_dict

            else:
                c = self.manager.get_elem_by_id_safe(f"{self.kind}.training_stage")
                stage = TRAINING_STAGES[self.running_data[c]] if c else None
                cmds = self.get_cmd_func(self.kind, stage) 

                running_log, running_progress, running_loss = \
                        get_trainer_info_x(output_path, self.kind)
                return_dict = {
                    output_box: running_log,
                }
                if progress_bar is not None and running_progress is not None:
                    return_dict[progress_bar] = running_progress
                if running_loss is not None and loss_viewer is not None:
                    return_dict[loss_viewer] = running_loss

                yield return_dict

            try:
                self.trainer.wait(2)
                self.trainer = None
            except TimeoutExpired:
                continue

        finish_info = ALERTS["info_finished"][lang]

        return_dict = {
            output_box: "```\n{}\n\n{}\n```".format(self._finalize(lang, finish_info), running_log),
        }
        if progress_bar is not None:
            return_dict[progress_bar] = gr.Slider(visible=False)

        yield return_dict

