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

import sys
from enum import Enum, unique

from .extras import logging
from .extras.env import VERSION, print_env
from .webui.interface import run_web_ui

USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   deepseekfactory-cli webui: launch LlamaBoard                        |\n"
    + "|   deepseekfactory-cli version: show version info                      |\n"
    + "-" * 70
)

WELCOME = (
    "-" * 58
    + "\n"
    + f"| Welcome to Deepseek Factory, version {VERSION}"
    + " " * (21 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "-" * 58
)

logger = logging.get_logger(__name__)


@unique
class Command(str, Enum):
    ENV = "env"
    WEBUI = "webui"
    VER = "version"
    HELP = "help"


def main():
    command = sys.argv.pop(1) if len(sys.argv) != 1 else Command.HELP
    if command == Command.ENV:
        print_env()
    elif command == Command.WEBUI:
        run_web_ui()
    elif command == Command.VER:
        print(WELCOME)
    elif command == Command.HELP:
        print(USAGE)
    elif command == "dataprocess":
        from .training.distill import run_distill
        run_distill()
    elif command == "sft":
        from .training.sft_train import run_sft
        run_sft()
    elif command == "rl":
        from .training.grpo_train import run_grpo
        run_grpo()
    elif command == "eval":
        from .training.inference import run_inference
        run_inference()
    else:
        raise NotImplementedError(f"Unknown command: {command}.")


if __name__ == "__main__":
    main()
