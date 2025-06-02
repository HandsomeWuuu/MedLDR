# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    log_file: Optional[str] = field(
        default=None,
        metadata={"help": ("The file to log to.")},
    )
# from dataclasses import dataclass,field
# from typing import List



@dataclass
class data_config:
    train_data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_data_path: str = field(default=None,
                                metadata={"help": "Path to the evaluation data."})
    test_data_path: str = field(default=None,
                                metadata={"help": "Path to the test data."})
    input_key: str = field(default="instruction",metadata={"help": "Key for the input data."})
    output_key: str = field(default="output",metadata={"help": "Key for the output data."})
    training_type: str = field(default="sft",metadata={"help": "Type of training."})
    
Task_Mapping_Table = {
    "lab": ["lab"],
    "lab+": ["lab", "ed_info", "family_history", "ed_chiefcomplaint","past_medical_history"],
    "lab_num": ["lab", "ed_info"],
    "lab_text": ["lab", "family_history", "ed_chiefcomplaint"],
    "lab_num+": ["lab", "ed_info", "micro_info"],
    "lab_text+": ["lab", "family_history", "ed_chiefcomplaint", "past_medical_history"],
    "all":  ["lab", "ed_info", "family_history", "ed_chiefcomplaint", "past_medical_history","physical_text","report"]
}

cut_off_len_dict = {
    "lab": 2048,
    "ed_info": 128, # max 72   
    "family_history": 512,
    "ed_chiefcomplaint": 32, # max 24
    "past_medical_history": 2048, # max 2613
    "physical_text": 1024, # max 3136
    "report": 2048 # 11067
}
