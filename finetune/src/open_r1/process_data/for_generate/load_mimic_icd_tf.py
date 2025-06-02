from torch.utils.data import Dataset
import torch
import transformers
import json
import os
import re
import random
import pandas as pd

from open_r1.configs import data_config as DataArguments
from open_r1.configs import Task_Mapping_Table, cut_off_len_dict

from open_r1.utils import rank0_print


from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
from .organize_data import top50_system_message

@dataclass
class DataCollatorForIcdDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, attention_mask, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids","attention_mask", "labels")
        )

        # 可以直接 pad 的，数据中不含 list

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0  # This gets the best result. Don't know why.
    

        input_ids = [torch.tensor(_input_ids[: self.tokenizer.model_max_length]) for _input_ids in input_ids]
        attention_mask = [torch.tensor(_attention_mask[: self.tokenizer.model_max_length]) for _attention_mask in attention_mask]
        # labels = [torch.tensor(_labels[: self.tokenizer.model_max_length]) for _labels in labels]
        labels = [_labels.clone().detach() for _labels in labels]

        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = self.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        

        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return batch
    
class MIMICICDSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(MIMICICDSupervisedDataset, self).__init__()

        # 读取数据
        self.list_data_dict = []
        self.tokenizer = tokenizer
        self.data_args = data_args


        rank0_print(f"Loading {data_path}")

        with open(data_path, "r") as file:
            try:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)
            except json.JSONDecodeError:
                file.seek(0)
                for line in file:
                    cur_data_dict = json.loads(line)
                    self.list_data_dict.append(cur_data_dict)
                rank0_print(f"Loaded {len(self.list_data_dict)} sample from {data_path}")

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")

        
        # Shuffle dataset
        rank0_print("Shuffling dataset...")
        random.shuffle(self.list_data_dict)

        # Print Case:
        self.input_list = Task_Mapping_Table[self.data_args.input_key]

        rank0_print('input_list:', self.input_list)
        selected_inputs = {key: self.list_data_dict[0]['input_dict'][key] for key in self.input_list}
        rank0_print('Selected inputs case :', selected_inputs)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.generate_and_tokenize_prompt(sources)
        data_dict['case_id'] = sources['case_id']

        return data_dict

    # def tokenize(self, prompt, add_eos_token=True):
    #     result = self.tokenizer(
    #         prompt,
    #         truncation=True,
    #         max_length=self.train_args.cutoff_len,
    #         padding=False,
    #         return_tensors=None,
    #     )
    #     # Add EOS token if not present
    #     if (
    #         result["input_ids"][-1] != self.tokenizer.eos_token_id
    #         and len(result["input_ids"]) < self.train_args.cutoff_len
    #         and add_eos_token
    #     ):
    #         result["input_ids"].append(self.tokenizer.eos_token_id)
    #         result["attention_mask"].append(1)
        
    #     # without mask
    #     result["labels"] = result["input_ids"].copy()

    #     return result
    
    def single_tokenize(self, prompt, cutoff_len, add_start_token =False, add_eos_token=False):
        if cutoff_len is None:
            result = self.tokenizer(
                prompt,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
        else:
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
        
        if not add_start_token:
            result["input_ids"] = result["input_ids"][1:]
            result["attention_mask"] = result["attention_mask"][1:]
    
         # Add EOS token if not present
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        # without mask
        result["labels"] = result["input_ids"].copy()
        return result
    
    def generate_and_tokenize_prompt(self, data_point):
        input_list = self.input_list

        input_prompt = top50_system_message
        input_tokenizer = self.single_tokenize(input_prompt, cutoff_len=None, add_start_token=True)

        print('prompt tokenizer len:', len(input_tokenizer['input_ids']))
        tokenized_full_prompt = {
            "input_ids": input_tokenizer['input_ids'][:],
            "attention_mask": input_tokenizer['attention_mask'][:],
            "labels": input_tokenizer['labels'][:]
        }

        # 1. 构造输入
        for key in input_list:
            cut_off_len = cut_off_len_dict[key]
            if isinstance(data_point['input_dict'][key], list):
                single_tokenized = self.single_tokenize('The report info:\n' + '\n---\n'.join(data_point['input_dict'][key]), cut_off_len)
            else:
                single_tokenized = self.single_tokenize(data_point['input_dict'][key], cut_off_len)

            # 将input_ids  decode 出来，打印出来
            input_ids = single_tokenized['input_ids']
            input_text = self.tokenizer.decode(input_ids)
            rank0_print(f'key : {key}, input_text: {input_text}')
            
            tokenized_full_prompt['input_ids'].extend(single_tokenized['input_ids'])
            tokenized_full_prompt['attention_mask'].extend(single_tokenized['attention_mask'])
            tokenized_full_prompt['labels'].extend(single_tokenized['labels'])

        ## 在最后加一个 Answer: token
        answer_token = self.single_tokenize("Answer:",cutoff_len=None)
        tokenized_full_prompt['input_ids'].extend(answer_token['input_ids'])
        tokenized_full_prompt['attention_mask'].extend(answer_token['attention_mask'])
        tokenized_full_prompt['labels'].extend(answer_token['labels'])

        # 2. 构造输出
        ## 先将之前的置为 -100
        tokenized_full_prompt['labels'] = [-100] * len(tokenized_full_prompt['labels'])

        output_icd_list = data_point['output_dict']['output_icd_id']

        output_str = 'The predicted icd code are: ' +  ', '.join(output_icd_list)
        output_tokenized = self.single_tokenize(output_str, cutoff_len=None, add_eos_token=True)
        tokenized_full_prompt['input_ids'].extend(output_tokenized['input_ids'])
        tokenized_full_prompt['attention_mask'].extend(output_tokenized['attention_mask'])
        tokenized_full_prompt['labels'].extend(output_tokenized['labels'])

        return tokenized_full_prompt
    