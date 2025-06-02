
import torch
import transformers
import json
import os

import random


from open_r1.configs import data_config as DataArguments
from open_r1.configs import Task_Mapping_Table, cut_off_len_dict
from open_r1.utils import rank0_print

from .organize_data import top50_system_message,top50_reasoning_system_message,top40_system_message

from datasets import load_dataset, Dataset  

from tqdm import tqdm
def load_json(file_path):
    with open(file_path, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            file.seek(0)
            data = [json.loads(line) for line in file]
    return data
    



def get_dataset(data_args: DataArguments, tokenizer: transformers.PreTrainedTokenizer, split: str):
    
    # 1. 用 from datasets import load_dataset 加载 data_args 的数据集
    # 2. 打印数据集的信息
    # 3. 组织数据集的输入和输出: 
    if split == 'train':
        data_path = data_args.train_data_path
        data = load_json(data_path)
    elif split == 'val':
        data_path = data_args.eval_data_path
        data = load_json(data_path)
    elif split == 'test':
        data_path = data_args.test_data_path
        data = load_json(data_path)
    else:
        raise ValueError(f"Invalid split: {split}")
    
    rank0_print(f"Loaded {len(data)} samples from {data_path}")

    # Shuffle dataset
    rank0_print("Shuffling dataset...")
    random.shuffle(data)

    # Print Case:
    input_list = Task_Mapping_Table[data_args.input_key]

    rank0_print('input_list:', input_list)
    selected_inputs = {key: data[0]['input_dict'][key] for key in input_list}
    rank0_print('Selected inputs case :', selected_inputs)

    # train_type = 'sft' # grpo
    train_type = data_args.training_type
    # 3. 组织数据集的输入和输出
    target_data_list = []

    for data_point in tqdm(data):
        target_data = {}
        target_data['case_id'] = data_point['case_id']
        if train_type == 'sft':
            # target_data['input_prompt'] = top50_system_message
            target_data['input_prompt'] = top40_system_message
        else:
            target_data['input_prompt'] = top50_reasoning_system_message
            
        # target_data['input'] = ['\n---\n'.join(data_point['input_dict'][key]) if isinstance(data_point['input_dict'][key], list) else data_point['input_dict'][key] for key in input_list]
        target_data['input'] = {key: 'The report results:\n' +'\n---\n'.join(data_point['input_dict'][key]) if isinstance(data_point['input_dict'][key], list) else data_point['input_dict'][key] for key in input_list}
        # target_data['output'] =  '[' + ', '.join(data_point['output_dict']['output_icd_id']) + ']'
        target_data['output'] = '{"diagnoses": [' + ', '.join([f'"{code}"' for code in data_point['output_dict']['output_icd_id']]) + ']}'
        # target_data['completion'] = 'The predicted icd code are: ' +  ', '.join(data_point['output_dict']['output_icd_id'])
        target_data['solution'] = ';'.join(data_point['output_dict']['output_icd_id'])

        target_data_list.append(target_data)

    dataset = Dataset.from_list(target_data_list)
    print(f"dataset: {dataset}")


    return dataset


def process_input_data(data_point, tokenizer):
    # patient info
    input_data = ''
    input_key_list = data_point.keys()
    # print(f"data_point type: {type(data_point)}")
    # print(f"input_key_list: {input_key_list}")
    for key in input_key_list:
        value = data_point[key]
        single_tokenized = single_tokenize(tokenizer,data_point[key], cutoff_len = cut_off_len_dict[key])
        input_ids = single_tokenized['input_ids']
        input_text = tokenizer.decode(input_ids)
        input_data += f"\n{input_text}"
        # print(f" {key}, input_text: {input_text}")

    

    return input_data

def single_tokenize(tokenizer, prompt, cutoff_len, add_start_token =False):
        if cutoff_len is None:
            result = tokenizer(
                prompt,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
        else:
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
        
        if not add_start_token:
            result["input_ids"] = result["input_ids"][1:]

        return result