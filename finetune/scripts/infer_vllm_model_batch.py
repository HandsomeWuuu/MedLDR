# function: 推理模型 生成推理结果
# 1. 用参数解析模块argparse解析命令行参数，得到 模型名称、输入测试json数据路径，推理结果输出路径
# 2. 加载模型，加载测试数据，加载推理结果输出文件
# 3. 遍历测试数据，对每个测试数据，生成推理结果
# 4. 保存推理结果到输出文件

import argparse
import torch
import os
import json
from tqdm import tqdm
# import shortuuid
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer,PreTrainedTokenizerFast

import math
import random
import pandas as pd

# from .src.open_r1.process_data.for_generate.load_mimic_icd import get_dataset,process_input_data
from open_r1.process_data.for_generate.load_mimic_icd import get_dataset,process_input_data
from open_r1.configs import data_config as DataArguments
# 推理 lab 任务

from vllm import LLM,SamplingParams

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

    
def load_json(file_path):

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except:
        data = []
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))

    return data


def prepare_tokenizer_and_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer.pad_token_id = 0
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # model = LlamaForCausalLM.from_pretrained(model_path)
    model = LLM(model=model_path,
                # tokenizer_mode='slow',
                )
    
    return tokenizer, model



def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)

    print('Infer Model Path:',model_path)

    tokenizer,model = prepare_tokenizer_and_model(model_path)


    # Load multi-question file
    data_config = DataArguments()
    data_config.test_data_path = args.question_file
    data_config.input_key = 'all'
    data_config.training_type = args.training_type # 'grpo'
    questions = get_dataset(data_config, tokenizer, 'test')
    questions = questions.to_list()

    # Choose a subset of questions
    if isinstance(args.choose_num,int):
        questions = questions[:int(args.choose_num)]

    print('Watch: total questions num:',len(questions))

    # 选择重复的数据 -- 用于再次推理
    if args.repeat_file is not None:
        repeat_file = os.path.expanduser(args.repeat_file)
        repeat_data = load_json(repeat_file)
        print('repeat_data num:',len(repeat_data))
        if not isinstance(repeat_data,list):
            repeat_data = [repeat_data]
        repeat_data_list = [x['case_id'] for x in repeat_data]
        questions = [x for x in questions if x['case_id'] in repeat_data_list]

    else:
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    print('questions num:',len(questions))


    if 'OpenBioLLM' in model_path:
        model_type = 'OpenBioLLM'
    elif 'DeepSeek-R1-Distill-Llama-8B' in model_path:
        model_type = 'Distill-Llama-8B'
    else:
        raise ValueError(f"Unknown model type: {model_path}")

    print('Model_type:',model_type)

    if data_config.training_type == 'sft':
        print('infer sft model')
        # Process input data
        def formatting_prompts_func(example):
            # print('example:',example)
            single_case = {}
            single_case['case_id'] = example['case_id']
            single_case['solution'] = example['solution']

            input_text = process_input_data(example['input'],tokenizer)

            if model_type == 'OpenBioLLM':
                messages = [
                            {"role": "system", "content": example['input_prompt']},
                            {"role": "user", "content": input_text},
                            ]
                output_texts = tokenizer.apply_chat_template(
                                messages, 
                                tokenize=False, 
                                add_generation_prompt=True)
                # 把输出的文本拼接起来 -- infer时，输出的文本是要预测的
            elif model_type == 'Distill-Llama-8B':
                # Distill-Llama-8B模型的tokenizer
                messages = [
                            {"role": "system", "content": example['input_prompt']},
                            {"role": "user", "content": input_text},
                            ]
                output_texts = tokenizer.apply_chat_template(
                                messages, 
                                tokenize=False, 
                                add_generation_prompt=True)
                # 去掉 <think> token
                output_texts = output_texts.replace("<think>","")
            else:
                # output_texts = f"### Question: {example['input_prompt']}\n{input_text}.\n ### Answer: {example['output']}"
                output_texts = f"### Question: {example['input_prompt']}\n{input_text}.\n ### Answer:"
    

            single_case['prompt'] = output_texts

            return single_case

        if model_type == 'OpenBioLLM':
            # OpenBioLLM-8B模型的tokenizer
            sampling_params = SamplingParams(temperature=0.7,top_p=0.95,max_tokens=200,include_stop_str_in_output=True,
                                                stop_token_ids=[
                                                   tokenizer.eos_token_id,
                                                   tokenizer.convert_tokens_to_ids("<|eot_id|>")
                                               ])
              
        elif model_type == 'Distill-Llama-8B':
            # Distill-Llama-8B模型的tokenizer
            sampling_params = SamplingParams(temperature=0.7,top_p=0.95,max_tokens=200,include_stop_str_in_output=True,
                                                stop_token_ids=[
                                                   tokenizer.eos_token_id,
                                                   tokenizer.convert_tokens_to_ids("<｜end▁of▁sentence｜>")
                                               ])
                                  
        else:
            sampling_params = SamplingParams(temperature=0.7, top_p=0.95,max_tokens=100,stop=["}"],include_stop_str_in_output=True)

        
        print('Process SFT data')
        print('sampling_params:',sampling_params)  
        formatted_questions = [formatting_prompts_func(q) for q in questions]
        
        
    elif data_config.training_type == 'grpo':
        print('infer grpo model')
        # 这里暂时只支持 Distill-Llama-8B模型
        # Format into conversation
        from trl.data_utils import apply_chat_template

        def make_conversation(example):
            prompt = []

            # if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": example["input_prompt"]})
            input_text = process_input_data(example['input'],tokenizer)
            prompt.append({"role": "user", "content":input_text})

            return {"prompt": apply_chat_template({'prompt':prompt},tokenizer), "case_id": example["case_id"], "solution": example["solution"]}
        
        # Apply make_conversation to each question in the dataset
        formatted_questions = [make_conversation(q) for q in questions]

        sampling_params = SamplingParams(temperature=0.7, top_p=0.95,max_tokens=4096)


    
    print('formatted_questions num:',len(formatted_questions))
    print('formatted_questions case:',formatted_questions[0])

    # Infer Loop    
    ans_file = open(answers_file, "w")
    
    # 将 formatted_questions 打成批次，再推理
    batch_size = 8
    formatted_questions = split_list(formatted_questions,batch_size)
    print('formatted_questions batch num:',len(formatted_questions))
    print('formatted_questions single batch num:',len(formatted_questions[0]))

    for line in tqdm(formatted_questions):
        
        # get batch data
        batch_input = [x['prompt'] for x in line]

        output_texts = model.generate(batch_input,sampling_params)
    
        # Process each item in the batch
        for i, outputs in enumerate(output_texts):
            ans_file.write(json.dumps({
                            "idx": line[i]["case_id"],
                            "input": outputs.prompt,
                            "input_tokens": len(outputs.prompt_token_ids),
                            "result": outputs.outputs[0].text,
                            "solution": line[i]["solution"], 
                            "result_tokens": len(outputs.outputs[0].token_ids),
                            }) + "\n")
            ans_file.flush()


    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--repeat-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--choose-num", type=str, default=None)
    parser.add_argument("--input-key", type=str, default=None)
    parser.add_argument("--training-type", type=str,default=None) # unuse
    args = parser.parse_args()

    print('args',args)

    if args.choose_num != 'None':
        args.choose_num = int(args.choose_num)
        print('args.choose_num:',args.choose_num)
    else:  
        args.choose_num = None
    # 将 args 写入到文件中

    args.repeat_file = '/data/h3571902/deep_r1/open_r1/results/grpo/DeepSeek-R1-Distill-Llama-8B-GRPO-mimic-v3/back_cp/checkpoint-980-light/40_test_infer_prompt_40/processed_results/invalid_cases.jsonl'

    args_dict = vars(args)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    args_settings_name = os.path.join(os.path.dirname(args.answers_file),"args_setting.json")
    with open(args_settings_name, "w") as f:
        json.dump(args_dict, f, indent=4)

    eval_model(args)


