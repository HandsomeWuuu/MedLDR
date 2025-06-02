import json
import argparse
from chat_api.chat_deepseek import chat_deepseek_v3,chat_deepseek_v3_0324_with_cursorai
from chat_api.chat_openai import chat_gpt4o, chat_o3_mini, chat_o3_mini_with_cursorai
from chat_api.chat_o4_mini import chat_o4_mini
from chat_api.chat_gpt_4_1 import chat_gpt_4_1
from process_data.organize_data import organize_input_data, system_message
from chat_api.chat_deepseek_r1 import chat_deepseek_r1_zzz,chat_deepseek_r1_sliliconflow,chat_deepseek_r1_tenxun
from chat_api.chat_qwen import chat_Distill_Qwen_sliliconflow
from chat_api.chat_llama import chat_Distill_Llama_sliliconflow

from chat_api.chat_grok3 import chat_grok_3_with_cursor_api, chat_grok_3
from chat_api.chat_gemini2_5 import chat_gemini_2_5

from chat_api.chat_qwen_235b import chat_qwen_235b_think,chat_qwen_235b
import os
from tqdm import tqdm
from openai import OpenAI
from multiprocessing import Pool

# 解析命令行参数
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--test_data_path', type=str, required=True, help='Test data path')
parser.add_argument('--split_list', type=str, default=None, help='Split list')
parser.add_argument('--task_type', type=str, required=True, help='Task type')
parser.add_argument('--chat_model', type=str, required=True, help='Chat model')
parser.add_argument('--num_processes', type=int, default=10, help='Number of processes')
parser.add_argument('--num_round', type=int, default=1, help='Number of processes')
parser.add_argument('--test_mode', action="store_true", help="启用测试模式")
args = parser.parse_args()

task_type = args.task_type
chat_model = args.chat_model

# 加载 dataset/split_by_patient/test_data.json 的数据
# test_data_path = './dataset/split_by_patient_v1/test_data.json'
# test_data_path = '/grp01/cs_yzyu/wushuai/model/llama_demo/datasets/mimic_multi_inputs_icd/split_by_patient/subset_top40_top1/infer_icd_report_data_top.json'
test_data_path = args.test_data_path

temp_postfix = ''
if 'split' in test_data_path.split('/')[-1]:
    spilt_postfix = test_data_path.split("/")[-1].replace(".json", "")
    save_data_path = f'./results/{task_type}_{chat_model}/result_{chat_model}_{task_type}_{spilt_postfix}.jsonl'
    temp_postfix = f'_{spilt_postfix}'
else:
    if args.num_round == 1:
        save_data_path = f'./results/{task_type}_{chat_model}/result_{chat_model}_{task_type}_full.jsonl'
    else:
        save_data_path = f'./results/subset_100/round_{args.num_round}/{task_type}_{chat_model}/result_{chat_model}_{task_type}_100.jsonl'

print('**** Test data path : **** \n', test_data_path)
print('**** Save data path : **** \n', save_data_path)

with open(test_data_path, 'r') as f:
    process_data = json.load(f)

TEST_MODE = args.test_mode

print('**** Test data len: **** \n', len(process_data))
print('**** Test model : **** \n', chat_model)
print('**** Test task type : **** \n', task_type)
print('**** num_processes : **** \n', args.num_processes)
print('**** num_round : **** \n', args.num_round)
print('**** Test mode : **** \n', TEST_MODE)
print('**** System message : **** \n', system_message)


# 准备服务器
if chat_model == 'deepseek_r1_tenxun':
    API_SECRET_KEY = "sk-xxx"
    BASE_URL = "https://api.lkeap.cloud.tencent.com/v1"
    client = OpenAI(
        api_key=API_SECRET_KEY, 
        base_url=BASE_URL
    )
elif 'slflow' in chat_model:
    pass
    print('**** Chat model is in sliliconflow ****')
elif 'cursorai' in chat_model:
    import requests
    client = requests.Session()
else:
    API_SECRET_KEY = "sk-xxx"
    BASE_URL = "https://api.zhizengzeng.com/v1"
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)


# Check has been completed cases
completed_cases = set()
if os.path.exists(save_data_path):
    with open(save_data_path, 'r') as f:
        for line in f:
            result_dict = json.loads(line)
            completed_cases.add(result_dict['case_id'])

print('**** Completed cases len: **** \n', len(completed_cases))
print('remaining cases:', len(process_data) - len(completed_cases))
# print('**** Completed cases : **** \n', completed_cases)

def infer_data(data_list, save_path):
    max_retries = 10
    # print('i am in infer_data')
    sub_completed_cases = set()
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            for line in f:
                result_dict = json.loads(line)
                sub_completed_cases.add(result_dict['case_id'])

    print('**** Sub completed cases len: **** \n', len(sub_completed_cases))
    print('**** Sub completed cases : **** \n', sub_completed_cases)

    with open(save_path, 'a') as f:
        for data_point in tqdm(data_list):
            if data_point['case_id'] in completed_cases or data_point['case_id'] in sub_completed_cases:
                print(f"Skipping {data_point['case_id']} as it has already been completed")
                continue
            if TEST_MODE:
                print('* Processing data_point:', data_point['case_id'])

            result_dict = {}
            result_dict['case_id'] = data_point['case_id']
            
            # 准备数据
            input_data, token_count = organize_input_data(data_point, task_type)

            if TEST_MODE:
                print('* Input_data:', input_data)

            print('* Estimate_token_count:', token_count)

            result_dict['input_data'] = input_data
            result_dict['estimate_token_count'] = token_count
            # 调用 chat_deepseek_v3
            retries = 0
            response, token_dict = None, None
            
            while retries < max_retries:
                if chat_model == 'deepseek_v3_zzz':
                    response, token_dict = chat_deepseek_v3(client, system_message, input_data)
                elif chat_model == 'gpt4o_zzz':
                    response, token_dict = chat_gpt4o(client, system_message, input_data)
                elif chat_model == 'o3_mini_zzz':
                    response, token_dict = chat_o3_mini(client, system_message, input_data)
                elif chat_model == 'deepseek_r1_zzz':
                    response, token_dict = chat_deepseek_r1_zzz(client, system_message, input_data)
                elif chat_model == 'cursorai_grok_3':
                    response, token_dict = chat_grok_3(client, system_message, input_data)
                elif chat_model == 'cursorai_grok_3_reasoning':
                    response, token_dict = chat_grok_3_with_cursor_api(client, system_message, input_data)
                elif chat_model == 'cursorai_gemini_2_5':
                    response, token_dict = chat_gemini_2_5(client, system_message, input_data)
                elif chat_model == 'cursorai_o4_mini':
                    response, token_dict = chat_o4_mini(client, system_message, input_data)
                elif chat_model == 'cursorai_gpt_4_1':
                    response, token_dict = chat_gpt_4_1(client, system_message, input_data)

                else:
                    raise ValueError(f"Invalid chat model: {chat_model}")

                if response is not None:
                    break

                retries += 1
                print(f"Retrying {data_point['case_id']} ({retries}/{max_retries})")
            
            if response is None:
                print(f"Failed to get response for {data_point['case_id']} after {max_retries} retries")
                continue
            
            # 提取推理部分和答案部分
            if 'deepseek_r1' in chat_model:
                result_dict['answer'] = response['answer']
                result_dict['reasoning'] = response['reasoning']
                result_dict['token_usage'] = token_dict
            else:
                result_dict['response'] = response
                result_dict['token_usage'] = token_dict

            if TEST_MODE:
                print('* Response:', response)
            
            print('* Token_usage:', token_dict)

            # 写入文件
            f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')


def merge_files(file_paths, output_path):
    with open(output_path, 'a') as outfile:
        for file_path in file_paths:
            with open(file_path, 'r') as infile:
                for line in infile:
                    outfile.write(line)

def main(test_data):
    num_processes = args.num_processes
    chunk_size = len(test_data) // num_processes

    save_data_root = os.path.dirname(save_data_path)
    
    if not os.path.exists(save_data_root):
        os.makedirs(save_data_root)

    temp_files = [os.path.join(save_data_root, f'temp_result_{i}_{temp_postfix}.jsonl') for i in range(num_processes)]

    with Pool(processes=num_processes) as pool:
        for i in range(num_processes):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i != num_processes - 1 else len(test_data)
            pool.apply_async(infer_data, args=(test_data[start_idx:end_idx], temp_files[i]))

        pool.close()
        pool.join()

    merge_files(temp_files, save_data_path)

    # 删除临时文件
    for temp_file in temp_files:
        os.remove(temp_file)


if __name__ == '__main__':
    main(process_data)