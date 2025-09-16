import json
import argparse

from eicu_process_data.organize_eicu_data import organize_input_data, eicu_system_message_dict
import os
from tqdm import tqdm
from multiprocessing import Pool

# think or not , temperature
from chat_api.chat_gemini2_5_flash import chat_gemini_2_5_flash
from chat_api.chat_claude_sonnet_4 import chat_claude_sonnet_4
from chat_api.chat_qwen_235b_a22b_official import chat_qwen_qwen3_235b_a22b_official
from chat_api.chat_qwen_plus_official import chat_qwen_plus_official
from chat_api.chat_qwen3_30b_a3b_official import chat_qwen3_30b_a3b_official
from chat_api.chat_gpt_5_mini import chat_gpt_5_mini
from chat_api.chat_deepseek import chat_deepseek_cursorai

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--test_data_path', type=str, required=True, help='Test data path')
parser.add_argument('--split_list', type=str, default=None, help='Split list')
parser.add_argument('--task_type', type=str, required=True, help='Task type')
parser.add_argument('--chat_model', type=str, required=True, help='Chat model')
parser.add_argument('--think', type=bool, help='Whether to use reasoning (think) or not')  # If not specified, defaults to False
parser.add_argument('--reasoning_effort', type=str, default=None, help='Reasoning effort: minimal, low, medium, high')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature')
parser.add_argument('--system_message_type', type=str, default=None, help='System message type')

parser.add_argument('--num_processes', type=int, default=10, help='Number of processes')
parser.add_argument('--num_round', type=int, default=1, help='Number of processes')
parser.add_argument('--test_mode', action="store_true", help="Enable test mode")
args = parser.parse_args()

task_type = args.task_type
chat_model = args.chat_model

system_message = eicu_system_message_dict[args.system_message_type]
test_data_path = args.test_data_path

temp_postfix = ''

think_postfix = 'think' if args.think else 'nothink'
prompt_group = '_'.join(args.system_message_type.split('_')[-2:])

if args.num_round == 1:
    if args.chat_model == 'cursorai_gpt_5_mini':
        save_data_path = f'./results_eicu/{task_type}_{chat_model}_{args.reasoning_effort}/prompt_{prompt_group}/prompt_{args.system_message_type}_temp_{args.temperature}/result_{chat_model}_{task_type}_full.jsonl'
    else:
        save_data_path = f'./results_eicu/{task_type}_{chat_model}_{think_postfix}/prompt_{prompt_group}/prompt_{args.system_message_type}_temp_{args.temperature}/result_{chat_model}_{task_type}_full.jsonl'
else:
    if args.chat_model == 'cursorai_gpt_5_mini':
        save_data_path = f'./results_eicu/subset_100/round_{args.num_round}/{task_type}_{chat_model}_{args.reasoning_effort}/prompt_{prompt_group}/prompt_{args.system_message_type}_temp_{args.temperature}/result_{chat_model}_{task_type}_full.jsonl'
    else:
        save_data_path = f'./results_eicu/subset_100/round_{args.num_round}/{task_type}_{chat_model}_{think_postfix}/prompt_{prompt_group}/prompt_{args.system_message_type}_temp_{args.temperature}/result_{chat_model}_{task_type}_full.jsonl'
        

os.makedirs(os.path.dirname(save_data_path), exist_ok=True)

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
print('**** Think : **** \n', args.think)
print('**** Reasoning effort : **** \n', args.reasoning_effort)
print('**** Temperature : **** \n', args.temperature)
print('**** System message type : **** \n', args.system_message_type)
print('**** System message : **** \n', system_message)

# sleep(5) # Wait for 5 seconds to ensure the logs are output correctly

import time
sleep_time= 3
print(f'**** Sleeping for {sleep_time} seconds ****')
time.sleep(sleep_time)

# Prepare server
if 'cursorai' in chat_model:
    import requests
    client = requests.Session()
else:
    # official API
    pass   


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

# Remove data points that do not need to be processed from process_data
process_data = [item for item in process_data if item['case_id'] not in completed_cases]
print('**** Remaining cases (to run) len: **** \n', len(process_data))

def infer_data(data_list, save_path):
    max_retries = 10
 
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
            
            # Prepare data
            input_data, token_count = organize_input_data(data_point, task_type)

            if TEST_MODE:
                print('* Input_data:', input_data)

            print('Case id:',data_point['case_id'],'* Estimate_token_count:', token_count)

            result_dict['input_data'] = input_data
            result_dict['estimate_token_count'] = token_count
    
            retries = 0
            response, token_dict = None, None
            
            while retries < max_retries:
                
                # think or not , temperature
                if chat_model == 'cursorai_gemini_2_5_flash' and args.think:
                    response, token_dict = chat_gemini_2_5_flash(client, system_message, input_data,think=True,temperature=args.temperature)
                elif chat_model == 'cursorai_gemini_2_5_flash' and not args.think:
                    response, token_dict = chat_gemini_2_5_flash(client, system_message, input_data,think=False,temperature=args.temperature)
                elif chat_model == 'cursorai_claude_sonnet_4' and args.think:
                    response, token_dict = chat_claude_sonnet_4(client, system_message, input_data,think=True,temperature=args.temperature)
                elif chat_model == 'cursorai_claude_sonnet_4' and not args.think:
                    response, token_dict = chat_claude_sonnet_4(client, system_message, input_data,think=False,temperature=args.temperature)
                elif chat_model == 'official_qwen3_235b_a22b' and args.think:
                    response, token_dict = chat_qwen_qwen3_235b_a22b_official(None, system_message, input_data,think=True,temperature=args.temperature)
                elif chat_model == 'official_qwen3_235b_a22b' and not args.think:
                    response, token_dict = chat_qwen_qwen3_235b_a22b_official(None, system_message, input_data,think=False,temperature=args.temperature)
            
                elif chat_model == 'official_qwen_plus' and args.think:
                    response, token_dict = chat_qwen_plus_official(None, system_message, input_data,think=True,temperature=args.temperature)
                elif chat_model == 'official_qwen_plus' and not args.think:
                    response, token_dict = chat_qwen_plus_official(None, system_message, input_data,think=False,temperature=args.temperature)
                
                elif chat_model == 'official_qwen3_30b_a3b' and args.think:
                    response, token_dict = chat_qwen3_30b_a3b_official(None, system_message, input_data,think=True,temperature=args.temperature)
                elif chat_model == 'official_qwen3_30b_a3b' and not args.think:
                    response, token_dict = chat_qwen3_30b_a3b_official(None, system_message, input_data,think=False,temperature=args.temperature)
                
                elif chat_model == 'cursorai_gpt_5_mini':
                    response, token_dict = chat_gpt_5_mini(client, system_message, input_data, reasoning_effort=args.reasoning_effort)

                elif chat_model == 'cursorai_deepseek' and args.think:
                    response, token_dict = chat_deepseek_cursorai(client, system_message, input_data, think=True, temperature=args.temperature)
                elif chat_model == 'cursorai_deepseek' and not args.think:
                    response, token_dict = chat_deepseek_cursorai(client, system_message, input_data, think=False, temperature=args.temperature)
                
                else:
                    raise ValueError(f"Unsupported chat model: {chat_model}")

                if response is not None:
                    break
                retries += 1
                print(f"Retrying {data_point['case_id']} ({retries}/{max_retries})")
            
            if response is None:
                print(f"Failed to get response for {data_point['case_id']} after {max_retries} retries")
                continue
            
            # Extract the reasoning part and the answer part
            if  args.think:
                print('extracting reasoning and response from response')
                if chat_model == 'cursorai_claude_sonnet_4' or chat_model == 'cursorai_grok_3' or chat_model == 'cursorai_deepseek':
                  
                    print('extracting reasoning from response')
                 
                    try:
                        reasoning_content = response['response'].split('<think>')[1].split('</think>')[0]
                        print('reasoning_content:', reasoning_content)
                        result_dict['reasoning'] = reasoning_content
                    
                        # Remove the reasoning part from the response
                        response['response'] = response['response'].replace(f'<think>{reasoning_content}</think>', '').strip()
                    
                    except IndexError:
                        print('Error extracting reasoning content, using response directly')
                        result_dict['reasoning'] = 'No reasoning found'
                    
                else:
                    result_dict['reasoning'] = response['reasoning']
            
            print('save response:')
            result_dict['response'] = response['response']
            result_dict['token_usage'] = token_dict
            result_dict['think'] = args.think
            result_dict['temperature'] = args.temperature
            result_dict['system_message_type'] = args.system_message_type
            result_dict['chat_model'] = chat_model
            
            if TEST_MODE:
                print('* Response:', response)
            
            print('* Token_usage:', token_dict)

            # Save the result
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


    for temp_file in temp_files:
        os.remove(temp_file)


if __name__ == '__main__':
    main(process_data)