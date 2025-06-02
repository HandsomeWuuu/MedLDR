import json
import pandas as pd
import re
import os
import argparse

top_50_icd = ['2762', '5990', '99592', '51881', '5849', '43411', '42832', '5845', '42822', '42833', \
            '41401', '42731', '42823', '5770', '34830', '40291', '431', '5070', '34982', '4280', '311', \
            '51884', '34831', '2761', '27651', '99591', '25040', '99859', '28419', '2851', '3485', '41071', \
            '0389', '40491', '29181', '486', '7802', '6827', '5762', '2724', '4019', '41519', '40391', '2875', \
                '25080', '43491', '5856', '4589', '262', '25000']

top_40_icd = ['4589', '27651', '5070', '5770', '51884', '0389', '34982', '25080', '41401',\
             '29181', '41071', '5845', '41519', '431', '2762', '262', '40291', '34830', '43491',\
            '34831', '99859', '40391', '42731', '2851', '99591', '4019', '28419', '40491', '486',\
            '5762', '5990', '6827', '42833', '43411', '7802', '51881', '311', '42823', '2761', '5849']



def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def check_consistency(top10, predictions):
    pred_num = len(predictions)
    if predictions == top10[:pred_num]:
        # print('Predictions are the top 3 of top10')
        return True
    else:
        return False

def check_invaild_case(response):
    # 1. 解决response中没有json字段的问题
    pattern = r"\{.*?\}"
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return matches

    return None

def extract_icd_code(result_list):
    icd_codes = [icd.split(':')[0] for icd in result_list]

    return icd_codes


def extract_top_data(data):
    
    valid_list = []
    invalid_list = []
    for item in data:
        single_result = {}
        single_result['case_id'] = item['case_id']
        
        response = item['answer']

        # Remove content between <think> and </think> tags
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

        target_pattern = r'\{.*?\}'
        matches = re.findall(target_pattern, response, re.DOTALL)

        # print('Matches:', matches)
    
        if not matches:
            # matches = check_invaild_case(response)
            invalid_list.append(item)
            continue

        # matches = next((match for match in matches if 'diagnoses' in match), None)
        matches = matches[0] if len(matches) > 0 else None

        print('Matches:', matches)
        # raise Exception('Stop here')
        if matches:
    
            try:
                json_data = json.loads(matches)
                print('Json Data:', json_data)
                    
                predictions = json_data.get('diagnoses', [])
                print('Predictions:', predictions)

            except Exception as e:
                print('Error:', e)
                invalid_list.append(item)
                continue
                
            if  len(predictions)>0:
                print('predictions:', predictions)
                predictions_icd = extract_icd_code(predictions)
                single_result['predictions'] = extract_icd_code(predictions_icd)    
                valid_list.append(single_result)
        
            else:  
                print('Invalid case:', item['case_id'])
                invalid_list.append(item)
        else:
            invalid_list.append(item)
    
    valid_df = pd.DataFrame(valid_list)
    return valid_df, invalid_list


def save_to_csv(df, output_file):
    df.to_csv(output_file, index=False, encoding='utf-8')

def save_to_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# task_target = 'statistics_token' # statistics_token,extract_data

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--top_num', type=int, default=50)
    args = parser.parse_args()
    input_file_list = [args.input_file]

    # input_file_list = ['/data/h3571902/deep_r1/open_r1/results/sft/Llama3-OpenBioLLM-8B-mimic-lr5e6/checkpoint-1080/50_test_infer_result_batch/result.jsonl']

    input_file = input_file_list[-1]
    output_valid_file = os.path.dirname(input_file) + '/processed_results'+ '/valid_results.csv'
    os.makedirs(os.path.dirname(output_valid_file), exist_ok=True)

    output_invalid_file = os.path.dirname(input_file) + '/processed_results'+ '/invalid_cases.jsonl'

    data = []
    for file in input_file_list:
        temp_data = load_jsonl(file)
        print('Data length:', len(temp_data))
        data += temp_data
    print('Data length:', len(data))

    # 把 idx 改成 case_id
    for item in data:
        item['case_id'] = item['idx']
        item['answer'] = item['result']

        del item['idx']
        del item['result']

    # if task_target == 'extract_data':
    valid_df, invalid_list = extract_top_data(data)
    
    print('Valid data length:', len(valid_df))
    print('Invalid data length:', len(invalid_list))
    # invalid_df = invaild_case()


    # 增加一个 valid_prediction, 用 apply 函数 把 predictions 在 top_40_icd 的icd 放到 valid_prediction 中
    if args.top_num == 50:
        valid_df['valid_prediction'] = valid_df['predictions'].apply(lambda x: [icd for icd in x if icd in top_50_icd])
        valid_df['error_prediction_len'] = valid_df['predictions'].apply(lambda x: len(x) - len([icd for icd in x if icd in top_50_icd]))
    elif args.top_num == 40:
        valid_df['valid_prediction'] = valid_df['predictions'].apply(lambda x: [icd for icd in x if icd in top_40_icd])
        valid_df['error_prediction_len'] = valid_df['predictions'].apply(lambda x: len(x) - len([icd for icd in x if icd in top_40_icd]))

    print('Error Prediction Sum:', valid_df['error_prediction_len'].sum())
    print('Error Prediction Case:', valid_df[valid_df['error_prediction_len'] != 0].shape[0])
    
    save_to_csv(valid_df, output_valid_file)
    save_to_jsonl(invalid_list, output_invalid_file)

    # elif task_target == 'statistics_token':
    data_list = []
    for item in data:
        sigle_case = {}
        sigle_case['case_id'] = item['case_id']
        sigle_case['prompt_tokens'] = item['input_tokens']
        sigle_case['completion_tokens'] = item['result_tokens']
        
        data_list.append(sigle_case)

    data_df = pd.DataFrame(data_list)
    print(data_df.columns)
    print(data_df.head())
    print(data_df.describe())

    # 模拟计算下价格 --  input_price, output_price
    input_price_M = 4
    output_price_M = 16
    input_tokens = data_df['prompt_tokens'].sum()/1000000
    output_tokens = data_df['completion_tokens'].sum()/1000000

    print('Input Tokens:', input_tokens,'M', 'Input Price:', (input_tokens * input_price_M))
    print('Output Tokens:', output_tokens,'M','Output Price:', (output_tokens * output_price_M))
    print('Token Price:', (output_tokens * output_price_M) + (input_tokens * input_price_M))
    
if __name__ == "__main__":
    main()
