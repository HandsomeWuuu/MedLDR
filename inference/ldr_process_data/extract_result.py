import json
import pandas as pd
import re
import os
import sys

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

def check_markdown_format(response):
    # 2. 检测markdown格式的top10和predictions
    try:
        # 尝试提取 **top10**: [...] 格式
        top10_pattern = r'\*\*top10\*\*:\s*\[(.*?)\]'
        predictions_pattern = r'\*\*predictions\*\*:\s*\[(.*?)\]'
        
        top10_match = re.search(top10_pattern, response, re.DOTALL)
        predictions_match = re.search(predictions_pattern, response, re.DOTALL)
        
        if top10_match and predictions_match:
            top10_str = top10_match.group(1)
            predictions_str = predictions_match.group(1)
            
            # 解析列表中的项目，匹配 "code:description" 格式
            def parse_icd_list(list_str):
                items = re.findall(r'"([^"]+)"', list_str)
                return items
            
            top10 = parse_icd_list(top10_str)
            predictions = parse_icd_list(predictions_str)
            
            if len(top10) > 0 and len(predictions) > 0:
                return {
                    'top10': top10,
                    'predictions': predictions
                }
    except Exception as e:
        print(f"Error parsing markdown format: {e}")
    
    return None

def extract_icd_code(result_list):
    icd_codes = [icd.split(':')[0] for icd in result_list]

    return icd_codes


def extract_top_data(data):
    
    valid_list = []
    invalid_list = []
    markdown_format_count = 0
    
    for item in data:
        single_result = {}
        single_result['case_id'] = item['case_id']
        
        response = item['response']
        json_data = None

        pattern = r"```json\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)

        if not matches:
            matches = check_invaild_case(response)
        
        # 如果还是没有找到JSON，尝试检测markdown格式
        if not matches:
            markdown_result = check_markdown_format(response)
            if markdown_result:
                json_data = markdown_result
                markdown_format_count += 1

        if matches:
            try:
                if len(matches) == 1:
                    matches = matches[0]
                    json_data = json.loads(matches)
            except:
                print('Invalid JSON case:', item['case_id'])
                invalid_list.append(item)
                continue
        
        if json_data:
            try:
                top10 = json_data.get('top10', [])
                predictions = json_data.get('predictions', [])

                if len(top10)>0 and len(predictions)>0:
                    top10_icd = extract_icd_code(top10)
                    predictions_icd = extract_icd_code(predictions)

                    single_result['top10'] = extract_icd_code(top10_icd)
                    single_result['predictions'] = extract_icd_code(predictions_icd)    
                    single_result['pred_consistency'] = check_consistency(top10_icd, predictions_icd)
                    single_result['top1_consistency'] = check_consistency(top10_icd, predictions_icd[:1])
                    valid_list.append(single_result)
            
                else:  
                    print('Invalid case:', item['case_id'])
                    invalid_list.append(item)
            except Exception as e:
                print('Invalid case:', item['case_id'], f'Error: {e}')
                invalid_list.append(item)
        else:
            invalid_list.append(item)
    
    print(f"从markdown格式提取的案例数量: {markdown_format_count}")
    valid_df = pd.DataFrame(valid_list)
    return valid_df, invalid_list


def save_to_csv(df, output_file):
    df.to_csv(output_file, index=False, encoding='utf-8')

def save_to_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


task_target = 'statistics_token' # statistics_token,extract_data

def main():
    base_path = './'
   
    input_file = sys.argv[1]

    print('Input file:', input_file)
    output_valid_file = os.path.dirname(input_file) + '/processed_results'+ '/valid_results.csv'
    os.makedirs(os.path.dirname(output_valid_file), exist_ok=True)

    output_invalid_file = os.path.dirname(input_file) + '/processed_results'+ '/invalid_cases.jsonl'

    data = load_jsonl(input_file)

    print('Data length:', len(data))

    # Part 1: if task_target == 'extract_data':
    valid_df, invalid_list = extract_top_data(data)
    
    print('Valid data length:', len(valid_df))
    print('Invalid data length:', len(invalid_list))
    # invalid_df = invaild_case()

    # Find duplicate case_id rows
    duplicate_case_ids = valid_df[valid_df.duplicated('case_id', keep=False)]
    print('Duplicate case_id rows:')
    print(duplicate_case_ids)
    # duplicate_case_ids.to_csv(os.path.dirname(input_file) + '/processed_results'+ '/duplicate_case_ids.csv', index=False, encoding='utf-8')
    valid_df = valid_df.drop_duplicates(subset='case_id', keep='first')

    print('Valid data length after drop duplicates:', len(valid_df))
    # raise ValueError('Duplicate case_id rows found')
    # valid_df = pd.concat([valid_df, invalid_df], ignore_index=True)
    consistency_true_ratio = valid_df['pred_consistency'].sum()
    top_1_consistency_true_ratio = valid_df['top1_consistency'].sum()
    print('Consistency True Ratio:', consistency_true_ratio,'/', len(valid_df))
    print('Top1 Consistency True Ratio:', top_1_consistency_true_ratio,'/', len(valid_df))

    # 增加一个 valid_prediction, 用 apply 函数 把 predictions 在 top_40_icd 的icd 放到 valid_prediction 中
    valid_df['valid_prediction'] = valid_df['predictions'].apply(lambda x: [icd for icd in x if icd in top_40_icd])
    valid_df['error_prediction_len'] = valid_df['predictions'].apply(lambda x: len(x) - len([icd for icd in x if icd in top_40_icd]))

    print('Error Prediction Sum:', valid_df['error_prediction_len'].sum())
    print('Error Prediction Case:', valid_df[valid_df['error_prediction_len'] != 0].shape[0])
    
    # 根据 case_id 去重
    valid_df = valid_df.drop_duplicates(subset='case_id', keep='first')

    print('Valid data length after drop duplicates:', len(valid_df))
    
    save_to_csv(valid_df, output_valid_file)
    save_to_jsonl(invalid_list, output_invalid_file)
    
    
    # Part 2: elif task_target == 'statistics_token':
    data_list = []
    for item in data:
        sigle_case = {}
        sigle_case['case_id'] = item['case_id']
        sigle_case['estimate_prompt_tokens'] = item['estimate_token_count']['total']
        sigle_case.update(item['token_usage'])
        data_list.append(sigle_case)

    data_df = pd.DataFrame(data_list)
    print(data_df.columns)
    print(data_df.head())
    print(data_df.describe())

    # 模拟计算下价格 --  input_price, output_price
    input_price_M = 0.3*1.66
    output_price_M = 2.5*1.66
    input_tokens = data_df['prompt_tokens'].sum()/1000000
    output_tokens = data_df['completion_tokens'].sum()/1000000

    print('Input Tokens:', input_tokens,'M', 'Input Price:', (input_tokens * input_price_M))
    print('Output Tokens:', output_tokens,'M','Output Price:', (output_tokens * output_price_M))
    print('Total Price:', (input_tokens * input_price_M) + (output_tokens * output_price_M))
    
    # remove_invalid = True
    remove_invalid = False

    if remove_invalid:
        # Remove invalid cases from original data
        invalid_case_ids = {item['case_id'] for item in invalid_list}
        filtered_data = [item for item in data if item['case_id'] not in invalid_case_ids]
        
        # Remove duplicate cases from filtered_data, keeping the last occurrence
        duplicate_case_ids_set = set(duplicate_case_ids['case_id'].tolist())
        seen_case_ids = set()
        final_filtered_data = []

        # Process in reverse order to keep the last occurrence
        for item in reversed(filtered_data):
            case_id = item['case_id']
            if case_id in duplicate_case_ids_set:
                if case_id not in seen_case_ids:
                    seen_case_ids.add(case_id)
                    final_filtered_data.append(item)
            else:
                final_filtered_data.append(item)

        # Reverse back to original order
        filtered_data = list(reversed(final_filtered_data))
        # Save filtered data back to original path
        save_to_jsonl(filtered_data, input_file)
        print(f'Removed {len(invalid_case_ids)} invalid cases from original file')
        print(f'Remaining cases: {len(filtered_data)}')


if __name__ == "__main__":
    main()
