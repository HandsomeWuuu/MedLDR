import json
import pandas as pd
import re
import os

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
        
        response = item['response']

        pattern = r"```json\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)


        if not matches:
            matches = check_invaild_case(response)
        

        if matches:
            try:
                if len(matches) == 1:
                    matches = matches[0]
                    json_data = json.loads(matches)
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
            except:
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


def main():

    input_file = '/xxxxx/infer_api/results/all_cursorai_gpt_4_1/result_cursorai_gpt_4_1_all_full.jsonl'
  
    output_valid_file = os.path.dirname(input_file) + '/processed_results'+ '/valid_results.csv'
    os.makedirs(os.path.dirname(output_valid_file), exist_ok=True)

    output_invalid_file = os.path.dirname(input_file) + '/processed_results'+ '/invalid_cases.jsonl'

    data = load_jsonl(input_file)

    print('Data length:', len(data))

    # Part 1: 
    valid_df, invalid_list = extract_top_data(data)
    
    print('Valid data length:', len(valid_df))
    print('Invalid data length:', len(invalid_list))

    print('Valid data length after drop duplicates:', len(valid_df))
    consistency_true_ratio = valid_df['pred_consistency'].sum()
    top_1_consistency_true_ratio = valid_df['top1_consistency'].sum()
    print('Consistency True Ratio:', consistency_true_ratio,'/', len(valid_df),'ratio:', round(consistency_true_ratio/len(valid_df), 4))
    print('Top1 Consistency True Ratio:', top_1_consistency_true_ratio,'/', len(valid_df), 'ratio:', round(top_1_consistency_true_ratio/len(valid_df), 4))

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
    
    
    # Part 2: 
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

    # price --  input_price, output_price
    input_price_M = 2
    output_price_M = 8
    print('Input Price:', input_price_M)
    print('Output Price:', output_price_M)
    
    input_tokens = data_df['prompt_tokens'].sum()/1000000
    output_tokens = data_df['completion_tokens'].sum()/1000000

    print('Input Tokens:', input_tokens,'M', 'Input Price:', (input_tokens * input_price_M))
    print('Output Tokens:', output_tokens,'M','Output Price:', (output_tokens * output_price_M))
    print('Total Price:', (input_tokens * input_price_M) + (output_tokens * output_price_M))

if __name__ == "__main__":
    main()
