import pandas as pd
import json
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, classification_report
import argparse

top_40_icd = ['4589', '27651', '5070', '5770', '51884', '0389', '34982', '25080', '41401',\
             '29181', '41071', '5845', '41519', '431', '2762', '262', '40291', '34830', '43491',\
            '34831', '99859', '40391', '42731', '2851', '99591', '4019', '28419', '40491', '486',\
            '5762', '5990', '6827', '42833', '43411', '7802', '51881', '311', '42823', '2761', '5849']


icd_name_code_map_file = '/grp01/cs_yzyu/wushuai/model/llama_demo/datasets/mimic_lab_icd/split_new/icd_and_name.csv'
icd_codes_df = pd.read_csv(icd_name_code_map_file, dtype=str)

icd_code_to_idx = {code: idx for idx, code in enumerate(top_40_icd)}
icd_idx_to_code = {idx: code for idx, code in enumerate(top_40_icd)}

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]


def main(args):
   
    labels = load_json(args.label_file)
    df_predictions = pd.read_csv(args.predict_result, dtype=str)

    # 将 df_predictions 中的 valid_prediction 字段中的 "['99591', '486', '42823']" 转换为 list ['99591', '486', '42823']
    import ast
    df_predictions['valid_prediction'] = df_predictions['valid_prediction'].apply(lambda x: ast.literal_eval(x) if x else [])
    # df_predictions['top10'] = df_predictions['top10'].apply(lambda x: ast.literal_eval(x) if x else [])
    
    # print(df_predictions.head())

    labels_list = []
    for item in labels:
        single_label = {}
        single_label['case_id'] = item['case_id']
        single_label['labels_icd'] = [str(case) for case in item['output_dict']['output_icd_id']]
        single_label['top_1_icd'] = str(item['top_1_icd']['code'])
        labels_list.append(single_label)
    
    df_labels = pd.DataFrame(labels_list)
    print(df_labels.head())
    assert len(df_predictions) == len(df_labels), f'Prediction and label count mismatch pred len {len(df_predictions)} !=  labels len {len(df_labels)}'

    df_merged = pd.merge(df_predictions, df_labels, on='case_id')


    # 2. 用 top_1_icd 和 labels_icd 计算 top1_accuracy
    top1_correct = 0
    
    # Group-wise accuracy calculation
    icd_group_metrics = {}
    for icd_code in top_40_icd:
        icd_group_metrics[icd_code] = {
            'correct': 0,
            'total': 0,
            'accuracy': 0.0
        }
    
    for _, row in df_merged.iterrows():
        top1_icd_code = row['top_1_icd'] 
        predict_ids = row['valid_prediction']
        
        # Overall accuracy
        if top1_icd_code in predict_ids:
            top1_correct += 1
        
        # Group-wise accuracy
        if top1_icd_code in top_40_icd:
            icd_group_metrics[top1_icd_code]['total'] += 1
            if top1_icd_code in predict_ids:
                icd_group_metrics[top1_icd_code]['correct'] += 1
    
    # Calculate overall accuracy
    top1_accuracy = top1_correct / len(df_merged)
    print(f'Top-1 ICD Accuracy: {top1_accuracy:.4f}')
    print(f'icd_group_metrics: {icd_group_metrics}')

    # Calculate group-wise accuracy
    for icd_code in top_40_icd:
        if icd_group_metrics[icd_code]['total'] > 0:
            icd_group_metrics[icd_code]['accuracy'] = icd_group_metrics[icd_code]['correct'] / icd_group_metrics[icd_code]['total']
        print(f"ICD {icd_code} Accuracy: {icd_group_metrics[icd_code]['accuracy']:.4f} ({icd_group_metrics[icd_code]['correct']}/{icd_group_metrics[icd_code]['total']})")
    

    # Save metrics
    metrics = {
        'top1_accuracy': round(top1_accuracy, 4),
        'icd_group_accuracy': {code: round(data['accuracy'], 4) for code, data in icd_group_metrics.items() if data['total'] > 0},
        'icd_group_counts': {code: {'correct': data['correct'], 'total': data['total']} for code, data in icd_group_metrics.items() if data['total'] > 0}
    }
    
    # Get ICD code names for better reporting
    icd_names = {}
    print(icd_codes_df.head())
    print(icd_codes_df.columns)

    for icd_code in top_40_icd:

        name_row = icd_codes_df[icd_codes_df['ICD Code'] == icd_code]
        if not name_row.empty:
            icd_names[icd_code] = name_row['Name'].values[0]
        else:
            icd_names[icd_code] = "Unknown"
    
    metrics['icd_names'] = icd_names
    
    # Define the output path for metrics
    metric_path = os.path.join(os.path.dirname(args.predict_result), 'metrics_group_by_icd.json')

    with open(metric_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    

    with open(metric_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_result', type=str, help='Path to the inference results CSV file')
    parser.add_argument('--label_file', type=str, help='Path to the labels JSON file')

    args = parser.parse_args()
    main(args)
