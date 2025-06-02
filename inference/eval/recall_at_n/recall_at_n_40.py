'''
Compute metrics for the Confidence Interval

'''

import pandas as pd
import json
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, classification_report
import argparse
from tqdm import tqdm
import ast
import pickle

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

def save_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def compute_bootstrap_ci(metric_fn, data,save_bootstrap_data,metric_name, n_bootstrap=1000, confidence=0.95):
        """
        Compute bootstrap confidence intervals for a given metric
        """
        n_samples = len(data)
        bootstrap_scores = []

        for _ in tqdm(range(n_bootstrap)):
            # Sample with replacement
            indices = np.random.randint(0, n_samples, n_samples)
            sample = data.iloc[indices]
            score = metric_fn(sample)
            bootstrap_scores.append(score)
        
        # Calculate confidence interval
        lower_percentile = ((1 - confidence) / 2) * 100
        upper_percentile = (confidence + (1 - confidence) / 2) * 100
        
        ci_lower = np.percentile(bootstrap_scores, lower_percentile)
        ci_upper = np.percentile(bootstrap_scores, upper_percentile)
        
        if save_bootstrap_data:
            # Save bootstrap scores for this metric
            save_path = os.path.join(save_bootstrap_data_path, f'{metric_name}_bootstrap_scores.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(bootstrap_scores, f)

        return ci_lower, ci_upper
        
def calculate_metrics_with_ci(df_merged,save_bootstrap_data):
    """
    Calculate all metrics with confidence intervals
    """
    metrics_with_ci = {}
  
    def calc_recall_at_k(data, k):
        recall = 0
        for _, row in data.iterrows():
            labels_icd = row['labels_icd']
            topk = row['top10'][:k]
            recall += len(set(labels_icd) & set(topk)) / len(labels_icd)
        return recall / len(data)
    
   
    if 'top10' in df_merged.columns:
        for k in range(1, 11):
            value = calc_recall_at_k(df_merged, k)
            lower, upper = compute_bootstrap_ci(
            lambda x: calc_recall_at_k(x, k), df_merged, save_bootstrap_data, f'recall@{k}')
            metrics_with_ci[f'recall@{k}'] = {
            'value': value,
            'ci': [lower, upper]
            }
        
    return metrics_with_ci


def main(args):
   
    labels = load_json(args.label_file)
    df_predictions = pd.read_csv(args.predict_result, dtype=str)

    df_predictions['valid_prediction'] = df_predictions['valid_prediction'].apply(lambda x: ast.literal_eval(x) if x else [])

    if 'top10' in df_predictions.columns:
        df_predictions['top10'] = df_predictions['top10'].apply(lambda x: ast.literal_eval(x) if x else [])
    
    # add labels: labels_icd, top_1_icd
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

    save_bootstrap_data = False

    if save_bootstrap_data:
        global save_bootstrap_data_path
        save_bootstrap_data_path = os.path.join(os.path.dirname(args.predict_result), 'bootstrap_40_metrics_data')
    
        os.makedirs(save_bootstrap_data_path, exist_ok=True)

    save_path = os.path.join(os.path.dirname(args.predict_result), 'metric_class_40_recall@N_ci.json')
    
    # 1. 用 valid_prediction 和 labels_icd 计算准确率,f1, classifcation report
    result = calculate_metrics_with_ci(df_merged,save_bootstrap_data)

    # Save the result to a JSON file
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)
        print(f"Metrics with CI saved to {save_path}")
        


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_result', type=str, help='Path to the inference results CSV file')
    parser.add_argument('--label_file', type=str, help='Path to the labels JSON file')

    args = parser.parse_args()
    main(args)
