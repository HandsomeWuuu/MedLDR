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
import pickle

merged_icd_dict = {
    "001-139": ["0389"],
    "240-279": ["25080", "262", "2761", "2762", "27651"],
    "280-289": ["28419", "2851"],
    "290-319": ["29181", "311"],
    "320-389": ["34830", "34831", "34982"],
    "390-459": ["4019", "40291", "40391", "40491", "41071", "41401", "41519", "42731", "42823", "42833", "431", "43411", "43491", "4589"],
    "460-519": ["486", "5070", "51881", "51884"],
    "520-579": ["5762", "5770"],
    "580-629": ["5845", "5849", "5990"],
    "680-709": ["6827"],
    "780-799": ["7802"],
    "800-999": ["99591", "99859"]
}


icd_name_code_map_file = '/grp01/cs_yzyu/wushuai/model/llama_demo/datasets/mimic_lab_icd/split_new/icd_and_name.csv'
icd_codes_df = pd.read_csv(icd_name_code_map_file, dtype=str)

icd_code_to_idx = {code: idx for idx, code in enumerate(merged_icd_dict.keys())}
icd_idx_to_code = {idx: code for idx, code in enumerate(merged_icd_dict.keys())}

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


def accuracy(y_true, y_pred):
    count = sum(sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i])) for i in range(y_true.shape[0]))
    return count / y_true.shape[0]

def transform_icd_to_class(icd_code_list):
    transformed_list = []
    # print('Before:',icd_code_list)
    for code in icd_code_list:
        for key, values in merged_icd_dict.items():
            if code in values:
                transformed_list.append(key)
                break
    
    transformed_list = list(set(transformed_list))

    return transformed_list



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
            top10 = [case[:3] for case in row['top10']]
            labels_icd = row['labels_icd']
            # print('Top10:', top10)
            # print('Labels:', labels_icd)
            
            topk = top10[:k]
            # print('Topk:', topk)
            # 要将 topk 中的 icd code 映射回 merged_icd_dict 的区间
            transformed_topk = []
            for icd in topk:
                for class_range, icd_codes in merged_icd_dict.items():
                    # Compare first 3 digits of the ICD code
                    icd_prefix = icd
                    for code in icd_codes:
                        if code.startswith(icd_prefix):
                            transformed_topk.append(class_range)
                            break
            # print('Transformed Topk:', transformed_topk)

            # Remove duplicates
            recall += len(set(labels_icd) & set(transformed_topk)) / len(labels_icd)
            
        return recall / len(data)
    
    # Calculate metrics and their CIs

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

    # 将 df_predictions 中的 valid_prediction 字段中的 "['99591', '486', '42823']" 转换为 list ['99591', '486', '42823']
    import ast
    df_predictions['valid_prediction'] = df_predictions['valid_prediction'].apply(lambda x: ast.literal_eval(x) if x else [])
    # df_predictions['valid_prediction'] = df_predictions['valid_prediction'].apply(lambda x: transform_icd_to_class(x))

    if 'top10' in df_predictions.columns:
        df_predictions['top10'] = df_predictions['top10'].apply(lambda x: ast.literal_eval(x) if x else [])
        # df_predictions['top10'] = df_predictions['top10'].apply(lambda x: transform_icd_to_class(x))

    labels_list = []
    for item in labels:
        single_label = {}
        single_label['case_id'] = item['case_id']
        single_label['labels_icd'] = [str(case) for case in item['output_dict']['output_icd_id']]
        single_label['top_1_icd'] = str(item['top_1_icd']['code'])
        labels_list.append(single_label)
    
    df_labels = pd.DataFrame(labels_list)

    df_labels['labels_icd'] = df_labels['labels_icd'].apply(lambda x: transform_icd_to_class(x))
    # df_labels['top_1_icd'] = df_labels['top_1_icd'].apply(lambda x: transform_icd_to_class([x]))

    
    assert len(df_predictions) == len(df_labels), f'Prediction and label count mismatch pred len {len(df_predictions)} !=  labels len {len(df_labels)}'

    df_merged = pd.merge(df_predictions, df_labels, on='case_id')

    print(df_merged.head())

    save_bootstrap_data = False

    if save_bootstrap_data:
        global save_bootstrap_data_path
        save_bootstrap_data_path = os.path.join(os.path.dirname(args.predict_result), 'bootstrap_12_metrics_data')
    
        os.makedirs(save_bootstrap_data_path, exist_ok=True)


    # ## 1. 计算大类指标的置信区间
    save_path = os.path.join(os.path.dirname(args.predict_result), 'metric_class_12_recall@N_ci.json')
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
