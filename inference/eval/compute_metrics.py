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


def load_icd_codes(lab_file):
    icd_codes_df = pd.read_csv(lab_file, dtype=str)
    icd_codes = icd_codes_df['ICD Code'].tolist()
    icd_code_to_idx = {code: idx for idx, code in enumerate(icd_codes)}
    icd_idx_to_code = {idx: code for idx, code in enumerate(icd_codes)}
    return icd_codes_df, icd_code_to_idx, icd_idx_to_code




def accuracy(y_true, y_pred):
    count = sum(sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i])) for i in range(y_true.shape[0]))
    return count / y_true.shape[0]


def compute_multi_label_metrics(label_list, pred_list):
    label_list = np.array(label_list)
    pred_list = np.array(pred_list)

    metrics = {
        'emr_accuracy': round(accuracy_score(label_list, pred_list), 4),
        'accuracy': round(accuracy(label_list, pred_list), 4),
        'precision': round(precision_score(label_list, pred_list, average='samples'), 4),
        'recall': round(recall_score(label_list, pred_list, average='samples'), 4),
        'f1': round(f1_score(label_list, pred_list, average='samples'), 4),
        'hamming_loss': round(hamming_loss(label_list, pred_list), 4)
    }
    return metrics


def get_valid_result(result_file, failed_result_file, icd_code_to_idx):
    result_df = pd.read_csv(result_file, dtype=str)
    failed_result_df = pd.read_csv(failed_result_file, dtype=str)
    combined_df = pd.concat([result_df, failed_result_df], ignore_index=True)

    case_ids = combined_df['Case ID'].unique()
    predictions_valid = {case_id: [code for code in combined_df[combined_df['Case ID'] == case_id]['Code'].tolist() if code in icd_code_to_idx] for case_id in case_ids}
    predictions_invalid = {case_id: [code for code in combined_df[combined_df['Case ID'] == case_id]['Code'].tolist() if code not in icd_code_to_idx] for case_id in case_ids}

    return predictions_valid, predictions_invalid


def main(args):
   
    labels = load_json(args.label_file)
    df_predictions = pd.read_csv(args.predict_result, dtype=str)

    # 将 df_predictions 中的 valid_prediction 字段中的 "['99591', '486', '42823']" 转换为 list ['99591', '486', '42823']
    import ast
    df_predictions['valid_prediction'] = df_predictions['valid_prediction'].apply(lambda x: ast.literal_eval(x) if x else [])
    df_predictions['top10'] = df_predictions['top10'].apply(lambda x: ast.literal_eval(x) if x else [])
    
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

    # 1. 用 valid_prediction 和 labels_icd 计算准确率,f1, classifcation report
    label_list = []
    pred_list = []
    for _, row in df_merged.iterrows():
        pred_id = np.zeros(40)
        for code in row['valid_prediction']:
            pred_id[icd_code_to_idx[code]] = 1

        pred_list.append(pred_id)
        # print(pred_id, row['valid_prediction'])

        label_id = np.zeros(40)
        for code in row['labels_icd']:
            label_id[icd_code_to_idx[code]] = 1
        # print(label_id, row['labels_icd'])

        # raise ValueError('Stop here')
        label_list.append(label_id)

    report = classification_report(label_list, pred_list, output_dict=True)

    avg_report = {k: v for k, v in report.items() if 'avg' in k}
    non_avg_report = {k: v for k, v in report.items() if 'avg' not in k}

    report_df = pd.DataFrame(non_avg_report).transpose().reset_index().rename(columns={'index': 'idx'})
    report_df['idx'] = report_df['idx'].astype(int)
    report_df['ICD_Code'] = report_df['idx'].map(icd_idx_to_code)
    report_df['ICD_Name'] = report_df['ICD_Code'].map(icd_codes_df.set_index('ICD Code')['Name'])
    report_df = report_df.sort_values(by='support', ascending=False)
    report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].applymap(lambda x: round(x, 4))

    report_df_path = os.path.join(os.path.dirname(args.predict_result), 'classification_report.csv')
    report_df.to_csv(report_df_path, index=False)

    metrics = compute_multi_label_metrics(label_list, pred_list)
    avg_report = {k: {metric: round(value, 4) for metric, value in v.items()} for k, v in avg_report.items()}
    metrics.update(avg_report)
    metric_path = os.path.join(os.path.dirname(args.predict_result), 'result_metrics.json')
    

    # 2. 用 top_1_icd 和 labels_icd 计算 top1_accuracy
    top1_correct = 0
    for _, row in df_merged.iterrows():
        top1_icd_code = row['top_1_icd'] 
        predict_ids = row['valid_prediction']
        if top1_icd_code in predict_ids:
            top1_correct += 1
        
    top1_accuracy = top1_correct / (len(df_merged))
    print(f'Top-1 ICD Accuracy: {top1_accuracy:.4f}')

    metrics['top1_accuracy'] = round(top1_accuracy, 4)

    with open(metric_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # 3. 用 valid_prediction 和 top10 计算 recall@10, recall@5
    top10_recall = 0
    top5_recall = 0
    top3_recall = 0
    for _, row in df_merged.iterrows():
        labels_icd = row['labels_icd']
        top10 = row['top10']

        top5 = top10[:5]
        top3 = top10[:3]

        top10_recall += len(set(labels_icd) & set(top10)) / len(labels_icd)
        top5_recall += len(set(labels_icd) & set(top5)) / len(labels_icd)
        top3_recall += len(set(labels_icd) & set(top3)) / len(labels_icd)
    
    top10_recall = top10_recall / len(df_merged)
    top5_recall = top5_recall / len(df_merged)
    top3_recall = top3_recall / len(df_merged)

    print(f'Recall@10: {top10_recall:.4f}')
    print(f'Recall@5: {top5_recall:.4f}')
    print(f'Recall@3: {top3_recall:.4f}')

    metrics['recall@10'] = round(top10_recall, 4)
    metrics['recall@5'] = round(top5_recall, 4)
    metrics['recall@3'] = round(top3_recall, 4)

    with open(metric_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_result', type=str, help='Path to the inference results CSV file')
    parser.add_argument('--label_file', type=str, help='Path to the labels JSON file')

    args = parser.parse_args()
    main(args)
