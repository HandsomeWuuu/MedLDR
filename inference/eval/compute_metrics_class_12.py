import pandas as pd
import json
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, classification_report
import argparse

# top_40_icd = ['4589', '27651', '5070', '5770', '51884', '0389', '34982', '25080', '41401',\
#              '29181', '41071', '5845', '41519', '431', '2762', '262', '40291', '34830', '43491',\
#             '34831', '99859', '40391', '42731', '2851', '99591', '4019', '28419', '40491', '486',\
#             '5762', '5990', '6827', '42833', '43411', '7802', '51881', '311', '42823', '2761', '5849']

overall_dict = {
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

icd_code_to_idx = {code: idx for idx, code in enumerate(overall_dict)}
icd_idx_to_code = {idx: code for idx, code in enumerate(overall_dict)}


def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]

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

def main(args):
   
    labels = load_json(args.label_file)
    df_predictions = pd.read_csv(args.predict_result, dtype=str)

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
        pred_id = np.zeros(len(icd_code_to_idx))
        for code in row['valid_prediction']:
            # For codes that don't match any interval, find their interval and set that
            for interval, codes in overall_dict.items():
                if any(code.startswith(c[:3]) for c in codes):
                    pred_id[icd_code_to_idx[interval]] = 1
                    break

        pred_list.append(pred_id)

        label_id = np.zeros(len(icd_code_to_idx))
        for code in row['labels_icd']:
            for interval, codes in overall_dict.items():
                if any(code.startswith(c[:3]) for c in codes):
                    label_id[icd_code_to_idx[interval]] = 1
                    break

        # print(label_id, row['labels_icd'])

        # raise ValueError('Stop here')
        label_list.append(label_id)

    report = classification_report(label_list, pred_list, output_dict=True)

    avg_report = {k: v for k, v in report.items() if 'avg' in k}
    non_avg_report = {k: v for k, v in report.items() if 'avg' not in k}

    report_df = pd.DataFrame(non_avg_report).transpose().reset_index().rename(columns={'index': 'idx'})
    report_df['idx'] = report_df['idx'].astype(int)
    report_df['ICD_Code_Interval'] = report_df['idx'].map(icd_idx_to_code)
    # report_df['ICD_Name'] = report_df['ICD_Code'].map(icd_codes_df.set_index('ICD Code')['Name'])
    report_df = report_df.sort_values(by='support', ascending=False)
    report_df[['precision', 'recall', 'f1-score']] = report_df[['precision', 'recall', 'f1-score']].applymap(lambda x: round(x, 4))

    report_df_path = os.path.join(os.path.dirname(args.predict_result), 'classification_report_12.csv')
    report_df.to_csv(report_df_path, index=False)

    metrics = compute_multi_label_metrics(label_list, pred_list)
    avg_report = {k: {metric: round(value, 4) for metric, value in v.items()} for k, v in avg_report.items()}
    metrics.update(avg_report)
    metric_path = os.path.join(os.path.dirname(args.predict_result), 'result_metrics_class_12.json')
    

    # 2. 用 top_1_icd 和 labels_icd 计算 top1_accuracy
    top1_correct = 0
    for _, row in df_merged.iterrows():
        top1_icd_code = row['top_1_icd']
        label_intervals = set()
        for interval, codes in overall_dict.items():
            if any(top1_icd_code.startswith(c[:3]) for c in codes):
                label_intervals.add(interval)
                break

        predict_intervals = set()
        for pred_code in row['valid_prediction']:
            for interval, codes in overall_dict.items():
                if any(pred_code.startswith(c[:3]) for c in codes):
                    predict_intervals.add(interval)
                    break
        
        if len(label_intervals.intersection(predict_intervals)) > 0:
            top1_correct += 1
        
    top1_accuracy = top1_correct / len(df_merged)
    print(f'Top-1 ICD Accuracy: {top1_accuracy:.4f}')

    raise ValueError('Stop here')
    metrics['top1_accuracy'] = round(top1_accuracy, 4)


    # 3. 用 valid_prediction 和 top10 计算 recall@10, recall@5
    top10_recall = 0
    top5_recall = 0
    top3_recall = 0

    for _, row in df_merged.iterrows():
        label_intervals = set()
        for code in row['labels_icd']:
            for interval, codes in overall_dict.items():
                if any(code.startswith(c[:3]) for c in codes):
                    label_intervals.add(interval)
                    break
        
    
        predict_intervals = set()
        for pred_code in row['top10'][:10]:  # Top 10
            for interval, codes in overall_dict.items():
                if any(pred_code.startswith(c[:3]) for c in codes):
                    predict_intervals.add(interval)
                    break
  
        
        top10_recall += len(label_intervals.intersection(predict_intervals)) / len(label_intervals) 

        # Calculate for top 5
        predict_intervals_5 = set()
        for pred_code in row['top10'][:5]:
            for interval, codes in overall_dict.items():
                if any(pred_code.startswith(c[:3]) for c in codes):
                    predict_intervals_5.add(interval)
                    break
        # print('predict_intervals_5',predict_intervals_5)
        top5_recall += len(label_intervals.intersection(predict_intervals_5)) / len(label_intervals) 
        # print('top5_recall',top5_recall)

        # Calculate for top 3
        predict_intervals_3 = set()
        for pred_code in row['top10'][:3]:
            for interval, codes in overall_dict.items():
                if any(pred_code.startswith(c[:3]) for c in codes):
                    predict_intervals_3.add(interval)
                    break
        
        # print('predict_intervals_3',predict_intervals)
        top3_recall += len(label_intervals.intersection(predict_intervals_3)) / len(label_intervals) 
        # print('top3_recall',top3_recall)
        # raise ValueError('Stop here')
    metrics['recall@10'] = round(top10_recall / len(df_merged), 4)
    metrics['recall@5'] = round(top5_recall / len(df_merged), 4) 
    metrics['recall@3'] = round(top3_recall / len(df_merged), 4)

    with open(metric_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_result', type=str, help='Path to the inference results CSV file')
    parser.add_argument('--label_file', type=str, help='Path to the labels JSON file')

    args = parser.parse_args()
    main(args)
