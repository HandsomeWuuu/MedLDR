import pandas as pd
import json
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, classification_report
import argparse
from tqdm import tqdm

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


def get_merge_df(label_file, predict_result):
    labels = load_json(label_file)
    df_predictions = pd.read_csv(predict_result, dtype=str)

    import ast
    df_predictions['valid_prediction'] = df_predictions['valid_prediction'].apply(lambda x: ast.literal_eval(x) if x else [])

    labels_list = []
    for item in labels:
        single_label = {}
        single_label['case_id'] = item['case_id']
        single_label['labels_icd'] = [str(case) for case in item['output_dict']['output_icd_id']]
        single_label['top_1_icd'] = str(item['top_1_icd']['code'])
        labels_list.append(single_label)

    df_labels = pd.DataFrame(labels_list)
    assert len(df_predictions) == len(df_labels), f'Prediction and label count mismatch pred len {len(df_predictions)} !=  labels len {len(df_labels)}'

    df_merged = pd.merge(df_predictions, df_labels, on='case_id')

    print(f'df merged head: {df_merged.head()}')

    return df_merged


def compute_primary_acc(df_merged):

    top1_correct = 0
    for _, row in df_merged.iterrows():
        top1_icd_code = row['top_1_icd']
        label_intervals = set()
        # make label_intervals
        for interval, codes in overall_dict.items():
            if any(top1_icd_code.startswith(c[:3]) for c in codes):
                label_intervals.add(interval)
                break

        # make predict_intervals
        predict_intervals = set()
        for pred_code in row['valid_prediction']:
            for interval, codes in overall_dict.items():
                if any(pred_code.startswith(c[:3]) for c in codes):
                    predict_intervals.add(interval)
                    break
        
        # check if label_intervals in predict_intervals
        if label_intervals.issubset(predict_intervals):
            top1_correct += 1
        
    top1_accuracy = top1_correct / len(df_merged)
    print(f'Top-1 ICD Accuracy: {top1_accuracy:.4f}')


    return primary_acc


def compute_p_value(model_1_df, model_2_df, model_names, n_resamples=10):
    """
    计算两个模型的准确率差异的p值，使用bootstrap方法。
    要求 model_1_df 和 model_2_df 均有 'case_id' 和 'valid_prediction' 列。
    """
    # 按case_id对齐
    merged = pd.merge(
        model_1_df[['case_id', 'valid_prediction', 'top_1_icd']],
        model_2_df[['case_id', 'valid_prediction']],
        on='case_id',
        suffixes=('_1', '_2')
    )

    print(f'merged head: {merged.head()}')
    
    ### 1. 计算每个模型的平均 primary_acc和p值
    def compute_primary_acc_for_df(df, pred_col):
        correct = 0
        for _, row in df.iterrows():
            top1_icd_code = row['top_1_icd']
            label_intervals = set()
            for interval, codes in overall_dict.items():
                if any(top1_icd_code.startswith(c[:3]) for c in codes):
                    label_intervals.add(interval)
                    break
            predict_intervals = set()
            for pred_code in row[pred_col]:
                for interval, codes in overall_dict.items():
                    if any(pred_code.startswith(c[:3]) for c in codes):
                        predict_intervals.add(interval)
                        break
            if label_intervals.issubset(predict_intervals):
                correct += 1
        return correct / len(df)

    acc_1 = compute_primary_acc_for_df(merged, 'valid_prediction_1')
    acc_2 = compute_primary_acc_for_df(merged, 'valid_prediction_2')

    # bootstrap p-value
    n = len(merged)
    diffs = []
    print(f'Bootstrap n : {n_resamples}')
    for _ in tqdm(range(n_resamples)):
        idx = np.random.choice(n, n, replace=True)
        sample_1 = merged.iloc[idx]
        sample_2 = merged.iloc[idx]
        acc1 = compute_primary_acc_for_df(sample_1, 'valid_prediction_1')
        acc2 = compute_primary_acc_for_df(sample_2, 'valid_prediction_2')
        diffs.append(acc1 - acc2)
        print(f'Bootstrap acc1: {acc1:.4f}, acc2: {acc2:.4f}, diff: {acc1 - acc2:.4f}')

    diffs = np.array(diffs)
    observed_diff = acc_1 - acc_2
    print(f'Observed diff: {observed_diff:.4f}')

    p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))

    print(f"模型1 {model_names[0]} 平均primary_acc: {acc_1:.4f}")
    print(f"模型2 {model_names[1]} 平均primary_acc: {acc_2:.4f}")
    print(f"平均primary_acc差异p值: {p_value:.6f}")
    

    ### 2. 计算每个大类的primary_acc和p值
    def compute_primary_acc_row(row, pred_col):
        top1_icd_code = row['top_1_icd']
        label_intervals = set()
        for interval, codes in overall_dict.items():
            if any(top1_icd_code.startswith(c[:3]) for c in codes):
                label_intervals.add(interval)
                break

        predict_intervals = set()
        for pred_code in row[pred_col]:
            for interval, codes in overall_dict.items():
                if any(pred_code.startswith(c[:3]) for c in codes):
                    predict_intervals.add(interval)
                    break

        return list(label_intervals)[0] if label_intervals else None, int(label_intervals.issubset(predict_intervals))

    # 统计每个大类的acc
    category_accs_1 = {cat: [] for cat in overall_dict}
    category_accs_2 = {cat: [] for cat in overall_dict}

    for _, row in merged.iterrows():
        cat1, acc1 = compute_primary_acc_row(row, 'valid_prediction_1')
        cat2, acc2 = compute_primary_acc_row(row, 'valid_prediction_2')
        if cat1:
            category_accs_1[cat1].append(acc1)
            category_accs_2[cat1].append(acc2)

    # 计算每个大类的p值
    category_p_values = {}
    for cat in overall_dict:
        accs_1 = np.array(category_accs_1[cat])
        accs_2 = np.array(category_accs_2[cat])
        if len(accs_1) == 0:
            print(f"Category {cat}: No samples.")
            category_p_values[cat] = None
            continue
        observed_diff = accs_1.mean() - accs_2.mean()
        n = len(accs_1)
        # print(f'n: {n}, observed_diff: {observed_diff:.4f}')
        diffs = []
        for _ in range(n_resamples):
            idx = np.random.choice(n, n, replace=True)
            diff = accs_1[idx].mean() - accs_2[idx].mean()
            diffs.append(diff)
        diffs = np.array(diffs)
        # print(f'diffs: {diffs.shape}')
        p_value = np.mean(np.abs(diffs) >= np.abs(observed_diff))
        print(f"Category {cat} - 模型1 {model_names[0]} primary_acc: {accs_1.mean():.4f}, 模型2 {model_names[1]} primary_acc: {accs_2.mean():.4f}, p值: {p_value:.6f} ({len(accs_1)} samples)")
        category_p_values[cat] = p_value

    return category_p_values

def main():
    parser = argparse.ArgumentParser(description="Compute p-value for model predictions.")
    parser.add_argument('--label_file', type=str, required=True, help='Path to the label file.')
    parser.add_argument('--model_1_result', type=str, required=True, help='Path to the first model prediction result.')
    parser.add_argument('--model_2_result', type=str, required=True, help='Path to the second model prediction result.')
    args = parser.parse_args()

    # Load data
    df_model_1 = get_merge_df(args.label_file, args.model_1_result)
    df_model_2 = get_merge_df(args.label_file, args.model_2_result)

    model_names = ['Gemini-2.5-Pro', 'GPT-4.1']

    # Compute p-value
    p_values = compute_p_value(df_model_1, df_model_2, model_names)
    print("P-values:", p_values)

if __name__ == "__main__":
    main()