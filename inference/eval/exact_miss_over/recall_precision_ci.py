'''
    统计 primary 评测结果
    查看 pred primary 离 label 的 top 1 icd 的距离
    1. 在不在 40 类中: correct
    2. 在不在 33 类中: very close
    3. 在不在 12 类中: close
    4. 在不在 3 类中: nothing related
'''
import pandas as pd
import json
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, classification_report
import argparse
from tqdm import tqdm


def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]
    
def count_recall_precision_diagnosis(df_merged):
    """
    Calculate recall and precision for diagnosis predictions
    
    Calculate:
    - Recall: TP / (TP + FN) = TP / total_true_positives
    - Precision: TP / (TP + FP) = TP / total_predicted_positives
    """
    missed_total = 0
    over_total = 0
    results = []

    # 为计算micro指标准备变量
    total_tp = 0  # 真阳性总数 - (correct diagnoses) == TP
    total_fn = 0  # 假阴性总数 (missed diagnoses) == FN
    total_fp = 0  # 假阳性总数 (over diagnoses) == FP
    
    # 为计算macro指标准备字典
    disease_tp = {}  # 每种疾病的真阳性数
    disease_fn = {}  # 每种疾病的假阴性数 (missed)
    disease_fp = {}  # 每种疾病的假阳性数 (over)
    all_diseases = set()  # 所有出现过的疾病
    
    for _, row in df_merged.iterrows():
        predictions = row['valid_prediction']
        true_labels = row['labels_icd']
        
        # 计算当前样本的TP、FN、FP
        tp = len(set(predictions).intersection(set(true_labels)))
        fn = len([label for label in true_labels if label not in predictions])
        fp = len([pred for pred in predictions if pred not in true_labels])
        
        # 累加micro指标的计数
        total_tp += tp
        total_fn += fn
        total_fp += fp
        
        # 收集macro指标的数据
        for disease in set(true_labels + predictions):
            all_diseases.add(disease)
            if disease in true_labels and disease in predictions:
                disease_tp[disease] = disease_tp.get(disease, 0) + 1
            elif disease in true_labels and disease not in predictions:
                disease_fn[disease] = disease_fn.get(disease, 0) + 1
            elif disease not in true_labels and disease in predictions:
                disease_fp[disease] = disease_fp.get(disease, 0) + 1

        # Find missed diagnoses (in true labels but not in predictions)
        missed = [label for label in true_labels if label not in predictions]
        missed_count = len(missed)
        missed_total += missed_count

        # Find overdiagnoses (in predictions but not in true labels)
        over = [pred for pred in predictions if pred not in true_labels]
        over_count = len(over)
        over_total += over_count


    result_dict = {}
    
    # 计算 micro recall 和 micro precision
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    
    result_dict['micro_recall'] = micro_recall
    result_dict['micro_precision'] = micro_precision
    

    # 计算 macro recall 和 macro precision
    recalls = []
    precisions = []
    
    for disease in all_diseases:
        tp = disease_tp.get(disease, 0)
        fn = disease_fn.get(disease, 0)
        fp = disease_fp.get(disease, 0)
        
        # 某些疾病可能没有真实样本，导致分母为0
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        
        # 计算精确率
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
    
    macro_recall = np.mean(recalls) if recalls else 0
    macro_precision = np.mean(precisions) if precisions else 0
    
    result_dict['macro_recall'] = macro_recall
    result_dict['macro_precision'] = macro_precision
    
    result_dict['avg_missed_per_case'] = missed_total/len(df_merged)
    result_dict['avg_over_per_case'] = over_total/len(df_merged)

    # Print the results to console
    # print(f"Micro Recall: {micro_recall:.4f}")
    # print(f"Micro Precision: {micro_precision:.4f}")
    # print(f"Macro Recall: {macro_recall:.4f}")
    # print(f"Macro Precision: {macro_precision:.4f}")
    # print(f"Average Missed per Case: {result_dict['avg_missed_per_case']:.4f}")
    # print(f"Average Over per Case: {result_dict['avg_over_per_case']:.4f}")
    
    
    return result_dict


def main(args):
    labels = load_json(args.label_file)
    df_predictions = pd.read_csv(args.predict_result, dtype=str)

    # 将 df_predictions 中的 valid_prediction 字段中的 "['99591', '486', '42823']" 转换为 list ['99591', '486', '42823']
    import ast
    df_predictions['valid_prediction'] = df_predictions['valid_prediction'].apply(lambda x: ast.literal_eval(x) if x else [])
    if 'top10' in df_predictions.columns:
        df_predictions['top10'] = df_predictions['top10'].apply(lambda x: ast.literal_eval(x) if x else [])

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

    save_dir = os.path.join(os.path.dirname(args.predict_result),'recall_precision')
    os.makedirs(save_dir, exist_ok=True)

    # 只计算 recall 和 precision
    result_dict = count_recall_precision_diagnosis(df_merged)

    # 计算置信区间 (Bootstrapping 1000次)
    def bootstrap_confidence_interval(df, n_bootstrap=1000, confidence=0.95):
        bootstrap_results = []
        n_samples = len(df)
        
        for _ in tqdm(range(n_bootstrap)):
            # 有放回地随机抽样
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = df.iloc[bootstrap_indices]
            
            # 计算bootstrap样本的recall和precision
            bootstrap_result = count_recall_precision_diagnosis(bootstrap_sample)
            bootstrap_results.append(bootstrap_result)
        
        # 计算各指标的置信区间
        result_with_ci = {}
        metrics = ['micro_recall', 'micro_precision', 'macro_recall', 'macro_precision','avg_missed_per_case', 'avg_over_per_case']
        
        for metric in metrics:
            values = [result[metric] for result in bootstrap_results]
            lower_bound = np.percentile(values, (1 - confidence) / 2 * 100)
            upper_bound = np.percentile(values, (1 + confidence) / 2 * 100)
            result_with_ci[metric] = {
                'value': result_dict[metric],
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        return result_with_ci

    # 执行bootstrapping并计算置信区间
    result_with_ci = bootstrap_confidence_interval(df_merged, n_bootstrap=1000)
    
    # 保存 recall 和 precision 的结果及其置信区间
    def save_json(data, file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    save_json(result_with_ci, os.path.join(save_dir, 'recall_precision_ci.json'))


    

    
    

            
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--label_file", type=str, required=True, help="Path to the label file")
    parser.add_argument("--predict_result", type=str, required=True, help="Path to the prediction result file")
    args = parser.parse_args()

    main(args)