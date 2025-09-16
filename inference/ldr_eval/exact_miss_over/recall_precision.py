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

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]

class_33_dict = {
    "038": ["0389"],
    "250": ["25080"],
    "262": ["262"],
    "276": ["2761", "2762", "27651"],
    "284": ["28419"],
    "285": ["2851"],
    "291": ["29181"],
    "311": ["311"],
    "348": ["34830", "34831"],
    "349": ["34982"],
    "401": ["4019"],
    "402": ["40291"],
    "403": ["40391"],
    "404": ["40491"],
    "410": ["41071"],
    "414": ["41401"],
    "415": ["41519"],
    "427": ["42731"],
    "428": ["42823", "42833"],
    "431": ["431"],
    "434": ["43411","43491"],
    "458": ["4589"],
    "486": ["486"],
    "507": ["5070"],
    "518": ["51881", "51884"],
    "576": ["5762"],
    "577": ["5770"],
    "584": ["5845", "5849"],
    "599": ["5990"],
    "682": ["6827"],
    "780": ["7802"],
    "995": ["99591"],
    "998": ["99859"]
}

class_12_dict = {
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
    

def count_miss_and_over_diagnosis(df_merged):
    """
    Count missed diagnoses and overdiagnoses
    - Missed diagnosis: diseases in true labels but not in predictions
    - Overdiagnosis: diseases in predictions but not in true labels
    
    Calculate:
    - Recall: TP / (TP + FN) = TP / total_true_positives
    - Precision: TP / (TP + FP) = TP / total_predicted_positives
    """
    missed_total = 0
    over_total = 0
    results = []
    
    miss_rates = []
    over_rates = []
    
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

        # Find missed diagnoses (in true labels but not in predictions)
        missed = [label for label in true_labels if label not in predictions]
        missed_count = len(missed)
        missed_total += missed_count
        
        # Find overdiagnoses (in predictions but not in true labels)
        over = [pred for pred in predictions if pred not in true_labels]
        over_count = len(over)
        over_total += over_count
        
        # Calculate miss rate and over rate for each case
        miss_rate = missed_count / len(true_labels) if len(true_labels) > 0 else 0
        over_rate = over_count / len(predictions) if len(predictions) > 0 else 0
        
        miss_rates.append(miss_rate)
        over_rates.append(over_rate)
        
        results.append({
            'case_id': row['case_id'],
            'missed_diagnosis': missed,
            'missed_count': missed_count,
            'miss_rate': miss_rate,
            'overdiagnosis': over,
            'over_count': over_count,
            'over_rate': over_rate
        })
        
        # 收集macro指标的数据
        for disease in set(true_labels + predictions):
            all_diseases.add(disease)
            if disease in true_labels and disease in predictions:
                disease_tp[disease] = disease_tp.get(disease, 0) + 1
            elif disease in true_labels and disease not in predictions:
                disease_fn[disease] = disease_fn.get(disease, 0) + 1
            elif disease not in true_labels and disease in predictions:
                disease_fp[disease] = disease_fp.get(disease, 0) + 1

    results_df = pd.DataFrame(results)
    
    result_dict = {}

    # Add summary statistics to result_dict
    result_dict['total_missed_diagnoses'] = missed_total
    result_dict['total_overdiagnoses'] = over_total
    result_dict['avg_missed_per_case'] = missed_total/len(df_merged)
    result_dict['avg_over_per_case'] = over_total/len(df_merged)
    result_dict['avg_miss_rate'] = np.mean(miss_rates)
    result_dict['avg_over_rate'] = np.mean(over_rates)
    
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
    
    # Print the results to console as well
    print(f"Total missed diagnoses: {result_dict['total_missed_diagnoses']}")
    print(f"Total overdiagnoses: {result_dict['total_overdiagnoses']}")
    print(f"Average missed diagnoses per case: {result_dict['avg_missed_per_case']:.2f}")
    print(f"Average overdiagnoses per case: {result_dict['avg_over_per_case']:.2f}")
    print(f"Average miss rate: {result_dict['avg_miss_rate']:.4f}")
    print(f"Average over rate: {result_dict['avg_over_rate']:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    

    # Disease level analysis
    # Count missed diagnoses by disease
    disease_missed = {}
    disease_total_in_labels = {}
    
    # Count overdiagnoses by disease
    disease_over = {}
    disease_total_in_preds = {}
    
    for _, row in df_merged.iterrows():
        predictions = row['valid_prediction']
        true_labels = row['labels_icd']
        
        # Count disease occurrences in true labels
        for label in true_labels:
            disease_total_in_labels[label] = disease_total_in_labels.get(label, 0) + 1
            if label not in predictions:
                disease_missed[label] = disease_missed.get(label, 0) + 1
        
        # Count disease occurrences in predictions
        for pred in predictions:
            disease_total_in_preds[pred] = disease_total_in_preds.get(pred, 0) + 1
            if pred not in true_labels:
                disease_over[pred] = disease_over.get(pred, 0) + 1

    # Calculate miss rate and over rate for each disease
    disease_analysis = []
    
    # Analyze missed diagnoses
    for disease, missed_count in disease_missed.items():
        total = disease_total_in_labels.get(disease, 0)
        miss_rate = missed_count / total if total > 0 else 0
        disease_analysis.append({
            'disease': disease,
            'total_in_labels': total,
            'missed_count': missed_count,
            'miss_rate': miss_rate,
            'over_count': 0,
            'over_rate': 0
        })
    
    # Analyze over-diagnoses
    for disease, over_count in disease_over.items():
        total = disease_total_in_preds.get(disease, 0)
        over_rate = over_count / total if total > 0 else 0
        
        # Check if disease already in analysis from missed diagnoses
        existing = next((item for item in disease_analysis if item['disease'] == disease), None)
        if existing:
            existing['over_count'] = over_count
            existing['over_rate'] = over_rate
        else:
            disease_analysis.append({
                'disease': disease,
                'total_in_labels': 0,
                'missed_count': 0,
                'miss_rate': 0,
                'over_count': over_count,
                'over_rate': over_rate,
                'total_in_predictions': total
            })
    
    # Convert to DataFrame for easier analysis
    disease_df = pd.DataFrame(disease_analysis)
    disease_df = disease_df.sort_values('total_in_labels', ascending=False)
   
    # Add disease-level analysis to result dictionary
    disease_dict = disease_df.to_dict(orient='records')

    # print('disease_dict:', disease_dict)
    
    return results_df, result_dict, disease_dict

 


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

    # 统计 missed 和 over
    results_df, result_dict, disease_dict = count_miss_and_over_diagnosis(df_merged)

    # Save results to CSV
    results_df.to_csv(os.path.join(save_dir, 'missed_over_diagnosis_results.csv'), index=False)

    # Save result_dict and disease_dict to JSON
    def save_json(data, file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    save_json(result_dict, os.path.join(save_dir, 'missed_over_diagnosis_summary.json'))
    save_json(disease_dict, os.path.join(save_dir, 'missed_over_diagnosis_disease_analysis.json'))

    # 单独保存 recall 和 precision 的结果，便于查看
    recall_precision_dict = {
        'micro_recall': result_dict['micro_recall'],
        'micro_precision': result_dict['micro_precision'],
        'macro_recall': result_dict['macro_recall'],
        'macro_precision': result_dict['macro_precision']
    }
    save_json(recall_precision_dict, os.path.join(save_dir, 'recall_precision.json'))




    

    
    

            
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--label_file", type=str, required=True, help="Path to the label file")
    parser.add_argument("--predict_result", type=str, required=True, help="Path to the prediction result file")
    args = parser.parse_args()

    main(args)