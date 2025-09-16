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

def distance_top_1_to_pred_all(df_merged):
    results = []
    distribution = {1: 0, 2: 0, 3: 0, 4: 0}  # 1: correct, 2: very close, 3: close, 4: nothing related
    
    for index, row in df_merged.iterrows():
        label_icd = row['top_1_icd']
        predictions = row['valid_prediction']
        label_prefix = label_icd[:3]
        
        # print(f"case_id: {row['case_id']}, label_icd: {label_icd}, predictions: {predictions}")
        case_scores = []
        for pred_icd in predictions:
            # Check if direct match (correct - in 40 classes)
            if pred_icd == label_icd:
                case_scores.append(1)
                continue
            
            # Check if in class_33 (very close)
            found_in_33 = False
            if label_prefix in class_33_dict and pred_icd in class_33_dict[label_prefix]:
                case_scores.append(2)
                found_in_33 = True
            if found_in_33:
                continue
            
            # Check if in class_12 (close)
            found_in_12 = False
            for category_range, icd_codes in class_12_dict.items():
                start, end = category_range.split('-')
                if start <= label_prefix <= end:  # First check if label is in this category range
                    if pred_icd in icd_codes:     # Then check if prediction is in the category's codes
                        case_scores.append(3)
                        found_in_12 = True
                        break
            if found_in_12:
                continue
            
            # If reached here, it's not related
            case_scores.append(4)
        # print(f"case_scores: {case_scores}")

        # Get the best relationship (smallest score)
        min_score = 4 if not case_scores else min(case_scores)
        # print(f"min_score: {min_score}")
    
        results.append({
            'case_id': row['case_id'],
            'label_icd': label_icd,
            'predictions': predictions,
            'case_scores': case_scores,
            'best_score': min_score
        })
    
        # Update distribution count
        distribution[min_score] += 1
    
    # Calculate percentages
    total_cases = len(results)
    distribution_percentage = {
        1: (distribution[1] / total_cases) * 100,
        2: (distribution[2] / total_cases) * 100,
        3: (distribution[3] / total_cases) * 100,
        4: (distribution[4] / total_cases) * 100
    }
    print(f"total_cases: {total_cases}, distribution: {distribution}, distribution_percentage: {distribution_percentage}")
    
    
    return {
        'total_cases': total_cases,
        'distribution': distribution,
        'distribution_percentage': distribution_percentage,
        'individual_results': results,
    }

    

def distance_label_all_to_pred_all(df_merged):
    results = []
    distribution = {1: 0, 2: 0, 3: 0, 4: 0}  # 1: correct, 2: very close, 3: close, 4: nothing related
    
    total_labels = 0
    
    for index, row in df_merged.iterrows():
        case_id = row['case_id']
        label_icds = row['labels_icd']
        predictions = row['valid_prediction']
        
        # print(f"case_id: {case_id}, label_icds: {label_icds}, predictions: {predictions}")
        
        case_labels_scores = {}
        
        for label_icd in label_icds:
            label_scores = []
            label_prefix = label_icd[:3]
            
            for pred_icd in predictions:
                # Check if direct match (correct - in 40 classes)
                if pred_icd == label_icd:
                    label_scores.append(1)
                    continue
                
                # Check if in class_33 (very close)
                found_in_33 = False
                if label_prefix in class_33_dict and pred_icd in class_33_dict[label_prefix]:
                    label_scores.append(2)
                    found_in_33 = True
                if found_in_33:
                    continue
                
                # Check if in class_12 (close)
                found_in_12 = False
                for category_range, icd_codes in class_12_dict.items():
                    start, end = category_range.split('-')
                    if start <= label_prefix <= end:  # Check if label is in this category range
                        if pred_icd in icd_codes:     # Check if prediction is in the category's codes
                            label_scores.append(3)
                            found_in_12 = True
                            break
                if found_in_12:
                    continue
                
                # If reached here, it's not related
                label_scores.append(4)
            
            # Get the best relationship (smallest score) for this label
            min_score = 4 if not label_scores else min(label_scores)
            # print(f"label_icd: {label_icd}, label_scores: {label_scores}, min_score: {min_score}")
            
            case_labels_scores[label_icd] = {
                'scores': label_scores,
                'best_score': min_score
            }
            
            # Update distribution count
            distribution[min_score] += 1
            total_labels += 1
        
        results.append({
            'case_id': case_id,
            'label_icds': label_icds,
            'predictions': predictions,
            'label_scores': case_labels_scores
        })

    # Calculate percentages
    distribution_percentage = {
        1: (distribution[1] / total_labels) * 100 if total_labels > 0 else 0,
        2: (distribution[2] / total_labels) * 100 if total_labels > 0 else 0,
        3: (distribution[3] / total_labels) * 100 if total_labels > 0 else 0,
        4: (distribution[4] / total_labels) * 100 if total_labels > 0 else 0
    }
    
    return {
        'total_labels': total_labels,
        'distribution': distribution,
        'distribution_percentage': distribution_percentage,
        'individual_results': results,
    }


def main(args):
    labels = load_json(args.label_file)
    df_predictions = pd.read_csv(args.predict_result, dtype=str)

    # 将 df_predictions 中的 valid_prediction 字段中的 "['99591', '486', '42823']" 转换为 list ['99591', '486', '42823']
    import ast
    df_predictions['valid_prediction'] = df_predictions['valid_prediction'].apply(lambda x: ast.literal_eval(x) if x else [])
    if 'top10' in df_predictions.columns:
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

    # 统计 primary 评测结果
    top_1_result = distance_top_1_to_pred_all(df_merged)
    save_path = os.path.join(os.path.dirname(args.predict_result), 'distance','top_1_to_pred_all.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(top_1_result, f, indent=4)

    # 统计 label 评测结果
    save_path = os.path.join(os.path.dirname(args.predict_result), 'distance','label_all_to_pred_all.json')
    all_result = distance_label_all_to_pred_all(df_merged)
    with open(save_path, 'w') as f:
        json.dump(all_result, f, indent=4)

    
    

            
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--label_file", type=str, required=True, help="Path to the label file")
    parser.add_argument("--predict_result", type=str, required=True, help="Path to the prediction result file")
    args = parser.parse_args()

    main(args)