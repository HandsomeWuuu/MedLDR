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

def distance_pred_1_to_label_1(df_merged):
    # 统计距离
    df_merged['pred_top_1_icd'] = df_merged['valid_prediction'].apply(lambda x: x[0] if len(x) > 0 else None)
    df_merged['label_top_1_icd'] = df_merged['top_1_icd']
    
    results = {
        'correct': 0,
        'very_close': 0,
        'close': 0,
        'nothing_related': 0
    }
    
    for _, row in df_merged.iterrows():
        pred = row['pred_top_1_icd']
        label = row['label_top_1_icd']
        
        if pred is None:
            results['nothing_related'] += 1
            continue
        
        # Check if prediction matches the label exactly (40 class)
        if pred == label:
            results['correct'] += 1
            continue
        
        # Check if prediction is in the same 33-class category
        pred_in_33 = False
        label_in_33 = False
        for category, codes in class_33_dict.items():
            if label in codes:
                label_in_33 = True
                if pred in codes:
                    pred_in_33 = True
                    break
        
        if pred_in_33 and label_in_33:
            results['very_close'] += 1
            continue
        
        # Check if prediction is in the same 12-class category
        pred_in_12 = False
        label_in_12 = False
        for category, codes in class_12_dict.items():
            if label in codes:
                label_in_12 = True
                if pred in codes:
                    pred_in_12 = True
                    break
        
        if pred_in_12 and label_in_12:
            results['close'] += 1
            continue
        
        # If we've reached here, they're not related
        results['nothing_related'] += 1
    
    total = sum(results.values())
    print(f"Total cases: {total}")
    print(f"Correct (exact match): {results['correct']} ({results['correct']/total*100:.2f}%)")
    print(f"Very close (same 33-class): {results['very_close']} ({results['very_close']/total*100:.2f}%)")
    print(f"Close (same 12-class): {results['close']} ({results['close']/total*100:.2f}%)")
    print(f"Nothing related: {results['nothing_related']} ({results['nothing_related']/total*100:.2f}%)")
    
    return results


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

    dict_result = distance_pred_1_to_label_1(df_merged)

    save_path = os.path.join(os.path.dirname(args.predict_result), 'distance','pred_1_to_top_1.json')
    # if not os.path.exists(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(dict_result, f, indent=4)
            
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--label_file", type=str, required=True, help="Path to the label file")
    parser.add_argument("--predict_result", type=str, required=True, help="Path to the prediction result file")
    args = parser.parse_args()

    main(args)