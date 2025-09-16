import json
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, classification_report
from collections import defaultdict

def load_json(file_path):
    """Load JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = []
            for line in file:
                data.append(json.loads(line.strip()))
            return data

def extract_icd_codes_from_mapping():
    """Extract all 74 ICD10 codes from diagnosis_icd_mapping.json"""
    try:
        mapping_file = 'eicu_process_data/diagnosis_icd_mapping.json'
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        icd_codes = list(data['icd10_to_diagnosis_mapping'].keys())
        print(f"Extracted {len(icd_codes)} ICD10 codes from mapping file")
        return icd_codes
    except Exception as e:
        print(f"Failed to read mapping file: {e}")
        # Hardcoded 74 ICD10 codes as fallback
        icd_codes = [
            'E86.1', 'D64.9', 'R65.21', 'I50.9', 'J18.9', 'I63.50', 'I10', 'J44.9',
            'I48.0', 'A41.9', 'I95.9', 'I25.10', 'J45', 'J69.0', 'J96.00', 'N18.6',
            'E87.5', 'E10.1', 'E87.2', 'N17.9', 'E03.9', 'E78.5', 'J44.1', 'N39.0',
            'J91.8', 'J80', 'N18.9', 'J96.91', 'R73.9', 'R41.82', 'G93.41', 'I50.1',
            'K92.2', 'I46.9', 'R65.2', 'G93.40', 'E87.70', 'I67.8', 'D72.829', 'E83.42',
            'F32.9', 'I21.3', 'I21.4', 'N30.9', 'E66.9', 'R65.20', 'R40.0', 'R10.9',
            'D62', 'F43.0', 'R56.9', 'J96.92', 'F10.239', 'F05', 'R11.0', 'G47.33',
            'R50.9', 'J96.10', 'E46', 'R07.9', 'R00.0', 'E83.51', 'R57.0', 'E16.2',
            'N18.3', 'I62.9', 'J98.11', 'D68.32', 'D69.6', 'D68.9', 'F03', 'E87.6', 
            'F41.9','E87.0'
        ]
        print(f"Using hardcoded {len(icd_codes)} ICD10 codes")
        return icd_codes

def create_icd_mapping(icd_codes):
    """Create mapping from ICD code to index"""
    icd_code_to_idx = {code: idx for idx, code in enumerate(icd_codes)}
    icd_idx_to_code = {idx: code for idx, code in enumerate(icd_codes)}
    return icd_code_to_idx, icd_idx_to_code

def accuracy(y_true, y_pred):
    """Calculate multilabel accuracy (IoU)"""
    count = sum(sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i])) for i in range(y_true.shape[0]))
    return count / y_true.shape[0]

def compute_multi_label_metrics(label_list, pred_list):
    """Compute multilabel classification metrics"""
    label_list = np.array(label_list)
    pred_list = np.array(pred_list)

    metrics = {
        'emr_accuracy': round(accuracy_score(label_list, pred_list), 4),
        'accuracy': round(accuracy(label_list, pred_list), 4),
        'precision': round(precision_score(label_list, pred_list, average='samples', zero_division=0), 4),
        'recall': round(recall_score(label_list, pred_list, average='samples', zero_division=0), 4),
        'f1': round(f1_score(label_list, pred_list, average='samples', zero_division=0), 4),
        'hamming_loss': round(hamming_loss(label_list, pred_list), 4)
    }
    return metrics

def compute_recall_for_disease_group(df_merged, pred_column, label_column, group_name):
    """Compute recall for a specific disease group"""
    recall_count = 0
    total_count = 0
    
    for _, row in df_merged.iterrows():
        label_diseases = row[label_column] if isinstance(row[label_column], list) else []
        pred_diseases = row[pred_column] if isinstance(row[pred_column], list) else []
        
        if label_diseases:  # Only compute if label group is not empty
            total_count += 1
            # Check if any disease in label group is in prediction
            if any(disease in pred_diseases for disease in label_diseases):
                recall_count += 1
    
    if total_count > 0:
        recall = recall_count / total_count
        print(f'{group_name} Recall: {recall:.4f} ({recall_count}/{total_count})')
        return round(recall, 4)
    else:
        print(f'{group_name} Recall: N/A (no samples)')
        return 0.0

def main(args):
    print("Start loading data...")
    
    # Load label data
    labels_data = load_json(args.label_file)
    print(f"Loaded {len(labels_data)} label samples")
    
    # Load prediction results
    df_predictions = pd.read_csv(args.predict_result, dtype=str)
    print(f"Loaded {len(df_predictions)} prediction samples")
    
    # Convert string list in prediction results to actual list
    def safe_eval_list(x):
        """Safely parse list string"""
        try:
            if pd.isna(x) or x == 'nan':
                return []
            if isinstance(x, str):
                x = x.strip()
                if x.startswith('[') and x.endswith(']'):
                    return eval(x)
                else:
                    return [x] if x else []
            elif isinstance(x, list):
                return x
            else:
                return []
        except:
            return []
    
    df_predictions['valid_prediction'] = df_predictions['valid_prediction'].apply(safe_eval_list)
    
    if 'top20' in df_predictions.columns:
        df_predictions['top20'] = df_predictions['top20'].apply(safe_eval_list)
    
    # Extract ICD codes from mapping file and create mapping
    icd_codes = extract_icd_codes_from_mapping()
    icd_code_to_idx, icd_idx_to_code = create_icd_mapping(icd_codes)
    num_classes = len(icd_codes)
    
    print(f"Using {num_classes} ICD codes for evaluation")
    
    # Process label data, create DataFrame
    labels_list = []
    for item in labels_data:
        single_label = {
            'case_id': str(item['case_id']),  # eICU uses case_id field directly
            'labels_icd': [],
            'primary_diseases': [],
            'major_diseases': [],
            'top_1_icd': None
        }
        
        # Extract all diseases from output field
        if 'output' in item and 'all_diseases' in item['output'] and isinstance(item['output']['all_diseases'], list) and len(item['output']['all_diseases']) > 0:
            single_label['labels_icd'] = [disease['icd10code'] for disease in item['output']['all_diseases'] if 'icd10code' in disease]
            single_label['labels_icd'] = [code for code in single_label['labels_icd'] if code and code != '']

        # Extract primary diseases from primary_diseases field
        if 'output' in item and 'primary_diseases' in item['output'] and isinstance(item['output']['primary_diseases'], list):
            single_label['primary_diseases'] = [disease['icd10code'] for disease in item['output']['primary_diseases'] if 'icd10code' in disease]
            single_label['primary_diseases'] = [code for code in single_label['primary_diseases'] if code and code != '']
            if single_label['primary_diseases']:
                single_label['top_1_icd'] = single_label['primary_diseases'][0]
        # Extract major diseases from major_diseases field
        if 'output' in item and 'major_diseases' in item['output'] and isinstance(item['output']['major_diseases'], list):
            single_label['major_diseases'] = [disease['icd10code'] for disease in item['output']['major_diseases'] if 'icd10code' in disease]
            single_label['major_diseases'] = [code for code in single_label['major_diseases'] if code and code != '']

        labels_list.append(single_label)
    
    df_labels = pd.DataFrame(labels_list)
    print(f"Processed label data: {len(df_labels)} samples")
    print("Label data example:")
    print(df_labels.head())
    
    # Merge prediction and label data
    df_merged = pd.merge(df_predictions, df_labels, on='case_id', how='inner')
    print(f"Merged data: {len(df_merged)} samples")

    if len(df_merged) == 0:
        print("Error: No data after merging, please check if case_id matches")
        return
    
    # 1. Compute multilabel classification metrics
    print("\nComputing multilabel classification metrics...")
    label_list = []
    pred_list = []
    
    for _, row in df_merged.iterrows():
        # Prediction vector
        pred_vector = np.zeros(num_classes)
        for code in row['valid_prediction']:
            if code in icd_code_to_idx:
                pred_vector[icd_code_to_idx[code]] = 1

        pred_list.append(pred_vector)
        
        # Label vector
        label_vector = np.zeros(num_classes)
        for code in row['labels_icd']:
            if code in icd_code_to_idx:
                label_vector[icd_code_to_idx[code]] = 1
        label_list.append(label_vector)
    
    # Classification report
    try:
        report = classification_report(label_list, pred_list, output_dict=True, zero_division=0)
        
        avg_report = {k: v for k, v in report.items() if 'avg' in k}
        non_avg_report = {k: v for k, v in report.items() if 'avg' not in k and k != 'accuracy'}
        
        # Create detailed report DataFrame
        if non_avg_report:
            detailed_df = pd.DataFrame(non_avg_report).T
            detailed_df['icd_code'] = [icd_idx_to_code.get(int(idx), f'class_{idx}') for idx in detailed_df.index if str(idx).isdigit()]
            print("Detailed classification report saved")
        
    except Exception as e:
        print(f"Error generating classification report: {e}")
        avg_report = {}
    
    # Compute main metrics
    metrics = compute_multi_label_metrics(label_list, pred_list)
    if avg_report:
        metrics.update({
            'macro_precision': round(avg_report.get('macro avg', {}).get('precision', 0), 4),
            'macro_recall': round(avg_report.get('macro avg', {}).get('recall', 0), 4),
            'macro_f1': round(avg_report.get('macro avg', {}).get('f1-score', 0), 4),
            'weighted_precision': round(avg_report.get('weighted avg', {}).get('precision', 0), 4),
            'weighted_recall': round(avg_report.get('weighted avg', {}).get('recall', 0), 4),
            'weighted_f1': round(avg_report.get('weighted avg', {}).get('f1-score', 0), 4)
        })
    
    # 2. Compute Top-1 accuracy
    print("Computing Top-1 accuracy...")
    top1_correct = 0
    
    # Use the first primary_diseases as top-1 label
    for _, row in df_merged.iterrows():
        if row['top_1_icd'] and len(row['valid_prediction']) > 0:
            if row['top_1_icd'] in row['valid_prediction']:
                top1_correct += 1
    
    # Count samples with top-1 label
    valid_top1_samples = sum(1 for _, row in df_merged.iterrows() if row['top_1_icd'])
    
    if valid_top1_samples > 0:
        top1_accuracy = top1_correct / valid_top1_samples
        print(f'Top-1 ICD Accuracy: {top1_accuracy:.4f} ({top1_correct}/{valid_top1_samples})')
        metrics['top1_accuracy'] = round(top1_accuracy, 4)
    else:
        print('Top-1 ICD Accuracy: N/A (no valid samples)')
        metrics['top1_accuracy'] = 0.0
    
    # 3. Compute Recall@K (if top20 data exists)
    if 'top20' in df_predictions.columns:
        print("Computing Recall@K metrics...")
        recall_at_20 = 0
        recall_at_10 = 0
        recall_at_5 = 0
        recall_at_3 = 0

        for _, row in df_merged.iterrows():
            true_labels = set(row['labels_icd'])
            if '' in true_labels:
                raise ValueError(f"Sample {row['case_id']} label list contains empty string, cannot compute recall")
            
            if not true_labels:
                raise ValueError(f"Sample {row['case_id']} label list is empty, cannot compute recall")
                
            top20_pred = set(row['top20'][:20]) if len(row['top20']) >= 20 else set(row['top20'])
            top10_pred = set(row['top20'][:10]) if len(row['top20']) >= 10 else set(row['top20'])
            top5_pred = set(row['top20'][:5]) if len(row['top20']) >= 5 else set(row['top20'])
            top3_pred = set(row['top20'][:3]) if len(row['top20']) >= 3 else set(row['top20'])

            if true_labels.intersection(top20_pred):
                recall_at_20 += 1
            if true_labels.intersection(top10_pred):
                recall_at_10 += 1
            if true_labels.intersection(top5_pred):
                recall_at_5 += 1
            if true_labels.intersection(top3_pred):
                recall_at_3 += 1

        total_samples = len(df_merged)
        recall_at_20 = recall_at_20 / total_samples
        recall_at_10 = recall_at_10 / total_samples
        recall_at_5 = recall_at_5 / total_samples
        recall_at_3 = recall_at_3 / total_samples

        print(f'Recall@20: {recall_at_20:.4f}')
        print(f'Recall@10: {recall_at_10:.4f}')
        print(f'Recall@5: {recall_at_5:.4f}')
        print(f'Recall@3: {recall_at_3:.4f}')
        
        metrics['recall@20'] = round(recall_at_20, 4)
        metrics['recall@10'] = round(recall_at_10, 4)
        metrics['recall@5'] = round(recall_at_5, 4)
        metrics['recall@3'] = round(recall_at_3, 4)

    # 4. Compute recall for primary_diseases and major_diseases
    print("\nComputing recall for specific disease groups...")
    metrics['primary_diseases_recall'] = compute_recall_for_disease_group(
        df_merged, 'valid_prediction', 'primary_diseases', 'Primary Diseases'
    )
    metrics['major_diseases_recall'] = compute_recall_for_disease_group(
        df_merged, 'valid_prediction', 'major_diseases', 'Major Diseases'
    )

    # Save metrics
    metric_path = os.path.join(os.path.dirname(args.predict_result), 'result_metrics.json')
    with open(metric_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nMetrics saved to: {metric_path}")
    print("Main metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute classification metrics for eICU dataset')
    parser.add_argument('--predict_result', type=str, required=True, 
                        help='Prediction result CSV file path')
    parser.add_argument('--label_file', type=str, required=True,
                        help='Label JSON file path')

    args = parser.parse_args()
    main(args)
