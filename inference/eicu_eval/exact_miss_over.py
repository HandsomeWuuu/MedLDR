'''
    Statistics for eICU evaluation results
    Calculate missed diagnosis and overdiagnosis scores
    1. Missed diagnosis: Diseases present in the true labels but not in the predictions
    2. Overdiagnosis: Diseases present in the predictions but not in the true labels
'''
import pandas as pd
import json
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, classification_report
import argparse

def load_json(file_path):
    """Load JSON file"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]

def count_miss_and_over_diagnosis(df_merged):
    """
    Count missed diagnosis and overdiagnosis
    - Missed diagnosis: Diseases present in the true labels but not in the predictions
    - Overdiagnosis: Diseases present in the predictions but not in the true labels
    """
    missed_total = 0
    over_total = 0
    results = []
    
    miss_rates = []
    over_rates = []
    
    for _, row in df_merged.iterrows():
        predictions = row['valid_prediction']
        true_labels = row['labels_icd']

        # Find missed diagnosis (in true labels but not in predictions)
        missed = [label for label in true_labels if label not in predictions]
        missed_count = len(missed)
        missed_total += missed_count
        
        # Find overdiagnosis (in predictions but not in true labels)
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
            'over_rate': over_rate,
            'true_labels_count': len(true_labels),
            'predictions_count': len(predictions)
        })

    results_df = pd.DataFrame(results)
    
    result_dict = {}

    # Add summary statistics to result dict
    result_dict['total_cases'] = len(df_merged)
    result_dict['total_missed_diagnoses'] = missed_total
    result_dict['total_overdiagnoses'] = over_total
    result_dict['avg_missed_per_case'] = missed_total/len(df_merged)
    result_dict['avg_over_per_case'] = over_total/len(df_merged)
    result_dict['avg_miss_rate'] = np.mean(miss_rates)
    result_dict['avg_over_rate'] = np.mean(over_rates)
    result_dict['std_miss_rate'] = np.std(miss_rates)
    result_dict['std_over_rate'] = np.std(over_rates)
    
    # Print results to console
    print(f"Total cases: {result_dict['total_cases']}")
    print(f"Total missed diagnoses: {result_dict['total_missed_diagnoses']}")
    print(f"Total overdiagnoses: {result_dict['total_overdiagnoses']}")
    print(f"Average missed diagnoses per case: {result_dict['avg_missed_per_case']:.2f}")
    print(f"Average overdiagnoses per case: {result_dict['avg_over_per_case']:.2f}")
    print(f"Average miss rate: {result_dict['avg_miss_rate']:.4f} ± {result_dict['std_miss_rate']:.4f}")
    print(f"Average over rate: {result_dict['avg_over_rate']:.4f} ± {result_dict['std_over_rate']:.4f}")
    

    # Disease-level analysis
    # Count missed diagnosis by disease
    disease_missed = {}
    disease_total_in_labels = {}
    
    # Count overdiagnosis by disease
    disease_over = {}
    disease_total_in_preds = {}
    
    for _, row in df_merged.iterrows():
        predictions = row['valid_prediction']
        true_labels = row['labels_icd']
        
        # Count occurrences in true labels
        for label in true_labels:
            disease_total_in_labels[label] = disease_total_in_labels.get(label, 0) + 1
            if label not in predictions:
                disease_missed[label] = disease_missed.get(label, 0) + 1
        
        # Count occurrences in predictions
        for pred in predictions:
            disease_total_in_preds[pred] = disease_total_in_preds.get(pred, 0) + 1
            if pred not in true_labels:
                disease_over[pred] = disease_over.get(pred, 0) + 1

    # Calculate miss rate and over rate for each disease
    disease_analysis = []
    
    # Analyze missed and overdiagnosis
    all_diseases = set(disease_total_in_labels.keys()) | set(disease_total_in_preds.keys())
    
    for disease in all_diseases:
        total_in_labels = disease_total_in_labels.get(disease, 0)
        total_in_preds = disease_total_in_preds.get(disease, 0)
        missed_count = disease_missed.get(disease, 0)
        over_count = disease_over.get(disease, 0)
        
        miss_rate = missed_count / total_in_labels if total_in_labels > 0 else 0
        over_rate = over_count / total_in_preds if total_in_preds > 0 else 0
        
        disease_analysis.append({
            'disease': disease,
            'total_in_labels': total_in_labels,
            'total_in_predictions': total_in_preds,
            'missed_count': missed_count,
            'miss_rate': miss_rate,
            'over_count': over_count,
            'over_rate': over_rate
        })
    
    # Convert to DataFrame for analysis
    disease_df = pd.DataFrame(disease_analysis)
    disease_df = disease_df.sort_values('total_in_labels', ascending=False)
   
    # Add disease-level analysis to result dict
    disease_dict = disease_df.to_dict(orient='records')

    print(f"\nDisease-level analysis completed, analyzed {len(disease_dict)} diseases")
    
    return results_df, result_dict, disease_dict

def analyze_severity(results_df, result_dict):
    """Analyze severity distribution of missed and overdiagnosis"""
    severity_analysis = {}
    
    # Missed diagnosis severity analysis
    miss_rate_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    miss_severity = {}
    
    for low, high in miss_rate_ranges:
        range_name = f"{low:.1f}-{high:.1f}"
        count = len(results_df[(results_df['miss_rate'] >= low) & (results_df['miss_rate'] < high)])
        if high == 1.0:  # Include 1.0
            count = len(results_df[(results_df['miss_rate'] >= low) & (results_df['miss_rate'] <= high)])
        miss_severity[range_name] = count
    
    # Overdiagnosis severity analysis
    over_severity = {}
    for low, high in miss_rate_ranges:
        range_name = f"{low:.1f}-{high:.1f}"
        count = len(results_df[(results_df['over_rate'] >= low) & (results_df['over_rate'] < high)])
        if high == 1.0:  # Include 1.0
            count = len(results_df[(results_df['over_rate'] >= low) & (results_df['over_rate'] <= high)])
        over_severity[range_name] = count
    
    severity_analysis['miss_rate_distribution'] = miss_severity
    severity_analysis['over_rate_distribution'] = over_severity
    
    print("\nMiss rate distribution:")
    for range_name, count in miss_severity.items():
        percentage = count / len(results_df) * 100
        print(f"  {range_name}: {count} cases ({percentage:.1f}%)")
    
    print("\nOver rate distribution:")
    for range_name, count in over_severity.items():
        percentage = count / len(results_df) * 100
        print(f"  {range_name}: {count} cases ({percentage:.1f}%)")
    
    return severity_analysis

def main(args):
    """Main function"""
    # Load data
    labels = load_json(args.label_file)
    df_predictions = pd.read_csv(args.predict_result, dtype=str)

    # Process list fields in prediction results
    import ast
    df_predictions['valid_prediction'] = df_predictions['valid_prediction'].apply(lambda x: ast.literal_eval(x) if x else [])
    if 'top20' in df_predictions.columns:
        df_predictions['top20'] = df_predictions['top20'].apply(lambda x: ast.literal_eval(x) if x else [])

    # Process label data, refer to compute_metrics.py logic
    labels_list = []
    for item in labels:
        single_label = {
            'case_id': str(item['case_id']),
            'labels_icd': [],
            'primary_diseases': [],
            'major_diseases': [],
            'top_1_icd': None
        }
        
        # Extract all diseases from output field
        if 'output' in item and 'all_diseases' in item['output'] and isinstance(item['output']['all_diseases'], list) and len(item['output']['all_diseases']) > 0:
            single_label['labels_icd'] = [disease['icd10code'] for disease in item['output']['all_diseases'] if 'icd10code' in disease]
            # Ensure no empty string in labels
            single_label['labels_icd'] = [code for code in single_label['labels_icd'] if code and code != '']

        # Extract primary diseases from primary_diseases field
        if 'output' in item and 'primary_diseases' in item['output'] and isinstance(item['output']['primary_diseases'], list):
            single_label['primary_diseases'] = [disease['icd10code'] for disease in item['output']['primary_diseases'] if 'icd10code' in disease]
            # Ensure no empty string in primary diseases
            single_label['primary_diseases'] = [code for code in single_label['primary_diseases'] if code and code != '']
            # Set top-1 as the first primary disease
            if single_label['primary_diseases']:
                single_label['top_1_icd'] = single_label['primary_diseases'][0]
        
        # Extract major diseases from major_diseases field
        if 'output' in item and 'major_diseases' in item['output'] and isinstance(item['output']['major_diseases'], list):
            single_label['major_diseases'] = [disease['icd10code'] for disease in item['output']['major_diseases'] if 'icd10code' in disease]
            # Ensure no empty string in major diseases
            single_label['major_diseases'] = [code for code in single_label['major_diseases'] if code and code != '']

        # If no primary_diseases but has all_diseases, use the first disease as top_1
        if single_label['top_1_icd'] is None and single_label['labels_icd']:
            single_label['top_1_icd'] = single_label['labels_icd'][0]
            print(f"Warning: Case {item['case_id']} has no primary_diseases, use the first disease as top_1_icd")
        
        labels_list.append(single_label)
    
    df_labels = pd.DataFrame(labels_list)
    print(f"Label data sample:")
    print(df_labels.head())
    
    # Merge data
    df_merged = pd.merge(df_predictions, df_labels, on='case_id', how='inner')
    print(f"Successfully merged data for {len(df_merged)} cases")

    if len(df_merged) == 0:
        print("Error: No data after merging, please check if case_id matches")
        return

    # Create save directory
    save_dir = os.path.join(os.path.dirname(args.predict_result), 'miss_over')
    os.makedirs(save_dir, exist_ok=True)

    # Count missed diagnosis and overdiagnosis
    print("\nStart analyzing missed diagnosis and overdiagnosis...")
    results_df, result_dict, disease_dict = count_miss_and_over_diagnosis(df_merged)

    # Analyze severity
    print("\nStart analyzing severity distribution...")
    severity_analysis = analyze_severity(results_df, result_dict)

    # Save results
    print(f"\nSaving results to {save_dir}")
    
    # Save detailed results to CSV
    results_df.to_csv(os.path.join(save_dir, 'eicu_missed_over_diagnosis_results.csv'), index=False)
    print(f"Detailed results saved to: eicu_missed_over_diagnosis_results.csv")

    # Save summary results to JSON
    def save_json(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

    # Merge all analysis results
    final_results = {
        'summary': result_dict,
        'severity_analysis': severity_analysis,
        'top_missed_diseases': sorted(disease_dict, key=lambda x: x['missed_count'], reverse=True)[:20],
        'top_over_diseases': sorted(disease_dict, key=lambda x: x['over_count'], reverse=True)[:20]
    }

    save_json(final_results, os.path.join(save_dir, 'eicu_missed_over_diagnosis_summary.json'))
    save_json(disease_dict, os.path.join(save_dir, 'eicu_missed_over_diagnosis_disease_analysis.json'))
    
    print(f"Summary results saved to: eicu_missed_over_diagnosis_summary.json")
    print(f"Disease-level analysis saved to: eicu_missed_over_diagnosis_disease_analysis.json")
    
    # Print top 10 most missed diseases
    top_missed = sorted(disease_dict, key=lambda x: x['missed_count'], reverse=True)[:10]
    print("\nTop 10 most missed diseases:")
    for i, disease in enumerate(top_missed, 1):
        print(f"{i:2d}. Disease {disease['disease']}: missed {disease['missed_count']}/{disease['total_in_labels']} times "
              f"(miss rate: {disease['miss_rate']:.3f})")
    
    # Print top 10 most overdiagnosed diseases
    top_over = sorted(disease_dict, key=lambda x: x['over_count'], reverse=True)[:10]
    print("\nTop 10 most overdiagnosed diseases:")
    for i, disease in enumerate(top_over, 1):
        print(f"{i:2d}. Disease {disease['disease']}: overdiagnosed {disease['over_count']}/{disease['total_in_predictions']} times "
              f"(over rate: {disease['over_rate']:.3f})")

    print(f"\nAnalysis completed! All results saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate missed diagnosis and overdiagnosis in eICU data")
    parser.add_argument("--label_file", type=str, required=True, help="Path to label file")
    parser.add_argument("--predict_result", type=str, required=True, help="Path to prediction result file")
    args = parser.parse_args()

    main(args)