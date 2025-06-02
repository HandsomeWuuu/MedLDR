import pandas as pd
import json
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, classification_report
import argparse
import ast
import pickle
from sklearn.utils import resample
import random


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


def main(args):
   
    labels = load_json(args.label_file)
    df_predictions = pd.read_csv(args.predict_result, dtype=str)

    
    df_predictions['valid_prediction'] = df_predictions['valid_prediction'].apply(lambda x: ast.literal_eval(x) if x else [])
    # df_predictions['top10'] = df_predictions['top10'].apply(lambda x: ast.literal_eval(x) if x else [])
    
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

    # Number of bootstrap iterations
    n_iterations = 1000
    # n_iterations = 5
    
    # Initialize dictionaries to store bootstrap results
    bootstrap_results = {
        'overall': [],
        'categories': {cat: [] for cat in overall_dict.keys()}
    }
    
    print(f"Performing bootstrap with {n_iterations} iterations...")
    
    # Perform bootstrap resampling
    for i in range(n_iterations):
        if i % 100 == 0:
            print(f"Bootstrap iteration {i}/{n_iterations}")
            
        # Resample with replacement
        bootstrap_indices = random.choices(range(len(df_merged)), k=len(df_merged))
        bootstrap_df = df_merged.iloc[bootstrap_indices]
        
        # Calculate metrics for this bootstrap sample
        top1_correct = 0
        category_correct = {cat: 0 for cat in overall_dict.keys()}
        category_total = {cat: 0 for cat in overall_dict.keys()}
        
        for _, row in bootstrap_df.iterrows():
            top1_icd_code = row['top_1_icd'] 
            predict_codes = row['valid_prediction']
            
            # Find which category the top1_icd belongs to
            top1_category = None
            for category, codes in overall_dict.items():
                for code in codes:
                    if top1_icd_code.startswith(code[:3]):
                        top1_category = category
                        break
                if top1_category:
                    break
            
            if top1_category:
                category_total[top1_category] += 1
                
                # Check if any prediction falls into the same category
                predict_categories = set()
                for pred_code in predict_codes:
                    for category, codes in overall_dict.items():
                        if any(pred_code.startswith(c[:3]) for c in codes):
                            predict_categories.add(category)
                            break
                
                if top1_category in predict_categories:
                    category_correct[top1_category] += 1
                    top1_correct += 1
        
        # Calculate accuracies for this bootstrap sample
        bootstrap_accuracy = top1_correct / len(bootstrap_df) if len(bootstrap_df) > 0 else 0
        bootstrap_results['overall'].append(bootstrap_accuracy)
        
        for category in overall_dict.keys():
            if category_total[category] > 0:
                cat_accuracy = category_correct[category] / category_total[category]
            else:
                cat_accuracy = 0
            bootstrap_results['categories'][category].append(cat_accuracy)
    
    # Calculate the actual metrics on the full dataset
    top1_correct = 0
    category_metrics = {}
    for category in overall_dict.keys():
        category_metrics[category] = {
            'correct': 0,
            'total': 0,
            'accuracy': 0.0
        }
    
    for _, row in df_merged.iterrows():
        top1_icd_code = row['top_1_icd'] 
        predict_codes = row['valid_prediction']
        
        # Find which category the top1_icd belongs to
        top1_category = None
        for category, codes in overall_dict.items():
            for code in codes:
                if top1_icd_code.startswith(code[:3]):
                    top1_category = category
                    break
            if top1_category:
                break
        
        if top1_category:
            category_metrics[top1_category]['total'] += 1
            
            # Check if any prediction falls into the same category
            predict_categories = set()
            for pred_code in predict_codes:
                for category, codes in overall_dict.items():
                    if any(pred_code.startswith(c[:3]) for c in codes):
                        predict_categories.add(category)
                        break
            
            if top1_category in predict_categories:
                category_metrics[top1_category]['correct'] += 1
                top1_correct += 1
    
    # Calculate overall accuracy
    top1_accuracy = top1_correct / len(df_merged)
    print(f'Top-1 Category Accuracy: {top1_accuracy:.4f}')
    
    # raise ValueError('Stop here')
    # Calculate confidence intervals and prepare final metrics
    final_metrics = {
        'overall': {
            'mean': top1_accuracy,
            'CI': [np.percentile(bootstrap_results['overall'], 2.5), 
                   np.percentile(bootstrap_results['overall'], 97.5)],
            'all': bootstrap_results['overall']
        },
        'categories': {}
    }
    
    # Calculate category-wise accuracy and confidence intervals
    for category in overall_dict.keys():
        if category_metrics[category]['total'] > 0:
            category_metrics[category]['accuracy'] = category_metrics[category]['correct'] / category_metrics[category]['total']
            
            final_metrics['categories'][category] = {
                'mean': category_metrics[category]['accuracy'],
                'CI': [np.percentile(bootstrap_results['categories'][category], 2.5),
                       np.percentile(bootstrap_results['categories'][category], 97.5)],
                'all': bootstrap_results['categories'][category]
            }
            
            print(f"Category {category} Accuracy: {category_metrics[category]['accuracy']:.4f} "
                  f"({category_metrics[category]['correct']}/{category_metrics[category]['total']}) "
                  f"95% CI: [{final_metrics['categories'][category]['CI'][0]:.4f}, {final_metrics['categories'][category]['CI'][1]:.4f}]")
    
    print('final_metrics', final_metrics)
    # Define the output path for metrics
    os.makedirs(os.path.join(os.path.dirname(args.predict_result), 'bootstrap_12_disease'), exist_ok=True)
    metric_path = os.path.join(os.path.dirname(args.predict_result), 'bootstrap_12_disease', 'primary_by_disease_category.pkl')
    
    # Save metrics as pickle
    with open(metric_path, 'wb') as f:
        pickle.dump(final_metrics, f)
    
    print(f"Metrics saved to {metric_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_result', type=str, help='Path to the inference results CSV file')
    parser.add_argument('--label_file', type=str, help='Path to the labels JSON file')

    args = parser.parse_args()
    main(args)
