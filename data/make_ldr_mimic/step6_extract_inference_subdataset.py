import json
import numpy as np
import os
import sys
from tqdm import tqdm

def extract_subset_data(input_file1, input_file2=None):
    with open(input_file1, 'r', encoding='utf-8') as f:
        process_data1 = json.load(f)
    print(f"len of process_data1:{len(process_data1)}")

    if input_file2 is not None:
        with open(input_file2, 'r', encoding='utf-8') as f:
            process_data2 = json.load(f)
        print(f"len of process_data2:{len(process_data2)}")
        process_data = process_data1 + process_data2
    else:
        process_data = process_data1

    print(f"len of process_data:{len(process_data)}")

    # 1. Collect unique ICD codes
    icd_code_list = []
    for case in tqdm(process_data):
        icd_code_list.extend(case['output_dict']['output_icd_id'])
    unique_icd_codes = list(set(icd_code_list))
    print(f"unique icd codes:{unique_icd_codes}")
    print(f"len of unique icd codes:{len(unique_icd_codes)}")

    # 2. For each ICD code, sample up to 30 cases (prefer top_1_icd, fill from others if needed)
    sample_case_ids = set()
    for icd_code in tqdm(unique_icd_codes):
        case_ids = [case['case_id'] for case in process_data if icd_code in case['output_dict']['output_icd_id']]
        top_1_case_ids = [case['case_id'] for case in process_data if icd_code == case['top_1_icd']['code']]
        remain_case_ids = list(set(case_ids) - set(top_1_case_ids))
        if len(top_1_case_ids) < 30:
            sample_cases = np.random.choice(remain_case_ids, 30 - len(top_1_case_ids), replace=False)
            sample_cases = np.concatenate([top_1_case_ids, sample_cases])
        else:
            sample_cases = np.random.choice(top_1_case_ids, 30, replace=False)
        print(f"sample len {len(sample_cases)} icd_code:{icd_code} from len of case_ids:{len(case_ids)}")
        sample_case_ids.update(sample_cases)

    print(f"len of sample_case_ids:{len(sample_case_ids)}")

    # 3. Extract and save sampled data
    sample_data = [case for case in process_data if case['case_id'] in sample_case_ids]
    save_sample_path = os.path.join(os.path.dirname(input_file1), 'subset_top', 'infer_icd_report_data.json')
    os.makedirs(os.path.dirname(save_sample_path), exist_ok=True)
    with open(save_sample_path, 'w') as file:
        json.dump(sample_data, file, indent=4)

if __name__ == '__main__':
    # Usage: python extract_inference_subdataset_5.py subset
    input_type = 'subset'  # Default to 'subset'
    if input_type == 'subset':
        val_data_path = '/xxx/Multi-modal/MMCaD_code2/data/pair_multi_input_icd/datasets/split_by_patient_seed1/report_hpi/val_data_top_icd_report.json'
        test_data_path = '/xxx/Multi-modal/MMCaD_code2/data/pair_multi_input_icd/datasets/split_by_patient_seed1/report_hpi/test_data_top_icd_report.json'
        extract_subset_data(test_data_path, val_data_path)
