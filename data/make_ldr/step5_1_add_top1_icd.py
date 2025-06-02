import json
import os
import pickle


# 从中检索 icd 10 --> 9 的映射的新名称
import pandas as pd
from tqdm import tqdm
from icdmappings import Mapper


mapper = Mapper()

def get_first_icd(input_file, output_file):
    '''
    To get the first icd code for each patient
    '''
    base_path = '/xxx/'
    core_mimiciv_path = base_path + 'data/physionet.org/files/mimiciv/2.2/'
    icd_map_table = pd.read_csv(core_mimiciv_path + 'hosp/d_icd_diagnoses.csv.gz', compression='gzip') 

    dataset_base_path = '/xxx/Multi-modal/data/physionet.org/files/mimic_mmcad2'
    # Step 1: Read the test_data.json file
    # input_file = '/userhome/cs/u3010415/Multi-modal/MMCaD_code2/data/pair_multi_input_icd/datasets/split_by_patient_seed1/test_data.json'
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Number of cases in the data: {len(data)}")

    target_data = []
    # Step 2: For each case_id, read the corresponding icd_diagnosis.pkl file
    for entry in tqdm(data):
        case_id = entry["case_id"]
        icd_file_path = os.path.join(dataset_base_path, 'img/data', case_id, 'icd_diagnosis.pkl')
        if not os.path.exists(icd_file_path):
            icd_file_path = os.path.join(dataset_base_path, 'unimg/data', case_id, 'icd_diagnosis.pkl')
        
        if os.path.exists(icd_file_path):
            with open(icd_file_path, 'rb') as icd_file:
                icd_data = pickle.load(icd_file)
            
            # print(f"Processing case_id: {case_id}")
            # print(f"Top ICD code: {icd_data}")
            # Step 3: Add a new field to store the icd_code with seq_num 1
            top_icd_entry = icd_data.loc[icd_data['seq_num'] == 1].to_dict('records')[0] if not icd_data.loc[icd_data['seq_num'] == 1].empty else None
            # print(f"Top ICD code: {top_icd_entry}")

            if top_icd_entry:
                icd_code = top_icd_entry['icd_code']
                icd_version = top_icd_entry['icd_version']
                
                if icd_version == 10:
                    mapped_code = mapper.map([icd_code], source='icd10', target='icd9')[0]
                    icd_name_row = icd_map_table[icd_map_table['icd_code'] == mapped_code]
                    if not icd_name_row.empty:
                        icd_name = icd_name_row['long_title'].values[0]
                        # entry['top_1_icd'] = {'code': mapped_code, 'name': icd_name}
                    else:
                        icd_name = 'Unknown'
                        # entry['top_1_icd'] = {'code': mapped_code, 'name': 'Unknown'}
                    icd_code = mapped_code

                elif icd_version == 9:
                    icd_name = top_icd_entry['long_title']
                # 如果 code 在 entry[output_dict][output_icd_id] 里面，就加上
                if icd_code in entry['output_dict']['output_icd_id']:    
                    entry['top_1_icd'] = {'code': icd_code, 'name': icd_name}
                    target_data.append(entry)
                else:
                    continue

        else:
            raise ValueError(f"ICD file not found for case_id: {case_id}")

    # Step 4: Save the updated data to test_data_top_icd.json
    print(f"Number of cases with top_1_icd: {len(target_data)}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, ensure_ascii=False, indent=4)
    
# Example usage
base_path = '/xxx/Multi-modal/MMCaD_code2/'
input_file = base_path + 'data/pair_multi_input_icd/datasets/split_by_patient_seed1/test_data.json'
output_file = base_path + 'data/pair_multi_input_icd/datasets/split_by_patient_seed1/test_data_top_icd.json'

get_first_icd(input_file, output_file)

