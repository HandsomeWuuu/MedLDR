'''
    Extract the top n  ICD diagnoses as labels and match the corresponding lab data.
'''
import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np
import time
import pickle
import re
from multiprocessing import Pool
import sys
from icdmappings import Mapper

mapper = Mapper()

base_path = '/xxxx/Multi-modal/'
core_mimiciv_path = base_path + 'data/physionet.org/files/mimiciv/2.2/'


check_img_path = base_path + 'data/physionet.org/files/mimic_mmcad2/img/data'
check_unimg_path = base_path + 'data/physionet.org/files/mimic_mmcad2/unimg/data'

# 将终端输入的第一个参数赋值给data_type
data_type = sys.argv[1]

# base_dir = f'count_data/{data_version}'
base_dir = f'data/pair_lab_and_icd_v2'


if not os.path.exists(base_dir):
    os.makedirs(base_dir)

if data_type == 'img_img':
    process_txt = base_path + 'MMCaD_code2/split_data/img_img_valid.txt'
    sava_data_path= f'{base_dir}/img_img/img_valid_pair.json'
    sava_lab_path = f'{base_dir}/img_img/img_valid_lab.csv'
elif data_type == 'img_unimg':
    process_txt = base_path + 'MMCaD_code2/split_data/img_unimg_valid.txt'
    sava_data_path= f'{base_dir}/img_unimg/img_unimg_valid_data_pair.json'
    sava_lab_path = f'{base_dir}/img_unimg/img_unimg_valid_lab.csv'
elif data_type == 'unimg':
    process_txt = base_path + 'MMCaD_code2/split_data/unimg_valid.txt'
    sava_data_path= f'{base_dir}/unimg/unimg_valid_data_pair.json'
    sava_lab_path = f'{base_dir}/unimg/unimg_valid_lab.csv'

if not os.path.exists(os.path.dirname(sava_data_path)):
    os.makedirs(os.path.dirname(sava_data_path))

with open(process_txt, "r") as file:
    content = file.readlines()
    process_data = [line.strip() for line in content]

print(f"Type of data being processed: {data_type}")
print("Length of the loaded content list:")
print(len(process_data))
print("Example from the loaded content list:")

# Retrieve new names from ICD-10 to ICD-9 mapping
icd_map_table = pd.read_csv(core_mimiciv_path + 'hosp/d_icd_diagnoses.csv.gz', compression='gzip') 
# print(icd_map_table.head())
# print(icd_map_table[icd_map_table['icd_code']== '5854'])


def process_csv(file_path):
    if os.stat(file_path).st_size <=1:
        return 0
    else:
        df = pd.read_csv(file_path)
        return df
    

def process_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        df = pd.DataFrame(data)
    
    return df
    
    
def read_csv_pkl(c_path):
    check_list = ['labevents.csv','hosp_ed_cxr_data_split_note.csv','icd_diagnosis.pkl']
    # check_list = ['labevents.csv','icd_diagnosis.pkl']
    
    one_row ={}
    
    for file_name in check_list:
        # print(f'check :{file_name}')
        file_path = os.path.join(c_path, file_name)
    
        if 'hosp_ed_cxr' in file_name:
            one_row['hosp_ed_cxr'] = process_csv(file_path)
        elif 'icd_diagnosis' in file_name:
            one_row['icd_diagnosis'] = process_pkl(file_path)
        elif 'microbio' in file_name:
            one_row['microbio'] = process_csv(file_path)
        elif 'labevents' in file_name:
            one_row['lab'] = process_csv(file_path)
        elif 'radiology_report' in file_name:
             one_row['radio_report'] = process_csv(file_path)
    
    return one_row

def process_single(single_case):
    subject_id,hadm_id,stay_id = single_case.split(' ')

    if  data_type == 'img_img' or data_type == 'img_unimg':
        case_path = os.path.join(check_img_path,subject_id,hadm_id,stay_id)
    elif data_type == 'unimg':
        case_path = os.path.join(check_unimg_path,subject_id,hadm_id,stay_id)
    
    case_dict = read_csv_pkl(case_path)
    
    return case_dict
    


def match_note_diagnosis(df_text):
    # Define the pattern to match Discharge Diagnosis: and Discharge Condition: sections
    pattern = r'Discharge Diagnosis:(.*?)(?=Discharge Condition:)'
    
    # Find all matches
    matches = re.findall(pattern, df_text, flags=re.DOTALL)
    # print(f'matches 1:{matches}')
    # Extract the disease information
    matches = [match.strip() for match in matches]
    matched_text = ', '.join(matches)
    # print(f'matches 2:{matched_text}')
   
    return matched_text

def get_top_n_icd(icd_table,n=3):
    # Keep rows where seq_num is less than or equal to n
    top_icd_table = icd_table[icd_table['seq_num'] <= n]

    icd_table_9 = top_icd_table[top_icd_table['icd_version'] == 9]

    icd_codes_9 = list(zip(icd_table_9['icd_code'].tolist(), icd_table_9['long_title'].tolist()))

    # Select rows where icd_version == 10
    icd_table_10 = top_icd_table[top_icd_table['icd_version'] == 10]
    if not icd_table_10.empty:
        icd_codes_10 = icd_table_10['icd_code'].tolist()
        conver_10_to_9_list = mapper.map(icd_codes_10, source='icd10', target='icd9')

        # Find the corresponding name from icd_map_table
        for code in conver_10_to_9_list:
            try:
                new_name = icd_map_table[icd_map_table['icd_code'] == code]['long_title'].values[0]
                icd_codes_9.append((code, new_name))
            except IndexError:
                print(f'code: {code} not in icd_map_table, icd_codes_10:{icd_codes_10}, conver_10_to_9_list:{conver_10_to_9_list}')
            

    return icd_codes_9
    


# Get Demographic Prompt
def match_lab_and_icd(case_result):
    # This version is intended to count diseases with icd diagnosis seq_num < n
    result_dict = {}
    hosp_ed_cxr = case_result['hosp_ed_cxr'].iloc[0]

    # 1. Get ICD Procedure
    icd_table = case_result['icd_diagnosis']
    icd_codes = get_top_n_icd(icd_table)

    # If this is 0, it means no match was found -- if no match is found, there is no need to count it
    if len(icd_codes) == 0:
        return 0,0
    
    # result_dict.update({'icd_match_note': note_icd_match})
    result_dict.update({'icd_codes': icd_codes})
    
    # Get Lab Prompt
    # if isinstance(case_result['lab'], pd.core.frame.DataFrame):
    # This cleaning is still not enough, some data is still not clean:
    # - Filter by timeline
    # - Remove data where item is 'Delete'
    # - Remove data with empty value

    lab = case_result['lab']
    # print(f'lab:{ len(lab)}')
    # Admission and discharge times from hosp_ed_cxr
    admittime = pd.Timestamp(hosp_ed_cxr['admittime'])
    dischtime = pd.Timestamp(hosp_ed_cxr['discharge_time']) 

    lab['charttime'] = pd.to_datetime(lab['charttime'])
    lab = lab[(lab['charttime'] >= admittime) & (lab['charttime'] <= dischtime)]
    # print(f'after time filter lab:{ len(lab)}')

    lab = lab[lab['label'] != 'Delete']
    # print(f'after Delete filter lab:{ len(lab)}')

    lab[['value', 'ref_range_lower', 'ref_range_upper']] = lab[['value', 'ref_range_lower', 'ref_range_upper']].apply(pd.to_numeric, errors='coerce')
    lab = lab.dropna(subset=['value','ref_range_lower','ref_range_upper'])
    # print(f'after Nan filter lab:{ len(lab)}')

    # lab = lab[(lab['value'] < 10000) & (lab['ref_range_lower'] < 10000) & (lab['ref_range_upper'] < 10000)]
    # print(f'after value filter lab:{ len(lab)}')
    # Count the lab items
    # Pack itemid and label into a tuple
    lab_items = list(zip(lab['itemid'].tolist(), lab['label'].tolist()))

    if len(lab_items) == 0:
        return 0,0  
    
    result_dict.update({'lab_items': lab_items})

    return result_dict,lab

def process_part(part_process_data):
    json_list = []
    valid_lab_df = pd.DataFrame()

    for case in tqdm(part_process_data):
        case_result = process_single(case)
        # Skip cases without hosp_ed_cxr data -- invalid data
        if case_result['hosp_ed_cxr'].empty:
            continue
        # If the lab data is empty, skip directly
        if isinstance(case_result['lab'], int) or case_result['lab'].empty:
            continue


        single_dict,valid_lab = match_lab_and_icd(case_result)

        if isinstance(single_dict, int):
            continue

        json_list.append(single_dict)
        valid_lab_df = pd.concat([valid_lab_df,valid_lab],axis=0)
    
    
    return json_list,valid_lab_df

def main(process_list):
    num_processes = 20
    chunk_size = len(process_list) // num_processes

    with Pool(processes=num_processes) as pool:
        ranges = []
        for i in range(num_processes):
            if i == num_processes - 1:
                ranges.append(process_list[i * chunk_size: len(process_list)])
            else:
                ranges.append(process_list[i * chunk_size: (i + 1) * chunk_size])

        print(len(ranges),len(ranges[0]))
        result = pool.map(process_part, ranges)
    
    ## Part 1: Save json format data
    merged_first = [item for sublist, _ in result for item in sublist]
    with open(sava_data_path, 'w') as file:
        json.dump(merged_first, file, indent=4)

    print(f'valid count:{len(merged_first)}/ {len(process_list)}')
    ## Part 2: Save count data
    merged_second = pd.concat([sublist for _, sublist in result], axis=0)
    df_second = merged_second.reset_index(drop=True)
    df_second.to_csv(sava_lab_path)

    print(f'valid_lab:{len(df_second)}')
 


main(process_data)

    






