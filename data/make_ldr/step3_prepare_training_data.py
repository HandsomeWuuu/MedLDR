'''
    和 pair_lab_icd_v2.py 一样，这个脚本也是用于构建 lab 和 icd code 的对应关系，但是这个脚本是用于训练的，所以需要实际的 lab 数据

    只筛选 data/pair_lab_and_icd_v2/lab_icd_pair_top_50.csv 中的 icd code，然后将这些 icd code 与 lab 数据对应起来，最后保存到 data/pair_lab_and_icd_v2 目录下

    遍历 img_img, img_unimg, unimg 三种数据类型，分别处理对应的 lab 数据和 icd code 数据
    输入：
        - data_type: img_img, img_unimg, unimg
        - lab_icd_pair_top_{top_num}.csv 
    输出：
        - data/pair_lab_and_icd_v2/datasets/top50_img/img_pairs.json
        - data/pair_lab_and_icd_v2/datasets/top50_img/img_lab.csv
        - data/pair_lab_and_icd_v2/datasets/top50_img_unimg/img_unimg_pairs.json
        - data/pair_lab_and_icd_v2/datasets/top50_img_unimg/img_unimg_lab.csv
        - data/pair_lab_and_icd_v2/datasets/top50_unimg/unimg_pairs.json
        - data/pair_lab_and_icd_v2/datasets/top50_unimg/unimg_lab.csv
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
# from collections import OrderedDict

mapper = Mapper()


base_path = '/userhome/cs/u3010415/Multi-modal/'
core_mimiciv_path = base_path + 'data/physionet.org/files/mimiciv/2.2/'


check_img_path = base_path + 'data/physionet.org/files/mimic_mmcad2/img/data'
check_unimg_path = base_path + 'data/physionet.org/files/mimic_mmcad2/unimg/data'

# 将终端输入的第一个参数赋值给data_type
data_type = sys.argv[1]

# base_dir = f'count_data/{data_version}'
base_dir = f'data/pair_multi_input_icd/datasets'

# 要筛选的目标 lab 数据
top_num = 50
freq_df = pd.read_csv(f'{base_path}/MMCaD_code2/data/pair_lab_and_icd_v2/lab_icd_pair_top_{top_num}.csv', dtype={'Code': str})
print(f"freq_df len: {len(freq_df)}")

# target_codes = set(freq_df['Code'].astype(str)) # 这种方式会让0开头的字符被去掉
target_codes = set(freq_df['Code'])
print(f"target_codes len: {len(target_codes)}")

if not os.path.exists(base_dir):
    print(f"create dir:{base_dir}")
    os.makedirs(base_dir)

if data_type == 'img_img':
    process_txt = base_path + 'MMCaD_code2/split_data/img_img_valid.txt'
    save_data_path= f'{base_dir}/top50_img/img_pairs.json'
    save_lab_path = f'{base_dir}/top50_img/img_lab.csv'
elif data_type == 'img_unimg':
    process_txt = base_path + 'MMCaD_code2/split_data/img_unimg_valid.txt'
    save_data_path= f'{base_dir}/top50_img_unimg/img_unimg_pairs.json'
    save_lab_path = f'{base_dir}/top50_img_unimg/img_unimg_lab.csv'
    
elif data_type == 'unimg':
    process_txt = base_path + 'MMCaD_code2/split_data/unimg_valid.txt'
    save_data_path= f'{base_dir}/top50_unimg/unimg_pairs.json'
    save_lab_path = f'{base_dir}/top50_unimg/unimg_lab.csv'


if not os.path.exists(os.path.dirname(save_data_path)):
    os.makedirs(os.path.dirname(save_data_path))

with open(process_txt, "r") as file:
    content = file.readlines()
    # 去除每行末尾的换行符
    process_data = [line.strip() for line in content]

print(f"处理的数据类型:{data_type}")

print("读取的内容列表长度:")
print(len(process_data))
print("读取的内容列表案例:")

# 从中检索 icd 10 --> 9 的映射的新名称
icd_map_table = pd.read_csv(core_mimiciv_path + 'hosp/d_icd_diagnoses.csv.gz', compression='gzip') 

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
    check_list = ['labevents.csv','hosp_ed_cxr_data_split_note.csv','icd_diagnosis.pkl','microbiologyevents.csv']
    
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
    
    # case_type = os.path.basename(process_txt).split('_')[0]

    if  data_type == 'img_img' or data_type == 'img_unimg':
        case_path = os.path.join(check_img_path,subject_id,hadm_id,stay_id)
    elif data_type == 'unimg':
        case_path = os.path.join(check_unimg_path,subject_id,hadm_id,stay_id)
    
    case_dict = read_csv_pkl(case_path)
    
    return case_dict

def get_top_n_icd(icd_table,n=3):
    # 保留 seq_num 小于等于 n 的数据
    top_icd_table = icd_table[icd_table['seq_num'] <= n]

    icd_table_9 = top_icd_table[top_icd_table['icd_version'] == 9]

    icd_codes_9 = list(zip(icd_table_9['icd_code'].tolist(), icd_table_9['long_title'].tolist()))

    # 将 icd_version==10 的数据表挑出来
    icd_table_10 = top_icd_table[top_icd_table['icd_version'] == 10]
    if not icd_table_10.empty:
        icd_codes_10 = icd_table_10['icd_code'].tolist()
        # print(f'icd_codes_10:{icd_codes_10}')
        conver_10_to_9_list = mapper.map(icd_codes_10, source='icd10', target='icd9')

        # print(f'conver_10_to_9_list:{conver_10_to_9_list}')

        # 从 icd_map_table 中找到对应的名称
        for code in conver_10_to_9_list:
            try:
                new_name = icd_map_table[icd_map_table['icd_code'] == code]['long_title'].values[0]
                icd_codes_9.append((code, new_name))
            except IndexError:
                print(f'code: {code} not in icd_map_table, icd_codes_10:{icd_codes_10}, conver_10_to_9_list:{conver_10_to_9_list}')
                # raise ValueError(f'code: {code} not in icd_map_table')

        # icd_codes_9.extend(conver_10_to_9_list)
    
    return icd_codes_9
    
def generate_lab_table(df_lines):

    base_pt = f"labevents name,measurement fluid,measurement category,value,valueuom,lower_bound,upper_bound\n" #flag,comments flag 会标识出是否正常
    for _, df_line in df_lines.iterrows(): 
        base_pt += f"{df_line['label']},{df_line['fluid']},{df_line['category']},{df_line['value']},{df_line['valueuom']},{df_line['ref_range_lower']},{df_line['ref_range_upper']}\n" #{df_line['flag']},{df_line['comments']}
    return base_pt

def generate_ed_table(hosp_ed_cxr):
    ed_base_pt = "gender,age approximation,temperature,heartrate,resprate,oxygen saturation,systolic blood pressure,diastolic blood pressure,pain level\n"
    info_pt = f"{hosp_ed_cxr['gender']},{hosp_ed_cxr['anchor_age']},{hosp_ed_cxr['ed_temperature']},{hosp_ed_cxr['ed_heartrate']},{hosp_ed_cxr['ed_resprate']},{hosp_ed_cxr['ed_o2sat']},{hosp_ed_cxr['ed_sbp']},{hosp_ed_cxr['ed_dbp']},{hosp_ed_cxr['ed_pain']}\n"
    return ed_base_pt + info_pt

def get_text_info(hosp_ed_cxr,ed_word):
        if ed_word == 'family_history':
            ed_text = 'The family history: '
        elif ed_word == 'ed_chiefcomplaint':
            # 所以 chief complaint 用的是 ed 表中的
            ed_text = 'The chief complaint: '
        
        try:
            if isinstance( hosp_ed_cxr[ed_word], str):
                ed_text += hosp_ed_cxr[ed_word]
            else:
                raise TypeError
        except (KeyError, TypeError):
            print(f"KeyError or TypeError: {hosp_ed_cxr[ed_word]}")
            ed_text += 'None'
        
        return ed_text
    
def get_past_medical_history(hosp_ed_cxr):
    medical_history = re.findall(r'Past Medical History:\n(.*?)Social History:', hosp_ed_cxr['discharge_note_text'], re.DOTALL)
    medical_history = medical_history[0].strip() if len(medical_history) > 0 else 'None'
    medical_history = 'The past medical history: ' + medical_history

    return medical_history

sense_interpretation = {"S": 'sensitive', "I": 'intermediate', "R": 'resistant',"P":'pending',"NaN": "NaN"}
def generate_micro_table(df_lines):
    if len(df_lines) > 15:
        df_lines = df_lines[:15]

    # 没有做数据裁剪
    df_lines = df_lines.astype(str).replace('nan', 'NaN')
    base_pt = f"test name,used specimen,grew organism,test antibiotic,antibiotic sensitivity,Comments\n"
    for _, df_line in df_lines.iterrows():
        base_pt += f"{df_line['test_name']},{df_line['spec_type_desc']},{df_line['org_name']},{df_line['ab_name']},{sense_interpretation[df_line['interpretation']]},{df_line['comments']}\n"
    return base_pt

def match_lab_and_icd(case_result):
    # 统计 icd 诊断的 seq_num < n 的疾病
    # 只筛选 target_codes 中的 icd code --  直接构造成数据集
    '''
        result_dict: {
            'input_dict': 
                    {'lab': str,
                     'family_history': str,
                     'ed_chiefcomplaint': str,
                     'ed_info': str,
                     'image': str,},
            'output_dict': 
                    {'output_icd_id':[icd_id,],
                     'output_icd_name':[icd_name,],
                     'output_icd_id_and_name':[icd_id:icd_name,]}
        }
    '''

    result_dict = {}
    input_dict = {}
    output_dict = {}

    hosp_ed_cxr = case_result['hosp_ed_cxr'].iloc[0]

    # ----------- 1 Get Output: ICD code and name ------------ #
    icd_table = case_result['icd_diagnosis']
    icd_codes = get_top_n_icd(icd_table)

    #  这里是 0的话，说明没有匹配到 -- 没有匹配到的话，就不用统计了
    if len(icd_codes) == 0:
        return 0,0
    
    # 判断和筛选 icd_codes 在 target_codes 中的数据
    valid_icd_codes = [code for code in icd_codes if code[0] in target_codes]

    if len(valid_icd_codes) == 0:
        return 0,0
    
    # 整理数据
    output_dict.update({'output_icd_id': [str(code[0]) for code in valid_icd_codes]})
    output_dict.update({'output_icd_name': [str(code[1]) for code in valid_icd_codes]})
    output_dict.update({'output_icd_id_and_name': [f"{code[0]}:{code[1]}" for code in valid_icd_codes]})
    result_dict.update({'output_dict': output_dict})

    # ------------- 2 Get Input: Lab and other information ------------ #
    # input_text = 'Please review the following lab test list to identify the potential disease the patient may have.\n'

    # ----------------- 2.1 Process Lab ----------------- #
    lab = case_result['lab']

    # 从 hosp_ed_cxr 入院出院的时间
    admittime = pd.Timestamp(hosp_ed_cxr['admittime'])
    dischtime = pd.Timestamp(hosp_ed_cxr['discharge_time']) 
    lab['charttime'] = pd.to_datetime(lab['charttime'])
    lab = lab[(lab['charttime'] >= admittime) & (lab['charttime'] <= dischtime)]
    lab = lab[lab['label'] != 'Delete']
    lab[['value', 'ref_range_lower', 'ref_range_upper']] = lab[['value', 'ref_range_lower', 'ref_range_upper']].apply(pd.to_numeric, errors='coerce')
    lab = lab.dropna(subset=['value','ref_range_lower','ref_range_upper'])

    if len(lab) == 0:
        return 0,0

    lab_text = 'The following lab tests are available for the patient:\n'
    lab_text = lab_text + generate_lab_table(lab)
    lab_text = lab_text.replace('Beta-2', 'Beta-two') # 把数值中的 Beta-2 替换成 Beta-two, 避免被当成 提取出负数
    input_dict.update({'lab': lab_text})


    # ----------------- 2.2 Process Family History ----------------- #
    family_history_text = get_text_info(hosp_ed_cxr,'family_history')
    input_dict.update({'family_history': family_history_text})

    # ----------------- 2.3 Process Chief Complaint ----------------- #
    ed_chiefcomplaint_text = get_text_info(hosp_ed_cxr,'ed_chiefcomplaint')
    input_dict.update({'ed_chiefcomplaint': ed_chiefcomplaint_text})

    # ----------------- 2.4 Process Past Medical History ----------------- #
    past_medical_history = get_past_medical_history(hosp_ed_cxr)
    input_dict.update({'past_medical_history': past_medical_history})

    # ----------------- 2.5 Process ED Info ----------------- #
    ed_info_text = 'The ED info:\n' + generate_ed_table(hosp_ed_cxr)
    input_dict.update({'ed_info': ed_info_text})

    # ----------------- 2.6 Process Micro Info ----------------- #
    if isinstance(case_result['microbio'], pd.core.frame.DataFrame):
        micro_prompt = generate_micro_table(case_result['microbio'])
        micro_prompt = 'The microbiology tests for the patient:\n' + micro_prompt
    else:
        micro_prompt = 'The microbiology tests for the patient: None'
    
    # print(f'micro_prompt:{micro_prompt}')
    input_dict.update({'micro_info': micro_prompt})
    # ----------------- 2.7 Process Image ----------------- #
    # pass -- 暂时不需要处理 image
    # raise ValueError('暂时不需要处理 image')
    result_dict.update({'input_dict': input_dict})

    return result_dict,lab

def process_part(part_process_data):
    json_list = []
    lab_df = pd.DataFrame() 

    for case in tqdm(part_process_data):
        print(f'case:{case}')
        idx = '/'.join(case.split(' '))
        case_result = process_single(case)

        # 有些没有hosp_ed_cxr的数据，直接跳过 -- 无效数据
        if case_result['hosp_ed_cxr'].empty:
            continue
        # 如果 lab 数据为空，直接跳过
        if isinstance(case_result['lab'], int) or case_result['lab'].empty:
            continue

        single_dict,single_lab = match_lab_and_icd(case_result)
        if isinstance(single_dict, int):
            continue
        
        single_dict.update({'case_id': idx})
        json_list.append(single_dict)
        lab_df = pd.concat([lab_df,single_lab],axis=0)

    
    return json_list,lab_df

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
    merged_first = [item for sublist in result for item in sublist[0]]
    with open(save_data_path, 'w') as file:
        json.dump(merged_first, file, indent=4)

    print(f'valid count:{len(merged_first)}/ {len(process_list)}')


main(process_data)
