import json
from tqdm import tqdm
import numpy as np
import time
import pickle
import re
from multiprocessing import Pool
import sys
import os
import pandas as pd
from collections import OrderedDict
from collections import Counter
import matplotlib.pyplot as plt

'''
    向 50 类疾病的 icd_name_path 里面添加 image report, physical, hpi 信息
'''
base_path = '/xxx/'
core_mimiciv_path = base_path + 'data/physionet.org/files/mimiciv/2.2/'


check_img_path = base_path + 'data/physionet.org/files/mimic_mmcad2/img/data'
check_unimg_path = base_path + 'data/physionet.org/files/mimic_mmcad2/unimg/data'


# 1. 加载 input_file
input_type = sys.argv[1]

if input_type == 'train':
    input_file = base_path + 'MMCaD_code2/data/pair_multi_input_icd/datasets/split_by_patient_seed1/train_data_top_icd.json'
    save_data_path = os.path.join(os.path.dirname(input_file), 'report_hpi/train_data_top_icd_report.json')
elif input_type == 'val':
    input_file = base_path + 'MMCaD_code2/data/pair_multi_input_icd/datasets/split_by_patient_seed1/val_data_top_icd.json'
    save_data_path = os.path.join(os.path.dirname(input_file), 'report_hpi/val_data_top_icd_report.json')
elif input_type == 'test':
    input_file = base_path + 'MMCaD_codes2/data/pair_multi_input_icd/datasets/split_by_patient_seed1/test_data_top_icd.json'
    save_data_path = os.path.join(os.path.dirname(input_file), 'report_hpi/test_data_top_icd_report.json')
elif input_type == 'analysis':
    input_file = base_path + 'MMCaD_code2/data/pair_multi_input_icd/datasets/split_by_patient_seed1/report_hpi/subset_top40_top1/infer_icd_report_data_top.json'


print(f'input_type:{input_type}')

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
    '''
    4.5K    ecg_info.csv
    29K     hosp_ed_cxr_data.csv
    33K     hosp_ed_cxr_data_split_note.csv: 切分 HPI,physical exam
    4.5K    icd_diagnosis.pkl 
    4.5K    labevents.csv
    4.5K    microbiologyevents.csv
    4.5K    radiology_report.csv: 切分 report
    '''
    check_list = ['hosp_ed_cxr_data_split_note.csv','radiology_report.csv']
    
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
    subject_id,hadm_id,stay_id = single_case.split('/')
    
    case_path = os.path.join(check_img_path, subject_id, hadm_id, stay_id)
    if not os.path.exists(case_path):
        case_path = os.path.join(check_unimg_path, subject_id, hadm_id, stay_id)
    
    case_dict = read_csv_pkl(case_path)
    
    return case_dict

input_sections = OrderedDict([
    ('History of Present illness', 'History of Present Illness'),
    ('Past Medical History:', 'Past Medical History:'),
    ('Social History:', 'Social History:'),
    ('Physical Exam', 'Physical Exam'),
    ('Pertinent Results:', 'Pertinent Results:'),
    ('Brief Hospital Course', 'Brief Hospital Course'),
    ('Medications on Admission', '[A-Za-z_]+ on Admission'),
    ('Discharge Medications', '[A-Za-z_]+ Medications'),
    ('Discharge Disposition', '[A-Za-z_]+ Disposition'),
    ('Discharge Diagnosis', '[A-Za-z_]+ Diagnosis'),
    ('Discharge Condition', '[A-Za-z_]+ Condition'),
    
])

def extract_physical_exam(discharge_summary):
    '''
    不仅要匹配出来，还得只保留  on admission 之后 on discharge 之前的内容
    '''
    target_text = re.findall(r'Physical Exam:\n(.*?)Pertinent Results:', discharge_summary, re.DOTALL)
    target_text = target_text[0].strip() if len(target_text) > 0 else 'None'
    
    # 保留 on admission 之后 discharge 之前的内容
    lower_text = target_text.lower()
    discharge_index = lower_text.find('discharge')
    if discharge_index != -1:
        target_text = target_text[:discharge_index]

    return target_text

def extract_from_discharge_note(hosp_ed_cxr,extract_type):
    discharge_summary = hosp_ed_cxr['discharge_note_text'][0]

    if extract_type == 'HPI':
        target_text = re.findall(r'History of Present Illness:\n(.*?)Past Medical History:', discharge_summary, re.DOTALL)
        target_text = target_text[0].strip() if len(target_text) > 0 else 'None'
        # target_text = 'The history of present illness : ' + target_text
    elif extract_type == 'Physical':
        target_text = extract_physical_exam(discharge_summary)
        # target_text = 'The physical exam on admission : ' + target_text

    return target_text

def extract_report_list(report_df):
    report_list = []
    def remove_findings_impression(df_text):
        # Define the pattern to match FINDINGS: and IMPRESSION: sections
        pattern = r'FINDINGS:.*?(?=IMPRESSION:)' # \n\nFINDINGS:  C
        # Use re.sub to remove the matched sections
        cleaned_text = re.sub(pattern, '', df_text, flags=re.DOTALL)
        return cleaned_text

    for idx,row in report_df.iterrows():
        single_dict = {}
        row_text = row['text']
        # 把 findings 去掉得了
        row_text = remove_findings_impression(row_text)
        report_list.append(row_text)
    
    return report_list
        



def add_more_data(case_dict,case):
    '''
    1. image report — impression (最好能分类存放 —  CXR,CT,**Ultrasound..**)
    2. physical exam — on admission
    3. History of present illness —  做对比实验用
    '''
    result_dict = case

    # raise ValueError('stop')
    # ----------------- 1.1 Process HPI ----------------- #
    hpi_text = extract_from_discharge_note(case_dict['hosp_ed_cxr'], 'HPI')
    hpi_text = 'The history of present illness : ' + hpi_text
    # print(f"hpi_text:{hpi_text}")
    if hpi_text == 'None':
        print(f"the case_dict:{case_dict['case_id']} has no hpi_text")
        result_dict['input_dict']['hpi_text'] = 'None'
    else:
        result_dict['input_dict']['hpi_text'] = hpi_text

    # ----------------- 1.2 Process Physical ----------------- #
    physical_text = extract_from_discharge_note(case_dict['hosp_ed_cxr'], 'Physical')
    physical_text = 'The physical exam on admission : ' + physical_text
    
    # print(f"physical_text:{physical_text}")
    if physical_text == 'None':
        print(f"the case_dict:{case_dict['case_id']} has no physical_text")
        result_dict['input_dict']['physical_text'] = 'None'
    else:
        result_dict['input_dict']['physical_text'] = physical_text
    
    # ----------------- 2 Process Report ----------------- #
    if isinstance(case_dict['radio_report'],int) and case_dict['radio_report'] == 0:
        print(f"the case_dict:{case_dict['case_id']} has no report")
        result_dict['report'] = []
    else:
        report_df = case_dict['radio_report']
        # print('len of report_df',len(report_df))
        # 需要按照时间筛选
        hosp_ed_cxr = case_dict['hosp_ed_cxr'].iloc[0]
        admittime = pd.Timestamp(hosp_ed_cxr['admittime'])
        dischtime = pd.Timestamp(hosp_ed_cxr['discharge_time']) 
        report_df['charttime'] = pd.to_datetime(report_df['charttime'])
        report_df = report_df[(report_df['charttime'] >= admittime) & (report_df['charttime'] <= dischtime)]
        # print('len of valid report_df',len(report_df))

        report_list = extract_report_list(report_df)
        result_dict['input_dict']['report'] = report_list

    return result_dict


def process_part(part_process_data):
    json_list = []

    for case in tqdm(part_process_data):
        # print(f'case:{case}')
        case_id = case['case_id']
        # idx = '/'.join(case_id.split(' '))
        case_result = process_single(case_id)

        single_dict = add_more_data(case_result,case)
        if isinstance(single_dict, int):
            continue
    
        json_list.append(single_dict)
    
    return json_list

def main(process_list):
    num_processes = 10
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
    merged_first = [item for sublist in result for item in sublist]
    with open(save_data_path, 'w') as file:
        json.dump(merged_first, file, indent=4)

    # Check the report length distribution
    report_lengths = [len(case['input_dict']['report']) for case in merged_first]
    length_distribution = dict(sorted(Counter(report_lengths).items()))
    print("Report length distribution:", length_distribution)

def plot_distribution(top_1_icd_distribution,save_base,img_type):
    plt.figure(figsize=(14, 6))
    plt.bar(top_1_icd_distribution.keys(), top_1_icd_distribution.values())
    plt.xticks(rotation=45)
    plt.xlabel(f'{img_type} ICD code')
    plt.ylabel('Frequency')
    plt.title(f'{input_type}: {img_type} ICD code distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(save_base, f'{input_type}_{img_type}_icd_distribution.png'))

def analysis_data(input_file):
    save_base = os.path.dirname(input_file)

    with open(input_file, 'r', encoding='utf-8') as f:
        process_data = json.load(f)

    print(f"len of process_data:{len(process_data)}")

    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # 1. 遍历数据统计 input_dict 中元素的 token 长度
    token_lengths = []
    data_dict = {key: [] for key in process_data[0]['input_dict'].keys()}
    data_dict['total_length'] = []
    data_dict['case_id'] = []
    data_dict['report_length'] = []
    data_dict['icd_num'] = []
    for case in tqdm(process_data):
        input_dict = case['input_dict']
        total_length = 0
        report_length = len(input_dict['report']) if 'report' in input_dict else 0
        for key, value in input_dict.items():
            if isinstance(value, list):
                value = ' '.join(value)
                
            tokens = tokenizer.encode(value)
            token_length = len(tokens)
            data_dict[key].append(token_length)
            total_length += token_length
        
        data_dict['total_length'].append(total_length)
        data_dict['case_id'].append(case['case_id'])
        data_dict['report_length'].append(report_length)
        data_dict['icd_num'].append(len(case['output_dict']['output_icd_id']))
        token_lengths.append(total_length)
    
    df = pd.DataFrame(data_dict)
    df.set_index('case_id', inplace=True)
    print(df.head())

    # 打印 token 长度的分布
    print('df.describe:\n',df.describe())
    # Save the description of the dataframe to a CSV file
    os.makedirs(os.path.join(save_base, 'analysis_top1'), exist_ok=True)
    
    df.describe().round(1).to_csv(os.path.join(save_base, 'analysis_top1', f'{input_type}_token_length_description.csv'))

    # 2. 统计 top_1_icd[code] 的分布
    top_1_icd_codes = [case['top_1_icd']['code'] for case in process_data]
    top_1_icd_distribution = dict(sorted(Counter(top_1_icd_codes).items(), key=lambda item: item[1], reverse=True))
    print(f"Top 1 ICD code distribution: {top_1_icd_distribution}")
    print(f"len of top_1_icd_distribution:{len(top_1_icd_distribution)}")

    # 将 top_1_icd_distribution 绘制成图
    plot_distribution(top_1_icd_distribution,os.path.join(save_base, 'analysis_top1'),'top_1')

    # 3. 统计 output_dict[output_icd_id] 的分布
    output_icd_ids = [icd_id for case in process_data for icd_id in case['output_dict']['output_icd_id']]
    output_icd_distribution = dict(sorted(Counter(output_icd_ids).items(), key=lambda item: item[1], reverse=True))
    print(f"Output ICD code distribution: {output_icd_distribution}")
    print(f"len of output_icd_distribution:{len(output_icd_distribution)}")
    plot_distribution(output_icd_distribution,os.path.join(save_base, 'analysis_top1'),'overall')

    # 用 output_icd_distribution 的key 减去 top_1_icd_distribution 的key，看看后者少了哪些
    lack_icd_codes = set(output_icd_distribution.keys()) - set(top_1_icd_distribution.keys())
    print(f"lack_icd_codes:{lack_icd_codes}")



if __name__ == '__main__':
    # input_type == 'analysis' # 'main'
    # *** Function 1: main ***
    if input_type in ['train', 'val', 'test']:
        main(process_data)
    # *** Function 2: analysis_data ***
    elif input_type == 'analysis':
        analysis_data(input_file)
        # analysis_data(save_data_path)
        # analysis_data(input_file)