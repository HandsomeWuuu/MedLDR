import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np
import time
import pickle

import multiprocessing as mp
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

from util2 import Patient_MIMIC

base_path = '/xxx/Multi-modal/data'

core_mimiciv_path = base_path + '/physionet.org/files/mimiciv/2.2/'
core_mimicED_path = base_path + '/physionet.org/files/mimic-iv-ed/2.2/'
core_mimicNote_path = base_path + '/physionet.org/files/mimic-iv-note/2.2/'
core_mimiciv_imgcxr_path = base_path + '/physionet.org/files/mimic-cxr-jpg/2.0.0/'

# TODO edit path here
mimic_MMCaD_path = base_path + '/physionet.org/files/mimic_mmcad2/'
process_type = 'unimg' # img or unimg

if process_type == 'img':
    save_dir = base_path + '/physionet.org/files/mimic_mmcad2/img/data/'
    MMCaD_patient_path = os.path.join(mimic_MMCaD_path, 'img')
else:
    save_dir = base_path + '/physionet.org/files/mimic_mmcad2/unimg/data/'
    MMCaD_patient_path = os.path.join(mimic_MMCaD_path, 'unimg')

os.makedirs(save_dir, exist_ok=True)



"""
MATCH MICROBIOLOGYEVENTS
"""
def match_micro(valid_idxes_list):
    for idx in tqdm(valid_idxes_list):
        cur_idx = int(idx)
        filename = f"{cur_idx:08d}" + '.pkl'
        with open(MMCaD_patient_path + '/patient_object/' + filename, 'rb') as input:
            patient = pickle.load(input)


        subject_id = patient.core['subject_id'].values[0]
        hadm_id = patient.core['hadm_id'].values[0]
        if not np.isnan(patient.core['stay_id'].values[0]):
            stay_id = int(patient.core['stay_id'].values[0])
        else:
            stay_id = 0

        if not os.path.exists(os.path.join(save_dir, str(subject_id))):
            os.mkdir(os.path.join(save_dir, str(subject_id)))

        if not os.path.exists(os.path.join(save_dir, str(subject_id), str(hadm_id))):
            os.mkdir(os.path.join(save_dir, str(subject_id), str(hadm_id)))

        if not os.path.exists(os.path.join(save_dir, str(subject_id), str(hadm_id), str(stay_id))):
            os.mkdir(os.path.join(save_dir, str(subject_id), str(hadm_id), str(stay_id)))

        fname_micro = os.path.join(save_dir, str(subject_id), str(hadm_id), str(stay_id), 'microbiologyevents.csv')
        fname_hosp_ed_cxr = os.path.join(save_dir, str(subject_id), str(hadm_id), str(stay_id), 'hosp_ed_cxr_data.csv')


        current_microbiologyevents = df_microbiologyevents[(df_microbiologyevents['subject_id']== subject_id) & (df_microbiologyevents['hadm_id']== hadm_id)]
        current_microbiologyevents = current_microbiologyevents[(current_microbiologyevents['charttime'] >= pd.Timestamp(patient.core['admittime'].values[0])) & (current_microbiologyevents['charttime'] <= pd.Timestamp(patient.discharge.charttime.values[0]))]
        unique_micro_test_names = list(current_microbiologyevents.test_name.unique())
        unique_micro_test_id = list(current_microbiologyevents.test_itemid.unique())

        micro_fusion=pd.DataFrame()
        for test in unique_micro_test_names:
            first_test = current_microbiologyevents[current_microbiologyevents['test_name']==test]
            # drop all rows with 'canceled' or '___'
            first_test.org_name.fillna('',inplace=True)
            first_test.comments.fillna('', inplace=True)
            first_test['org_name'] = list(first_test['org_name'].str.lower())
            first_test['comments'] = list(first_test['comments'].str.lower())
            first_test = first_test[~first_test['org_name'].str.contains('cancel')]
            first_test = first_test[~first_test['comments'].str.contains('cancel|___')]

            if first_test.ab_itemid.isnull().values.sum()>0 or first_test.org_name.isnull().values.sum()>0:
                first_test = first_test.sort_values(by=['charttime'])[:1]
            else:
                first_test = first_test.sort_values(by=['charttime'])

            micro_fusion = pd.concat([micro_fusion, first_test], sort=False)


        micro_fusion.to_csv(fname_micro, index=False)


"""
MATCH LABEVENTS
"""
def match_labevents_all(valid_idxes_list):

    for idx in tqdm(valid_idxes_list):
        cur_idx = int(idx)

        filename = f"{cur_idx:08d}" + '.pkl'
        with open(MMCaD_patient_path + '/patient_object/' + filename, 'rb') as input:
            patient = pickle.load(input)

        subject_id = patient.core['subject_id'].values[0]
        hadm_id = patient.core['hadm_id'].values[0]
        if not np.isnan(patient.core['stay_id'].values[0]):
            stay_id = int(patient.core['stay_id'].values[0])
        else:
            stay_id = 0

        if not os.path.exists(os.path.join(save_dir, str(subject_id))):
            os.mkdir(os.path.join(save_dir, str(subject_id)))

        if not os.path.exists(os.path.join(save_dir, str(subject_id), str(hadm_id))):
            os.mkdir(os.path.join(save_dir, str(subject_id), str(hadm_id)))

        if not os.path.exists(os.path.join(save_dir, str(subject_id), str(hadm_id), str(stay_id))):
            os.mkdir(os.path.join(save_dir, str(subject_id), str(hadm_id), str(stay_id)))

        fname_lab = os.path.join(save_dir, str(subject_id), str(hadm_id), str(stay_id), 'labevents.csv')

        if os.path.exists(fname_lab):
            continue
        
        f_df_labevents = df_labevents[
            (df_labevents.subject_id == subject_id) & (df_labevents.hadm_id == hadm_id)]
        f_df_labevents = f_df_labevents[
            (f_df_labevents['charttime'] >= pd.Timestamp(patient.core['admittime'].values[0])) & (
                    f_df_labevents['charttime'] <= pd.Timestamp(patient.discharge.charttime.values[0]))]

        unique_labevent_test_id = list(f_df_labevents.itemid.unique())

        first_labevents = pd.DataFrame()
        for test_id in unique_labevent_test_id:
            first_test = f_df_labevents[f_df_labevents['itemid'] == test_id]
            first_test.comments.fillna('', inplace=True)
            first_test['comments'] = list(first_test['comments'].str.lower())

            first_test = first_test[~first_test['comments'].str.contains('cancel')]
            first_test = first_test.sort_values(by=['charttime'])[:1]

            first_labevents = pd.concat([first_labevents, first_test], sort=False)

        first_labevents.to_csv(fname_lab,index=False)


def process_lab_micro(sub_valid_idxes):

    print(f'PID name: {mp.current_process().name}, Process len: {len(sub_valid_idxes)}')
    match_micro(sub_valid_idxes)
    # match_labevents_all(sub_valid_idxes)


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

        print(ranges)
        pool.map(process_lab_micro, ranges)

# load data
print('load microbiologyevents')
df_microbiologyevents = pd.read_csv(core_mimiciv_path + 'hosp/microbiologyevents.csv.gz',
                                    compression='gzip', dtype={'comments': 'object', 'quantity': 'object'})

df_microbiologyevents['charttime']  = pd.to_datetime(df_microbiologyevents['charttime'])

# load data # labevents_dropnan.csv.gz, 
print('load labevents_dropnan')
df_labevents = pd.read_csv(core_mimiciv_path + 'hosp/labevents_dropnan.csv.gz', compression='gzip',
                               dtype={'storetime': 'object', 'valueuom': 'object', 'flag': 'str', 'priority': 'str',
                                      'comments': 'str'})

print('load d_labitems')
df_d_labitems = pd.read_csv(core_mimiciv_path + 'hosp/d_labitems.csv.gz', compression='gzip',
                            dtype={'loinc_code': 'object'})

print('merge labevents and d_labitems')
df_labevents = pd.merge(df_labevents,df_d_labitems, on=['itemid'],how='inner')
df_labevents['charttime'] = pd.to_datetime(df_labevents['charttime'])



if __name__ == '__main__':
    print('load valid idx')
    if process_type == 'img':
        with open('img_valid_idx_all.json', 'r') as f:
            valid_idxes = json.load(f)
        print(f'process img len: {len(valid_idxes)}')
    elif process_type == 'unimg':
        with open('unimg_valid_idx_all.json', 'r') as f:
            valid_idxes = json.load(f)
        print(f'process unimg len: {len(valid_idxes)}')

    else:
        raise Exception('type error')

    target_idxes_list = list(valid_idxes.keys()) # [:20] # only need idx_key

    main(target_idxes_list)