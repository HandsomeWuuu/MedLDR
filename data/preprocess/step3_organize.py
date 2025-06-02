import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import time
import pickle
from util2 import Patient_MIMIC, date_diff_hrs,get_timebound_patient_stay

import multiprocessing as mp
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
'''
    Goal: Organize all data:

    - Output:
        - 'hosp_ed_cxr_data.csv': After matching imaging data, merge the entire table and save
        - 'radiology_report.csv': Save directly
        - 'icd_diagnosis.pkl': Save directly
        - Plan to process ECG (electrocardiogram) information
'''
# TODO edit path here
base_path = '/xxx/Multi-modal/data'

core_mimiciv_path = base_path + '/physionet.org/files/mimiciv/2.2/'
core_mimicED_path = base_path + '/physionet.org/files/mimic-iv-ed/2.2/'
core_mimicNote_path = base_path + '/physionet.org/files/mimic-iv-note/2.2/'
core_mimiciv_imgcxr_path = base_path + '/physionet.org/files/mimic-cxr-jpg/2.1.0/'

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

def organize_data(valid_idxes_list):
    # print(f'Processing len:{len(valid_idxes_list)}, list:{valid_idxes_list}')

    for idx in tqdm(valid_idxes_list):
        pickle_filename = f"{int(idx):08d}" + '.pkl'

        with open(MMCaD_patient_path + '/patient_object/' + pickle_filename, 'rb') as input:
            patient = pickle.load(input)

        subject_id = patient.core['subject_id'].values[0]
        hadm_id = patient.core['hadm_id'].values[0]
        if not np.isnan(patient.core['stay_id'].values[0]):
            stay_id = int(patient.core['stay_id'].values[0])
        else:
            stay_id = 0
        fname = os.path.join(save_dir, str(subject_id), str(hadm_id), str(stay_id), 'hosp_ed_cxr_data.csv')

        if not os.path.exists(os.path.join(save_dir, str(subject_id))):
            os.mkdir(os.path.join(save_dir, str(subject_id)))

        if not os.path.exists(os.path.join(save_dir, str(subject_id), str(hadm_id))):
            os.mkdir(os.path.join(save_dir, str(subject_id), str(hadm_id)))

        if not os.path.exists(os.path.join(save_dir, str(subject_id), str(hadm_id), str(stay_id))):
            os.mkdir(os.path.join(save_dir, str(subject_id), str(hadm_id), str(stay_id)))

        # read and write to csv
        radiology_report = patient.radiology_report
        radiology_report.to_csv(
            os.path.join(save_dir, str(subject_id), str(hadm_id), str(stay_id), 'radiology_report.csv'), index=False)

        # Get information of chest x-rays conducted within this patient stay
        df_cxr = patient.cxr

        # admittime = pd.Timestamp(patient.core.admittime.values[0])
        # dischtime = pd.Timestamp(patient.discharge.charttime.values[0])

        if patient.EDstay.intime.empty:
            admittime = None
        else:
            admittime = pd.Timestamp(patient.EDstay.intime.values[0])

        dischtime = pd.Timestamp(patient.discharge.charttime.values[0])

        if len(df_cxr) > 0 and admittime != None:
            df_stay_cxr = df_cxr.loc[(df_cxr['charttime'] >= admittime) & (df_cxr['charttime'] <= dischtime)]
        else:
            df_stay_cxr = pd.DataFrame()

        if not df_stay_cxr.empty:
            current_df_fusion = pd.DataFrame()
            for idx, df_stay_cxr_row in df_stay_cxr.iterrows():
                study_id = df_stay_cxr_row['study_id']

                # Get stay anchor times
                img_charttime = df_stay_cxr_row['charttime']
                img_deltacharttime = df_stay_cxr_row['deltacharttime']

                # Get time to discharge and discharge location/status
                img_id = df_stay_cxr_row["dicom_id"]
                img_length_of_stay = date_diff_hrs(dischtime, img_charttime)
                discharge_location = patient.core['discharge_location'][0]
                if discharge_location == "DIED":
                    death_status = 1
                else:
                    death_status = 0

                # Select allowed timestamp range
                start_hr = None
                end_hr = None

                dt_patient = get_timebound_patient_stay(patient, start_hr, end_hr)
                is_included = True

                df_fusion = pd.DataFrame(
                    [[subject_id, hadm_id, stay_id, admittime, dischtime, img_id, img_charttime,
                      discharge_location, img_length_of_stay, death_status]],
                    columns=['subject_id', 'hadm_id', 'stay_id', 'admittime', 'discharge_time', 'img_id',
                             'img_charttime', 'discharge_location', 'img_length_of_stay',
                             'death_status'])

                df_fusion['omr_result_name'] = [list(dt_patient.omr['result_name'].values)]
                df_fusion['omr_result_value'] = [list(dt_patient.omr['result_value'].values)]
                df_fusion['omr_chartdate'] = [list(dt_patient.omr['chartdate'].values)]

                # df_fusion['ed_charttime'] = dt_patient.EDtriage['charttime']
                df_fusion['gender'] = dt_patient.core['gender'].values[0]
                df_fusion['race'] = dt_patient.core['race'].values[0]
                df_fusion['arrival_transport'] = dt_patient.core['admission_location'].values[0]
                df_fusion['anchor_age'] = dt_patient.core['anchor_age'].values[0]

                if not dt_patient.EDmedrecon.empty:
                    df_fusion['ed_medrecon_name'] = [list(dt_patient.EDmedrecon['name'].values)]
                    df_fusion['ed_medrecon_charttime'] = [list(dt_patient.EDmedrecon['charttime'].values)]
                    df_fusion['ed_medrecon_gsn'] = [list(dt_patient.EDmedrecon['gsn'].values)]
                    df_fusion['ed_medrecon_ndc'] = [list(dt_patient.EDmedrecon['ndc'].values)]
                    df_fusion['ed_medrecon_etc_rn'] = [list(dt_patient.EDmedrecon['etc_rn'].values)]
                    df_fusion['ed_medrecon_etccode'] = [list(dt_patient.EDmedrecon['etccode'].values)]
                    df_fusion['ed_medrecon_etcdescription'] = [list(dt_patient.EDmedrecon['etcdescription'].values)]
                else:
                    df_fusion['ed_medrecon_name'] = None
                    df_fusion['ed_medrecon_charttime'] = None
                    df_fusion['ed_medrecon_gsn'] = None
                    df_fusion['ed_medrecon_ndc'] = None
                    df_fusion['ed_medrecon_etc_rn'] = None
                    df_fusion['ed_medrecon_etccode'] = None
                    df_fusion['ed_medrecon_etcdescription'] = None

                if not dt_patient.EDtriage.empty:
                    df_fusion['ed_temperature'] = dt_patient.EDtriage['temperature'].values[0]
                    df_fusion['ed_heartrate'] = dt_patient.EDtriage['heartrate'].values[0]
                    df_fusion['ed_resprate'] = dt_patient.EDtriage['resprate'].values[0]
                    df_fusion['ed_o2sat'] = dt_patient.EDtriage['o2sat'].values[0]
                    df_fusion['ed_sbp'] = dt_patient.EDtriage['sbp'].values[0]
                    df_fusion['ed_dbp'] = dt_patient.EDtriage['dbp'].values[0]
                    df_fusion['ed_acuity'] = dt_patient.EDtriage['acuity'].values[0]
                    df_fusion['ed_pain'] = dt_patient.EDtriage['pain'].values[0]
                    df_fusion['ed_chiefcomplaint'] = dt_patient.EDtriage['chiefcomplaint'].values[0]
                else:
                    df_fusion['ed_temperature'] = None
                    df_fusion['ed_heartrate'] = None
                    df_fusion['ed_resprate'] = None
                    df_fusion['ed_o2sat'] = None
                    df_fusion['ed_sbp'] = None
                    df_fusion['ed_dbp'] = None
                    df_fusion['ed_acuity'] = None
                    df_fusion['ed_pain'] = None
                    df_fusion['ed_chiefcomplaint'] = None

                df_fusion['discharge_note_text'] = dt_patient.discharge['text'].values[0]
                df_fusion['discharge_note_id'] = dt_patient.discharge['note_id'].values[0]
                # df_fusion['discharge_note_charttime']=dt_patient.discharge['charttime'].values[0]

                current_df_fusion = pd.concat([current_df_fusion, df_fusion], sort=False)

            # fname
            current_df_fusion.to_csv(fname, index=False)
        else:
            current_df_fusion = pd.DataFrame()
            # Get stay anchor times
            img_charttime = None
            img_deltacharttime = None

            # Get time to discharge and discharge location/status
            img_id = None
            study_id = None
            img_length_of_stay = None
            discharge_location = patient.core['discharge_location'][0]
            if discharge_location == "DIED":
                death_status = 1
            else:
                death_status = 0

            # Select allowed timestamp range
            start_hr = None
            end_hr = None

            dt_patient = get_timebound_patient_stay(patient, start_hr, end_hr)
            # save_patient_object(dt_patient, core_mimiciv_path + 'pickle/' + f"{patient_idx:08d}" + '_dt.pkl')
            is_included = True

            df_fusion = pd.DataFrame(
                [[subject_id, hadm_id, stay_id, admittime, dischtime, img_id, study_id, img_charttime,
                  discharge_location, img_length_of_stay, death_status]],
                columns=['subject_id', 'hadm_id', 'stay_id', 'admittime', 'discharge_time', 'img_id', 'study_id',
                         'img_charttime', 'discharge_location', 'img_length_of_stay',
                         'death_status'])

            df_fusion['omr_result_name'] = [list(dt_patient.omr['result_name'].values)]
            df_fusion['omr_result_value'] = [list(dt_patient.omr['result_value'].values)]
            df_fusion['omr_chartdate'] = [list(dt_patient.omr['chartdate'].values)]

            # df_fusion['ed_charttime'] = dt_patient.EDtriage['charttime']
            df_fusion['gender'] = dt_patient.core['gender'].values[0]
            df_fusion['race'] = dt_patient.core['race'].values[0]
            df_fusion['arrival_transport'] = dt_patient.core['admission_location'].values[0]
            df_fusion['anchor_age'] = dt_patient.core['anchor_age'].values[0]

            if not dt_patient.EDmedrecon.empty:
                df_fusion['ed_medrecon_name'] = [list(dt_patient.EDmedrecon['name'].values)]
                df_fusion['ed_medrecon_charttime'] = [list(dt_patient.EDmedrecon['charttime'].values)]
                df_fusion['ed_medrecon_gsn'] = [list(dt_patient.EDmedrecon['gsn'].values)]
                df_fusion['ed_medrecon_ndc'] = [list(dt_patient.EDmedrecon['ndc'].values)]
                df_fusion['ed_medrecon_etc_rn'] = [list(dt_patient.EDmedrecon['etc_rn'].values)]
                df_fusion['ed_medrecon_etccode'] = [list(dt_patient.EDmedrecon['etccode'].values)]
                df_fusion['ed_medrecon_etcdescription'] = [list(dt_patient.EDmedrecon['etcdescription'].values)]
            else:
                df_fusion['ed_medrecon_name'] = None
                df_fusion['ed_medrecon_charttime'] = None
                df_fusion['ed_medrecon_gsn'] = None
                df_fusion['ed_medrecon_ndc'] = None
                df_fusion['ed_medrecon_etc_rn'] = None
                df_fusion['ed_medrecon_etccode'] = None
                df_fusion['ed_medrecon_etcdescription'] = None

            if not dt_patient.EDtriage.empty:
                df_fusion['ed_temperature'] = dt_patient.EDtriage['temperature'].values[0]
                df_fusion['ed_heartrate'] = dt_patient.EDtriage['heartrate'].values[0]
                df_fusion['ed_resprate'] = dt_patient.EDtriage['resprate'].values[0]
                df_fusion['ed_o2sat'] = dt_patient.EDtriage['o2sat'].values[0]
                df_fusion['ed_sbp'] = dt_patient.EDtriage['sbp'].values[0]
                df_fusion['ed_dbp'] = dt_patient.EDtriage['dbp'].values[0]
                df_fusion['ed_acuity'] = dt_patient.EDtriage['acuity'].values[0]
                df_fusion['ed_pain'] = dt_patient.EDtriage['pain'].values[0]
                df_fusion['ed_chiefcomplaint'] = dt_patient.EDtriage['chiefcomplaint'].values[0]
            else:
                df_fusion['ed_temperature'] = None
                df_fusion['ed_heartrate'] = None
                df_fusion['ed_resprate'] = None
                df_fusion['ed_o2sat'] = None
                df_fusion['ed_sbp'] = None
                df_fusion['ed_dbp'] = None
                df_fusion['ed_acuity'] = None
                df_fusion['ed_pain'] = None
                df_fusion['ed_chiefcomplaint'] = None

            df_fusion['discharge_note_text'] = dt_patient.discharge['text'].values[0]
            df_fusion['discharge_note_id'] = dt_patient.discharge['note_id'].values[0]
            # df_fusion['discharge_note_charttime']=dt_patient.discharge['charttime'].values[0]

            current_df_fusion = pd.concat([current_df_fusion, df_fusion], sort=False)
            current_df_fusion.to_csv(fname, index=False)


        drop_keywords = ['micro', 'ed_continuous', 'lab']
        drop_columns = []
        for i in drop_keywords:
            for column in current_df_fusion.columns:
                if i in column:
                    drop_columns.append(column)
        current_df_fusion.drop(drop_columns, axis=1, inplace=True)

        # Write
        current_icd_diagnosis = patient.icd_diagnosis
        with open(os.path.join(save_dir, str(subject_id), str(hadm_id), str(stay_id), 'icd_diagnosis.pkl'), 'wb') as f:
            pickle.dump(current_icd_diagnosis, f)

def main(process_list):
    num_processes = 1
    chunk_size = len(process_list) // num_processes

    with Pool(processes=num_processes) as pool:
        ranges = []
        for i in range(num_processes):
            if i == num_processes - 1:
                ranges.append(process_list[i * chunk_size: len(process_list)])
            else:
                ranges.append(process_list[i * chunk_size: (i + 1) * chunk_size])

        print(f'Split data: all {len(ranges)}, single {len(ranges[0])}')
        pool.map(organize_data, ranges)


if __name__ == '__main__':
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

    target_idxes_list = list(valid_idxes.keys()) # only need idx_key

    print(len(target_idxes_list))
    main(target_idxes_list)

