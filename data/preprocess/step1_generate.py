import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
import math
import datetime as dt
from tqdm import tqdm
from util2 import date_diff_hrs, save_patient_object, load_patient_object, Patient_MIMIC, get_timebound_patient_stay

import multiprocessing as mp
from multiprocessing import Pool


debug = False

# define paths
base_path = '/xxxx/xxxx/Multi-modal/data'
core_mimiciv_path = base_path + '/physionet.org/files/mimiciv/2.2/'
core_mimicED_path = base_path + '/physionet.org/files/mimic-iv-ed/2.2/'
core_mimicNote_path = base_path + '/physionet.org/files/mimic-iv-note/2.2/'
core_mimiciv_imgcxr_path = base_path + '/physionet.org/files/mimic-cxr-jpg/2.1.0/'


mimic_MMCaD_path = base_path + '/physionet.org/files/mimic_mmcad2/'



if not debug:
    # Initialize and load data
    df_admissions = pd.read_csv(core_mimiciv_path + 'hosp/admissions.csv.gz', compression='gzip')
    df_patients = pd.read_csv(core_mimiciv_path + 'hosp/patients.csv.gz', compression='gzip')
    df_transfers = pd.read_csv(core_mimiciv_path + 'hosp/transfers.csv.gz', compression='gzip')

    ## HOSP
    icd_diagnosis_dict = pd.read_csv(core_mimiciv_path + 'hosp/d_icd_diagnoses.csv.gz', compression='gzip')
    icd_diagnosis = pd.read_csv(core_mimiciv_path + 'hosp/diagnoses_icd.csv.gz', compression='gzip')
    icd_diagnosis = pd.merge(icd_diagnosis, icd_diagnosis_dict, on=['icd_code', 'icd_version'])

    df_omr = pd.read_csv(core_mimiciv_path + 'hosp/omr.csv.gz', compression='gzip')


    ## CXR IMG
    df_mimic_cxr_split = pd.read_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-split.csv.gz', compression='gzip')
    df_mimic_cxr_chexpert = pd.read_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-chexpert.csv.gz', compression='gzip')
    df_mimic_cxr_metadata = pd.read_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-metadata.csv.gz',compression='gzip')
    df_mimic_cxr_negbio = pd.read_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-negbio.csv.gz', compression='gzip')


    ## ED (subject_id, stay_id)
    df_ED_stays = pd.read_csv(core_mimicED_path + 'ed/edstays.csv.gz', compression='gzip')
    df_ED_medrecon = pd.read_csv(core_mimicED_path + 'ed/medrecon.csv.gz', compression='gzip')
    # df_ED_pyxis = pd.read_csv(core_mimicED_path + 'ed/pyxis.csv.gz', compression='gzip')
    df_ED_triage = pd.read_csv(core_mimicED_path + 'ed/triage.csv.gz', compression='gzip')
    df_ED_vitalsign = pd.read_csv(core_mimicED_path + 'ed/vitalsign.csv.gz', compression='gzip')

    ## DISCHARGE NOTES
    df_Note_discharge = pd.read_csv(core_mimicNote_path + 'note/discharge.csv.gz',compression='gzip')
    df_Note_radiology = pd.read_csv(core_mimicNote_path + 'note/radiology.csv.gz', compression='gzip')

    # HOSP
    # transfer time
    df_admissions['admittime'] = pd.to_datetime(df_admissions['admittime'])
    df_admissions['dischtime'] = pd.to_datetime(df_admissions['dischtime'])
    df_admissions['deathtime'] = pd.to_datetime(df_admissions['deathtime'])
    df_admissions['edregtime'] = pd.to_datetime(df_admissions['edregtime'])
    df_admissions['edouttime'] = pd.to_datetime(df_admissions['edouttime'])

    df_transfers['intime'] = pd.to_datetime(df_transfers['intime'])
    df_transfers['outtime'] = pd.to_datetime(df_transfers['outtime'])

    df_omr['chartdate'] = pd.to_datetime(df_omr['chartdate'])

    # CXR
    df_mimic_cxr_metadata = pd.read_csv(core_mimiciv_imgcxr_path + 'mimic-cxr-2.0.0-metadata_cxrtime.csv', dtype={'dicom_id': 'object'})
    df_mimic_cxr_metadata['cxrtime'] = pd.to_datetime(df_mimic_cxr_metadata['cxrtime'])

    # ED
    df_ED_stays['intime'] = pd.to_datetime(df_ED_stays['intime'])
    df_ED_stays['outtime'] = pd.to_datetime(df_ED_stays['outtime'])
    df_ED_vitalsign['charttime'] = pd.to_datetime(df_ED_vitalsign['charttime'])

    # Note
    df_Note_discharge['charttime'] = pd.to_datetime(df_Note_discharge['charttime'])
    df_Note_discharge['storetime'] = pd.to_datetime(df_Note_discharge['storetime'])
    df_Note_radiology['charttime'] = pd.to_datetime(df_Note_radiology['charttime'])
    df_Note_radiology['storetime'] = pd.to_datetime(df_Note_radiology['storetime'])

    # Sort
    df_admissions = df_admissions.sort_values(by=['subject_id','hadm_id']) # (431231, 16)
    df_patients = df_patients.sort_values(by=['subject_id'])
    df_transfers = df_transfers.sort_values(by=['subject_id','hadm_id'])

    df_omr = df_omr.sort_values(by=['subject_id'])

    ## ****  CXR  ****
    df_mimic_cxr_split = df_mimic_cxr_split.sort_values(by=['subject_id'])
    df_mimic_cxr_chexpert = df_mimic_cxr_chexpert.sort_values(by=['subject_id'])
    df_mimic_cxr_metadata = df_mimic_cxr_metadata.sort_values(by=['subject_id'])
    df_mimic_cxr_negbio = df_mimic_cxr_negbio.sort_values(by=['subject_id'])

    ## ****  ED   ****
    # ? 这里为什么从 df_ED_vitalsign 来
    # df_ED_stay = df_ED_vitalsign.sort_values(by=['subject_id','stay_id'])
    df_ED_medrecon = df_ED_medrecon.sort_values(by=['subject_id','stay_id'])
    df_ED_vitalsign = df_ED_vitalsign.sort_values(by=['subject_id','stay_id'])
    df_ED_triage = df_ED_triage.sort_values(by=['subject_id','stay_id'])

    ## ****  NOTES ****
    df_Note_discharge = df_Note_discharge.sort_values(by=['subject_id','hadm_id'])



def get_patient_micmic(key_subject_id, key_hadm_id, key_stay_id):
    # Inputs:
    #   key_subject_id -> subject_id is unique to a patient
    #   key_hadm_id    -> hadm_id is unique to a patient hospital stay
    #   key_stay_id    -> stay_id is unique to a patient ward stay
    #
    #   NOTES: Identifiers which specify the patient. More information about
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # Outputs:
    #   Patient_ICUstay -> ICU patient stay structure

    # -> FILTER data
    f_df_admissions = df_admissions[
        (df_admissions.subject_id == key_subject_id) & (df_admissions.hadm_id == key_hadm_id)]
    f_df_patients = df_patients[(df_patients.subject_id == key_subject_id)]
    f_df_transfers = df_transfers[(df_transfers.subject_id == key_subject_id) & (df_transfers.hadm_id == key_hadm_id)]
    ###-> Merge data into single patient structure
    f_df_core = f_df_admissions
    f_df_core = f_df_core.merge(f_df_patients, how='left')
    f_df_core = f_df_core.merge(f_df_transfers, how='left')
    f_df_core['stay_id'] = [key_stay_id]*len(f_df_core)


    f_df_omr = df_omr[(df_omr.subject_id == key_subject_id)]

    f_df_chartevents = None

    ##-> CXR
    f_df_mimic_cxr_split = df_mimic_cxr_split[(df_mimic_cxr_split.subject_id == key_subject_id)]
    f_df_mimic_cxr_chexpert = df_mimic_cxr_chexpert[(df_mimic_cxr_chexpert.subject_id == key_subject_id)]
    f_df_mimic_cxr_metadata = df_mimic_cxr_metadata[(df_mimic_cxr_metadata.subject_id == key_subject_id)]
    f_df_mimic_cxr_negbio = df_mimic_cxr_negbio[(df_mimic_cxr_negbio.subject_id == key_subject_id)]
    ###-> Merge data into single patient structure
    f_df_cxr = f_df_mimic_cxr_split
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_chexpert, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_metadata, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_negbio, how='left')

    ##-> ED
    if key_stay_id is not None:
        f_df_ED_stays = df_ED_stays[(df_ED_stays.subject_id == key_subject_id) & (df_ED_stays.hadm_id == key_hadm_id) & (
                df_ED_stays.stay_id == key_stay_id)]
        f_df_ED_vitalsign = df_ED_vitalsign[(df_ED_vitalsign.subject_id == key_subject_id)  & (
                df_ED_vitalsign.stay_id == key_stay_id)]
        f_df_ED_medrecon = df_ED_medrecon[(df_ED_medrecon.subject_id == key_subject_id)  & (
                df_ED_medrecon.stay_id == key_stay_id)]
        f_df_ED_triage = df_ED_triage[(df_ED_triage.subject_id == key_subject_id) & (
                df_ED_triage.stay_id == key_stay_id)]
    else:
        f_df_ED_stays=pd.DataFrame()
        f_df_ED_vitalsign=pd.DataFrame()
        f_df_ED_medrecon=pd.DataFrame()
        f_df_ED_triage=pd.DataFrame()

    ##-> NOTES
    f_df_discharge = df_Note_discharge[
        (df_Note_discharge.subject_id == key_subject_id) & (df_Note_discharge.hadm_id == key_hadm_id)]
    f_df_radiology_report = df_Note_radiology[
        (df_Note_radiology.subject_id == key_subject_id) & (df_Note_radiology.hadm_id == key_hadm_id)]

    # -> Create & Populate patient structure
    ## CORE
    core = f_df_core

    ## HOSP
    omr = f_df_omr

    ## ICU
    chartevents = f_df_chartevents
    icustays=None
    datetimeevents=None

    ## CXR
    cxr = f_df_cxr

    ## ED
    ED_stay = f_df_ED_stays
    ED_medrecon = f_df_ED_medrecon
    ED_vitalsign = f_df_ED_vitalsign
    ED_triage = f_df_ED_triage

    ## NOTES
    Note_discharge =f_df_discharge
    Note_radiology_report = f_df_radiology_report

    current_icd_diagnosis = icd_diagnosis[
        (icd_diagnosis['subject_id'] == int(key_subject_id)) & (icd_diagnosis['hadm_id'] == int(key_hadm_id))]

    # Create patient object and return
    Patient_MIMIC_stay = Patient_MIMIC(f_df_core, omr, icustays, datetimeevents, \
                                  chartevents, cxr, ED_stay,ED_medrecon, ED_vitalsign,ED_triage, Note_discharge, current_icd_diagnosis,Note_radiology_report)

    return Patient_MIMIC_stay

def extract_single_patient_records_mimiciv(patient_idx, df_ids, start_hr=None, end_hr=None):

    key_subject_id = int(df_ids.iloc[patient_idx].subject_id)
    key_hadm_id = int(df_ids.iloc[patient_idx].hadm_id)
    if not np.isnan(df_ids.iloc[patient_idx].stay_id):
        key_stay_id = int(df_ids.iloc[patient_idx].stay_id)
    else:
        key_stay_id = np.nan
    start_hr = start_hr  # Select timestamps
    end_hr = end_hr  # Select timestamps
    patient = get_patient_micmic(key_subject_id, key_hadm_id, key_stay_id)
    filename = f"{patient_idx:08d}" + '.pkl'
    dt_patient = get_timebound_patient_stay(patient, start_hr, end_hr)

    return key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient


def process_patient(df_ids_list):
    df_ids_list = df_ids_list
    valid_idx_id_dict = {}
    for patient_idx in tqdm(df_ids_list):
        key_subject_id = int(df_ids.iloc[patient_idx].subject_id)
        key_hadm_id = int(df_ids.iloc[patient_idx].hadm_id)

        if not np.isnan(df_ids.iloc[patient_idx].stay_id):
            key_stay_id = int(df_ids.iloc[patient_idx].stay_id)
        else:
            key_stay_id = np.nan
        
        key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient = extract_single_patient_records_mimiciv(
            patient_idx, df_ids)
    

        if len(patient.icd_diagnosis) < 1:
            continue

        valid_idx_id_dict[patient_idx] = [key_subject_id, key_hadm_id, key_stay_id]

        filename = f"{patient_idx:08d}" + '.pkl'
        # process the interrupt station
        file_save_path = os.path.join(save_path ,filename)
        if not os.path.exists(file_save_path):
            save_patient_object(patient, file_save_path)

    return valid_idx_id_dict


def main(gen_df_ids):
    num_processes = 10
    process_list = range(len(gen_df_ids))
    chunk_size = len(process_list) // num_processes

    with Pool(processes=num_processes) as pool:
        ranges = []
        for i in range(num_processes):
            if i == num_processes - 1:
                ranges.append(range(i * chunk_size, len(process_list)))
            else:
                ranges.append(range(i * chunk_size, (i + 1) * chunk_size))

        print(ranges)

        results = pool.map(process_patient, ranges)

    valid_idx_id_dict = {}
    for d in results:
        valid_idx_id_dict.update(d)
    print(f'Generate Finished: {len(valid_idx_id_dict)}')

    if 'unimg' in save_path:
        save_json_path = 'unimg_valid_idx_all.json'
    else:
        save_json_path = 'img_valid_idx_all.json'

    with open(save_json_path, 'w') as f:
        json.dump(valid_idx_id_dict, f)


# process_type = 'img'
process_type = 'unimg'

if process_type == 'img':
    save_path= os.path.join(mimic_MMCaD_path, 'img','patient_object')
    df_ids = pd.read_csv(core_mimiciv_path + 'sid_hadm_stay_mimiciv_key_ids.csv')   # img
    # df_ids = pd.read_csv(core_mimiciv_path + 'sid_hadm_stay_mimiciv_clindiag_ids.csv') # all
    os.makedirs(save_path, exist_ok=True)
    print('Image data len: ' + str(len(df_ids)))
elif process_type == 'unimg':
    # process unimg data
    save_path= os.path.join(mimic_MMCaD_path, 'unimg','patient_object')
    df_ids = pd.read_csv(core_mimiciv_path + 'sid_hadm_stay_mimiciv_nimg_key_ids.csv')
    print('UN Image data len: ' + str(len(df_ids)))
    os.makedirs(save_path, exist_ok=True)
else:
    raise  Exception('process_type is wrong')

if __name__ == '__main__':
    ## clip part
    clip_df_ids = df_ids # [:100]
    print(f'clip_df_ids len: {len(clip_df_ids)}')

    main(clip_df_ids)



