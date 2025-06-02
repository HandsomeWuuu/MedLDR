import pickle
import math
import numpy as np
import pandas as pd
import datetime as dt

def date_diff_hrs(t1, t0):
    # Inputs:
    #   t1 -> Final timestamp in a patient hospital stay
    #   t0 -> Initial timestamp in a patient hospital stay

    # Outputs:
    #   delta_t -> Patient stay structure bounded by allowed timestamps

    try:
        delta_t = (t1 - t0).total_seconds() / 3600  # Result in hrs
    except:
        delta_t = math.nan

    return delta_t

# SAVE SINGLE PATIENT ICU STAY RECORDS FOR MIMIC-IV
def save_patient_object(obj, filepath):
    # Inputs:
    #   obj -> Timebound ICU patient stay object
    #   filepath -> Pickle file path to save object to
    #
    # Outputs:
    #   VOID -> Object is saved in filename path
    # Overwrites any existing file.
    
    with open(filepath, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# LOAD SINGLE PATIENT ICU STAY RECORDS FOR MIMIC-IV
def load_patient_object(filepath):
    # Inputs:
    #   filepath -> Pickle file path to save object to
    #
    # Outputs:
    #   obj -> Loaded timebound ICU patient stay object

    # Overwrites any existing file.
    with open(filepath, 'rb') as input:
        return pickle.load(input)

class Patient_MIMIC(object):
    def __init__(self, core, omr, icustays, datetimeevents,
                                  chartevents, cxr, ED_stay,EDmedrecon, ED_vitalsign,ED_triage, Note_discharge, icd_diagnosis, radiology_report):
        ## CORE
        self.core = core

        ## HOSP
        self.omr = omr
        # self.diagnoses_icd = diagnoses_icd
        # self.drgcodes = drgcodes
        # self.emar = emar
        # self.emar_detail = emar_detail
        # self.hcpcsevents = hcpcsevents
        # self.labevents = labevents
        # self.microbiologyevents = microbiologyevents
        # self.poe = poe
        # self.poe_detail = poe_detail
        # self.prescriptions = prescriptions
        # self.procedures_icd = procedures_icd
        # self.services = services
        ## ICU
        # self.procedureevents = procedureevents
        # self.outputevents = outputevents
        # self.inputevents = inputevents
        self.icustays = icustays
        self.datetimeevents = datetimeevents
        self.chartevents = chartevents
        ## CXR
        self.cxr = cxr

        ## ED
        self.EDstay = ED_stay
        self.EDmedrecon = EDmedrecon
        self.EDvitalsign = ED_vitalsign
        self.EDtriage = ED_triage
        ## NOTES
        self.discharge = Note_discharge
        self.icd_diagnosis = icd_diagnosis
        self.radiology_report = radiology_report


def get_timebound_patient_stay(Patient_MIMIC, start_hr=None, end_hr=None):
    # Inputs:
    #   Patient_MIMIC -> Patient ICU stay structure
    #   start_hr -> start_hr indicates the first valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #   end_hr -> end_hr indicates the last valid time (in hours) from the admition time "admittime" for all retreived features, input "None" to avoid time bounding
    #
    #   NOTES: Identifiers which specify the patient. More information about
    #   these identifiers is available at https://mimic-iv.mit.edu/basics/identifiers

    # Outputs:
    #   Patient_MIMIC -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any

    # %% EXAMPLE OF USE
    ## Let's select a single patient
    '''
    key_subject_id = 10000032
    key_hadm_id = 29079034
    key_stay_id = 39553978
    start_hr = 0
    end_hr = 24
    patient = get_Patient_MIMIC(key_subject_id, key_hadm_id, key_stay_id)
    dt_patient = get_timebound_Patient_MIMIC(patient, start_hr , end_hr)
    '''

    # Create a deep copy so that it is not the same object
    # Patient_MIMIC = copy.deepcopy(Patient_MIMIC)

    ## --> Process Event Structure Calculations
    admittime = pd.Timestamp(Patient_MIMIC.core['intime'].values[0])
    dischtime = pd.Timestamp(Patient_MIMIC.discharge['charttime'].values[0])

    if len(Patient_MIMIC.omr) != 0:
        # Patient_MIMIC.omr['deltacharttime'] = Patient_MIMIC.omr.apply(
        #     lambda x: date_diff_hrs(x['chartdate'], admittime) if not x.empty else None, axis=1)
        
        omr_copy = Patient_MIMIC.omr.copy()
        omr_copy['deltacharttime'] = omr_copy.apply(lambda x: date_diff_hrs(x['chartdate'], admittime) if not x.empty else None, axis=1)
        Patient_MIMIC.omr = omr_copy
        del omr_copy

    # if len(Patient_MIMIC.emar) != 0:
    #     Patient_MIMIC.emar['deltacharttime'] = Patient_MIMIC.emar.apply(
    #         lambda x: date_diff_hrs(x['charttime'], admittime) if not x.empty else None, axis=1)
    # Patient_MIMIC.labevents['deltacharttime'] = Patient_MIMIC.labevents.apply(
    #     lambda x: date_diff_hrs(x['charttime'], admittime) if not x.empty else None, axis=1)
    # if len(Patient_MIMIC.microbiologyevents) != 0:
    #     Patient_MIMIC.microbiologyevents['deltacharttime'] = Patient_MIMIC.microbiologyevents.apply(
    #         lambda x: date_diff_hrs(x['charttime'], admittime) if not x.empty else None, axis=1)
    # Patient_MIMIC.outputevents['deltacharttime'] = Patient_MIMIC.outputevents.apply(
    #     lambda x: date_diff_hrs(x['charttime'], admittime) if not x.empty else None, axis=1)
    # ICU need to check for empty
    # if len(Patient_MIMIC.datetimeevents)!=0:
    #     Patient_MIMIC.datetimeevents['deltacharttime'] = Patient_MIMIC.datetimeevents.apply(
    #         lambda x: date_diff_hrs(x['charttime'], admittime) if not x.empty else None, axis=1)
    # Patient_MIMIC.chartevents['deltacharttime'] = Patient_MIMIC.chartevents.apply(
    #     lambda x: date_diff_hrs(x['charttime'], admittime) if not x.empty else None, axis=1)
    # print("#################   EDvitalsign\n",Patient_MIMIC.EDvitalsign)
    # print("#################   discharge\n",Patient_MIMIC.discharge)
    if len(Patient_MIMIC.EDvitalsign) != 0:
        # Patient_MIMIC.EDvitalsign['deltacharttime'] = Patient_MIMIC.EDvitalsign.apply(
        #     lambda x: date_diff_hrs(x['charttime'], admittime) if not x.empty else None, axis=1)

        EDvitalsign_copy = Patient_MIMIC.EDvitalsign.copy()
        EDvitalsign_copy['deltacharttime'] = Patient_MIMIC.EDvitalsign.apply(
            lambda x: date_diff_hrs(x['charttime'], admittime) if not x.empty else None, axis=1)
        Patient_MIMIC.EDvitalsign = EDvitalsign_copy
        del EDvitalsign_copy

        # Patient_MIMIC.discharge['deltacharttime'] = Patient_MIMIC.discharge.apply(
        #     lambda x: date_diff_hrs(x['charttime'], admittime) if not x.empty else None, axis=1)

    # Re-calculate times of CXR database, charttime=crxtime
    if len(Patient_MIMIC.cxr) != 0:
        Patient_MIMIC.cxr['StudyDateForm'] = pd.to_datetime(Patient_MIMIC.cxr['StudyDate'], format='%Y%m%d')
        Patient_MIMIC.cxr['StudyTimeForm'] = Patient_MIMIC.cxr.apply(lambda x: '%#010.3f' % x['StudyTime'], 1)
        Patient_MIMIC.cxr['StudyTimeForm'] = pd.to_datetime(Patient_MIMIC.cxr['StudyTimeForm'],
                                                              format='%H%M%S.%f').dt.time
        Patient_MIMIC.cxr['charttime'] = Patient_MIMIC.cxr.apply(
            lambda r: dt.datetime.combine(r['StudyDateForm'], r['StudyTimeForm']), 1)
        Patient_MIMIC.cxr['charttime'] = Patient_MIMIC.cxr['charttime'].dt.floor('Min')
        Patient_MIMIC.cxr['deltacharttime'] = Patient_MIMIC.cxr.apply(
            lambda x: date_diff_hrs(x['charttime'], admittime) if not x.empty else None, axis=1)

    ## --> Filter by allowable time stamps
    if not (start_hr == None):
        if len(Patient_MIMIC.omr) != 0:
            Patient_MIMIC.omr = Patient_MIMIC.omr[
                (Patient_MIMIC.omr.deltacharttime >= start_hr) | pd.isnull(Patient_MIMIC.omr.deltacharttime)]
        # if len(Patient_MIMIC.emar) != 0:
        #     Patient_MIMIC.emar = Patient_MIMIC.emar[
        #         (Patient_MIMIC.emar.deltacharttime >= start_hr) | pd.isnull(Patient_MIMIC.emar.deltacharttime)]
        # Patient_MIMIC.labevents = Patient_MIMIC.labevents[
        #     (Patient_MIMIC.labevents.deltacharttime >= start_hr) | pd.isnull(
        #         Patient_MIMIC.labevents.deltacharttime)]
        # if len(Patient_MIMIC.microbiologyevents) != 0:
        #     Patient_MIMIC.microbiologyevents = Patient_MIMIC.microbiologyevents[
        #         (Patient_MIMIC.microbiologyevents.deltacharttime >= start_hr) | pd.isnull(
        #             Patient_MIMIC.microbiologyevents.deltacharttime)]
        # Patient_MIMIC.outputevents = Patient_MIMIC.outputevents[
        #     (Patient_MIMIC.outputevents.deltacharttime >= start_hr) | pd.isnull(
        #         Patient_MIMIC.outputevents.deltacharttime)]
        # if len(Patient_MIMIC.datetimeevents)!=0:
        #     Patient_MIMIC.datetimeevents = Patient_MIMIC.datetimeevents[
        #         (Patient_MIMIC.datetimeevents.deltacharttime >= start_hr) | pd.isnull(
        #             Patient_MIMIC.datetimeevents.deltacharttime)]
        # Patient_MIMIC.chartevents = Patient_MIMIC.chartevents[
        #     (Patient_MIMIC.chartevents.deltacharttime >= start_hr) | pd.isnull(
        #         Patient_MIMIC.chartevents.deltacharttime)]
        if len(Patient_MIMIC.cxr) != 0:
            Patient_MIMIC.cxr = Patient_MIMIC.cxr[
                (Patient_MIMIC.cxr.deltacharttime >= start_hr) | pd.isnull(Patient_MIMIC.cxr.deltacharttime)]
            Patient_MIMIC.imcxr = [Patient_MIMIC.imcxr[i] for i, x in enumerate(
                (Patient_MIMIC.cxr.deltacharttime >= start_hr) | pd.isnull(Patient_MIMIC.cxr.deltacharttime)) if x]
        # ED
        # if len(Patient_MIMIC.EDvitalsign) != 0:
        # Patient_MIMIC.EDvitalsign = Patient_MIMIC.EDvitalsign[
        #     (Patient_MIMIC.EDvitalsign.deltacharttime >= start_hr) | pd.isnull(
        #         Patient_MIMIC.EDvitalsign.deltacharttime)]
        # Notes
        # Patient_MIMIC.discharge = Patient_MIMIC.discharge[
        #     (Patient_MIMIC.Note_discharge.deltacharttime >= start_hr) | pd.isnull(Patient_MIMIC.Note_discharge.deltacharttime)]

    if not (end_hr == None):
        if len(Patient_MIMIC.omr) != 0:
            Patient_MIMIC.omr = Patient_MIMIC.omr[
                (Patient_MIMIC.omr.deltacharttime <= end_hr) | pd.isnull(Patient_MIMIC.omr.deltacharttime)]
        # if len(Patient_MIMIC.emar) != 0:
        #     Patient_MIMIC.emar = Patient_MIMIC.emar[
        #         (Patient_MIMIC.emar.deltacharttime <= end_hr) | pd.isnull(Patient_MIMIC.emar.deltacharttime)]
        # Patient_MIMIC.labevents = Patient_MIMIC.labevents[
        #     (Patient_MIMIC.labevents.deltacharttime <= end_hr) | pd.isnull(Patient_MIMIC.labevents.deltacharttime)]
        # if len(Patient_MIMIC.microbiologyevents) != 0:
        #     Patient_MIMIC.microbiologyevents = Patient_MIMIC.microbiologyevents[
        #         (Patient_MIMIC.microbiologyevents.deltacharttime <= end_hr) | pd.isnull(
        #             Patient_MIMIC.microbiologyevents.deltacharttime)]
        # Patient_MIMIC.outputevents = Patient_MIMIC.outputevents[
        #     (Patient_MIMIC.outputevents.deltacharttime <= end_hr) | pd.isnull(
        #         Patient_MIMIC.outputevents.deltacharttime)]
        # if len(Patient_MIMIC.datetimeevents)!=0:
        #     Patient_MIMIC.datetimeevents = Patient_MIMIC.datetimeevents[
        #         (Patient_MIMIC.datetimeevents.deltacharttime <= end_hr) | pd.isnull(
        #             Patient_MIMIC.datetimeevents.deltacharttime)]
        # Patient_MIMIC.chartevents = Patient_MIMIC.chartevents[
        #     (Patient_MIMIC.chartevents.deltacharttime <= end_hr) | pd.isnull(
        #         Patient_MIMIC.chartevents.deltacharttime)]
        Patient_MIMIC.cxr = Patient_MIMIC.cxr[
            (Patient_MIMIC.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_MIMIC.cxr.deltacharttime)]
        # Patient_MIMIC.imcxr = [Patient_MIMIC.imcxr[i] for i, x in enumerate(
        #     (Patient_MIMIC.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_MIMIC.cxr.deltacharttime)) if x]

        # ED
        if len(Patient_MIMIC.EDvitalsign) != 0:
            Patient_MIMIC.EDvitalsign = Patient_MIMIC.EDvitalsign[
                (Patient_MIMIC.EDvitalsign.deltacharttime <= end_hr) | pd.isnull(
                    Patient_MIMIC.EDvitalsign.deltacharttime)]
        # Notes
        # Patient_MIMIC.discharge = Patient_MIMIC.discharge[
        #     (Patient_MIMIC.Note_discharge.deltacharttime <= end_hr) | pd.isnull(Patient_MIMIC.Note_discharge.deltacharttime)]

        # Filter CXR to match allowable patient stay
        if len(Patient_MIMIC.cxr) != 0:
            Patient_MIMIC.cxr = Patient_MIMIC.cxr[(Patient_MIMIC.cxr.charttime <= dischtime)]

    # dt_patient.labevents = dt_patient.labevents.reset_index(drop=True)

    return Patient_MIMIC