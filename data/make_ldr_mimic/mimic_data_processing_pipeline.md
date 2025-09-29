# MIMIC Data Processing Pipeline

This document summarizes the full data processing pipeline for MIMIC data in this project, including both the preprocessing and main data construction steps. All scripts referenced are located in the `data/make_ldr_mimic/` directory and its subfolders.

---

## 0. Preprocessing (preprocess_mimic)

### 0.1 Generate Structured Patient Stay Objects
- **Script:** `preprocess_mimic/step1_generate.py`
- **Description:**
    - Iterates over all patient/hospital/ED stays.
    - Integrates core info, ICD diagnoses, imaging, ED, discharge notes, etc.
    - Saves each stay as a structured `Patient_MIMIC` object (`.pkl` file) for fast access.
- **Output:** One `.pkl` file per stay.

### 0.2 Match Laboratory and Microbiology Events
- **Script:** `preprocess_mimic/step2_match_lab.py`
- **Description:**
    - For each stay, matches all laboratory (labevents) and microbiology events during the hospital stay.
    - Saves as CSV files for each stay.
- **Output:** `labevents.csv`, `microbiologyevents.csv` per stay.

### 0.3 Organize and Merge Stay Information
- **Script:** `preprocess_mimic/step3_organize.py`
- **Description:**
    - Merges imaging, ICD diagnoses, discharge notes, and other information at the stay level.
    - Produces standardized files for downstream processing.
- **Output:** `hosp_ed_cxr_data.csv`, `radiology_report.csv`, `icd_diagnosis.pkl` per stay.

---

## 1. Main Data Construction Pipeline (make_ldr_mimic)

### 1.1 Pair ICD Diagnoses and Lab Data
- **Script:** `step1_pair_icd_and_lab.py`
- **Description:**
    - Extracts ICD diagnoses (ICD-9/ICD-10) for each case and matches corresponding lab data during the hospital stay.
    - Generates paired data for different types (img_img, img_unimg, unimg).
- **Output:** Paired JSON and lab CSV files for each type.

### 1.2 Count Frequent Diseases
- **Script:** `step2_get_top_diseases.py`
- **Description:**
    - Counts the frequency of all ICD diagnoses across cases.
    - Selects the most frequent ICD codes for downstream use.
- **Output:** ICD code frequency list (CSV).

### 1.3 Prepare Dataset
- **Script:** `step3_prepare_data.py`
- **Description:**
    - Organizes input (lab, chief complaint, family history, etc.) and output (ICD diagnoses) into a standardized format for model training and evaluation.
- **Output:** Standardized JSON and CSV files for each data type.


### 1.4 Add Top-1 ICD Diagnosis and Enrich with Reports/Text

#### 1.4.1 Add Top-1 ICD Diagnosis
- **Script:** `step4_1_add_top1_icd.py`
- **Description:**
    - Adds the primary ICD diagnosis (seq_num=1) for each sample for further analysis and inference subset extraction.
- **Output:** JSON files with top-1 ICD info.

#### 1.4.2 Add Imaging Reports, Physical Exam, and More Text
- **Script:** `step4_2_add_report_physical_info.py`
- **Description:**
    - Adds imaging reports, physical exam, history of present illness, and other text information to each sample.
- **Output:** JSON files with multi-modal text fields.

### 1.5 Extract Inference Subset
- **Script:** `step6_extract_inference_subdataset.py`
- **Description:**
    - Samples a fixed number of cases per ICD from validation/test sets to create a subset for inference and evaluation.
- **Output:** Inference subset JSON.

---

**Note:**
- All steps must be run in order, starting from preprocessing, to ensure downstream scripts have the required input files.
- For details on parameters or running commands, refer to the comments in each script.
