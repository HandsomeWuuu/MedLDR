# eICU Data Processing Pipeline Documentation

This document describes the complete data processing pipeline for preparing eICU patient data for large language model (LLM) inference. The pipeline consists of four main scripts, each responsible for a specific stage of data extraction, cleaning, structuring, and formatting.

---

## 1. `eicu_data_extraction.py`

**Purpose:**
- Extract the top N most frequent diseases (typically top 100) from the original eICU diagnosis data.
- Strictly filter and select high-quality patients whose diagnoses are all within the top diseases and meet priority requirements.
- Sample a fixed number of cases for each disease.
- Output a sample case list and a patient-disease dictionary for downstream processing.

**Key Outputs:**
- `eicu_extracted_data/eicu_sample_cases.json`: List of sampled patient cases.

---

## 2. `eicu_build_dataset.py`

**Purpose:**
- For each sampled patient, extract all relevant raw data from the original eICU tables (e.g., patient info, diagnosis, labs, history, etc.).
- Create a dedicated folder for each patient, containing all their raw data files and case information.

**Key Outputs:**
- `eicu_patient_datasets/<patient_id>/`: One folder per patient, containing all raw CSVs and metadata.

---

## 3. `eicu_preprocess.py`

**Purpose:**
- Clean and structure each patient's raw data files (diagnosis, past history, physical exam, labs, etc.).
- Remove duplicates, aggregate measurements, and filter invalid or redundant records.
- Output structured `post_*.csv` files for each patient, ready for downstream analysis or modeling.

**Key Outputs:**
- `eicu_patient_datasets/<patient_id>/post_*.csv`: Cleaned and structured data files for each patient.

---

## 4. `eicu_llm_dataset.py` / `step_4_eicu_llm_dataset.py`

**Purpose:**
- Convert each patient's structured data into a unified input-output format suitable for LLM inference or fine-tuning.
- For each patient, generate a dictionary with:
  - **Input:**
    - Demographics (gender, age)
    - Past medical history (as a comma-separated string)
    - Physical exam findings (as a list of name: result pairs)
    - Laboratory results (as a CSV string)
  - **Output:**
    - All diagnoses, and diagnoses grouped by priority (primary, major, other), each as a list of (name, icd9code, icd10code) tuples.
- Aggregate all patient samples into a single JSON file for LLM use.

**Key Outputs:**
- `eicu_extracted_data/eicu_llm_dataset.json`: The final dataset for LLM tasks.

---

## **Overall Workflow**

1. **Sample Extraction:**
   - Run `eicu_data_extraction.py` to select high-quality patients and sample cases.
2. **Raw Data Extraction:**
   - Run `eicu_build_dataset.py` to extract all raw data for each sampled patient.
3. **Data Cleaning & Structuring:**
   - Run `eicu_preprocess.py` to clean and structure each patient's data files.
4. **LLM Dataset Construction:**
   - Run `eicu_llm_dataset.py` (or `step_4_eicu_llm_dataset.py`) to generate the final LLM-ready dataset.

---

## **Directory Structure Overview**

```
/your_workspace/
├── eicu_data_extraction.py
├── eicu_build_dataset.py
├── eicu_preprocess.py
├── eicu_llm_dataset.py
├── eicu_extracted_data/
│   ├── eicu_sample_cases.json
│   └── eicu_llm_dataset.json
├── eicu_patient_datasets/
│   └── <patient_id>/
│       ├── patient.csv
│       ├── diagnosis.csv
│       ├── post_diagnosis.csv
│       ├── post_lab.csv
│       └── ...
└── ...
```

---

## **Notes**
- Each script is modular and can be run independently, but the recommended order is as described above.
- The pipeline is designed for reproducibility and scalability for large-scale medical data processing and LLM applications.



