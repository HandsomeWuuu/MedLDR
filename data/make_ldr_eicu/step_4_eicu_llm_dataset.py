"""
Convert each patient's files in eicu_patient_datasets to string and organize into jsonl format for LLM inference

File processing as follows:
Input fields:
1. Read patient.csv file
   Required columns: gender, age -- format as text, e.g.: gender: Male, age: 65
2. post_pastHistory.csv file
   Use the pasthistorypath column, iterate each row, remove the prefix "notes/Progress Notes/Past History/", and join the rest with commas.
3. post_physicalExam.csv
   Use columns physicalexampath (split by "/" and take the last part as the exam name), physicalexamtext as the result, format as: physicalexam: [exam_name: result, ...]
4. post_lab.csv
    Use columns labresultoffset, labname, labresult, labmeasurenamesystem. No extra formatting, just convert the csv to string.

Output fields: load post_diagnosis.csv and organize as
- all_diseases: list of (name, icd9code, icd10code)
- primary_diseases: list of (name, icd9code, icd10code)
- major_diseases: list of (name, icd9code, icd10code)
- other_diseases: list of (name, icd9code, icd10code)
"""

import os
import pandas as pd
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_patient_info(patient_csv_path):
    """Process patient basic information"""
    try:
        df = pd.read_csv(patient_csv_path)
        if df.empty:
            return ""
        
        row = df.iloc[0]  # Get the first row of data
        gender = row.get('gender', 'Unknown')
        age = row.get('age', 'Unknown')
        
        return f"Patient demographics and basic information: gender: {gender}, age: {age}"
    except Exception as e:
        logger.error(f"Error processing patient info from {patient_csv_path}: {e}")
        return ""


def process_past_history(past_history_csv_path):
    """Process past medical history"""
    try:
        df = pd.read_csv(past_history_csv_path)
        if df.empty:
            return ""
        
        history_items = []
        for _, row in df.iterrows():
            path = row.get('pasthistorypath', '')
            if path.startswith('notes/Progress Notes/Past History/'):
                # Remove the prefix
                history_item = path.replace('notes/Progress Notes/Past History/', '')
                history_items.append(history_item)
        
        return f"Patient's medical history and past conditions: {', '.join(history_items)}"
    except Exception as e:
        logger.error(f"Error processing past history from {past_history_csv_path}: {e}")
        return ""


def process_physical_exam(physical_exam_csv_path):
    """Process physical examination"""
    try:
        df = pd.read_csv(physical_exam_csv_path)
        if df.empty:
            return ""
        
        exam_items = []
        for _, row in df.iterrows():
            path = row.get('physicalexampath', '')
            text = row.get('physicalexamtext', '')
            
            if path:
                # Get the last part of the path as the exam name
                exam_name = path.split('/')[-1]
                exam_items.append(f"{exam_name}: {text}")
        
        return f"Physical examination findings and clinical observations: [{', '.join(exam_items)}]"
    except Exception as e:
        logger.error(f"Error processing physical exam from {physical_exam_csv_path}: {e}")
        return ""


def process_lab_results(lab_csv_path):
    """Process laboratory test results"""
    try:
        df = pd.read_csv(lab_csv_path)
        if df.empty:
            return ""
        
        # Select required columns
        required_columns = ['labresultoffset', 'labname', 'labresult', 'labmeasurenamesystem']
        available_columns = [col for col in required_columns if col in df.columns]
        
        if not available_columns:
            return ""
        
        # Select existing columns and convert to string
        lab_df = df[available_columns]
        lab_csv_data = lab_df.to_csv(index=False)
        
        return f"Laboratory test results with timestamps and measurements:\n{lab_csv_data}"
    except Exception as e:
        logger.error(f"Error processing lab results from {lab_csv_path}: {e}")
        return ""


def parse_icd_codes(icd_string):
    """Parse ICD code string, separating ICD9 and ICD10 codes"""
    if pd.isna(icd_string) or not icd_string:
        return "", ""
    
    # Split ICD9 and ICD10 codes (format: "icd9_code, icd10_code")
    codes = icd_string.split(', ')
    icd9_code = codes[0].strip() if len(codes) > 0 else ""
    icd10_code = codes[1].strip() if len(codes) > 1 else ""
    
    return icd9_code, icd10_code


def process_diagnosis(diagnosis_csv_path):
    """Process diagnosis information"""
    try:
        df = pd.read_csv(diagnosis_csv_path)
        if df.empty:
            return {
                "all_diseases": [],
                "primary_diseases": [],
                "major_diseases": [],
                "other_diseases": []
            }
        
        all_diseases = []
        primary_diseases = []
        major_diseases = []
        other_diseases = []
        
        for _, row in df.iterrows():
            diagnosis_string = row.get('diagnosisstring', '')
            icd_codes = row.get('icd9code', '')
            priority = row.get('diagnosispriority', 'Other')
            
            # Parse ICD codes
            icd9_code, icd10_code = parse_icd_codes(icd_codes)
            
            disease_info = {
                "name": diagnosis_string,
                "icd9code": icd9_code,
                "icd10code": icd10_code
            }
            
            # Add to all diseases list
            all_diseases.append(disease_info)
            
            # Classify by priority
            if priority == 'Primary':
                primary_diseases.append(disease_info)
            elif priority == 'Major':
                major_diseases.append(disease_info)
            else:  # Other
                other_diseases.append(disease_info)
        
        return {
            "all_diseases": all_diseases,
            "primary_diseases": primary_diseases,
            "major_diseases": major_diseases,
            "other_diseases": other_diseases
        }
    except Exception as e:
        logger.error(f"Error processing diagnosis from {diagnosis_csv_path}: {e}")
        return {
            "all_diseases": [],
            "primary_diseases": [],
            "major_diseases": [],
            "other_diseases": []
        }


def process_patient_folder(patient_folder_path):
    """Process individual patient folder"""
    patient_id = os.path.basename(patient_folder_path)
    logger.info(f"Processing patient {patient_id}")
    
    # Define file paths
    patient_csv = os.path.join(patient_folder_path, 'patient.csv')
    past_history_csv = os.path.join(patient_folder_path, 'post_pastHistory.csv')
    physical_exam_csv = os.path.join(patient_folder_path, 'post_physicalExam.csv')
    lab_csv = os.path.join(patient_folder_path, 'post_lab.csv')
    diagnosis_csv = os.path.join(patient_folder_path, 'post_diagnosis.csv')
    
    # Process input fields
    patient_info = process_patient_info(patient_csv) if os.path.exists(patient_csv) else ""
    past_history = process_past_history(past_history_csv) if os.path.exists(past_history_csv) else ""
    physical_exam = process_physical_exam(physical_exam_csv) if os.path.exists(physical_exam_csv) else ""
    lab_results = process_lab_results(lab_csv) if os.path.exists(lab_csv) else ""
    
    # Process output fields
    diagnosis_info = process_diagnosis(diagnosis_csv) if os.path.exists(diagnosis_csv) else {
        "all_diseases": [],
        "primary_diseases": [],
        "major_diseases": [],
        "other_diseases": []
    }
    
    # Build input field structure
    input_data = {}
    
    if patient_info:
        input_data["patient_info"] = patient_info
    if past_history:
        input_data["past_history"] = past_history
    if physical_exam:
        input_data["physical_exam"] = physical_exam
    if lab_results:
        input_data["lab_results"] = lab_results
    
    # Build output field structure
    output_data = {
        "all_diseases": diagnosis_info["all_diseases"],
        "primary_diseases": diagnosis_info["primary_diseases"],
        "major_diseases": diagnosis_info["major_diseases"],
        "other_diseases": diagnosis_info["other_diseases"]
    }
    
    # Build result dictionary
    result = {
        "case_id": patient_id,
        "input": input_data,
        "output": output_data
    }
    
    return result


def main():
    """Main function"""
    # Set input and output paths
    input_dir = "/Users/wushuai/Documents/research/dataset/eicu_patient_datasets"
    output_file = "/Users/wushuai/Documents/research/dataset/eicu_extracted_data/eicu_llm_dataset.json"

    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    # Get all patient folders
    patient_folders = [
        os.path.join(input_dir, folder_name)
        for folder_name in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, folder_name)) 
        and folder_name.isdigit()  # Only process folders named with digits
    ]
    
    logger.info(f"Found {len(patient_folders)} patient folders")
    
    # Process all patients and save as JSON array
    processed_count = 0
    all_results = []
    
    for patient_folder in patient_folders:
        try:
            result = process_patient_folder(patient_folder)
            all_results.append(result)
            processed_count += 1
            
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count} patients")
                
        except Exception as e:
            logger.error(f"Error processing {patient_folder}: {e}")
            continue
    
    # Write all results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Processing completed. Total processed: {processed_count} patients")
    logger.info(f"Output saved to: {output_file}")


if __name__ == "__main__":
    main()




