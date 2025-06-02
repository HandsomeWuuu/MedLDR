import os
import json

def check_json_files():
    # Define the directory path
    directory = "/grp01/cs_yzyu/wushuai/model/llama_demo/datasets/mimic_multi_inputs_icd/split_by_patient_seed1_report_hpi"

    # List all JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    # Load and count each JSON file
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # If data is a list, get its length
            if isinstance(data, list):
                count = len(data)
            # If data is a dictionary, count its keys
            elif isinstance(data, dict):
                count = len(data)
            else:
                count = 1
                
        print(f"File: {json_file} - Count: {count}")
    
check_json_files()