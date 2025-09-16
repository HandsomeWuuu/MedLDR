import json
import pandas as pd
import re
import os
import sys

model_price = {
    'gemini_2_5_flash':{'input': 0.3 * 1.66, 'output': 2.5 * 1.66},  # USD/million tokens * exchange rate
    'gpt_5_mini':{'input': 0.25 * 1.66, 'output': 2 * 1.66},  # USD/million tokens * exchange rate
    'claude_sonnet_4':{'input': 3 * 1.66, 'output': 15 * 1.66},  # USD/million tokens * exchange rate
    'qwen_plus_think':{'input': 0.8, 'output': 8},  # USD/million tokens * exchange rate
    'qwen_plus_nothink':{'input': 0.8, 'output': 2},  # USD/million tokens * exchange rate
    'qwen3_30b':{'input': 0.75, 'output': 7.5},  # USD/million tokens * exchange rate
    'deepseek_v3':{'input': 0.27 * 7, 'output': 1.1 *7 },  # USD/million tokens * exchange rate
    'deepseek_r1':{'input': 0.55 * 7, 'output': 2.19 *7 },  # USD/million tokens * exchange rate
}

def extract_eicu_74_icd_codes():
    """Extract 74 ICD10 codes for the eICU task"""
    try:
        import sys
        sys.path.append('.')
        with open('process_eicu_data/diagnosis_icd_mapping.json', 'r') as f:
            data = json.load(f)
        icd10_codes = list(data['icd10_to_diagnosis_mapping'].keys())
        return icd10_codes
    except Exception as e:
        print(f"Error loading ICD codes: {e}")
        # Fallback to hardcoded list
        return [
            'E86.1', 'D64.9', 'R65.21', 'I50.9', 'J18.9', 'I63.50', 'I10', 'J44.9',
            'I48.0', 'A41.9', 'I95.9', 'I25.10', 'J45', 'J69.0', 'J96.00', 'N18.6',
            'E87.5', 'E10.1', 'E87.2', 'N17.9', 'E03.9', 'E78.5', 'J44.1', 'N39.0',
            'J91.8', 'J80', 'N18.9', 'J96.91', 'R73.9', 'R41.82', 'G93.41', 'I50.1',
            'K92.2', 'I46.9', 'R65.2', 'G93.40', 'E87.70', 'I67.8', 'D72.829', 'E83.42',
            'F32.9', 'I21.3', 'I21.4', 'N30.9', 'E66.9', 'R65.20', 'R40.0', 'R10.9',
            'D62', 'F43.0', 'R56.9', 'J96.92', 'F10.239', 'F05', 'R11.0', 'G47.33',
            'R50.9', 'J96.10', 'E46', 'R07.9', 'R00.0', 'E83.51', 'R57.0', 'E16.2',
            'N18.3', 'I62.9', 'J98.11', 'D68.32', 'D69.6', 'D68.9', 'F03', 'E87.6', 
            'F41.9','E87.0'
        ]

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def check_consistency(top20, predictions):
    """Check prediction consistency"""
    pred_num = len(predictions)
    if predictions == top20[:pred_num]:
        return True
    else:
        return False

def fix_json_format(json_str):
    """Fix common JSON format issues"""
    # 1. Remove inline comments (e.g. "D64.9", // Anemia of critical illness)
    json_str = re.sub(r'//.*?(?=\n|$)', '', json_str)
    # 2. Add quotes to ICD codes without quotes (e.g. I50.9 -> "I50.9")
    pattern = r'(\s*)([A-Z]\d{1,3}\.?\d*)(,|\s*[\]\}])'
    json_str = re.sub(pattern, r'\1"\2"\3', json_str)
    # 3. Remove trailing commas in arrays/objects
    json_str = re.sub(r',(\s*[\]\}])', r'\1', json_str)
    # 4. (Optional) Fix keys without quotes
    # json_str = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', json_str)
    # 5. Clean up extra whitespace
    json_str = re.sub(r'\s+', ' ', json_str)
    json_str = re.sub(r'\s*([,\[\]{}:])\s*', r'\1', json_str)
    return json_str

def check_invalid_case(response):
    """Check invalid response and try to extract JSON - enhanced fault tolerance"""
    # 1. Try to parse the whole response as JSON
    try:
        json_data = json.loads(response.strip())
        return [json.dumps(json_data)]
    except json.JSONDecodeError:
        pass
    # 2. Find JSON objects in response - look for last JSON (usually final answer)
    json_objects = []
    lines = response.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                json_data = json.loads(line)
                json_objects.append(line)
            except json.JSONDecodeError:
                pass
    if json_objects:
        return [json_objects[-1]]
    # 3. Greedy match for JSON object
    pattern = r"\{.*\}"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        longest_match = max(matches, key=len)
        fixed_json = fix_json_format(longest_match)
        try:
            json.loads(fixed_json)
            return [fixed_json]
        except json.JSONDecodeError as e:
            print(f"JSON still invalid after fix: {e}")
            print(f"Fixed JSON snippet: {fixed_json[:200]}...")
            try:
                json.loads(longest_match)
                return [longest_match]
            except json.JSONDecodeError:
                pass
    # 4. Use bracket balancing to extract JSON
    json_extracted = extract_balanced_json(response)
    if json_extracted:
        fixed_json = fix_json_format(json_extracted)
        try:
            json.loads(fixed_json)
            return [fixed_json]
        except json.JSONDecodeError:
            try:
                json.loads(json_extracted)
                return [json_extracted]
            except json.JSONDecodeError:
                pass
    # 5. Non-greedy match as last resort
    pattern = r"\{.*?\}"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches
    return None

def extract_balanced_json(text):
    """Extract complete JSON object using bracket balancing"""
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    brace_count = 0
    in_string = False
    escape_next = False
    for i, char in enumerate(text[start_idx:], start_idx):
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[start_idx:i+1]
                    fixed_json = fix_json_format(json_str)
                    try:
                        json.loads(fixed_json)
                        return fixed_json
                    except json.JSONDecodeError:
                        try:
                            json.loads(json_str)
                            return json_str
                        except json.JSONDecodeError:
                            continue
    return None

def extract_icd_code(result_list):
    """Extract ICD codes, handle possible colon in code"""
    if not result_list:
        return []
    icd_codes = []
    for icd in result_list:
        if ':' in str(icd):
            code = str(icd).split(':')[0].strip()
        else:
            code = str(icd).strip()
        icd_codes.append(code)
    return icd_codes

def extract_eicu_results(data, valid_icd_codes):
    """Extract eICU task results"""
    valid_list = []
    invalid_list = []
    for item in data:
        single_result = {}
        single_result['case_id'] = item['case_id']
        response = item['response']
        # Try to extract ```json code block first
        pattern = r"```json\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)
        if not matches:
            matches = check_invalid_case(response)
        if matches:
            try:
                if len(matches) == 1:
                    json_str = matches[0]
                else:
                    json_str = matches[-1]
                json_data = json.loads(json_str)
                top20 = json_data.get('top20', [])
                predictions = json_data.get('predictions', [])
                if len(top20) > 0 and len(predictions) > 0:
                    top20_icd = extract_icd_code(top20)
                    predictions_icd = extract_icd_code(predictions)
                    single_result['top20'] = top20_icd
                    single_result['predictions'] = predictions_icd
                    single_result['pred_consistency'] = check_consistency(top20_icd, predictions_icd)
                    single_result['top1_consistency'] = check_consistency(top20_icd, predictions_icd[:1])
                    valid_list.append(single_result)
                else:
                    print(f'Invalid case (empty results): {item["case_id"]}')
                    invalid_list.append(item)
            except json.JSONDecodeError as e:
                print(f'JSON decode error for case {item["case_id"]}: {e}, trying to fix...')
                try:
                    fixed_json = fix_json_format(json_str)
                    json_data = json.loads(fixed_json)
                    top20 = json_data.get('top20', [])
                    predictions = json_data.get('predictions', [])
                    if len(top20) > 0 and len(predictions) > 0:
                        top20_icd = extract_icd_code(top20)
                        predictions_icd = extract_icd_code(predictions)
                        single_result['top20'] = top20_icd
                        single_result['predictions'] = predictions_icd
                        single_result['pred_consistency'] = check_consistency(top20_icd, predictions_icd)
                        single_result['top1_consistency'] = check_consistency(top20_icd, predictions_icd[:1])
                        valid_list.append(single_result)
                        print(f'Successfully fixed and extracted case {item["case_id"]}')
                    else:
                        print(f'Fixed JSON but empty results for case: {item["case_id"]}')
                        invalid_list.append(item)
                except json.JSONDecodeError as e2:
                    print(f'JSON fix failed for case {item["case_id"]}: {e2}')
                    invalid_list.append(item)
                except Exception as e2:
                    print(f'Other error during JSON fix for case {item["case_id"]}: {e2}')
                    invalid_list.append(item)
            except Exception as e:
                print(f'Other error for case {item["case_id"]}: {e}')
                invalid_list.append(item)
        else:
            print(f'No JSON found for case: {item["case_id"]}')
            invalid_list.append(item)
    valid_df = pd.DataFrame(valid_list) if valid_list else pd.DataFrame()
    return valid_df, invalid_list

def validate_predictions(df, valid_icd_codes):
    """Validate predictions and count errors"""
    if df.empty:
        return df
    df['valid_prediction'] = df['predictions'].apply(
        lambda x: [icd for icd in x if icd in valid_icd_codes]
    )
    df['error_prediction_len'] = df['predictions'].apply(
        lambda x: len(x) - len([icd for icd in x if icd in valid_icd_codes])
    )
    return df

def save_to_csv(df, output_file):
    """Save as CSV file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8')

def save_to_jsonl(data, output_file):
    """Save as JSONL file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # Extract valid ICD code list
    valid_icd_codes = extract_eicu_74_icd_codes()
    print(f"Number of valid ICD codes: {len(valid_icd_codes)}")
    print(f"First 10 ICD codes: {valid_icd_codes[:10]}")

    
    # Input file path
    input_file = sys.argv[1] 
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: file does not exist {input_file}")
        return
    # Output file paths
    output_valid_file = os.path.join(os.path.dirname(input_file), 'processed_results', 'valid_results.csv')
    output_invalid_file = os.path.join(os.path.dirname(input_file), 'processed_results', 'invalid_cases.jsonl')
    # Load data
    data = load_jsonl(input_file)
    print(f'Total data count: {len(data)}')
    # Remove case_id == 643420 or 338999 (empty data)
    data = [item for item in data if item['case_id'] not in ["643420", "338999"]]
    print(f'Total after removing empty data: {len(data)}')
    # Extract results
    valid_df, invalid_list = extract_eicu_results(data, valid_icd_codes)
    print(f'Valid data count: {len(valid_df)}')
    print(f'Invalid data count: {len(invalid_list)}')
    if not valid_df.empty:
        # Check duplicate case_id
        duplicate_case_ids = valid_df[valid_df.duplicated('case_id', keep=False)]
        if not duplicate_case_ids.empty:
            print('Duplicate case_id found:')
            print(duplicate_case_ids[['case_id']].value_counts())
            # Keep last duplicate
            valid_df = valid_df.drop_duplicates(subset='case_id', keep='last')
            print(f'Valid data count after deduplication: {len(valid_df)}')
        # Consistency statistics
        consistency_true_ratio = valid_df['pred_consistency'].sum()
        top_1_consistency_true_ratio = valid_df['top1_consistency'].sum()
        print(f'Prediction consistency ratio: {consistency_true_ratio}/{len(valid_df)} = {consistency_true_ratio/len(valid_df):.3f}')
        print(f'Top1 consistency ratio: {top_1_consistency_true_ratio}/{len(valid_df)} = {top_1_consistency_true_ratio/len(valid_df):.3f}')
        # Validate predictions
        valid_df = validate_predictions(valid_df, valid_icd_codes)
        print(f'Total error predictions: {valid_df["error_prediction_len"].sum()}')
        print(f'Number of cases with error predictions: {valid_df[valid_df["error_prediction_len"] != 0].shape[0]}')
        # Save results
        save_to_csv(valid_df, output_valid_file)
        print(f'Valid results saved to: {output_valid_file}')
    if invalid_list:
        save_to_jsonl(invalid_list, output_invalid_file)
        print(f'Invalid results saved to: {output_invalid_file}')
    # Token usage statistics (if token info exists)
    if data and 'token_usage' in data[0]:
        print('\n=== Token Usage Statistics ===')
        token_data = []
        for item in data:
            single_case = {}
            single_case['case_id'] = item['case_id']
            single_case['estimate_prompt_tokens'] = item.get('estimate_token_count', {}).get('total', 0)
            single_case.update(item.get('token_usage', {}))
            token_data.append(single_case)
        token_df = pd.DataFrame(token_data)
        print(token_df.head())
        if not token_df.empty:
            model_name = 'gemini_2_5_flash'
            print(f"Token statistics summary:")
            print(token_df.describe())
            # Price calculation (example: Gemini 2.5 Flash)
            input_price_M = model_price[model_name]['input']
            output_price_M = model_price[model_name]['output']
            if 'prompt_tokens' in token_df.columns and 'completion_tokens' in token_df.columns and 'reasoning_tokens' not in token_df.columns:
                input_tokens = token_df['prompt_tokens'].sum() / 1000000
                output_tokens = token_df['completion_tokens'].sum() / 1000000
                print(f'Input tokens: {input_tokens:.3f}M, Price: {input_tokens * input_price_M:.2f} CNY')
                print(f'Output tokens: {output_tokens:.3f}M, Price: {output_tokens * output_price_M:.2f} CNY')
                print(f'Total price: {(input_tokens * input_price_M) + (output_tokens * output_price_M):.2f} CNY')
            elif 'prompt_tokens' in token_df.columns and 'completion_tokens' in token_df.columns and 'reasoning_tokens' in token_df.columns:
                input_tokens = token_df['prompt_tokens'].sum() / 1000000
                output_tokens = token_df['completion_tokens'].sum() / 1000000 + token_df['reasoning_tokens'].sum() / 1000000
                print(f'Input tokens: {input_tokens:.3f}M, Price: {input_tokens * input_price_M:.2f} CNY')
                print(f'Output tokens: {output_tokens:.3f}M, Price: {output_tokens * output_price_M:.2f} CNY')
                print(f'Total price: {(input_tokens * input_price_M) + (output_tokens * output_price_M):.2f} CNY')
            else:
                print("Token columns not found or incomplete in the data")
    # Remove invalid data from input_file using case_id
    if invalid_list:
        invalid_case_ids = {item['case_id'] for item in invalid_list}
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(input_file, 'w', encoding='utf-8') as f:
            for line in lines:
                item = json.loads(line.strip())
                if item['case_id'] not in invalid_case_ids:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'Removed {len(invalid_list)} invalid data entries')

if __name__ == "__main__":
    main()
