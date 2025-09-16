
from process_data.config import Task_Mapping_Table,cut_off_len_dict
import tiktoken
import os
encoding = tiktoken.get_encoding("cl100k_base")
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


def load_icd_code_and_name():
    """Load ICD code and disease name mapping from txt file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    txt_file_path = os.path.join(current_dir, 'icd10_disease_map.txt')
    
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    
    return content

def generate_icd_prompt():
    """Generate a prompt containing ICD codes and disease names"""
    icd_content = load_icd_code_and_name()
    
    prompt = "Available ICD-10 codes and their corresponding diseases (Note: only use these exact codes in the output):\n\n"
    prompt += icd_content
    prompt += "\n\nPlease select appropriate ICD-10 codes from the above list based on the patient's clinical data.\n"
    
    return prompt

top_74_icd_code_and_name = generate_icd_prompt()

# 1. prompt_expert_designed
prompt_expert_designed = """As a disease diagnosis expert, analyze the patient data and provide:

1. Top 20 most likely diseases (ICD codes only, ranked by confidence)
2. Final diagnosis: most likely diseases with highest confidence (may include codes not in top 20)

Output format:
{
    "top20": ["code1", "code2", ..., "code20"],
    "predictions": ["final_code1", "final_code2", ...]
}

Only use ICD codes from the provided disease list."""

# 2. prompt_severity_based
prompt_severity_based = """As an expert diagnostician, analyze the clinical data systematically:

- Identify key clinical findings and patient characteristics
- Consider appropriate differential diagnoses
- Rank diseases by clinical severity and evidence strength

Provide:
- Top 20 differential diagnoses (ICD codes only)
- Final diagnosis: most confident diagnoses based on clinical evidence

Output format:
{
    "top20": ["icd1", "icd2", ..., "icd20"],
    "predictions": ["primary_diagnosis", "secondary_diagnosis", ...]
}

Use only ICD codes from the provided list."""

# 3. CoT-Based 
prompt_cot_based = """As a disease diagnosis expert, let me analyze this clinical case step by step:

1. **Clinical Review**: What key findings do I see?
2. **Pattern Recognition**: What conditions do these findings suggest?
3. **Differential Ranking**: Which diagnoses are most likely?
4. **Final Assessment**: What are my most confident diagnoses?

Provide the following diagnostic output:
- top20: Top 20 most likely diseases (ICD codes only, ranked by confidence)
- predictions: Final diagnosis with highest confidence (may include codes not in top20)

Output format:
{
    "top20": ["code1", "code2", ..., "code20"],
    "predictions": ["most_confident_diagnosis1", "most_confident_diagnosis2", ...]
}

IMPORTANT: Use only ICD codes from the disease list provided. Ensure your output follows the exact JSON format above."""

# Combine the prompt and disease_icdCode_and_name into a single system message for different strategies
eicu_system_message_dict = {
    'expert_designed': f"{prompt_expert_designed}\n{top_74_icd_code_and_name}",
    'severity_based': f"{prompt_severity_based}\n{top_74_icd_code_and_name}",
    'cot_based': f"{prompt_cot_based}\n{top_74_icd_code_and_name}"
}


# 4. Combine the prompt, disease_icdCode_and_name, and various types of information into a single input, and return the input content along with the token length
# Various types of information: patient_info + past_history + physical_exam + lab_results
def organize_input_data(data_point, data_type):
    input_data = ''
    token_count = {'total': 0}
    lab_token_cut_off_len = 8168
    for key,value in data_point['input'].items():  
        if key == 'lab_results':
        
            lab_tokens = tokenizer.encode(value)
            if len(lab_tokens) > lab_token_cut_off_len:
          
                lab_tokens = lab_tokens[:lab_token_cut_off_len]

                value = tokenizer.decode(lab_tokens)
                print(f"Warning: lab_results exceed {lab_token_cut_off_len} tokens and have been truncated.")
            
             
        input_data += f"{value}\n"
        token_count[key] = len(tokenizer.encode(value))

    token_count['total'] += sum(token_count[key] for key in data_point['input'].keys())

    return input_data, token_count
