from open_r1.configs import Task_Mapping_Table,cut_off_len_dict
import tiktoken
from typing import Union

# encoding = tiktoken.get_encoding("cl100k_base")
# tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


# top5_Task_Prompt  = "Please select the answer from the following diseases, each disease is composed of icd_id:icd_name, and output the corresponding icd_id:"
top50_Task_Prompt  = "You are a medical diagnosis assistant system. Based on the patient's visit information, please analyze the case and select the 1 to 3 most likely diagnoses from the provided list of 50 diseases."

top50_Task_Prompt_distill = """\
Act as a clinical coding assistant. Based on the patient's medical records, \
output ONLY the top 1-3 most probable ICD-10 codes from the provided disease list. \
Strictly format as valid JSON with a "diagnoses" array containing codes ordered by \
descending probability. Use only codes from the predefined 50-disease ICD-10 list. \
Omit all explanatory text. Example output format: {"diagnoses": ["code1", "code2", ...]}"""

reasoning_prompt = "You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"

top_50_icd_code_and_name = "Each disease corresponds to the format icd_code:icd_name \
 [ 34830:Encephalopathy, unspecified; 42833:Acute on chronic diastolic heart failure; 4589:Hypotension, unspecified; 311:Depressive disorder, not elsewhere classified; 5990:Urinary tract infection, site not specified; 6827:Cellulitis and abscess of foot, except toes; 2724:Other and unspecified hyperlipidemia; \
 43411:Cerebral embolism with cerebral infarction; 40491:Hypertensive heart and chronic kidney disease, unspecified, with heart failure and with chronic kidney disease stage I through stage IV, or unspecified; 2762:Acidosis; 7802:Syncope and collapse; 99591:Sepsis; 41071:Subendocardial infarction, initial episode of care; \
 42822:Chronic systolic heart failure; 28419:Other pancytopenia; 42823:Acute on chronic systolic heart failure; 27651:Dehydration; 99592:Severe sepsis; 25080:Diabetes with other specified manifestations, type II or unspecified type, not stated as uncontrolled; 4280:Congestive heart failure, unspecified; 51881:Acute respiratory failure; \
 5856:End stage renal disease; 42832:Chronic diastolic heart failure; 2761:Hyposmolality and/or hyponatremia; 25000:Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled; 25040:Diabetes with renal manifestations, type II or unspecified type, not stated as uncontrolled; 34831:Metabolic encephalopathy; \
 5849:Acute kidney failure, unspecified; 5070:Pneumonitis due to inhalation of food or vomitus; 2851:Acute posthemorrhagic anemia; 262:Other severe protein-calorie malnutrition; 40291:Unspecified hypertensive heart disease with heart failure; 5762:Obstruction of bile duct; 5770:Acute pancreatitis; 431:Intracerebral hemorrhage; 4019:Unspecified essential hypertension; \
 486:Pneumonia, organism unspecified; 42731:Atrial fibrillation; 3485:Cerebral edema; 5845:Acute kidney failure with lesion of tubular necrosis; 51884:Acute and chronic respiratory failure; 43491:Cerebral artery occlusion, unspecified with cerebral infarction; 41401:Coronary atherosclerosis of native coronary artery; 99859:Other postoperative infection; \
 41519:Other pulmonary embolism and infarction; 2875:Thrombocytopenia, unspecified; 34982:Toxic encephalopathy; 40391:Hypertensive chronic kidney disease, unspecified, with chronic kidney disease stage V or end stage renal disease; 29181:Alcohol withdrawal]"


top_40_icd_code_and_name_distill = "Each disease corresponds to the format icd_code:icd_name (Note: only use codes in the output):\n\
[ 5849:Acute kidney failure, unspecified; 34982:Toxic encephalopathy;\
42823:Acute on chronic systolic heart failure; 262:Other severe protein-calorie malnutrition;\
43491:Cerebral artery occlusion, unspecified with cerebral infarction; 29181:Alcohol withdrawal;\
43411:Cerebral embolism with cerebral infarction; 5990:Urinary tract infection, site not specified;\
99591:Sepsis; 40491:Hypertensive heart and chronic kidney disease, unspecified, with heart failure and with chronic kidney disease stage I through stage IV, or unspecified; \
40291:Unspecified hypertensive heart disease with heart failure; 5770:Acute pancreatitis; 486:Pneumonia, organism unspecified; 99859:Other postoperative infection; \
431:Intracerebral hemorrhage; 34830:Encephalopathy, unspecified; 40391:Hypertensive chronic kidney disease, unspecified, with chronic kidney disease stage V or end stage renal disease; \
51881:Acute respiratory failure; 5070:Pneumonitis due to inhalation of food or vomitus; 0389:Unspecified septicemia; 41071:Subendocardial infarction, initial episode of care; 28419:Other pancytopenia; \
41401:Coronary atherosclerosis of native coronary artery; 7802:Syncope and collapse; 25080:Diabetes with other specified manifestations, type II or unspecified type, not stated as uncontrolled;\
42731:Atrial fibrillation; 27651:Dehydration; 4589:Hypotension, unspecified; 2851:Acute posthemorrhagic anemia; 5762:Obstruction of bile duct; 42833:Acute on chronic diastolic heart failure;\
34831:Metabolic encephalopathy; 2762:Acidosis; 4019:Unspecified essential hypertension; 6827:Cellulitis and abscess of foot, except toes; 2761:Hyposmolality and/or hyponatremia; 5845:Acute kidney failure with lesion of tubular necrosis; \
51884:Acute and chronic respiratory failure; 311:Depressive disorder, not elsewhere classified; 41519:Other pulmonary embolism and infarction ]"

top50_system_message = f"{top50_Task_Prompt_distill}\n{top_50_icd_code_and_name}"
top50_reasoning_system_message = f"{top50_Task_Prompt_distill}\n{top_50_icd_code_and_name}\n{reasoning_prompt}\n"

top40_system_message = f"{top50_Task_Prompt_distill}\n{top_40_icd_code_and_name_distill}"

# top40_system_message = f"{prompt_ds_distill}\n{top_40_icd_code_and_name_distill}"


# token_lengths = cut_off_len_dict

# # 4. 将 prompt + 各种种类信息 + disease_icdCode_and_name 的内容组织到一起形成输入，返回 输入的内容以及 token 长度
# def organize_input_data(data_point, data_tpye):
#     # patient info
#     input_data = ''
#     token_count = {'total': 0}

#     input_key_list = Task_Mapping_Table[data_tpye]

#     for key in input_key_list:
#         # print('key:', key)
#         value = data_point['input_dict'][key]
#         # print('value', value)
#         # if len(value.split(' ')) > token_lengths[key]:
#         #     print(f"Warning: {data_point['case_id']} {key} is too long. Truncating to {token_lengths[key]} tokens.")
#         #     value = ' '.join(value.split(' ')[:token_lengths[key]]) + '...'
#         if isinstance(value, list):
#             value = 'The report info:\n' + '\n---\n'.join(value)
        
#         input_data += f"\n{value}"
#         token_count[key] = len(tokenizer.encode(value))

#     token_count['total'] += sum(token_count[key] for key in input_key_list)

#     return input_data, token_count



# organized_data, token_length = organize_input_data()




class Prompter(object):
    
    def generate_prompt(
        self,
        instruction: str,
        label: Union[None, str] = None,
    ) -> str:     

        res = f"{instruction}\nAnswer: "
               
        if label:
            res = f"{res}{label}"
         
        return res


    def get_response(self, output: str) -> str:
        return output.split("Answer:")[1].strip().replace("/", "\u00F7").replace("*", "\u00D7")
        # return output