from process_data.config import Task_Mapping_Table
import tiktoken


tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


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


prompt_ds_distill = """As a medical diagnosis assistant, strictly follow the steps below:

1. Generate the top 10 diseases with the highest confidence from the 40 diseases listed below (use only ICD codes, do not include disease names), sorted in descending order of confidence.
   - Ensure all codes come from the given list.
   - Correct example format: ["5849", "34982", ...]

2. From the top 10, filter 1-3 diseases with extremely high confidence (codes only), ensuring:
   - Only retain diseases with confidence significantly higher than other options.
   - Maintain descending order.
   - Correct example format: ["42823", "51884"]

Strictly follow the JSON format below (violations will result in system errors):
{
    "top10": ["icd_code1", "icd_code2", ..., "icd_code10"],  // Must contain 10 strings
    "predictions": ["highest_code"]  // 1-3 strings
}

Do not include:
- Disease names or mixed formats (e.g., "Acute kidney failure")
- Object structures (e.g., {"icd_code": ...})
- Codes not in the list (e.g., ".77")
- Extra comments or unclosed code blocks"""


system_message = f"{prompt_ds_distill}\n{top_40_icd_code_and_name_distill}"



#  Organize prompt + various types of information + disease_icdCode_and_name content together to form the input, return the input content and token length

def organize_input_data(data_point, data_tpye):
    # patient info
    input_data = ''
    token_count = {'total': 0}

    input_key_list = Task_Mapping_Table[data_tpye]

    for key in input_key_list:
        value = data_point['input_dict'][key]
    
        if isinstance(value, list):
            value = 'The report info:\n' + '\n---\n'.join(value)
        
        input_data += f"\n{value}"
        token_count[key] = len(tokenizer.encode(value))

    token_count['total'] += sum(token_count[key] for key in input_key_list)

    return input_data, token_count

