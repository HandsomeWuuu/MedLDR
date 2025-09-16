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


prompt_strict_expert_designed = """As a medical diagnosis assistant, strictly follow the steps below:

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

prompt_standard_expert_designed = """Medical Diagnosis Task:

Given patient data, predict the most likely diseases from the provided ICD code list.

Instructions:
1. Analyze patient symptoms, lab results, and clinical data
2. Select top 10 most probable diseases (ICD codes only)
3. Choose 1-3 highest confidence predictions as final diagnoses
4. Return results in JSON format

Output format:
{
    "top10": ["code1", "code2", ..., "code10"],
    "predictions": ["primary_code"]
}

Only use codes from the disease list below."""

prompt_conservative_expert_designed = """Medical Diagnosis Task:

Given patient data, predict the most likely diseases from the provided ICD code list.

Instructions:
1. Analyze patient symptoms, lab results, and clinical data
2. Select top 10 most probable diseases (ICD codes only)
3. Choose 1-3 highest confidence predictions as final diagnoses (must include at least 1)
4. Return results in JSON format

**Conservative Diagnostic Principles:**
- Only include diseases with STRONG supporting evidence
- Avoid diagnoses based on single symptoms or weak associations
- Consider "ruling out" vs "ruling in" - be selective in final predictions

**Evidence Requirements:**
- For predictions: Sufficient supporting clinical indicators required 
- For top10: At least one clear supporting finding required
- Avoid speculative diagnoses without clear clinical correlation

Output format:
{
    "top10": ["code1", "code2", ..., "code10"],
    "predictions": ["primary_code"]  // Must include at least 1, only include if highly confident
}

Only use codes from the disease list below (Only provide the code and do not need the name of the disease). If confidence is low, predictions may contain fewer than 3 codes, but must contain at least 1."""



prompt_severity_based_low = """As an expert medical diagnostician, analyze the patient data using clinical reasoning principles:

**Clinical Analysis Framework:**
1. **Symptom-Disease Mapping**: Identify key symptoms and their associated differential diagnoses
2. **Pattern Recognition**: Look for disease patterns, clusters, and typical presentations
3. **Risk Stratification**: Consider patient demographics, comorbidities, and severity indicators
4. **Differential Diagnosis**: Apply systematic exclusion and inclusion criteria

**Diagnostic Process:**
- First, identify the most likely disease category (cardiac, renal, infectious, etc.)
- Then narrow down to specific conditions within that category
- Consider both primary diagnoses and complications
- Prioritize based on clinical severity and treatment urgency

**Output Requirements:**
Return exactly this JSON format:
{
    "top10": ["icd1", "icd2", "icd3", "icd4", "icd5", "icd6", "icd7", "icd8", "icd9", "icd10"],
    "predictions": ["primary_icd", "secondary_icd"]
}

Only use ICD codes from the provided list. Ensure top10 contains exactly 10 codes, predictions contains 1-3 codes."""

prompt_severity_based_moderate = """Emergency medicine diagnosis - prioritize by clinical urgency:

**Triage Priorities:**
1. Life-threatening: sepsis, MI, stroke, respiratory failure (treat immediately)
2. Urgent: organ dysfunction, severe bleeding, infections (treat rapidly)  
3. Stable: chronic conditions, supportive care (monitor/treat routinely)

**Analysis Steps:**
- Identify emergency conditions first
- Check for organ failure signs
- Consider treatment urgency
- Rank by severity + probability

**CRITICAL: Output must be valid JSON only:**
{
    "top10": ["icd1", "icd2", "icd3", "icd4", "icd5", "icd6", "icd7", "icd8", "icd9", "icd10"],
    "predictions": ["urgent_icd1", "urgent_icd2"]
}

Use only provided ICD codes. Rank top10 by urgency + probability (HIGH to LOW). Select 1-3 most critical for predictions."""


prompt_severity_based_high = """As an expert emergency medicine physician, prioritize diagnoses by clinical severity and urgency:

**Triage-Based Diagnostic Framework:**
1. **Immediate Life-Threatening Conditions** (Priority 1):
   - Cardiac arrest, acute MI, severe sepsis, stroke, respiratory failure
   - Require immediate intervention
2. **Urgent Conditions** (Priority 2):
   - Organ dysfunction, significant bleeding, severe infections
   - Need rapid diagnosis and treatment
3. **Less Urgent Conditions** (Priority 3):
   - Chronic conditions, stable presentations, supportive care needs

**Clinical Severity Indicators:**
- Vital sign instability (hypotension, tachycardia, hypoxia)
- Altered mental status or neurological deficits
- Laboratory markers of organ failure (elevated creatinine, lactate)
- Evidence of systemic illness or sepsis

**Diagnostic Strategy:**
- Always consider worst-case scenarios first
- Rule out reversible causes of clinical deterioration
- Focus on conditions requiring immediate treatment
- Balance sensitivity vs specificity based on clinical context

**Output Requirements:**
- **top10**: List exactly 10 ICD codes ranked by CLINICAL URGENCY and treatment priority
- **predictions**: Select 1-3 most critical diagnoses

**Output Format:**
{
    "top10": ["most_urgent_icd1", "urgent_icd2", "icd3", "icd4", "icd5", "icd6", "icd7", "icd8", "icd9", "stable_icd10"],
    "predictions": ["highest_priority_urgent_icd", "secondary_urgent_icd"]
}

Only use ICD codes from the provided list. Remember: This is SEVERITY-BASED ranking, not just probability."""


# 3. CoT-based (链式思维推理版本)
prompt_strict_cot_based = """As a professional medical diagnosis assistant, let's work through this medical diagnosis step by step.

**Step 1: Information Analysis**
First, I'll examine the patient's clinical data systematically:
- Laboratory values and abnormalities
- Vital signs and physical findings  
- Symptoms and chief complaints
- Medical history and risk factors

**Step 2: Pattern Recognition**
Next, I'll identify clinical patterns:
- What symptoms cluster together?
- Which lab values suggest specific organ involved?
- Are there signs of acute vs chronic conditions?
- What age/demographic considerations apply?

**Step 3: Differential Diagnosis Formation**
Now I'll generate potential diagnoses:
- List conditions that could explain the presentation
- Consider common diagnoses first (horses not zebras)
- Include both primary conditions and complications
- Think about system-based differential diagnoses

**Step 4: Evidence Weighing**
For each potential diagnosis, I'll evaluate:
- How well does it explain the clinical picture?
- What supporting evidence exists?
- What evidence argues against it?
- How severe/urgent is this condition?

**Step 5: Final Ranking**
Based on my analysis, I'll rank the most likely diagnoses considering:
- Strength of supporting evidence
- Clinical probability
- Potential for serious complications
- Treatment implications

**Output Format:**
{
    "reasoning_summary": "Brief explanation of my diagnostic reasoning",
    "top10": ["code1", "code2", ..., "code10"],
    "predictions": ["most_likely_code1", "most_likely_code2"]
}

Remember: Only use ICD codes from the provided disease list."""



prompt_standard_cot_based = """Think through this medical case step by step, then provide your diagnosis.

Consider: What are the key clinical findings? What conditions could explain them? Which diagnoses have the strongest evidence?

**Output Requirements:**
- top10: List exactly 10 most likely ICD codes in descending order of probability
- predictions: Select 1-3 final diagnoses with highest confidence from the top10

**Output Format:**
{
    "reasoning": "Brief explanation of your diagnostic thinking",
    "top10": ["code1", "code2", ..., "code10"],  // Must contain exactly 10 codes
    "predictions": ["most_likely_code"]  // 1-3 final diagnoses with highest confidence
}

Use only the provided ICD codes."""


prompt_conservative_cot_based = """Think through this medical case step by step, then provide your diagnosis.

**Step-by-step reasoning:**
- What are the key clinical findings and abnormal values?
- Which conditions could explain these findings?
- What evidence supports each potential diagnosis?
- Which diagnoses have the strongest clinical correlation?

**Diagnostic considerations:**
- Require sufficient supporting findings for each diagnosis (not just one symptom)
- Only include predictions with strong, consistent clinical evidence
- Avoid diagnoses based on marginal lab values or isolated symptoms

**Output Requirements:**
- top10: List exactly 10 most likely ICD codes in descending order of probability
- predictions: Select 1-3 final diagnoses with highest confidence from the top10

**Output Format:**
{
    "reasoning": "Brief explanation of your diagnostic thinking",
    "top10": ["code1", "code2", ..., "code10"],  // Must contain exactly 10 codes
    "predictions": ["most_likely_code"]  // 1-3 final diagnoses with highest confidence
}

Use only the provided ICD codes."""

# system_message = f"{prompt_ds_distill}\n{top_40_icd_code_and_name_distill}"
system_message_dict = {
    'strict_expert_designed': f"{prompt_strict_expert_designed}\n{top_40_icd_code_and_name_distill}",
    'standard_expert_designed': f"{prompt_standard_expert_designed}\n{top_40_icd_code_and_name_distill}",
    'conservative_expert_designed': f"{prompt_conservative_expert_designed}\n{top_40_icd_code_and_name_distill}",

    'severity_based_low': f"{prompt_severity_based_low}\n{top_40_icd_code_and_name_distill}",
    'severity_based_moderate': f"{prompt_severity_based_moderate}\n{top_40_icd_code_and_name_distill}",
    'severity_based_high': f"{prompt_severity_based_high}\n{top_40_icd_code_and_name_distill}",

    'strict_cot_based': f"{prompt_strict_cot_based}\n{top_40_icd_code_and_name_distill}",
    'standard_cot_based': f"{prompt_standard_cot_based}\n{top_40_icd_code_and_name_distill}",
    'conservative_cot_based': f"{prompt_conservative_cot_based}\n{top_40_icd_code_and_name_distill}"
}



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

