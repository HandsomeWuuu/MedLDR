'''
读取 examples/eicu_dataset/eicu_llm_sample_100_subset.json 文件，遍历所有的 case, 将  "primary_diseases": [],的数据去掉，保存到 examples/eicu_dataset/eicu_llm_sample_100_subset_no_empty_primary_diseases.json 文件中

看看删除了多少 case (N个)
然后去 /grp01/cs_yzyu/wushuai/model/llama_demo/infer_api/infer_api/dataset/eicu_ldr/eicu_llm_dataset.json 文件中，去里面抓N个 case (case_id 不在 sample 100 里, 且 primary_diseases 不为空) , 添加到 examples/eicu_dataset/eicu_llm_sample_100_subset_no_empty_primary_diseases.json 文件中，覆盖保存
'''
import json
import os

def remove_empty_primary_diseases(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 过滤掉 primary_diseases 为空的 case
    print('data case', data[0].keys())
    print(f"Total cases before filtering: {len(data)}") 
    filtered = [case for case in data if len(case.get('output', {}).get('primary_diseases', [])) > 0]
    N = len(data) - len(filtered)
    print(f"Total cases after filtering: {len(filtered)}") 
    print(f"Removed cases: {N}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    return filtered, N, set([case.get('case_id') for case in data])

def supplement_cases(filtered_path, N, sample_case_ids, supplement_path):
    # 读取补充数据集
    with open(supplement_path, 'r', encoding='utf-8') as f:
        all_cases = json.load(f)
    # 过滤出 case_id 不在 sample_case_ids 且 primary_diseases 不为空的 case
    candidates = [case for case in all_cases if (case.get('case_id') not in sample_case_ids and len(case.get('output', {}).get('primary_diseases', [])) > 0)]
    print(f"Available candidates for supplement: {len(candidates)}")
    # 取前N个
    supplement = candidates[:N]
    print(f"Supplemented cases: {len(supplement)}")
    # 合并并保存
    with open(filtered_path, 'r', encoding='utf-8') as f:
        filtered_cases = json.load(f)
    merged = filtered_cases + supplement
    with open(filtered_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Final total cases: {len(merged)}")

if __name__ == '__main__':
    input_file = 'examples/eicu_dataset/eicu_llm_sample_100_subset.json'
    output_file = 'examples/eicu_dataset/eicu_llm_sample_100_subset_no_empty_primary_diseases.json'
    supplement_file = '/grp01/cs_yzyu/wushuai/model/llama_demo/infer_api/infer_api/dataset/eicu_ldr/eicu_llm_dataset.json'
    filtered, N, sample_case_ids = remove_empty_primary_diseases(input_file, output_file)
    if N > 0:
        supplement_cases(output_file, N, sample_case_ids, supplement_file)