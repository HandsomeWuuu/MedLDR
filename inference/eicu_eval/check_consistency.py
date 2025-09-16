import sys
import pandas as pd
import ast
import json
import os

#!/usr/bin/env python3

def compute_pair_consistency(dfA, dfB):
    top20_cons = {}
    # 1. Calculate top20 consistency
    for n in [1, 2, 3, 5, 10, 20]:
        ratios = []
        for idx, row in dfA.iterrows():
            case_id = row['case_id']
            listA = dfA[dfA['case_id'] == case_id]['top20'].values[0]
            listB = dfB[dfB['case_id'] == case_id]['top20'].values[0]
            
            topA = listA[:n]
            topB = listB[:n]
            # Calculate intersection over union for the first n elements
            inter_ratio = len(set(topA) & set(topB)) / len(set(topA) | set(topB))

            ratios.append(inter_ratio)
        top20_cons[f"top_{n}"] = round(sum(ratios) / len(ratios), 4) if ratios else 0.0
    
    pred_cons = {}
    # 2. Calculate prediction consistency - similar to top20 consistency, but compare top1 and all
    for col in ['top1', 'all']:
        ratios = []
        for idx, row in dfA.iterrows():
            case_id = row['case_id']
            listA = dfA[dfA['case_id'] == case_id]['valid_prediction'].values[0]
            listB = dfB[dfB['case_id'] == case_id]['valid_prediction'].values[0]
            
            # print(f"listA: {listA}, listB: {listB}")
            # print(f'dfA row: {row}')
            # Calculate intersection over union
            if col == 'top1':
                inter_ratio = len(set(listA[:1]) & set(listB[:1])) / 1
            else:
                inter_ratio = len(set(listA) & set(listB)) / len(set(listA) | set(listB))

            ratios.append(inter_ratio)
        pred_cons[col] = round(sum(ratios) / len(ratios), 4) if ratios else 0.0
    
    return {"top20": top20_cons, "prediction": pred_cons}

def compute_three_consistency(df1, df2, df3):
    top20_cons = {}
    # 1. Calculate top20 consistency
    for n in [1, 2, 3, 5, 10, 20]:
        ratios = []
        for idx, row in df1.iterrows():
            case_id = row['case_id']
            list1 = df1[df1['case_id'] == case_id]['top20'].values[0]
            list2 = df2[df2['case_id'] == case_id]['top20'].values[0]
            list3 = df3[df3['case_id'] == case_id]['top20'].values[0]
            
            top1 = list1[:n]
            top2 = list2[:n]
            top3 = list3[:n]
            # Calculate intersection over union for the first n elements
            inter_ratio = len(set(top1) & set(top2) & set(top3)) / len(set(top1) | set(top2) | set(top3))

            ratios.append(inter_ratio)
        top20_cons[f"top_{n}"] = round(sum(ratios) / len(ratios), 4) if ratios else 0.0
    
    pred_cons = {}
    # 2. Calculate prediction consistency - similar to top20 consistency, but compare top1 and all
    for col in ['top1', 'all']:
        ratios = []
        for idx, row in df1.iterrows():
            case_id = row['case_id']
            list1 = df1[df1['case_id'] == case_id]['valid_prediction'].values[0]
            list2 = df2[df2['case_id'] == case_id]['valid_prediction'].values[0]
            list3 = df3[df3['case_id'] == case_id]['valid_prediction'].values[0]
            
            if col == 'top1':
                inter_ratio = len(set(list1[:1]) & set(list2[:1]) & set(list3[:1])) / 1
            else:
                inter_ratio = len(set(list1) & set(list2) & set(list3)) / len(set(list1) | set(list2) | set(list3))

            ratios.append(inter_ratio)
        pred_cons[col] = round(sum(ratios) / len(ratios), 4) if ratios else 0.0
    
    return {"top20_cons": top20_cons, "prediction_cons": pred_cons}


def main(file1, file2, file3, save_path=None):
    # 1. Read 3 csv files
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    # Compare if case_id in df2 and df3 are consistent
    if not df2['case_id'].isin(df3['case_id']).all():
        missing_ids = df2.loc[~df2['case_id'].isin(df3['case_id']), 'case_id'].unique()
        print("The following case_id exist in df2 but not in df3:")
        print(missing_ids)
    
    print(f"case_id in df2 and df3 are consistent, both have {len(df2)} rows")

    df1 = pd.read_csv(file1)
    # Filter df1 to only include case_id in df2
    df1 = df1[df1['case_id'].isin(df2['case_id'])]

    df1['valid_prediction'] = df1['valid_prediction'].apply(lambda x: ast.literal_eval(x) if x else [])
    df1['top20'] = df1['top20'].apply(lambda x: ast.literal_eval(x) if x else [])
    df2['valid_prediction'] = df2['valid_prediction'].apply(lambda x: ast.literal_eval(x) if x else [])
    df2['top20'] = df2['top20'].apply(lambda x: ast.literal_eval(x) if x else [])
    df3['valid_prediction'] = df3['valid_prediction'].apply(lambda x: ast.literal_eval(x) if x else [])
    df3['top20'] = df3['top20'].apply(lambda x: ast.literal_eval(x) if x else [])

    print(f"df1 len {len(df1)} is consistent with df2 and df3, ready for comparison")

    assert df1.shape[0] == df2.shape[0] == df3.shape[0], "Row numbers are inconsistent, cannot compare"

    # Set columns to compare
    rounds = ["1 and 2", "1 and 3", "2 and 3", "1, 2, 3"]

    results_dict = {
        "1 and 2": compute_pair_consistency(df1, df2),
        "1 and 3": compute_pair_consistency(df1, df3),
        "2 and 3": compute_pair_consistency(df2, df3),
        "1, 2, 3": compute_three_consistency(df1, df2, df3)
    }

    print("Consistency Results:")
    for round in rounds:
        print(f"Round {round}:")
        print(results_dict[round])
        print("\n")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Check consistency of eicu results")
    parser.add_argument('--compare_model', type=str, required=True, help='Model to compare (claude-sonnet-4, qwen-plus, gemini_2_5_flash)')
    parser.add_argument('--think_mode', type=str, required=True, help='Think mode (think, nothink)')
    parser.add_argument('--temperature', type=float, required=True, help='Temperature value (0.1, 0.7, 1.0)')
    args = parser.parse_args()

    compare_model = args.compare_model # claude-sonnet-4, qwen-plus, gemini_2_5_flash, 

    think_mode = args.think_mode # think, nothink
    temperature = args.temperature # 0.1, 0.7, 1.0

    prompt_name = 'current_human_designed'
    prompt_type = '_'.join(prompt_name.split('_')[-2:]) # current_human_designed, current_automated_designed, current_human_automated_designed

    base_path = 'xxx/MedLDR/'

    print(f"compare_model: {compare_model}, think_mode: {think_mode}, temperature: {temperature}, prompt_name: {prompt_name}, prompt_type: {prompt_type}")

    if compare_model == 'gemini_2_5_flash':
        sub_path = f'all_cursorai_gemini_2_5_flash_{think_mode}/prompt_{prompt_type}/prompt_{prompt_name}_temp_{temperature}/processed_results/valid_results.csv'
        compare_list = [f'results_eicu/subset_100/round_4/{sub_path}',
                        f'results_eicu/subset_100/round_2/{sub_path}',
                        f'results_eicu/subset_100/round_3/{sub_path}']
    elif compare_model == 'claude-sonnet-4':
        sub_path = f'all_cursorai_claude_sonnet_4_{think_mode}/prompt_{prompt_type}/prompt_{prompt_name}_temp_{temperature}/processed_results/valid_results.csv'
        compare_list = [f'results_eicu/subset_100/round_4/{sub_path}',
                        f'results_eicu/subset_100/round_2/{sub_path}',
                        f'results_eicu/subset_100/round_3/{sub_path}']
    
    elif compare_model == 'qwen-plus':
        sub_path = f'all_official_qwen_plus_{think_mode}/prompt_{prompt_type}/prompt_{prompt_name}_temp_{temperature}/processed_results/valid_results.csv'
        compare_list = [f'results_eicu/subset_100/round_4/{sub_path}',
                        f'results_eicu/subset_100/round_2/{sub_path}',
                        f'results_eicu/subset_100/round_3/{sub_path}']
    elif compare_model == 'qwen3_30b_a3b':
        sub_path = f'all_official_qwen3_30b_a3b_{think_mode}/prompt_{prompt_type}/prompt_{prompt_name}_temp_{temperature}/processed_results/valid_results.csv'
        compare_list = [f'results_eicu/subset_100/round_4/{sub_path}',
                        f'results_eicu/subset_100/round_2/{sub_path}',
                        f'results_eicu/subset_100/round_3/{sub_path}']
    elif compare_model == 'gpt_5_mini':
        if think_mode == 'think':
            think_type = 'high'
        else:
            think_type = 'minimal'

        sub_path = f'all_cursorai_gpt_5_mini_{think_type}/prompt_{prompt_type}/prompt_{prompt_name}_temp_{temperature}/processed_results/valid_results.csv'
        compare_list = [f'results_eicu/subset_100/round_4/{sub_path}',
                        f'results_eicu/subset_100/round_2/{sub_path}',
                        f'results_eicu/subset_100/round_3/{sub_path}']
        
    elif compare_model == 'deepseek':
        sub_path = f'all_official_deepseek_{think_mode}/prompt_{prompt_type}/prompt_{prompt_name}_temp_{temperature}/processed_results/valid_results.csv'
        compare_list = [f'results_eicu/subset_100/round_4/{sub_path}',
                        f'results_eicu/subset_100/round_2/{sub_path}',
                        f'results_eicu/subset_100/round_3/{sub_path}']
    else:
        raise ValueError(f"Unsupported compare_model: {compare_model}")

    save_path = base_path + f'results_eicu/subset_100/consistency/{compare_model}/{think_mode}/prompt_{prompt_name}_temp_{temperature}/consistency_new.json'

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    main(base_path + compare_list[0], base_path + compare_list[1], base_path + compare_list[2], save_path)
