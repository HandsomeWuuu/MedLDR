import json
import random
from collections import Counter
import os
import pandas as pd

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def split_data(data, test_size=2000):
    random.shuffle(data)
    test_data = data[:test_size]
    train_data = data[test_size:]
    return train_data, test_data

def count_icd_distribution(data):

    icd_codes = [icd for item in data for icd in item['output_dict']['output_icd_id']]
    # 打印 ICD 代码的种类
    print(f"ICD codes length: {len(set(icd_codes))}")
    
    return Counter(icd_codes)

def plt_icd_distribution(train_dict, val_dict, test_dict, output_path):
    import numpy as np
    import matplotlib.pyplot as plt

    # Combine keys from all dictionaries
    keys = list(set(train_dict.keys()).union(set(val_dict.keys())).union(set(test_dict.keys())))
    keys.sort(key=lambda k: train_dict.get(k, 0), reverse=True)

    # Get values for train, val, and test dictionaries
    train_values = [train_dict.get(key, 0) for key in keys]
    val_values = [val_dict.get(key, 0) for key in keys]
    test_values = [test_dict.get(key, 0) for key in keys]

    # Set the positions and width for the bars
    x = np.arange(len(keys))
    width = 0.25

    # Plot the bars
    fig, ax = plt.subplots(figsize=(15, 8))
    bars1 = ax.bar(x - width, train_values, width, label='Train')
    bars2 = ax.bar(x, val_values, width, label='Validation')
    bars3 = ax.bar(x + width, test_values, width, label='Test')

    # Add some text for labels, title and axes ticks
    ax.set_xlabel('ICD Codes')
    ax.set_ylabel('Counts')
    ax.set_title('Distribution of ICD Codes in Train, Validation, and Test Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=90)
    ax.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'icd_code_distribution.png'))
    plt.show()

def split_by_type(file_paths):

    for file_path in file_paths:
        data = load_json(file_path)
        train_data, test_data = split_data(data)
        all_train_data.extend(train_data)
        all_test_data.extend(test_data)

    return all_train_data, all_test_data

def split_by_patient(file_paths):
    # 固定随机种子
    random.seed(2025)

    all_data = []
    for file_path in file_paths:
        data = load_json(file_path)
        all_data.extend(data)

    print(f"Combined data length: {len(all_data)}")
    # Count the distribution of subjects
    subject_counts = Counter(item["case_id"].split('/')[0] for item in all_data)
    print(f"Number of subjects: {len(subject_counts)}")
    # print(f"Subject distribution: {subject_counts}")
    subjects = list(subject_counts.keys())
    random.shuffle(subjects)

    # Split subjects into train, test, and validation sets
    num_subjects = len(subjects)
    train_subjects = subjects[:int(0.8 * num_subjects)]
    test_subjects = subjects[int(0.8 * num_subjects):int(0.9 * num_subjects)]
    val_subjects = subjects[int(0.9 * num_subjects):]

    # Split data based on subject sets
    train_data = [item for item in all_data if item["case_id"].split('/')[0] in train_subjects]
    test_data = [item for item in all_data if item["case_id"].split('/')[0] in test_subjects]
    val_data = [item for item in all_data if item["case_id"].split('/')[0] in val_subjects]

    print(f"Training set length: {len(train_data)}")
    print(f"Validation set length: {len(val_data)}")
    print(f"Testing set length: {len(test_data)}")

    return train_data, val_data, test_data, 



def main():
   
    base_path = '/xxx/Multi-modal/MMCaD_code2/data/pair_multi_input_icd/datasets'

    file_paths = [
        f'{base_path}/top50_img/img_pairs.json',
        f'{base_path}/top50_img_unimg/img_unimg_pairs.json',
        f'{base_path}/top50_unimg/unimg_pairs.json'
    ]

    all_train_data = []
    all_val_data = []
    all_test_data = []

    # all_train_data, all_test_data = split_by_type(file_paths)
    all_train_data, all_val_data, all_test_data = split_by_patient(file_paths)

    # 打印训练集和测试集的 ICD 分布
    train_distribution = count_icd_distribution(all_train_data)
    val_distribution = count_icd_distribution(all_val_data)
    test_distribution = count_icd_distribution(all_test_data)

    # Combine distributions into a single DataFrame
    combined_distribution = pd.DataFrame({
        'ICD Code': list(train_distribution.keys()),
        'Train Count': list(train_distribution.values()),
        'Validation Count': [val_distribution.get(icd, 0) for icd in train_distribution.keys()],
        'Test Count': [test_distribution.get(icd, 0) for icd in train_distribution.keys()]
    })

    # Sort the DataFrame by Train Count in descending order
    combined_distribution = combined_distribution.sort_values(by='Train Count', ascending=False)

    # Print the combined distribution
    print("Combined ICD distribution:")
    print(combined_distribution)

    # 遍历 train_distribution 中的 icd 
    core_mimiciv_path = '/xxx/' + 'data/physionet.org/files/mimiciv/2.2/'
    icd_map_table = pd.read_csv(core_mimiciv_path + 'hosp/d_icd_diagnoses.csv.gz', compression='gzip') 
    icd_names = []
    for icd_code in train_distribution.keys():
        icd_name = icd_map_table[icd_map_table['icd_code'] == icd_code]['long_title'].values
        if len(icd_name) > 0:
            icd_names.append((icd_code, icd_name[0]))
        else:
            icd_names.append((icd_code, 'Unknown'))

    icd_df = pd.DataFrame(icd_names, columns=['ICD Code', 'Name'])

    output_dir = f'{base_path}/split_by_patient_seed1'

    # 保存训练集和测试集
    save = True
    if save:
        os.makedirs(output_dir, exist_ok=True)
        icd_df.to_csv(os.path.join(output_dir, 'icd_and_name.csv'), index=False)

        with open(os.path.join(output_dir, 'train_data.json'), 'w') as f:
            json.dump(all_train_data, f,indent=4)

        with open(os.path.join(output_dir, 'val_data.json'), 'w') as f:
            json.dump(all_val_data, f,indent=4)

        with open(os.path.join(output_dir, 'test_data.json'), 'w') as f:
            json.dump(all_test_data, f,indent=4)
    
     # 绘制 ICD 分布图
    plt_icd_distribution(train_distribution, val_distribution, test_distribution, output_dir)

if __name__ == "__main__":
    main()