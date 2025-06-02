import json
import pandas as pd
from collections import Counter
from tqdm import tqdm

'''
    文件描述：
    1. 统计 lab 和 icd code 的频次



'''
# Load the dataset
base_path = '/userhome/cs/u3010415/Multi-modal/MMCaD_code2/data/pair_lab_and_icd_v2'
multiple_files = [
    f'{base_path}/img_img/img_valid_pair.json',
    f'{base_path}/img_unimg/img_unimg_valid_data_pair.json',
    f'{base_path}/unimg/unimg_valid_data_pair.json'
]

def process_multiple_files(file_paths):
    combined_counter = Counter()
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
        df = pd.DataFrame(data)

        # 打印 file_path 和 df 的长度
        print(f"file_path: {file_path.split('/')[-1]}, df len: {len(df)}")
        
        combined_counter.update([code[0] for codes in df['icd_codes'] for code in codes])

    combined_freq_df = pd.DataFrame(combined_counter.items(), columns=['Code', 'Frequency'])
    combined_freq_df = combined_freq_df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    return combined_freq_df

def summary_multiple_files():
    multiple_files_freq_df = process_multiple_files(multiple_files)
    print(multiple_files_freq_df)

    # Save ICD codes with frequency higher than freq_num to a file (such as top 50)
    top_num = 50
    freq_df = multiple_files_freq_df.iloc[:top_num]
    print(f'top {top_num} ICD codes len: {len(freq_df)}')

    print(f'Top {top_num} ICD codes frequency:', freq_df['Frequency'].sum())
    freq_df.to_csv(f'{base_path}/lab_icd_pair_top_{top_num}.csv', index=False)



if __name__ == "__main__":
    summary_multiple_files()