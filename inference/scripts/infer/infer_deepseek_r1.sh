base_path=/grp01/cs_yzyu/wushuai/model/llama_demo
python mimic_infer.py \
    --test_data_path $base_path/datasets/mimic_multi_inputs_icd/split_by_patient/subset_top40_top1/infer_icd_report_data_100_subset.json\
    --task_type all \
    --chat_model deepseek_r1_zzz \
    --num_processes 5 \