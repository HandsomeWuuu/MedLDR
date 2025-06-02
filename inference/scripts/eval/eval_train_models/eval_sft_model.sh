base_path=/grp01/cs_yzyu/wushuai/model/llama_demo

predict_result=/grp01/cs_yzyu/wushuai/deep_r1/open-r1/infer_model/sft/DeepSeek-R1-Distill-Llama-8B_lr5e6/checkpoint-1080/40_test_infer_prompt_40/processed_results/valid_results.csv
# predict_result=/grp01/cs_yzyu/wushuai/deep_r1/open-r1/infer_model/grpo/DeepSeek-R1-Distill-Llama-8B-GRPO-mimic-v3/back_cp/checkpoint-980-light/40_test_infer_prompt_40/processed_results/merge_results.csv
# predict_result=/grp01/cs_yzyu/wushuai/deep_r1/open-r1/infer_model/sft/Llama3-OpenBioLLM-8B-mimic-lr5e6/checkpoint-1080/40_test_infer_prompt_40/processed_results/valid_results.csv
label_file=$base_path/datasets/mimic_multi_inputs_icd/split_by_patient/subset_top40_top1/infer_icd_report_data_top.json

# subset - 100
# predict_result=$base_path/infer_api/infer_api/results/subset_100/round_3/all_cursorai_gemini_2_5/processed_results/valid_results.csv
# label_file=$base_path/datasets/mimic_multi_inputs_icd/split_by_patient/subset_top40_top1/infer_icd_report_data_100_subset.json

# 计算 40 类的指标
# python eval/compute_metrics.py \
#     --predict_result $predict_result \
#     --label_file $label_file


# 计算 33 类的指标
# python eval/compute_metrics_class_33.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# # 计算 12 类的指标
# python eval/compute_metrics_class_12.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# 计算置信区间 40 类
# python eval/compute_metrics_ci.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# # 计算置信区间 class 33 类
# python eval/compute_metrics_class_33_ci.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# # 计算置信区间 class 12 类
# python eval/compute_metrics_class_12_ci.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# ### 计算 40 类的 每种疾病的 primary accuracy
# python eval/disease_metrics/compute_40_metrics.py \
#     --predict_result $predict_result \
#     --label_file $label_file

### 计算 12 类的 每种疾病的 primary accuracy
# python eval/disease_metrics/compute_12_metrics.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# ### 计算 12 类的 每种疾病的 primary accuracy ci
# python eval/disease_metrics/compute_12_metrics_ci.py \
#     --predict_result $predict_result \
#     --label_file $label_file

## 计算 distance 
# python eval/close_eval/close_eval_primary_v2.py \
#     --predict_result $predict_result \
#     --label_file $label_file

python eval/exact_miss_over/recall_precision_ci.py \
    --predict_result $predict_result \
    --label_file $label_file