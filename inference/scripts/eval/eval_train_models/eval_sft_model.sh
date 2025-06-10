base_path=/grp01/cs_yzyu/wushuai/model/llama_demo

predict_result=/grp01/cs_yzyu/wushuai/deep_r1/open-r1/infer_model/sft/DeepSeek-R1-Distill-Llama-8B_lr5e6/checkpoint-1080/40_test_infer_prompt_40/processed_results/valid_results.csv
# predict_result=/grp01/cs_yzyu/wushuai/deep_r1/open-r1/infer_model/grpo/DeepSeek-R1-Distill-Llama-8B-GRPO-mimic-v3/back_cp/checkpoint-980-light/40_test_infer_prompt_40/processed_results/merge_results.csv
# predict_result=/grp01/cs_yzyu/wushuai/deep_r1/open-r1/infer_model/sft/Llama3-OpenBioLLM-8B-mimic-lr5e6/checkpoint-1080/40_test_infer_prompt_40/processed_results/valid_results.csv
label_file=$base_path/datasets/mimic_multi_inputs_icd/split_by_patient/subset_top40_top1/infer_icd_report_data_top.json

# subset - 100
# predict_result=$base_path/infer_api/infer_api/results/subset_100/round_3/all_cursorai_gemini_2_5/processed_results/valid_results.csv
# label_file=$base_path/datasets/mimic_multi_inputs_icd/split_by_patient/subset_top40_top1/infer_icd_report_data_100_subset.json

# Calculate metrics for 40 classes
# python eval/compute_metrics.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# Calculate metrics for 33 classes
# python eval/compute_metrics_class_33.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# # Calculate metrics for 12 classes
# python eval/compute_metrics_class_12.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# Calculate confidence interval for 40 classes
# python eval/compute_metrics_ci.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# # Calculate confidence interval for 33 classes
# python eval/compute_metrics_class_33_ci.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# # Calculate confidence interval for 12 classes
# python eval/compute_metrics_class_12_ci.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# ### Calculate primary accuracy for each disease in 40 classes
# python eval/disease_metrics/compute_40_metrics.py \
#     --predict_result $predict_result \
#     --label_file $label_file

### Calculate primary accuracy for each disease in 12 classes
# python eval/disease_metrics/compute_12_metrics.py \
#     --predict_result $predict_result \
#     --label_file $label_file

# ### Calculate primary accuracy confidence interval for each disease in 12 classes
# python eval/disease_metrics/compute_12_metrics_ci.py \
#     --predict_result $predict_result \
#     --label_file $label_file

## Calculate distance 
# python eval/close_eval/close_eval_primary_v2.py \
#     --predict_result $predict_result \
#     --label_file $label_file

python eval/exact_miss_over/recall_precision_ci.py \
    --predict_result $predict_result \
    --label_file $label_file
