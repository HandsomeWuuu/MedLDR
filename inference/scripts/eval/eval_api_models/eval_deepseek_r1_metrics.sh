base_path=/grp01/cs_yzyu/wushuai/model/llama_demo
predict_result=$base_path/infer_api/infer_api/results/all_deepseek_r1_zzz/processed_results/valid_results.csv
label_file=$base_path/datasets/mimic_multi_inputs_icd/split_by_patient/subset_top40_top1/infer_icd_report_data_top.json

########## Basic Metrics ##############
python eval/compute_metrics.py \
    --predict_result $predict_result \
    --label_file $label_file

# 计算 12 类的指标
python eval/compute_metrics_class_12.py \
    --predict_result $predict_result \
    --label_file $label_file

######### Primary Accuracy of every disease ##############
### 计算 40 类的 每种疾病的 primary accuracy
python eval/disease_metrics/compute_40_metrics.py \
    --predict_result $predict_result \
    --label_file $label_file

# ### 计算 12 类的 每种疾病的 primary accuracy ci
python eval/disease_metrics/compute_12_metrics_ci.py \
    --predict_result $predict_result \
    --label_file $label_file

############ Recall at N (Top-N accurcy) ##############
## 计算 12 类的 recall N  ci
python eval/recall_at_n/recall_at_n_12.py \
    --predict_result $predict_result \
    --label_file $label_file

## 计算 40 类的 recall N  ci
python eval/recall_at_n/recall_at_n_40.py \
    --predict_result $predict_result \
    --label_file $label_file

############ Distance Metrics ##############
## 计算 distance 
python eval/close_eval/close_eval_primary_v2.py \
    --predict_result $predict_result \
    --label_file $label_file

############ Miss Over Metrics ##############
python eval/exact_miss_over/exact_miss_over.py \
    --predict_result $predict_result \
    --label_file $label_file
