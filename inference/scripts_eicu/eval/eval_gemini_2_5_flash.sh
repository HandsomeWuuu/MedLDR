base_path=./
label_file=$base_path/examples/eicu_dataset/eicu_llm_sample_100_subset.json

# ## First, run the extract_eicu_results.py script to generate the prediction results file
# python inference/eicu_eval/extract_eicu_results.py \
#     $base_path/all_results/results_eicu/xxx/valid_results.jsonl \

# Wait for the script to finish, then calculate metrics
pred_file=$base_path/all_results/results_eicu/xxx/processed_results/valid_results.csv

# Calculate metrics
python inference/eicu_eval/compute_metrics.py \
    --predict_result $pred_file \
    --label_file $label_file

# Calculate missed and over-diagnosis
python inference/eicu_eval/exact_miss_over.py \
    --predict_result $pred_file \
    --label_file $label_file \

# Calculate Recall@K
python inference/eicu_eval/compute_recall_n.py \
    --predict_result $pred_file \
    --label_file $label_file
