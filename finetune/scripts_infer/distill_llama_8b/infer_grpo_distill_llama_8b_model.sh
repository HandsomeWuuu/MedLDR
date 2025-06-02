# Assign the command line arguments to variables
# base_path
model_path=/data/h3571902/deep_r1/open_r1/results/grpo/DeepSeek-R1-Distill-Llama-8B-GRPO-mimic-subset-only-acc/back_cp/checkpoint-480-light
question_path=/data/h3571902/deep_r1/data/split_by_patient_seed1_report_hpi/data/test_data_top_icd_report.json
answer_path=$model_path/50_test_infer_result_batch/result.jsonl

# parent_dir=$(dirname "$file_path")
# mkdir -p "$parent_dir"

training_type=grpo
script="scripts/infer_vllm_model_batch.py"
choose_sample_num=None

CUDA_VISIBLE_DEVICES=2 python3 "$script" \
        --model-path "$model_path" \
        --question-file "$question_path" \
        --answers-file "$answer_path" \
        --choose-num $choose_sample_num \
        --training-type $training_type \