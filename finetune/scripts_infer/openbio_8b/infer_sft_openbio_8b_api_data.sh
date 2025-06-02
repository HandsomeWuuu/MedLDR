# Assign the command line arguments to variables
# base_path
model_path=/data/h3571902/deep_r1/open_r1/results/sft/Llama3-OpenBioLLM-8B-mimic-lr5e6/checkpoint-1080
question_path=/data/h3571902/deep_r1/data/subset_top40_top1/infer_data/infer_icd_report_data_top.json
top_num=40
answer_path=$model_path/40_test_infer_prompt_${top_num}/result.jsonl

# parent_dir=$(dirname "$file_path")
# mkdir -p "$parent_dir"

training_type=sft
script="scripts/infer_vllm_model_batch.py"
choose_sample_num=None

CUDA_VISIBLE_DEVICES=7 python3 "$script" \
        --model-path "$model_path" \
        --question-file "$question_path" \
        --answers-file "$answer_path" \
        --choose-num $choose_sample_num \
        --training-type $training_type \

# 提取结果：
python eval/extract_sft_openbio_result.py --input_file $answer_path --top_num $top_num