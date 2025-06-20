# Assign the command line arguments to variables
# base_path
model_path=/xxx/open_r1/results/sft/DeepSeek-R1-Distill-Llama-8B_lr5e6/checkpoint-1080
question_path=/xxx/data/subset_top40_top1/infer_data/infer_icd_report_data_top.json
top_num=50
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

# 提取结果： -- SFT 哪个模型都可以用这个
python eval/extract_sft_openbio_result.py --input_file $answer_path --top_num 40
