# Assign the command line arguments to variables
# base_path
model_path=/data/h3571902/deep_r1/open_r1/results/grpo/DeepSeek-R1-Distill-Llama-8B-GRPO-mimic-v3/back_cp/checkpoint-980-light
question_path=/data/h3571902/deep_r1/data/subset_top40_top1/infer_data/infer_icd_report_data_top.json
top_num=40
answer_path=$model_path/40_test_infer_prompt_${top_num}/result.jsonl


training_type=grpo
script="scripts/infer_vllm_model_batch.py"
choose_sample_num=None

CUDA_VISIBLE_DEVICES=7 python3 "$script" \
        --model-path "$model_path" \
        --question-file "$question_path" \
        --answers-file "$answer_path" \
        --choose-num $choose_sample_num \
        --training-type $training_type \

# 提取结果： -- SFT 哪个模型都可以用这个, GRPO 应该也可以用这个
python eval/extract_sft_openbio_result.py --input_file $answer_path --top_num $top_num
