base_path=.
test_data_path=$base_path/examples/ldr_dataset/mimic_llm_sample_100_subset.json

## Think
# cot_based, expert_designed, severity_based

prompt_type=expert_designed
# num_round_list=(2 3)
num_round_list=(1)

chat_model=cursorai_gemini_2_5_flash

for num_round in ${num_round_list[@]}
do
    echo "Prompt type: $prompt_type"
    python eicu_infer.py \
        --test_data_path $test_data_path \
        --task_type all \
        --chat_model $chat_model \
        --system_message_type $prompt_type \
        --think True \
        --temperature 1.0 \
        --num_processes 10 \
        --num_round $num_round &

    # sleep 5s
    # no think
    python eicu_infer.py \
        --test_data_path $test_data_path \
        --task_type all \
        --chat_model $chat_model \
        --system_message_type $prompt_type \
        --temperature 1.0 \
        --num_processes 10 \
        --num_round $num_round &
done