base_path=./
test_data_path=$base_path/examples/ldr_dataset/mimic_llm_sample_100_subset.json

# set your API key to environment variable or directly assign it here
export API_KEY="sk-xxxx"  # replace with your actual API key

## Think
# cot_based, machine_optimized, current_human_designed

# strict_expert_designed, standard_expert_designed, conservative_expert_designed
# severity_based_low, severity_based_moderate, severity_based_high
# strict_cot_based, standard_cot_based, conservative_cot_based

prompt_type=standard_expert_designed
# temperature_list=(0.1 0.7 1.0)
temperature_list=(1.0)
num_rounds=(1 2 3)

for temperature in "${temperature_list[@]}"; do
    echo "Running with temperature: $temperature"

    for num_round in "${num_rounds[@]}"; do
        # echo "Running with num_round: $num_round"

        python mimic_infer.py \
        --test_data_path $test_data_path \
        --task_type all \
        --chat_model cursorai_claude_sonnet_4 \
        --system_message_type $prompt_type \
        --think True \
        --temperature $temperature \
        --num_processes 10 \
        --num_round $num_round &

        # sleep 5s

        python mimic_infer.py \
            --test_data_path $test_data_path \
            --task_type all \
            --chat_model cursorai_claude_sonnet_4 \
            --system_message_type $prompt_type \
            --temperature $temperature \
            --num_processes 10 \
            --num_round $num_round &

    done
done
