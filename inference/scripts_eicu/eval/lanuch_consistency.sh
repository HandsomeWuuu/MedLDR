
# algorithm_list=(claude-sonnet-4  gemini_2_5_flash qwen-plus )
algorithm_list=(gemini_2_5_flash)
# algorithm_list=(qwen3_30b_a3b)
think_mode_type_list=(think nothink)
temperature_list=(1.0)

for algorithm in ${algorithm_list[@]}; do
    for think_mode in ${think_mode_type_list[@]}; do
        for temperature in ${temperature_list[@]}; do
            python inference/eicu_eval/check_consistency.py \
                --compare_model $algorithm \
                --think_mode $think_mode \
                --temperature $temperature
        done
    done
done

