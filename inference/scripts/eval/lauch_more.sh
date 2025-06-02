
#!/bin/bash
# eval_gpt_4_1.sh
bash_list=(eval_gemini_2_5.sh
          eval_deepseek_r1_metrics.sh
           eval_deepseek_v3_metrics.sh
           eval_grok_3_metrics.sh
           eval_grok_3_reasoning_metrics.sh
           eval_o3_mini_metrics.sh
           eval_o4_mini.sh
           eval_gpt_4o_metrics.sh
           )

for script in "${bash_list[@]}"; do
    echo "Running $script..."
    bash "./scripts/eval/eval_api_models/$script" &
    echo "$script completed."
    echo "----------------------------------------"
done

echo "All scripts executed successfully."