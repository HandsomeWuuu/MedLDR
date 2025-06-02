
log_file=./logs_gen/gen_mimic/all/sft/openbio_8b/sft_all_openbio_8b_out.log
mkdir -p $(dirname $log_file)


base_path=/data/h3571902/deep_r1

# train_data_top_icd_report_1_10, train_data_top_icd_report
accelerate launch --config_file=recipes/accelerate_configs/zero2.yaml src/open_r1/sft_mimic.py \
    --model_name_or_path $base_path/models/aaditya/Llama3-OpenBioLLM-8B \
    --train_data_path $base_path/data/split_by_patient_seed1_report_hpi/data/train_data_top_icd_report.json\
    --input_key all \
    --output_dir results/sft/Llama3-OpenBioLLM-8B-mimic-lr1e5/ \
    --learning_rate 1.0e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --num_train_epochs 3 \
    --max_length 8192 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy no \
    --eval_steps 50 \
    --save_strategy epoch \
    --save_only_model True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --seed 42 \
    --log_file $log_file
    # --save_steps 100 \
    # --save_total_limit 5 \
    # --torch_empty_cache_steps 1 \
    # --packing \
# --max_seq_length 9216 \
# --eval_data_path $base_path/llama_demo/datasets/mimic_multi_inputs_icd/split_by_patient_seed1_report_hpi/val_data_top_icd_report.json \
#     --test_data_path $base_path/llama_demo/datasets/mimic_multi_inputs_icd/split_by_patient_seed1_report_hpi/test_data_top_icd_report.json \