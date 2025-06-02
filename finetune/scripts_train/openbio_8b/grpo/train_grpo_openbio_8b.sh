

log_file=./logs_gen/gen_mimic/all/grpo/openbio-8b/grpo/grpo_mimic_openbio_8b_all_only_acc_base_sft_out.log
mkdir -p $(dirname $log_file)
base_path=/data/h3571902/deep_r1
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=7 src/open_r1/grpo_mimic.py \
    --train_data_path $base_path/data/split_by_patient_seed1_report_hpi/data/train_data_top_icd_report.json\
    --input_key all \
    --log_file $log_file \
    --training_type grpo \
    --config recipes/Llama3-OpenBioLLM-8B/grpo/openbio_8b_grpo_config.yaml 2>&1 | tee $log_file
