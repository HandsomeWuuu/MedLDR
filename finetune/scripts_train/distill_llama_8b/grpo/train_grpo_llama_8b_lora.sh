#!/bin/bash
#SBATCH --job-name=train_grpo_lora          # Job name
##SBATCH --mail-type=BEGIN,END,FAIL                  # Mail events
##SBATCH --mail-user=u3010415@connect.hku.hk                 # Set your email address
#SBATCH --partition=gpu_shared,gpu             # Specific Partition (gpu/gpu_shared)
#####SBATCH --nodelist=SPGL-1-15
#SBATCH --qos=normal                           # Specific QoS (debug/normal)
#SBATCH --time=3-00:05:00                      # Wall time limit (days-hrs:min:sec)
#SBATCH --nodes=1                              # Single compute node used
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16                      # CPUs used
#SBATCH --gpus-per-node=8                  # GPUs used
#SBATCH --mem=256G
#SBATCH --output=./logs/distill-llama-8b/grpo/lora_grpo_math_220k_v2_out.log                       # Standard output file
#SBATCH --error=./logs/distill-llama-8b/grpo/lora_grpo_math_220k_v2_err.log                        # Standard error file
#SBATCH --exclude=SPGL-1-12


export PATH=/grp01/cs_yzyu/wushuai/anaconda/condabin:/grp01/cs_yzyu/wushuai/.conda/envs/open_r1/bin:$PATH
export PYTHONPATH=/grp01/cs_yzyu/wushuai/model/llama_demo:$PYTHONPATH
export HOME=/grp01/cs_yzyu/wushuai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export VLLM_USE_MODELSCOPE=True
# echo 'Cuda home: ' $CUDA_HOME
source /grp01/cs_yzyu/wushuai/anaconda/bin/activate
conda activate open_r1


ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/DeepSeek-R1-Distill-Llama-8B/grpo/llama_8b_grpo_config_lora.yaml
    # --config recipes/DeepSeek-R1-Distill-Llama-8B/grpo/llama_8b_grpo_config.yaml
    # --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
# export VLLM_DEVICE_ID=7        # 指定 vLLM 使用的 GPU

# ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
#     --num_processes=7 src/open_r1/grpo_lora.py \
#     --config recipes/DeepSeek-R1-Distill-Llama-8B/grpo/llama_8b_grpo_config_lora.yaml