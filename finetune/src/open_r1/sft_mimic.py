# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import SFTConfig
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    DataCollatorForCompletionOnlyLM,
)
from open_r1.configs import data_config as DataArguments

# from open_r1.process_data.for_generate.load_mimic_icd import MIMICICDSupervisedDataset
# from open_r1.process_data.for_generate.load_mimic_icd_tf import MIMICICDSupervisedDataset,DataCollatorForIcdDataset
from open_r1.process_data.for_generate.load_mimic_icd import get_dataset,process_input_data

from typing import Dict, Optional, Sequence, List
from open_r1.utils import rank0_print


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, model_type) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    import torch.distributed as dist

    train_dataset = get_dataset(data_args,tokenizer,'train')  # 这里传入的是 train dataset
        # 这里传入的是 eval dataset
    if data_args.eval_data_path is not None:
        eval_dataset = get_dataset(data_args,tokenizer,'val')
    else:
        eval_dataset = None

    if data_args.test_data_path is not None:
        test_dataset = get_dataset(data_args,tokenizer,'test')
    else:
        test_dataset = None

    if model_type == 'OpenBioLLM-8B':
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    elif model_type == 'Distill-Llama-8B':
        response_template = "<｜Assistant｜>"
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    else:
        response_template = "### Answer:"
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[1:]
    
    print("response_template:", response_template)
    print("response_template_ids:", response_template_ids)
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    # collator = DataCollatorForIcdDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                test_dataset=test_dataset,
                collator=collator)




def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
    import numpy as np
    
    predictions, label_ids = eval_pred.predictions, eval_pred.label_ids
    
    # Logit-level loss
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    logit_loss = np.mean(np.square(logits - label_ids))
    
    # Decode predictions and labels to get ICD codes
    # Assuming predictions and labels are tokenized text that needs to be decoded
    # and contain ICD codes in a consistent format
    def extract_icds(text):
        # Split text and extract ICD codes
        # Modify this based on your exact format
        icds = set(code.strip() for code in text.split() if code.strip())
        return icds
    
    pred_icds = extract_icds(predictions)
    label_icds = extract_icds(label_ids)
    
    # Calculate overlap accuracy
    overlap = len(pred_icds.intersection(label_icds))
    total = len(label_icds)
    icd_accuracy = overlap / total if total > 0 else 0
    
    metrics = {
        "logit_loss": float(logit_loss),
        "icd_accuracy": float(icd_accuracy)
    }
    
    return metrics


logger = logging.getLogger(__name__)


def main(data_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            # logging.StreamHandler(sys.stdout),
            logging.FileHandler(training_args.log_file)
        ],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {data_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if 'OpenBioLLM-8B' in model_args.model_name_or_path:
        model_type = 'OpenBioLLM-8B'
    else:
        model_type = 'Distill-Llama-8B'

    # print("数据集:", dataset)
    # # 打印数据集一个案例
    # print("script_args:", script_args)  
    # print("数据集一个案例:", dataset[script_args.dataset_train_split][0])

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"
    ################
    # Load datasets
    ################
    # data = get_dataset(data_args,tokenizer,'train')

    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    data_module = make_supervised_data_module(tokenizer, data_args, model_type)

    def formatting_prompts_func(example):

        input_text = process_input_data(example['input'],tokenizer)

        if model_type == 'OpenBioLLM-8B':
            messages = [
                        {"role": "system", "content": example['input_prompt']},
                        {"role": "user", "content": input_text},
                        ]
            output_texts = tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True)
            # 把输出的文本拼接起来
            output_texts += f"{example['output']}<|eot_id|>"

        elif model_type == 'Distill-Llama-8B':
            messages = [
                        {"role": "system", "content": example['input_prompt']},
                        {"role": "user", "content": input_text},
                        ]
            output_texts = tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True)
            # <｜Assistant｜><think> 把think去掉
            output_texts = output_texts.replace("<think>", "")

            # 把输出的文本拼接起来
            output_texts += f"{example['output']}<｜end▁of▁sentence｜>"

        else:
            output_texts = f"### Question: {example['input_prompt']}\n{input_text}.\n ### Answer: {example['output']}"
        
        # print("output_texts:", output_texts)
        return output_texts
    
    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    ############################
    # Initialize the SFT Trainer
    ############################
    # trainer = CustomSFTTrainer(
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["collator"],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        formatting_func=formatting_prompts_func,
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    # train
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(data_module['train_dataset'])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": data_args.train_data_path,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(data_module["test_dataset"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)




if __name__ == "__main__":
    parser = TrlParser((DataArguments, SFTConfig, ModelConfig))
    data_args, training_args, model_args = parser.parse_args_and_config()
    rank0_print("data_args:", data_args)
    rank0_print("training_args:", training_args)
    rank0_print("model_args:", model_args)

    main(data_args, training_args, model_args)
