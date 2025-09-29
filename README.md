# The reasoning paradox of large language models in clinical diagnostics

This is an official implementation of "[The reasoning paradox of large language models in clinical diagnostics]()".

## Introduction

This study constructed two long-context diagnostic reasoning (LDR) benchmarks using the MIMIC-IV and eICU databases to evaluate six state-of-the-art matched model pairs: GPT-5 (Mini), Gemini 2.5 (Flash), Claude 4 (Sonnet), DeepSeek, Qwen3-235B, and Qwen3-30B. Each pair comprises a thinking-augmented model (referred to as LRM) and a standard model (LLM), both operating on the identical base model.
Our evaluation framework assessed two complementary dimensions: diagnostic performance and clinical reliability/safety.
The results reveal a key insight: although LRMs achieve superior diagnostic accuracy and comprehensiveness, these benefits are counterbalanced by notable limitations in clinical reliability and safety, such as overdiagnosis and inconsistencies at the level of individual diagnoses.



![Overview of model evaluation framework and LDR dataset construction pipeline](image/main_pipeline_public.png)
> *Overview of model evaluation framework and LDR-MIMIC dataset construction pipeline*



## Data:

For buliding LDR-MIMIC dataset, you need to download [the MIMIC-IV dataset ](https://physionet.org/content/mimiciv/2.2/) (requires passing a qualification exam).
Then use our code in the `data/` directory to process and generate the LDR training and evaluation datasets.

Similarly, to bulid the LDR-eICU dataset, you need to download [the eICU dataset](https://eicu-crd.mit.edu/).


## File Structure
Our code structure consists of three main parts: 
- Data processing: scripts for preparing and processing the MIMIC-IV dataset into the LDR benchmark.
- Inference code: scripts for model inference and evaluation via API.
- Examples: single-case inference to check case input and output; 100 cases from each dataset are provided for consistency reproduction.

The detailed structure of the code is as follows:

```
├── README.md
├── LICENSE
├── data/                # Dataset related (download scripts, preprocessing, example data)
│   ├── preprocess/      # Preprocessing MIMIC-IV dataset, integrating each patient's admission information
│   └── make_ldr/        # Scripts and workflow for constructing the LDR dataset
├── inference/           # Inference model code (model loading, inference scripts, evaluation)
├── finetune/            # Fine-tuning code (training scripts, configs, logs)
├── models/              # Pre-trained/fine-tuned model weights (optional, or download links)
└── examples/            # Example use cases, notebooks, etc.
```

## Inference API models
### 1. Requirements
To run inference, please install the required Python packages:

```bash
pip install openai requests
```

To perform inference with APIs (e.g., OpenAI, DeepSeek), you must obtain and set your API keys. You can directly insert your API key in the code where required.

### 2. Data Preparation
- First, download the [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) dataset from PhysioNet (requires credential approval).

- Next, use the scripts in the `data/preprocess` directory to process the raw MIMIC-IV data and organize all admission information for each patient.

- Finally, run the scripts in `data/make_ldr` to construct the LDR training and evaluation datasets.

### 3. Inference Evaluation Data
As an example, to perform inference using the DeepSeek-R1 model, follow these steps:

1. **Run Inference**  
  Use the provided shell script to generate predictions:
  ```bash
  bash ./inference/scripts/infer/infer_deepseek_r1.sh
  ```

2. **Extract Results**  
  After inference, extract and format the results:
  ```bash
  python process_data/extract_result.py
  ```

3. **Evaluate Predictions**  
  Evaluate the predictions using the evaluation script:
  ```bash
  bash ./inference/scripts/eval/eval_api_models/eval_deepseek_r1_metrics.sh
  ```

Replace the placeholder paths in the scripts with your actual prediction and reference files as needed.


##  Fine-tuning models
### 1. Requirements
**Model Preparation**
Download the models from the links provided in `models/download_link.txt`:
- [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- [Llama3-OpenBioLLM-8B](https://huggingface.co/aaditya/Llama3-OpenBioLLM-8B)

**Training Environment**
Our training code is based on [open-r1](https://github.com/huggingface/open-r). Please set up your environment according to the requirements of open-r1.

### 2. Data Preparation

Use the training dataset of the LDR dataset.

### 3. Train

**Supervised Fine-tuning Training**

- Train DeepSeek-R1-Distill-Llama-8B
```
bash ./finetune/scripts_train/distill_llama_8b/sft/train_sft_llama_8b_mimic.sh
```
- Train Llama3-OpenBioLLM-8B
```
bash ./finetune/scripts_train/distill_llama_8b/grpo/train_grpo_llama_8b.sh
```

**Reinforcement Learning Training (GPRO)**
- Train DeepSeek-R1-Distill-Llama-8B
```
bash ./finetune/scripts_train/openbio_8b/sft/train_sft_openbio_8b_mimic.sh
```
### 4. Inference Evaluation Data

1. **Run Inference**  
  Use the provided shell script to generate predictions:
  ```bash
  # Inference SFT models
  bash ./finetune/scripts_infer/distill_llama_8b/infer_sft_distill_llama_api_data.sh
  # Inference RL models
  bash ./finetune/scripts_infer/distill_llama_8b/infer_grpo_distill_llama_8b_model.sh
  ```

2. **Extract Results**  
  After inference, extract and format the results:
  ```bash
  python finetune/eval/extract_sft_openbio_result.py
  ```

3. **Evaluate Predictions**  
  Evaluate the predictions using the evaluation script:
  ```bash
  bash ./inference/scripts/eval/eval_train_models/eval_sft_model.sh
  ```

## Examples

### One case testing:
**Inference with API Models**

Use the [`examples/inference_api.ipynb`](examples/inference_api.ipynb) notebook to perform inference with API models.

**Inference with Fine-tuned Models**

Use the [`examples/inference_ft.ipynb`](examples/inference_ft.ipynb) notebook to perform inference with fine-tuned models.

### One hundred cases for batch testing:

We provide 100 LDR test cases in `examples/ldr_dataset/mimic_llm_sample_100_subset.json` for consistency verification.

Additionally, 100 external eICU dataset test cases are available in `examples/eicu_dataset/eicu_llm_sample_100_subset.json` for consistency verification.

After configuring your API KEY:

**Testing the LDR dataset**

1. You can run batch testing on the LDR dataset using `inference/scripts/infer/infer_gemini_2_5_flash.sh`.
2. Then, extract the inference results for each model and each round:
  ```
  python inference/ldr_process_data/extract_result.py model_xx_output.jsonl
  ```
3. Calculate repeat consistency:
  ```
  bash inference/scripts/eval/lanuch_consistency.sh
  ```
4. Compute metrics (note: metrics on only 100 cases may differ from those on the full dataset):
  ```
  bash inference/scripts/eval/eval_api_models/eval_gemini_2_5_flash.sh
  ```

**Testing the external eICU dataset**

1. Run testing using `inference/scripts_eicu/infer/infer_gemini_2_5_flash.sh`.
2. Then, extract the inference results for each model and each round:
  ```
  python inference/eicu_eval/extract_eicu_results.py model_xx_output.jsonl
  ```
3. Calculate repeat consistency:
  ```
  bash inference/scripts_eicu/eval/lanuch_consistency.sh
  ```
4. Compute metrics (note: metrics on only 100 cases may differ from those on the full dataset):
  ```
  bash inference/scripts_eicu/eval/eval_gemini_2_5_flash.sh
  ```

<!-- # Citation
If you find this project useful for your research, please consider citing:
```
@inproceedings{shuai2025MedLDR,
  title={Uncovering the Limits of Reasoning Large Language Models in Medical Diagnostics},
  author={Hongyu Zhuo, Shuai Wu, Meng Lou, Yizhou Yu},
  booktitle={},
  year={2025}
}
``` -->

# Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.
> [Open-r1](https://github.com/huggingface/open-r1), [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)

# Contact

If you have any questions, please feel free to [create issues]()❓ or [contact me](u3010415@connect.hku.hk) 📧