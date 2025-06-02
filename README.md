# Uncovering the Limits of Reasoning Large Language Models in Medical Diagnostics

This is an official implementation of "[Uncovering the Limits of Reasoning Large Language Models in Medical Diagnostics]()".

## Introduction

We constructed a long-context diagnostic reasoning (LDR) benchmark from the MIMIC-IV database. We systematically evaluated 5 state-of-the-art reasoning language models (RLMs) including Gemini-2.5-Pro, O4-Mini (High), O3-Mini (High), DeepSeek-R1, and Grok-3-Reasoner, along with 4 large language models including GPT-4.1, GPT-4o, DeepSeek-V3, and Grok-3. Our evaluation framework spans 3 dimensions (Accuracy, Reliability, Assistance) using 5 metrics: primary accuracy, exact match accuracy, miss and overdiagnosis rates, diagnostic reliability, and quality of differential diagnosis and clinical relevance.

**Key Findings**

- Our study reveals RLMs outperform LLMs in primary diagnoses (61.8% vs 52.5% accuracy), but still fall below clinical deployment thresholds.
- Even the best models achieve less than 25% exact match accuracy, with RLMs showing 43% higher overdiagnosis rates despite better identifying relevant conditions.
- RLMs demonstrate lower reliability, with 62% primary self-consistency compared to LLMs' 87%, suggesting reasoning introduces variability.
- As diagnostic assistants, RLMs provide superior differential diagnoses and higher clinical relevance, with only 29.2% completely unrelated predictions versus 37.6% for LLMs.
- All evaluated models still produce 30%+ clinically irrelevant predictions, indicating significant room for improvement before clinical deployment.

## Data:

First, you need to download [the MIMIC-IV dataset ](https://physionet.org/content/mimiciv/2.2/) (requires passing a qualification exam).
Then use our code in the `data` directory to process and obtain the data

## File Structure
Our code structure consists of three main parts: 
- Data processing
- Inference code
- Fine-tuning code

The detailed structure of the code is as follows:

```
├── README.md
├── LICENSE
├── data/                # Dataset related (download scripts, preprocessing, example data)
├── inference/           # Inference model code (model loading, inference scripts, evaluation)
├── finetune/            # Fine-tuning code (training scripts, configs, logs)
├── models/              # Pre-trained/fine-tuned model weights (optional, or download links)
├── utils/               # Utility functions, common scripts
├── scripts/             # One-click run scripts, batch processing
├── docs/                # Documentation, manuals, paper-related materials
└── examples/            # Example use cases, notebooks, etc.
```


## Supervised Fine-tuning Experiment
### 1. Requirements

### 2. Data Preparation

### 3. Inference

## Inference Experiment
### 1. Requirements

### 2. Data Preparation

### 3. Train


# Citation
If you find this project useful for your research, please consider citing:
```
@inproceedings{shuai2025MedLDR,
  title={Uncovering the Limits of Reasoning Large Language Models in Medical Diagnostics},
  author={Hongyu Zhuo, Shuai Wu, Meng Lou, Yizhou Yu},
  booktitle={},
  year={2025}
}
```
# Acknowledgment
Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.
> [Open-r1](https://github.com/huggingface/open-r1), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

# Contact

If you have any questions, please feel free to [create issues]()❓ or [contact me](u3010415@connect.hku.hk) 📧