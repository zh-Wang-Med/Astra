# Astra: a generalizable report generation foundation model for 3D computed tomography

**Intrinsic inconsistencies in reporting style and diagnostic terminology** across cohorts make naive joint training prone to noisy textual supervision, thereby limiting model generalizability. Here we present Astra, a generalizable CT report generation foundation model trained on 90,678 thoracoabdominal CT–report pairs (CTRgDB) with 353,671 abnormalities spanning eight organ systems. By harmonizing report style and further refining diagnostic consistency via reinforcement learning, Astra achieves style-consistent and diagnostically accurate report generation across diverse anatomical regions and institutions. 

We systematically evaluate Astra across three dimensions: **methodological generalizability**, **clinical utility**, and **foundational extensibility**. Methodological generalizability is established through rigorous benchmarking across CTRgDB and six independent out-of-distribution clinical cohorts, where Astra consistently surpasses existing state-of-the-art architectures. Clinical utility is corroborated via a real-world human-AI collaboration study, where Astra, on average, accelerates chest CT report drafting by 29.6\% and enhances abdominal CT report completeness by 11.3\% across diverse levels of clinical expertise. Finally, foundational extensibility is evidenced by the capacity to catalyze broader AI development paradigms, where Astra facilitates diagnostic model development through ensemble strategies with pretrained vision encoder and scales vision-language pretraining by synthesizing diagnostic reports for previously unreported scans. 

Taken together, **Astra represents a transformative step toward generalizable foundation models for 3D CT report generation, moving beyond the traditional single-cohort paradigm and highlighting their broader potential across clinical and research settings**.


## 📖 Overview
![Overview](fig/overview.png)

Code including:

* 📝 **Supervised Fine-Tuning (SFT)** for CT report generation
* 🎯 **Reinforcement Learning with GRPO** for reward-guided report generation optimization
* 🏷️ **Ensemble Classification** for multi-label disease classification
* 🔬 **Scaling Pretraining** for large-scale 3D CT visual encoder pretraining 

## 📁 Repository Structure

```text
astra_open/
├── pretrain/                           # Scaling pretraining
│   └── scripts/
│       └── run_train.sh                # Pretraining launch script
├── sft/                                # Supervised fine-tuning
│   └── scripts/                        # SFT launch scripts
├── r1_v/                               # Reinforcement learning with GRPO
│   └── src/
│       └── r1-v/
│           └── run_grpo.sh             # GRPO launch script
├── ensemble/                           # Ensemble classification
│   └── train.sh                        # Classification launch script
├── requirements_pretrain.txt           # Dependencies for pretraining
├── requirements_sft_classification.txt # Dependencies for SFT and classification
├── requirements_rl.txt                 # Dependencies for RL with GRPO
└── README.md
```

## 🛠️ Environment Setup

Astra uses three independent environments for different training stages. We recommend using `conda` to manage them separately.

### Environment 1: Scaling Pretraining

This environment is used for large-scale 3D CT visual encoder pretraining.

```bash
# Create environment
conda create -n astra_pretrain python=3.9 -y
conda activate astra_pretrain

# Install dependencies
pip install -r requirements_pretrain.txt
```

### Environment 2: SFT and Ensemble Classification

This environment is used for supervised fine-tuning and multi-label disease classification.

```bash
# Create environment
conda create -n astra_sft python=3.10 -y
conda activate astra_sft

# Install dependencies
pip install -r requirements_sft_classification.txt
```

### Environment 3: Reinforcement Learning with GRPO

This environment is used for GRPO-based reinforcement learning optimization.

```bash
# Create environment
conda create -n astra_rl python=3.10 -y
conda activate astra_rl

# Install dependencies
pip install -r requirements_rl.txt
```

## 🚀 Quick Start

### Supervised Fine-Tuning

```bash
conda activate astra_sft
cd sft/scripts

bash sft/scripts/train.sh
```

### Reinforcement Learning with GRPO

```bash
conda activate astra_rl
cd r1_v/src/r1-v

bash run_grpo.sh
```

### Ensemble Classification

```bash
conda activate astra_sft
cd ensemble

bash train.sh
```

### Scaling Pretraining

```bash
conda activate astra_pretrain
cd pretrain/scripts

bash run_train.sh
```

## 📌 Notes

Please make sure that the required datasets, pretrained checkpoints, and configuration files are correctly prepared before launching each training stage.


