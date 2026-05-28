# Astra

## A generalizable report generation foundation model for 3D computed tomography

**Astra** is a generalizable foundation model designed for automated report generation from 3D computed tomography (CT) scans.

## 📖 Overview
![Overview](fig/overview.png)

Astra supports multiple training and evaluation stages for 3D CT understanding, including:

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


