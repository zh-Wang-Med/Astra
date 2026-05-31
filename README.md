# Astra: a generalizable report generation foundation model for 3D computed tomography

**Intrinsic inconsistencies in reporting style and diagnostic terminology** across cohorts make naive joint training prone to noisy textual supervision, thereby limiting model generalizability. Here we present Astra, a generalizable CT report generation foundation model trained on 90,678 thoracoabdominal CT–report pairs (CTRgDB) with 353,671 abnormalities spanning eight organ systems. By harmonizing report style and further refining diagnostic consistency via reinforcement learning, Astra achieves style-consistent and diagnostically accurate report generation across diverse anatomical regions and institutions. 

We systematically evaluate Astra across three dimensions: **methodological generalizability**, **clinical utility**, and **foundational extensibility**. Methodological generalizability is established through rigorous benchmarking across CTRgDB and six independent out-of-distribution clinical cohorts, where Astra consistently surpasses existing state-of-the-art architectures. Clinical utility is corroborated via a real-world human-AI collaboration study, where Astra, on average, accelerates chest CT report drafting by 29.6\% and enhances abdominal CT report completeness by 11.3\% across diverse levels of clinical expertise. Finally, foundational extensibility is evidenced by the capacity to catalyze broader AI development paradigms, where Astra facilitates diagnostic model development through ensemble strategies with pretrained vision encoder and scales vision-language pretraining by synthesizing diagnostic reports for previously unreported scans. 

Taken together, **Astra represents a transformative step toward generalizable foundation models for 3D CT report generation, moving beyond the traditional single-cohort paradigm and highlighting their broader potential across clinical and research settings**.


## 📖 Overview
![Overview](fig/overview.png)

## 📖 Benchmark
![Benchmark](fig/fig3_merge.jpg)

## 📁 Repository Structure
Code including:

* 📝 **Supervised Fine-Tuning (SFT)** for CT report generation
* 🎯 **Reinforcement Learning with GRPO** for reward-guided report generation optimization
* 🏷️ **Ensemble Classification** for multi-label disease classification
* 🔬 **Scaling Pretraining** for large-scale 3D CT visual encoder pretraining
  
code is coming soon




