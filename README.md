# Legal RLHF — Reward Modeling and PPO for Legal Document Relevance

End-to-end RLHF pipeline applied to legal document relevance ranking,
using the EURLEX dataset from lex_glue (55,000 EU legal documents).

Motivated by prior work showing that supervised fine-tuning with limited 
labeled data collapses to constant outputs (Norel et al., under review).
RLHF with preference pairs provides a more robust training signal —
relative preferences are easier to learn than absolute scores.

## Overview

This project implements a three-stage RLHF pipeline:

1. **Reward Model** — trained on preference pairs (chosen vs rejected documents)
2. **PPO Loop** — policy optimization using reward model as signal
3. **Evaluation** — statistical validation of reward distribution and policy improvement

All data sourced from HuggingFace (public). No PHI involved.

## Dataset

**Source:** [lex_glue / eurlex](https://huggingface.co/datasets/lex_glue)
**Documents:** 55,000 EU legal texts with multi-label legal category annotations
**Preference pairs:** 5,000 (chosen: ≥5 labels, rejected: ≤2 labels)

## Design Decisions

**Truncation at 2,048 characters:** For a reward model learning relative preferences,
the header and opening context are sufficient signal. Chunking would complicate
preference pair construction without clear benefit for this task.

**SFT step omitted:** Reward model initialized directly from pre-trained DistilBERT.
In production, SFT on domain-specific data would precede reward model training.

**Distant supervision via n_labels:** In production RLHF, preference pairs come from
human annotators or LLM-as-a-Judge. Here we use n_labels as a proxy —
documents with more legal category labels are more topically specific.
Known limitation: topical specificity ≠ true legal relevance.

## Stack

- Python, pandas — data processing
- HuggingFace datasets, transformers — model and data loading
- trl — PPO implementation
- scikit-learn — evaluation metrics
- Matplotlib — visualizations
- Jupyter — EDA notebook

## Project Structure
```
legal-rlhf/
├── data/                        # not tracked in git
│   └── preference_pairs.csv     # 5,000 preference pairs
├── figures/
│   ├── 00_eda_distributions.png
│   └── 01_preference_pairs.png
├── notebooks/
│   └── 00_eda.ipynb             # data exploration and pair construction
├── src/
│   ├── 01_reward_model.py       # reward model training
│   ├── 02_ppo.py                # PPO loop
│   └── 03_eval.py               # evaluation and metrics
├── .gitignore
├── requirements.txt
└── README.md
```
## Setup

```bash
pyenv virtualenv 3.11 legal-rlhf
pyenv local legal-rlhf
pip install -r requirements.txt
jupyter notebook
```

## Status

🔄 In progress

---

**Author:** Raquel (Kely) Norel, PhD
**Domain:** Legal AI / RLHF

