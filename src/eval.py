"""
Evaluation of Reward Model and GRPO Policy

Statistical validation of the full RLHF pipeline:
1. Reward model: pairwise accuracy, score distributions
2. Policy: selection accuracy improvement over random baseline
3. Statistical significance: binomial test on selection accuracy
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats
from transformers import DistilBertTokenizer
import sys
import os

sys.path.append(os.path.dirname(__file__))
from reward_model import RewardModel

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256

# ── Load models and data ──────────────────────────────────────────────────────
print("Loading reward model...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
reward_model = RewardModel(MODEL_NAME)
reward_model.load_state_dict(
    torch.load("data/reward_model.pt", map_location=DEVICE)
)
reward_model.to(DEVICE)
reward_model.eval()

df = pd.read_csv("data/preference_pairs.csv")
print(f"Evaluating on {len(df)} preference pairs")

# ── 1. Reward model evaluation ────────────────────────────────────────────────
print("\n--- Reward Model Evaluation ---")

def score_texts(texts, batch_size=32):
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors='pt'
        ).to(DEVICE)
        with torch.no_grad():
            scores = reward_model(enc['input_ids'], enc['attention_mask'])
        all_scores.extend(scores.cpu().numpy())
    return np.array(all_scores)

chosen_scores   = score_texts(df['chosen'].tolist())
rejected_scores = score_texts(df['rejected'].tolist())

# Pairwise accuracy
pairwise_acc = (chosen_scores > rejected_scores).mean()
print(f"Pairwise accuracy (chosen > rejected): {pairwise_acc:.3f}")

# Mean scores
print(f"Mean chosen score:   {chosen_scores.mean():.4f} ± {chosen_scores.std():.4f}")
print(f"Mean rejected score: {rejected_scores.mean():.4f} ± {rejected_scores.std():.4f}")

# Statistical significance — t-test
t_stat, p_value = stats.ttest_ind(chosen_scores, rejected_scores)
print(f"t-test: t={t_stat:.3f}, p={p_value:.2e}")

# ── 2. GRPO policy evaluation ─────────────────────────────────────────────────
print("\n--- GRPO Policy Evaluation ---")

with open("data/grpo_metrics.json", "r") as f:
    grpo_metrics = json.load(f)

final_acc = grpo_metrics["selection_accuracy"][-1]
n_trials  = 1000  # approximate number of selections

# Binomial test: is final accuracy significantly above 0.5 (random)?
binom_result = stats.binomtest(
    int(final_acc * n_trials),
    n_trials,
    p=0.5,
    alternative='greater'
)
print(f"Final selection accuracy: {final_acc:.3f}")
print(f"Binomial test vs random (p=0.5): p={binom_result.pvalue:.2e}")

# ── 3. Score distribution plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Reward distributions
axes[0].hist(chosen_scores, bins=40, alpha=0.6, 
             label=f'Chosen (μ={chosen_scores.mean():.3f})', color='steelblue')
axes[0].hist(rejected_scores, bins=40, alpha=0.6, 
             label=f'Rejected (μ={rejected_scores.mean():.3f})', color='salmon')
axes[0].set_title('Reward Model — Score Distributions')
axes[0].set_xlabel('Reward Score')
axes[0].set_ylabel('Count')
axes[0].legend()

# Selection accuracy vs random baseline
epochs = range(1, len(grpo_metrics["selection_accuracy"]) + 1)
axes[1].plot(epochs, grpo_metrics["selection_accuracy"], 
             marker='o', color='green', label='Policy')
axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Random baseline')
axes[1].fill_between(epochs, 0.5, grpo_metrics["selection_accuracy"],
                     alpha=0.2, color='green')
axes[1].set_title(f'Policy Selection Accuracy\n(binomial p={binom_result.pvalue:.2e})')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0, 1)
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/04_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved to figures/04_evaluation.png")

# ── 4. Summary ────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("EVALUATION SUMMARY")
print("="*50)
print(f"Reward model pairwise accuracy:  {pairwise_acc:.3f}")
p_str = f"{p_value:.2e}" if p_value > 1e-300 else "< 2.2e-16"
print(f"t-test: t={t_stat:.3f}, p={p_str}")
print(f"Reward model t-test p-value:     {p_str}")
print(f"Policy final selection accuracy: {final_acc:.3f}")
print(f"Policy binomial test p-value:    {binom_result.pvalue:.2e}")
print("="*50)