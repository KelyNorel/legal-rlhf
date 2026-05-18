"""
Plot GRPO training curves from saved metrics.
Run after grpo.py
"""

import json
import matplotlib.pyplot as plt

with open("data/grpo_metrics.json", "r") as f:
    metrics = json.load(f)

epochs = range(1, len(metrics["train_losses"]) + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Selection Accuracy
axes[0].plot(epochs, metrics["selection_accuracy"], 
             marker='o', color='green')
axes[0].set_title('Policy — Selection Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, 1)
axes[0].axhline(y=0.5, color='gray', linestyle='--', label='random baseline')
axes[0].legend()

# KL Divergence
axes[1].plot(epochs, metrics["kl_values"], 
             marker='o', color='orange')
axes[1].set_title('KL Penalty (policy vs reference)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('L2 Distance')

# Mean Reward
axes[2].plot(epochs, metrics["mean_rewards"], 
             marker='o', color='steelblue')
axes[2].set_title('Mean Reward per Epoch')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Reward Score')

plt.tight_layout()
plt.savefig('figures/03_grpo_training.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to figures/03_grpo_training.png")