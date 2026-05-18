"""
Plot training curves from saved reward model metrics.
Run after 01_reward_model.py
"""

import json
import matplotlib.pyplot as plt

# Load metrics
with open("data/reward_model_metrics.json", "r") as f:
    metrics = json.load(f)

train_losses   = metrics["train_losses"]
val_losses     = metrics["val_losses"]
val_accuracies = metrics["val_accuracies"]

epochs = range(1, len(train_losses) + 1)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, train_losses, label='Train Loss', marker='o')
axes[0].plot(epochs, val_losses,   label='Val Loss',   marker='o')
axes[0].set_title('Reward Model — Loss Curves')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MarginRankingLoss')
axes[0].legend()

axes[1].plot(epochs, val_accuracies, label='Val Accuracy', marker='o', color='green')
axes[1].set_title('Reward Model — Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0.9, 1.0)
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/02_reward_model_training.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to figures/02_reward_model_training.png")