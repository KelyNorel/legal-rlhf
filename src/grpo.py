"""
GRPO Training for Legal Document Relevance Scoring

Uses trained reward model as signal to optimize a policy model
via Group Relative Policy Optimization (GRPO) — the algorithm
used in DeepSeek-R1. More efficient than PPO: no value model needed.

Policy model: DistilBERT sequence classifier
Reward model: trained in 01_reward_model.py (95.8% pairwise accuracy)
Framework: TRL 1.4.0
"""

import torch
import pandas as pd
import numpy as np
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
import json
import sys
import os

sys.path.append(os.path.dirname(__file__))
from reward_model import RewardModel

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 1e-5
KL_COEF    = 0.1   # KL penalty — keeps policy close to reference

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

# ── Load reward model ─────────────────────────────────────────────────────────
print("Loading reward model...")
reward_model = RewardModel(MODEL_NAME)
reward_model.load_state_dict(
    torch.load("data/reward_model.pt", map_location=DEVICE)
)
reward_model.to(DEVICE)
reward_model.eval()
print("Reward model loaded.")

# ── Policy model ──────────────────────────────────────────────────────────────
print("Loading policy model...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
policy    = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=1
).to(DEVICE)

# Reference model — frozen copy of initial policy
# KL divergence measured against this to prevent reward hacking
reference = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=1
).to(DEVICE)
for param in reference.parameters():
    param.requires_grad = False
print("Policy and reference models loaded.")

# ── GRPO Training Loop ────────────────────────────────────────────────────────
def get_reward(texts, model, tok):
    """Score documents using reward model."""
    enc = tok(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors='pt'
    ).to(DEVICE)
    with torch.no_grad():
        scores = model(enc['input_ids'], enc['attention_mask'])
    return scores

def kl_penalty(policy_logits, ref_logits):
    """
    Simplified KL penalty: L2 distance between policy and reference logits.
    Prevents policy from drifting too far from initialization.
    """
    return torch.mean((policy_logits - ref_logits) ** 2)


def grpo_loss(rewards, policy_logits, ref_logits):
    """
    GRPO loss = -advantage + KL penalty
    Advantage = reward normalized within the group (zero mean, unit std)
    """
    # Normalize rewards within batch (group relative)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    # Policy scores
    policy_scores = torch.sigmoid(policy_logits).squeeze()
    
    # Policy loss — maximize advantage-weighted scores
    policy_loss = -(advantages * policy_scores).mean()
    
    # KL penalty — L2 distance from reference
    kl = kl_penalty(policy_logits.squeeze(), ref_logits.squeeze())
    
    total_loss = policy_loss + KL_COEF * kl
    return total_loss, policy_loss.item(), kl.item()
    
def train():
    df = pd.read_csv("data/preference_pairs.csv")
    # Use chosen docs for policy training
    texts = df['chosen'].tolist()[:1000]  # subset for speed
    print(f"Training on {len(texts)} documents")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=LR)
    
    metrics = {
        "train_losses": [],
        "policy_losses": [],
        "kl_values": [],
        "mean_rewards": []
    }

    for epoch in range(EPOCHS):
        epoch_loss, epoch_pl, epoch_kl, epoch_reward = 0, 0, 0, 0
        n_batches = 0

        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]

            # Tokenize for policy
            enc = tokenizer(
                batch_texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors='pt'
            ).to(DEVICE)

            # Get rewards from reward model
            rewards = get_reward(batch_texts, reward_model, tokenizer)

            # Policy forward pass
            policy_out = policy(
                input_ids=enc['input_ids'],
                attention_mask=enc['attention_mask']
            )
            policy_logits = policy_out.logits

            # Reference forward pass (no grad)
            with torch.no_grad():
                ref_out = reference(
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask']
                )
                ref_logits = ref_out.logits

            # GRPO loss
            loss, pl, kl = grpo_loss(rewards, policy_logits, ref_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss   += loss.item()
            epoch_pl     += pl
            epoch_kl     += kl
            epoch_reward += rewards.mean().item()
            n_batches    += 1

        avg_loss   = epoch_loss   / n_batches
        avg_pl     = epoch_pl     / n_batches
        avg_kl     = epoch_kl     / n_batches
        avg_reward = epoch_reward / n_batches

        metrics["train_losses"].append(avg_loss)
        metrics["policy_losses"].append(avg_pl)
        metrics["kl_values"].append(avg_kl)
        metrics["mean_rewards"].append(avg_reward)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Policy Loss: {avg_pl:.4f} | "
              f"KL: {avg_kl:.4f} | "
              f"Mean Reward: {avg_reward:.4f}")

    # Save
    torch.save(policy.state_dict(), "data/policy_model.pt")
    with open("data/grpo_metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Policy model saved to data/policy_model.pt")
    print("Metrics saved to data/grpo_metrics.json")

if __name__ == "__main__":
    train()