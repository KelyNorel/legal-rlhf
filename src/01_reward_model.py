"""
Reward Model Training for Legal Document Relevance

Trains a reward model on preference pairs (chosen vs rejected documents).
The reward model learns to assign higher scores to more relevant documents.

Architecture: DistilBERT + linear head -> scalar reward score
Training: pairwise ranking loss (chosen score > rejected score)
Data: 5,000 preference pairs from EURLEX (lex_glue)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LENGTH   = 256        # tokens (not chars — faster than 512)
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
MARGIN       = 0.5        # ranking loss margin
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Device: {DEVICE}")

# ── Dataset ──────────────────────────────────────────────────────────────────
class PreferenceDataset(Dataset):
    """
    Each sample: (chosen_text, rejected_text)
    The reward model should score chosen > rejected.
    """
    def __init__(self, df, tokenizer, max_length):
        self.chosen   = df['chosen'].tolist()
        self.rejected = df['rejected'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, idx):
        chosen_enc = self.tokenizer(
            self.chosen[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        rejected_enc = self.tokenizer(
            self.rejected[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'chosen_input_ids':      chosen_enc['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_enc['attention_mask'].squeeze(),
            'rejected_input_ids':      rejected_enc['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_enc['attention_mask'].squeeze(),
        }

# ── Reward Model ─────────────────────────────────────────────────────────────
class RewardModel(nn.Module):
    """
    DistilBERT encoder + linear head -> scalar reward score.
    Uses [CLS] token representation as document embedding.
    """
    def __init__(self, model_name):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        reward = self.reward_head(cls_output)
        return reward.squeeze(-1)  # scalar per document

# ── Training Loop ─────────────────────────────────────────────────────────────
def train_reward_model():
    # Load data
    df = pd.read_csv("data/preference_pairs.csv")
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED)
    print(f"Train pairs: {len(train_df)} | Val pairs: {len(val_df)}")

    # Tokenizer and datasets
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = PreferenceDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset   = PreferenceDataset(val_df, tokenizer, MAX_LENGTH)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Model, optimizer, loss
    model = RewardModel(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MarginRankingLoss(margin=MARGIN)

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            chosen_ids   = batch['chosen_input_ids'].to(DEVICE)
            chosen_mask  = batch['chosen_attention_mask'].to(DEVICE)
            rejected_ids  = batch['rejected_input_ids'].to(DEVICE)
            rejected_mask = batch['rejected_attention_mask'].to(DEVICE)

            chosen_scores   = model(chosen_ids, chosen_mask)
            rejected_scores = model(rejected_ids, rejected_mask)

            # target=1 means chosen should score higher than rejected
            target = torch.ones(chosen_scores.size(0)).to(DEVICE)
            loss = loss_fn(chosen_scores, rejected_scores, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

       # ── Validate ──
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                chosen_ids   = batch['chosen_input_ids'].to(DEVICE)
                chosen_mask  = batch['chosen_attention_mask'].to(DEVICE)
                rejected_ids  = batch['rejected_input_ids'].to(DEVICE)
                rejected_mask = batch['rejected_attention_mask'].to(DEVICE)

                chosen_scores   = model(chosen_ids, chosen_mask)
                rejected_scores = model(rejected_ids, rejected_mask)

                target = torch.ones(chosen_scores.size(0)).to(DEVICE)
                loss = loss_fn(chosen_scores, rejected_scores, target)
                val_loss += loss.item()

                # Accuracy: how often chosen > rejected?
                correct += (chosen_scores > rejected_scores).sum().item()
                total += chosen_scores.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.3f}")

    return model, tokenizer, train_losses, val_losses, val_accuracies


if __name__ == "__main__":
    model, tokenizer, train_losses, val_losses, val_accuracies = train_reward_model()

    # Save model
    torch.save(model.state_dict(), "data/reward_model.pt")
    print("Model saved to data/reward_model.pt")

    # Save metrics
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies
    }
    with open("data/reward_model_metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Metrics saved to data/reward_model_metrics.json")

        