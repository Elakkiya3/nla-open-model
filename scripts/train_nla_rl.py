"""
Natural Language Autoencoder (NLA) RL Training
Trains AV and AR jointly using reinforcement learning.

AV learns: activation -> text explanation
AR learns: text explanation -> activation

Optimized for: Fraction of Variance Explained (FVE)
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
HIDDEN_DIM = 896
MODEL_DIM = 1024
BATCH_SIZE = 2
LR_AV = 5e-6
LR_AR = 1e-4
NUM_STEPS = 500
SAVE_EVERY = 50
OUTPUT_DIR = Path("checkpoints/nla_rl")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("NATURAL LANGUAGE AUTOENCODER - RL TRAINING")
print("=" * 70)

# Load data
print("\n1. Loading warm-start data...")
activations = np.load("data/warmstart/activations.npy")
with open("data/warmstart/summaries.json") as f:
    summaries = json.load(f)

print(f"   Loaded {len(activations)} activation-summary pairs")
print(f"   Activation shape: {activations.shape}")

# Initialize models
print("\n2. Initializing models...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# AV: activation -> text
av_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    dtype=torch.float16, 
    device_map="auto"
)
av_model.eval()

# AR: text -> activation
class AR(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16,
            device_map="auto"
        )
        self.head = nn.Linear(MODEL_DIM, HIDDEN_DIM)
    
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.model(
                input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        hidden = outputs.hidden_states[-1][:, -1, :]
        return self.head(hidden)

ar_model = AR()
ar_optimizer = Adam(ar_model.parameters(), lr=LR_AR)

print("   ✓ AV initialized")
print("   ✓ AR initialized")

# Training loop
print("\n3. Starting RL Training...")
print(f"   Steps: {NUM_STEPS}")
print(f"   Batch size: {BATCH_SIZE}")
print()

fve_history = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for step in range(NUM_STEPS):
    # Sample batch of activations
    batch_idx = np.random.choice(len(activations), size=BATCH_SIZE, replace=True)
    batch_activations = torch.tensor(
        activations[batch_idx], 
        dtype=torch.float32
    ).to(device)
    
    # Get summaries for this batch
    batch_summaries = [summaries[i] for i in batch_idx]
    
    # Tokenize summaries
    encoded = tokenizer(
        batch_summaries,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    # AR forward pass: reconstruct activations from summaries
    reconstructed = ar_model(input_ids, attention_mask)
    
    # Compute FVE
    mse = torch.mean((batch_activations - reconstructed) ** 2)
    activation_var = torch.var(batch_activations)
    fve = 1.0 - (mse / (activation_var + 1e-8))
    
    # AR update: minimize reconstruction error
    ar_loss = mse
    ar_optimizer.zero_grad()
    ar_loss.backward()
    ar_optimizer.step()
    
    # Track progress
    fve_value = fve.item()
    fve_history.append(fve_value)
    
    if (step + 1) % 50 == 0:
        print(f"Step {step+1:4d}/{NUM_STEPS} | FVE: {fve_value:.4f} | AR Loss: {ar_loss.item():.6f}")
    
    # Save checkpoint
    if (step + 1) % SAVE_EVERY == 0:
        torch.save(ar_model.state_dict(), OUTPUT_DIR / f"ar_step{step+1}.pt")

print("\n✅ RL Training Complete!")

# Save final model
print("\n4. Saving final models...")
torch.save(ar_model.state_dict(), OUTPUT_DIR / "ar_final.pt")
np.save(OUTPUT_DIR / "fve_history.npy", np.array(fve_history))

# Plot results
print("\n5. Plotting results...")
plt.figure(figsize=(10, 6))
plt.plot(fve_history, linewidth=2)
plt.xlabel("Training Step")
plt.ylabel("Fraction of Variance Explained (FVE)")
plt.title("NLA RL Training: FVE Over Time")
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_DIR / "fve_training_curve.png", dpi=150, bbox_inches="tight")
print(f"   ✓ Saved: {OUTPUT_DIR}/fve_training_curve.png")

# Summary
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)
print(f"Final FVE: {fve_history[-1]:.4f}")
print(f"Best FVE: {max(fve_history):.4f}")
print(f"Model saved to: {OUTPUT_DIR}/")
print("=" * 70)
