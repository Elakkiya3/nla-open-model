"""
Standalone Activation Autoencoder Baseline
Proper reconstruction baseline before NLA language bottleneck
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# CONFIG
# ==========================================

HIDDEN_DIM = 896
LATENT_DIM = 128

NUM_EPOCHS = 300
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

OUTPUT_DIR = Path("results/autoencoder_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("AUTOENCODER BASELINE TRAINING")
print("=" * 70)

# ==========================================
# LOAD DATA
# ==========================================

print("\n[1/5] Loading activation data...")

activations = np.load("data/warmstart/activations.npy")

with open("data/warmstart/summaries.json") as f:
    summaries = json.load(f)

train_acts, test_acts = train_test_split(
    activations,
    test_size=0.2,
    random_state=42
)

print(f"Train activations: {train_acts.shape}")
print(f"Test activations:  {test_acts.shape}")

train_tensor = torch.tensor(train_acts, dtype=torch.float32)
test_tensor = torch.tensor(test_acts, dtype=torch.float32)

# ==========================================
# AUTOENCODER MODEL
# ==========================================

print("\n[2/5] Building autoencoder...")

class ActivationAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 512),
            nn.ReLU(),

            nn.Linear(512, LATENT_DIM),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.ReLU(),

            nn.Linear(512, HIDDEN_DIM)
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        return reconstructed, latent

model = ActivationAutoencoder().to(device)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

print(f"✓ Model initialized")
print(f"✓ Parameters: {sum(p.numel() for p in model.parameters())}")

# ==========================================
# TRAINING
# ==========================================

print("\n[3/5] Training autoencoder...")

loss_history = []
fve_history = []

model.train()

for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):

    permutation = torch.randperm(train_tensor.size(0))

    epoch_loss = 0

    for i in range(0, train_tensor.size(0), BATCH_SIZE):

        indices = permutation[i:i+BATCH_SIZE]

        batch = train_tensor[indices].to(device)

        reconstructed, latent = model(batch)

        mse = torch.mean((batch - reconstructed) ** 2)

        optimizer.zero_grad()

        mse.backward()

        optimizer.step()

        epoch_loss += mse.item()

    # Compute FVE
    activation_var = torch.var(train_tensor.to(device))

    fve = 1.0 - (mse / (activation_var + 1e-8))

    loss_history.append(epoch_loss)
    fve_history.append(fve.item())

print("\n✓ Training complete")

# ==========================================
# TEST EVALUATION
# ==========================================

print("\n[4/5] Evaluating on test set...")

model.eval()

with torch.no_grad():

    test_batch = test_tensor.to(device)

    reconstructed, latent = model(test_batch)

    test_mse = torch.mean((test_batch - reconstructed) ** 2)

    test_var = torch.var(test_batch)

    test_fve = 1.0 - (test_mse / (test_var + 1e-8))

print(f"\nTEST FVE: {test_fve.item():.4f}")

# ==========================================
# SAVE RESULTS
# ==========================================

print("\n[5/5] Saving results...")

torch.save(
    model.state_dict(),
    OUTPUT_DIR / "autoencoder.pt"
)

stats = {
    "test_fve": float(test_fve.item()),
    "final_loss": float(test_mse.item()),
    "epochs": NUM_EPOCHS,
    "latent_dim": LATENT_DIM
}

with open(OUTPUT_DIR / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# Plot training curves

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(loss_history)
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")

axes[1].plot(fve_history)
axes[1].set_title("Training FVE")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("FVE")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "training_curves.png",
    dpi=150
)

print(f"✓ Model saved")
print(f"✓ Curves saved")

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"Test FVE: {test_fve.item():.4f}")
print(f"Final Loss: {test_mse.item():.4f}")

print("=" * 70)