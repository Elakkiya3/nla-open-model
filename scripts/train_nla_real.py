"""
NLA RL Training - Simplified for Stability
Optimized for working on free-tier GPU with real data
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
from sklearn.model_selection import train_test_split


# CONFIG
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
HIDDEN_DIM = 896
MODEL_DIM = 1024
NUM_STEPS = 300  # Reduced from 500 for faster training
BATCH_SIZE = 1   # Reduced to 1 to avoid memory issues
LR_AV = 1e-5
LR_AR = 1e-4
SAVE_EVERY = 30
OUTPUT_DIR = Path("results/nla_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("NLA RL TRAINING - REAL DATA")
print("=" * 70)

# LOAD DATA
print("\n[1/5] Loading warm-start data...")

activations = np.load("data/warmstart_unnormalized/activations.npy")
with open("data/warmstart_unnormalized/summaries.json") as f:
    summaries_dict = json.load(f)
    summaries = list(summaries_dict.values())  # Convert dict to list

train_acts, test_acts, train_sum, test_sum = train_test_split(
    activations,
    summaries,
    test_size=0.2,
    random_state=42
)
# Add after loading data:
# Ensure activations have reasonable variance
if np.std(activations) < 0.01:
    print("⚠️  WARNING: Activations are very small, likely normalized")
    print("   Regenerate data WITHOUT normalization")
    exit()

print(f"Train activations: {train_acts.shape}")
print(f"Test activations: {test_acts.shape}")

print(f"  ✓ Loaded {len(activations)} pairs")
print(f"  ✓ Shape: {activations.shape}")
assert len(activations) == len(summaries), "Mismatch in data"

# TOKENIZER
# TOKENIZER + MODEL
print("\n[2/5] Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

model.eval()

print(f"  ✓ Tokenizer loaded")
print(f"  ✓ Qwen model loaded")

# AR MODEL (Simple, Memory-Efficient)
print("\n[3/5] Initializing AR model...")

class SimpleAR(nn.Module):
    """Lightweight AR model to avoid memory issues"""
    def __init__(self):
        super().__init__()
        # Use smaller projection to save memory
        self.fc1 = nn.Linear(896, 512)
        self.fc2 = nn.Linear(512, HIDDEN_DIM)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, text_embedding):
        x = torch.relu(self.fc1(text_embedding))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

ar_model = SimpleAR()
ar_optimizer = Adam(ar_model.parameters(), lr=LR_AR)

print(f"  ✓ AR model initialized")
print(f"  ✓ Parameters: {sum(p.numel() for p in ar_model.parameters())}")

# TRAINING
print("\n[4/5] Starting RL Training...")
print(f"  Steps: {NUM_STEPS}")
print(f"  Batch size: {BATCH_SIZE}\n")

fve_history = []
loss_history = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ar_model = ar_model.to(device)
global_var = np.var(train_acts)

try:
    for step in tqdm(range(NUM_STEPS), desc="Training"):
        # Sample activation
        idx = np.random.randint(0, len(train_acts))
        activation = torch.tensor(
            train_acts[idx], 
            dtype=torch.float32
        ).unsqueeze(0).to(device)
        
        # Get summary
        summary = train_sum[idx]
        
        # Encode summary to simple embedding
        tokens = tokenizer(
            summary,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        hidden = outputs.hidden_states[-1]   # [1, seq_len, 896]

        # Mean pooling over sequence
        text_embedding = hidden.mean(dim=1)  # [1, 896]
        
        # AR forward
        reconstructed = ar_model(text_embedding)
        
        # Compute FVE
        mse = torch.mean((activation - reconstructed) ** 2)
        fve = 1.0 - (mse.item() / (global_var + 1e-8))
        ## activation_var = torch.var(activation)
        ##fve = 1.0 - (mse / (activation_var + 1e-8))
        
        # AR update
        ar_loss = mse
        ar_optimizer.zero_grad()
        ar_loss.backward()
        torch.nn.utils.clip_grad_norm_(ar_model.parameters(), 1.0)
        ar_optimizer.step()
        
        # Track
        fve_history.append(fve.item())
        loss_history.append(ar_loss.item())
        
        # Save checkpoint
        if (step + 1) % SAVE_EVERY == 0:
            torch.save(ar_model.state_dict(), OUTPUT_DIR / f"ar_step{step+1}.pt")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)

except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
# ==========================================
# TEST SET EVALUATION
# ==========================================

print("\nEvaluating on TEST set...")

ar_model.eval()

test_fves = []
test_mses = []

with torch.no_grad():

    # Mean activation baseline
    mean_activation = np.mean(train_acts, axis=0)

    baseline_errors = []

    for i in range(len(test_acts)):

        activation = torch.tensor(
            test_acts[i],
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        summary = test_sum[i]

        tokens = tokenizer(
            summary,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        hidden = outputs.hidden_states[-1]

        text_embedding = hidden.mean(dim=1)

        # AR forward

        reconstructed = ar_model(text_embedding)

        mse = torch.mean(
            (activation - reconstructed) ** 2
        )
        baseline_mse_sample = np.mean(
   
            (test_acts[i] - mean_activation) ** 2
        )

        fve = 1.0 - (
            mse.item() / (baseline_mse_sample + 1e-8)
        )
        ##activation_var = torch.var(activation)

        ## fve = 1.0 - (mse / (activation_var + 1e-8))

        test_fves.append(fve.item())
        test_mses.append(mse.item())

        # Baseline error
        baseline_mse_sample = np.mean(
            (test_acts[i] - mean_activation) ** 2
        )

        baseline_errors.append(baseline_mse_sample)

mean_test_fve = np.mean(test_fves)
mean_test_mse = np.mean(test_mses)
baseline_mse = np.mean(baseline_errors)

print(f"\nTEST FVE: {mean_test_fve:.4f}")
print(f"TEST MSE: {mean_test_mse:.4f}")
print(f"BASELINE MSE: {baseline_mse:.4f}")
# SAVE RESULTS
print("\n[5/5] Saving results...")

# Save model
torch.save(ar_model.state_dict(), OUTPUT_DIR / "ar_final.pt")
np.save(OUTPUT_DIR / "fve_history.npy", np.array(fve_history))
np.save(OUTPUT_DIR / "loss_history.npy", np.array(loss_history))

# Summary stats
stats = {
    "final_fve": float(fve_history[-1]) if fve_history else 0,
    "best_fve": float(max(fve_history)) if fve_history else 0,
    "mean_fve": float(np.mean(fve_history)) if fve_history else 0,
    "std_fve": float(np.std(fve_history)) if fve_history else 0,
    "test_fve": float(mean_test_fve),
    "test_mse": float(mean_test_mse),
    "baseline_mse": float(baseline_mse),
    "num_steps": len(fve_history),

}

with open(OUTPUT_DIR / "training_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"  ✓ Model saved: {OUTPUT_DIR}/ar_final.pt")
print(f"  ✓ FVE history saved")
print(f"  ✓ Stats saved")

# PLOT RESULTS
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# FVE curve
axes[0, 0].plot(fve_history, linewidth=2, color="steelblue")
axes[0, 0].set_xlabel("Step")
axes[0, 0].set_ylabel("FVE")
axes[0, 0].set_title("FVE Over Training")
axes[0, 0].grid(True, alpha=0.3)

# Loss curve
axes[0, 1].plot(loss_history, linewidth=2, color="coral")
axes[0, 1].set_xlabel("Step")
axes[0, 1].set_ylabel("MSE Loss")
axes[0, 1].set_title("Reconstruction Loss")
axes[0, 1].grid(True, alpha=0.3)

# FVE distribution
axes[1, 0].hist(fve_history, bins=15, color="steelblue", alpha=0.7, edgecolor="black")
axes[1, 0].set_xlabel("FVE Value")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("FVE Distribution")
axes[1, 0].grid(True, alpha=0.3, axis="y")

# Stats text
stats_text = f"""
TRAINING STATISTICS

Final FVE:      {stats['final_fve']:.4f}
Best FVE:       {stats['best_fve']:.4f}
Mean FVE:       {stats['mean_fve']:.4f}
Std Dev:        {stats['std_fve']:.4f}

Steps completed: {stats['num_steps']}
Batch size:      {BATCH_SIZE}
Model:          {MODEL_NAME}
"""

axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_results.png", dpi=150, bbox_inches="tight")
print(f"  ✓ Saved: training_results.png")

# FINAL SUMMARY
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Final FVE:        {stats['final_fve']:.4f}")
print(f"Best FVE:         {stats['best_fve']:.4f}")
print(f"Mean FVE:         {stats['mean_fve']:.4f}")
print(f"Test FVE:         {stats['test_fve']:.4f}")
# Compare against mean-prediction baseline
print(f"Baseline MSE:     {stats['baseline_mse']:.4f}")
## print(f"Baseline MSE: {baseline_mse:.4f}")
print(f"Total steps:      {stats['num_steps']}")
print(f"Results saved to: {OUTPUT_DIR}/")
print("=" * 70)
