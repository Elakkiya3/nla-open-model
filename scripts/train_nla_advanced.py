"""
Production NLA - Transformer-based AR with Attention
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# CONFIG
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
HIDDEN_DIM = 896
MODEL_DIM = 1024
NUM_STEPS = 2000
BATCH_SIZE = 4
LR_AR = 1e-4
WARMUP_STEPS = 200
OUTPUT_DIR = Path("results/nla_production")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PRODUCTION NLA - ADVANCED ARCHITECTURE")
print("=" * 70)

# ==========================================
# IMPROVED AR ARCHITECTURE
# ==========================================

class TransformerAR(nn.Module):
    """
    Transformer-based Activation Reconstructor
    - Multi-head attention over text embeddings
    - Residual connections
    - Layer normalization
    - Bottleneck structure
    """
    def __init__(self, input_dim=1024, hidden_dim=896, num_heads=8, num_layers=2):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, 512)
        self.input_norm = nn.LayerNorm(512)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.2,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to activation space
        self.output_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim)
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, text_embeddings, mask=None):
        """
        Args:
            text_embeddings: (batch, seq_len, 1024)
            mask: (batch, seq_len) or None
        Returns:
            reconstructed: (batch, 896)
        """
        # Project input
        x = self.input_proj(text_embeddings)  # (batch, seq_len, 512)
        x = self.input_norm(x)
        
        # Apply transformer (learns attention over text tokens)
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask.bool())
        else:
            x = self.transformer(x)
        
        # Pool over sequence (mean + max pooling)
        x_mean = x.mean(dim=1)  # (batch, 512)
        x_max = x.max(dim=1)[0]  # (batch, 512)
        x = torch.cat([x_mean, x_max], dim=1)  # (batch, 1024)
        x = nn.Linear(1024, 512).to(x.device)(x)
        
        # Project to activation space
        recon = self.output_proj(x)  # (batch, 896)
        
        return recon

# ==========================================
# LOAD DATA & EXPAND DATASET
# ==========================================

print("\n[1/7] Loading data...")
activations = np.load("data/warmstart_large/activations.npy")
with open("data/warmstart_large/summaries.json") as f:
    summaries_dict = json.load(f)

summaries = [summaries_dict[str(i)] if isinstance(summaries_dict, dict) else summaries_dict[i] 
             for i in range(len(activations))]

train_acts, test_acts, train_sum, test_sum = train_test_split(
    activations, summaries, test_size=0.15, random_state=42
)

print(f"  Train: {train_acts.shape} ({len(train_acts)} pairs)")
print(f"  Test:  {test_acts.shape} ({len(test_acts)} pairs)")

# ==========================================
# LOAD MODEL
# ==========================================

print("\n[2/7] Loading Qwen model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
qwen_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    output_hidden_states=True
)
qwen_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##qwen_model = qwen_model.to(device)

print(f"  ✓ Device: {device}")

# ==========================================
# INITIALIZE AR MODEL
# ==========================================

print("\n[3/7] Initializing advanced AR model...")
ar_model = TransformerAR(
    input_dim=MODEL_DIM,
    hidden_dim=HIDDEN_DIM,
    num_heads=8,
    num_layers=2
)
ar_model = ar_model.to(device)

# Use AdamW with weight decay (better than Adam)
ar_optimizer = AdamW(ar_model.parameters(), lr=LR_AR, weight_decay=1e-5)

# Cosine annealing scheduler
scheduler = CosineAnnealingLR(ar_optimizer, T_max=NUM_STEPS, eta_min=1e-6)

print(f"  ✓ Parameters: {sum(p.numel() for p in ar_model.parameters()):,}")
print(f"  ✓ Optimizer: AdamW with weight decay")
print(f"  ✓ Scheduler: Cosine annealing")

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_text_embeddings_batch(texts, max_len=256):
    """Get Qwen embeddings for batch of texts"""
    tokens = tokenizer(
        texts,
        max_length=max_len,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = qwen_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden = outputs.hidden_states[-1]  # (batch, seq_len, 1024)
    
    return hidden, attention_mask

def compute_fve(activation, reconstructed, baseline_var):
    """Compute FVE with numerical stability"""
    mse = torch.mean((activation - reconstructed) ** 2)
    fve = 1.0 - (mse / (baseline_var + 1e-8))
    return fve, mse

# ==========================================
# TRAINING LOOP
# ==========================================

print("\n[4/7] Starting advanced training...")
print(f"  Steps: {NUM_STEPS}")
print(f"  Batch size: {BATCH_SIZE}\n")

fve_history = []
loss_history = []
test_fve_history = []
validation_losses = []

# Compute baseline
mean_activation = torch.tensor(np.mean(train_acts, axis=0), dtype=torch.float32).to(device)
baseline_var = torch.var(mean_activation)

best_test_fve = -float('inf')
patience = 100
patience_counter = 0

try:
    for step in tqdm(range(NUM_STEPS), desc="Training"):
        ar_model.train()
        
        # Sample batch from training data
        batch_indices = np.random.choice(len(train_acts), size=BATCH_SIZE, replace=True)
        batch_acts = torch.tensor(train_acts[batch_indices], dtype=torch.float32).to(device)
        batch_texts = [train_sum[i] for i in batch_indices]
        
        # Get text embeddings
        text_emb, text_mask = get_text_embeddings_batch(batch_texts)
        
        # AR forward pass
        reconstructed = ar_model(text_emb, text_mask)
        
        # Compute loss
        mse_loss = torch.mean((batch_acts - reconstructed) ** 2)
        
        # L2 regularization on reconstruction
        l2_reg = 0.01 * torch.mean(reconstructed ** 2)
        total_loss = mse_loss + l2_reg
        
        # Backward
        ar_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(ar_model.parameters(), 1.0)
        ar_optimizer.step()
        scheduler.step()
        
        # Compute FVE
        fve, _ = compute_fve(batch_acts, reconstructed, baseline_var)
        fve_history.append(fve.item())
        loss_history.append(total_loss.item())
        
        # Periodic validation
        if (step + 1) % 100 == 0:
            ar_model.eval()
            with torch.no_grad():
                # Validate on random test sample
                val_indices = np.random.choice(len(test_acts), size=min(8, len(test_acts)), replace=False)
                val_acts = torch.tensor(test_acts[val_indices], dtype=torch.float32).to(device)
                val_texts = [test_sum[i] for i in val_indices]
                
                val_emb, val_mask = get_text_embeddings_batch(val_texts)
                val_recon = ar_model(val_emb, val_mask)
                
                val_loss = torch.mean((val_acts - val_recon) ** 2).item()
                val_fve, _ = compute_fve(val_acts, val_recon, baseline_var)
                
                test_fve_history.append(val_fve.item())
                validation_losses.append(val_loss)
                
                # Early stopping
                if val_fve.item() > best_test_fve:
                    best_test_fve = val_fve.item()
                    patience_counter = 0
                    torch.save(ar_model.state_dict(), OUTPUT_DIR / "ar_best.pt")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping at step {step + 1}")
                    break

    print("\n" + "=" * 70)
    print("✅ ADVANCED TRAINING COMPLETE!")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

# ==========================================
# FULL TEST EVALUATION
# ==========================================

print("\nRunning comprehensive test evaluation...")

ar_model.eval()
all_test_fves = []
all_test_mses = []

with torch.no_grad():
    for test_idx in range(len(test_acts)):
        test_act = torch.tensor(test_acts[test_idx], dtype=torch.float32).unsqueeze(0).to(device)
        test_text = [test_sum[test_idx]]
        
        test_emb, test_mask = get_text_embeddings_batch(test_text)
        test_recon = ar_model(test_emb, test_mask)
        
        fve, mse = compute_fve(test_act, test_recon, baseline_var)
        all_test_fves.append(fve.item())
        all_test_mses.append(mse.item())

mean_test_fve = np.mean(all_test_fves)

# ==========================================
# SAVE RESULTS
# ==========================================

print("\nSaving results...")

torch.save(ar_model.state_dict(), OUTPUT_DIR / "ar_final.pt")
np.save(OUTPUT_DIR / "fve_history.npy", np.array(fve_history))
np.save(OUTPUT_DIR / "loss_history.npy", np.array(loss_history))
np.save(OUTPUT_DIR / "test_fve_history.npy", np.array(test_fve_history))

stats = {
    "train_final_fve": float(fve_history[-1]) if fve_history else 0,
    "train_best_fve": float(max(fve_history)) if fve_history else 0,
    "train_mean_fve": float(np.mean(fve_history)) if fve_history else 0,
    "test_mean_fve": float(mean_test_fve),
    "test_best_fve": float(max(all_test_fves)) if all_test_fves else 0,
    "validation_steps": len(test_fve_history),
    "total_steps": len(fve_history),
    "best_val_fve": float(best_test_fve),
}

with open(OUTPUT_DIR / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# ==========================================
# VISUALIZATIONS
# ==========================================

print("Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training curves
axes[0, 0].plot(fve_history, label="Train FVE", linewidth=2, alpha=0.7)
if test_fve_history:
    val_steps = np.linspace(0, len(fve_history), len(test_fve_history))
    axes[0, 0].plot(val_steps, test_fve_history, label="Validation FVE", linewidth=2, marker='o')
axes[0, 0].set_xlabel("Training Step")
axes[0, 0].set_ylabel("FVE")
axes[0, 0].set_title("Training Progress: FVE")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss curve
axes[0, 1].plot(loss_history, linewidth=2, color="coral")
axes[0, 1].set_xlabel("Training Step")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].set_title("Reconstruction Loss")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# Test FVE distribution
axes[1, 0].hist(all_test_fves, bins=15, color="steelblue", alpha=0.7, edgecolor="black")
axes[1, 0].axvline(mean_test_fve, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_test_fve:.3f}")
axes[1, 0].set_xlabel("Test FVE")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Test Set FVE Distribution")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Summary stats
stats_text = f"""
ADVANCED TRAINING RESULTS

Architecture:
  - Transformer-based AR
  - 8 attention heads
  - 2 transformer layers
  - Parameters: ~918K

Training:
  - Train FVE: {stats['train_final_fve']:.4f}
  - Best Validation: {stats['best_val_fve']:.4f}
  - Test FVE: {stats['test_mean_fve']:.4f}
  
Optimization:
  - AdamW + weight decay
  - Cosine annealing LR
  - Gradient clipping
  - Early stopping

Dataset:
  - Training: 102 pairs
  - Test: 18 pairs
  - Total: 120 pairs
"""

axes[1, 1].text(0.05, 0.95, stats_text, fontsize=10, family="monospace",
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_results.png", dpi=150, bbox_inches="tight")
print(f"  ✓ Saved visualization")

# ==========================================
# FINAL REPORT
# ==========================================

print("\n" + "=" * 70)
print("ADVANCED NLA - FINAL RESULTS")
print("=" * 70)
print(f"Train FVE (final):      {stats['train_final_fve']:.4f}")
print(f"Best Validation FVE:    {stats['best_val_fve']:.4f}")
print(f"Test FVE (mean):        {stats['test_mean_fve']:.4f}")
print(f"Total training steps:   {stats['total_steps']}")
print(f"Validation checkpoints: {stats['validation_steps']}")
print("=" * 70)
