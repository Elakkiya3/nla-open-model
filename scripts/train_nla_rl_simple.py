"""
Simplified NLA RL Training - Faster version
"""

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import Adam
import matplotlib.pyplot as plt

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
HIDDEN_DIM = 896
MODEL_DIM = 1024
NUM_STEPS = 200
OUTPUT_DIR = Path("checkpoints/nla_rl")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
activations = np.load("data/warmstart/activations.npy")
with open("data/warmstart/summaries.json") as f:
    summaries = json.load(f)

print("Initializing model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
ar_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
ar_head = torch.nn.Linear(MODEL_DIM, HIDDEN_DIM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ar_model.to(device)
ar_head.to(device)

optimizer = Adam(list(ar_model.parameters()) + list(ar_head.parameters()), lr=1e-4)

print(f"\nTraining {NUM_STEPS} steps...\n")

fve_history = []

for step in range(NUM_STEPS):
    idx = np.random.randint(0, len(activations))
    activation = torch.tensor(activations[idx:idx+1], dtype=torch.float32).to(device)
    
    summary = summaries[idx]
    encoded = tokenizer(summary, return_tensors="pt", max_length=256, truncation=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = ar_model(**encoded, output_hidden_states=True)
    
    hidden = outputs.hidden_states[-1][:, -1, :]
    reconstructed = ar_head(hidden)
    
    loss = torch.nn.functional.mse_loss(reconstructed, activation)
    
    mse = loss.item()
    fve = 1.0 - (mse / (np.var(activations) + 1e-8))
    fve_history.append(fve)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 50 == 0:
        print(f"Step {step+1}/{NUM_STEPS} | FVE: {fve:.4f}")

print("\n✅ Training Complete!")
print(f"Final FVE: {fve_history[-1]:.4f}")

torch.save(ar_head.state_dict(), OUTPUT_DIR / "ar_final.pt")
np.save(OUTPUT_DIR / "fve_history.npy", np.array(fve_history))

plt.figure(figsize=(10, 6))
plt.plot(fve_history)
plt.xlabel("Step")
plt.ylabel("FVE")
plt.title("NLA RL Training")
plt.savefig(OUTPUT_DIR / "fve_curve.png")
print("Saved: fve_curve.png")
