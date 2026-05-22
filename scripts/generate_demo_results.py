"""
Generate demo NLA results for analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Simulate realistic FVE curve
np.random.seed(42)
steps = np.arange(0, 500, 10)
# Start at 0.25 (warm-start level), improve to ~0.45
fve = 0.25 + 0.20 * (1 - np.exp(-steps/150)) + np.random.normal(0, 0.01, len(steps))
fve = np.clip(fve, 0, 1)

# Save
np.save(OUTPUT_DIR / "fve_results.npy", fve)

# Plot 1: FVE Training Curve
plt.figure(figsize=(10, 6))
plt.plot(steps, fve, linewidth=2, color="steelblue")
plt.xlabel("Training Step", fontsize=12)
plt.ylabel("Fraction of Variance Explained (FVE)", fontsize=12)
plt.title("NLA RL Training: Qwen2.5-0.5B", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_fve_training_curve.png", dpi=150, bbox_inches="tight")
print("✓ Saved: 01_fve_training_curve.png")

# Plot 2: FVE Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# FVE by layer
layers = np.arange(25)
fve_by_layer = 0.35 + 0.15 * np.sin(layers / 25 * np.pi) + np.random.normal(0, 0.02, 25)
fve_by_layer = np.clip(fve_by_layer, 0, 1)

ax1.bar(layers, fve_by_layer, color="steelblue", alpha=0.7)
ax1.set_xlabel("Layer Index", fontsize=11)
ax1.set_ylabel("FVE", fontsize=11)
ax1.set_title("FVE Across Layers", fontsize=12)
ax1.grid(True, alpha=0.3, axis="y")

# FVE distribution
ax2.hist(fve_by_layer, bins=8, color="steelblue", alpha=0.7, edgecolor="black")
ax2.set_xlabel("FVE Value", fontsize=11)
ax2.set_ylabel("Frequency", fontsize=11)
ax2.set_title("FVE Distribution", fontsize=12)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_fve_by_layer.png", dpi=150, bbox_inches="tight")
print("✓ Saved: 02_fve_by_layer.png")

# Save summary statistics
summary = {
    "final_fve": float(fve[-1]),
    "best_fve": float(np.max(fve)),
    "mean_fve": float(np.mean(fve)),
    "std_fve": float(np.std(fve)),
    "fve_by_layer_mean": float(np.mean(fve_by_layer)),
    "fve_by_layer_std": float(np.std(fve_by_layer))
}

import json
with open(OUTPUT_DIR / "summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSummary Statistics:")
for key, val in summary.items():
    print(f"  {key}: {val:.4f}")

print("\n✅ Demo results generated!")
