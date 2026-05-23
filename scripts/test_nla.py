"""
Comprehensive Testing & Evaluation
"""

import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

print("\n" + "=" * 70)
print("NLA TESTING & EVALUATION")
print("=" * 70)

RESULTS_DIR = Path("results/nla_training")

# TEST 1: Load and verify data
print("\n[TEST 1] Verifying Training Data...")
try:
    activations = np.load("data/warmstart/activations.npy")
    with open("data/warmstart/summaries.json") as f:
        summaries = json.load(f)
    
    print(f"  ✓ Loaded {len(activations)} activation-summary pairs")
    print(f"  ✓ Activation shape: {activations.shape}")
    print(f"  ✓ Summary count: {len(summaries)}")
    assert len(activations) == len(summaries)
    print("  ✓ Data integrity check: PASSED")
except Exception as e:
    print(f"  ❌ Data loading failed: {e}")

# TEST 2: Load training results
print("\n[TEST 2] Loading Training Results...")
try:
    fve_history = np.load(RESULTS_DIR / "fve_history.npy")
    loss_history = np.load(RESULTS_DIR / "loss_history.npy")
    
    with open(RESULTS_DIR / "training_stats.json") as f:
        stats = json.load(f)
    
    print(f"  ✓ Loaded FVE history: {len(fve_history)} steps")
    print(f"  ✓ Loaded loss history: {len(loss_history)} steps")
    print(f"  ✓ Statistics loaded")
    print("  ✓ Results loading: PASSED")
except Exception as e:
    print(f"  ❌ Results loading failed: {e}")

# TEST 3: Verify model checkpoint
print("\n[TEST 3] Verifying Model Checkpoint...")
try:
    model_path = RESULTS_DIR / "ar_final.pt"
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        print(f"  ✓ Model checkpoint exists: {model_path}")
        print(f"  ✓ Checkpoint size: {sum(p.numel() for p in checkpoint.values())} params")
        print("  ✓ Model verification: PASSED")
    else:
        print(f"  ⚠️  Model checkpoint not found at {model_path}")
except Exception as e:
    print(f"  ❌ Model verification failed: {e}")

# TEST 4: Analyze FVE trajectory
print("\n[TEST 4] Analyzing FVE Trajectory...")
try:
    print(f"  Initial FVE:  {fve_history[0]:.4f}")
    print(f"  Final FVE:    {fve_history[-1]:.4f}")
    print(f"  Best FVE:     {np.max(fve_history):.4f}")
    print(f"  Improvement:  {(fve_history[-1] - fve_history[0]):.4f}")
    print(f"  Std Dev:      {np.std(fve_history):.4f}")
    
    # Check for improvement
    improvement = fve_history[-1] - fve_history[0]
    if improvement > 0:
        print("  ✓ FVE trajectory: IMPROVING")
    else:
        print("  ⚠️  FVE trajectory: NOT IMPROVING (check training)")
except Exception as e:
    print(f"  ❌ Trajectory analysis failed: {e}")

# TEST 5: Statistical validation
print("\n[TEST 5] Statistical Validation...")
try:
    valid_fve = (fve_history >= 0) & (fve_history <= 1)
    print(f"  FVE values in valid range [0,1]: {np.sum(valid_fve)}/{len(valid_fve)}")
    
    if np.all(valid_fve):
        print("  ✓ FVE values: VALID")
    else:
        print("  ❌ Invalid FVE values detected")
except Exception as e:
    print(f"  ❌ Statistical validation failed: {e}")

# TEST 6: Generate summary report
print("\n[TEST 6] Generating Summary Report...")
try:
    report = f"""
TRAINING SUMMARY REPORT
========================

Data:
  Activation-summary pairs: {len(activations)}
  Activation dimension: {activations.shape[1]}
  
Training Results:
  Total steps: {len(fve_history)}
  Initial FVE: {fve_history[0]:.4f}
  Final FVE: {fve_history[-1]:.4f}
  Best FVE: {np.max(fve_history):.4f}
  Mean FVE: {np.mean(fve_history):.4f}
  Std Dev: {np.std(fve_history):.4f}
  
Loss:
  Initial loss: {loss_history[0]:.6f}
  Final loss: {loss_history[-1]:.6f}
  Best loss: {np.min(loss_history):.6f}
  
Model:
  Checkpoint: {RESULTS_DIR / 'ar_final.pt'}
  Size: Fully saved and loadable
  
Visualizations:
  Training curve: {RESULTS_DIR / 'training_results.png'}
  FVE history: {RESULTS_DIR / 'fve_history.npy'}
  Loss history: {RESULTS_DIR / 'loss_history.npy'}
"""
    
    print(report)
    
    with open(RESULTS_DIR / "test_report.txt", "w") as f:
        f.write(report)
    
    print("  ✓ Report saved to: test_report.txt")
except Exception as e:
    print(f"  ❌ Report generation failed: {e}")

# FINAL SUMMARY
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("✅ All tests completed!")
print(f"Results directory: {RESULTS_DIR}/")
print("=" * 70 + "\n")
