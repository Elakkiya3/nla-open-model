
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32)

model.eval()

text = "Machine learning helps researchers understand neural networks."

inputs = tokenizer(text, return_tensors="pt")

print("Running model...")

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True
    )

hidden_states = outputs.hidden_states

print("\nNumber of layers:")
print(len(hidden_states))

print("\nShape of final layer activations:")
print(hidden_states[-1].shape)

print("\nExample activation vector:")
print(hidden_states[-1][0, 0, :10])

# SAve Activationns
os.makedirs("data", exist_ok=True)

final_activations = hidden_states[-1].cpu().numpy()

save_path = "data/sample_activations.npy"

np.save(save_path, final_activations)

print(f"\nSaved activations to: {save_path}")