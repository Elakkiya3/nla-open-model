"""
Train Activation Verbalizer (AV) to predict summaries from activations.

AV learns: activation (896-dim) -> text summary
"""

import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
HIDDEN_DIM = 896
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
OUTPUT_DIR = Path("checkpoints/av_warmstart")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ActivationDataset(Dataset):
    """Dataset of (activation, summary) pairs."""
    
    def __init__(self, activation_path, summary_path, tokenizer, max_length=256):
        self.activations = np.load(activation_path)
        with open(summary_path) as f:
            self.summaries = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        # Activation as tensor
        activation = torch.tensor(self.activations[idx], dtype=torch.float32)
        
        # Summary as tokens
        summary = self.summaries[idx]
        encoded = self.tokenizer(
            summary,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "activation": activation,
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": encoded["input_ids"].squeeze()
        }

def train_av():
    print("=" * 60)
    print("Training Activation Verbalizer (AV)")
    print("=" * 60)
    
    # Load tokenizer and model
    print("\n1. Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto"
    )
    
    # Load dataset
    print("2. Loading (activation, summary) pairs...")
    dataset = ActivationDataset(
        "data/warmstart/activations.npy",
        "data/warmstart/summaries.json",
        tokenizer
    )
    print(f"   Dataset size: {len(dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=20,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=2,
        warmup_steps=50,
    )
    
    # Trainer
    print("\n3. Starting supervised fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    trainer.train()
    
    # Save final checkpoint
    print("\n4. Saving model...")
    model.save_pretrained(OUTPUT_DIR / "final")
    tokenizer.save_pretrained(OUTPUT_DIR / "final")
    
    print(f"✅ AV warm-start complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_av()
