"""
Train Activation Reconstructor (AR) to predict activations from summaries.

AR learns: text summary -> activation (896-dim)
"""

import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
HIDDEN_DIM = 896
MODEL_DIM = 1024  # Qwen hidden dimension
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
OUTPUT_DIR = Path("checkpoints/ar_warmstart")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class SummaryDataset(Dataset):
    """Dataset of (summary, activation) pairs."""
    
    def __init__(self, activation_path, summary_path, tokenizer, max_length=256):
        self.activations = np.load(activation_path)
        with open(summary_path) as f:
            self.summaries = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        summary = self.summaries[idx]
        activation = torch.tensor(self.activations[idx], dtype=torch.float32)
        
        encoded = self.tokenizer(
            summary,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "target_activation": activation
        }

class ActivationReconstructorHead(nn.Module):
    """
    AR architecture:
    1. Process text through Qwen
    2. Extract final token's hidden state
    3. Project to 896-dimensional activation space
    """
    
    def __init__(self, model_name, hidden_dim=HIDDEN_DIM, model_dim=MODEL_DIM):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="auto"
        )
        self.reconstruction_head = nn.Linear(model_dim, hidden_dim)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            reconstructed_activation: [batch, 896]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Get hidden state of final token
        final_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch, 1024]
        
        # Project to activation space
        reconstructed = self.reconstruction_head(final_hidden)  # [batch, 896]
        
        return reconstructed

def train_ar():
    print("=" * 60)
    print("Training Activation Reconstructor (AR)")
    print("=" * 60)
    
    # Load
    print("\n1. Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ar_model = ActivationReconstructorHead(MODEL_NAME)
    
    print("2. Loading dataset...")
    dataset = SummaryDataset(
        "data/warmstart/activations.npy",
        "data/warmstart/summaries.json",
        tokenizer
    )
    print(f"   Dataset size: {len(dataset)}")
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        ar_model.parameters(),
        lr=LEARNING_RATE
    )
    
    # Training loop
    print("\n3. Starting supervised training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ar_model = ar_model.to(device)
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_activation = batch["target_activation"].to(device)
            
            # Forward pass
            reconstructed = ar_model(input_ids, attention_mask)
            
            # Loss (MSE between predicted and target activation)
            loss = torch.nn.functional.mse_loss(reconstructed, target_activation)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"   Epoch {epoch+1}/{NUM_EPOCHS}, "
                      f"Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Avg Loss: {avg_loss:.4f}")
        
        print(f"Epoch {epoch+1} complete. Avg Loss: {total_loss/num_batches:.4f}")
    
    # Save
    print("\n4. Saving model...")
    torch.save(ar_model.state_dict(), OUTPUT_DIR / "ar_model.pt")
    
    print(f"✅ AR warm-start complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_ar()
