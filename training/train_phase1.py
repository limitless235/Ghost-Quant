import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from model.config import ModelConfig
from model.shared_transformer import GhostTransformer, mrl_loss, get_spectral_norm
import time
import os
import gc

# Set MPS watermark to 0.0 to allow full unified memory usage
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def save_checkpoint(model, optimizer, step, config, path="checkpoint.pt"):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, path)
    print(f"Checkpoint saved at step {step}")

def load_checkpoint(model, optimizer, path="checkpoint.pt"):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='mps', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step']
    return 0

def train():
    config = ModelConfig()
    device = torch.device(config.device)
    
    # Initialize Model
    model = GhostTransformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Resume from checkpoint if exists
    start_step = load_checkpoint(model, optimizer)
    if start_step > 0:
        print(f"Resuming from step {start_step}")
    
    # Load Dataset (C4 en streaming)
    print("Loading C4 dataset...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    if start_step > 0:
        print(f"Fast-forwarding dataset by {start_step} examples...")
        # Skip at the dataset level to avoid pulling and discarding thousands of HTTP streams,
        # which causes semaphore leaks and memory bloat.
        dataset = dataset.skip(start_step)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Simple data generator for streaming
    def data_generator():
        for example in dataset:
            text = example['text']
            tokens = tokenizer(text, truncation=True, max_length=config.max_seq_len + 1, return_tensors="pt")
            if tokens.input_ids.size(1) < 2:
                continue
            yield tokens.input_ids[0]

    # Manual batching for streaming dataset
    batch_size = 1
    gen = data_generator()
    
    step = start_step
    print(f"Starting training loop at step {step}...")
    while True:
        batch_ids = []
        try:
            for _ in range(batch_size):
                batch_ids.append(next(gen))
        except StopIteration:
            break
            
        # Pad batch
        max_len = max(len(ids) for ids in batch_ids)
        input_ids = torch.stack([torch.cat([ids, torch.full((max_len - len(ids),), tokenizer.pad_token_id)]) for ids in batch_ids])
        input_ids = input_ids.to(device)
        
        # Targets are shifted input_ids
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        optimizer.zero_grad()
        
        # Forward pass (returns final hidden states)
        hidden_states = model(input_ids)
        
        # Compute MRL loss sequentially to save memory
        loss = mrl_loss(hidden_states, targets, model)
        
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
        # Spectral Norm Monitor and Checkpoint every 500 steps
        if step > 0 and step % 500 == 0:
            spectral_metrics = get_spectral_norm(model.shared_block, steps=3)
            print(f"--- Spectral Norms (Step {step}) ---")
            for name, sigma in spectral_metrics.items():
                print(f"  {name}: {sigma:.4f}")
            save_checkpoint(model, optimizer, step, config)
        
        # Aggressive memory cleanup EVERY STEP
        del loss, hidden_states, targets, input_ids
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        step += 1
        if step > 10000: # Limit for now
            break

if __name__ == "__main__":
    train()
