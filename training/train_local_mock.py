import torch
import torch.optim as optim
from model.config import ModelConfig
from model.shared_transformer import GhostTransformer, mrl_loss, get_spectral_norm
import time

def train_local():
    config = ModelConfig()
    # For local mock, we can use a smaller vocab or seq_len if desired, 
    # but let's stick to the config.
    device = torch.device(config.device)
    
    print(f"Initializing Ghost-Quant model (~100M params) on {device}...")
    model = GhostTransformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Synthetic Data Generator (Mocking C4)
    def mock_data_generator(batch_size, seq_len, vocab_size):
        while True:
            # Random tokens for testing
            yield torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    batch_size = 4
    seq_len = 128 # Smaller for mock speed
    gen = mock_data_generator(batch_size, seq_len + 1, config.vocab_size)
    
    print("Starting training loop (MOCK MODE)...")
    step = 0
    try:
        while step < 1000:
            input_ids_raw = next(gen)
            
            # Targets are shifted input_ids
            targets = input_ids_raw[:, 1:].contiguous()
            input_ids = input_ids_raw[:, :-1].contiguous()
            
            optimizer.zero_grad()
            
            # Forward pass (returns final hidden states)
            hidden_states = model(input_ids)
            
            # Compute MRL loss sequentially to save memory
            loss = mrl_loss(hidden_states, targets, model)
            
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f}")
                
            # Spectral Norm Monitor every 100 steps for mock
            if step % 100 == 0:
                spectral_metrics = get_spectral_norm(model.shared_block, steps=3)
                print(f"--- Spectral Norms (Step {step}) ---")
                for name, sigma in spectral_metrics.items():
                    print(f"  {name}: {sigma:.4f}")
            
            step += 1
            
    except KeyboardInterrupt:
        print("Training stopped by user.")

if __name__ == "__main__":
    train_local()
