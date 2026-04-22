import torch
import torch.optim as optim
from datasets import load_dataset
from transformers import GPT2Tokenizer
from model.config import ModelConfig
from model.shared_transformer import GhostTransformer, mrl_loss, get_spectral_norm
from model.sparse_gate import apply_2_4_mask_ste
from training.losses import composite_sparsity_loss
import time
import os
import gc

# Set MPS watermark
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def save_checkpoint(model, optimizer, step, config, path="checkpoint_phase2.pt"):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, path)
    print(f"Phase 2 Checkpoint saved at step {step}")

def load_phase1_checkpoint(model, path="checkpoint.pt"):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='mps', weights_only=False)
        # Load weights into the model
        # Note: Phase 2 model has the same parameter names as Phase 1, 
        # so this should work even with the architectural change in forward()
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"Loaded Phase 1 weights from {path}")
        return checkpoint['step']
    else:
        print(f"Phase 1 checkpoint not found at {path}")
        return 0

def check_sparsity_metrics(model):
    """
    Computes gate collapse and active sparsity statistics.
    """
    total_blocks = 0
    collapsed_blocks = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'mlp_fc_offset.A' in name or 'mlp_fc_offset.B' in name:
                # This is more complex since we apply mask to the EFFECTIVE weight
                # For simplicity, we'll check the base shared weight if we want global metrics
                pass
        
        # Let's check the shared weight which is the primary source of gates
        w = model.shared_block.mlp_fc_shared
        # We chunk it like in forward: (2*d_model, d_model)
        w_gate, _ = w.chunk(2, dim=0)
        
        # Reshape into blocks of 4
        x = w_gate.view(-1, 4)
        total_blocks = x.size(0)
        
        # Identify elements that would be kept (top 2 by magnitude)
        _, indices = torch.topk(x.abs(), k=2, dim=-1, largest=True)
        
        # Check if the kept elements are essentially zero (collapse)
        # We'll use a small epsilon
        eps = 1e-6
        vals = torch.gather(x, 1, indices)
        collapsed_mask = (vals.abs() < eps).all(dim=-1)
        collapsed_blocks = collapsed_mask.sum().item()
        
    return {
        "gate_collapse_pct": (collapsed_blocks / total_blocks) * 100,
        "active_sparsity": 50.0  # Fixed for 2:4
    }

def load_checkpoint(model, optimizer, path="checkpoint_phase2.pt"):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='mps', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resuming Phase 2 from step {checkpoint['step']}")
        return checkpoint['step']
    return None

def train():
    config = ModelConfig()
    config.use_sparse_glu = True  # Enable for Phase 2
    device = torch.device(config.device)
    
    # Initialize Model
    model = GhostTransformer(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5) # Slightly lower LR for fine-tuning
    
    # 1. Try to resume from Phase 2 checkpoint
    print("Checking for Phase 2 checkpoint...")
    start_step = load_checkpoint(model, optimizer)
    
    # 2. If no Phase 2 checkpoint, load Phase 1 weights and start fresh
    if start_step is None:
        print("No Phase 2 checkpoint found. Loading Phase 1 weights...")
        load_phase1_checkpoint(model)
        start_step = 0
    
    # Load Dataset (C4 en streaming)
    print("Initializing Hugging Face stream (allenai/c4)...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    if start_step > 0:
        print(f"Fast-forwarding dataset by {start_step} examples (this may take 1-3 minutes)...")
        dataset = dataset.skip(start_step)

    print("Loading tokenizer...")
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

    print("Preparing data generator...")
    batch_size = 1
    gen = data_generator()
    
    total_steps = 10000
    anneal_steps = int(0.3 * total_steps)
    target_lambda = config.sparse_lambda
    
    print(f"Starting Phase 2 training loop at step {start_step}...")
    step = start_step
    while step < total_steps:
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
        
        targets = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        # Calculate annealed lambda
        current_lambda = target_lambda * min(1.0, step / anneal_steps)
        
        optimizer.zero_grad()
        
        # Forward pass
        hidden_states = model(input_ids)
        
        # Task Loss (MRL)
        task_loss = mrl_loss(hidden_states, targets, model)
        
        # Composite Loss (Task + Lambda * Sparsity)
        loss = composite_sparsity_loss(task_loss, model, current_lambda)
        
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Task: {task_loss.item():.4f} | Lambda: {current_lambda:.4f}")
            
        if step % 500 == 0:
            metrics = check_sparsity_metrics(model)
            print(f"--- Sparsity Metrics (Step {step}) ---")
            print(f"  Gate Collapse: {metrics['gate_collapse_pct']:.2f}%")
            print(f"  Active Sparsity: {metrics['active_sparsity']:.1f}%")
            save_checkpoint(model, optimizer, step, config)
        
        # Cleanup
        del loss, task_loss, hidden_states, targets, input_ids
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        step += 1

if __name__ == "__main__":
    train()
