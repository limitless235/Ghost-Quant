import torch
from model.config import ModelConfig
from model.shared_transformer import GhostTransformer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

config = ModelConfig()
# Adjust vocab size to get closer to 100M if needed
# 50257 is standard, but let's check
model = GhostTransformer(config)
total_params = count_parameters(model)
print(f"Total Parameters: {total_params / 1e6:.2f}M")

# Break down
shared_params = sum(p.numel() for p in model.shared_block.parameters())
offset_params = sum(p.numel() for p in model.layers.parameters())
emb_params = sum(p.numel() for p in model.token_emb.parameters()) + model.pos_emb.numel()
head_params = sum(p.numel() for p in model.head.parameters())

print(f"Embedding: {emb_params / 1e6:.2f}M")
print(f"Shared Block: {shared_params / 1e6:.2f}M")
print(f"Offset Layers (12): {offset_params / 1e6:.2f}M")
print(f"Head: {head_params / 1e6:.2f}M")
