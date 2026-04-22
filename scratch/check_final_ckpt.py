import torch
from model.config import ModelConfig

ckpt = torch.load('checkpoint.pt', map_location='cpu', weights_only=False)
print(f"Final Step: {ckpt['step']}")
# We don't save the loss in the checkpoint, but we can see the config
print(f"Config: {ckpt['config']}")
