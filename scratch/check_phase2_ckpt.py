import torch
from model.config import ModelConfig
from training.train_phase2 import check_sparsity_metrics

ckpt = torch.load('checkpoint_phase2.pt', map_location='cpu', weights_only=False)
model_state = ckpt['model_state_dict']
step = ckpt['step']
config = ckpt['config']

from model.shared_transformer import GhostTransformer
model = GhostTransformer(config)
model.load_state_dict(model_state)

metrics = check_sparsity_metrics(model)

print(f"Step: {step}")
print(f"Gate Collapse: {metrics['gate_collapse_pct']:.2f}%")
print(f"Active Sparsity: {metrics['active_sparsity']:.1f}%")
