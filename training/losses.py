import torch
import torch.nn as nn

def composite_sparsity_loss(task_loss, model, lmbda):
    """
    Computes task_loss + lambda * sparsity_loss.
    Sparsity loss here is the L1 norm of the unmasked gate weights to encourage
    sparse-friendly weight distributions.
    """
    sparsity_loss = 0
    count = 0
    
    # Target the shared gating weights directly
    # In our model, the first half of mlp_fc_shared is used as the gate
    w_gate, _ = model.shared_block.mlp_fc_shared.chunk(2, dim=0)
    sparsity_loss = w_gate.abs().mean()
    
    return task_loss + lmbda * sparsity_loss
