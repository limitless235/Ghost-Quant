import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_2_4_mask_ste(weight):
    """
    Applies 2:4 structured sparsity mask to the weight tensor.
    Expects weight to have a dimension divisible by 4 along the channel dim (last dim).
    Uses Straight-Through Estimator (STE) for gradients.
    """
    # Shape: (..., N) -> (..., N/4, 4)
    original_shape = weight.shape
    x = weight.view(-1, 4)
    
    # Identify 2 smallest magnitude elements in each block of 4
    _, indices = torch.topk(x.abs(), k=2, dim=-1, largest=False)
    
    # Create mask: 1 for elements to keep, 0 for elements to remove
    mask = torch.ones_like(x)
    mask.scatter_(1, indices, 0)
    
    # Apply mask
    sparse_weight = x * mask
    
    # Reshape back
    sparse_weight = sparse_weight.view(original_shape)
    
    # Straight-Through Estimator trick: weight + (sparse_weight - weight).detach()
    # This keeps the sparse value for forward pass but uses dense weight for backward pass gradients
    return weight + (sparse_weight - weight).detach()

class SparseGLU(nn.Module):
    """
    Gated Linear Unit with 2:4 structured sparsity applied to the gating projection.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Gate and Value projections
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        
    def forward(self, x):
        # Apply 2:4 sparsity to the gate projection weights during forward pass
        # Note: We apply it to the weights of the linear layer
        sparse_w1 = apply_2_4_mask_ste(self.w1.weight)
        
        # GLU Operation: (x @ W_gate) * (x @ W_value)
        gate = F.linear(x, sparse_w1)
        value = self.w2(x)
        
        return F.gelu(gate) * value
