import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple
from .config import ModelConfig
from .sparse_gate import apply_2_4_mask_ste

class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, device: str = "mps"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Low-rank decomposition: ΔW = A @ B
        # A is (out_features, rank), B is (rank, in_features)
        self.A = nn.Parameter(torch.zeros(out_features, rank, device=device))
        self.B = nn.Parameter(torch.zeros(rank, in_features, device=device))
        
        # Initialize A and B such that ΔW is small/zero initially
        nn.init.normal_(self.A, std=0.02)
        nn.init.zeros_(self.B)

    def forward(self, shared_weight: torch.Tensor) -> torch.Tensor:
        # returns W_shared + A @ B
        return shared_weight + (self.A @ self.B)

class SharedBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        
        # Shared weights for Attention
        self.qkv_shared = nn.Parameter(torch.empty(3 * d_model, d_model, device=config.device))
        self.attn_out_shared = nn.Parameter(torch.empty(d_model, d_model, device=config.device))
        
        # Shared weights for MLP
        self.mlp_fc_shared = nn.Parameter(torch.empty(4 * d_model, d_model, device=config.device))
        self.mlp_proj_shared = nn.Parameter(torch.empty(d_model, 4 * d_model, device=config.device))
        
        # Initialize shared weights
        nn.init.normal_(self.qkv_shared, std=0.02)
        nn.init.normal_(self.attn_out_shared, std=0.02)
        nn.init.normal_(self.mlp_fc_shared, std=0.02)
        nn.init.normal_(self.mlp_proj_shared, std=0.02)

class OffsetLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model = config.d_model
        rank = config.lora_rank
        device = config.device
        
        self.qkv_offset = LowRankLinear(d_model, 3 * d_model, rank, device)
        self.attn_out_offset = LowRankLinear(d_model, d_model, rank, device)
        self.mlp_fc_offset = LowRankLinear(d_model, 4 * d_model, rank, device)
        self.mlp_proj_offset = LowRankLinear(4 * d_model, d_model, rank, device)
        
        # LayerNorms are per-layer
        self.ln_1 = nn.LayerNorm(d_model, device=device)
        self.ln_2 = nn.LayerNorm(d_model, device=device)

class MatryoshkaHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Main projection to vocab
        # We use a single large weight and slice it for MRL
        self.weight = nn.Parameter(torch.empty(config.vocab_size, config.d_model, device=config.device))
        nn.init.normal_(self.weight, std=0.02)
        
    def forward(self, x: torch.Tensor, M=768) -> torch.Tensor:
        # Slice the weights for MRL inference
        weight_slice = self.weight[:, :M]
        return F.linear(x[:, :, :M], weight_slice)

class GhostTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model, device=config.device)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model, device=config.device))
        
        # Single Shared Block (weights)
        self.shared_block = SharedBlock(config)
        
        # Per-layer offsets
        self.layers = nn.ModuleList([OffsetLayer(config) for _ in range(config.n_layers)])
        
        self.ln_f = nn.LayerNorm(config.d_model, device=config.device)
        self.head = MatryoshkaHead(config)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx: torch.Tensor, M=768) -> torch.Tensor:
        b, t = idx.size()
        assert t <= self.config.max_seq_len
        
        x = self.token_emb(idx) + self.pos_emb[:, :t, :]
        
        for layer in self.layers:
            # 1. Attention
            residual = x
            x = layer.ln_1(x)
            
            # Compute effective weights
            qkv_w = layer.qkv_offset(self.shared_block.qkv_shared)
            attn_out_w = layer.attn_out_offset(self.shared_block.attn_out_shared)
            
            # QKV Projection
            qkv = F.linear(x, qkv_w)
            q, k, v = qkv.split(self.config.d_model, dim=-1)
            
            # Multi-head attention (simplified implementation)
            q = q.view(b, t, self.config.n_heads, -1).transpose(1, 2)
            k = k.view(b, t, self.config.n_heads, -1).transpose(1, 2)
            v = v.view(b, t, self.config.n_heads, -1).transpose(1, 2)
            
            # Causal mask
            mask = torch.tril(torch.ones(t, t, device=x.device)).view(1, 1, t, t)
            
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(b, t, -1)
            
            # Attn output projection
            x = residual + F.linear(y, attn_out_w)
            
            # 2. MLP
            residual = x
            x = layer.ln_2(x)
            
            mlp_fc_w = layer.mlp_fc_offset(self.shared_block.mlp_fc_shared)
            mlp_proj_w = layer.mlp_proj_offset(self.shared_block.mlp_proj_shared)
            
            if self.config.use_sparse_glu:
                # GLU implementation: Split 4*d into two 2*d projections
                # w_gate, w_value: (2*d_model, d_model)
                w_gate, w_value = mlp_fc_w.chunk(2, dim=0)
                
                # Apply 2:4 sparsity mask to the gating projection
                w_gate_sparse = apply_2_4_mask_ste(w_gate)
                
                gate = F.linear(x, w_gate_sparse)
                value = F.linear(x, w_value)
                
                # GLU: activation(gate) * value
                # We use GELU as requested
                x = F.gelu(gate) * value
                
                # Projection back: mlp_proj_w is (d_model, 4*d_model)
                # We chunk it to match the 2*d intermediate size
                mlp_proj_w_reduced, _ = mlp_proj_w.chunk(2, dim=1)
                x = residual + F.linear(x, mlp_proj_w_reduced)
            else:
                # Original Phase 1 MLP
                x = F.linear(x, mlp_fc_w)
                x = F.gelu(x)
                x = residual + F.linear(x, mlp_proj_w)
            
        x = self.ln_f(x)
        return self.head(x, M=M)

def mrl_loss(hidden_states: torch.Tensor, targets: torch.Tensor, model: GhostTransformer) -> torch.Tensor:
    config = model.config
    total_loss = 0
    for dim, weight in zip(config.matryoshka_slices, config.matryoshka_weights):
        # Project only the current slice to save memory
        x_slice = hidden_states[:, :, :dim]
        weight_slice = model.head.weight[:, :dim]
        logits = F.linear(x_slice, weight_slice)
        
        # Compute loss for this slice
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += weight * loss
        
        # Explicitly delete logits to free memory immediately
        del logits
        
    return total_loss

def get_spectral_norm(shared_block: SharedBlock, steps: int = 3) -> dict:
    """Computes spectral norm for shared weights using power iteration."""
    metrics = {}
    with torch.no_grad():
        for name, weight in [
            ("qkv_shared", shared_block.qkv_shared),
            ("attn_out_shared", shared_block.attn_out_shared),
            ("mlp_fc_shared", shared_block.mlp_fc_shared),
            ("mlp_proj_shared", shared_block.mlp_proj_shared)
        ]:
            # Approximate spectral norm (max singular value)
            # W is (out, in)
            w = weight
            if w.dim() > 2:
                w = w.view(w.size(0), -1)
            
            u = torch.randn(w.size(0), 1, device=w.device)
            for _ in range(steps):
                v = F.normalize(w.t() @ u, dim=0)
                u = F.normalize(w @ v, dim=0)
            
            sigma = (u.t() @ w @ v).item()
            metrics[name] = sigma
    return metrics
