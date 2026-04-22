import torch
import torch.nn as nn
import torch.nn.functional as F

class MatryoshkaRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dims = [96, 192, 384, 768]
        self.threshold = 0.9

    def forward(self, hidden_states, heads):
        for i, dim in enumerate(self.dims):
            logits = heads[i](hidden_states[:, :, :dim])
            probs = F.softmax(logits, dim=-1)
            max_prob, _ = torch.max(probs, dim=-1)
            
            if max_prob.mean() > self.threshold or i == len(self.dims) - 1:
                return logits, i
        return logits, len(self.dims) - 1

class GhostAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.embed_dim
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).reshape(B, T, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        att = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        att = F.softmax(att, dim=-1)
        y = att @ v
        
        return self.proj(y.reshape(B, T, C))
