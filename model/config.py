from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelConfig:
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 50257  # Default GPT-2 vocab size
    max_seq_len: int = 512
    
    # Matryoshka Slice Dimensions
    matryoshka_slices: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    matryoshka_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])
    
    # Low-rank Offset Configuration
    lora_rank: int = 64  # Rank for ΔW tensors
    
    # Phase 2: Sparsity Settings
    use_sparse_glu: bool = False
    sparse_lambda: float = 0.01
    
    # Device
    device: str = "mps"
