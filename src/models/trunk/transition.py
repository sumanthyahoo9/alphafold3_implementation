"""
AlphaFold3 Transition Layer

Implements Algorithm 11: Transition layer

This is a feed-forward MLP with SwiGLU activation used throughout
the architecture (MSA Module, Pairformer, etc).

Key features:
- SwiGLU activation: swish(a) ⊙ b (gated activation)
- Expansion factor n (typically 4)
- No bias in linear layers
- Used with residual connections in parent modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transition(nn.Module):
    """
    Transition layer with SwiGLU activation.
    
    Implements Algorithm 11 from AF3 supplementary.
    
    This is a standard feed-forward layer with gated activation.
    The SwiGLU activation (swish(a) ⊙ b) provides better gradient
    flow than ReLU and allows the network to gate information.
    
    Args:
        c: Input/output channels
        n: Expansion factor (default: 4)
            Hidden dimension = n * c
    
    References:
        SwiGLU: "GLU Variants Improve Transformer" (Shazeer 2020)
        Used in PaLM, LLaMA, and modern transformers
    """
    
    def __init__(self, c: int, n: int = 4):
        super().__init__()
        
        self.c = c
        self.n = n
        self.hidden_dim = n * c
        
        # Line 2: Project to n*c for first gate
        self.linear_a = nn.Linear(c, n * c, bias=False)
        
        # Line 3: Project to n*c for second gate
        self.linear_b = nn.Linear(c, n * c, bias=False)
        
        # Line 4: Project back to c
        self.linear_out = nn.Linear(n * c, c, bias=False)
        
        # Layer normalization
        self.norm = nn.LayerNorm(c)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transition layer with SwiGLU activation.
        
        Args:
            x: [..., c] input tensor
        
        Returns:
            x_out: [..., c] output tensor (same shape as input)
        
        Algorithm 11:
        1: x ← LayerNorm(x)
        2: a = LinearNoBias(x)    # a ∈ R^(n·c)
        3: b = LinearNoBias(x)    # b ∈ R^(n·c)
        4: x ← LinearNoBias(swish(a) ⊙ b)  # x ∈ R^c
        5: return x
        """
        # Line 1: Layer normalization
        x_norm = self.norm(x)  # [..., c]
        
        # Lines 2-3: Project to hidden dimension (n*c)
        a = self.linear_a(x_norm)  # [..., n*c]
        b = self.linear_b(x_norm)  # [..., n*c]
        
        # Line 4: SwiGLU activation
        # swish(a) = a * sigmoid(a)
        # SwiGLU = swish(a) ⊙ b
        a_swish = F.silu(a)  # silu = swish = x * sigmoid(x)
        gated = a_swish * b  # [..., n*c]
        
        # Project back to original dimension
        x_out = self.linear_out(gated)  # [..., c]
        
        return x_out


def create_dummy_transition_input(
    batch_size: int = 2,
    seq_len: int = 10,
    c: int = 128
) -> torch.Tensor:
    """
    Create dummy input for testing Transition.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        c: Channel dimension
        
    Returns:
        x: [batch_size, seq_len, c] dummy tensor
    """
    return torch.randn(batch_size, seq_len, c)