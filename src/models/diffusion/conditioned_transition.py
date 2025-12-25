"""
AlphaFold3 Conditioned Transition Block

File: src/models/diffusion/conditioned_transition.py

Implements Algorithm 25: Conditioned Transition Block

A SwiGLU-based feed-forward layer with adaptive normalization and gating
controlled by conditioning signal. Combines AdaLN for input normalization
and adaptive output gating (adaLN-Zero trick).

Key features:
- AdaLN at input (conditioning-dependent normalization)
- SwiGLU for non-linear expansion
- Adaptive output gating (adaLN-Zero)
- Bias initialized to -2.0 for stable training

Mathematical intuition:
    Traditional MLP: Linear → Activation → Linear
    SwiGLU: swish(Linear(x)) ⊙ Linear(x)
    Conditioned: AdaLN → SwiGLU → Adaptive Gate
    
    The conditioning controls:
    1. Input normalization (via AdaLN)
    2. Output gating (via sigmoid gating)
    
Why adaLN-Zero:
- Initialize output gate near 0 (bias=-2.0 → sigmoid≈0.12)
- Early training: small updates (stable)
- Later training: gate opens as needed
- Prevents initial instability from random conditioning
"""

import torch
import torch.nn as nn
from .adaln import AdaptiveLayerNorm


class ConditionedTransitionBlock(nn.Module):
    """
    Conditioned Transition Block - SwiGLU with adaptive normalization.
    
    Implements Algorithm 25 from AF3 supplementary.
    
    This is a feed-forward layer that adapts its behavior based on
    conditioning signal (typically containing timestep + trunk info).
    
    Structure:
    1. AdaLN for conditioning-dependent input normalization
    2. SwiGLU expansion (swish activation)
    3. Adaptive output gating (adaLN-Zero trick)
    
    The adaLN-Zero trick initializes the output gate near zero,
    ensuring the layer starts with small updates during training.
    
    Args:
        dim: Input/output dimension
        dim_cond: Conditioning dimension
        n: Expansion factor for hidden dimension (default: 2)
            Hidden dim = n * dim
    """
    
    def __init__(
        self,
        dim: int,
        dim_cond: int,
        n: int = 2
    ):
        super().__init__()
        
        self.dim = dim
        self.dim_cond = dim_cond
        self.n = n
        self.dim_hidden = n * dim
        
        # Algorithm 25, line 1: AdaLN for input normalization
        self.adaln = AdaptiveLayerNorm(dim_input=dim, dim_cond=dim_cond)
        
        # Algorithm 25, line 2: SwiGLU expansion
        # Two parallel projections (no bias as per paper)
        self.linear_1 = nn.Linear(dim, self.dim_hidden, bias=False)
        self.linear_2 = nn.Linear(dim, self.dim_hidden, bias=False)
        
        # Algorithm 25, line 3: Output projection (no bias)
        self.linear_out = nn.Linear(self.dim_hidden, dim, bias=False)
        
        # Algorithm 25, line 3: Adaptive gating (adaLN-Zero)
        # Initialized with bias=-2.0 for stability
        self.gate = nn.Linear(dim_cond, dim)
        # Initialize bias to -2.0 (sigmoid(-2) ≈ 0.12)
        nn.init.constant_(self.gate.bias, -2.0)
    
    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply conditioned transition.
        
        Args:
            a: Input activations
               Shape: [..., dim]
            s: Conditioning signal
               Shape: [..., dim_cond]
               Contains timestep embedding + trunk representations
        
        Returns:
            update: Output update to add to activations
                   Shape: [..., dim]
        
        Algorithm 25:
        1: a ← AdaLN(a, s)
        2: b ← swish(LinearNoBias(a)) ⊙ LinearNoBias(a)
        3: a ← sigmoid(Linear(s, bias_init=-2.0)) ⊙ LinearNoBias(b)
        4: return a
        """
        # Algorithm 25, line 1: Adaptive normalization
        a = self.adaln(a, s)  # [..., dim]
        
        # Algorithm 25, line 2: SwiGLU expansion
        # swish(x) = x * sigmoid(x)
        swish_out = torch.nn.functional.silu(self.linear_1(a))  # [..., n*dim]
        gate_out = self.linear_2(a)  # [..., n*dim]
        b = swish_out * gate_out  # [..., n*dim]
        
        # Project back to original dimension
        b = self.linear_out(b)  # [..., dim]
        
        # Algorithm 25, line 3: Adaptive output gating (adaLN-Zero)
        # sigmoid(Linear(s, bias_init=-2.0))
        gate_value = torch.sigmoid(self.gate(s))  # [..., dim]
        
        # Apply gating
        a = gate_value * b  # [..., dim]
        
        # Algorithm 25, line 4: Return update
        return a


def create_dummy_conditioned_transition_input(
    batch_size: int = 0, # Change to None to clear unit tests
    dim: int = 768,
    dim_cond: int = 384,
    device: str = 'cpu'
) -> tuple:
    """
    Create dummy inputs for testing ConditionedTransitionBlock.
    
    Args:
        batch_size: If None, returns unbatched. Otherwise, batched.
        dim: Dimension of activations
        dim_cond: Dimension of conditioning
        device: Device for tensors
    
    Returns:
        (a, s): Tuple of activation and conditioning tensors
    """
    if batch_size is None:
        # Unbatched: single sequence
        a = torch.randn(10, dim, device=device)  # 10 tokens
        s = torch.randn(10, dim_cond, device=device)
    else:
        # Batched
        a = torch.randn(batch_size, 10, dim, device=device)
        s = torch.randn(batch_size, 10, dim_cond, device=device)
    
    return a, s