"""
AlphaFold3 Adaptive LayerNorm (AdaLN)

File: src/models/diffusion/adaln.py

Implements Algorithm 26: Adaptive LayerNorm

Modulates activations using a conditioning signal. Unlike standard LayerNorm
with fixed scale and offset parameters, AdaLN learns to predict both scale
and shift from conditioning (typically timestep + trunk information).

Key features:
- Conditioning-dependent normalization
- Learned scale via sigmoid (bounded 0-1)
- Learned shift via linear projection
- More expressive than standard LayerNorm

Mathematical intuition:
    Standard LN: (x - mean) / std
    AdaLN: scale(conditioning) ⊙ LN(x) + shift(conditioning)
    
    The conditioning signal dynamically controls how much to scale
    and shift the normalized activations.

Why adaptive:
- Diffusion needs different behavior at different timesteps
- Trunk representations provide context
- AdaLN allows conditioning to modulate information flow
"""

import torch
import torch.nn as nn


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive LayerNorm - modulates activations with conditioning.
    
    Implements Algorithm 26 from AF3 supplementary.
    
    Unlike standard LayerNorm which has learned but fixed scale and offset,
    AdaLN predicts scale and shift from a conditioning signal. This allows
    the network to adapt its behavior based on context (e.g., diffusion
    timestep, trunk representations).
    
    The scale is bounded via sigmoid (0 to 1), while shift is unbounded.
    
    Args:
        dim_input: Dimension of input activations to normalize
        dim_cond: Dimension of conditioning signal
    """
    
    def __init__(self, dim_input: int, dim_cond: int):
        super().__init__()
        
        self.dim_input = dim_input
        self.dim_cond = dim_cond
        
        # LayerNorm for input (no learnable scale/offset)
        # Algorithm 26, line 1: scale=False, offset=False
        self.norm_input = nn.LayerNorm(dim_input, elementwise_affine=False)
        
        # LayerNorm for conditioning (no offset)
        # Algorithm 26, line 2: offset=False
        # In PyTorch, we need to use affine=True but will only use scale
        self.norm_cond = nn.LayerNorm(dim_cond, elementwise_affine=True)
        # Set bias to zero and make it non-trainable
        self.norm_cond.bias.data.zero_()
        self.norm_cond.bias.requires_grad = False
        
        # Algorithm 26, line 3: Linear projections from conditioning
        # Scale projection (with sigmoid)
        self.linear_scale = nn.Linear(dim_cond, dim_input)
        
        # Shift projection (no bias, as stated in algorithm)
        self.linear_shift = nn.Linear(dim_cond, dim_input, bias=False)
    
    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply adaptive layer normalization.
        
        Args:
            a: Input activations to normalize
               Shape: [..., dim_input]
            s: Conditioning signal
               Shape: [..., dim_cond]
               Typically contains timestep embedding + trunk representations
        
        Returns:
            modulated: Normalized and modulated activations
                      Shape: [..., dim_input]
        
        Algorithm 26:
        1: a ← LayerNorm(a, scale=False, offset=False)
        2: s ← LayerNorm(s, offset=False)
        3: a ← sigmoid(Linear(s)) ⊙ a + LinearNoBias(s)
        4: return a
        """
        # Algorithm 26, line 1: Normalize input (no learnable params)
        a_norm = self.norm_input(a)  # [..., dim_input]
        
        # Algorithm 26, line 2: Normalize conditioning (no offset)
        s_norm = self.norm_cond(s)  # [..., dim_cond]
        
        # Algorithm 26, line 3: Compute adaptive scale and shift
        # Scale: sigmoid(Linear(s)) - bounded to [0, 1]
        scale = torch.sigmoid(self.linear_scale(s_norm))  # [..., dim_input]
        
        # Shift: LinearNoBias(s) - unbounded
        shift = self.linear_shift(s_norm)  # [..., dim_input]
        
        # Apply modulation: scale ⊙ a + shift
        modulated = scale * a_norm + shift  # [..., dim_input]
        
        return modulated


def create_dummy_adaln_input(
    batch_size: int = None,
    dim_input: int = 384,
    dim_cond: int = 384,
    device: str = 'cpu'
) -> tuple:
    """
    Create dummy inputs for testing AdaLN.
    
    Args:
        batch_size: If None, returns unbatched. Otherwise, batched.
        dim_input: Dimension of activations
        dim_cond: Dimension of conditioning
        device: Device for tensors
    
    Returns:
        (a, s): Tuple of activation and conditioning tensors
    """
    if batch_size is None:
        # Unbatched: single sequence
        a = torch.randn(10, dim_input, device=device)  # 10 tokens
        s = torch.randn(10, dim_cond, device=device)
    else:
        # Batched
        a = torch.randn(batch_size, 10, dim_input, device=device)
        s = torch.randn(batch_size, 10, dim_cond, device=device)
    
    return a, s