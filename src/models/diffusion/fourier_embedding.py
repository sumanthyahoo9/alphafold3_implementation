"""
AlphaFold3 Fourier Embedding

File: src/models/diffusion/fourier_embedding.py

Implements Algorithm 22: Fourier Embedding

Encodes diffusion timesteps using random Fourier features. This provides
a rich, high-dimensional representation of time that helps the network
learn different denoising behaviors at different noise levels.

Key features:
- Random Fourier features for time encoding
- Fixed random weights (not learned!)
- Cosine-based encoding for smoothness
- High-dimensional (default 256) for rich representation

Mathematical intuition:
    Each dimension captures a different frequency of time
    cos(2π(t·w + b)) creates periodic patterns
    Multiple frequencies together create unique encoding for each t
    
Why random Fourier features:
- Proven to approximate kernel functions
- Rich encoding from simple operation
- Computationally efficient
- Helps with positional encoding of continuous values
"""
import math
import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    """
    Fourier embedding for timestep encoding in diffusion.
    
    Implements Algorithm 22 from AF3 supplementary.
    
    Uses random Fourier features to encode continuous timestep values
    into high-dimensional representations. The weights and biases are
    randomly initialized once and then FROZEN (never updated).
    
    This is similar to positional encoding in transformers but for
    continuous scalar values (timesteps) rather than discrete positions.
    
    Args:
        dim: Embedding dimension (default: 256)
            Paper uses c=256 in Algorithm 21, line 8
    """
    
    def __init__(self, dim: int = 256):
        super().__init__()
        
        self.dim = dim
        
        # Algorithm 22, line 1: Randomly generate weight/bias ONCE before training
        # These are FIXED (registered as buffer, not parameters)
        # w, b ~ N(0, I_c)
        w = torch.randn(dim)
        b = torch.randn(dim)
        
        # Register as buffers (not parameters - won't be updated by optimizer)
        self.register_buffer('w', w)
        self.register_buffer('b', b)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute Fourier embedding of timestep(s).
        
        Args:
            t: Timestep value(s)
               Can be:
               - Scalar: [] or [1]
               - Batch: [batch_size]
               All values should be preprocessed (e.g., log-scaled)
        
        Returns:
            embedding: [dim] if t is scalar
                      [batch_size, dim] if t is batch
        
        Algorithm 22:
        1: w, b ~ N(0, I_c)  # Done in __init__
        2: return cos(2π(t·w + b))
        """
        # Handle different input shapes
        if t.dim() == 0:
            # Scalar input
            t = t.unsqueeze(0)  # [1]
            squeeze_output = True
        elif t.dim() == 1:
            # Batch input [batch_size]
            squeeze_output = False
        else:
            raise ValueError(f"Expected 0D or 1D tensor, got {t.dim()}D")
        
        # Algorithm 22, line 2: cos(2π(t·w + b))
        # t: [batch_size] or [1]
        # w: [dim]
        # Result: [batch_size, dim]
        
        # Expand t for broadcasting: [batch_size, 1]
        t = t.unsqueeze(-1)
        
        # Compute: t·w + b
        # [batch_size, 1] * [dim] + [dim] = [batch_size, dim]
        angle = 2 * math.pi * (t * self.w + self.b)
        
        # Apply cosine
        embedding = torch.cos(angle)  # [batch_size, dim]
        
        # Squeeze if input was scalar
        if squeeze_output:
            embedding = embedding.squeeze(0)  # [dim]
        
        return embedding


def create_dummy_timestep(batch_size: int = None, device: str = 'cpu') -> torch.Tensor:
    """
    Create dummy timestep for testing FourierEmbedding.
    
    Args:
        batch_size: If None, returns scalar timestep. Otherwise, batch.
        device: Device for tensor
    
    Returns:
        Timestep tensor
    """
    if batch_size is None:
        # Scalar timestep (typical usage in Algorithm 21)
        t = torch.tensor(0.5, device=device)
    else:
        # Batch of timesteps
        t = torch.rand(batch_size, device=device)
    
    return t