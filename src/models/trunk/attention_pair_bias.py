"""
AlphaFold3 Attention with Pair Bias

File: src/models/trunk/attention_pair_bias.py

Implements attention for single representation with pair bias.

This is the single representation attention used in Pairformer (Algorithm 17, line 7).
Similar to standard multi-head attention but adds pair representation as bias to
attention logits.

Key features:
- Standard Q·K attention (unlike MSAPairWeightedAveraging!)
- Pair bias added to attention logits
- Gating mechanism for selective updates
- Multi-head for different perspectives

Mathematical intuition:
    weights = softmax(Q @ K^T / √c + pair_bias)
    
    The pair representation guides which tokens should attend to each other,
    based on the current structural hypothesis.
"""

import torch
import torch.nn as nn
import math


class AttentionPairBias(nn.Module):
    """
    Multi-head attention for single representation with pair bias.
    
    Used in Pairformer Stack (Algorithm 17, line 7).
    
    This is standard scaled dot-product attention with pair bias added to
    the attention logits. The pair representation acts as a learned bias
    that modulates which positions attend to each other based on structural
    information.
    
    Key difference from MSAPairWeightedAveraging:
    - This uses Q·K attention (standard)
    - MSAPairWeightedAveraging uses only pair bias (no Q·K)
    
    Args:
        c_single: Single representation channels (default: 384)
        c_pair: Pair representation channels (default: 128)
        c: Attention dimension per head (default: 32)  
        n_heads: Number of attention heads (default: 16)
    """
    
    def __init__(
        self,
        c_single: int = 384,
        c_pair: int = 128,
        c: int = 32,
        n_heads: int = 16
    ):
        super().__init__()
        
        self.c_single = c_single
        self.c_pair = c_pair
        self.c = c
        self.n_heads = n_heads
        
        # Layer normalizations
        self.norm_single = nn.LayerNorm(c_single)
        self.norm_pair = nn.LayerNorm(c_pair)
        
        # QKV projections (multi-head)
        self.linear_q = nn.Linear(c_single, c * n_heads, bias=False)
        self.linear_k = nn.Linear(c_single, c * n_heads, bias=False)
        self.linear_v = nn.Linear(c_single, c * n_heads, bias=False)
        
        # Pair bias projection (multi-head)
        self.linear_b = nn.Linear(c_pair, n_heads, bias=False)
        
        # Gating projection (multi-head)
        self.linear_g = nn.Linear(c_single, c * n_heads, bias=False)
        
        # Output projection
        self.linear_out = nn.Linear(c * n_heads, c_single, bias=False)
    
    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply single representation attention with pair bias.
        
        Args:
            single: [N_token, c_single] single representation
                    Or [batch, N_token, c_single] if batched
            pair: [N_token, N_token, c_pair] pair representation
                  Or [batch, N_token, N_token, c_pair] if batched
            mask: Optional [N_token, N_token] attention mask
                  Or [batch, N_token, N_token] if batched
        
        Returns:
            single_update: Same shape as single input
        
        Similar to row-wise attention in AlphaFold 2 but for single sequence.
        Pair bias added to attention logits:
        attention_weights = softmax(Q @ K^T / √c + pair_bias)
        """
        # Handle batching
        is_batched = single.dim() == 3
        if not is_batched:
            single = single.unsqueeze(0)
            pair = pair.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        batch_size, n_token, _ = single.shape
        
        # Layer normalization
        single_norm = self.norm_single(single)  # [batch, N_token, c_single]
        pair_norm = self.norm_pair(pair)  # [batch, N_token, N_token, c_pair]
        
        # Project to Q, K, V (multi-head)
        q = self.linear_q(single_norm)  # [batch, N_token, c*n_heads]
        k = self.linear_k(single_norm)  # [batch, N_token, c*n_heads]
        v = self.linear_v(single_norm)  # [batch, N_token, c*n_heads]
        
        # Reshape to separate heads
        q = q.view(batch_size, n_token, self.n_heads, self.c)
        k = k.view(batch_size, n_token, self.n_heads, self.c)
        v = v.view(batch_size, n_token, self.n_heads, self.c)
        # Shape: [batch, N_token, n_heads, c]
        
        # Project pair to bias (multi-head)
        b = self.linear_b(pair_norm)  # [batch, N_token, N_token, n_heads]
        
        # Compute gate (multi-head)
        g = torch.sigmoid(self.linear_g(single_norm))  # [batch, N_token, c*n_heads]
        g = g.view(batch_size, n_token, self.n_heads, self.c)
        # Shape: [batch, N_token, n_heads, c]
        
        # Compute attention logits: Q @ K^T / √c
        # q: [batch, i, heads, c]
        # k: [batch, j, heads, c]
        # Result: [batch, i, j, heads]
        attn_logits = torch.einsum('bihc,bjhc->bijh', q, k)
        attn_logits = attn_logits / math.sqrt(self.c)  # Scale
        
        # Add pair bias
        # b: [batch, i, j, heads]
        attn_logits = attn_logits + b  # [batch, i, j, heads]
        
        # Apply mask if provided
        if mask is not None:
            # mask: [batch, i, j] → [batch, i, j, 1]
            mask = mask.unsqueeze(-1)
            # Set masked positions to large negative value
            attn_logits = attn_logits.masked_fill(~mask, -1e9)
        
        # Softmax over j dimension (keys)
        attn_weights = torch.softmax(attn_logits, dim=2)  # [batch, i, j, heads]
        
        # Apply attention to values
        # v: [batch, j, heads, c]
        # attn_weights: [batch, i, j, heads]
        # Result: [batch, i, heads, c]
        o = torch.einsum('bijh,bjhc->bihc', attn_weights, v)
        
        # Apply gating
        o = g * o  # [batch, N_token, n_heads, c]
        
        # Concatenate heads and project to output
        o = o.reshape(batch_size, n_token, self.n_heads * self.c)
        single_update = self.linear_out(o)  # [batch, N_token, c_single]
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            single_update = single_update.squeeze(0)
        
        return single_update