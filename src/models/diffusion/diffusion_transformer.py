"""
AlphaFold3 Diffusion Transformer

File: src/models/diffusion/diffusion_transformer.py

Implements Algorithm 23: Diffusion Transformer
Implements Algorithm 24: Diffusion AttentionPairBias

The core transformer for diffusion-based structure generation.
24 blocks of attention + conditioned transitions operating on token-level
representations.

Key features:
- Multi-head attention with pair bias
- Adaptive LayerNorm conditioning
- Conditioned transition blocks
- Adaptive output gating (adaLN-Zero)

Architecture:
    24 blocks × (AttentionPairBias + ConditionedTransitionBlock)
    
This is the token-level processor between AtomAttentionEncoder and
AtomAttentionDecoder in the diffusion module.
"""
from typing import Optional
import torch
import torch.nn as nn

from src.models.diffusion.adaln import AdaptiveLayerNorm
from src.models.diffusion.conditioned_transition import ConditionedTransitionBlock


class DiffusionAttentionPairBias(nn.Module):
    """
    Attention with pair bias for diffusion module.
    
    Implements Algorithm 24 from AF3 supplementary.
    
    Similar to trunk's AttentionPairBias but with:
    - Adaptive LayerNorm (instead of standard LayerNorm)
    - Adaptive output gating (adaLN-Zero)
    
    Args:
        c_token: Token dimension (default: 768)
        c_pair: Pair dimension (default: 128)  
        n_heads: Number of attention heads (default: 16)
        c_s: Conditioning dimension (default: 384), can be None
    """
    
    def __init__(
        self,
        c_token: int = 768,
        c_pair: int = 128,
        n_heads: int = 16,
        c_s: int = 384
    ):
        super().__init__()
        
        self.c_token = c_token
        self.c_pair = c_pair
        self.n_heads = n_heads
        self.c_s = c_s
        self.c_hidden = c_token // n_heads
        
        assert c_token % n_heads == 0, f"c_token={c_token} must be divisible by n_heads={n_heads}"
        
        # Algorithm 24, lines 1-5: Input normalization
        # If conditioning provided, use AdaLN, else standard LayerNorm
        if c_s is not None:
            self.adaln = AdaptiveLayerNorm(dim_input=c_token, dim_cond=c_s)
            self.use_conditioning = True
        else:
            self.layer_norm = nn.LayerNorm(c_token)
            self.use_conditioning = False
        
        # Algorithm 24, line 6: Query projection (with bias)
        self.linear_q = nn.Linear(c_token, c_token)
        
        # Algorithm 24, line 7: Key and Value projections (no bias)
        self.linear_k = nn.Linear(c_token, c_token, bias=False)
        self.linear_v = nn.Linear(c_token, c_token, bias=False)
        
        # Algorithm 24, line 8: Pair bias projection
        self.linear_pair_bias = nn.Linear(c_pair, n_heads, bias=False)
        self.pair_norm = nn.LayerNorm(c_pair)
        
        # Algorithm 24, line 9: Gating
        self.linear_g = nn.Linear(c_token, c_token, bias=False)
        
        # Algorithm 24, line 11: Output projection (no bias)
        self.linear_out = nn.Linear(c_token, c_token, bias=False)
        
        # Algorithm 24, lines 12-14: Adaptive output gating (adaLN-Zero)
        if c_s is not None:
            self.output_gate = nn.Linear(c_s, c_token)
            # Initialize bias to -2.0 for stability
            nn.init.constant_(self.output_gate.bias, -2.0)
    
    def forward(
        self,
        a: torch.Tensor,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        bias_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply attention with pair bias.
        
        Args:
            a: Token activations
               Shape: [N_token, c_token] or [batch, N_token, c_token]
            s: Conditioning (can be None)
               Shape: [N_token, c_s] or [batch, N_token, c_s]
            z: Pair representation
               Shape: [N_token, N_token, c_pair] or [batch, N_token, N_token, c_pair]
            bias_mask: Optional attention bias mask (β_ij in algorithm)
                      Shape: [N_token, N_token] or [batch, N_token, N_token]
        
        Returns:
            output: Updated token activations
                   Shape: same as input a
        
        Algorithm 24:
        1-5: Adaptive normalization (AdaLN if conditioning, else LayerNorm)
        6-7: Q, K, V projections
        8: Pair bias projection
        9: Gating values
        10-11: Attention + output projection
        12-14: Adaptive output gating if conditioning
        """
        # Handle batching - detect from ORIGINAL input shape before any transforms
        is_batched = a.dim() == 3
        
        # Algorithm 24, lines 1-5: Input normalization
        if self.use_conditioning and s is not None:
            # Line 2: AdaLN
            a_norm = self.adaln(a, s)
        else:
            # Line 4: Standard LayerNorm
            a_norm = self.layer_norm(a)
        
        # Algorithm 24, line 6: Query projection
        q = self.linear_q(a_norm)  # [..., N_token, c_token]
        
        # Algorithm 24, line 7: Key and Value projections
        k = self.linear_k(a_norm)  # [..., N_token, c_token]
        v = self.linear_v(a_norm)  # [..., N_token, c_token]
        
        # Reshape for multi-head attention
        # [..., N_token, c_token] -> [..., N_token, n_heads, c_hidden]
        q = q.view(*q.shape[:-1], self.n_heads, self.c_hidden)
        k = k.view(*k.shape[:-1], self.n_heads, self.c_hidden)
        v = v.view(*v.shape[:-1], self.n_heads, self.c_hidden)
        
        # Transpose for attention: [..., n_heads, N_token, c_hidden]
        if is_batched:
            q = q.transpose(-3, -2)  # [batch, n_heads, N_token, c_hidden]
            k = k.transpose(-3, -2)
            v = v.transpose(-3, -2)
        else:
            q = q.transpose(-3, -2)  # [n_heads, N_token, c_hidden]
            k = k.transpose(-3, -2)
            v = v.transpose(-3, -2)
        
        # Algorithm 24, line 8: Compute pair bias
        # [..., N_token, N_token, c_pair] -> [..., N_token, N_token, n_heads]
        pair_bias = self.linear_pair_bias(self.pair_norm(z))
        
        # Transpose to [..., n_heads, N_token, N_token]
        if is_batched:
            pair_bias = pair_bias.permute(0, 3, 1, 2)  # [batch, n_heads, N_token, N_token]
        else:
            pair_bias = pair_bias.permute(2, 0, 1)  # [n_heads, N_token, N_token]
        
        # Add optional bias mask (β_ij)
        if bias_mask is not None:
            if is_batched:
                if bias_mask.dim() == 2:  # [N_token, N_token]
                    bias_mask = bias_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N_token, N_token]
                elif bias_mask.dim() == 3:  # [batch, N_token, N_token]
                    bias_mask = bias_mask.unsqueeze(1)  # [batch, 1, N_token, N_token]
            else:
                # Unbatched: bias_mask is [N_token, N_token]
                bias_mask = bias_mask.unsqueeze(0)  # [1, N_token, N_token]
            pair_bias = pair_bias + bias_mask
        
        # Algorithm 24, line 10: Compute attention
        # Q @ K^T / sqrt(c_hidden) + pair_bias
        # [..., n_heads, N_token, c_hidden] @ [..., n_heads, c_hidden, N_token]
        # -> [..., n_heads, N_token, N_token]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.c_hidden ** 0.5)
        attn_logits = attn_logits + pair_bias
        
        # Softmax over keys (last dimension)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [..., n_heads, N_token, N_token]
        
        # Algorithm 24, line 9: Compute gating values
        g = torch.sigmoid(self.linear_g(a_norm))  # [..., N_token, c_token]
        g = g.view(*g.shape[:-1], self.n_heads, self.c_hidden)  # [..., N_token, n_heads, c_hidden]
        if is_batched:
            g = g.transpose(-3, -2)  # [batch, n_heads, N_token, c_hidden]
        else:
            g = g.transpose(-3, -2)  # [n_heads, N_token, c_hidden]
        
        # Apply attention to values
        # [..., n_heads, N_token, N_token] @ [..., n_heads, N_token, c_hidden]
        # -> [..., n_heads, N_token, c_hidden]
        attn_out = torch.matmul(attn_weights, v)
        
        # Apply gating (element-wise)
        attn_out = g * attn_out  # [..., n_heads, N_token, c_hidden]
        
        # Transpose back and reshape
        if is_batched:
            attn_out = attn_out.transpose(-3, -2)  # [batch, N_token, n_heads, c_hidden]
        else:
            attn_out = attn_out.transpose(-3, -2)  # [N_token, n_heads, c_hidden]
        
        attn_out = attn_out.reshape(*attn_out.shape[:-2], self.c_token)  # [..., N_token, c_token]
        
        # Algorithm 24, line 11: Output projection
        output = self.linear_out(attn_out)  # [..., N_token, c_token]
        
        # Algorithm 24, lines 12-14: Adaptive output gating (adaLN-Zero)
        if self.use_conditioning and s is not None:
            # Line 13: sigmoid(Linear(s, bias_init=-2.0)) ⊙ output
            gate = torch.sigmoid(self.output_gate(s))
            output = gate * output
        
        return output


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer - 24 blocks of attention + transitions.
    
    Implements Algorithm 23 from AF3 supplementary.
    
    The main token-level processor in the diffusion module.
    Each block:
    1. AttentionPairBias (with conditioning)
    2. ConditionedTransitionBlock
    
    Args:
        c_token: Token dimension (default: 768)
        c_pair: Pair dimension (default: 128)
        c_s: Conditioning dimension (default: 384)
        n_blocks: Number of transformer blocks (default: 24)
        n_heads: Number of attention heads (default: 16)
    """
    
    def __init__(
        self,
        c_token: int = 768,
        c_pair: int = 128,
        c_s: int = 384,
        n_blocks: int = 24,
        n_heads: int = 16
    ):
        super().__init__()
        
        self.c_token = c_token
        self.c_pair = c_pair
        self.c_s = c_s
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        
        # Create 24 transformer blocks
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                c_token=c_token,
                c_pair=c_pair,
                c_s=c_s,
                n_heads=n_heads
            )
            for _ in range(n_blocks)
        ])
    
    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        bias_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply diffusion transformer.
        
        Args:
            a: Token activations
               Shape: [N_token, c_token] or [batch, N_token, c_token]
            s: Single conditioning
               Shape: [N_token, c_s] or [batch, N_token, c_s]
            z: Pair conditioning
               Shape: [N_token, N_token, c_pair] or [batch, N_token, N_token, c_pair]
            bias_mask: Optional attention bias (β_ij), typically 0
                      Shape: [N_token, N_token] or [batch, N_token, N_token]
        
        Returns:
            a: Updated token activations
               Shape: same as input a
        
        Algorithm 23:
        1: for n in [1...24]:
        2:     b = AttentionPairBias(a, s, z, β, heads=16)
        3:     a = b + ConditionedTransitionBlock(a, s)
        5: return a
        """
        # Algorithm 23, lines 1-4: Apply 24 blocks
        for block in self.blocks:
            a = block(a, s, z, bias_mask)
        
        # Algorithm 23, line 5: Return
        return a


class DiffusionTransformerBlock(nn.Module):
    """
    Single block of DiffusionTransformer.
    
    One block = AttentionPairBias + ConditionedTransitionBlock
    """
    
    def __init__(
        self,
        c_token: int = 768,
        c_pair: int = 128,
        c_s: int = 384,
        n_heads: int = 16
    ):
        super().__init__()
        
        # Algorithm 23, line 2: AttentionPairBias
        self.attention = DiffusionAttentionPairBias(
            c_token=c_token,
            c_pair=c_pair,
            n_heads=n_heads,
            c_s=c_s
        )
        
        # Algorithm 23, line 3: ConditionedTransitionBlock
        self.transition = ConditionedTransitionBlock(
            dim=c_token,
            dim_cond=c_s,
            n=2  # Expansion factor
        )
    
    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        bias_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply one transformer block.
        
        Algorithm 23, lines 2-3:
        2: b = AttentionPairBias(a, s, z, β, heads)
        3: a = b + ConditionedTransitionBlock(a, s)
        """
        # Line 2: Attention with pair bias
        b = self.attention(a, s, z, bias_mask)
        
        # Line 3: Add residual + transition
        a = b + self.transition(a, s)
        
        return a


def create_dummy_diffusion_transformer_input(
    batch_size: int = None,
    n_token: int = 10,
    c_token: int = 768,
    c_pair: int = 128,
    c_s: int = 384,
    device: str = 'cpu'
) -> tuple:
    """
    Create dummy inputs for testing DiffusionTransformer.
    
    Args:
        batch_size: If None, unbatched. Otherwise, batched.
        n_token: Number of tokens
        c_token: Token dimension
        c_pair: Pair dimension
        c_s: Conditioning dimension
        device: Device for tensors
    
    Returns:
        (a, s, z, bias_mask): Tuple of inputs
    """
    if batch_size is None:
        a = torch.randn(n_token, c_token, device=device)
        s = torch.randn(n_token, c_s, device=device)
        z = torch.randn(n_token, n_token, c_pair, device=device)
        bias_mask = None  # Typically 0 in Algorithm 20, line 5
    else:
        a = torch.randn(batch_size, n_token, c_token, device=device)
        s = torch.randn(batch_size, n_token, c_s, device=device)
        z = torch.randn(batch_size, n_token, n_token, c_pair, device=device)
        bias_mask = None
    
    return a, s, z, bias_mask