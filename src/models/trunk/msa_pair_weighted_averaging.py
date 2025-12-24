"""
AlphaFold3 MSA Pair Weighted Averaging

File: src/models/trunk/msa_pair_weighted_averaging.py

Implements Algorithm 10: MSA row-wise gated self-attention using only pair bias

This layer performs attention over MSA rows where the attention weights
are derived entirely from the pair representation, not from query-key
interactions. This is computationally efficient as all MSA rows use the
same attention pattern.

Key features:
- No query-key attention mechanism
- Attention weights projected from pair representation only
- Gating mechanism for selective updates
- Multi-head attention for different perspectives
- All MSA rows attend identically (more efficient!)

Mathematical intuition:
    Traditional attention: weights = softmax(Q @ K^T)
    MSA pair weighted:     weights = softmax(pair_bias)
    
    The pair representation controls how positions attend to each other,
    making this independent of the specific MSA row content.
"""

import torch
import torch.nn as nn


class MSAPairWeightedAveraging(nn.Module):
    """
    MSA row-wise gated self-attention using only pair bias.
    
    Implements Algorithm 10 from AF3 supplementary.
    
    This is a unique attention mechanism where attention weights are derived
    entirely from the pair representation rather than query-key products.
    This means all MSA rows use the same attention pattern, which is:
    1. More efficient (compute weights once, apply to all rows)
    2. Enforces consistency (all sequences attend the same way)
    
    The pair representation acts as a learned attention pattern that
    determines which token positions should attend to each other based
    on the current structural hypothesis.
    
    Args:
        c_msa: MSA representation channels (default: 64)
        c_pair: Pair representation channels (default: 128)
        c: Attention dimension per head (default: 32)
        n_heads: Number of attention heads (default: 8)
    """
    
    def __init__(
        self,
        c_msa: int = 64,
        c_pair: int = 128,
        c: int = 32,
        n_heads: int = 8
    ):
        super().__init__()
        
        self.c_msa = c_msa
        self.c_pair = c_pair
        self.c = c
        self.n_heads = n_heads
        
        # Layer normalizations
        self.norm_msa = nn.LayerNorm(c_msa)
        self.norm_pair = nn.LayerNorm(c_pair)
        
        # Line 2: Value projection (multi-head)
        self.linear_v = nn.Linear(c_msa, c * n_heads, bias=False)
        
        # Line 3: Pair bias projection (multi-head)
        self.linear_b = nn.Linear(c_pair, n_heads, bias=False)
        
        # Line 4: Gating projection (multi-head)
        self.linear_g = nn.Linear(c_msa, c * n_heads, bias=False)
        
        # Line 7: Output projection
        self.linear_out = nn.Linear(c * n_heads, c_msa, bias=False)
    
    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply MSA pair weighted averaging.
        
        Args:
            msa: [N_msa, N_token, c_msa] MSA representation
                 Or [batch, N_msa, N_token, c_msa] if batched
            pair: [N_token, N_token, c_pair] pair representation
                  Or [batch, N_token, N_token, c_pair] if batched
        
        Returns:
            msa_update: Same shape as msa input
        
        Algorithm 10:
        1: m_si ← LayerNorm(m_si)
        2: v^h_si = LinearNoBias(m_si)
        3: b^h_ij = LinearNoBias(LayerNorm(z_ij))
        4: g^h_si = sigmoid(LinearNoBias(m_si))
        5: w^h_ij = softmax_j(b^h_ij)
        6: o^h_si = g^h_si ⊙ Σ_j w^h_ij v^h_sj
        7: m̃_si = LinearNoBias(concat_h(o^h_si))
        8: return {m̃_si}
        """
        # Handle batching
        is_batched = msa.dim() == 4
        if not is_batched:
            msa = msa.unsqueeze(0)
            pair = pair.unsqueeze(0)
        
        batch_size, n_msa, n_token, _ = msa.shape
        
        # Line 1: Layer normalization
        msa_norm = self.norm_msa(msa)  # [batch, N_msa, N_token, c_msa]
        pair_norm = self.norm_pair(pair)  # [batch, N_token, N_token, c_pair]
        
        # Line 2: Project to values (multi-head)
        v = self.linear_v(msa_norm)  # [batch, N_msa, N_token, c*n_heads]
        v = v.view(batch_size, n_msa, n_token, self.n_heads, self.c)
        # v: [batch, N_msa, N_token, n_heads, c]
        
        # Line 3: Project pair to bias (multi-head)
        b = self.linear_b(pair_norm)  # [batch, N_token, N_token, n_heads]
        
        # Line 4: Compute gate (multi-head)
        g = torch.sigmoid(self.linear_g(msa_norm))  # [batch, N_msa, N_token, c*n_heads]
        g = g.view(batch_size, n_msa, n_token, self.n_heads, self.c)
        # g: [batch, N_msa, N_token, n_heads, c]
        
        # Line 5: Compute attention weights from pair bias
        # w^h_ij = softmax_j(b^h_ij)
        # b: [batch, i, j, heads]
        # We want softmax over j (last token dimension)
        w = torch.softmax(b, dim=2)  # [batch, N_token, N_token, n_heads]
        
        # Line 6: Apply attention to values with gating
        # For each sequence s and position i:
        # o^h_si = g^h_si ⊙ Σ_j w^h_ij v^h_sj
        #
        # v: [batch, s, j, heads, c]
        # w: [batch, i, j, heads]
        # Result: [batch, s, i, heads, c]
        
        # Use einsum for the weighted sum
        # w[i,j,h] * v[s,j,h,c] summed over j → [s,i,h,c]
        o = torch.einsum('bijh,bsjhc->bsihc', w, v)
        # o: [batch, N_msa, N_token, n_heads, c]
        
        # Apply gating
        o = g * o  # [batch, N_msa, N_token, n_heads, c]
        
        # Line 7: Concatenate heads and project to output
        o = o.reshape(batch_size, n_msa, n_token, self.n_heads * self.c)
        msa_update = self.linear_out(o)  # [batch, N_msa, N_token, c_msa]
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            msa_update = msa_update.squeeze(0)
        
        return msa_update


def create_dummy_msa_pair_input(
    n_msa: int = 512,
    n_token: int = 10,
    c_msa: int = 64,
    c_pair: int = 128,
    batch_size: int = None
) -> tuple:
    """
    Create dummy MSA and pair inputs for testing MSAPairWeightedAveraging.
    
    Args:
        n_msa: Number of MSA sequences
        n_token: Number of tokens
        c_msa: MSA representation channels
        c_pair: Pair representation channels
        batch_size: Optional batch size (if None, returns unbatched)
        
    Returns:
        (msa, pair): Tuple of dummy tensors
    """
    if batch_size is None:
        msa = torch.randn(n_msa, n_token, c_msa)
        pair = torch.randn(n_token, n_token, c_pair)
    else:
        msa = torch.randn(batch_size, n_msa, n_token, c_msa)
        pair = torch.randn(batch_size, n_token, n_token, c_pair)
    
    return msa, pair