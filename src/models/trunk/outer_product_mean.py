"""
AlphaFold3 Outer Product Mean

File: src/models/trunk/outer_product_mean.py

Implements Algorithm 9: Outer product mean

This layer transfers evolutionary information from the MSA representation
to the pair representation by computing the outer product of MSA row projections
and averaging over sequences.

Key features:
- Captures co-evolution patterns between token pairs
- Mean pooling over MSA sequences (flattened outer product)
- Critical for MSA → pair information flow
- No attention mechanism (just outer product + pooling)
"""

import torch
import torch.nn as nn


class OuterProductMean(nn.Module):
    """
    Outer product mean for MSA-to-pair communication.
    
    Implements Algorithm 9 from AF3 supplementary.
    
    This layer computes the outer product of projected MSA representations
    and averages over sequences. This captures co-evolutionary information:
    if positions i and j co-evolve, their projections will be correlated
    across MSA sequences.
    
    Args:
        c_msa: MSA representation channels (default: 64)
        c_pair: Pair representation channels (default: 128)
        c: Intermediate projection dimension (default: 32)
    
    Mathematical formulation:
        a_si, b_si = Linear(LayerNorm(m_si))  # Project MSA
        o_ij = mean_over_s(flatten(a_si ⊗ b_sj))  # Outer product + mean
        z_ij = Linear(o_ij)  # Project to pair dimension
    
    Why this works:
        If positions i and j co-evolve (e.g., forming a contact), their
        amino acid distributions across sequences will be correlated.
        The outer product captures these pairwise correlations.
    """
    
    def __init__(
        self,
        c_msa: int = 64,
        c_pair: int = 128,
        c: int = 32
    ):
        super().__init__()
        
        self.c_msa = c_msa
        self.c_pair = c_pair
        self.c = c
        
        # Layer normalization for MSA
        self.norm = nn.LayerNorm(c_msa)
        
        # Line 2: Project to intermediate dimension
        # Two separate projections for outer product
        self.linear_a = nn.Linear(c_msa, c, bias=False)
        self.linear_b = nn.Linear(c_msa, c, bias=False)
        
        # Line 4: Project flattened outer product to pair dimension
        # Flattened outer product has dimension c*c
        self.linear_out = nn.Linear(c * c, c_pair, bias=True)
    
    def forward(self, msa: torch.Tensor) -> torch.Tensor:
        """
        Compute outer product mean over MSA sequences.
        
        Args:
            msa: [N_msa, N_token, c_msa] MSA representation
                 Can also be [batch, N_msa, N_token, c_msa]
        
        Returns:
            pair_update: [N_token, N_token, c_pair] pair representation update
                         Or [batch, N_token, N_token, c_pair] if batched
        
        Algorithm 9:
        1: m_si ← LayerNorm(m_si)
        2: a_si, b_si = LinearNoBias(m_si)  # a_si, b_si ∈ R^c
        3: o_ij = flatten(mean_s(a_si ⊗ b_sj))  # o_ij ∈ R^(c·c)
        4: z_ij = Linear(o_ij)  # z_ij ∈ R^c_pair
        5: return {z_ij}
        """
        # Handle both batched and unbatched inputs
        is_batched = msa.dim() == 4
        if not is_batched:
            msa = msa.unsqueeze(0)  # Add batch dimension
        
        batch_size, n_msa, n_token, _ = msa.shape
        
        # Line 1: Layer normalization
        msa_norm = self.norm(msa)  # [batch, N_msa, N_token, c_msa]
        
        # Line 2: Project to intermediate dimension
        a = self.linear_a(msa_norm)  # [batch, N_msa, N_token, c]
        b = self.linear_b(msa_norm)  # [batch, N_msa, N_token, c]
        
        # Line 3: Compute outer product and mean over sequences
        # a_si ⊗ b_sj creates [batch, N_msa, N_token, N_token, c, c]
        # We want: for each sequence s, compute a[s,i] ⊗ b[s,j]
        # Then average over s
        
        # Reshape for outer product
        # a: [batch, N_msa, N_token, 1, c, 1]
        # b: [batch, N_msa, 1, N_token, 1, c]
        a_outer = a.unsqueeze(3).unsqueeze(-1)  # [batch, N_msa, N_token, 1, c, 1]
        b_outer = b.unsqueeze(2).unsqueeze(-2)  # [batch, N_msa, 1, N_token, 1, c]
        
        # Outer product: [batch, N_msa, N_token, N_token, c, c]
        outer = a_outer * b_outer
        
        # Mean over MSA sequences (dimension 1)
        outer_mean = outer.mean(dim=1)  # [batch, N_token, N_token, c, c]
        
        # Flatten the last two dimensions
        outer_flat = outer_mean.reshape(batch_size, n_token, n_token, self.c * self.c)
        
        # Line 4: Project to pair dimension
        pair_update = self.linear_out(outer_flat)  # [batch, N_token, N_token, c_pair]
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            pair_update = pair_update.squeeze(0)
        
        return pair_update


def create_dummy_msa_input(
    n_msa: int = 512,
    n_token: int = 10,
    c_msa: int = 64,
    batch_size: int = 0
) -> torch.Tensor:
    """
    Create dummy MSA input for testing OuterProductMean.
    
    Args:
        n_msa: Number of MSA sequences
        n_token: Number of tokens
        c_msa: MSA representation channels
        batch_size: Optional batch size (if None, returns unbatched)
        
    Returns:
        msa: [N_msa, N_token, c_msa] or [batch, N_msa, N_token, c_msa]
    """
    if batch_size is None:
        return torch.randn(n_msa, n_token, c_msa)
    else:
        return torch.randn(batch_size, n_msa, n_token, c_msa)