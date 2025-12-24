"""
AlphaFold3 MSA Module

File: src/models/trunk/msa_module.py

Implements Algorithm 8: MSA Module

The MSA Module is a critical component that processes Multiple Sequence Alignment
(MSA) data to enrich the pair representation with evolutionary information. It
consists of 4 homogeneous blocks that alternate between:
1. MSA→Pair communication (OuterProductMean)
2. MSA processing (attention + transition)
3. Pair processing (triangle updates + attention + transition)

Key features:
- Samples random MSA subset each recycling iteration
- 4 homogeneous blocks for iterative refinement
- Transfers evolutionary co-evolution patterns to pair representation
- All information flows through pair (no direct MSA row-to-row communication)
- Uses dropout for regularization during training

Mathematical intuition:
    MSA contains evolutionary information about which residues co-evolve.
    OuterProductMean transfers this to pair representation.
    Triangle updates enforce geometric consistency.
    MSAPairWeightedAveraging allows pair to guide MSA processing.
"""

import torch
import torch.nn as nn
from typing import Optional

from .outer_product_mean import OuterProductMean
from .msa_pair_weighted_averaging import MSAPairWeightedAveraging
from .transition import Transition
from .triangle_updates import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming
)
from .triangle_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)


class MSAModule(nn.Module):
    """
    MSA Module - processes MSA to enrich pair representation.
    
    Implements Algorithm 8 from AF3 supplementary.
    
    The MSA Module fulfills a similar role to the Extra MSA Stack in
    AlphaFold 2. It samples a new random subset of the MSA for each
    recycling iteration, embeds the sequences, then processes them
    through 4 homogeneous blocks.
    
    Each block:
    1. Communication: MSA → Pair (OuterProductMean)
    2. MSA Stack: Pair-weighted attention + Transition
    3. Pair Stack: Triangle updates + Triangle attention + Transition
    
    Conceptual difference from AF2: No direct MSA row-to-row communication.
    All information flows via pair representation. This encourages the pair
    to contain maximal structural information.
    
    Args:
        c_msa: MSA representation channels (default: 64)
        c_pair: Pair representation channels (default: 128)
        c_single: Single representation channels (default: 384)
        n_blocks: Number of processing blocks (default: 4)
        c_outer: OuterProductMean intermediate dimension (default: 32)
        c_attn: MSA attention dimension per head (default: 8)
        n_heads_msa: MSA attention heads (default: 8)
        n_heads_pair: Pair attention heads (default: 4)
        dropout_msa: MSA dropout rate (default: 0.15)
        dropout_pair_row: Pair row-wise dropout rate (default: 0.25)
        dropout_pair_col: Pair column-wise dropout rate (default: 0.25)
    """
    
    def __init__(
        self,
        c_msa: int = 64,
        c_pair: int = 128,
        c_single: int = 384,
        n_blocks: int = 4,
        c_outer: int = 32,
        c_attn: int = 8,
        n_heads_msa: int = 8,
        n_heads_pair: int = 4,
        dropout_msa: float = 0.15,
        dropout_pair_row: float = 0.25,
        dropout_pair_col: float = 0.25
    ):
        super().__init__()
        
        self.c_msa = c_msa
        self.c_pair = c_pair
        self.c_single = c_single
        self.n_blocks = n_blocks
        
        # MSA embedding (lines 1, 3-4)
        # Note: Line 1 concatenates f_msa features (done externally)
        # Line 2: MSA sampling (done externally)
        self.linear_msa = nn.Linear(c_msa, c_msa, bias=False)  # Line 3
        self.linear_single_to_msa = nn.Linear(c_single, c_msa, bias=False)  # Line 4
        
        # Create blocks (lines 5-14)
        self.blocks = nn.ModuleList([
            MSAModuleBlock(
                c_msa=c_msa,
                c_pair=c_pair,
                c_outer=c_outer,
                c_attn=c_attn,
                n_heads_msa=n_heads_msa,
                n_heads_pair=n_heads_pair,
                dropout_msa=dropout_msa,
                dropout_pair_row=dropout_pair_row,
                dropout_pair_col=dropout_pair_col
            )
            for _ in range(n_blocks)
        ])
    
    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        single: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process MSA to enrich pair representation.
        
        Args:
            msa: [N_msa, N_token, c_msa] raw MSA features (already concatenated)
                 Or [batch, N_msa, N_token, c_msa] if batched
            pair: [N_token, N_token, c_pair] pair representation
                  Or [batch, N_token, N_token, c_pair] if batched
            single: [N_token, c_single] single representation (s_inputs)
                    Or [batch, N_token, c_single] if batched
            msa_mask: Optional [N_msa, N_token] mask for MSA
                      Or [batch, N_msa, N_token] if batched
        
        Returns:
            pair: [N_token, N_token, c_pair] updated pair representation
                  Or [batch, N_token, N_token, c_pair] if batched
        
        Algorithm 8:
        1: m_Si = concat(f_msa_Si, f_has_deletion_Si, f_deletion_value_Si)
        2: {s} = SampleRandomWithoutReplacement({S})
        3: m_si ← LinearNoBias(m_si)
        4: m_si += LinearNoBias({s_inputs_i})
        5: for all l ∈ [1, ..., N_block] do
        6:   {z_ij} += OuterProductMean({m_si})
        7:   {m_si} += Dropout_Rowwise_0.15(MSAPairWeightedAveraging({m_si}, {z_ij}))
        8:   {m_si} += Transition({m_si})
        9:   {z_ij} += Dropout_Rowwise_0.25(TriangleMultiplicationOutgoing({z_ij}))
        10:  {z_ij} += Dropout_Rowwise_0.25(TriangleMultiplicationIncoming({z_ij}))
        11:  {z_ij} += Dropout_Rowwise_0.25(TriangleAttentionStartingNode({z_ij}))
        12:  {z_ij} += Dropout_Columnwise_0.25(TriangleAttentionEndingNode({z_ij}))
        13:  {z_ij} += Transition({z_ij})
        14: end for
        15: return {z_ij}
        """
        # Handle batching
        is_batched = msa.dim() == 4
        if not is_batched:
            msa = msa.unsqueeze(0)
            pair = pair.unsqueeze(0)
            single = single.unsqueeze(0)
            if msa_mask is not None:
                msa_mask = msa_mask.unsqueeze(0)
        
        # Lines 1-2: MSA concatenation and sampling done externally
        # (User provides already concatenated and sampled MSA)
        
        # Line 3: Embed MSA
        msa = self.linear_msa(msa)  # [batch, N_msa, N_token, c_msa]
        
        # Line 4: Add single representation to MSA
        # Broadcast single over MSA sequences
        single_to_msa = self.linear_single_to_msa(single)  # [batch, N_token, c_single]
        single_to_msa = single_to_msa.unsqueeze(1)  # [batch, 1, N_token, c_msa]
        msa = msa + single_to_msa  # [batch, N_msa, N_token, c_msa]
        
        # Lines 5-14: Process through blocks
        for block in self.blocks:
            pair = block(msa, pair, msa_mask)
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            pair = pair.squeeze(0)
        
        # Line 15: Return updated pair
        return pair


class MSAModuleBlock(nn.Module):
    """
    Single block of MSA Module processing.
    
    Implements lines 6-13 of Algorithm 8.
    
    Each block performs:
    1. MSA → Pair communication via OuterProductMean
    2. MSA processing via attention and transition
    3. Pair processing via triangle updates, attention, and transition
    """
    
    def __init__(
        self,
        c_msa: int = 64,
        c_pair: int = 128,
        c_outer: int = 32,
        c_attn: int = 8,
        n_heads_msa: int = 8,
        n_heads_pair: int = 4,
        dropout_msa: float = 0.15,
        dropout_pair_row: float = 0.25,
        dropout_pair_col: float = 0.25
    ):
        super().__init__()
        
        self.dropout_msa = dropout_msa
        self.dropout_pair_row = dropout_pair_row
        self.dropout_pair_col = dropout_pair_col
        
        # Line 6: MSA → Pair communication
        self.outer_product_mean = OuterProductMean(
            c_msa=c_msa,
            c_pair=c_pair,
            c=c_outer
        )
        
        # Line 7: MSA attention
        self.msa_attention = MSAPairWeightedAveraging(
            c_msa=c_msa,
            c_pair=c_pair,
            c=c_attn,
            n_heads=n_heads_msa
        )
        
        # Line 8: MSA transition
        self.msa_transition = Transition(c=c_msa, n=4)
        
        # Lines 9-10: Triangle multiplicative updates
        self.triangle_mult_outgoing = TriangleMultiplicationOutgoing(
            c_pair=c_pair,
            c=128  # Algorithm 12 default
        )
        self.triangle_mult_incoming = TriangleMultiplicationIncoming(
            c_pair=c_pair,
            c=128  # Algorithm 13 default
        )
        
        # Lines 11-12: Triangle attention
        self.triangle_attn_starting = TriangleAttentionStartingNode(
            c_pair=c_pair,
            c=32,  # Algorithm 14 default
            n_heads=n_heads_pair
        )
        self.triangle_attn_ending = TriangleAttentionEndingNode(
            c_pair=c_pair,
            c=32,  # Algorithm 15 default
            n_heads=n_heads_pair
        )
        
        # Line 13: Pair transition
        self.pair_transition = Transition(c=c_pair, n=4)
        
        # Dropout layers
        self.dropout_rowwise = nn.Dropout(dropout_pair_row) if dropout_pair_row > 0 else nn.Identity()
        self.dropout_columnwise = nn.Dropout(dropout_pair_col) if dropout_pair_col > 0 else nn.Identity()
        self.dropout_msa_layer = nn.Dropout(dropout_msa) if dropout_msa > 0 else nn.Identity()
    
    def forward(
        self,
        msa: torch.Tensor,
        pair: torch.Tensor,
        msa_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process one block of MSA Module.
        
        Args:
            msa: [batch, N_msa, N_token, c_msa]
            pair: [batch, N_token, N_token, c_pair]
            msa_mask: Optional [batch, N_msa, N_token]
        
        Returns:
            pair: [batch, N_token, N_token, c_pair] updated pair
        """
        # Line 6: Communication - MSA to Pair
        pair = pair + self.outer_product_mean(msa)
        
        # Lines 7-8: MSA stack
        msa_update = self.msa_attention(msa, pair)
        msa_update = self.apply_dropout_rowwise_msa(msa_update)
        msa = msa + msa_update
        
        msa = msa + self.msa_transition(msa)
        
        # Lines 9-13: Pair stack
        # Triangle multiplicative updates (rowwise dropout)
        pair_update = self.triangle_mult_outgoing(pair)
        pair_update = self.apply_dropout_rowwise_pair(pair_update)
        pair = pair + pair_update
        
        pair_update = self.triangle_mult_incoming(pair)
        pair_update = self.apply_dropout_rowwise_pair(pair_update)
        pair = pair + pair_update
        
        # Triangle attention (rowwise and columnwise dropout)
        pair_update = self.triangle_attn_starting(pair)
        pair_update = self.apply_dropout_rowwise_pair(pair_update)
        pair = pair + pair_update
        
        pair_update = self.triangle_attn_ending(pair)
        pair_update = self.apply_dropout_columnwise_pair(pair_update)
        pair = pair + pair_update
        
        # Transition
        pair = pair + self.pair_transition(pair)
        
        return pair
    
    def apply_dropout_rowwise_msa(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rowwise dropout to MSA (drop entire sequences)."""
        if not self.training or self.dropout_msa == 0:
            return x
        # Dropout along sequence dimension (dim=1 in batched [batch, N_msa, N_token, c])
        # Create mask: [batch, N_msa, 1, 1] and broadcast
        batch_size, n_msa = x.shape[:2]
        mask = torch.bernoulli(
            torch.ones(batch_size, n_msa, 1, 1, device=x.device) * (1 - self.dropout_msa)
        )
        return x * mask / (1 - self.dropout_msa)
    
    def apply_dropout_rowwise_pair(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rowwise dropout to pair (drop entire rows)."""
        if not self.training or self.dropout_pair_row == 0:
            return x
        # Dropout along first pair dimension (dim=1 in batched [batch, N_token, N_token, c])
        batch_size, n_token = x.shape[:2]
        mask = torch.bernoulli(
            torch.ones(batch_size, n_token, 1, 1, device=x.device) * (1 - self.dropout_pair_row)
        )
        return x * mask / (1 - self.dropout_pair_row)
    
    def apply_dropout_columnwise_pair(self, x: torch.Tensor) -> torch.Tensor:
        """Apply columnwise dropout to pair (drop entire columns)."""
        if not self.training or self.dropout_pair_col == 0:
            return x
        # Dropout along second pair dimension (dim=2 in batched [batch, N_token, N_token, c])
        batch_size, _, n_token = x.shape[:3]
        mask = torch.bernoulli(
            torch.ones(batch_size, 1, n_token, 1, device=x.device) * (1 - self.dropout_pair_col)
        )
        return x * mask / (1 - self.dropout_pair_col)