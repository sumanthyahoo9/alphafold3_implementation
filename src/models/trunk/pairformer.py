"""
AlphaFold3 Pairformer Stack

File: src/models/trunk/pairformer.py

Implements Algorithm 17: Pairformer Stack

The Pairformer Stack is the main trunk of AlphaFold3, replacing the Evoformer
from AlphaFold2. It's a simplified architecture that processes only single + pair
representations (no MSA in the core).

Key simplifications vs Evoformer:
- No MSA representation in core (handled separately in MSA Module)
- No column-wise attention
- No outer product mean from single→pair (no information flow single→pair)
- Pair only influences single (via attention bias), not vice versa

Architecture (48 blocks):
- Pair stack: Triangle updates + Triangle attention + Transition
- Single stack: Attention (pair-biased) + Transition

Mathematical intuition:
    Pair representation contains all structural information.
    Single representation is like the "query sequence" being folded.
    Pair biases single's attention - structure guides sequence processing.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .transition import Transition
from .triangle_updates import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming
)
from .triangle_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)
from .attention_pair_bias import AttentionPairBias


class PairformerStack(nn.Module):
    """
    Pairformer Stack - main trunk of AlphaFold3.
    
    Implements Algorithm 17 from AF3 supplementary.
    
    The Pairformer Stack fulfills a similar role to the Evoformer in
    AlphaFold 2, but with a simplified architecture. It uses only a
    single representation (like the privileged first MSA row in AF2)
    and a pair representation.
    
    Key architectural choices:
    1. No column-wise attention (consequence of single vs MSA)
    2. Single attention with pair bias (like row-wise attention in AF2)
    3. NO single→pair information flow (unlike AF2's outer product mean)
    4. Pair influences single via attention biasing only
    
    This means the pair representation is the "backbone" containing
    all structural information, and the single just processes the
    query sequence guided by pair.
    
    Args:
        c_single: Single representation channels (default: 384)
        c_pair: Pair representation channels (default: 128)
        n_blocks: Number of blocks (default: 48)
        c_hidden_tri: Triangle update hidden dimension (default: 128)
        c_hidden_attn_tri: Triangle attention dimension per head (default: 32)
        c_hidden_attn_single: Single attention dimension per head (default: 32)
        n_heads_tri: Triangle attention heads (default: 4)
        n_heads_single: Single attention heads (default: 16)
        dropout_pair_row: Pair row-wise dropout rate (default: 0.25)
        dropout_pair_col: Pair column-wise dropout rate (default: 0.25)
    """
    
    def __init__(
        self,
        c_single: int = 384,
        c_pair: int = 128,
        n_blocks: int = 48,
        c_hidden_tri: int = 128,
        c_hidden_attn_tri: int = 32,
        c_hidden_attn_single: int = 32,
        n_heads_tri: int = 4,
        n_heads_single: int = 16,
        dropout_pair_row: float = 0.25,
        dropout_pair_col: float = 0.25
    ):
        super().__init__()
        
        self.c_single = c_single
        self.c_pair = c_pair
        self.n_blocks = n_blocks
        
        # Create blocks
        self.blocks = nn.ModuleList([
            PairformerBlock(
                c_single=c_single,
                c_pair=c_pair,
                c_hidden_tri=c_hidden_tri,
                c_hidden_attn_tri=c_hidden_attn_tri,
                c_hidden_attn_single=c_hidden_attn_single,
                n_heads_tri=n_heads_tri,
                n_heads_single=n_heads_single,
                dropout_pair_row=dropout_pair_row,
                dropout_pair_col=dropout_pair_col
            )
            for _ in range(n_blocks)
        ])
    
    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        single_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process single and pair through Pairformer stack.
        
        Args:
            single: [N_token, c_single] single representation
                    Or [batch, N_token, c_single] if batched
            pair: [N_token, N_token, c_pair] pair representation
                  Or [batch, N_token, N_token, c_pair] if batched
            single_mask: Optional [N_token] mask
                         Or [batch, N_token] if batched
            pair_mask: Optional [N_token, N_token] attention mask
                       Or [batch, N_token, N_token] if batched
        
        Returns:
            (single, pair): Updated representations
        
        Algorithm 17:
        1: for all l ∈ [1, ..., N_block] do
        2:   {z_ij} += Dropout_Rowwise_0.25(TriangleMultiplicationOutgoing({z_ij}))
        3:   {z_ij} += Dropout_Rowwise_0.25(TriangleMultiplicationIncoming({z_ij}))
        4:   {z_ij} += Dropout_Rowwise_0.25(TriangleAttentionStartingNode({z_ij}))
        5:   {z_ij} += Dropout_Columnwise_0.25(TriangleAttentionEndingNode({z_ij}))
        6:   {z_ij} += Transition({z_ij})
        7:   {s_i} += AttentionPairBias({s_i}, ∅, {z_ij}, β_ij=0, N_head=16)
        8:   {s_i} += Transition({s_i})
        9: end for
        10: return {s_i}, {z_ij}
        """
        # Handle batching
        is_batched = single.dim() == 3
        if not is_batched:
            single = single.unsqueeze(0)
            pair = pair.unsqueeze(0)
            if single_mask is not None:
                single_mask = single_mask.unsqueeze(0)
            if pair_mask is not None:
                pair_mask = pair_mask.unsqueeze(0)
        
        # Process through blocks (lines 1-9)
        for block in self.blocks:
            single, pair = block(single, pair, single_mask, pair_mask)
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            single = single.squeeze(0)
            pair = pair.squeeze(0)
        
        # Line 10: Return both representations
        return single, pair


class PairformerBlock(nn.Module):
    """
    Single block of Pairformer processing.
    
    Implements lines 2-8 of Algorithm 17.
    
    Each block:
    1. Pair stack: Triangle updates + attention + transition
    2. Single stack: Attention (pair-biased) + transition
    
    Note: Single does NOT influence pair (key difference from AF2)!
    """
    
    def __init__(
        self,
        c_single: int = 384,
        c_pair: int = 128,
        c_hidden_tri: int = 128,
        c_hidden_attn_tri: int = 32,
        c_hidden_attn_single: int = 32,
        n_heads_tri: int = 4,
        n_heads_single: int = 16,
        dropout_pair_row: float = 0.25,
        dropout_pair_col: float = 0.25
    ):
        super().__init__()
        
        self.dropout_pair_row = dropout_pair_row
        self.dropout_pair_col = dropout_pair_col
        
        # Lines 2-3: Triangle multiplicative updates
        self.triangle_mult_outgoing = TriangleMultiplicationOutgoing(
            c_pair=c_pair,
            c=c_hidden_tri
        )
        self.triangle_mult_incoming = TriangleMultiplicationIncoming(
            c_pair=c_pair,
            c=c_hidden_tri
        )
        
        # Lines 4-5: Triangle attention
        self.triangle_attn_starting = TriangleAttentionStartingNode(
            c_pair=c_pair,
            c=c_hidden_attn_tri,
            n_heads=n_heads_tri
        )
        self.triangle_attn_ending = TriangleAttentionEndingNode(
            c_pair=c_pair,
            c=c_hidden_attn_tri,
            n_heads=n_heads_tri
        )
        
        # Line 6: Pair transition
        self.pair_transition = Transition(c=c_pair, n=4)
        
        # Line 7: Single attention with pair bias
        self.single_attention = AttentionPairBias(
            c_single=c_single,
            c_pair=c_pair,
            c=c_hidden_attn_single,
            n_heads=n_heads_single
        )
        
        # Line 8: Single transition
        self.single_transition = Transition(c=c_single, n=4)
    
    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        single_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process one Pairformer block.
        
        Args:
            single: [batch, N_token, c_single]
            pair: [batch, N_token, N_token, c_pair]
            single_mask: Optional [batch, N_token]
            pair_mask: Optional [batch, N_token, N_token]
        
        Returns:
            (single, pair): Updated representations
        """
        # Lines 2-6: Pair stack
        # Triangle multiplicative updates (rowwise dropout)
        pair_update = self.triangle_mult_outgoing(pair)
        pair_update = self.apply_dropout_rowwise(pair_update)
        pair = pair + pair_update
        
        pair_update = self.triangle_mult_incoming(pair)
        pair_update = self.apply_dropout_rowwise(pair_update)
        pair = pair + pair_update
        
        # Triangle attention (rowwise and columnwise dropout)
        pair_update = self.triangle_attn_starting(pair)
        pair_update = self.apply_dropout_rowwise(pair_update)
        pair = pair + pair_update
        
        pair_update = self.triangle_attn_ending(pair)
        pair_update = self.apply_dropout_columnwise(pair_update)
        pair = pair + pair_update
        
        # Pair transition
        pair = pair + self.pair_transition(pair)
        
        # Lines 7-8: Single stack
        # Attention with pair bias (no dropout specified in algorithm)
        single = single + self.single_attention(single, pair, pair_mask)
        
        # Single transition
        single = single + self.single_transition(single)
        
        return single, pair
    
    def apply_dropout_rowwise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rowwise dropout to pair."""
        if not self.training or self.dropout_pair_row == 0:
            return x
        # Drop entire rows (dim=1 in [batch, N_token, N_token, c])
        batch_size, n_token = x.shape[:2]
        mask = torch.bernoulli(
            torch.ones(batch_size, n_token, 1, 1, device=x.device) * (1 - self.dropout_pair_row)
        )
        return x * mask / (1 - self.dropout_pair_row)
    
    def apply_dropout_columnwise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply columnwise dropout to pair."""
        if not self.training or self.dropout_pair_col == 0:
            return x
        # Drop entire columns (dim=2 in [batch, N_token, N_token, c])
        batch_size, _, n_token = x.shape[:3]
        mask = torch.bernoulli(
            torch.ones(batch_size, 1, n_token, 1, device=x.device) * (1 - self.dropout_pair_col)
        )
        return x * mask / (1 - self.dropout_pair_col)