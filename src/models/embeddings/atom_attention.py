"""
AlphaFold3 Atom Attention Encoder and Decoder

Implements Algorithm 5: Atom attention encoder
Implements Algorithm 6: Atom attention decoder
Implements Algorithm 7: Atom Transformer (sequence-local attention)

This module processes per-atom features and aggregates them to per-token representations
using sequence-local attention. Critical for learning local geometry independent of
tokenization scheme.

Key features:
- Embeds atom metadata (positions, charges, elements, names)
- Computes atom-atom pair features with distance-based embeddings
- Applies sequence-local attention (32x128 windows)
- Aggregates atoms to tokens via mean pooling (permutation invariant)
- Decodes token activations back to atom position updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class AtomTransformer(nn.Module):
    """
    Applies sequence-local attention to atom representations.
    
    Implements Algorithm 7 from AF3 supplementary.
    
    This creates a block-sparse attention pattern where each subset of 32 atoms
    (queries) attends to nearby 128 atoms (keys/values) in sequence space.
    
    Args:
        c_atom: Atom representation channels (default: 128)
        c_atompair: Atom pair representation channels (default: 16)
        n_blocks: Number of transformer blocks (default: 3)
        n_heads: Number of attention heads (default: 4)
        n_queries: Query window size (default: 32)
        n_keys: Key/value window size (default: 128)
    """
    
    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128
    ):
        super().__init__()
        
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_keys = n_keys
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            AtomTransformerBlock(
                c_atom=c_atom,
                c_atompair=c_atompair,
                n_heads=n_heads
            )
            for _ in range(n_blocks)
        ])
    
    def forward(
        self,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply sequence-local attention.
        
        Args:
            ql: [N_atom, c_atom] atom queries
            cl: [N_atom, c_atom] atom conditioning
            plm: [N_atom, N_atom, c_atompair] atom pair features
        
        Returns:
            ql: [N_atom, c_atom] transformed atom representations
        
        Algorithm 7:
        1: Compute sequence-local attention mask βlm
        2: Apply DiffusionTransformer with mask
        3: Return transformed queries
        """
        n_atoms = ql.shape[0]
        
        # Line 1: Create sequence-local attention mask
        # βlm = 0 if |l - c| < Nqueries/2 AND |m - c| < Nkeys/2 for any center c
        # βlm = -10^10 else (mask out)
        beta_mask = self._create_sequence_local_mask(n_atoms)
        
        # Move mask to same device as input
        beta_mask = beta_mask.to(ql.device)
        
        # Line 2: Apply transformer blocks with mask
        for block in self.blocks:
            ql = block(ql, cl, plm, beta_mask)
        
        return ql
    
    def _create_sequence_local_mask(self, n_atoms: int) -> torch.Tensor:
        """
        Create sequence-local attention mask.
        
        Each subset of n_queries atoms attends to nearby n_keys atoms.
        
        Args:
            n_atoms: Total number of atoms
        
        Returns:
            beta_mask: [N_atom, N_atom] attention mask
                       0 = allowed, -1e10 = masked
        
        Algorithm 7 Line 1:
        βlm = 0 if |l - c| < Nqueries/2 AND |m - c| < Nkeys/2 for any center c
        βlm = -10^10 else
        
        Subset centers: {15.5, 47.5, 79.5, ...} (every n_queries=32 atoms)
        
        Implementation: For each query atom l, we check if there exists any center c
        such that |l - c| < n_queries/2. If yes, that atom can attend to keys m where
        |m - c| < n_keys/2 for the same center c.
        """
        # Initialize mask to -1e10 (all masked)
        beta_mask = torch.full((n_atoms, n_atoms), -1e10, dtype=torch.float32)
        
        # Compute subset centers: 15.5, 47.5, 79.5, ...
        center_offset = self.n_queries / 2 - 0.5
        
        # For each center
        center_idx = 0
        while True:
            center = center_offset + center_idx * self.n_queries
            
            # Stop if center is beyond atoms
            if center >= n_atoms:
                break
            
            # Query range: atoms where |l - center| < n_queries/2
            # This means: center - n_queries/2 < l < center + n_queries/2
            query_start = max(0, int(center - self.n_queries / 2 + 1))
            query_end = min(n_atoms, int(center + self.n_queries / 2 + 1))
            
            # Key range: atoms where |m - center| < n_keys/2
            # This means: center - n_keys/2 < m < center + n_keys/2
            key_start = max(0, int(center - self.n_keys / 2 + 1))
            key_end = min(n_atoms, int(center + self.n_keys / 2 + 1))
            
            # Ensure key range is at least n_keys (for edge cases)
            if key_end - key_start < self.n_keys and key_end < n_atoms:
                key_end = min(n_atoms, key_start + self.n_keys)
            
            # Unmask this block
            beta_mask[query_start:query_end, key_start:key_end] = 0.0
            
            center_idx += 1
        
        return beta_mask


class AtomTransformerBlock(nn.Module):
    """
    Single transformer block with pair-biased attention.
    
    Simplified version of DiffusionTransformer (Algorithm 23-24).
    """
    
    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        n_heads: int = 4
    ):
        super().__init__()
        
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.n_heads = n_heads
        self.c_head = c_atom // n_heads
        
        # Layer norms
        self.norm_q = nn.LayerNorm(c_atom)
        self.norm_pair = nn.LayerNorm(c_atompair)
        
        # Attention projections
        self.q_proj = nn.Linear(c_atom, c_atom)
        self.k_proj = nn.Linear(c_atom, c_atom, bias=False)
        self.v_proj = nn.Linear(c_atom, c_atom, bias=False)
        self.pair_bias_proj = nn.Linear(c_atompair, n_heads, bias=False)
        self.gate_proj = nn.Linear(c_atom, c_atom, bias=False)
        self.out_proj = nn.Linear(c_atom, c_atom, bias=False)
        
        # Transition (MLP)
        self.transition = nn.Sequential(
            nn.LayerNorm(c_atom),
            nn.Linear(c_atom, c_atom * 4),
            nn.ReLU(),
            nn.Linear(c_atom * 4, c_atom)
        )
    
    def forward(
        self,
        ql: torch.Tensor,
        cl: torch.Tensor,
        plm: torch.Tensor,
        beta_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply attention with pair bias and mask.
        
        Args:
            ql: [N_atom, c_atom] queries
            cl: [N_atom, c_atom] conditioning
            plm: [N_atom, N_atom, c_atompair] pair features
            beta_mask: [N_atom, N_atom] attention mask
        
        Returns:
            ql: [N_atom, c_atom] updated queries
        """
        n_atoms = ql.shape[0]
        
        # Residual connection
        ql_input = ql
        
        # Normalize
        ql_norm = self.norm_q(ql)
        
        # Project to Q, K, V
        q = self.q_proj(ql_norm)  # [N_atom, c_atom]
        k = self.k_proj(ql_norm)  # [N_atom, c_atom]
        v = self.v_proj(ql_norm)  # [N_atom, c_atom]
        
        # Reshape for multi-head attention
        q = q.view(n_atoms, self.n_heads, self.c_head)  # [N_atom, n_heads, c_head]
        k = k.view(n_atoms, self.n_heads, self.c_head)
        v = v.view(n_atoms, self.n_heads, self.c_head)
        
        # Compute attention logits
        # [N_atom, n_heads, c_head] @ [N_atom, n_heads, c_head]^T
        # -> [N_atom, n_heads, N_atom]
        attn_logits = torch.einsum('ihc,jhc->hij', q, k) / math.sqrt(self.c_head)
        
        # Add pair bias
        pair_bias = self.pair_bias_proj(self.norm_pair(plm))  # [N_atom, N_atom, n_heads]
        pair_bias = pair_bias.permute(2, 0, 1)  # [n_heads, N_atom, N_atom]
        attn_logits = attn_logits + pair_bias
        
        # Add sequence-local mask
        attn_logits = attn_logits + beta_mask.unsqueeze(0)  # [n_heads, N_atom, N_atom]
        
        # Softmax
        attn_weights = F.softmax(attn_logits, dim=-1)  # [n_heads, N_atom, N_atom]
        
        # Apply attention
        # [n_heads, N_atom, N_atom] @ [N_atom, n_heads, c_head]
        # -> [n_heads, N_atom, c_head]
        attn_out = torch.einsum('hij,jhc->ihc', attn_weights, v)
        attn_out = attn_out.reshape(n_atoms, self.c_atom)  # [N_atom, c_atom]
        
        # Gating
        gate = torch.sigmoid(self.gate_proj(ql_norm))
        attn_out = gate * attn_out
        
        # Output projection
        attn_out = self.out_proj(attn_out)
        
        # Residual
        ql = ql_input + attn_out
        
        # Transition (MLP)
        ql = ql + self.transition(ql)
        
        return ql


class AtomAttentionEncoder(nn.Module):
    """
    Encodes atom-level features into token-level representations.
    
    Implements Algorithm 5 from AF3 supplementary.
    
    Args:
        c_atom: Atom single representation channels (default: 128)
        c_atompair: Atom pair representation channels (default: 16)
        c_token: Token representation channels (default: 384)
        n_blocks: Number of transformer blocks (default: 3)
        n_heads: Number of attention heads (default: 4)
    """
    
    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
        n_blocks: int = 3,
        n_heads: int = 4
    ):
        super().__init__()
        
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        
        # Line 1: Embed per-atom metadata
        # Input: concat(ref_pos[3], ref_charge[1], ref_mask[1], ref_element[128], ref_atom_name_chars[4*64])
        # Total input dim: 3 + 1 + 1 + 128 + 256 = 389
        self.atom_single_embed = nn.Linear(389, c_atom, bias=False)
        
        # Line 4: Embed position offsets (3D vectors)
        self.pos_offset_embed = nn.Linear(3, c_atompair, bias=False)
        
        # Line 5: Embed inverse squared distances
        self.inv_distance_embed = nn.Linear(1, c_atompair, bias=False)
        
        # Line 6: Embed valid mask
        self.mask_embed = nn.Linear(1, c_atompair, bias=False)
        
        # Lines 9-10: Trunk embeddings (if provided)
        self.trunk_single_proj = nn.Linear(384, c_atom, bias=False)  # Assuming c_single=384
        self.trunk_pair_proj = nn.Linear(128, c_atompair, bias=False)  # Assuming c_pair=128
        
        # Line 11: Noisy position embedding
        self.noisy_pos_embed = nn.Linear(3, c_atom, bias=False)
        
        # Line 13: Single conditioning to pair (2 projections)
        self.single_to_pair_1 = nn.Linear(c_atom, c_atompair, bias=False)
        self.single_to_pair_2 = nn.Linear(c_atom, c_atompair, bias=False)
        
        # Line 14: MLP on pair activations (3-layer)
        self.pair_mlp = nn.Sequential(
            nn.Linear(c_atompair, c_atompair, bias=False),
            nn.ReLU(),
            nn.Linear(c_atompair, c_atompair, bias=False),
            nn.ReLU(),
            nn.Linear(c_atompair, c_atompair, bias=False)
        )
        
        # Line 15: AtomTransformer (sequence-local attention)
        self.atom_transformer = AtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_blocks=n_blocks,
            n_heads=n_heads
        )
        
        # Line 16: Aggregate to token representation
        self.atom_to_token_proj = nn.Linear(c_atom, c_token, bias=False)
        
        # Layer norms
        self.trunk_single_norm = nn.LayerNorm(384)
        self.trunk_pair_norm = nn.LayerNorm(128)
    
    def forward(
        self,
        atom_features: Dict[str, torch.Tensor],
        atom_to_token_idx: torch.Tensor,
        noisy_positions: Optional[torch.Tensor] = None,
        trunk_single: Optional[torch.Tensor] = None,
        trunk_pair: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode atom features to token representations.
        
        Args:
            atom_features: Dict containing:
                - ref_pos: [N_atom, 3] positions
                - ref_charge: [N_atom] charges
                - ref_mask: [N_atom] mask
                - ref_element: [N_atom, 128] one-hot elements
                - ref_atom_name_chars: [N_atom, 4, 64] character encoding
                - ref_space_uid: [N_atom] residue grouping
            atom_to_token_idx: [N_atom] mapping atoms to tokens
            noisy_positions: [N_atom, 3] optional noisy coordinates
            trunk_single: [N_token, 384] optional trunk single repr
            trunk_pair: [N_token, N_token, 128] optional trunk pair repr
        
        Returns:
            ai: [N_token, c_token] aggregated token representation
            q_skip: [N_atom, c_atom] atom single (for decoder skip)
            c_skip: [N_atom, c_atom] atom conditioning (for decoder skip)
            p_skip: [N_atom, N_atom, c_atompair] atom pair (for decoder skip)
        
        Algorithm 5:
        1: cl = LinearNoBias(concat(ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars))
        2-6: Compute atom pair features plm from positions and masks
        7: Initialize ql = cl
        8-12: Optionally add trunk embeddings and noisy positions
        13-14: Add single conditioning to pair, run MLP
        15: Apply AtomTransformer
        16: Aggregate atoms to tokens via mean pooling
        17-18: Return outputs and skip connections
        """
        # Extract features
        ref_pos = atom_features['ref_pos']  # [N_atom, 3]
        ref_charge = atom_features['ref_charge'].unsqueeze(-1)  # [N_atom, 1]
        ref_mask = atom_features['ref_mask'].unsqueeze(-1)  # [N_atom, 1]
        ref_element = atom_features['ref_element']  # [N_atom, 128]
        ref_atom_name_chars = atom_features['ref_atom_name_chars']  # [N_atom, 4, 64]
        ref_space_uid = atom_features['ref_space_uid']  # [N_atom]
        
        n_atoms = ref_pos.shape[0]
        n_tokens = atom_to_token_idx.max().item() + 1
        
        # Flatten atom name chars
        ref_atom_name_chars_flat = ref_atom_name_chars.reshape(n_atoms, -1)  # [N_atom, 256]
        
        # Line 1: Embed per-atom metadata
        atom_input = torch.cat([
            ref_pos,
            ref_charge,
            ref_mask,
            ref_element,
            ref_atom_name_chars_flat
        ], dim=-1)  # [N_atom, 389]
        
        cl = self.atom_single_embed(atom_input)  # [N_atom, c_atom]
        
        # Lines 2-6: Compute atom pair features
        # Line 2: Position offsets
        dlm = ref_pos.unsqueeze(0) - ref_pos.unsqueeze(1)  # [N_atom, N_atom, 3]
        
        # Line 3: Valid mask (same residue)
        vlm = (ref_space_uid.unsqueeze(0) == ref_space_uid.unsqueeze(1)).float()  # [N_atom, N_atom]
        
        # Line 4: Embed position offsets
        plm = self.pos_offset_embed(dlm) * vlm.unsqueeze(-1)  # [N_atom, N_atom, c_atompair]
        
        # Line 5: Inverse squared distance
        distances_sq = (dlm ** 2).sum(dim=-1, keepdim=True)  # [N_atom, N_atom, 1]
        inv_sq_dist = 1.0 / (1.0 + distances_sq)  # [N_atom, N_atom, 1]
        plm = plm + self.inv_distance_embed(inv_sq_dist) * vlm.unsqueeze(-1)
        
        # Line 6: Embed valid mask
        plm = plm + self.mask_embed(vlm.unsqueeze(-1)) * vlm.unsqueeze(-1)
        
        # Line 7: Initialize atom single representation
        ql = cl.clone()  # [N_atom, c_atom]
        
        # Lines 8-12: Add trunk embeddings and noisy positions (if provided)
        if noisy_positions is not None or trunk_single is not None or trunk_pair is not None:
            # Line 9: Broadcast trunk single to atoms
            if trunk_single is not None:
                trunk_single_atoms = trunk_single[atom_to_token_idx]  # [N_atom, 384]
                cl = cl + self.trunk_single_proj(self.trunk_single_norm(trunk_single_atoms))
            
            # Line 10: Broadcast trunk pair to atoms
            if trunk_pair is not None:
                token_idx_i = atom_to_token_idx.unsqueeze(1)  # [N_atom, 1]
                token_idx_j = atom_to_token_idx.unsqueeze(0)  # [1, N_atom]
                trunk_pair_atoms = trunk_pair[token_idx_i, token_idx_j]  # [N_atom, N_atom, 128]
                plm = plm + self.trunk_pair_proj(self.trunk_pair_norm(trunk_pair_atoms))
            
            # Line 11: Add noisy positions
            if noisy_positions is not None:
                ql = ql + self.noisy_pos_embed(noisy_positions)
        
        # Line 13: Add single conditioning to pair
        cl_relu = torch.relu(cl)
        single_to_pair_i = self.single_to_pair_1(cl_relu).unsqueeze(1)  # [N_atom, 1, c_atompair]
        single_to_pair_j = self.single_to_pair_2(cl_relu).unsqueeze(0)  # [1, N_atom, c_atompair]
        plm = plm + single_to_pair_i + single_to_pair_j
        
        # Line 14: Run MLP on pair activations
        plm = plm + self.pair_mlp(plm)
        
        # Line 15: AtomTransformer with sequence-local attention
        ql = self.atom_transformer(ql, cl, plm)
        
        # Line 16: Aggregate per-atom to per-token via mean pooling
        # ai = mean over atoms where tok_idx(l) = i
        ql_relu = torch.relu(self.atom_to_token_proj(ql))  # [N_atom, c_token]
        
        # Mean pooling: for each token, average its atoms
        ai = torch.zeros(n_tokens, self.c_token, device=ql.device, dtype=ql.dtype)
        atom_counts = torch.zeros(n_tokens, device=ql.device, dtype=ql.dtype)
        
        # Scatter add for aggregation
        ai.index_add_(0, atom_to_token_idx, ql_relu)
        atom_counts.index_add_(0, atom_to_token_idx, torch.ones(n_atoms, device=ql.device))
        
        # Divide by counts for mean
        ai = ai / atom_counts.unsqueeze(-1).clamp(min=1.0)  # [N_token, c_token]
        
        # Lines 17-18: Store skip connections
        q_skip = ql
        c_skip = cl
        p_skip = plm
        
        return ai, q_skip, c_skip, p_skip


class AtomAttentionDecoder(nn.Module):
    """
    Decodes token-level activations to atom-level position updates.
    
    Implements Algorithm 6 from AF3 supplementary.
    
    This is the inverse of AtomAttentionEncoder - it broadcasts token
    representations back to atoms and applies sequence-local attention
    to produce position updates for diffusion denoising.
    
    Args:
        c_atom: Atom representation channels (default: 128)
        c_atompair: Atom pair representation channels (default: 16)
        c_token: Token representation channels (default: 384)
        n_blocks: Number of transformer blocks (default: 3)
        n_heads: Number of attention heads (default: 4)
    """
    
    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
        n_blocks: int = 3,
        n_heads: int = 4
    ):
        super().__init__()
        
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        
        # Line 1: Broadcast token activations to atoms
        self.token_to_atom = nn.Linear(c_token, c_atom, bias=False)
        
        # Line 2: AtomTransformer (sequence-local attention)
        self.atom_transformer = AtomTransformer(
            c_atom=c_atom,
            c_atompair=c_atompair,
            n_blocks=n_blocks,
            n_heads=n_heads
        )
        
        # Line 3: Map to position updates
        self.to_position_update = nn.Linear(c_atom, 3, bias=False)
        
        # Layer norm for position projection
        self.norm = nn.LayerNorm(c_atom)
    
    def forward(
        self,
        ai: torch.Tensor,
        q_skip: torch.Tensor,
        c_skip: torch.Tensor,
        p_skip: torch.Tensor,
        atom_to_token_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode token activations to atom position updates.
        
        Args:
            ai: [N_token, c_token] token activations
            q_skip: [N_atom, c_atom] atom queries from encoder
            c_skip: [N_atom, c_atom] atom conditioning from encoder
            p_skip: [N_atom, N_atom, c_atompair] atom pairs from encoder
            atom_to_token_idx: [N_atom] mapping atoms to tokens
        
        Returns:
            r_update: [N_atom, 3] position updates
        
        Algorithm 6:
        1: ql = LinearNoBias(a[tok_idx(l)]) + q_skip_l
        2: {ql} = AtomTransformer({ql}, {c_skip_l}, {p_skip_lm}, Nblock=3, Nhead=4)
        3: r_update_l = LinearNoBias(LayerNorm(ql))
        4: return {r_update_l}
        """
        n_atoms = q_skip.shape[0]
        
        # Line 1: Broadcast per-token activations to per-atom activations
        # For each atom l, get token activation a[tok_idx(l)]
        token_activations = ai[atom_to_token_idx]  # [N_atom, c_token]
        
        # Project to atom dimension and add skip connection
        ql = self.token_to_atom(token_activations) + q_skip  # [N_atom, c_atom]
        
        # Line 2: Apply cross attention transformer (sequence-local)
        # Uses skip connections c_skip and p_skip from encoder
        ql = self.atom_transformer(ql, c_skip, p_skip)  # [N_atom, c_atom]
        
        # Line 3: Map to position updates
        ql_norm = self.norm(ql)  # [N_atom, c_atom]
        r_update = self.to_position_update(ql_norm)  # [N_atom, 3]
        
        return r_update


def create_dummy_atom_features(
    n_atoms: int = 40,
    n_tokens: int = 10
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Create dummy atom features for testing AtomAttentionEncoder.
    
    Args:
        n_atoms: Number of atoms
        n_tokens: Number of tokens
        
    Returns:
        atom_features: Dict of dummy atom features
        atom_to_token_idx: [N_atom] mapping
    """
    atom_features = {
        'ref_pos': torch.randn(n_atoms, 3),
        'ref_charge': torch.randn(n_atoms),
        'ref_mask': torch.ones(n_atoms),
        'ref_element': torch.randn(n_atoms, 128),
        'ref_atom_name_chars': torch.randn(n_atoms, 4, 64),
        'ref_space_uid': torch.randint(0, n_tokens, (n_atoms,))
    }
    
    # Map atoms to tokens (e.g., 4 atoms per token)
    atom_to_token_idx = torch.arange(n_atoms) // (n_atoms // n_tokens)
    atom_to_token_idx = atom_to_token_idx.clamp(max=n_tokens - 1)
    
    return atom_features, atom_to_token_idx