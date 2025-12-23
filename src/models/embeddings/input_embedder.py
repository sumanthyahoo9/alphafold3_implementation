"""
AlphaFold3 Input Feature Embedder

Implements Algorithm 2: Construct an initial 1D embedding (single representation)

This module embeds token-level features (restype, profile, deletion_mean) and 
atom-level features (reference conformer) into a single representation s_i.

The embedding process:
1. Embed atoms → aggregated per-token representation via AtomAttentionEncoder
2. Concatenate with restype, profile, deletion_mean
3. Return single representation [N_token, c_token]
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class InputFeatureEmbedder(nn.Module):
    """
    Embeds input features into initial single representation.
    
    Implements Algorithm 2 from AF3 supplementary.
    
    Args:
        c_token: Single representation channels (default: 384)
        c_atom: Atom representation channels for AtomAttentionEncoder (default: 128)
        c_atompair: Atom pair representation channels (default: 16)
    """
    
    def __init__(
        self,
        c_token: int = 384,
        c_atom: int = 128,
        c_atompair: int = 16
    ):
        super().__init__()
        
        self.c_token = c_token
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        
        # AtomAttentionEncoder will be imported/implemented separately
        # For now, we'll use a placeholder
        # self.atom_attention_encoder = AtomAttentionEncoder(...)
        
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Embed input features into single representation.
        
        Args:
            features: Dict containing:
                - restype: [N_token, 32] one-hot residue types
                - profile: [N_token, 32] MSA profile
                - deletion_mean: [N_token] mean deletions
                - ref_pos: [N_atom, 3] atom positions
                - ref_charge: [N_atom] atom charges
                - ref_mask: [N_atom] atom mask
                - ref_element: [N_atom, 128] one-hot elements
                - ref_atom_name_chars: [N_atom, 4, 64] atom names
                - ref_space_uid: [N_atom] residue grouping
                - (atom_to_token_idx: [N_atom] mapping - will be computed)
        
        Returns:
            s_inputs: [N_token, c_token] single representation
        
        Algorithm 2:
        1: {ai}, _, _, _ = AtomAttentionEncoder({f*}, ∅, ∅, ∅, ...)
        2: si = concat(ai, f_restype_i, f_profile_i, f_deletion_mean_i)
        3: return {si}
        """
        # Extract features
        restype = features['restype']  # [N_token, 32]
        profile = features['profile']  # [N_token, 32]
        deletion_mean = features['deletion_mean']  # [N_token]
        
        n_tokens = restype.shape[0]
        
        # Step 1: Embed per-atom features via AtomAttentionEncoder
        # NOTE: This will call AtomAttentionEncoder which we'll implement next
        # For now, placeholder returns zeros
        # {ai} = AtomAttentionEncoder({f*}, ∅, ∅, ∅, ...)
        # ai shape: [N_token, c_token] after aggregation
        
        # PLACEHOLDER: In actual implementation, this calls AtomAttentionEncoder
        # which aggregates atom features to per-token representation
        ai = torch.zeros(
            n_tokens, self.c_token,
            dtype=restype.dtype,
            device=restype.device
        )
        
        # Step 2: Concatenate per-token features
        # si = concat(ai, f_restype_i, f_profile_i, f_deletion_mean_i)
        
        # Expand deletion_mean to [N_token, 1] for concatenation
        deletion_mean_expanded = deletion_mean.unsqueeze(-1)  # [N_token, 1]
        
        # Concatenate: ai [c_token] + restype [32] + profile [32] + deletion_mean [1]
        # Total: c_token + 32 + 32 + 1 = c_token + 65
        s_inputs = torch.cat([
            ai,                      # [N_token, c_token]
            restype,                 # [N_token, 32]
            profile,                 # [N_token, 32]
            deletion_mean_expanded   # [N_token, 1]
        ], dim=-1)  # [N_token, c_token + 65]
        
        # Note: In the actual AF3, the final dimension might be projected
        # back to c_token, but Algorithm 2 just shows concatenation
        
        return s_inputs
    
    def get_output_dim(self) -> int:
        """
        Get output dimension of concatenated features.
        
        Returns:
            c_token + 65 (65 = 32 + 32 + 1 for restype, profile, deletion_mean)
        """
        return self.c_token + 65


def create_dummy_input_features(
    n_tokens: int = 10,
    n_atoms: int = 40,
    c_token: int = 384
) -> Dict[str, torch.Tensor]:
    """
    Create dummy input features for testing InputFeatureEmbedder.
    
    Args:
        n_tokens: Number of tokens
        n_atoms: Number of atoms
        c_token: Token representation dimension
        
    Returns:
        Dict of dummy features matching AF3 Table 5
    """
    return {
        # Token features
        'restype': torch.randn(n_tokens, 32),
        'profile': torch.randn(n_tokens, 32),
        'deletion_mean': torch.randn(n_tokens),
        
        # Atom features (for AtomAttentionEncoder)
        'ref_pos': torch.randn(n_atoms, 3),
        'ref_charge': torch.randn(n_atoms),
        'ref_mask': torch.ones(n_atoms),
        'ref_element': torch.randn(n_atoms, 128),
        'ref_atom_name_chars': torch.randn(n_atoms, 4, 64),
        'ref_space_uid': torch.randint(0, n_tokens, (n_atoms,)),
    }