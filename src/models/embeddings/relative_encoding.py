"""
AlphaFold3 Relative Position Encoding

Implements Algorithm 3: Relative position encoding

This module encodes relative positional information to break symmetries
across identical residues and chains. Critical for distinguishing between
chemically identical but spatially distinct tokens.

Key features:
- Relative residue positions (residue_index differences)
- Relative token positions (token_index differences within same residue)
- Relative chain positions (sym_id differences for different chains)
- Same entity mask (whether tokens are from same sequence)
"""

import torch
import torch.nn as nn
from typing import Dict


class RelativePositionEncoding(nn.Module):
    """
    Encodes relative positional information for token pairs.
    
    Implements Algorithm 3 from AF3 supplementary.
    
    Args:
        r_max: Maximum relative position (default: 32)
        s_max: Maximum relative chain position (default: 2)
        c_z: Pair representation channels (default: 128)
    """
    
    def __init__(
        self,
        r_max: int = 32,
        s_max: int = 2,
        c_z: int = 128
    ):
        super().__init__()
        
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        
        # Total one-hot dimensions:
        # rel_pos: 2*r_max + 2 = 66 (values 0-65)
        # rel_token: 2*r_max + 2 = 66 (values 0-65)
        # same_entity: 1 (binary)
        # rel_chain: 2*s_max + 2 = 6 (values 0-5)
        # Total: 66 + 66 + 1 + 6 = 139
        total_dims = (2 * r_max + 2) + (2 * r_max + 2) + 1 + (2 * s_max + 2)
        
        # Linear projection to pair representation
        self.linear = nn.Linear(total_dims, c_z, bias=False)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute relative position encodings.
        
        Args:
            features: Dict containing:
                - token_index: [N_token] token indices
                - residue_index: [N_token] residue indices
                - asym_id: [N_token] chain IDs
                - entity_id: [N_token] sequence IDs
                - sym_id: [N_token] symmetry IDs
        
        Returns:
            p_ij: [N_token, N_token, c_z] relative position encodings
        
        Algorithm 3:
        1: Compute same_chain, same_residue, same_entity masks
        2-5: Compute relative residue positions
        6-7: Compute relative token positions
        8-9: Compute relative chain positions
        10: Concatenate and project
        11: Return pair features
        """
        # Extract features
        token_index = features['token_index']  # [N_token]
        residue_index = features['residue_index']  # [N_token]
        asym_id = features['asym_id']  # [N_token]
        entity_id = features['entity_id']  # [N_token]
        sym_id = features['sym_id']  # [N_token]
        
        n_tokens = token_index.shape[0]
        device = token_index.device
        
        # Line 1: Compute binary masks
        # same_chain[i,j] = (asym_id[i] == asym_id[j])
        asym_id_i = asym_id.unsqueeze(1)  # [N_token, 1]
        asym_id_j = asym_id.unsqueeze(0)  # [1, N_token]
        same_chain = (asym_id_i == asym_id_j)  # [N_token, N_token]
        
        # Line 2: same_residue[i,j] = (residue_index[i] == residue_index[j])
        residue_index_i = residue_index.unsqueeze(1)
        residue_index_j = residue_index.unsqueeze(0)
        same_residue = (residue_index_i == residue_index_j)
        
        # Line 3: same_entity[i,j] = (entity_id[i] == entity_id[j])
        entity_id_i = entity_id.unsqueeze(1)
        entity_id_j = entity_id.unsqueeze(0)
        same_entity = (entity_id_i == entity_id_j).float()  # [N_token, N_token]
        
        # Lines 4-5: Relative residue positions
        residue_diff = residue_index_i - residue_index_j  # [N_token, N_token]
        
        # d_residue = clip(diff + r_max, 0, 2*r_max) if same_chain else 2*r_max + 1
        d_residue = torch.where(
            same_chain,
            torch.clamp(residue_diff + self.r_max, 0, 2 * self.r_max),
            torch.full_like(residue_diff, 2 * self.r_max + 1)
        )  # [N_token, N_token]
        
        # One-hot encode: [N_token, N_token, 2*r_max + 2]
        a_rel_pos = self._one_hot(d_residue, num_classes=2 * self.r_max + 2)
        
        # Lines 6-7: Relative token positions
        token_diff = token_index.unsqueeze(1) - token_index.unsqueeze(0)
        
        # d_token = clip(diff + r_max, 0, 2*r_max) if same_chain AND same_residue else 2*r_max + 1
        d_token = torch.where(
            same_chain & same_residue,
            torch.clamp(token_diff + self.r_max, 0, 2 * self.r_max),
            torch.full_like(token_diff, 2 * self.r_max + 1)
        )
        
        # One-hot encode: [N_token, N_token, 2*r_max + 2]
        a_rel_token = self._one_hot(d_token, num_classes=2 * self.r_max + 2)
        
        # Lines 8-9: Relative chain positions
        sym_diff = sym_id.unsqueeze(1) - sym_id.unsqueeze(0)
        
        # d_chain = clip(diff + s_max, 0, 2*s_max) if NOT same_chain else 2*s_max + 1
        d_chain = torch.where(
            ~same_chain,
            torch.clamp(sym_diff + self.s_max, 0, 2 * self.s_max),
            torch.full_like(sym_diff, 2 * self.s_max + 1)
        )
        
        # One-hot encode: [N_token, N_token, 2*s_max + 2]
        a_rel_chain = self._one_hot(d_chain, num_classes=2 * self.s_max + 2)
        
        # Line 10: Concatenate all features
        # same_entity needs to be expanded to [N_token, N_token, 1]
        same_entity_expanded = same_entity.unsqueeze(-1)
        
        # Concatenate: [rel_pos, rel_token, same_entity, rel_chain]
        concat_features = torch.cat([
            a_rel_pos,              # [N_token, N_token, 66]
            a_rel_token,            # [N_token, N_token, 66]
            same_entity_expanded,   # [N_token, N_token, 1]
            a_rel_chain             # [N_token, N_token, 6]
        ], dim=-1)  # [N_token, N_token, 139]
        
        # Linear projection to pair representation
        p_ij = self.linear(concat_features)  # [N_token, N_token, c_z]
        
        return p_ij
    
    def _one_hot(self, indices: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        One-hot encode integer indices.
        
        Args:
            indices: [N_token, N_token] integer indices
            num_classes: Number of classes
        
        Returns:
            one_hot: [N_token, N_token, num_classes] one-hot vectors
        """
        # Create one-hot encoding
        one_hot = torch.zeros(
            indices.shape[0], indices.shape[1], num_classes,
            dtype=torch.float32,
            device=indices.device
        )
        
        # Scatter 1s at the appropriate positions
        one_hot.scatter_(-1, indices.unsqueeze(-1).long(), 1.0)
        
        return one_hot


def create_dummy_relative_features(
    n_tokens: int = 10,
    n_chains: int = 2
) -> Dict[str, torch.Tensor]:
    """
    Create dummy features for testing RelativePositionEncoding.
    
    Args:
        n_tokens: Number of tokens
        n_chains: Number of chains
        
    Returns:
        Dict of dummy features
    """
    tokens_per_chain = n_tokens // n_chains
    
    # Ensure all tensors have exactly n_tokens elements
    features = {
        'token_index': torch.arange(n_tokens),
        'residue_index': torch.arange(n_tokens) % tokens_per_chain,
        'asym_id': torch.arange(n_tokens) // tokens_per_chain,
        'entity_id': torch.arange(n_tokens) // tokens_per_chain,
        'sym_id': torch.arange(n_tokens) // tokens_per_chain,
    }
    
    return features