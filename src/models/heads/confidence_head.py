"""
AlphaFold3 Confidence Head

File: src/models/heads/confidence_head.py

Implements Algorithm 31: Confidence Head

Predicts quality metrics for generated structures:
- pLDDT: Per-atom confidence scores
- PAE: Predicted aligned error between token pairs
- PDE: Predicted distance error
- Resolved: Whether atoms are experimentally resolved

Key features:
- Uses trunk representations (s, z) and predicted structure
- Small Pairformer for refinement (4 blocks)
- Binned predictions for all metrics

Architecture:
    (s, z, x_pred) → Distance embedding → Pairformer → Confidence outputs
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import math

from src.models.trunk.pairformer import PairformerStack


class ConfidenceHead(nn.Module):
    """
    Predict confidence metrics for structures.
    
    Implements Algorithm 31 from AF3 supplementary.
    
    Predicts per-atom confidence (pLDDT), pairwise errors (PAE, PDE),
    and experimental resolution.
    
    Args:
        c_single: Single representation dimension (default: 384)
        c_pair: Pair representation dimension (default: 128)
        n_blocks: Number of Pairformer blocks (default: 4)
        n_bins_plddt: Number of pLDDT bins (default: 50)
        n_bins_pae: Number of PAE bins (default: 64)
        n_bins_pde: Number of PDE bins (default: 64)
        n_bins_resolved: Number of resolved bins (default: 2)
        max_atoms_per_token: Maximum atoms per token (default: 64)
    """
    
    def __init__(
        self,
        c_single: int = 384,
        c_pair: int = 128,
        n_blocks: int = 4,
        n_bins_plddt: int = 50,
        n_bins_pae: int = 64,
        n_bins_pde: int = 64,
        n_bins_resolved: int = 2,
        max_atoms_per_token: int = 64
    ):
        super().__init__()
        
        self.c_single = c_single
        self.c_pair = c_pair
        self.n_bins_plddt = n_bins_plddt
        self.n_bins_pae = n_bins_pae
        self.n_bins_pde = n_bins_pde
        self.n_bins_resolved = n_bins_resolved
        self.max_atoms_per_token = max_atoms_per_token
        
        # Algorithm 31, line 1: Add s_inputs to pair
        self.s_inputs_proj_i = nn.Linear(c_single, c_pair, bias=False)
        self.s_inputs_proj_j = nn.Linear(c_single, c_pair, bias=False)
        
        # Algorithm 31, line 3: Distance bins for representative atoms
        # Bins from 3 3/8 Å to 21 3/8 Å
        self.register_buffer(
            'distance_bins',
            torch.linspace(3.375, 21.375, 64)  # 64 bins
        )
        
        # Project one-hot distances to pair representation
        self.distance_proj = nn.Linear(64, c_pair, bias=False)
        
        # Algorithm 31, line 4: Small Pairformer for refinement
        self.pairformer = PairformerStack(
            c_single=c_single,
            c_pair=c_pair,
            n_blocks=n_blocks
        )
        
        # Algorithm 31, line 5: PAE head (64 bins, 0-32Å)
        self.pae_proj = nn.Linear(c_pair, n_bins_pae, bias=False)
        
        # Algorithm 31, line 6: PDE head (64 bins, 0-32Å)
        self.pde_proj = nn.Linear(c_pair, n_bins_pde, bias=False)
        
        # Algorithm 31, line 7: pLDDT head (50 bins, 0-1)
        # Per-atom confidence
        self.plddt_proj = nn.Linear(
            c_single,
            max_atoms_per_token * n_bins_plddt,
            bias=False
        )
        
        # Algorithm 31, line 8: Resolved head (2 bins)
        # Per-atom experimental resolution
        self.resolved_proj = nn.Linear(
            c_single,
            max_atoms_per_token * n_bins_resolved,
            bias=False
        )
    
    def forward(
        self,
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        x_pred: torch.Tensor,
        atom_to_token_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict confidence metrics.
        
        Args:
            s_inputs: Input single representation [N_token, c_single]
            s: Final single representation [N_token, c_single]
            z: Final pair representation [N_token, N_token, c_pair]
            x_pred: Predicted atom positions [N_atoms, 3]
            atom_to_token_idx: Atom to token mapping [N_atoms]
        
        Returns:
            p_plddt: Per-atom pLDDT predictions [N_atoms, n_bins_plddt]
            p_pae: PAE predictions [N_token, N_token, n_bins_pae]
            p_pde: PDE predictions [N_token, N_token, n_bins_pde]
            p_resolved: Per-atom resolved predictions [N_atoms, n_bins_resolved]
        
        Algorithm 31:
        1: z += Linear(s_inputs_i) + Linear(s_inputs_j)
        2: d_ij = ||x_pred[rep(i)] - x_pred[rep(j)]||
        3: z += Linear(one_hot(d_ij, bins))
        4: s, z = PairformerStack(s, z, N_block=4)
        5: p_pae = softmax(Linear(z))
        6: p_pde = softmax(Linear(z + z^T))
        7: p_plddt = softmax(Linear(s)[token_atom_idx])
        8: p_resolved = softmax(Linear(s)[token_atom_idx])
        9: return p_plddt, p_pae, p_pde, p_resolved
        """
        n_tokens = s.shape[0]
        n_atoms = x_pred.shape[0]
        
        # Stop gradients as per paper
        s = s.detach()
        z = z.detach()
        x_pred = x_pred.detach()
        
        # Algorithm 31, line 1: Add s_inputs to pair
        # z_ij += Linear(s_inputs_i) + Linear(s_inputs_j)
        s_proj_i = self.s_inputs_proj_i(s_inputs).unsqueeze(1)  # [N_token, 1, c_pair]
        s_proj_j = self.s_inputs_proj_j(s_inputs).unsqueeze(0)  # [1, N_token, c_pair]
        z = z + s_proj_i + s_proj_j  # [N_token, N_token, c_pair]
        
        # Algorithm 31, line 2-3: Embed pairwise distances
        # Get representative atom per token (first atom of each token)
        # For simplicity, use first atom of each token as representative
        rep_atom_indices = []
        for token_idx in range(n_tokens):
            # Find first atom belonging to this token
            atom_mask = (atom_to_token_idx == token_idx)
            if atom_mask.any():
                first_atom = atom_mask.nonzero()[0].item()
                rep_atom_indices.append(first_atom)
            else:
                # Fallback if no atoms for this token
                rep_atom_indices.append(0)
        
        rep_atom_indices = torch.tensor(rep_atom_indices, device=x_pred.device)
        rep_positions = x_pred[rep_atom_indices]  # [N_token, 3]
        
        # Compute pairwise distances
        # d_ij = ||x_pred[rep(i)] - x_pred[rep(j)]||
        diff = rep_positions.unsqueeze(0) - rep_positions.unsqueeze(1)  # [N_token, N_token, 3]
        distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [N_token, N_token]
        
        # One-hot encode distances
        # Bins from 3.375Å to 21.375Å
        distance_one_hot = self._one_hot_encode_distances(distances)  # [N_token, N_token, 64]
        
        # Project and add to pair representation
        z = z + self.distance_proj(distance_one_hot)
        
        # Algorithm 31, line 4: Run Pairformer
        s, z = self.pairformer(s, z)
        
        # Algorithm 31, line 5: PAE prediction
        # p_pae_ij = softmax(Linear(z_ij))
        p_pae_logits = self.pae_proj(z)  # [N_token, N_token, n_bins_pae]
        p_pae = torch.softmax(p_pae_logits, dim=-1)
        
        # Algorithm 31, line 6: PDE prediction
        # p_pde_ij = softmax(Linear(z_ij + z_ji))
        z_symmetric = z + z.transpose(0, 1)  # Symmetrize
        p_pde_logits = self.pde_proj(z_symmetric)  # [N_token, N_token, n_bins_pde]
        p_pde = torch.softmax(p_pde_logits, dim=-1)
        
        # Algorithm 31, line 7: pLDDT prediction (per-atom)
        # p_plddt_l = softmax(Linear[token_atom_idx(l)](s_i(l)))
        plddt_logits = self.plddt_proj(s)  # [N_token, max_atoms * n_bins_plddt]
        plddt_logits = plddt_logits.reshape(
            n_tokens, self.max_atoms_per_token, self.n_bins_plddt
        )  # [N_token, max_atoms, n_bins_plddt]
        
        # Map to atoms
        p_plddt_all = plddt_logits[atom_to_token_idx]  # [N_atoms, max_atoms, n_bins_plddt]
        # Take first slice for each atom (simplified - could use atom index within token)
        p_plddt = torch.softmax(p_plddt_all[:, 0, :], dim=-1)  # [N_atoms, n_bins_plddt]
        
        # Algorithm 31, line 8: Resolved prediction (per-atom)
        # p_resolved_l = softmax(Linear[token_atom_idx(l)](s_i(l)))
        resolved_logits = self.resolved_proj(s)  # [N_token, max_atoms * n_bins_resolved]
        resolved_logits = resolved_logits.reshape(
            n_tokens, self.max_atoms_per_token, self.n_bins_resolved
        )  # [N_token, max_atoms, n_bins_resolved]
        
        # Map to atoms
        p_resolved_all = resolved_logits[atom_to_token_idx]  # [N_atoms, max_atoms, n_bins_resolved]
        p_resolved = torch.softmax(p_resolved_all[:, 0, :], dim=-1)  # [N_atoms, n_bins_resolved]
        
        # Algorithm 31, line 9: Return all predictions
        return p_plddt, p_pae, p_pde, p_resolved
    
    def _one_hot_encode_distances(self, distances: torch.Tensor) -> torch.Tensor:
        """
        One-hot encode distances into bins.
        
        Args:
            distances: [N_token, N_token] pairwise distances
        
        Returns:
            one_hot: [N_token, N_token, 64] one-hot encoded
        """
        # Find nearest bin for each distance
        # distances: [N_token, N_token]
        # bins: [64]
        
        # Compute distances to each bin center
        dist_to_bins = torch.abs(
            distances.unsqueeze(-1) - self.distance_bins
        )  # [N_token, N_token, 64]
        
        # Find nearest bin
        nearest_bin = torch.argmin(dist_to_bins, dim=-1)  # [N_token, N_token]
        
        # One-hot encode
        one_hot = torch.nn.functional.one_hot(
            nearest_bin, num_classes=64
        ).float()  # [N_token, N_token, 64]
        
        return one_hot
    
    def get_plddt_scores(self, p_plddt: torch.Tensor) -> torch.Tensor:
        """
        Convert pLDDT probabilities to scores.
        
        Args:
            p_plddt: [N_atoms, n_bins_plddt] probabilities
        
        Returns:
            plddt: [N_atoms] scores in [0, 1]
        """
        # Bin centers from 0 to 1
        bin_centers = torch.linspace(
            0.0, 1.0, self.n_bins_plddt,
            device=p_plddt.device
        )
        
        # Expectation: sum over bins weighted by probabilities
        plddt = (p_plddt * bin_centers).sum(dim=-1)
        
        return plddt
    
    def get_pae_scores(self, p_pae: torch.Tensor) -> torch.Tensor:
        """
        Convert PAE probabilities to scores.
        
        Args:
            p_pae: [N_token, N_token, n_bins_pae] probabilities
        
        Returns:
            pae: [N_token, N_token] scores in Angstroms
        """
        # Bin centers from 0 to 32 Angstroms in 0.5 Angstrom increments
        bin_centers = torch.linspace(
            0.0, 32.0, self.n_bins_pae,
            device=p_pae.device
        )
        
        # Expectation
        pae = (p_pae * bin_centers).sum(dim=-1)
        
        return pae
    
    def get_pde_scores(self, p_pde: torch.Tensor) -> torch.Tensor:
        """
        Convert PDE probabilities to scores.
        
        Args:
            p_pde: [N_token, N_token, n_bins_pde] probabilities
        
        Returns:
            pde: [N_token, N_token] scores in Angstroms
        """
        # Bin centers from 0 to 32 Angstroms in 0.5 Angstrom increments
        bin_centers = torch.linspace(
            0.0, 32.0, self.n_bins_pde,
            device=p_pde.device
        )
        
        # Expectation
        pde = (p_pde * bin_centers).sum(dim=-1)
        
        return pde