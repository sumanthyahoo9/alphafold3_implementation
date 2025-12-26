"""
AlphaFold3 Smooth LDDT Loss

File: src/models/losses/smooth_lddt_loss.py

Implements Algorithm 27: Smooth LDDT Loss

Measures local structure quality by comparing predicted and ground truth
pairwise distances with smooth, differentiable thresholds.

Key features:
- Multiple distance thresholds (0.5Å, 1Å, 2Å, 4Å)
- Smooth sigmoids for differentiability
- Different inclusion radii for proteins (15Å) vs nucleotides (30Å)
- Evaluates local structure preservation

LDDT (Local Distance Difference Test):
- Standard metric in protein structure validation
- More robust than RMSD for local structure quality
- Used in CASP competitions

Architecture:
    (x_pred, x_gt) → Pairwise distances → Distance errors → Smooth score → LDDT
"""
from typing import Optional
import torch
import torch.nn as nn


class SmoothLDDTLoss(nn.Module):
    """
    Smooth LDDT loss for structure prediction.
    
    Implements Algorithm 27 from AF3 supplementary.
    
    Computes Local Distance Difference Test (LDDT) with smooth sigmoids
    for differentiability during training.
    
    Args:
        eps: Small epsilon for numerical stability (default: 1e-8)
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
        # Distance thresholds for LDDT computation (in Angstroms)
        self.register_buffer(
            'thresholds',
            torch.tensor([0.5, 1.0, 2.0, 4.0])
        )
        
        # Inclusion radii (in Angstroms)
        self.radius_protein = 15.0  # For protein atoms
        self.radius_nucleotide = 30.0  # For DNA/RNA atoms
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        is_dna: Optional[torch.Tensor] = None,
        is_rna: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Smooth LDDT loss.
        
        Args:
            x_pred: Predicted atom positions [N_atoms, 3]
            x_gt: Ground truth atom positions [N_atoms, 3]
            is_dna: DNA atom mask [N_atoms] (default: all False)
            is_rna: RNA atom mask [N_atoms] (default: all False)
            mask: Valid atom mask [N_atoms] (default: all True)
        
        Returns:
            loss: Scalar loss value (1 - LDDT)
        
        Algorithm 27:
        1: δx_lm = ||x_pred_l - x_pred_m||
        2: δx_GT_lm = ||x_gt_l - x_gt_m||
        3: δ_lm = |δx_GT_lm - δx_lm|
        4: ε_lm = (1/4) Σ sigmoid(threshold - δ_lm)
        5: is_nucleotide = is_dna + is_rna
        6: c_lm = (δx_GT < radius) mask
        7: lddt = mean(c_lm * ε_lm) / mean(c_lm)
        8: return 1 - lddt
        """
        n_atoms = x_pred.shape[0]
        device = x_pred.device
        
        # Default masks if not provided
        if is_dna is None:
            is_dna = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        if is_rna is None:
            is_rna = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        if mask is None:
            mask = torch.ones(n_atoms, dtype=torch.bool, device=device)
        
        # Algorithm 27, line 1: Compute predicted pairwise distances
        # δx_lm = ||x_pred_l - x_pred_m||
        diff_pred = x_pred.unsqueeze(0) - x_pred.unsqueeze(1)  # [N, N, 3]
        dist_pred = torch.sqrt(
            (diff_pred ** 2).sum(dim=-1) + self.eps
        )  # [N, N]
        
        # Algorithm 27, line 2: Compute ground truth pairwise distances
        # δx_GT_lm = ||x_gt_l - x_gt_m||
        diff_gt = x_gt.unsqueeze(0) - x_gt.unsqueeze(1)  # [N, N, 3]
        dist_gt = torch.sqrt(
            (diff_gt ** 2).sum(dim=-1) + self.eps
        )  # [N, N]
        
        # Algorithm 27, line 3: Compute distance difference
        # δ_lm = |δx_GT_lm - δx_lm|
        dist_error = torch.abs(dist_gt - dist_pred)  # [N, N]
        
        # Algorithm 27, line 4: Compute smooth score with sigmoids
        # ε_lm = (1/4) * [sigmoid(0.5-δ) + sigmoid(1-δ) + sigmoid(2-δ) + sigmoid(4-δ)]
        # Using multiple thresholds: 0.5Å, 1Å, 2Å, 4Å
        scores = []
        for threshold in self.thresholds:
            # sigmoid(threshold - dist_error)
            # When dist_error << threshold → sigmoid → 1 (good)
            # When dist_error >> threshold → sigmoid → 0 (bad)
            score = torch.sigmoid(threshold - dist_error)
            scores.append(score)
        
        epsilon = torch.stack(scores, dim=-1).mean(dim=-1)  # [N, N]
        
        # Algorithm 27, line 5: Nucleotide mask
        # is_nucleotide = is_dna + is_rna
        is_nucleotide = is_dna.float() + is_rna.float()  # [N]
        is_nucleotide = is_nucleotide.clamp(0, 1)  # Ensure binary
        
        # Algorithm 27, line 6: Inclusion radius mask
        # c_lm = (δx_GT < 30Å) * is_nucleotide + (δx_GT < 15Å) * (1 - is_nucleotide)
        # For nucleotides: include if ground truth distance < 30Å
        # For proteins: include if ground truth distance < 15Å
        
        # Create pairwise nucleotide mask [N, N]
        is_nucleotide_pair = is_nucleotide.unsqueeze(0) + is_nucleotide.unsqueeze(1)
        is_nucleotide_pair = (is_nucleotide_pair > 0).float()  # At least one is nucleotide
        
        # Inclusion mask based on ground truth distances
        mask_nucleotide = (dist_gt < self.radius_nucleotide).float()
        mask_protein = (dist_gt < self.radius_protein).float()
        
        # Combine: use nucleotide radius if either atom is nucleotide
        inclusion_mask = (
            mask_nucleotide * is_nucleotide_pair +
            mask_protein * (1 - is_nucleotide_pair)
        )  # [N, N]
        
        # Apply valid atom mask (exclude invalid atoms)
        valid_pair_mask = mask.unsqueeze(0) * mask.unsqueeze(1)  # [N, N]
        inclusion_mask = inclusion_mask * valid_pair_mask
        
        # Exclude self-pairs (diagonal)
        eye_mask = 1 - torch.eye(n_atoms, device=device)
        inclusion_mask = inclusion_mask * eye_mask
        
        # Algorithm 27, line 7: Compute LDDT
        # lddt = mean(c_lm * ε_lm) / mean(c_lm)
        numerator = (inclusion_mask * epsilon).sum()
        denominator = inclusion_mask.sum()
        
        # Avoid division by zero
        if denominator > 0:
            lddt = numerator / (denominator + self.eps)
        else:
            # If no valid pairs, return 0 LDDT (worst score)
            lddt = torch.tensor(0.0, device=device)
        
        # Algorithm 27, line 8: Return loss (1 - LDDT)
        # LDDT is in [0, 1], higher is better
        # Loss is in [0, 1], lower is better
        loss = 1.0 - lddt
        
        return loss
    
    def compute_lddt_score(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        is_dna: Optional[torch.Tensor] = None,
        is_rna: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute LDDT score (not loss).
        
        Returns:
            lddt: LDDT score in [0, 1], higher is better
        """
        loss = self.forward(x_pred, x_gt, is_dna, is_rna, mask)
        lddt = 1.0 - loss
        return lddt


def smooth_lddt_loss(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    is_dna: Optional[torch.Tensor] = None,
    is_rna: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Functional interface for Smooth LDDT loss.
    
    Args:
        x_pred: Predicted positions [N_atoms, 3]
        x_gt: Ground truth positions [N_atoms, 3]
        is_dna: DNA atom mask [N_atoms]
        is_rna: RNA atom mask [N_atoms]
        mask: Valid atom mask [N_atoms]
    
    Returns:
        loss: Scalar loss (1 - LDDT)
    """
    loss_fn = SmoothLDDTLoss()
    return loss_fn(x_pred, x_gt, is_dna, is_rna, mask)