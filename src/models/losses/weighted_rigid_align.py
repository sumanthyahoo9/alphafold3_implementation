"""
AlphaFold3 Weighted Rigid Alignment

File: src/models/losses/weighted_rigid_align.py

Implements Algorithm 28: Weighted Rigid Align

Computes optimal rigid transformation (rotation + translation) to align
predicted coordinates to ground truth using weighted Kabsch algorithm.

Key features:
- Weighted centroid calculation
- SVD-based optimal rotation (Kabsch algorithm)
- Handles reflections (ensures proper rotation)
- Stop gradient on output (for training stability)

Used in:
- Weighted MSE loss computation
- Structure alignment and RMSD calculation

Architecture:
    (x_pred, x_gt, weights) → Center → SVD → Rotation → Aligned coords
"""
from typing import Optional
import torch
import torch.nn as nn


def weighted_rigid_align(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Align predicted coordinates to ground truth via weighted rigid transformation.
    
    Implements Algorithm 28 from AF3 supplementary.
    
    Uses weighted Kabsch algorithm to find optimal rotation and translation.
    
    Args:
        x: Predicted atom positions [N_atoms, 3]
        x_gt: Ground truth atom positions [N_atoms, 3]
        weights: Per-atom weights [N_atoms] (default: all 1.0)
        eps: Small epsilon for numerical stability
    
    Returns:
        x_aligned: Ground truth aligned to prediction [N_atoms, 3]
                   (with stop_gradient applied)
    
    Algorithm 28:
    1: μ = mean(w * x) / mean(w)
    2: μ_GT = mean(w * x_GT) / mean(w)
    3: x = x - μ
    4: x_GT = x_GT - μ_GT
    5: U, V = svd(Σ w * x_GT ⊗ x)
    6: R = U @ V^T
    7: if det(R) < 0:  # Remove reflection
    8:     F = diag(1, 1, -1)
    9:     R = U @ F @ V^T
    10: end if
    11: x_align = R @ x_GT + μ
    12: return stop_gradient(x_align)
    """
    n_atoms = x.shape[0]
    device = x.device
    
    # Default weights if not provided
    if weights is None:
        weights = torch.ones(n_atoms, device=device)
    
    # Ensure weights are [N_atoms, 1] for broadcasting
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1)  # [N_atoms, 1]
    
    # Algorithm 28, line 1-2: Weighted mean centering
    # μ = Σ(w_l * x_l) / Σ(w_l)
    weight_sum = weights.sum() + eps
    
    mu = (weights * x).sum(dim=0, keepdim=True) / weight_sum  # [1, 3]
    mu_gt = (weights * x_gt).sum(dim=0, keepdim=True) / weight_sum  # [1, 3]
    
    # Algorithm 28, line 3-4: Center coordinates
    x_centered = x - mu  # [N_atoms, 3]
    x_gt_centered = x_gt - mu_gt  # [N_atoms, 3]
    
    # Algorithm 28, line 5: Find optimal rotation via SVD
    # Compute weighted covariance matrix: C = Σ w_l * x_l ⊗ x_GT_l
    # We want R such that: R @ x_gt_centered ≈ x_centered
    # So covariance H = x_centered^T @ x_gt_centered
    # Shape: [3, 3]
    
    # Apply weights to both coordinate sets
    x_weighted = x_centered * torch.sqrt(weights)  # [N_atoms, 3]
    x_gt_weighted = x_gt_centered * torch.sqrt(weights)  # [N_atoms, 3]
    
    # Covariance matrix: H = x^T @ x_GT (NOT x_GT^T @ x!)
    # H[i,j] = Σ_l w_l * x_l[i] * x_GT_l[j]
    H = x_weighted.T @ x_gt_weighted  # [3, 3]
    
    # SVD: H = U @ S @ V^T
    U, S, Vt = torch.linalg.svd(H)  # U: [3,3], S: [3], Vt: [3,3]
    V = Vt.T  # [3, 3]
    
    # Algorithm 28, line 6: Compute rotation
    # R = U @ V^T
    R = U @ V.T  # [3, 3]
    
    # Algorithm 28, line 7-10: Handle reflections
    # If det(R) < 0, we have a reflection, not a rotation
    # Fix by flipping the sign of the last column of V
    det_R = torch.det(R)
    
    if det_R < 0:
        # Flip last column of V to remove reflection
        F = torch.eye(3, device=device)
        F[2, 2] = -1.0  # diag(1, 1, -1)
        
        R = U @ F @ V.T  # [3, 3]
    
    # Algorithm 28, line 11: Apply alignment
    # Rotate GT around its center, then translate to prediction's center
    # x_align = R @ x_GT_centered + μ
    x_aligned = (R @ x_gt_centered.T).T + mu  # [N_atoms, 3]
    
    # Algorithm 28, line 12: Stop gradient
    # In training, we don't backprop through the alignment
    x_aligned = x_aligned.detach()
    
    return x_aligned


class WeightedRigidAlign(nn.Module):
    """
    Module wrapper for weighted rigid alignment.
    
    Useful for including in model pipelines.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        x: torch.Tensor,
        x_gt: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Align ground truth to prediction.
        
        Args:
            x: Predicted positions [N_atoms, 3]
            x_gt: Ground truth positions [N_atoms, 3]
            weights: Per-atom weights [N_atoms]
        
        Returns:
            x_aligned: Aligned ground truth [N_atoms, 3]
        """
        return weighted_rigid_align(x, x_gt, weights, self.eps)


def compute_aligned_rmsd(
    x: torch.Tensor,
    x_gt: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute RMSD after optimal alignment.
    
    Args:
        x: Predicted positions [N_atoms, 3]
        x_gt: Ground truth positions [N_atoms, 3]
        weights: Per-atom weights [N_atoms]
    
    Returns:
        rmsd: Root mean squared deviation (scalar)
    """
    # Align GT to prediction
    x_gt_aligned = weighted_rigid_align(x, x_gt, weights)
    
    # Compute RMSD
    if weights is None:
        weights = torch.ones(x.shape[0], device=x.device)
    
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1)
    
    # Weighted squared differences
    sq_diff = ((x - x_gt_aligned) ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
    weighted_sq_diff = (weights * sq_diff).sum()
    weight_sum = weights.sum()
    
    # RMSD
    rmsd = torch.sqrt(weighted_sq_diff / (weight_sum + 1e-8))
    
    return rmsd