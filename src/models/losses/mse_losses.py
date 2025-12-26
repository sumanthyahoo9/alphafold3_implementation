"""
AlphaFold3 Weighted MSE and Bond Length Losses

File: src/models/losses/mse_losses.py

Implements:
- Weighted MSE Loss (Equation 3-4)
- Bond Length Loss (Equation 5)

Used in diffusion training to ensure accurate coordinate prediction
with emphasis on challenging molecules (nucleotides, ligands).

Key features:
- Molecule-specific upweighting (DNA/RNA: 5x, ligands: 10x)
- Rigid alignment before MSE computation
- Bond length preservation for bonded ligands

Architecture:
    (x_pred, x_gt, weights) → Align → Weighted MSE
    (bonds) → Bond length errors → MSE
"""
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
from src.models.losses.weighted_rigid_align import weighted_rigid_align


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss after rigid alignment.
    
    Implements Equations 3-4 from AF3 supplementary.
    
    Computes MSE between predicted and ground truth coordinates
    after optimal alignment, with upweighting for nucleotides and ligands.
    
    Args:
        alpha_dna: DNA upweighting factor (default: 5.0)
        alpha_rna: RNA upweighting factor (default: 5.0)
        alpha_ligand: Ligand upweighting factor (default: 10.0)
        eps: Small epsilon for numerical stability
    """
    
    def __init__(
        self,
        alpha_dna: float = 5.0,
        alpha_rna: float = 5.0,
        alpha_ligand: float = 10.0,
        eps: float = 1e-8
    ):
        super().__init__()
        self.alpha_dna = alpha_dna
        self.alpha_rna = alpha_rna
        self.alpha_ligand = alpha_ligand
        self.eps = eps
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        is_dna: Optional[torch.Tensor] = None,
        is_rna: Optional[torch.Tensor] = None,
        is_ligand: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss after alignment.
        
        Args:
            x_pred: Predicted atom positions [N_atoms, 3]
            x_gt: Ground truth atom positions [N_atoms, 3]
            is_dna: DNA atom mask [N_atoms] (default: all False)
            is_rna: RNA atom mask [N_atoms] (default: all False)
            is_ligand: Ligand atom mask [N_atoms] (default: all False)
            mask: Valid atom mask [N_atoms] (default: all True)
        
        Returns:
            loss: Scalar weighted MSE loss
        
        Equation 3-4:
        w_l = 1 + f^is_dna_l * α_dna + f^is_rna_l * α_rna + f^is_ligand_l * α_ligand
        L_MSE = (1/3) * mean_l(w_l * ||x_l - x_GT-aligned_l||^2)
        """
        n_atoms = x_pred.shape[0]
        device = x_pred.device
        
        # Default masks if not provided
        if is_dna is None:
            is_dna = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        if is_rna is None:
            is_rna = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        if is_ligand is None:
            is_ligand = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        if mask is None:
            mask = torch.ones(n_atoms, dtype=torch.bool, device=device)
        
        # Equation 4: Compute per-atom weights
        # w_l = 1 + f^is_dna_l * α_dna + f^is_rna_l * α_rna + f^is_ligand_l * α_ligand
        weights = (
            1.0 +
            is_dna.float() * self.alpha_dna +
            is_rna.float() * self.alpha_rna +
            is_ligand.float() * self.alpha_ligand
        )  # [N_atoms]
        
        # Apply mask to weights
        weights = weights * mask.float()
        
        # Equation 2: Align ground truth to prediction
        # x_GT-aligned = weighted_rigid_align(x_GT, x_pred, weights)
        x_gt_aligned = weighted_rigid_align(x_pred, x_gt, weights)
        
        # Equation 3: Compute weighted MSE
        # L_MSE = (1/3) * mean_l(w_l * ||x_l - x_GT-aligned_l||^2)
        
        # Squared distances: ||x_pred - x_gt_aligned||^2
        sq_dist = ((x_pred - x_gt_aligned) ** 2).sum(dim=-1)  # [N_atoms]
        
        # Weighted squared distances
        weighted_sq_dist = weights * sq_dist  # [N_atoms]
        
        # Mean over atoms (only valid atoms)
        weight_sum = weights.sum() + self.eps
        mean_weighted_sq_dist = weighted_sq_dist.sum() / weight_sum
        
        # Divide by 3 (as per Equation 3)
        loss = mean_weighted_sq_dist / 3.0
        
        return loss


class BondLengthLoss(nn.Module):
    """
    Bond length loss for bonded ligands.
    
    Implements Equation 5 from AF3 supplementary.
    
    Ensures bonds between bonded ligands and their parent chains
    have correct lengths during fine-tuning.
    
    Args:
        eps: Small epsilon for numerical stability
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        bonds: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """
        Compute bond length loss.
        
        Args:
            x_pred: Predicted atom positions [N_atoms, 3]
            x_gt: Ground truth atom positions [N_atoms, 3]
            bonds: List of (start_atom_idx, end_atom_idx) tuples
        
        Returns:
            loss: Scalar bond length loss
        
        Equation 5:
        L_bond = mean_{(l,m) in B} (||x_l - x_m|| - ||x_GT_l - x_GT_m||)^2
        """
        if len(bonds) == 0:
            # No bonds to compute loss for
            return torch.tensor(0.0, device=x_pred.device)
        
        # Convert bonds to tensor
        bonds_tensor = torch.tensor(bonds, dtype=torch.long, device=x_pred.device)
        
        # Extract atom pairs
        start_indices = bonds_tensor[:, 0]  # [N_bonds]
        end_indices = bonds_tensor[:, 1]    # [N_bonds]
        
        # Get atom positions for each bond
        x_pred_start = x_pred[start_indices]  # [N_bonds, 3]
        x_pred_end = x_pred[end_indices]      # [N_bonds, 3]
        
        x_gt_start = x_gt[start_indices]      # [N_bonds, 3]
        x_gt_end = x_gt[end_indices]          # [N_bonds, 3]
        
        # Compute bond lengths in prediction
        # ||x_l - x_m||
        pred_bond_lengths = torch.sqrt(
            ((x_pred_start - x_pred_end) ** 2).sum(dim=-1) + self.eps
        )  # [N_bonds]
        
        # Compute bond lengths in ground truth
        # ||x_GT_l - x_GT_m||
        gt_bond_lengths = torch.sqrt(
            ((x_gt_start - x_gt_end) ** 2).sum(dim=-1) + self.eps
        )  # [N_bonds]
        
        # Equation 5: Squared difference in bond lengths
        # (||x_l - x_m|| - ||x_GT_l - x_GT_m||)^2
        bond_length_errors = (pred_bond_lengths - gt_bond_lengths) ** 2
        
        # Mean over all bonds
        loss = bond_length_errors.mean()
        
        return loss


class DiffusionLoss(nn.Module):
    """
    Combined diffusion loss (Equation 6).
    
    Implements full diffusion training loss:
    L_diffusion = weight(t) * (L_MSE + α_bond * L_bond) + L_smooth_lddt
    
    where weight(t) = (t^2 + σ_data^2) / (t + σ_data)^2
    
    Args:
        alpha_bond: Bond loss weight (0 for training, 1 for fine-tuning)
        sigma_data: Data variance constant (default: 16.0)
        alpha_dna: DNA upweighting (default: 5.0)
        alpha_rna: RNA upweighting (default: 5.0)
        alpha_ligand: Ligand upweighting (default: 10.0)
    """
    
    def __init__(
        self,
        alpha_bond: float = 0.0,
        sigma_data: float = 16.0,
        alpha_dna: float = 5.0,
        alpha_rna: float = 5.0,
        alpha_ligand: float = 10.0
    ):
        super().__init__()
        
        self.alpha_bond = alpha_bond
        self.sigma_data = sigma_data
        
        self.mse_loss = WeightedMSELoss(
            alpha_dna=alpha_dna,
            alpha_rna=alpha_rna,
            alpha_ligand=alpha_ligand
        )
        
        self.bond_loss = BondLengthLoss()
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        noise_level: torch.Tensor,
        smooth_lddt_loss: torch.Tensor,
        is_dna: Optional[torch.Tensor] = None,
        is_rna: Optional[torch.Tensor] = None,
        is_ligand: Optional[torch.Tensor] = None,
        bonds: Optional[List[Tuple[int, int]]] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute full diffusion loss.
        
        Args:
            x_pred: Predicted denoised positions [N_atoms, 3]
            x_gt: Ground truth positions [N_atoms, 3]
            noise_level: Sampled noise level t (scalar)
            smooth_lddt_loss: Pre-computed smooth LDDT loss (scalar)
            is_dna: DNA atom mask [N_atoms]
            is_rna: RNA atom mask [N_atoms]
            is_ligand: Ligand atom mask [N_atoms]
            bonds: List of bonded atom pairs
            mask: Valid atom mask [N_atoms]
        
        Returns:
            loss: Total diffusion loss (scalar)
        
        Equation 6:
        L_diffusion = (t^2 + σ_data^2)/(t + σ_data)^2 * (L_MSE + α_bond * L_bond) + L_smooth_lddt
        """
        # Compute MSE loss
        mse = self.mse_loss(x_pred, x_gt, is_dna, is_rna, is_ligand, mask)
        
        # Compute bond loss if bonds provided
        if bonds is not None and len(bonds) > 0:
            bond = self.bond_loss(x_pred, x_gt, bonds)
        else:
            bond = torch.tensor(0.0, device=x_pred.device)
        
        # Equation 6: Time-dependent weighting
        # weight(t) = (t^2 + σ_data^2) / (t + σ_data)^2
        t = noise_level
        numerator = t ** 2 + self.sigma_data ** 2
        denominator = (t + self.sigma_data) ** 2
        time_weight = numerator / denominator
        
        # Combined weighted reconstruction loss
        reconstruction_loss = mse + self.alpha_bond * bond
        
        # Total loss
        loss = time_weight * reconstruction_loss + smooth_lddt_loss
        
        return loss


# Functional interfaces
def weighted_mse_loss(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    is_dna: Optional[torch.Tensor] = None,
    is_rna: Optional[torch.Tensor] = None,
    is_ligand: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    alpha_dna: float = 5.0,
    alpha_rna: float = 5.0,
    alpha_ligand: float = 10.0
) -> torch.Tensor:
    """Functional interface for weighted MSE loss."""
    loss_fn = WeightedMSELoss(alpha_dna, alpha_rna, alpha_ligand)
    return loss_fn(x_pred, x_gt, is_dna, is_rna, is_ligand, mask)


def bond_length_loss(
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    bonds: List[Tuple[int, int]]
) -> torch.Tensor:
    """Functional interface for bond length loss."""
    loss_fn = BondLengthLoss()
    return loss_fn(x_pred, x_gt, bonds)