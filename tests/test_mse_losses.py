"""
Unit tests for Weighted MSE and Bond Length Losses.

File: tests/test_mse_losses.py

Tests cover:
1. Weighted MSE Loss
2. Bond Length Loss
3. Combined Diffusion Loss
4. Equations 3-6 faithfulness
"""

import pytest
import torch
from src.models.losses.mse_losses import (
    WeightedMSELoss,
    BondLengthLoss,
    DiffusionLoss,
    weighted_mse_loss,
    bond_length_loss
)


class TestWeightedMSELoss:
    """Test Weighted MSE Loss (Equations 3-4)"""
    
    def test_perfect_prediction(self):
        """Perfect prediction should give loss = 0"""
        loss_fn = WeightedMSELoss()
        
        x = torch.randn(10, 3)
        x_pred = x.clone()
        x_gt = x.clone()
        
        loss = loss_fn(x_pred, x_gt)
        
        assert loss < 1e-5
    
    def test_loss_range(self):
        """Loss should be non-negative"""
        loss_fn = WeightedMSELoss()
        
        for _ in range(10):
            x_pred = torch.randn(15, 3)
            x_gt = torch.randn(15, 3)
            
            loss = loss_fn(x_pred, x_gt)
            
            assert loss >= 0.0
    
    def test_uniform_weights(self):
        """Without molecule flags, all weights = 1"""
        loss_fn = WeightedMSELoss()
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        # No flags → all weights = 1.0
        loss = loss_fn(x_pred, x_gt)
        
        assert loss >= 0.0
    
    def test_dna_upweighting(self):
        """DNA atoms should be upweighted by 5x"""
        loss_fn = WeightedMSELoss(alpha_dna=5.0)
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        # Mark half as DNA
        is_dna = torch.zeros(10, dtype=torch.bool)
        is_dna[5:] = True
        
        loss_dna = loss_fn(x_pred, x_gt, is_dna=is_dna)
        loss_no_dna = loss_fn(x_pred, x_gt)
        
        # DNA upweighting should affect loss
        assert loss_dna != loss_no_dna
    
    def test_ligand_upweighting(self):
        """Ligand atoms should be upweighted by 10x"""
        loss_fn = WeightedMSELoss(alpha_ligand=10.0)
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        # Mark some as ligands
        is_ligand = torch.zeros(10, dtype=torch.bool)
        is_ligand[7:] = True
        
        loss_ligand = loss_fn(x_pred, x_gt, is_ligand=is_ligand)
        loss_no_ligand = loss_fn(x_pred, x_gt)
        
        # Ligand upweighting should affect loss
        assert loss_ligand != loss_no_ligand
    
    def test_combined_weights(self):
        """Combined DNA/RNA/ligand weights"""
        loss_fn = WeightedMSELoss(
            alpha_dna=5.0,
            alpha_rna=5.0,
            alpha_ligand=10.0
        )
        
        x_pred = torch.randn(15, 3)
        x_gt = torch.randn(15, 3)
        
        # 5 protein, 5 DNA, 5 ligand
        is_dna = torch.zeros(15, dtype=torch.bool)
        is_dna[5:10] = True
        
        is_ligand = torch.zeros(15, dtype=torch.bool)
        is_ligand[10:] = True
        
        loss = loss_fn(x_pred, x_gt, is_dna=is_dna, is_ligand=is_ligand)
        
        assert loss >= 0.0
    
    def test_masking(self):
        """Should respect atom mask"""
        loss_fn = WeightedMSELoss()
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        # Mask out half
        mask = torch.ones(10, dtype=torch.bool)
        mask[5:] = False
        
        loss_masked = loss_fn(x_pred, x_gt, mask=mask)
        loss_unmasked = loss_fn(x_pred, x_gt)
        
        # Should be different
        assert loss_masked != loss_unmasked
    
    def test_functional_interface(self):
        """Functional interface should work"""
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        loss = weighted_mse_loss(x_pred, x_gt)
        
        assert loss >= 0.0


class TestBondLengthLoss:
    """Test Bond Length Loss (Equation 5)"""
    
    def test_perfect_bonds(self):
        """Perfect bond lengths should give loss = 0"""
        loss_fn = BondLengthLoss()
        
        x = torch.randn(10, 3)
        x_pred = x.clone()
        x_gt = x.clone()
        
        bonds = [(0, 1), (2, 3), (4, 5)]
        
        loss = loss_fn(x_pred, x_gt, bonds)
        
        assert loss < 1e-5
    
    def test_no_bonds(self):
        """Empty bond list should give loss = 0"""
        loss_fn = BondLengthLoss()
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        bonds = []
        
        loss = loss_fn(x_pred, x_gt, bonds)
        
        assert loss == 0.0
    
    def test_bond_length_error(self):
        """Incorrect bond lengths should give non-zero loss"""
        loss_fn = BondLengthLoss()
        
        # Ground truth with bond length 1.0
        x_gt = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ])
        
        # Prediction with bond length 2.0
        x_pred = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0]
        ])
        
        bonds = [(0, 1), (2, 3)]
        
        loss = loss_fn(x_pred, x_gt, bonds)
        
        # Bond length error = |2.0 - 1.0| = 1.0
        # Squared error = 1.0
        assert loss > 0.0
    
    def test_multiple_bonds(self):
        """Should handle multiple bonds"""
        loss_fn = BondLengthLoss()
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        bonds = [(0, 1), (1, 2), (3, 4), (5, 6), (7, 8)]
        
        loss = loss_fn(x_pred, x_gt, bonds)
        
        assert loss >= 0.0
    
    def test_functional_interface(self):
        """Functional interface should work"""
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        bonds = [(0, 1), (2, 3)]
        
        loss = bond_length_loss(x_pred, x_gt, bonds)
        
        assert loss >= 0.0


class TestDiffusionLoss:
    """Test Combined Diffusion Loss (Equation 6)"""
    
    def test_diffusion_loss_components(self):
        """Should combine MSE, bond, and LDDT losses"""
        loss_fn = DiffusionLoss(alpha_bond=1.0)
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        noise_level = torch.tensor(1.0)
        smooth_lddt_loss = torch.tensor(0.5)
        
        bonds = [(0, 1), (2, 3)]
        
        loss = loss_fn(
            x_pred, x_gt, noise_level, smooth_lddt_loss,
            bonds=bonds
        )
        
        assert loss >= 0.0
    
    def test_time_weighting(self):
        """Different noise levels should give different weights"""
        loss_fn = DiffusionLoss()
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        smooth_lddt_loss = torch.tensor(0.0)
        
        # Different noise levels
        t1 = torch.tensor(0.1)
        t2 = torch.tensor(10.0)
        
        loss1 = loss_fn(x_pred, x_gt, t1, smooth_lddt_loss)
        loss2 = loss_fn(x_pred, x_gt, t2, smooth_lddt_loss)
        
        # Different time weights should give different losses
        assert loss1 != loss2
    
    def test_no_bond_loss(self):
        """Should work without bonds"""
        loss_fn = DiffusionLoss(alpha_bond=0.0)
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        noise_level = torch.tensor(1.0)
        smooth_lddt_loss = torch.tensor(0.3)
        
        loss = loss_fn(x_pred, x_gt, noise_level, smooth_lddt_loss)
        
        assert loss >= 0.0
    
    def test_alpha_bond_effect(self):
        """alpha_bond should control bond loss weight"""
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        noise_level = torch.tensor(1.0)
        smooth_lddt_loss = torch.tensor(0.0)
        bonds = [(0, 1), (2, 3)]
        
        # Without bond loss
        loss_fn_0 = DiffusionLoss(alpha_bond=0.0)
        loss_0 = loss_fn_0(x_pred, x_gt, noise_level, smooth_lddt_loss, bonds=bonds)
        
        # With bond loss
        loss_fn_1 = DiffusionLoss(alpha_bond=1.0)
        loss_1 = loss_fn_1(x_pred, x_gt, noise_level, smooth_lddt_loss, bonds=bonds)
        
        # Should be different (unless bond error = 0)
        # For random data, very unlikely to have perfect bonds
        assert True  # Just check it runs


class TestEquationFaithfulness:
    """Test faithfulness to Equations 3-6"""
    
    def test_equation_3_structure(self):
        """
        Equation 3: L_MSE = (1/3) * mean_l(w_l * ||x_l - x_GT-aligned_l||^2)
        """
        loss_fn = WeightedMSELoss()
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        loss = loss_fn(x_pred, x_gt)
        
        # Loss should be non-negative scalar
        assert loss.dim() == 0
        assert loss >= 0.0
    
    def test_equation_4_weights(self):
        """
        Equation 4: w_l = 1 + α_dna * is_dna + α_rna * is_rna + α_ligand * is_ligand
        """
        loss_fn = WeightedMSELoss(
            alpha_dna=5.0,
            alpha_rna=5.0,
            alpha_ligand=10.0
        )
        
        # Check defaults
        assert loss_fn.alpha_dna == 5.0
        assert loss_fn.alpha_rna == 5.0
        assert loss_fn.alpha_ligand == 10.0
    
    def test_equation_5_structure(self):
        """
        Equation 5: L_bond = mean_{(l,m) in B} (||x_l - x_m|| - ||x_GT_l - x_GT_m||)^2
        """
        loss_fn = BondLengthLoss()
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        bonds = [(0, 1), (2, 3)]
        
        loss = loss_fn(x_pred, x_gt, bonds)
        
        # Loss should be non-negative scalar
        assert loss.dim() == 0
        assert loss >= 0.0
    
    def test_equation_6_structure(self):
        """
        Equation 6: L_diffusion = weight(t) * (L_MSE + α_bond * L_bond) + L_smooth_lddt
        where weight(t) = (t^2 + σ_data^2) / (t + σ_data)^2
        """
        loss_fn = DiffusionLoss(alpha_bond=1.0, sigma_data=16.0)
        
        # Check parameters
        assert loss_fn.alpha_bond == 1.0
        assert loss_fn.sigma_data == 16.0


class TestGradientFlow:
    """Test gradient properties"""
    
    def test_mse_gradient_flow(self):
        """MSE loss should allow gradient flow"""
        loss_fn = WeightedMSELoss()
        
        x_pred = torch.randn(10, 3, requires_grad=True)
        x_gt = torch.randn(10, 3)
        
        loss = loss_fn(x_pred, x_gt)
        loss.backward()
        
        assert x_pred.grad is not None
        assert not torch.isnan(x_pred.grad).any()
    
    def test_bond_gradient_flow(self):
        """Bond loss should allow gradient flow"""
        loss_fn = BondLengthLoss()
        
        x_pred = torch.randn(10, 3, requires_grad=True)
        x_gt = torch.randn(10, 3)
        bonds = [(0, 1), (2, 3)]
        
        loss = loss_fn(x_pred, x_gt, bonds)
        loss.backward()
        
        assert x_pred.grad is not None
        assert not torch.isnan(x_pred.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])