"""
Unit tests for Smooth LDDT Loss.

File: tests/test_smooth_lddt_loss.py

Tests cover:
1. Loss computation
2. Score ranges and properties
3. Algorithm 27 faithfulness
4. Different molecule types
5. Edge cases
"""

import pytest
import torch
from src.models.losses.smooth_lddt_loss import SmoothLDDTLoss, smooth_lddt_loss


class TestLossComputation:
    """Test basic loss computation"""
    
    def test_perfect_prediction(self):
        """Perfect prediction should give low loss"""
        loss_fn = SmoothLDDTLoss()
        
        x = torch.randn(10, 3)
        x_pred = x.clone()
        x_gt = x.clone()
        
        loss = loss_fn(x_pred, x_gt)
        
        # Perfect match → LDDT ~0.8 (due to sigmoid saturation) → loss ~0.2
        # This is expected: sigmoid(threshold) for small thresholds < 1.0
        assert loss < 0.3  # Low loss for perfect prediction
    
    def test_random_prediction(self):
        """Random prediction should give high loss"""
        loss_fn = SmoothLDDTLoss()
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3) * 10  # Very different
        
        loss = loss_fn(x_pred, x_gt)
        
        # Bad prediction → LDDT near 0 → loss near 1
        assert loss > 0.5
        assert loss <= 1.0
    
    def test_loss_range(self):
        """Loss should be in [0, 1]"""
        loss_fn = SmoothLDDTLoss()
        
        for _ in range(10):
            x_pred = torch.randn(15, 3)
            x_gt = torch.randn(15, 3)
            
            loss = loss_fn(x_pred, x_gt)
            
            assert loss >= 0.0
            assert loss <= 1.0
    
    def test_functional_interface(self):
        """Functional interface should work"""
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        loss = smooth_lddt_loss(x_pred, x_gt)
        
        assert loss.shape == ()  # Scalar
        assert loss >= 0.0
        assert loss <= 1.0


class TestMoleculeTypes:
    """Test different molecule types"""
    
    def test_protein_atoms(self):
        """Protein atoms use 15Å radius"""
        loss_fn = SmoothLDDTLoss()
        
        n_atoms = 20
        x_pred = torch.randn(n_atoms, 3)
        x_gt = torch.randn(n_atoms, 3)
        
        # No DNA/RNA flags → all protein
        loss = loss_fn(x_pred, x_gt)
        
        assert loss >= 0.0
        assert loss <= 1.0
    
    def test_dna_atoms(self):
        """DNA atoms use 30Å radius"""
        loss_fn = SmoothLDDTLoss()
        
        n_atoms = 20
        x_pred = torch.randn(n_atoms, 3)
        x_gt = torch.randn(n_atoms, 3)
        is_dna = torch.ones(n_atoms, dtype=torch.bool)
        
        loss = loss_fn(x_pred, x_gt, is_dna=is_dna)
        
        assert loss >= 0.0
        assert loss <= 1.0
    
    def test_rna_atoms(self):
        """RNA atoms use 30Å radius"""
        loss_fn = SmoothLDDTLoss()
        
        n_atoms = 20
        x_pred = torch.randn(n_atoms, 3)
        x_gt = torch.randn(n_atoms, 3)
        is_rna = torch.ones(n_atoms, dtype=torch.bool)
        
        loss = loss_fn(x_pred, x_gt, is_rna=is_rna)
        
        assert loss >= 0.0
        assert loss <= 1.0
    
    def test_mixed_atoms(self):
        """Mixed protein/DNA/RNA atoms"""
        loss_fn = SmoothLDDTLoss()
        
        n_atoms = 30
        x_pred = torch.randn(n_atoms, 3)
        x_gt = torch.randn(n_atoms, 3)
        
        # 10 protein, 10 DNA, 10 RNA
        is_dna = torch.zeros(n_atoms, dtype=torch.bool)
        is_dna[10:20] = True
        
        is_rna = torch.zeros(n_atoms, dtype=torch.bool)
        is_rna[20:30] = True
        
        loss = loss_fn(x_pred, x_gt, is_dna=is_dna, is_rna=is_rna)
        
        assert loss >= 0.0
        assert loss <= 1.0


class TestAlgorithm27Faithfulness:
    """Test faithfulness to Algorithm 27"""
    
    def test_algorithm_structure(self):
        """
        Algorithm 27:
        1: δx_lm = ||x_pred_l - x_pred_m||
        2: δx_GT_lm = ||x_gt_l - x_gt_m||
        3: δ_lm = |δx_GT_lm - δx_lm|
        4: ε_lm = (1/4) Σ sigmoid(threshold - δ_lm)
        5: is_nucleotide = is_dna + is_rna
        6: c_lm = inclusion mask
        7: lddt = mean(c_lm * ε_lm) / mean(c_lm)
        8: return 1 - lddt
        """
        loss_fn = SmoothLDDTLoss()
        
        # Check thresholds
        assert len(loss_fn.thresholds) == 4
        assert torch.allclose(
            loss_fn.thresholds,
            torch.tensor([0.5, 1.0, 2.0, 4.0])
        )
        
        # Check radii
        assert loss_fn.radius_protein == 15.0
        assert loss_fn.radius_nucleotide == 30.0
    
    def test_distance_thresholds(self):
        """Should use 0.5, 1, 2, 4 Angstrom thresholds"""
        loss_fn = SmoothLDDTLoss()
        
        expected = torch.tensor([0.5, 1.0, 2.0, 4.0])
        assert torch.allclose(loss_fn.thresholds, expected)
    
    def test_inclusion_radii(self):
        """Should use 15Å for protein, 30Å for nucleotides"""
        loss_fn = SmoothLDDTLoss()
        
        assert loss_fn.radius_protein == 15.0
        assert loss_fn.radius_nucleotide == 30.0


class TestScoreComputation:
    """Test LDDT score computation"""
    
    def test_compute_lddt_score(self):
        """Should compute LDDT score (not loss)"""
        loss_fn = SmoothLDDTLoss()
        
        x = torch.randn(10, 3)
        x_pred = x.clone()
        x_gt = x.clone()
        
        lddt = loss_fn.compute_lddt_score(x_pred, x_gt)
        
        # Perfect match → LDDT ~0.8 (due to sigmoid saturation at thresholds)
        assert lddt > 0.7  # High score
        assert lddt <= 1.0
    
    def test_score_range(self):
        """LDDT score should be in [0, 1]"""
        loss_fn = SmoothLDDTLoss()
        
        for _ in range(10):
            x_pred = torch.randn(15, 3)
            x_gt = torch.randn(15, 3)
            
            lddt = loss_fn.compute_lddt_score(x_pred, x_gt)
            
            assert lddt >= 0.0
            assert lddt <= 1.0
    
    def test_loss_score_relationship(self):
        """Loss = 1 - LDDT score"""
        loss_fn = SmoothLDDTLoss()
        
        x_pred = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        loss = loss_fn(x_pred, x_gt)
        lddt = loss_fn.compute_lddt_score(x_pred, x_gt)
        
        assert torch.allclose(loss, 1.0 - lddt, atol=1e-6)


class TestMasking:
    """Test atom masking"""
    
    def test_with_mask(self):
        """Should respect atom mask"""
        loss_fn = SmoothLDDTLoss()
        
        n_atoms = 20
        x_pred = torch.randn(n_atoms, 3)
        x_gt = torch.randn(n_atoms, 3)
        
        # Mask out half the atoms
        mask = torch.ones(n_atoms, dtype=torch.bool)
        mask[10:] = False
        
        loss = loss_fn(x_pred, x_gt, mask=mask)
        
        assert loss >= 0.0
        assert loss <= 1.0
    
    def test_all_masked(self):
        """All masked should handle gracefully"""
        loss_fn = SmoothLDDTLoss()
        
        n_atoms = 10
        x_pred = torch.randn(n_atoms, 3)
        x_gt = torch.randn(n_atoms, 3)
        
        # Mask out everything
        mask = torch.zeros(n_atoms, dtype=torch.bool)
        
        loss = loss_fn(x_pred, x_gt, mask=mask)
        
        # No valid atoms → worst loss
        assert loss == 1.0


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_atom(self):
        """Single atom should handle gracefully"""
        loss_fn = SmoothLDDTLoss()
        
        x_pred = torch.randn(1, 3)
        x_gt = torch.randn(1, 3)
        
        loss = loss_fn(x_pred, x_gt)
        
        # Only one atom → no pairs → worst loss
        assert loss == 1.0
    
    def test_two_atoms(self):
        """Two atoms should work"""
        loss_fn = SmoothLDDTLoss()
        
        x_pred = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        x_gt = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        
        loss = loss_fn(x_pred, x_gt)
        
        # Perfect match but sigmoid saturation gives ~0.8 LDDT
        assert loss < 0.3
    
    def test_identical_positions(self):
        """All atoms at same position"""
        loss_fn = SmoothLDDTLoss()
        
        x_pred = torch.zeros(10, 3)
        x_gt = torch.zeros(10, 3)
        
        loss = loss_fn(x_pred, x_gt)
        
        # All distances are 0 → perfect (but sigmoid saturation)
        assert loss < 0.3
    
    def test_gradient_flow(self):
        """Loss should allow gradient flow"""
        loss_fn = SmoothLDDTLoss()
        
        x_pred = torch.randn(10, 3, requires_grad=True)
        x_gt = torch.randn(10, 3)
        
        loss = loss_fn(x_pred, x_gt)
        loss.backward()
        
        assert x_pred.grad is not None
        assert not torch.isnan(x_pred.grad).any()


class TestDifferentiability:
    """Test smooth/differentiable properties"""
    
    def test_sigmoid_smoothness(self):
        """Using sigmoids should make loss smooth"""
        loss_fn = SmoothLDDTLoss()
        
        # Create similar positions
        x_gt = torch.randn(5, 3)
        
        # Gradually move prediction away
        losses = []
        for scale in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]:
            x_pred = x_gt + torch.randn(5, 3) * scale
            loss = loss_fn(x_pred, x_gt)
            losses.append(loss.item())
        
        # Loss should increase monotonically (roughly)
        # But smoothly, not in discrete jumps
        assert losses[-1] > losses[0]  # Worse when farther


if __name__ == "__main__":
    pytest.main([__file__, "-v"])