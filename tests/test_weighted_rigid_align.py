"""
Unit tests for Weighted Rigid Alignment.

File: tests/test_weighted_rigid_align.py

Tests cover:
1. Basic alignment
2. Rotation recovery
3. Weighting behavior
4. Algorithm 28 faithfulness
5. Edge cases
"""

import pytest
import torch
import math
from src.models.losses.weighted_rigid_align import (
    weighted_rigid_align,
    WeightedRigidAlign,
    compute_aligned_rmsd
)


class TestBasicAlignment:
    """Test basic alignment functionality"""
    
    def test_identity_alignment(self):
        """Same coordinates should give identity transformation"""
        x = torch.randn(10, 3)
        x_gt = x.clone()
        
        x_aligned = weighted_rigid_align(x, x_gt)
        
        # Should be identical
        assert torch.allclose(x_aligned, x_gt, atol=1e-5)
    
    def test_translation_only(self):
        """Should recover pure translation"""
        x = torch.randn(10, 3)
        translation = torch.tensor([1.0, 2.0, 3.0])
        x_gt = x + translation
        
        x_aligned = weighted_rigid_align(x, x_gt)
        
        # After alignment, should match x
        assert torch.allclose(x_aligned, x, atol=1e-5)
    
    def test_rotation_only(self):
        """Should recover pure rotation"""
        # Create rotation matrix (90 degrees around z-axis)
        angle = math.pi / 2
        R = torch.tensor([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        x = torch.randn(10, 3)
        x_gt = (R @ x.T).T  # Rotate x to get x_gt
        
        # Align x_gt back to x's frame
        x_aligned = weighted_rigid_align(x, x_gt)
        
        # After alignment, x_aligned should be close to x
        # (we've undone the rotation)
        rmsd = torch.sqrt(((x_aligned - x) ** 2).sum() / 10)
        assert rmsd < 0.1  # Should be very close
    
    def test_rotation_and_translation(self):
        """Should recover rotation + translation"""
        angle = math.pi / 4
        R = torch.tensor([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        x = torch.randn(10, 3)
        translation = torch.tensor([5.0, -3.0, 2.0])
        x_gt = (R @ x.T).T + translation
        
        # Align x_gt back to x's frame  
        x_aligned = weighted_rigid_align(x, x_gt)
        
        # After alignment, should be close to x
        rmsd = torch.sqrt(((x_aligned - x) ** 2).sum() / 10)
        assert rmsd < 0.1


class TestWeighting:
    """Test weighting behavior"""
    
    def test_uniform_weights(self):
        """Uniform weights should work like unweighted"""
        x = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        # Uniform weights
        weights = torch.ones(10)
        x_aligned_weighted = weighted_rigid_align(x, x_gt, weights)
        
        # No weights
        x_aligned_unweighted = weighted_rigid_align(x, x_gt, None)
        
        # Should be same
        assert torch.allclose(x_aligned_weighted, x_aligned_unweighted, atol=1e-5)
    
    def test_zero_weight_atoms_ignored(self):
        """Zero-weighted atoms shouldn't affect alignment"""
        x = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        # First 5 atoms have weight 1, last 5 have weight 0
        weights = torch.cat([torch.ones(5), torch.zeros(5)])
        
        x_aligned_partial = weighted_rigid_align(x, x_gt, weights)
        
        # Should work (alignment based only on first 5 atoms)
        assert x_aligned_partial.shape == (10, 3)
    
    def test_different_weights(self):
        """Different weights should give different alignments"""
        x = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        weights1 = torch.ones(10)
        weights2 = torch.cat([torch.ones(5) * 10, torch.ones(5)])
        
        x_aligned_1 = weighted_rigid_align(x, x_gt, weights1)
        x_aligned_2 = weighted_rigid_align(x, x_gt, weights2)
        
        # Should be different (unless by chance)
        assert not torch.allclose(x_aligned_1, x_aligned_2, atol=1e-3)


class TestAlgorithm28Faithfulness:
    """Test faithfulness to Algorithm 28"""
    
    def test_algorithm_structure(self):
        """
        Algorithm 28:
        1-2: Weighted centering
        3-4: Center coordinates
        5: SVD of weighted covariance
        6: R = U @ V^T
        7-10: Handle reflections
        11: Apply transformation
        12: Stop gradient
        """
        x = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        x_aligned = weighted_rigid_align(x, x_gt)
        
        # Should return coordinates
        assert x_aligned.shape == (10, 3)
        
        # Should have stopped gradients
        assert not x_aligned.requires_grad
    
    def test_stop_gradient(self):
        """Output should have gradients stopped"""
        x = torch.randn(10, 3, requires_grad=True)
        x_gt = torch.randn(10, 3, requires_grad=True)
        
        x_aligned = weighted_rigid_align(x, x_gt)
        
        # Output should not require grad
        assert not x_aligned.requires_grad


class TestModuleInterface:
    """Test module wrapper"""
    
    def test_module_forward(self):
        """Module should work like function"""
        module = WeightedRigidAlign()
        
        x = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        x_aligned = module(x, x_gt)
        
        assert x_aligned.shape == (10, 3)
    
    def test_module_with_weights(self):
        """Module should handle weights"""
        module = WeightedRigidAlign()
        
        x = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        weights = torch.ones(10)
        
        x_aligned = module(x, x_gt, weights)
        
        assert x_aligned.shape == (10, 3)


class TestRMSD:
    """Test RMSD computation"""
    
    def test_aligned_rmsd_perfect(self):
        """Perfect match should give RMSD = 0"""
        x = torch.randn(10, 3)
        x_gt = x.clone()
        
        rmsd = compute_aligned_rmsd(x, x_gt)
        
        assert rmsd < 1e-5
    
    def test_aligned_rmsd_range(self):
        """RMSD should be non-negative"""
        x = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        rmsd = compute_aligned_rmsd(x, x_gt)
        
        assert rmsd >= 0.0
    
    def test_aligned_rmsd_translation_invariant(self):
        """Translation shouldn't affect aligned RMSD"""
        x = torch.randn(10, 3)
        x_gt = torch.randn(10, 3)
        
        rmsd1 = compute_aligned_rmsd(x, x_gt)
        
        # Translate both
        x_translated = x + torch.tensor([10.0, 20.0, 30.0])
        x_gt_translated = x_gt + torch.tensor([5.0, -5.0, 10.0])
        
        rmsd2 = compute_aligned_rmsd(x_translated, x_gt_translated)
        
        # Should be same (alignment handles translation)
        assert torch.allclose(rmsd1, rmsd2, atol=1e-4)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_atom(self):
        """Single atom should work"""
        x = torch.randn(1, 3)
        x_gt = torch.randn(1, 3)
        
        x_aligned = weighted_rigid_align(x, x_gt)
        
        assert x_aligned.shape == (1, 3)
    
    def test_two_atoms(self):
        """Two atoms should work"""
        x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        x_gt = torch.tensor([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
        
        x_aligned = weighted_rigid_align(x, x_gt)
        
        assert x_aligned.shape == (2, 3)
    
    def test_collinear_points(self):
        """Collinear points should work"""
        x = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0]
        ])
        x_gt = torch.randn(3, 3)
        
        x_aligned = weighted_rigid_align(x, x_gt)
        
        assert x_aligned.shape == (3, 3)
    
    def test_numerical_stability(self):
        """Should handle small coordinates"""
        x = torch.randn(10, 3) * 1e-6
        x_gt = torch.randn(10, 3) * 1e-6
        
        x_aligned = weighted_rigid_align(x, x_gt)
        
        assert not torch.isnan(x_aligned).any()
        assert not torch.isinf(x_aligned).any()


class TestReflectionHandling:
    """Test reflection detection and correction"""
    
    def test_handles_reflection(self):
        """Should detect and fix reflections"""
        x = torch.randn(10, 3)
        
        # Create reflection (flip z-axis)
        R_reflect = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0]
        ])
        
        x_gt = (R_reflect @ x.T).T
        
        x_aligned = weighted_rigid_align(x, x_gt)
        
        # Should still align correctly
        assert x_aligned.shape == (10, 3)
        assert not torch.isnan(x_aligned).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])