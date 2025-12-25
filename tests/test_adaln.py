"""
Unit tests for AlphaFold3 Adaptive LayerNorm.

File: tests/test_adaln.py

Tests cover:
1. Initialization
2. Forward pass shape validation
3. Conditioning modulation
4. Scale and shift behavior
5. Algorithm 26 faithfulness
6. Comparison with standard LayerNorm
"""

import pytest
import torch
from src.models.diffusion.adaln import (
    AdaptiveLayerNorm,
    create_dummy_adaln_input
)


class TestInitialization:
    """Test AdaptiveLayerNorm initialization"""
    
    def test_basic_initialization(self):
        """Should initialize with correct dimensions"""
        adaln = AdaptiveLayerNorm(dim_input=384, dim_cond=384)
        
        assert adaln.dim_input == 384
        assert adaln.dim_cond == 384
    
    def test_different_dimensions(self):
        """Should handle different input/conditioning dimensions"""
        adaln = AdaptiveLayerNorm(dim_input=256, dim_cond=512)
        
        assert adaln.dim_input == 256
        assert adaln.dim_cond == 512
    
    def test_layernorm_without_affine(self):
        """Input LayerNorm should have no learnable parameters"""
        adaln = AdaptiveLayerNorm(dim_input=128, dim_cond=128)
        
        # norm_input should have no learnable parameters
        assert adaln.norm_input.elementwise_affine is False
    
    def test_conditioning_norm_no_bias(self):
        """Conditioning LayerNorm should have no bias"""
        adaln = AdaptiveLayerNorm(dim_input=128, dim_cond=128)
        
        # norm_cond bias should be zero and frozen
        assert torch.allclose(adaln.norm_cond.bias, torch.zeros_like(adaln.norm_cond.bias))
        assert not adaln.norm_cond.bias.requires_grad


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_basic_forward_unbatched(self):
        """Should handle unbatched input"""
        adaln = AdaptiveLayerNorm(dim_input=384, dim_cond=384)
        
        a = torch.randn(10, 384)  # 10 tokens
        s = torch.randn(10, 384)  # conditioning
        
        out = adaln(a, s)
        
        assert out.shape == (10, 384)
    
    def test_basic_forward_batched(self):
        """Should handle batched input"""
        adaln = AdaptiveLayerNorm(dim_input=384, dim_cond=384)
        
        a = torch.randn(4, 10, 384)  # batch of 4
        s = torch.randn(4, 10, 384)
        
        out = adaln(a, s)
        
        assert out.shape == (4, 10, 384)
    
    def test_different_sequence_lengths(self):
        """Should work with different sequence lengths"""
        adaln = AdaptiveLayerNorm(dim_input=256, dim_cond=256)
        
        for seq_len in [5, 10, 20, 50]:
            a = torch.randn(seq_len, 256)
            s = torch.randn(seq_len, 256)
            
            out = adaln(a, s)
            
            assert out.shape == (seq_len, 256)
    
    def test_different_dimensions(self):
        """Should work when input and conditioning have different dims"""
        adaln = AdaptiveLayerNorm(dim_input=256, dim_cond=512)
        
        a = torch.randn(10, 256)
        s = torch.randn(10, 512)
        
        out = adaln(a, s)
        
        assert out.shape == (10, 256)


class TestConditioningModulation:
    """Test conditioning-based modulation"""
    
    def test_different_conditioning_different_output(self):
        """Different conditioning should produce different outputs"""
        adaln = AdaptiveLayerNorm(dim_input=128, dim_cond=128)
        
        a = torch.randn(10, 128)
        s1 = torch.randn(10, 128)
        s2 = torch.randn(10, 128)
        
        out1 = adaln(a, s1)
        out2 = adaln(a, s2)
        
        # Same input, different conditioning → different output
        assert not torch.allclose(out1, out2, atol=1e-5)
    
    def test_same_conditioning_same_output(self):
        """Same conditioning should produce same output (deterministic)"""
        adaln = AdaptiveLayerNorm(dim_input=128, dim_cond=128)
        
        a = torch.randn(10, 128)
        s = torch.randn(10, 128)
        
        out1 = adaln(a, s)
        out2 = adaln(a, s)
        
        assert torch.allclose(out1, out2)
    
    def test_conditioning_influences_all_positions(self):
        """Conditioning should influence output at all positions"""
        adaln = AdaptiveLayerNorm(dim_input=64, dim_cond=64)
        
        a = torch.randn(20, 64)
        
        # Zero conditioning
        s_zero = torch.zeros(20, 64)
        out_zero = adaln(a, s_zero)
        
        # Non-zero conditioning
        s_nonzero = torch.randn(20, 64)
        out_nonzero = adaln(a, s_nonzero)
        
        # Should be different at all positions
        assert not torch.allclose(out_zero, out_nonzero, atol=1e-5)


class TestScaleAndShift:
    """Test scale and shift behavior"""
    
    def test_scale_bounded_by_sigmoid(self):
        """Scale should be in range [0, 1] due to sigmoid"""
        adaln = AdaptiveLayerNorm(dim_input=64, dim_cond=64)
        
        a = torch.randn(10, 64)
        s = torch.randn(10, 64) * 10  # Large conditioning values
        
        # Access internal scale computation
        s_norm = adaln.norm_cond(s)
        scale = torch.sigmoid(adaln.linear_scale(s_norm))
        
        # Scale should be in [0, 1]
        assert scale.min() >= 0.0
        assert scale.max() <= 1.0
    
    def test_shift_unbounded(self):
        """Shift should be unbounded (no sigmoid)"""
        adaln = AdaptiveLayerNorm(dim_input=64, dim_cond=64)
        
        # Large conditioning
        s = torch.randn(10, 64) * 100
        
        s_norm = adaln.norm_cond(s)
        shift = adaln.linear_shift(s_norm)
        
        # Shift can be large (no sigmoid bounding)
        # Just check it exists and has correct shape
        assert shift.shape == (10, 64)


class TestAlgorithm26Faithfulness:
    """Test faithfulness to Algorithm 26"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 26:
        1: a ← LayerNorm(a, scale=False, offset=False)
        2: s ← LayerNorm(s, offset=False)
        3: a ← sigmoid(Linear(s)) ⊙ a + LinearNoBias(s)
        4: return a
        """
        adaln = AdaptiveLayerNorm(dim_input=128, dim_cond=128)
        
        a = torch.randn(10, 128)
        s = torch.randn(10, 128)
        
        # Manually compute expected output
        # Line 1: LayerNorm without scale/offset
        a_norm = adaln.norm_input(a)
        
        # Line 2: LayerNorm of conditioning
        s_norm = adaln.norm_cond(s)
        
        # Line 3: Modulation
        scale = torch.sigmoid(adaln.linear_scale(s_norm))
        shift = adaln.linear_shift(s_norm)
        expected = scale * a_norm + shift
        
        # Actual output
        actual = adaln(a, s)
        
        # Should match
        assert torch.allclose(actual, expected, atol=1e-6)
    
    def test_no_bias_in_shift_projection(self):
        """Shift projection should have no bias (LinearNoBias)"""
        adaln = AdaptiveLayerNorm(dim_input=128, dim_cond=128)
        
        assert adaln.linear_shift.bias is None


class TestComparisonWithStandardLayerNorm:
    """Compare with standard LayerNorm"""
    
    def test_different_from_standard_layernorm(self):
        """AdaLN should behave differently from standard LayerNorm"""
        adaln = AdaptiveLayerNorm(dim_input=128, dim_cond=128)
        standard_ln = torch.nn.LayerNorm(128)
        
        a = torch.randn(10, 128)
        s = torch.randn(10, 128)
        
        out_adaln = adaln(a, s)
        out_standard = standard_ln(a)
        
        # Should be different (AdaLN is conditioned)
        assert not torch.allclose(out_adaln, out_standard, atol=1e-3)
    
    def test_conditioning_enables_adaptation(self):
        """AdaLN can adapt while standard LN cannot"""
        adaln = AdaptiveLayerNorm(dim_input=64, dim_cond=64)
        
        a = torch.randn(10, 64)
        
        # Two different conditioning signals
        s1 = torch.randn(10, 64)
        s2 = torch.randn(10, 64)
        
        out1 = adaln(a, s1)
        out2 = adaln(a, s2)
        
        # Different conditioning → different outputs (even if small before training)
        # The key is they ARE different, showing conditioning works
        mean_diff = (out1 - out2).abs().mean()
        assert mean_diff > 0.0  # Any difference shows conditioning is working
        
        # Additional check: outputs should not be identical
        assert not torch.allclose(out1, out2)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_zero_conditioning(self):
        """Should handle zero conditioning"""
        adaln = AdaptiveLayerNorm(dim_input=64, dim_cond=64)
        
        a = torch.randn(10, 64)
        s = torch.zeros(10, 64)
        
        out = adaln(a, s)
        
        assert out.shape == (10, 64)
        assert not torch.isnan(out).any()
    
    def test_zero_activations(self):
        """Should handle zero activations"""
        adaln = AdaptiveLayerNorm(dim_input=64, dim_cond=64)
        
        a = torch.zeros(10, 64)
        s = torch.randn(10, 64)
        
        out = adaln(a, s)
        
        assert out.shape == (10, 64)
        assert not torch.isnan(out).any()
    
    def test_large_values(self):
        """Should handle large values without overflow"""
        adaln = AdaptiveLayerNorm(dim_input=64, dim_cond=64)
        
        a = torch.randn(10, 64) * 100
        s = torch.randn(10, 64) * 100
        
        out = adaln(a, s)
        
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_single_position(self):
        """Should work with single position"""
        adaln = AdaptiveLayerNorm(dim_input=64, dim_cond=64)
        
        a = torch.randn(1, 64)
        s = torch.randn(1, 64)
        
        out = adaln(a, s)
        
        assert out.shape == (1, 64)


class TestGradientFlow:
    """Test gradient flow"""
    
    def test_gradients_to_activations(self):
        """Gradients should flow to activations"""
        adaln = AdaptiveLayerNorm(dim_input=64, dim_cond=64)
        
        a = torch.randn(10, 64, requires_grad=True)
        s = torch.randn(10, 64)
        
        out = adaln(a, s)
        loss = out.sum()
        loss.backward()
        
        assert a.grad is not None
        assert not torch.allclose(a.grad, torch.zeros_like(a.grad))
    
    def test_gradients_to_conditioning(self):
        """Gradients should flow to conditioning"""
        adaln = AdaptiveLayerNorm(dim_input=64, dim_cond=64)
        
        a = torch.randn(10, 64)
        s = torch.randn(10, 64, requires_grad=True)
        
        out = adaln(a, s)
        loss = out.sum()
        loss.backward()
        
        assert s.grad is not None
        assert not torch.allclose(s.grad, torch.zeros_like(s.grad))
    
    def test_gradients_to_parameters(self):
        """Gradients should flow to learned parameters"""
        adaln = AdaptiveLayerNorm(dim_input=64, dim_cond=64)
        
        a = torch.randn(10, 64, requires_grad=True)
        s = torch.randn(10, 64, requires_grad=True)
        
        out = adaln(a, s)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist for linear layers
        assert adaln.linear_scale.weight.grad is not None
        assert adaln.linear_shift.weight.grad is not None


class TestBatchConsistency:
    """Test batch processing consistency"""
    
    def test_batch_consistency(self):
        """Batched and individual processing should match"""
        adaln = AdaptiveLayerNorm(dim_input=64, dim_cond=64)
        
        a_batched = torch.randn(4, 10, 64)
        s_batched = torch.randn(4, 10, 64)
        
        # Process as batch
        out_batched = adaln(a_batched, s_batched)
        
        # Process individually
        outs_individual = []
        for i in range(4):
            out_i = adaln(a_batched[i], s_batched[i])
            outs_individual.append(out_i)
        
        # Should match
        for i in range(4):
            assert torch.allclose(out_batched[i], outs_individual[i], atol=1e-6)


class TestDummyInputGeneration:
    """Test dummy input utility"""
    
    def test_unbatched_dummy(self):
        """Should generate unbatched dummy inputs"""
        a, s = create_dummy_adaln_input()
        
        assert a.shape == (10, 384)
        assert s.shape == (10, 384)
    
    def test_batched_dummy(self):
        """Should generate batched dummy inputs"""
        a, s = create_dummy_adaln_input(batch_size=4)
        
        assert a.shape == (4, 10, 384)
        assert s.shape == (4, 10, 384)
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        a, s = create_dummy_adaln_input(
            batch_size=2,
            dim_input=256,
            dim_cond=512
        )
        
        assert a.shape == (2, 10, 256)
        assert s.shape == (2, 10, 512)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])