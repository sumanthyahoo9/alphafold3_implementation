"""
Unit tests for AlphaFold3 Conditioned Transition Block.

File: tests/test_conditioned_transition.py

Tests cover:
1. Initialization
2. Forward pass shape validation
3. AdaLN integration
4. SwiGLU behavior
5. Adaptive gating (adaLN-Zero)
6. Algorithm 25 faithfulness
"""

import pytest
import torch
from src.models.diffusion.conditioned_transition import (
    ConditionedTransitionBlock,
    create_dummy_conditioned_transition_input
)


class TestInitialization:
    """Test ConditionedTransitionBlock initialization"""
    
    def test_basic_initialization(self):
        """Should initialize with correct dimensions"""
        block = ConditionedTransitionBlock(dim=768, dim_cond=384)
        
        assert block.dim == 768
        assert block.dim_cond == 384
        assert block.n == 2  # Default expansion factor
        assert block.dim_hidden == 1536  # 2 * 768
    
    def test_custom_expansion_factor(self):
        """Should accept custom expansion factor"""
        block = ConditionedTransitionBlock(dim=512, dim_cond=256, n=4)
        
        assert block.n == 4
        assert block.dim_hidden == 2048  # 4 * 512
    
    def test_gate_bias_initialization(self):
        """Gate bias should be initialized to -2.0"""
        block = ConditionedTransitionBlock(dim=256, dim_cond=256)
        
        # Check gate bias is -2.0
        assert torch.allclose(
            block.gate.bias,
            torch.ones_like(block.gate.bias) * -2.0
        )
    
    def test_has_adaln(self):
        """Should have AdaptiveLayerNorm component"""
        block = ConditionedTransitionBlock(dim=256, dim_cond=256)
        
        assert hasattr(block, 'adaln')
        assert block.adaln.dim_input == 256
        assert block.adaln.dim_cond == 256


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_basic_forward_unbatched(self):
        """Should handle unbatched input"""
        block = ConditionedTransitionBlock(dim=768, dim_cond=384)
        
        a = torch.randn(10, 768)
        s = torch.randn(10, 384)
        
        out = block(a, s)
        
        assert out.shape == (10, 768)
    
    def test_basic_forward_batched(self):
        """Should handle batched input"""
        block = ConditionedTransitionBlock(dim=768, dim_cond=384)
        
        a = torch.randn(4, 10, 768)
        s = torch.randn(4, 10, 384)
        
        out = block(a, s)
        
        assert out.shape == (4, 10, 768)
    
    def test_different_sequence_lengths(self):
        """Should work with different sequence lengths"""
        block = ConditionedTransitionBlock(dim=512, dim_cond=256)
        
        for seq_len in [5, 10, 20, 50]:
            a = torch.randn(seq_len, 512)
            s = torch.randn(seq_len, 256)
            
            out = block(a, s)
            
            assert out.shape == (seq_len, 512)
    
    def test_different_dimensions(self):
        """Should work with different dimensions"""
        for dim in [256, 512, 768, 1024]:
            block = ConditionedTransitionBlock(dim=dim, dim_cond=dim)
            
            a = torch.randn(10, dim)
            s = torch.randn(10, dim)
            
            out = block(a, s)
            
            assert out.shape == (10, dim)


class TestAdaLNIntegration:
    """Test AdaLN integration"""
    
    def test_uses_adaln(self):
        """Should use AdaLN for input normalization"""
        block = ConditionedTransitionBlock(dim=256, dim_cond=256)
        
        a = torch.randn(10, 256)
        s1 = torch.randn(10, 256)
        s2 = torch.randn(10, 256)
        
        # Different conditioning → different outputs (via AdaLN)
        out1 = block(a, s1)
        out2 = block(a, s2)
        
        assert not torch.allclose(out1, out2, atol=1e-5)
    
    def test_conditioning_affects_output(self):
        """Conditioning should affect output through AdaLN"""
        block = ConditionedTransitionBlock(dim=128, dim_cond=128)
        
        a = torch.randn(10, 128)
        
        # Zero vs non-zero conditioning
        s_zero = torch.zeros(10, 128)
        s_nonzero = torch.randn(10, 128)
        
        out_zero = block(a, s_zero)
        out_nonzero = block(a, s_nonzero)
        
        # Should produce different outputs
        assert not torch.allclose(out_zero, out_nonzero, atol=1e-5)


class TestSwiGLUBehavior:
    """Test SwiGLU activation behavior"""
    
    def test_expansion_factor(self):
        """Hidden dimension should be n times input dimension"""
        for n in [2, 4, 8]:
            block = ConditionedTransitionBlock(dim=256, dim_cond=128, n=n)
            
            assert block.dim_hidden == 256 * n
            assert block.linear_1.out_features == 256 * n
            assert block.linear_2.out_features == 256 * n
    
    def test_output_same_dimension_as_input(self):
        """Output should have same dimension as input"""
        block = ConditionedTransitionBlock(dim=512, dim_cond=256, n=4)
        
        a = torch.randn(10, 512)
        s = torch.randn(10, 256)
        
        out = block(a, s)
        
        # Output dimension matches input
        assert out.shape[-1] == 512


class TestAdaptiveGating:
    """Test adaptive output gating (adaLN-Zero)"""
    
    def test_gate_near_zero_initially(self):
        """Gate should be near zero initially (bias=-2.0)"""
        block = ConditionedTransitionBlock(dim=256, dim_cond=256)
        
        # With zero conditioning, gate ≈ sigmoid(-2.0) ≈ 0.12
        s_zero = torch.zeros(10, 256)
        
        # Compute gate value
        gate_value = torch.sigmoid(block.gate(s_zero))
        
        # Should be near 0.12 (sigmoid(-2))
        expected = torch.sigmoid(torch.tensor(-2.0))
        assert torch.allclose(gate_value.mean(), expected, atol=0.1)
    
    def test_gate_modulates_output(self):
        """Gate should modulate the output"""
        block = ConditionedTransitionBlock(dim=128, dim_cond=128)
        
        a = torch.randn(10, 128)
        
        # Different conditioning → different gating
        s1 = torch.randn(10, 128) * -10  # Push gate toward 0
        s2 = torch.randn(10, 128) * 10   # Push gate toward 1
        
        out1 = block(a, s1)
        out2 = block(a, s2)
        
        # Different gating → different outputs
        assert not torch.allclose(out1, out2, atol=1e-3)
    
    def test_gate_has_bias(self):
        """Gate linear layer should have bias (unlike other projections)"""
        block = ConditionedTransitionBlock(dim=256, dim_cond=256)
        
        # Gate has bias (initialized to -2.0)
        assert block.gate.bias is not None
        
        # Other projections have no bias
        assert block.linear_1.bias is None
        assert block.linear_2.bias is None
        assert block.linear_out.bias is None


class TestAlgorithm25Faithfulness:
    """Test faithfulness to Algorithm 25"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 25:
        1: a ← AdaLN(a, s)
        2: b ← swish(LinearNoBias(a)) ⊙ LinearNoBias(a)
        3: a ← sigmoid(Linear(s, bias_init=-2.0)) ⊙ LinearNoBias(b)
        4: return a
        """
        block = ConditionedTransitionBlock(dim=256, dim_cond=256, n=2)
        
        a_input = torch.randn(10, 256)
        s = torch.randn(10, 256)
        
        # Manually compute expected output
        # Line 1: AdaLN
        a = block.adaln(a_input, s)
        
        # Line 2: SwiGLU
        swish_out = torch.nn.functional.silu(block.linear_1(a))
        gate_out = block.linear_2(a)
        b = swish_out * gate_out
        b = block.linear_out(b)
        
        # Line 3: Adaptive gating
        gate_value = torch.sigmoid(block.gate(s))
        expected = gate_value * b
        
        # Actual output
        actual = block(a_input, s)
        
        # Should match
        assert torch.allclose(actual, expected, atol=1e-6)
    
    def test_default_expansion_factor(self):
        """Default expansion factor should be n=2"""
        block = ConditionedTransitionBlock(dim=768, dim_cond=384)
        
        assert block.n == 2


class TestEdgeCases:
    """Test edge cases"""
    
    def test_zero_activations(self):
        """Should handle zero activations"""
        block = ConditionedTransitionBlock(dim=128, dim_cond=128)
        
        a = torch.zeros(10, 128)
        s = torch.randn(10, 128)
        
        out = block(a, s)
        
        assert out.shape == (10, 128)
        assert not torch.isnan(out).any()
    
    def test_zero_conditioning(self):
        """Should handle zero conditioning"""
        block = ConditionedTransitionBlock(dim=128, dim_cond=128)
        
        a = torch.randn(10, 128)
        s = torch.zeros(10, 128)
        
        out = block(a, s)
        
        assert out.shape == (10, 128)
        assert not torch.isnan(out).any()
    
    def test_large_values(self):
        """Should handle large values without overflow"""
        block = ConditionedTransitionBlock(dim=128, dim_cond=128)
        
        a = torch.randn(10, 128) * 100
        s = torch.randn(10, 128) * 100
        
        out = block(a, s)
        
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_single_position(self):
        """Should work with single position"""
        block = ConditionedTransitionBlock(dim=128, dim_cond=128)
        
        a = torch.randn(1, 128)
        s = torch.randn(1, 128)
        
        out = block(a, s)
        
        assert out.shape == (1, 128)


class TestGradientFlow:
    """Test gradient flow"""
    
    def test_gradients_to_activations(self):
        """Gradients should flow to activations"""
        block = ConditionedTransitionBlock(dim=128, dim_cond=128)
        
        a = torch.randn(10, 128, requires_grad=True)
        s = torch.randn(10, 128)
        
        out = block(a, s)
        loss = out.sum()
        loss.backward()
        
        assert a.grad is not None
        assert not torch.allclose(a.grad, torch.zeros_like(a.grad))
    
    def test_gradients_to_conditioning(self):
        """Gradients should flow to conditioning"""
        block = ConditionedTransitionBlock(dim=128, dim_cond=128)
        
        a = torch.randn(10, 128)
        s = torch.randn(10, 128, requires_grad=True)
        
        out = block(a, s)
        loss = out.sum()
        loss.backward()
        
        assert s.grad is not None
        assert not torch.allclose(s.grad, torch.zeros_like(s.grad))
    
    def test_gradients_to_all_parameters(self):
        """Gradients should flow to all learnable parameters"""
        block = ConditionedTransitionBlock(dim=128, dim_cond=128)
        
        a = torch.randn(10, 128, requires_grad=True)
        s = torch.randn(10, 128, requires_grad=True)
        
        out = block(a, s)
        loss = out.sum()
        loss.backward()
        
        # Check all major components have gradients
        assert block.linear_1.weight.grad is not None
        assert block.linear_2.weight.grad is not None
        assert block.linear_out.weight.grad is not None
        assert block.gate.weight.grad is not None
        assert block.gate.bias.grad is not None


class TestBatchConsistency:
    """Test batch processing consistency"""
    
    def test_batch_consistency(self):
        """Batched and individual processing should match"""
        block = ConditionedTransitionBlock(dim=256, dim_cond=256)
        
        a_batched = torch.randn(4, 10, 256)
        s_batched = torch.randn(4, 10, 256)
        
        # Process as batch
        out_batched = block(a_batched, s_batched)
        
        # Process individually
        outs_individual = []
        for i in range(4):
            out_i = block(a_batched[i], s_batched[i])
            outs_individual.append(out_i)
        
        # Should match
        for i in range(4):
            assert torch.allclose(out_batched[i], outs_individual[i], atol=1e-6)


class TestDummyInputGeneration:
    """Test dummy input utility"""
    
    def test_unbatched_dummy(self):
        """Should generate unbatched dummy inputs"""
        a, s = create_dummy_conditioned_transition_input()
        
        assert a.shape == (10, 768)
        assert s.shape == (10, 384)
    
    def test_batched_dummy(self):
        """Should generate batched dummy inputs"""
        a, s = create_dummy_conditioned_transition_input(batch_size=4)
        
        assert a.shape == (4, 10, 768)
        assert s.shape == (4, 10, 384)
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        a, s = create_dummy_conditioned_transition_input(
            batch_size=2,
            dim=512,
            dim_cond=256
        )
        
        assert a.shape == (2, 10, 512)
        assert s.shape == (2, 10, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])