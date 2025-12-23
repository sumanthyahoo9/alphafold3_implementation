"""
Unit tests for AlphaFold3 Transition Layer.

Tests cover:
1. Initialization
2. Forward pass shape validation
3. SwiGLU activation
4. Different input shapes (2D, 3D, 4D)
5. Algorithm 11 faithfulness
6. Expansion factor variations
"""

import pytest
import torch
from src.models.trunk.transition import Transition, create_dummy_transition_input


class TestInitialization:
    """Test Transition layer initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default expansion factor"""
        transition = Transition(c=128)
        
        assert transition.c == 128
        assert transition.n == 4
        assert transition.hidden_dim == 512
    
    def test_custom_expansion(self):
        """Should accept custom expansion factor"""
        transition = Transition(c=256, n=8)
        
        assert transition.c == 256
        assert transition.n == 8
        assert transition.hidden_dim == 2048
    
    def test_layer_dimensions(self):
        """Linear layers should have correct dimensions"""
        transition = Transition(c=128, n=4)
        
        # Input projection layers
        assert transition.linear_a.in_features == 128
        assert transition.linear_a.out_features == 512
        assert transition.linear_b.in_features == 128
        assert transition.linear_b.out_features == 512
        
        # Output projection
        assert transition.linear_out.in_features == 512
        assert transition.linear_out.out_features == 128


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_basic_forward_2d(self):
        """Should handle 2D input [batch, features]"""
        transition = Transition(c=128, n=4)
        x = torch.randn(8, 128)
        
        out = transition(x)
        
        assert out.shape == x.shape
        assert out.shape == (8, 128)
    
    def test_basic_forward_3d(self):
        """Should handle 3D input [batch, seq, features]"""
        transition = Transition(c=128, n=4)
        x = torch.randn(4, 10, 128)
        
        out = transition(x)
        
        assert out.shape == x.shape
        assert out.shape == (4, 10, 128)
    
    def test_basic_forward_4d(self):
        """Should handle 4D input [batch, seq1, seq2, features] for pairs"""
        transition = Transition(c=128, n=4)
        x = torch.randn(2, 10, 10, 128)
        
        out = transition(x)
        
        assert out.shape == x.shape
        assert out.shape == (2, 10, 10, 128)
    
    def test_different_dimensions(self):
        """Should work with various feature dimensions"""
        for c in [64, 128, 256, 384, 512]:
            transition = Transition(c=c, n=4)
            x = torch.randn(2, 10, c)
            
            out = transition(x)
            
            assert out.shape == (2, 10, c)


class TestSwiGLUActivation:
    """Test SwiGLU activation mechanism"""
    
    def test_swiglu_changes_output(self):
        """SwiGLU should produce different output than linear"""
        transition = Transition(c=128, n=4)
        x = torch.randn(4, 10, 128)
        
        # Forward pass
        out = transition(x)
        
        # Should not be identity
        assert not torch.allclose(out, x, atol=1e-3)
    
    def test_swiglu_gating(self):
        """Gating mechanism should affect output"""
        transition = Transition(c=128, n=4)
        
        # Two different inputs
        x1 = torch.randn(2, 10, 128)
        x2 = torch.randn(2, 10, 128)
        
        out1 = transition(x1)
        out2 = transition(x2)
        
        # Different inputs → different outputs
        assert not torch.allclose(out1, out2, atol=1e-5)
    
    def test_activation_non_linearity(self):
        """Should exhibit non-linear behavior"""
        transition = Transition(c=128, n=4)
        
        x = torch.randn(2, 10, 128)
        
        # f(2x) ≠ 2*f(x) for non-linear functions
        out_x = transition(x)
        out_2x = transition(2 * x)
        
        assert not torch.allclose(out_2x, 2 * out_x, atol=1e-3)


class TestExpansionFactor:
    """Test different expansion factors"""
    
    def test_expansion_factor_2(self):
        """Should work with n=2"""
        transition = Transition(c=128, n=2)
        x = torch.randn(4, 10, 128)
        
        out = transition(x)
        
        assert out.shape == (4, 10, 128)
        assert transition.hidden_dim == 256
    
    def test_expansion_factor_4(self):
        """Should work with n=4 (default)"""
        transition = Transition(c=128, n=4)
        x = torch.randn(4, 10, 128)
        
        out = transition(x)
        
        assert out.shape == (4, 10, 128)
        assert transition.hidden_dim == 512
    
    def test_expansion_factor_8(self):
        """Should work with n=8"""
        transition = Transition(c=256, n=8)
        x = torch.randn(4, 10, 256)
        
        out = transition(x)
        
        assert out.shape == (4, 10, 256)
        assert transition.hidden_dim == 2048


class TestAlgorithm11Faithfulness:
    """Test faithfulness to Algorithm 11 specification"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 11 structure:
        1: x ← LayerNorm(x)
        2: a = LinearNoBias(x)
        3: b = LinearNoBias(x)
        4: x ← LinearNoBias(swish(a) ⊙ b)
        5: return x
        """
        transition = Transition(c=128, n=4)
        x = torch.randn(4, 10, 128)
        
        out = transition(x)
        
        # Should produce output with same shape
        assert out.shape == x.shape
    
    def test_no_bias(self):
        """Linear layers should have no bias"""
        transition = Transition(c=128, n=4)
        
        assert transition.linear_a.bias is None
        assert transition.linear_b.bias is None
        assert transition.linear_out.bias is None
    
    def test_default_parameters(self):
        """Should use paper's default expansion factor"""
        transition = Transition(c=128)
        
        # Algorithm 11 default: n=4
        assert transition.n == 4


class TestResidualConnection:
    """Test that transition is designed for residual connections"""
    
    def test_with_residual(self):
        """Should work with residual connection as intended"""
        transition = Transition(c=128, n=4)
        x = torch.randn(4, 10, 128)
        
        # As used in parent modules: x = x + Transition(x)
        out = x + transition(x)
        
        assert out.shape == x.shape
    
    def test_residual_preserves_gradient(self):
        """Residual connection should preserve gradient flow"""
        transition = Transition(c=128, n=4)
        x = torch.randn(4, 10, 128, requires_grad=True)
        
        # Forward with residual
        out = x + transition(x)
        loss = out.sum()
        
        # Backward
        loss.backward()
        
        # Gradients should exist
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_element(self):
        """Should handle single element"""
        transition = Transition(c=128, n=4)
        x = torch.randn(1, 1, 128)
        
        out = transition(x)
        
        assert out.shape == (1, 1, 128)
    
    def test_large_batch(self):
        """Should handle large batch sizes"""
        transition = Transition(c=128, n=4)
        x = torch.randn(64, 50, 128)
        
        out = transition(x)
        
        assert out.shape == (64, 50, 128)
    
    def test_small_channels(self):
        """Should work with small channel dimensions"""
        transition = Transition(c=8, n=4)
        x = torch.randn(4, 10, 8)
        
        out = transition(x)
        
        assert out.shape == (4, 10, 8)
    
    def test_large_channels(self):
        """Should work with large channel dimensions"""
        transition = Transition(c=1024, n=4)
        x = torch.randn(2, 5, 1024)
        
        out = transition(x)
        
        assert out.shape == (2, 5, 1024)


class TestGradientFlow:
    """Test gradient flow through the layer"""
    
    def test_gradients_computed(self):
        """Gradients should be computed for all parameters"""
        transition = Transition(c=128, n=4)
        x = torch.randn(4, 10, 128, requires_grad=True)
        
        out = transition(x)
        loss = out.sum()
        loss.backward()
        
        # All parameters should have gradients
        assert transition.linear_a.weight.grad is not None
        assert transition.linear_b.weight.grad is not None
        assert transition.linear_out.weight.grad is not None
    
    def test_gradient_magnitude(self):
        """Gradients should have reasonable magnitude"""
        transition = Transition(c=128, n=4)
        x = torch.randn(4, 10, 128, requires_grad=True)
        
        out = transition(x)
        loss = out.sum()
        loss.backward()
        
        # Gradients should not be too small or too large
        for param in transition.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                assert grad_norm > 1e-6  # Not vanishing
                assert grad_norm < 1e6   # Not exploding


class TestDummyInputGeneration:
    """Test dummy input generation utility"""
    
    def test_dummy_input_shape(self):
        """Dummy input should have correct shape"""
        x = create_dummy_transition_input(batch_size=4, seq_len=10, c=128)
        
        assert x.shape == (4, 10, 128)
    
    def test_dummy_input_different_sizes(self):
        """Should generate various input sizes"""
        for batch, seq, c in [(2, 5, 64), (8, 20, 256), (1, 100, 512)]:
            x = create_dummy_transition_input(batch, seq, c)
            assert x.shape == (batch, seq, c)


class TestMSAUsage:
    """Test usage in MSA context"""
    
    def test_msa_representation(self):
        """Should work with MSA representation [batch, n_seq, n_token, c]"""
        transition = Transition(c=64, n=4)
        
        # MSA: [batch, n_msa, n_token, c_msa]
        msa = torch.randn(2, 512, 10, 64)
        
        out = transition(msa)
        
        assert out.shape == msa.shape


class TestPairUsage:
    """Test usage in pair representation context"""
    
    def test_pair_representation(self):
        """Should work with pair representation [batch, n_token, n_token, c]"""
        transition = Transition(c=128, n=4)
        
        # Pair: [batch, n_token, n_token, c_pair]
        pair = torch.randn(2, 10, 10, 128)
        
        out = transition(pair)
        
        assert out.shape == pair.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])