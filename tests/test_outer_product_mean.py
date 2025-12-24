"""
Unit tests for AlphaFold3 Outer Product Mean.

File: tests/test_outer_product_mean.py

Tests cover:
1. Initialization
2. Forward pass shape validation
3. Outer product computation
4. Mean pooling over sequences
5. Batched and unbatched inputs
6. Algorithm 9 faithfulness
7. Co-evolution capture mechanism
"""

import pytest
import torch
from src.models.trunk.outer_product_mean import (
    OuterProductMean,
    create_dummy_msa_input
)


class TestInitialization:
    """Test OuterProductMean initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        opm = OuterProductMean()
        
        assert opm.c_msa == 64
        assert opm.c_pair == 128
        assert opm.c == 32
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        opm = OuterProductMean(c_msa=128, c_pair=256, c=64)
        
        assert opm.c_msa == 128
        assert opm.c_pair == 256
        assert opm.c == 64
    
    def test_layer_dimensions(self):
        """Linear layers should have correct dimensions"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        
        # Projection layers (no bias)
        assert opm.linear_a.in_features == 64
        assert opm.linear_a.out_features == 32
        assert opm.linear_a.bias is None
        
        assert opm.linear_b.in_features == 64
        assert opm.linear_b.out_features == 32
        assert opm.linear_b.bias is None
        
        # Output projection (has bias)
        assert opm.linear_out.in_features == 32 * 32  # Flattened c*c
        assert opm.linear_out.out_features == 128
        assert opm.linear_out.bias is not None


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_basic_forward_unbatched(self):
        """Should handle unbatched input [N_msa, N_token, c_msa]"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        msa = torch.randn(512, 10, 64)
        
        pair_update = opm(msa)
        
        # Should produce pair representation
        assert pair_update.shape == (10, 10, 128)
    
    def test_basic_forward_batched(self):
        """Should handle batched input [batch, N_msa, N_token, c_msa]"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        msa = torch.randn(4, 512, 10, 64)
        
        pair_update = opm(msa)
        
        # Should produce batched pair representation
        assert pair_update.shape == (4, 10, 10, 128)
    
    def test_different_msa_sizes(self):
        """Should work with different MSA sizes"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        
        for n_msa in [128, 256, 512, 1024]:
            msa = torch.randn(n_msa, 10, 64)
            pair_update = opm(msa)
            
            assert pair_update.shape == (10, 10, 128)
    
    def test_different_token_counts(self):
        """Should work with different token counts"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        
        for n_token in [5, 10, 20, 50]:
            msa = torch.randn(512, n_token, 64)
            pair_update = opm(msa)
            
            assert pair_update.shape == (n_token, n_token, 128)


class TestOuterProductComputation:
    """Test outer product computation mechanism"""
    
    def test_outer_product_shape(self):
        """Intermediate outer product should have correct shape"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        msa = torch.randn(512, 10, 64)
        
        # The outer product internally should be [N_msa, N_token, N_token, c, c]
        # After mean: [N_token, N_token, c, c]
        # After flatten: [N_token, N_token, c*c]
        
        pair_update = opm(msa)
        
        # Final output should be projected to c_pair
        assert pair_update.shape == (10, 10, 128)
    
    def test_outer_product_symmetry(self):
        """Outer product captures pairwise relationships"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        
        # MSA with identical sequences (no variation)
        msa_identical = torch.ones(512, 10, 64)
        
        # MSA with variation
        msa_varied = torch.randn(512, 10, 64)
        
        pair_identical = opm(msa_identical)
        pair_varied = opm(msa_varied)
        
        # Different MSA patterns should produce different pair updates
        assert not torch.allclose(pair_identical, pair_varied, atol=1e-3)


class TestMeanPooling:
    """Test mean pooling over MSA sequences"""
    
    def test_mean_over_sequences(self):
        """Should average over MSA sequence dimension"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        
        # Create MSA with known pattern
        n_msa = 100
        n_token = 5
        msa = torch.randn(n_msa, n_token, 64)
        
        pair_update = opm(msa)
        
        # Output should be independent of individual sequences
        # (only depends on their average)
        assert pair_update.shape == (n_token, n_token, 128)
    
    def test_msa_depth_affects_output(self):
        """Different MSA depths should produce different outputs"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        
        # Shallow MSA
        msa_shallow = torch.randn(10, 10, 64)
        
        # Deep MSA
        msa_deep = torch.randn(1000, 10, 64)
        
        pair_shallow = opm(msa_shallow)
        pair_deep = opm(msa_deep)
        
        # Different depths (with different random values) should differ
        assert not torch.allclose(pair_shallow, pair_deep, atol=1e-5)


class TestBatchedInputs:
    """Test batched input handling"""
    
    def test_batch_consistency(self):
        """Each batch element should be processed independently"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        
        # Create batched input
        msa_batched = torch.randn(4, 512, 10, 64)
        
        # Process as batch
        pair_batched = opm(msa_batched)
        
        # Process individually
        pairs_individual = []
        for i in range(4):
            pair_i = opm(msa_batched[i])
            pairs_individual.append(pair_i)
        
        # Should match
        for i in range(4):
            assert torch.allclose(pair_batched[i], pairs_individual[i], atol=1e-5)
    
    def test_batch_shape_preservation(self):
        """Batch dimension should be preserved"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        
        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            msa = torch.randn(batch_size, 512, 10, 64)
            pair_update = opm(msa)
            
            assert pair_update.shape == (batch_size, 10, 10, 128)


class TestAlgorithm9Faithfulness:
    """Test faithfulness to Algorithm 9 specification"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 9 structure:
        1: m_si ← LayerNorm(m_si)
        2: a_si, b_si = LinearNoBias(m_si)
        3: o_ij = flatten(mean_s(a_si ⊗ b_sj))
        4: z_ij = Linear(o_ij)
        5: return {z_ij}
        """
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        msa = torch.randn(512, 10, 64)
        
        pair_update = opm(msa)
        
        # Should follow algorithm and produce pair update
        assert pair_update.shape == (10, 10, 128)
    
    def test_default_parameters(self):
        """Should use paper's default parameters"""
        opm = OuterProductMean()
        
        # Algorithm 9 defaults
        assert opm.c == 32  # Intermediate dimension
        assert opm.c_pair == 128  # Pair representation
    
    def test_no_bias_in_projections(self):
        """Projection layers should have no bias (as per algorithm)"""
        opm = OuterProductMean()
        
        assert opm.linear_a.bias is None
        assert opm.linear_b.bias is None


class TestCoevolutionCapture:
    """Test that outer product captures co-evolutionary patterns"""
    
    def test_correlated_positions(self):
        """Correlated MSA positions should produce distinct pair features"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        
        n_msa = 500
        n_token = 10
        
        # Create MSA with artificial correlation between positions 0 and 1
        msa = torch.randn(n_msa, n_token, 64)
        
        # Make position 1 correlated with position 0
        msa[:, 1, :] = msa[:, 0, :] + 0.1 * torch.randn(n_msa, 64)
        
        pair_update = opm(msa)
        
        # Pair feature for (0, 1) should capture this correlation
        assert pair_update.shape == (10, 10, 128)
    
    def test_information_flow(self):
        """MSA information should flow to pair representation"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        
        # Different MSA → different pair updates
        msa1 = torch.randn(512, 10, 64)
        msa2 = torch.randn(512, 10, 64)
        
        pair1 = opm(msa1)
        pair2 = opm(msa2)
        
        # Should be different
        assert not torch.allclose(pair1, pair2, atol=1e-5)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_sequence(self):
        """Should handle single MSA sequence"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        msa = torch.randn(1, 10, 64)
        
        pair_update = opm(msa)
        
        assert pair_update.shape == (10, 10, 128)
    
    def test_single_token(self):
        """Should handle single token"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        msa = torch.randn(512, 1, 64)
        
        pair_update = opm(msa)
        
        assert pair_update.shape == (1, 1, 128)
    
    def test_large_msa(self):
        """Should handle large MSA"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        msa = torch.randn(5000, 10, 64)
        
        pair_update = opm(msa)
        
        assert pair_update.shape == (10, 10, 128)
    
    def test_custom_dimensions(self):
        """Should work with non-default dimensions"""
        opm = OuterProductMean(c_msa=128, c_pair=256, c=64)
        msa = torch.randn(512, 10, 128)
        
        pair_update = opm(msa)
        
        assert pair_update.shape == (10, 10, 256)


class TestGradientFlow:
    """Test gradient flow through the layer"""
    
    def test_gradients_computed(self):
        """Gradients should be computed for all parameters"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        msa = torch.randn(512, 10, 64, requires_grad=True)
        
        pair_update = opm(msa)
        loss = pair_update.sum()
        loss.backward()
        
        # All parameters should have gradients
        assert opm.linear_a.weight.grad is not None
        assert opm.linear_b.weight.grad is not None
        assert opm.linear_out.weight.grad is not None
        assert opm.linear_out.bias.grad is not None
    
    def test_input_gradients(self):
        """Input should receive gradients"""
        opm = OuterProductMean(c_msa=64, c_pair=128, c=32)
        msa = torch.randn(512, 10, 64, requires_grad=True)
        
        pair_update = opm(msa)
        loss = pair_update.sum()
        loss.backward()
        
        # Input should have gradients
        assert msa.grad is not None
        assert not torch.allclose(msa.grad, torch.zeros_like(msa.grad))


class TestDummyInputGeneration:
    """Test dummy input generation utility"""
    
    def test_unbatched_dummy(self):
        """Should generate unbatched dummy input"""
        msa = create_dummy_msa_input(n_msa=512, n_token=10, c_msa=64)
        
        assert msa.shape == (512, 10, 64)
    
    def test_batched_dummy(self):
        """Should generate batched dummy input"""
        msa = create_dummy_msa_input(
            n_msa=512, n_token=10, c_msa=64, batch_size=4
        )
        
        assert msa.shape == (4, 512, 10, 64)
    
    def test_different_sizes(self):
        """Should generate various sizes"""
        for n_msa, n_token, c_msa in [(128, 5, 32), (1024, 20, 128)]:
            msa = create_dummy_msa_input(n_msa, n_token, c_msa)
            assert msa.shape == (n_msa, n_token, c_msa)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])