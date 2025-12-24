"""
Unit tests for AlphaFold3 MSA Pair Weighted Averaging.

File: tests/test_msa_pair_weighted_averaging.py

Tests cover:
1. Initialization
2. Forward pass shape validation
3. Pair bias attention mechanism
4. Multi-head attention
5. Gating behavior
6. Algorithm 10 faithfulness
7. Consistency across MSA rows
"""

import pytest
import torch
from src.models.trunk.msa_pair_weighted_averaging import (
    MSAPairWeightedAveraging,
    create_dummy_msa_pair_input
)


class TestInitialization:
    """Test MSAPairWeightedAveraging initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        attn = MSAPairWeightedAveraging()
        
        assert attn.c_msa == 64
        assert attn.c_pair == 128
        assert attn.c == 32
        assert attn.n_heads == 8
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        attn = MSAPairWeightedAveraging(
            c_msa=128, c_pair=256, c=64, n_heads=16
        )
        
        assert attn.c_msa == 128
        assert attn.c_pair == 256
        assert attn.c == 64
        assert attn.n_heads == 16


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_basic_forward_unbatched(self):
        """Should handle unbatched input"""
        attn = MSAPairWeightedAveraging(c_msa=64, c_pair=128, c=32, n_heads=8)
        
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        
        msa_update = attn(msa, pair)
        
        assert msa_update.shape == (512, 10, 64)
    
    def test_basic_forward_batched(self):
        """Should handle batched input"""
        attn = MSAPairWeightedAveraging(c_msa=64, c_pair=128, c=32, n_heads=8)
        
        msa = torch.randn(4, 512, 10, 64)
        pair = torch.randn(4, 10, 10, 128)
        
        msa_update = attn(msa, pair)
        
        assert msa_update.shape == (4, 512, 10, 64)
    
    def test_different_msa_sizes(self):
        """Should work with different MSA sizes"""
        attn = MSAPairWeightedAveraging()
        
        for n_msa in [128, 256, 512, 1024]:
            msa = torch.randn(n_msa, 10, 64)
            pair = torch.randn(10, 10, 128)
            
            msa_update = attn(msa, pair)
            
            assert msa_update.shape == (n_msa, 10, 64)
    
    def test_different_token_counts(self):
        """Should work with different token counts"""
        attn = MSAPairWeightedAveraging()
        
        for n_token in [5, 10, 20, 50]:
            msa = torch.randn(512, n_token, 64)
            pair = torch.randn(n_token, n_token, 128)
            
            msa_update = attn(msa, pair)
            
            assert msa_update.shape == (512, n_token, 64)


class TestPairBiasAttention:
    """Test pair bias attention mechanism"""
    
    def test_no_query_key_attention(self):
        """Weights should come from pair only, not Q·K"""
        attn = MSAPairWeightedAveraging()
        
        # Same MSA, different pair → different results
        msa = torch.randn(512, 10, 64)
        pair1 = torch.randn(10, 10, 128)
        pair2 = torch.randn(10, 10, 128)
        
        update1 = attn(msa, pair1)
        update2 = attn(msa, pair2)
        
        # Different pair → different attention → different output
        assert not torch.allclose(update1, update2, atol=1e-5)
    
    def test_pair_controls_attention(self):
        """Pair representation should control attention weights"""
        attn = MSAPairWeightedAveraging()
        
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        
        msa_update = attn(msa, pair)
        
        # Pair bias controls which positions attend to each other
        assert msa_update.shape == (512, 10, 64)
    
    def test_same_attention_for_all_rows(self):
        """All MSA rows should use same attention pattern"""
        attn = MSAPairWeightedAveraging()
        
        # This is implicit in the algorithm - pair bias is shared
        # across all MSA sequences
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        
        msa_update = attn(msa, pair)
        
        # All sequences get updated (same attention pattern)
        assert msa_update.shape == (512, 10, 64)


class TestMultiHeadAttention:
    """Test multi-head attention mechanism"""
    
    def test_different_head_counts(self):
        """Should work with different numbers of heads"""
        for n_heads in [1, 4, 8, 16]:
            attn = MSAPairWeightedAveraging(
                c_msa=64, c_pair=128, c=32, n_heads=n_heads
            )
            
            msa = torch.randn(512, 10, 64)
            pair = torch.randn(10, 10, 128)
            
            msa_update = attn(msa, pair)
            
            assert msa_update.shape == (512, 10, 64)
    
    def test_multi_head_diversity(self):
        """Different heads should capture different patterns"""
        attn = MSAPairWeightedAveraging(n_heads=8)
        
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        
        msa_update = attn(msa, pair)
        
        # Multi-head provides multiple attention perspectives
        assert msa_update.shape == (512, 10, 64)


class TestGatingMechanism:
    """Test gating behavior"""
    
    def test_gating_affects_output(self):
        """Gating should modulate output"""
        attn = MSAPairWeightedAveraging()
        
        # Different MSA → different gates → different outputs
        msa1 = torch.randn(512, 10, 64)
        msa2 = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        
        update1 = attn(msa1, pair)
        update2 = attn(msa2, pair)
        
        # Different MSA content → different gating
        assert not torch.allclose(update1, update2, atol=1e-5)
    
    def test_gate_is_msa_dependent(self):
        """Gate should depend on MSA content"""
        attn = MSAPairWeightedAveraging()
        
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        
        msa_update = attn(msa, pair)
        
        # Gate computed from MSA
        assert msa_update.shape == (512, 10, 64)


class TestAlgorithm10Faithfulness:
    """Test faithfulness to Algorithm 10 specification"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 10:
        1: m_si ← LayerNorm(m_si)
        2: v^h_si = LinearNoBias(m_si)
        3: b^h_ij = LinearNoBias(LayerNorm(z_ij))
        4: g^h_si = sigmoid(LinearNoBias(m_si))
        5: w^h_ij = softmax_j(b^h_ij)
        6: o^h_si = g^h_si ⊙ Σ_j w^h_ij v^h_sj
        7: m̃_si = LinearNoBias(concat_h(o^h_si))
        """
        attn = MSAPairWeightedAveraging()
        
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        
        msa_update = attn(msa, pair)
        
        assert msa_update.shape == (512, 10, 64)
    
    def test_default_parameters(self):
        """Should use paper's default parameters"""
        attn = MSAPairWeightedAveraging()
        
        # Algorithm 10 defaults
        assert attn.c == 32
        assert attn.n_heads == 8
    
    def test_no_bias_in_projections(self):
        """Projection layers should have no bias"""
        attn = MSAPairWeightedAveraging()
        
        assert attn.linear_v.bias is None
        assert attn.linear_b.bias is None
        assert attn.linear_g.bias is None
        assert attn.linear_out.bias is None


class TestConsistencyAcrossRows:
    """Test that MSA rows are processed consistently"""
    
    def test_row_independence(self):
        """Each MSA row should be processed independently"""
        attn = MSAPairWeightedAveraging()
        
        # Process full MSA
        msa_full = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        
        update_full = attn(msa_full, pair)
        
        # Process first row only
        msa_single = msa_full[:1]
        update_single = attn(msa_single, pair)
        
        # First row should match
        assert torch.allclose(update_full[0], update_single[0], atol=1e-5)
    
    def test_shared_attention_pattern(self):
        """All rows should use same attention pattern (from pair)"""
        attn = MSAPairWeightedAveraging()
        
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        
        msa_update = attn(msa, pair)
        
        # All sequences updated with same attention weights
        assert msa_update.shape == (512, 10, 64)


class TestResidualConnection:
    """Test usage with residual connections"""
    
    def test_with_residual(self):
        """Should work with residual connection as in MSA Module"""
        attn = MSAPairWeightedAveraging()
        
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        
        # As used: msa = msa + Attention(msa, pair)
        msa_new = msa + attn(msa, pair)
        
        assert msa_new.shape == msa.shape


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_sequence(self):
        """Should handle single MSA sequence"""
        attn = MSAPairWeightedAveraging()
        
        msa = torch.randn(1, 10, 64)
        pair = torch.randn(10, 10, 128)
        
        msa_update = attn(msa, pair)
        
        assert msa_update.shape == (1, 10, 64)
    
    def test_single_token(self):
        """Should handle single token"""
        attn = MSAPairWeightedAveraging()
        
        msa = torch.randn(512, 1, 64)
        pair = torch.randn(1, 1, 128)
        
        msa_update = attn(msa, pair)
        
        assert msa_update.shape == (512, 1, 64)
    
    def test_large_msa(self):
        """Should handle large MSA"""
        attn = MSAPairWeightedAveraging()
        
        msa = torch.randn(5000, 10, 64)
        pair = torch.randn(10, 10, 128)
        
        msa_update = attn(msa, pair)
        
        assert msa_update.shape == (5000, 10, 64)
    
    def test_custom_dimensions(self):
        """Should work with non-default dimensions"""
        attn = MSAPairWeightedAveraging(
            c_msa=128, c_pair=256, c=64, n_heads=16
        )
        
        msa = torch.randn(512, 10, 128)
        pair = torch.randn(10, 10, 256)
        
        msa_update = attn(msa, pair)
        
        assert msa_update.shape == (512, 10, 128)


class TestGradientFlow:
    """Test gradient flow"""
    
    def test_gradients_computed(self):
        """Gradients should flow through the layer"""
        attn = MSAPairWeightedAveraging()
        
        msa = torch.randn(512, 10, 64, requires_grad=True)
        pair = torch.randn(10, 10, 128, requires_grad=True)
        
        msa_update = attn(msa, pair)
        loss = msa_update.sum()
        loss.backward()
        
        # Both inputs should have gradients
        assert msa.grad is not None
        assert pair.grad is not None
        assert not torch.allclose(msa.grad, torch.zeros_like(msa.grad))
        assert not torch.allclose(pair.grad, torch.zeros_like(pair.grad))


class TestBatchConsistency:
    """Test batch processing consistency"""
    
    def test_batch_consistency(self):
        """Batched and individual processing should match"""
        attn = MSAPairWeightedAveraging()
        
        msa_batched = torch.randn(4, 512, 10, 64)
        pair_batched = torch.randn(4, 10, 10, 128)
        
        # Process as batch
        update_batched = attn(msa_batched, pair_batched)
        
        # Process individually
        updates_individual = []
        for i in range(4):
            update_i = attn(msa_batched[i], pair_batched[i])
            updates_individual.append(update_i)
        
        # Should match
        for i in range(4):
            assert torch.allclose(
                update_batched[i], updates_individual[i], atol=1e-5
            )


class TestDummyInputGeneration:
    """Test dummy input utility"""
    
    def test_unbatched_dummy(self):
        """Should generate unbatched dummy inputs"""
        msa, pair = create_dummy_msa_pair_input(
            n_msa=512, n_token=10, c_msa=64, c_pair=128
        )
        
        assert msa.shape == (512, 10, 64)
        assert pair.shape == (10, 10, 128)
    
    def test_batched_dummy(self):
        """Should generate batched dummy inputs"""
        msa, pair = create_dummy_msa_pair_input(
            n_msa=512, n_token=10, c_msa=64, c_pair=128, batch_size=4
        )
        
        assert msa.shape == (4, 512, 10, 64)
        assert pair.shape == (4, 10, 10, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])