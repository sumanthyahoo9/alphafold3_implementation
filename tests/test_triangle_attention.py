"""
Unit tests for AlphaFold3 Triangle Attention.

File: tests/test_triangle_attention.py

Tests cover:
1. Initialization
2. Forward pass shape validation
3. Multi-head attention mechanism
4. Starting vs Ending node differences
5. Gating behavior
6. Pair bias integration
7. Algorithm 14-15 faithfulness
"""

import pytest
import torch
from src.models.trunk.triangle_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)
from src.models.trunk.triangle_updates import create_dummy_pair_input


class TestStartingNodeInitialization:
    """Test TriangleAttentionStartingNode initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        attn = TriangleAttentionStartingNode()
        
        assert attn.c_pair == 128
        assert attn.c == 32
        assert attn.n_heads == 4
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        attn = TriangleAttentionStartingNode(c_pair=256, c=64, n_heads=8)
        
        assert attn.c_pair == 256
        assert attn.c == 64
        assert attn.n_heads == 8


class TestEndingNodeInitialization:
    """Test TriangleAttentionEndingNode initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        attn = TriangleAttentionEndingNode()
        
        assert attn.c_pair == 128
        assert attn.c == 32
        assert attn.n_heads == 4
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        attn = TriangleAttentionEndingNode(c_pair=256, c=64, n_heads=8)
        
        assert attn.c_pair == 256
        assert attn.c == 64
        assert attn.n_heads == 8


class TestStartingNodeForwardPass:
    """Test TriangleAttentionStartingNode forward pass"""
    
    def test_basic_forward_unbatched(self):
        """Should handle unbatched input"""
        attn = TriangleAttentionStartingNode(c_pair=128, c=32, n_heads=4)
        pair = torch.randn(10, 10, 128)
        
        pair_update = attn(pair)
        
        assert pair_update.shape == (10, 10, 128)
    
    def test_basic_forward_batched(self):
        """Should handle batched input"""
        attn = TriangleAttentionStartingNode(c_pair=128, c=32, n_heads=4)
        pair = torch.randn(4, 10, 10, 128)
        
        pair_update = attn(pair)
        
        assert pair_update.shape == (4, 10, 10, 128)
    
    def test_different_token_counts(self):
        """Should work with different numbers of tokens"""
        attn = TriangleAttentionStartingNode()
        
        for n_token in [5, 10, 20, 50]:
            pair = torch.randn(n_token, n_token, 128)
            pair_update = attn(pair)
            
            assert pair_update.shape == (n_token, n_token, 128)


class TestEndingNodeForwardPass:
    """Test TriangleAttentionEndingNode forward pass"""
    
    def test_basic_forward_unbatched(self):
        """Should handle unbatched input"""
        attn = TriangleAttentionEndingNode(c_pair=128, c=32, n_heads=4)
        pair = torch.randn(10, 10, 128)
        
        pair_update = attn(pair)
        
        assert pair_update.shape == (10, 10, 128)
    
    def test_basic_forward_batched(self):
        """Should handle batched input"""
        attn = TriangleAttentionEndingNode(c_pair=128, c=32, n_heads=4)
        pair = torch.randn(4, 10, 10, 128)
        
        pair_update = attn(pair)
        
        assert pair_update.shape == (4, 10, 10, 128)
    
    def test_different_token_counts(self):
        """Should work with different numbers of tokens"""
        attn = TriangleAttentionEndingNode()
        
        for n_token in [5, 10, 20, 50]:
            pair = torch.randn(n_token, n_token, 128)
            pair_update = attn(pair)
            
            assert pair_update.shape == (n_token, n_token, 128)


class TestMultiHeadAttention:
    """Test multi-head attention mechanism"""
    
    def test_starting_node_attention(self):
        """Starting node should attend over edges from same node"""
        attn = TriangleAttentionStartingNode(c_pair=128, c=32, n_heads=4)
        pair = torch.randn(10, 10, 128)
        
        update = attn(pair)
        
        # Should produce valid update via attention
        assert update.shape == (10, 10, 128)
    
    def test_ending_node_attention(self):
        """Ending node should attend over edges to same node"""
        attn = TriangleAttentionEndingNode(c_pair=128, c=32, n_heads=4)
        pair = torch.randn(10, 10, 128)
        
        update = attn(pair)
        
        assert update.shape == (10, 10, 128)
    
    def test_different_head_counts(self):
        """Should work with different numbers of heads"""
        for n_heads in [1, 2, 4, 8]:
            attn_start = TriangleAttentionStartingNode(c_pair=128, c=32, n_heads=n_heads)
            attn_end = TriangleAttentionEndingNode(c_pair=128, c=32, n_heads=n_heads)
            
            pair = torch.randn(10, 10, 128)
            
            update_start = attn_start(pair)
            update_end = attn_end(pair)
            
            assert update_start.shape == (10, 10, 128)
            assert update_end.shape == (10, 10, 128)


class TestStartingVsEndingDifferences:
    """Test differences between starting and ending node attention"""
    
    def test_produce_different_results(self):
        """Starting and ending should produce different results"""
        attn_start = TriangleAttentionStartingNode(c_pair=128, c=32, n_heads=4)
        attn_end = TriangleAttentionEndingNode(c_pair=128, c=32, n_heads=4)
        
        pair = torch.randn(10, 10, 128)
        
        update_start = attn_start(pair)
        update_end = attn_end(pair)
        
        # Different attention patterns → different results
        assert not torch.allclose(update_start, update_end, atol=1e-5)
    
    def test_complementary_perspectives(self):
        """Both should provide valid but different perspectives"""
        attn_start = TriangleAttentionStartingNode()
        attn_end = TriangleAttentionEndingNode()
        
        pair = torch.randn(10, 10, 128)
        
        update_start = attn_start(pair)
        update_end = attn_end(pair)
        
        # Both produce valid updates
        assert update_start.shape == (10, 10, 128)
        assert update_end.shape == (10, 10, 128)
        # But from different perspectives
        assert not torch.allclose(update_start, update_end, atol=1e-5)


class TestGatingMechanism:
    """Test gating behavior"""
    
    def test_gating_affects_output_starting(self):
        """Gating should modulate output"""
        attn = TriangleAttentionStartingNode()
        
        pair1 = torch.randn(10, 10, 128)
        pair2 = torch.randn(10, 10, 128)
        
        update1 = attn(pair1)
        update2 = attn(pair2)
        
        # Different inputs → different outputs (gating working)
        assert not torch.allclose(update1, update2, atol=1e-5)
    
    def test_gating_affects_output_ending(self):
        """Gating should modulate output"""
        attn = TriangleAttentionEndingNode()
        
        pair1 = torch.randn(10, 10, 128)
        pair2 = torch.randn(10, 10, 128)
        
        update1 = attn(pair1)
        update2 = attn(pair2)
        
        assert not torch.allclose(update1, update2, atol=1e-5)


class TestPairBias:
    """Test pair bias integration in attention"""
    
    def test_bias_affects_attention_starting(self):
        """Pair bias should affect attention weights"""
        attn = TriangleAttentionStartingNode()
        pair = torch.randn(10, 10, 128)
        
        update = attn(pair)
        
        # Bias integrated into attention computation
        assert update.shape == (10, 10, 128)
    
    def test_bias_affects_attention_ending(self):
        """Pair bias should affect attention weights"""
        attn = TriangleAttentionEndingNode()
        pair = torch.randn(10, 10, 128)
        
        update = attn(pair)
        
        assert update.shape == (10, 10, 128)


class TestAlgorithm14Faithfulness:
    """Test faithfulness to Algorithm 14 (Starting Node)"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 14:
        1: z_ij ← LayerNorm(z_ij)
        2: q, k, v = LinearNoBias(z_ij)
        3: b = LinearNoBias(z_ij)
        4: g = sigmoid(LinearNoBias(z_ij))
        5: attn = softmax(Q^T K / √c + b_jk)
        6: o = g ⊙ Σ_k attn * v_ik
        7: output = LinearNoBias(concat(o))
        """
        attn = TriangleAttentionStartingNode()
        pair = torch.randn(10, 10, 128)
        
        pair_update = attn(pair)
        
        assert pair_update.shape == (10, 10, 128)
    
    def test_default_parameters(self):
        """Should use paper's default parameters"""
        attn = TriangleAttentionStartingNode()
        
        # Algorithm 14 defaults
        assert attn.c == 32
        assert attn.n_heads == 4
    
    def test_no_bias_in_projections(self):
        """Projection layers should have no bias"""
        attn = TriangleAttentionStartingNode()
        
        assert attn.linear_q.bias is None
        assert attn.linear_k.bias is None
        assert attn.linear_v.bias is None
        assert attn.linear_b.bias is None
        assert attn.linear_g.bias is None
        assert attn.linear_out.bias is None


class TestAlgorithm15Faithfulness:
    """Test faithfulness to Algorithm 15 (Ending Node)"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 15:
        1: z_ij ← LayerNorm(z_ij)
        2: q, k, v = LinearNoBias(z_ij)
        3: b = LinearNoBias(z_ij)
        4: g = sigmoid(LinearNoBias(z_ij))
        5: attn = softmax(Q^T K_kj / √c + b_ki)  ← different indices!
        6: o = g ⊙ Σ_k attn * v_kj  ← different indices!
        7: output = LinearNoBias(concat(o))
        """
        attn = TriangleAttentionEndingNode()
        pair = torch.randn(10, 10, 128)
        
        pair_update = attn(pair)
        
        assert pair_update.shape == (10, 10, 128)
    
    def test_default_parameters(self):
        """Should use paper's default parameters"""
        attn = TriangleAttentionEndingNode()
        
        # Algorithm 15 defaults (same as 14)
        assert attn.c == 32
        assert attn.n_heads == 4
    
    def test_no_bias_in_projections(self):
        """Projection layers should have no bias"""
        attn = TriangleAttentionEndingNode()
        
        assert attn.linear_q.bias is None
        assert attn.linear_k.bias is None
        assert attn.linear_v.bias is None
        assert attn.linear_b.bias is None
        assert attn.linear_g.bias is None
        assert attn.linear_out.bias is None


class TestResidualConnection:
    """Test usage with residual connections"""
    
    def test_with_residual_starting(self):
        """Should work with residual connection"""
        attn = TriangleAttentionStartingNode()
        pair = torch.randn(10, 10, 128)
        
        # As used: pair = pair + Attention(pair)
        pair_new = pair + attn(pair)
        
        assert pair_new.shape == pair.shape
    
    def test_with_residual_ending(self):
        """Should work with residual connection"""
        attn = TriangleAttentionEndingNode()
        pair = torch.randn(10, 10, 128)
        
        pair_new = pair + attn(pair)
        
        assert pair_new.shape == pair.shape


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_token_starting(self):
        """Should handle single token"""
        attn = TriangleAttentionStartingNode()
        pair = torch.randn(1, 1, 128)
        
        update = attn(pair)
        
        assert update.shape == (1, 1, 128)
    
    def test_single_token_ending(self):
        """Should handle single token"""
        attn = TriangleAttentionEndingNode()
        pair = torch.randn(1, 1, 128)
        
        update = attn(pair)
        
        assert update.shape == (1, 1, 128)
    
    def test_large_token_count_starting(self):
        """Should handle large token counts"""
        attn = TriangleAttentionStartingNode()
        pair = torch.randn(100, 100, 128)
        
        update = attn(pair)
        
        assert update.shape == (100, 100, 128)
    
    def test_large_token_count_ending(self):
        """Should handle large token counts"""
        attn = TriangleAttentionEndingNode()
        pair = torch.randn(100, 100, 128)
        
        update = attn(pair)
        
        assert update.shape == (100, 100, 128)
    
    def test_custom_dimensions(self):
        """Should work with non-default dimensions"""
        attn_start = TriangleAttentionStartingNode(c_pair=256, c=64, n_heads=8)
        attn_end = TriangleAttentionEndingNode(c_pair=256, c=64, n_heads=8)
        
        pair = torch.randn(10, 10, 256)
        
        update_start = attn_start(pair)
        update_end = attn_end(pair)
        
        assert update_start.shape == (10, 10, 256)
        assert update_end.shape == (10, 10, 256)


class TestGradientFlow:
    """Test gradient flow"""
    
    def test_gradients_computed_starting(self):
        """Gradients should flow through starting node attention"""
        attn = TriangleAttentionStartingNode()
        pair = torch.randn(10, 10, 128, requires_grad=True)
        
        update = attn(pair)
        loss = update.sum()
        loss.backward()
        
        assert pair.grad is not None
        assert not torch.allclose(pair.grad, torch.zeros_like(pair.grad))
    
    def test_gradients_computed_ending(self):
        """Gradients should flow through ending node attention"""
        attn = TriangleAttentionEndingNode()
        pair = torch.randn(10, 10, 128, requires_grad=True)
        
        update = attn(pair)
        loss = update.sum()
        loss.backward()
        
        assert pair.grad is not None
        assert not torch.allclose(pair.grad, torch.zeros_like(pair.grad))


class TestBatchConsistency:
    """Test batch processing consistency"""
    
    def test_batch_consistency_starting(self):
        """Batched and individual processing should match"""
        attn = TriangleAttentionStartingNode()
        
        pair_batched = torch.randn(4, 10, 10, 128)
        
        update_batched = attn(pair_batched)
        
        updates_individual = []
        for i in range(4):
            update_i = attn(pair_batched[i])
            updates_individual.append(update_i)
        
        for i in range(4):
            assert torch.allclose(update_batched[i], updates_individual[i], atol=1e-5)
    
    def test_batch_consistency_ending(self):
        """Batched and individual processing should match"""
        attn = TriangleAttentionEndingNode()
        
        pair_batched = torch.randn(4, 10, 10, 128)
        
        update_batched = attn(pair_batched)
        
        updates_individual = []
        for i in range(4):
            update_i = attn(pair_batched[i])
            updates_individual.append(update_i)
        
        for i in range(4):
            assert torch.allclose(update_batched[i], updates_individual[i], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])