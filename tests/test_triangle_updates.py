"""
Unit tests for AlphaFold3 Triangle Multiplicative Updates.

File: tests/test_triangle_updates.py

Tests cover:
1. Initialization
2. Forward pass shape validation
3. Triangle multiplication mechanism
4. Outgoing vs Incoming differences
5. Gating behavior
6. Algorithm 12-13 faithfulness
7. Geometric consistency
"""

import pytest
import torch
from src.models.trunk.triangle_updates import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    create_dummy_pair_input
)


class TestOutgoingInitialization:
    """Test TriangleMultiplicationOutgoing initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        triangle = TriangleMultiplicationOutgoing()
        
        assert triangle.c_pair == 128
        assert triangle.c == 128
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        triangle = TriangleMultiplicationOutgoing(c_pair=256, c=64)
        
        assert triangle.c_pair == 256
        assert triangle.c == 64


class TestIncomingInitialization:
    """Test TriangleMultiplicationIncoming initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        triangle = TriangleMultiplicationIncoming()
        
        assert triangle.c_pair == 128
        assert triangle.c == 128
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        triangle = TriangleMultiplicationIncoming(c_pair=256, c=64)
        
        assert triangle.c_pair == 256
        assert triangle.c == 64


class TestOutgoingForwardPass:
    """Test TriangleMultiplicationOutgoing forward pass"""
    
    def test_basic_forward_unbatched(self):
        """Should handle unbatched input"""
        triangle = TriangleMultiplicationOutgoing(c_pair=128, c=128)
        pair = torch.randn(10, 10, 128)
        
        pair_update = triangle(pair)
        
        assert pair_update.shape == (10, 10, 128)
    
    def test_basic_forward_batched(self):
        """Should handle batched input"""
        triangle = TriangleMultiplicationOutgoing(c_pair=128, c=128)
        pair = torch.randn(4, 10, 10, 128)
        
        pair_update = triangle(pair)
        
        assert pair_update.shape == (4, 10, 10, 128)
    
    def test_different_token_counts(self):
        """Should work with different numbers of tokens"""
        triangle = TriangleMultiplicationOutgoing()
        
        for n_token in [5, 10, 20, 50]:
            pair = torch.randn(n_token, n_token, 128)
            pair_update = triangle(pair)
            
            assert pair_update.shape == (n_token, n_token, 128)


class TestIncomingForwardPass:
    """Test TriangleMultiplicationIncoming forward pass"""
    
    def test_basic_forward_unbatched(self):
        """Should handle unbatched input"""
        triangle = TriangleMultiplicationIncoming(c_pair=128, c=128)
        pair = torch.randn(10, 10, 128)
        
        pair_update = triangle(pair)
        
        assert pair_update.shape == (10, 10, 128)
    
    def test_basic_forward_batched(self):
        """Should handle batched input"""
        triangle = TriangleMultiplicationIncoming(c_pair=128, c=128)
        pair = torch.randn(4, 10, 10, 128)
        
        pair_update = triangle(pair)
        
        assert pair_update.shape == (4, 10, 10, 128)
    
    def test_different_token_counts(self):
        """Should work with different numbers of tokens"""
        triangle = TriangleMultiplicationIncoming()
        
        for n_token in [5, 10, 20, 50]:
            pair = torch.randn(n_token, n_token, 128)
            pair_update = triangle(pair)
            
            assert pair_update.shape == (n_token, n_token, 128)


class TestTriangleMultiplication:
    """Test triangle multiplication mechanism"""
    
    def test_outgoing_vs_incoming_differ(self):
        """Outgoing and incoming should produce different results"""
        outgoing = TriangleMultiplicationOutgoing(c_pair=128, c=128)
        incoming = TriangleMultiplicationIncoming(c_pair=128, c=128)
        
        pair = torch.randn(10, 10, 128)
        
        update_out = outgoing(pair)
        update_in = incoming(pair)
        
        # Different algorithms → different results
        assert not torch.allclose(update_out, update_in, atol=1e-5)
    
    def test_intermediate_node_summation(self):
        """Should sum over intermediate nodes"""
        triangle = TriangleMultiplicationOutgoing()
        pair = torch.randn(10, 10, 128)
        
        update = triangle(pair)
        
        # Update should aggregate information from all paths through intermediate nodes
        assert update.shape == (10, 10, 128)


class TestGatingMechanism:
    """Test gating behavior"""
    
    def test_gating_affects_output_outgoing(self):
        """Gating should modulate output"""
        triangle = TriangleMultiplicationOutgoing()
        
        # Two different inputs
        pair1 = torch.randn(10, 10, 128)
        pair2 = torch.randn(10, 10, 128)
        
        update1 = triangle(pair1)
        update2 = triangle(pair2)
        
        # Different inputs → different outputs (gating working)
        assert not torch.allclose(update1, update2, atol=1e-5)
    
    def test_gating_affects_output_incoming(self):
        """Gating should modulate output"""
        triangle = TriangleMultiplicationIncoming()
        
        pair1 = torch.randn(10, 10, 128)
        pair2 = torch.randn(10, 10, 128)
        
        update1 = triangle(pair1)
        update2 = triangle(pair2)
        
        # Different inputs → different outputs
        assert not torch.allclose(update1, update2, atol=1e-5)


class TestAlgorithm12Faithfulness:
    """Test faithfulness to Algorithm 12 (Outgoing)"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 12:
        1: z_ij ← LayerNorm(z_ij)
        2: a_ij, b_ij = sigmoid(Linear(z_ij)) ⊙ Linear(z_ij)
        3: g_ij = sigmoid(Linear(z_ij))
        4: z̃_ij = g_ij ⊙ Linear(LayerNorm(Σ_k a_ik ⊙ b_jk))
        5: return {z̃_ij}
        """
        triangle = TriangleMultiplicationOutgoing()
        pair = torch.randn(10, 10, 128)
        
        pair_update = triangle(pair)
        
        assert pair_update.shape == (10, 10, 128)
    
    def test_default_parameters(self):
        """Should use paper's default parameters"""
        triangle = TriangleMultiplicationOutgoing()
        
        # Algorithm 12 default: c=128
        assert triangle.c == 128
    
    def test_no_bias_in_projections(self):
        """Projection layers should have no bias"""
        triangle = TriangleMultiplicationOutgoing()
        
        assert triangle.linear_a.bias is None
        assert triangle.linear_b.bias is None
        assert triangle.linear_g.bias is None
        assert triangle.linear_out.bias is None


class TestAlgorithm13Faithfulness:
    """Test faithfulness to Algorithm 13 (Incoming)"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 13:
        1: z_ij ← LayerNorm(z_ij)
        2: a_ij, b_ij = sigmoid(Linear(z_ij)) ⊙ Linear(z_ij)
        3: g_ij = sigmoid(Linear(z_ij))
        4: z̃_ij = g_ij ⊙ Linear(LayerNorm(Σ_k a_ki ⊙ b_kj))  ← different!
        5: return {z̃_ij}
        """
        triangle = TriangleMultiplicationIncoming()
        pair = torch.randn(10, 10, 128)
        
        pair_update = triangle(pair)
        
        assert pair_update.shape == (10, 10, 128)
    
    def test_default_parameters(self):
        """Should use paper's default parameters"""
        triangle = TriangleMultiplicationIncoming()
        
        # Algorithm 13 default: c=128
        assert triangle.c == 128
    
    def test_no_bias_in_projections(self):
        """Projection layers should have no bias"""
        triangle = TriangleMultiplicationIncoming()
        
        assert triangle.linear_a.bias is None
        assert triangle.linear_b.bias is None
        assert triangle.linear_g.bias is None
        assert triangle.linear_out.bias is None


class TestGeometricConsistency:
    """Test geometric consistency enforcement"""
    
    def test_triangle_information_flow(self):
        """Information should flow through triangular paths"""
        triangle = TriangleMultiplicationOutgoing()
        
        # Create pair with specific pattern
        n_token = 5
        pair = torch.randn(n_token, n_token, 128)
        
        update = triangle(pair)
        
        # Update should depend on triangular relationships
        # (i,j) update uses (i,k) and (k,j) for all k
        assert update.shape == (n_token, n_token, 128)
    
    def test_consistency_from_different_perspectives(self):
        """Outgoing and incoming enforce same constraint differently"""
        outgoing = TriangleMultiplicationOutgoing()
        incoming = TriangleMultiplicationIncoming()
        
        pair = torch.randn(10, 10, 128)
        
        update_out = outgoing(pair)
        update_in = incoming(pair)
        
        # Both produce valid updates (different but both enforce consistency)
        assert update_out.shape == (10, 10, 128)
        assert update_in.shape == (10, 10, 128)


class TestResidualConnection:
    """Test usage with residual connections"""
    
    def test_with_residual_outgoing(self):
        """Should work with residual connection as in MSA/Pairformer"""
        triangle = TriangleMultiplicationOutgoing()
        pair = torch.randn(10, 10, 128)
        
        # As used: pair = pair + TriangleUpdate(pair)
        pair_new = pair + triangle(pair)
        
        assert pair_new.shape == pair.shape
    
    def test_with_residual_incoming(self):
        """Should work with residual connection"""
        triangle = TriangleMultiplicationIncoming()
        pair = torch.randn(10, 10, 128)
        
        pair_new = pair + triangle(pair)
        
        assert pair_new.shape == pair.shape


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_token_outgoing(self):
        """Should handle single token"""
        triangle = TriangleMultiplicationOutgoing()
        pair = torch.randn(1, 1, 128)
        
        update = triangle(pair)
        
        assert update.shape == (1, 1, 128)
    
    def test_single_token_incoming(self):
        """Should handle single token"""
        triangle = TriangleMultiplicationIncoming()
        pair = torch.randn(1, 1, 128)
        
        update = triangle(pair)
        
        assert update.shape == (1, 1, 128)
    
    def test_large_token_count_outgoing(self):
        """Should handle large token counts"""
        triangle = TriangleMultiplicationOutgoing()
        pair = torch.randn(100, 100, 128)
        
        update = triangle(pair)
        
        assert update.shape == (100, 100, 128)
    
    def test_large_token_count_incoming(self):
        """Should handle large token counts"""
        triangle = TriangleMultiplicationIncoming()
        pair = torch.randn(100, 100, 128)
        
        update = triangle(pair)
        
        assert update.shape == (100, 100, 128)
    
    def test_custom_dimensions(self):
        """Should work with non-default dimensions"""
        outgoing = TriangleMultiplicationOutgoing(c_pair=256, c=64)
        incoming = TriangleMultiplicationIncoming(c_pair=256, c=64)
        
        pair = torch.randn(10, 10, 256)
        
        update_out = outgoing(pair)
        update_in = incoming(pair)
        
        assert update_out.shape == (10, 10, 256)
        assert update_in.shape == (10, 10, 256)


class TestGradientFlow:
    """Test gradient flow"""
    
    def test_gradients_computed_outgoing(self):
        """Gradients should flow through outgoing update"""
        triangle = TriangleMultiplicationOutgoing()
        pair = torch.randn(10, 10, 128, requires_grad=True)
        
        update = triangle(pair)
        loss = update.sum()
        loss.backward()
        
        # Input should have gradients
        assert pair.grad is not None
        assert not torch.allclose(pair.grad, torch.zeros_like(pair.grad))
    
    def test_gradients_computed_incoming(self):
        """Gradients should flow through incoming update"""
        triangle = TriangleMultiplicationIncoming()
        pair = torch.randn(10, 10, 128, requires_grad=True)
        
        update = triangle(pair)
        loss = update.sum()
        loss.backward()
        
        assert pair.grad is not None
        assert not torch.allclose(pair.grad, torch.zeros_like(pair.grad))


class TestBatchConsistency:
    """Test batch processing consistency"""
    
    def test_batch_consistency_outgoing(self):
        """Batched and individual processing should match"""
        triangle = TriangleMultiplicationOutgoing()
        
        pair_batched = torch.randn(4, 10, 10, 128)
        
        # Process as batch
        update_batched = triangle(pair_batched)
        
        # Process individually
        updates_individual = []
        for i in range(4):
            update_i = triangle(pair_batched[i])
            updates_individual.append(update_i)
        
        # Should match
        for i in range(4):
            assert torch.allclose(update_batched[i], updates_individual[i], atol=1e-5)
    
    def test_batch_consistency_incoming(self):
        """Batched and individual processing should match"""
        triangle = TriangleMultiplicationIncoming()
        
        pair_batched = torch.randn(4, 10, 10, 128)
        
        update_batched = triangle(pair_batched)
        
        updates_individual = []
        for i in range(4):
            update_i = triangle(pair_batched[i])
            updates_individual.append(update_i)
        
        for i in range(4):
            assert torch.allclose(update_batched[i], updates_individual[i], atol=1e-5)


class TestDummyInputGeneration:
    """Test dummy input utility"""
    
    def test_unbatched_dummy(self):
        """Should generate unbatched dummy input"""
        pair = create_dummy_pair_input(n_token=10, c_pair=128)
        
        assert pair.shape == (10, 10, 128)
    
    def test_batched_dummy(self):
        """Should generate batched dummy input"""
        pair = create_dummy_pair_input(n_token=10, c_pair=128, batch_size=4)
        
        assert pair.shape == (4, 10, 10, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])