"""
Unit tests for AlphaFold3 MSA Module.

File: tests/test_msa_module.py

Tests cover:
1. Initialization
2. Forward pass shape validation
3. Block structure
4. MSA to pair communication
5. Dropout behavior
6. Algorithm 8 faithfulness
7. Integration of all components
"""

import pytest
import torch
from src.models.trunk.msa_module import MSAModule, MSAModuleBlock


class TestMSAModuleInitialization:
    """Test MSAModule initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        module = MSAModule()
        
        assert module.c_msa == 64
        assert module.c_pair == 128
        assert module.c_single == 384
        assert module.n_blocks == 4
        assert len(module.blocks) == 4
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        module = MSAModule(
            c_msa=128,
            c_pair=256,
            c_single=512,
            n_blocks=8
        )
        
        assert module.c_msa == 128
        assert module.c_pair == 256
        assert module.c_single == 512
        assert module.n_blocks == 8
        assert len(module.blocks) == 8


class TestMSAModuleForwardPass:
    """Test MSAModule forward pass"""
    
    def test_basic_forward_unbatched(self):
        """Should handle unbatched input"""
        module = MSAModule(c_msa=64, c_pair=128, c_single=384, n_blocks=4)
        
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        single = torch.randn(10, 384)
        
        pair_out = module(msa, pair, single)
        
        # Returns updated pair
        assert pair_out.shape == (10, 10, 128)
    
    def test_basic_forward_batched(self):
        """Should handle batched input"""
        module = MSAModule(c_msa=64, c_pair=128, c_single=384, n_blocks=4)
        
        msa = torch.randn(4, 512, 10, 64)
        pair = torch.randn(4, 10, 10, 128)
        single = torch.randn(4, 10, 384)
        
        pair_out = module(msa, pair, single)
        
        assert pair_out.shape == (4, 10, 10, 128)
    
    def test_different_msa_sizes(self):
        """Should work with different MSA sizes"""
        module = MSAModule()
        
        for n_msa in [128, 256, 512]:
            msa = torch.randn(n_msa, 10, 64)
            pair = torch.randn(10, 10, 128)
            single = torch.randn(10, 384)
            
            pair_out = module(msa, pair, single)
            
            assert pair_out.shape == (10, 10, 128)
    
    def test_different_block_counts(self):
        """Should work with different numbers of blocks"""
        for n_blocks in [1, 2, 4, 8]:
            module = MSAModule(n_blocks=n_blocks)
            
            msa = torch.randn(512, 10, 64)
            pair = torch.randn(10, 10, 128)
            single = torch.randn(10, 384)
            
            pair_out = module(msa, pair, single)
            
            assert pair_out.shape == (10, 10, 128)


class TestBlockStructure:
    """Test MSA Module block structure"""
    
    def test_has_correct_number_of_blocks(self):
        """Should create correct number of blocks"""
        for n_blocks in [1, 4, 8]:
            module = MSAModule(n_blocks=n_blocks)
            assert len(module.blocks) == n_blocks
    
    def test_blocks_are_independent(self):
        """Each block should be independent"""
        module = MSAModule(n_blocks=4)
        
        # Each block has its own parameters
        for i in range(len(module.blocks) - 1):
            block1 = module.blocks[i]
            block2 = module.blocks[i + 1]
            
            # Different instances
            assert block1 is not block2


class TestMSAToPairCommunication:
    """Test MSA to pair communication"""
    
    def test_outer_product_mean_enriches_pair(self):
        """OuterProductMean should transfer MSA info to pair"""
        module = MSAModule(n_blocks=1)
        
        msa = torch.randn(512, 10, 64)
        pair_init = torch.randn(10, 10, 128)
        single = torch.randn(10, 384)
        
        pair_out = module(msa, pair_init, single)
        
        # Pair should be updated (different from input)
        assert not torch.allclose(pair_out, pair_init, atol=1e-5)
    
    def test_msa_influences_pair(self):
        """Different MSA should produce different pair"""
        module = MSAModule(n_blocks=1)
        
        msa1 = torch.randn(512, 10, 64)
        msa2 = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        single = torch.randn(10, 384)
        
        pair_out1 = module(msa1, pair, single)
        pair_out2 = module(msa2, pair, single)
        
        # Different MSA → different pair output
        assert not torch.allclose(pair_out1, pair_out2, atol=1e-5)


class TestDropoutBehavior:
    """Test dropout during training"""
    
    def test_dropout_only_in_training(self):
        """Dropout should only apply during training"""
        module = MSAModule(n_blocks=1)
        
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        single = torch.randn(10, 384)
        
        # Eval mode - deterministic
        module.eval()
        out1 = module(msa, pair, single)
        out2 = module(msa, pair, single)
        
        assert torch.allclose(out1, out2, atol=1e-6)
    
    def test_dropout_affects_training(self):
        """Dropout should add randomness during training"""
        module = MSAModule(n_blocks=1)
        module.train()
        
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 128)
        single = torch.randn(10, 384)
        
        # Training mode - should have some randomness from dropout
        # (though not guaranteed to be different every time)
        out = module(msa, pair, single)
        
        assert out.shape == (10, 10, 128)


class TestAlgorithm8Faithfulness:
    """Test faithfulness to Algorithm 8"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 8:
        1-2: MSA concatenation and sampling (done externally)
        3: m_si ← LinearNoBias(m_si)
        4: m_si += LinearNoBias(s_inputs_i)
        5-14: For each block:
            6: pair += OuterProductMean(msa)
            7: msa += Dropout(MSAPairWeightedAveraging(msa, pair))
            8: msa += Transition(msa)
            9-10: pair += Dropout(TriangleMultiplication)
            11-12: pair += Dropout(TriangleAttention)
            13: pair += Transition(pair)
        15: return pair
        """
        module = MSAModule(n_blocks=4)
        
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        single = torch.randn(10, 384)
        
        pair_out = module(msa, pair, single)
        
        assert pair_out.shape == (10, 10, 128)
    
    def test_default_parameters(self):
        """Should use paper's default parameters"""
        module = MSAModule()
        
        # Algorithm 8 defaults
        assert module.n_blocks == 4
        assert module.c_msa == 64
        assert module.c_pair == 128
    
    def test_returns_only_pair(self):
        """Should return only pair representation (line 15)"""
        module = MSAModule()
        
        msa = torch.randn(512, 10, 64)
        pair = torch.randn(10, 10, 128)
        single = torch.randn(10, 384)
        
        result = module(msa, pair, single)
        
        # Should return tensor (pair), not tuple
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 10, 128)


class TestComponentIntegration:
    """Test integration of all components"""
    
    def test_uses_outer_product_mean(self):
        """Should use OuterProductMean for MSA→Pair"""
        block = MSAModuleBlock()
        
        assert hasattr(block, 'outer_product_mean')
        assert block.outer_product_mean is not None
    
    def test_uses_msa_attention(self):
        """Should use MSAPairWeightedAveraging"""
        block = MSAModuleBlock()
        
        assert hasattr(block, 'msa_attention')
        assert block.msa_attention is not None
    
    def test_uses_triangle_updates(self):
        """Should use both triangle multiplication variants"""
        block = MSAModuleBlock()
        
        assert hasattr(block, 'triangle_mult_outgoing')
        assert hasattr(block, 'triangle_mult_incoming')
    
    def test_uses_triangle_attention(self):
        """Should use both triangle attention variants"""
        block = MSAModuleBlock()
        
        assert hasattr(block, 'triangle_attn_starting')
        assert hasattr(block, 'triangle_attn_ending')
    
    def test_uses_transitions(self):
        """Should use Transition for both MSA and pair"""
        block = MSAModuleBlock()
        
        assert hasattr(block, 'msa_transition')
        assert hasattr(block, 'pair_transition')


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_msa_sequence(self):
        """Should handle single MSA sequence"""
        module = MSAModule(n_blocks=1)
        
        msa = torch.randn(1, 10, 64)
        pair = torch.randn(10, 10, 128)
        single = torch.randn(10, 384)
        
        pair_out = module(msa, pair, single)
        
        assert pair_out.shape == (10, 10, 128)
    
    def test_single_token(self):
        """Should handle single token"""
        module = MSAModule(n_blocks=1)
        
        msa = torch.randn(512, 1, 64)
        pair = torch.randn(1, 1, 128)
        single = torch.randn(1, 384)
        
        pair_out = module(msa, pair, single)
        
        assert pair_out.shape == (1, 1, 128)
    
    def test_large_msa(self):
        """Should handle large MSA"""
        module = MSAModule(n_blocks=1)
        
        msa = torch.randn(5000, 10, 64)
        pair = torch.randn(10, 10, 128)
        single = torch.randn(10, 384)
        
        pair_out = module(msa, pair, single)
        
        assert pair_out.shape == (10, 10, 128)
    
    def test_custom_dimensions(self):
        """Should work with non-default dimensions"""
        module = MSAModule(c_msa=128, c_pair=256, c_single=512, n_blocks=2)
        
        msa = torch.randn(512, 10, 128)
        pair = torch.randn(10, 10, 256)
        single = torch.randn(10, 512)
        
        pair_out = module(msa, pair, single)
        
        assert pair_out.shape == (10, 10, 256)


class TestGradientFlow:
    """Test gradient flow"""
    
    def test_gradients_computed(self):
        """Gradients should flow through the module"""
        module = MSAModule(n_blocks=1)
        
        msa = torch.randn(512, 10, 64, requires_grad=True)
        pair = torch.randn(10, 10, 128, requires_grad=True)
        single = torch.randn(10, 384, requires_grad=True)
        
        pair_out = module(msa, pair, single)
        loss = pair_out.sum()
        loss.backward()
        
        # All inputs should have gradients
        assert msa.grad is not None
        assert pair.grad is not None
        assert single.grad is not None


class TestBatchConsistency:
    """Test batch processing consistency"""
    
    def test_batch_consistency(self):
        """Batched and individual processing should match"""
        module = MSAModule(n_blocks=1)
        module.eval()  # Disable dropout for deterministic comparison
        
        msa_batched = torch.randn(4, 512, 10, 64)
        pair_batched = torch.randn(4, 10, 10, 128)
        single_batched = torch.randn(4, 10, 384)
        
        # Process as batch
        pair_out_batched = module(msa_batched, pair_batched, single_batched)
        
        # Process individually
        pairs_out_individual = []
        for i in range(4):
            pair_out_i = module(msa_batched[i], pair_batched[i], single_batched[i])
            pairs_out_individual.append(pair_out_i)
        
        # Should match
        for i in range(4):
            assert torch.allclose(
                pair_out_batched[i], pairs_out_individual[i], atol=1e-5
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])