"""
Unit tests for AlphaFold3 Diffusion Transformer.

File: tests/test_diffusion_transformer.py

Tests cover:
1. Initialization
2. Forward pass shape validation
3. AttentionPairBias behavior
4. Transformer block structure
5. Algorithm 23 & 24 faithfulness
6. Gradient flow
"""

import pytest
import torch
from src.models.diffusion.diffusion_transformer import (
    DiffusionAttentionPairBias,
    DiffusionTransformer,
    DiffusionTransformerBlock,
    create_dummy_diffusion_transformer_input
)


class TestDiffusionAttentionPairBias:
    """Test DiffusionAttentionPairBias (Algorithm 24)"""
    
    def test_initialization(self):
        """Should initialize with correct dimensions"""
        attn = DiffusionAttentionPairBias(c_token=768, c_pair=128, n_heads=16, c_s=384)
        
        assert attn.c_token == 768
        assert attn.c_pair == 128
        assert attn.n_heads == 16
        assert attn.c_s == 384
        assert attn.c_hidden == 48  # 768 / 16
    
    def test_with_conditioning(self):
        """Should use AdaLN when conditioning provided"""
        attn = DiffusionAttentionPairBias(c_token=768, c_pair=128, n_heads=16, c_s=384)
        
        assert attn.use_conditioning is True
        assert hasattr(attn, 'adaln')
        assert hasattr(attn, 'output_gate')
    
    def test_without_conditioning(self):
        """Should use standard LayerNorm when no conditioning"""
        # For the parameter c_s, change it to None as default
        attn = DiffusionAttentionPairBias(c_token=768, c_pair=128, n_heads=16, c_s=0)
        
        assert attn.use_conditioning is False
        assert hasattr(attn, 'layer_norm')
        assert not hasattr(attn, 'output_gate')
    
    def test_forward_unbatched(self):
        """Should handle unbatched input"""
        attn = DiffusionAttentionPairBias(c_token=768, c_pair=128, n_heads=16, c_s=384)
        
        a = torch.randn(10, 768)
        s = torch.randn(10, 384)
        z = torch.randn(10, 10, 128)
        
        out = attn(a, s, z)
        
        assert out.shape == (10, 768)
    
    def test_forward_batched(self):
        """Should handle batched input"""
        attn = DiffusionAttentionPairBias(c_token=768, c_pair=128, n_heads=16, c_s=384)
        
        a = torch.randn(4, 10, 768)
        s = torch.randn(4, 10, 384)
        z = torch.randn(4, 10, 10, 128)
        
        out = attn(a, s, z)
        
        assert out.shape == (4, 10, 768)
    
    def test_pair_bias_applied(self):
        """Pair should bias attention"""
        attn = DiffusionAttentionPairBias(c_token=768, c_pair=128, n_heads=16, c_s=384)
        
        a = torch.randn(10, 768)
        s = torch.randn(10, 384)
        z1 = torch.randn(10, 10, 128)
        z2 = torch.randn(10, 10, 128)
        
        out1 = attn(a, s, z1)
        out2 = attn(a, s, z2)
        
        # Different pair → different output
        assert not torch.allclose(out1, out2, atol=1e-5)
    
    def test_output_gate_initialization(self):
        """Output gate bias should be -2.0"""
        attn = DiffusionAttentionPairBias(c_token=768, c_pair=128, n_heads=16, c_s=384)
        
        # Check bias initialized to -2.0
        assert torch.allclose(
            attn.output_gate.bias,
            torch.ones_like(attn.output_gate.bias) * -2.0
        )


class TestDiffusionTransformerBlock:
    """Test single transformer block"""
    
    def test_initialization(self):
        """Should initialize attention and transition"""
        block = DiffusionTransformerBlock(c_token=768, c_pair=128, c_s=384, n_heads=16)
        
        assert hasattr(block, 'attention')
        assert hasattr(block, 'transition')
    
    def test_forward_unbatched(self):
        """Should handle unbatched input"""
        block = DiffusionTransformerBlock(c_token=768, c_pair=128, c_s=384, n_heads=16)
        
        a = torch.randn(10, 768)
        s = torch.randn(10, 384)
        z = torch.randn(10, 10, 128)
        
        out = block(a, s, z)
        
        assert out.shape == (10, 768)
    
    def test_forward_batched(self):
        """Should handle batched input"""
        block = DiffusionTransformerBlock(c_token=768, c_pair=128, c_s=384, n_heads=16)
        
        a = torch.randn(4, 10, 768)
        s = torch.randn(4, 10, 384)
        z = torch.randn(4, 10, 10, 128)
        
        out = block(a, s, z)
        
        assert out.shape == (4, 10, 768)
    
    def test_residual_connection(self):
        """Output should include residual from attention"""
        block = DiffusionTransformerBlock(c_token=768, c_pair=128, c_s=384, n_heads=16)
        
        a = torch.randn(10, 768)
        s = torch.randn(10, 384)
        z = torch.randn(10, 10, 128)
        
        # Forward pass includes residual
        out = block(a, s, z)
        
        # Output shape preserved
        assert out.shape == a.shape


class TestDiffusionTransformer:
    """Test full DiffusionTransformer (Algorithm 23)"""
    
    def test_initialization(self):
        """Should initialize with 24 blocks by default"""
        transformer = DiffusionTransformer()
        
        assert transformer.n_blocks == 24
        assert len(transformer.blocks) == 24
        assert transformer.c_token == 768
        assert transformer.c_pair == 128
        assert transformer.c_s == 384
        assert transformer.n_heads == 16
    
    def test_custom_blocks(self):
        """Should accept custom number of blocks"""
        transformer = DiffusionTransformer(n_blocks=12)
        
        assert transformer.n_blocks == 12
        assert len(transformer.blocks) == 12
    
    def test_forward_unbatched(self):
        """Should handle unbatched input"""
        transformer = DiffusionTransformer(n_blocks=2)  # Use fewer blocks for speed
        
        a = torch.randn(10, 768)
        s = torch.randn(10, 384)
        z = torch.randn(10, 10, 128)
        
        out = transformer(a, s, z)
        
        assert out.shape == (10, 768)
    
    def test_forward_batched(self):
        """Should handle batched input"""
        transformer = DiffusionTransformer(n_blocks=2)
        
        a = torch.randn(4, 10, 768)
        s = torch.randn(4, 10, 384)
        z = torch.randn(4, 10, 10, 128)
        
        out = transformer(a, s, z)
        
        assert out.shape == (4, 10, 768)
    
    def test_different_token_counts(self):
        """Should work with different numbers of tokens"""
        transformer = DiffusionTransformer(n_blocks=2)
        
        for n_token in [5, 10, 20]:
            a = torch.randn(n_token, 768)
            s = torch.randn(n_token, 384)
            z = torch.randn(n_token, n_token, 128)
            
            out = transformer(a, s, z)
            
            assert out.shape == (n_token, 768)
    
    def test_conditioning_affects_output(self):
        """Different conditioning should affect output"""
        transformer = DiffusionTransformer(n_blocks=2)
        
        a = torch.randn(10, 768)
        s1 = torch.randn(10, 384)
        s2 = torch.randn(10, 384)
        z = torch.randn(10, 10, 128)
        
        out1 = transformer(a, s1, z)
        out2 = transformer(a, s2, z)
        
        # Different conditioning → different output
        assert not torch.allclose(out1, out2, atol=1e-5)
    
    def test_pair_affects_output(self):
        """Different pair should affect output"""
        transformer = DiffusionTransformer(n_blocks=2)
        
        a = torch.randn(10, 768)
        s = torch.randn(10, 384)
        z1 = torch.randn(10, 10, 128)
        z2 = torch.randn(10, 10, 128)
        
        out1 = transformer(a, s, z1)
        out2 = transformer(a, s, z2)
        
        # Different pair → different output
        assert not torch.allclose(out1, out2, atol=1e-5)


class TestAlgorithm23Faithfulness:
    """Test faithfulness to Algorithm 23"""
    
    def test_algorithm_structure(self):
        """
        Algorithm 23:
        1: for n in [1...24]:
        2:     b = AttentionPairBias(a, s, z, β=0, heads=16)
        3:     a = b + ConditionedTransitionBlock(a, s)
        5: return a
        """
        transformer = DiffusionTransformer(n_blocks=24, n_heads=16)
        
        # Should have 24 blocks
        assert len(transformer.blocks) == 24
        
        # Each block should have attention + transition
        for block in transformer.blocks:
            assert hasattr(block, 'attention')
            assert hasattr(block, 'transition')
            assert block.attention.n_heads == 16
    
    def test_default_parameters_match_paper(self):
        """Default parameters should match Algorithm 23"""
        transformer = DiffusionTransformer()
        
        # From Algorithm 23 and Algorithm 20
        assert transformer.n_blocks == 24  # Nblock = 24
        assert transformer.n_heads == 16  # Nhead = 16
        assert transformer.c_token == 768  # ctoken = 768


class TestAlgorithm24Faithfulness:
    """Test faithfulness to Algorithm 24"""
    
    def test_adaln_when_conditioning(self):
        """Should use AdaLN when s is provided (lines 1-2)"""
        attn = DiffusionAttentionPairBias(c_token=768, c_pair=128, n_heads=16, c_s=384)
        
        # Line 1: if {si} ≠ ∅
        # Line 2: ai ← AdaLN(ai, si)
        assert attn.use_conditioning is True
        assert hasattr(attn, 'adaln')
    
    def test_layernorm_without_conditioning(self):
        """Should use LayerNorm when s is None (lines 3-4)"""
        # For the parameter c_s, change it to None as default
        attn = DiffusionAttentionPairBias(c_token=768, c_pair=128, n_heads=16, c_s=0)
        
        # Line 4: ai ← LayerNorm(ai)
        assert attn.use_conditioning is False
        assert hasattr(attn, 'layer_norm')
    
    def test_adaptive_output_gating(self):
        """Should use adaptive gating at output (lines 12-14)"""
        attn = DiffusionAttentionPairBias(c_token=768, c_pair=128, n_heads=16, c_s=384)
        
        # Lines 12-14: if {si} ≠ ∅ then ai ← sigmoid(Linear(si, biasinit=-2.0)) ⊙ ai
        assert hasattr(attn, 'output_gate')
        # Bias initialized to -2.0
        assert torch.allclose(
            attn.output_gate.bias,
            torch.ones_like(attn.output_gate.bias) * -2.0
        )


class TestGradientFlow:
    """Test gradient flow"""
    
    def test_gradients_to_activations(self):
        """Gradients should flow to activations"""
        transformer = DiffusionTransformer(n_blocks=2)
        
        a = torch.randn(10, 768, requires_grad=True)
        s = torch.randn(10, 384)
        z = torch.randn(10, 10, 128)
        
        out = transformer(a, s, z)
        loss = out.sum()
        loss.backward()
        
        assert a.grad is not None
        assert not torch.allclose(a.grad, torch.zeros_like(a.grad))
    
    def test_gradients_to_conditioning(self):
        """Gradients should flow to conditioning"""
        transformer = DiffusionTransformer(n_blocks=2)
        
        a = torch.randn(10, 768)
        s = torch.randn(10, 384, requires_grad=True)
        z = torch.randn(10, 10, 128)
        
        out = transformer(a, s, z)
        loss = out.sum()
        loss.backward()
        
        assert s.grad is not None
        assert not torch.allclose(s.grad, torch.zeros_like(s.grad))
    
    def test_gradients_to_pair(self):
        """Gradients should flow to pair"""
        transformer = DiffusionTransformer(n_blocks=2)
        
        a = torch.randn(10, 768)
        s = torch.randn(10, 384)
        z = torch.randn(10, 10, 128, requires_grad=True)
        
        out = transformer(a, s, z)
        loss = out.sum()
        loss.backward()
        
        assert z.grad is not None
        assert not torch.allclose(z.grad, torch.zeros_like(z.grad))


class TestBatchConsistency:
    """Test batch processing consistency"""
    
    def test_batch_consistency(self):
        """Batched and individual processing should match"""
        transformer = DiffusionTransformer(n_blocks=2)
        
        a_batched = torch.randn(4, 10, 768)
        s_batched = torch.randn(4, 10, 384)
        z_batched = torch.randn(4, 10, 10, 128)
        
        # Process as batch
        out_batched = transformer(a_batched, s_batched, z_batched)
        
        # Process individually
        for i in range(4):
            out_i = transformer(a_batched[i], s_batched[i], z_batched[i])
            
            # Should match
            assert torch.allclose(out_batched[i], out_i, atol=1e-5)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_token(self):
        """Should work with single token"""
        transformer = DiffusionTransformer(n_blocks=2)
        
        a = torch.randn(1, 768)
        s = torch.randn(1, 384)
        z = torch.randn(1, 1, 128)
        
        out = transformer(a, s, z)
        
        assert out.shape == (1, 768)
    
    def test_with_bias_mask(self):
        """Should accept optional bias mask"""
        transformer = DiffusionTransformer(n_blocks=2)
        
        a = torch.randn(10, 768)
        s = torch.randn(10, 384)
        z = torch.randn(10, 10, 128)
        bias_mask = torch.zeros(10, 10)  # β_ij = 0 as in Algorithm 20
        
        out = transformer(a, s, z, bias_mask)
        
        assert out.shape == (10, 768)


class TestDummyInputGeneration:
    """Test dummy input utility"""
    
    def test_unbatched_dummy(self):
        """Should generate unbatched inputs"""
        a, s, z, bias = create_dummy_diffusion_transformer_input()
        
        assert a.shape == (10, 768)
        assert s.shape == (10, 384)
        assert z.shape == (10, 10, 128)
    
    def test_batched_dummy(self):
        """Should generate batched inputs"""
        a, s, z, bias = create_dummy_diffusion_transformer_input(batch_size=4)
        
        assert a.shape == (4, 10, 768)
        assert s.shape == (4, 10, 384)
        assert z.shape == (4, 10, 10, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])