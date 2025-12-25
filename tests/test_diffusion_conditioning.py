"""
Unit tests for AlphaFold3 Diffusion Conditioning.

File: tests/test_diffusion_conditioning.py

Tests cover:
1. Initialization
2. Forward pass shape validation
3. Pair conditioning pathway
4. Single conditioning pathway
5. Timestep embedding integration
6. Algorithm 21 faithfulness
"""

import pytest
import torch
from src.models.diffusion.diffusion_conditioning import (
    DiffusionConditioning,
    create_dummy_diffusion_conditioning_input
)


class TestInitialization:
    """Test DiffusionConditioning initialization"""
    
    def test_basic_initialization(self):
        """Should initialize with correct dimensions"""
        conditioning = DiffusionConditioning()
        
        assert conditioning.c_token == 384
        assert conditioning.c_pair_trunk == 128
        assert conditioning.c_single == 384
        assert conditioning.c_pair == 128
        assert conditioning.n_transition == 2
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        conditioning = DiffusionConditioning(
            c_token=512,
            c_pair_trunk=256,
            c_single=768,
            c_pair=256,
            n_transition=4
        )
        
        assert conditioning.c_token == 512
        assert conditioning.c_pair_trunk == 256
        assert conditioning.c_single == 768
        assert conditioning.c_pair == 256
        assert conditioning.n_transition == 4
    
    def test_has_required_components(self):
        """Should have all required components"""
        conditioning = DiffusionConditioning()
        
        # Pair components
        assert hasattr(conditioning, 'relative_position_encoding')
        assert hasattr(conditioning, 'pair_projection')
        assert hasattr(conditioning, 'pair_transitions')
        assert len(conditioning.pair_transitions) == 2
        
        # Single components
        assert hasattr(conditioning, 'single_projection')
        assert hasattr(conditioning, 'fourier_embedding')
        assert hasattr(conditioning, 'timestep_projection')
        assert hasattr(conditioning, 'single_transitions')
        assert len(conditioning.single_transitions) == 2


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_basic_forward_unbatched(self):
        """Should handle unbatched input"""
        conditioning = DiffusionConditioning()
        
        s_inputs, s_trunk, z_trunk, t, features = \
            create_dummy_diffusion_conditioning_input()
        
        s_cond, z_cond = conditioning(s_inputs, s_trunk, z_trunk, t, features)
        
        assert s_cond.shape == (10, 384)  # [N_token, c_single]
        assert z_cond.shape == (10, 10, 128)  # [N_token, N_token, c_pair]
    
    def test_basic_forward_batched(self):
        """Should handle batched input"""
        conditioning = DiffusionConditioning()
        
        s_inputs, s_trunk, z_trunk, t, features = \
            create_dummy_diffusion_conditioning_input(batch_size=4)
        
        s_cond, z_cond = conditioning(s_inputs, s_trunk, z_trunk, t, features)
        
        assert s_cond.shape == (4, 10, 384)  # [batch, N_token, c_single]
        assert z_cond.shape == (4, 10, 10, 128)  # [batch, N_token, N_token, c_pair]
    
    def test_different_token_counts(self):
        """Should work with different numbers of tokens"""
        conditioning = DiffusionConditioning()
        
        for n_token in [5, 10, 20, 50]:
            s_inputs, s_trunk, z_trunk, t, features = \
                create_dummy_diffusion_conditioning_input(n_token=n_token)
            
            s_cond, z_cond = conditioning(s_inputs, s_trunk, z_trunk, t, features)
            
            assert s_cond.shape == (n_token, 384)
            assert z_cond.shape == (n_token, n_token, 128)


class TestPairConditioning:
    """Test pair conditioning pathway"""
    
    def test_includes_relative_positions(self):
        """Pair conditioning should include relative positions"""
        conditioning = DiffusionConditioning()
        
        s_inputs, s_trunk, z_trunk, t, features = \
            create_dummy_diffusion_conditioning_input()
        
        # Get relative position encoding
        rel_pos = conditioning.relative_position_encoding(features)
        
        # Should have correct shape
        assert rel_pos.shape == (10, 10, 128)  # [N_token, N_token, c_pair_trunk]
    
    def test_pair_transitions_applied(self):
        """Should apply transition blocks to pair"""
        conditioning = DiffusionConditioning()
        
        s_inputs, s_trunk, z_trunk, t, features = \
            create_dummy_diffusion_conditioning_input()
        
        _, z_cond = conditioning(s_inputs, s_trunk, z_trunk, t, features)
        
        # Different trunk pair → different conditioned pair
        z_trunk_2 = torch.randn_like(z_trunk)
        _, z_cond_2 = conditioning(s_inputs, s_trunk, z_trunk_2, t, features)
        
        assert not torch.allclose(z_cond, z_cond_2, atol=1e-5)
    
    def test_pair_output_dimension(self):
        """Pair output should have correct dimension"""
        conditioning = DiffusionConditioning(
            c_pair_trunk=128,
            c_pair=256  # Different output dimension
        )
        
        s_inputs, s_trunk, z_trunk, t, features = \
            create_dummy_diffusion_conditioning_input()
        
        _, z_cond = conditioning(s_inputs, s_trunk, z_trunk, t, features)
        
        # Output should be c_pair=256, not c_pair_trunk=128
        assert z_cond.shape[-1] == 256


class TestSingleConditioning:
    """Test single conditioning pathway"""
    
    def test_combines_trunk_and_inputs(self):
        """Should combine trunk and input single representations"""
        conditioning = DiffusionConditioning()
        
        s_inputs, s_trunk, z_trunk, t, features = \
            create_dummy_diffusion_conditioning_input()
        
        # Different inputs → different output
        s_inputs_2 = torch.randn_like(s_inputs)
        
        s_cond_1, _ = conditioning(s_inputs, s_trunk, z_trunk, t, features)
        s_cond_2, _ = conditioning(s_inputs_2, s_trunk, z_trunk, t, features)
        
        assert not torch.allclose(s_cond_1, s_cond_2, atol=1e-5)
    
    def test_single_transitions_applied(self):
        """Should apply transition blocks to single"""
        conditioning = DiffusionConditioning()
        
        s_inputs, s_trunk, z_trunk, t, features = \
            create_dummy_diffusion_conditioning_input()
        
        s_cond, _ = conditioning(s_inputs, s_trunk, z_trunk, t, features)
        
        # Different trunk single → different conditioned single
        s_trunk_2 = torch.randn_like(s_trunk)
        s_cond_2, _ = conditioning(s_inputs, s_trunk_2, z_trunk, t, features)
        
        assert not torch.allclose(s_cond, s_cond_2, atol=1e-5)
    
    def test_single_output_dimension(self):
        """Single output should have correct dimension"""
        conditioning = DiffusionConditioning(
            c_token=384,
            c_single=768  # Different output dimension
        )
        
        s_inputs, s_trunk, z_trunk, t, features = \
            create_dummy_diffusion_conditioning_input()
        
        s_cond, _ = conditioning(s_inputs, s_trunk, z_trunk, t, features)
        
        # Output should be c_single=768, not c_token=384
        assert s_cond.shape[-1] == 768


class TestTimestepEmbedding:
    """Test timestep embedding integration"""
    
    def test_different_timesteps_different_outputs(self):
        """Different timesteps should produce different outputs"""
        conditioning = DiffusionConditioning()
        
        s_inputs, s_trunk, z_trunk, t1, features = \
            create_dummy_diffusion_conditioning_input()
        
        t2 = torch.tensor(10.0)  # Different timestep
        
        s_cond_1, _ = conditioning(s_inputs, s_trunk, z_trunk, t1, features)
        s_cond_2, _ = conditioning(s_inputs, s_trunk, z_trunk, t2, features)
        
        # Different timesteps → different single conditioning
        assert not torch.allclose(s_cond_1, s_cond_2, atol=1e-5)
    
    def test_timestep_preprocessing(self):
        """Timestep should be preprocessed: 0.25 * log(t/sigma_data)"""
        conditioning = DiffusionConditioning()
        
        t = torch.tensor(5.0)
        sigma_data = 16.0
        
        # Expected preprocessing
        t_expected = 0.25 * torch.log(t / sigma_data)
        
        # Embed
        n = conditioning.fourier_embedding(t_expected)
        
        assert n.shape == (256,)  # Fourier embedding dimension
    
    def test_timestep_affects_only_single(self):
        """Timestep should only affect single, not pair"""
        conditioning = DiffusionConditioning()
        
        s_inputs, s_trunk, z_trunk, t1, features = \
            create_dummy_diffusion_conditioning_input()
        
        t2 = torch.tensor(10.0)
        
        s_cond_1, z_cond_1 = conditioning(s_inputs, s_trunk, z_trunk, t1, features)
        s_cond_2, z_cond_2 = conditioning(s_inputs, s_trunk, z_trunk, t2, features)
        
        # Single should differ
        assert not torch.allclose(s_cond_1, s_cond_2, atol=1e-5)
        
        # Pair should be identical (no timestep dependence)
        assert torch.allclose(z_cond_1, z_cond_2, atol=1e-6)


class TestAlgorithm21Faithfulness:
    """Test faithfulness to Algorithm 21"""
    
    def test_pair_pathway_structure(self):
        """
        Pair pathway (lines 1-5):
        1: z = concat([z_trunk, RelativePositionEncoding])
        2: z ← LinearNoBias(LayerNorm(z))
        3-5: for b in [1,2]: z += Transition(z)
        """
        conditioning = DiffusionConditioning(n_transition=2)
        
        # Should have 2 pair transitions
        assert len(conditioning.pair_transitions) == 2
    
    def test_single_pathway_structure(self):
        """
        Single pathway (lines 6-12):
        6: s = concat([s_trunk, s_inputs])
        7: s ← LinearNoBias(LayerNorm(s))
        8: n = FourierEmbedding(0.25 * log(t/sigma_data), 256)
        9: s += LinearNoBias(LayerNorm(n))
        10-12: for b in [1,2]: s += Transition(s)
        """
        conditioning = DiffusionConditioning(n_transition=2)
        
        # Should have 2 single transitions
        assert len(conditioning.single_transitions) == 2
        
        # Should have FourierEmbedding with dim=256
        assert conditioning.fourier_embedding.dim == 256
    
    def test_default_parameters_match_paper(self):
        """Default parameters should match paper"""
        conditioning = DiffusionConditioning()
        
        # From Algorithm 21
        assert conditioning.c_pair == 128  # c_z = 128
        assert conditioning.c_single == 384  # c_s = 384
        assert conditioning.n_transition == 2  # 2 blocks each


class TestEdgeCases:
    """Test edge cases"""
    
    def test_zero_timestep(self):
        """Should handle very small timesteps"""
        conditioning = DiffusionConditioning()
        
        s_inputs, s_trunk, z_trunk, _, features = \
            create_dummy_diffusion_conditioning_input()
        
        t = torch.tensor(0.001)  # Very small timestep
        
        s_cond, z_cond = conditioning(s_inputs, s_trunk, z_trunk, t, features)
        
        assert not torch.isnan(s_cond).any()
        assert not torch.isnan(z_cond).any()
    
    def test_large_timestep(self):
        """Should handle large timesteps"""
        conditioning = DiffusionConditioning()
        
        s_inputs, s_trunk, z_trunk, _, features = \
            create_dummy_diffusion_conditioning_input()
        
        t = torch.tensor(1000.0)  # Large timestep
        
        s_cond, z_cond = conditioning(s_inputs, s_trunk, z_trunk, t, features)
        
        assert not torch.isnan(s_cond).any()
        assert not torch.isnan(z_cond).any()


class TestGradientFlow:
    """Test gradient flow"""
    
    def test_gradients_to_inputs(self):
        """Gradients should flow to all inputs"""
        conditioning = DiffusionConditioning()
        
        s_inputs, s_trunk, z_trunk, t, features = \
            create_dummy_diffusion_conditioning_input()
        
        s_inputs.requires_grad = True
        s_trunk.requires_grad = True
        z_trunk.requires_grad = True
        
        s_cond, z_cond = conditioning(s_inputs, s_trunk, z_trunk, t, features)
        
        loss = s_cond.sum() + z_cond.sum()
        loss.backward()
        
        assert s_inputs.grad is not None
        assert s_trunk.grad is not None
        assert z_trunk.grad is not None


class TestBatchConsistency:
    """Test batch processing consistency"""
    
    def test_batch_consistency(self):
        """Batched and individual processing should match"""
        conditioning = DiffusionConditioning()
        
        # Create batched input
        s_inputs_batched = torch.randn(4, 10, 384)
        s_trunk_batched = torch.randn(4, 10, 384)
        z_trunk_batched = torch.randn(4, 10, 10, 128)
        t_batched = torch.rand(4) * 10
        
        features = {
            'residue_index': torch.arange(10),
            'token_index': torch.arange(10),
            'asym_id': torch.zeros(10, dtype=torch.long),
            'entity_id': torch.zeros(10, dtype=torch.long),
            'sym_id': torch.zeros(10, dtype=torch.long),
        }
        
        # Process as batch
        s_batched, z_batched = conditioning(
            s_inputs_batched, s_trunk_batched, z_trunk_batched, 
            t_batched, features
        )
        
        # Process individually
        for i in range(4):
            s_i, z_i = conditioning(
                s_inputs_batched[i], s_trunk_batched[i], z_trunk_batched[i],
                t_batched[i], features
            )
            
            # Should match
            assert torch.allclose(s_batched[i], s_i, atol=1e-5)
            assert torch.allclose(z_batched[i], z_i, atol=1e-5)


class TestDummyInputGeneration:
    """Test dummy input utility"""
    
    def test_unbatched_dummy(self):
        """Should generate unbatched dummy inputs"""
        s_inputs, s_trunk, z_trunk, t, features = \
            create_dummy_diffusion_conditioning_input()
        
        assert s_inputs.shape == (10, 384)
        assert s_trunk.shape == (10, 384)
        assert z_trunk.shape == (10, 10, 128)
        assert t.dim() == 0  # Scalar
    
    def test_batched_dummy(self):
        """Should generate batched dummy inputs"""
        s_inputs, s_trunk, z_trunk, t, features = \
            create_dummy_diffusion_conditioning_input(batch_size=4)
        
        assert s_inputs.shape == (4, 10, 384)
        assert s_trunk.shape == (4, 10, 384)
        assert z_trunk.shape == (4, 10, 10, 128)
        assert t.shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])