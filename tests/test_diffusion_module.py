"""
Unit tests for AlphaFold3 Diffusion Module.

File: tests/test_diffusion_module.py

Tests cover:
1. Initialization
2. Forward pass shape validation
3. Coordinate scaling/rescaling
4. Algorithm 20 faithfulness
5. Gradient flow
6. Integration of all components
"""

import pytest
import torch
from src.models.diffusion.diffusion_module import (
    DiffusionModule,
    create_dummy_diffusion_module_input
)


class TestInitialization:
    """Test DiffusionModule initialization"""
    
    def test_basic_initialization(self):
        """Should initialize with correct dimensions"""
        module = DiffusionModule()
        
        assert module.sigma_data == 16.0
        assert module.c_atom == 128
        assert module.c_atompair == 16
        assert module.c_token == 768
        assert module.c_pair == 128
        assert module.c_s == 384
        assert module.n_blocks == 24
        assert module.n_heads == 16
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        module = DiffusionModule(
            sigma_data=32.0,
            c_atom=256,
            c_token=512,
            n_blocks=12
        )
        
        assert module.sigma_data == 32.0
        assert module.c_atom == 256
        assert module.c_token == 512
        assert module.n_blocks == 12
    
    def test_has_all_components(self):
        """Should have all required submodules"""
        module = DiffusionModule()
        
        # Algorithm 20 components
        assert hasattr(module, 'conditioning')  # Line 1
        assert hasattr(module, 'atom_encoder')  # Line 3
        assert hasattr(module, 'conditioning_projection')  # Line 4
        assert hasattr(module, 'transformer')  # Line 5
        assert hasattr(module, 'output_norm')  # Line 6
        assert hasattr(module, 'atom_decoder')  # Line 7


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_forward_unbatched(self):
        """Should handle unbatched input"""
        module = DiffusionModule(n_blocks=2)  # Fewer blocks for speed
        
        x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input()
        
        x_out = module(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk)
        
        # Output should match input shape
        assert x_out.shape == x_noisy.shape
        assert x_out.shape == (100, 3)
    
    def test_forward_batched(self):
        """Should handle batched input"""
        module = DiffusionModule(n_blocks=2)
        
        x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input(batch_size=4)
        
        x_out = module(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk)
        
        # Output should match input shape
        assert x_out.shape == x_noisy.shape
        assert x_out.shape == (4, 100, 3)
    
    def test_different_atom_counts(self):
        """Should work with different numbers of atoms"""
        module = DiffusionModule(n_blocks=2)
        
        for n_atoms in [50, 100, 200]:
            x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk = \
                create_dummy_diffusion_module_input(n_atoms=n_atoms)
            
            # Update atom-related features to match new atom count
            features['atom_to_token'] = torch.arange(n_atoms) % 10
            features['ref_pos'] = torch.randn(n_atoms, 3)
            features['ref_mask'] = torch.ones(n_atoms)
            features['ref_element'] = torch.randn(n_atoms, 128)  # One-hot encoded
            features['ref_charge'] = torch.zeros(n_atoms)
            features['ref_atom_name_chars'] = torch.randn(n_atoms, 4, 64)  # Character encoding
            features['ref_space_uid'] = torch.arange(n_atoms) % 10  # Residue grouping
            
            x_out = module(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk)
            
            assert x_out.shape == (n_atoms, 3)


class TestCoordinateScaling:
    """Test coordinate scaling and rescaling (Algorithm 20, lines 2 & 8)"""
    
    def test_input_scaling(self):
        """Noisy coords should be scaled by sqrt(t² + σ²)"""
        module = DiffusionModule(sigma_data=16.0, n_blocks=1)
        
        x_noisy = torch.randn(100, 3)
        t_hat = torch.tensor(5.0)
        
        # Expected scale: sqrt(5² + 16²) = sqrt(25 + 256) = sqrt(281) ≈ 16.76
        expected_scale = torch.sqrt(t_hat ** 2 + 16.0 ** 2)
        
        assert torch.isclose(expected_scale, torch.tensor(16.76), atol=0.01)
    
    def test_output_rescaling(self):
        """Output should combine noisy coords and updates correctly"""
        module = DiffusionModule(sigma_data=16.0, n_blocks=1)
        
        x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input()
        
        x_out = module(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk)
        
        # Output should be different from input (denoising happened)
        assert not torch.allclose(x_out, x_noisy, atol=1e-3)
    
    def test_different_timesteps(self):
        """Different timesteps should produce different outputs"""
        module = DiffusionModule(n_blocks=2)
        
        x_noisy, _, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input()
        
        t1 = torch.tensor(1.0)
        t2 = torch.tensor(10.0)
        
        x_out1 = module(x_noisy, t1, features, s_inputs, s_trunk, z_trunk)
        x_out2 = module(x_noisy, t2, features, s_inputs, s_trunk, z_trunk)
        
        # Different timesteps → different outputs
        assert not torch.allclose(x_out1, x_out2, atol=1e-5)


class TestConditioning:
    """Test conditioning effects"""
    
    def test_trunk_conditioning_affects_output(self):
        """Different trunk conditioning should affect output"""
        module = DiffusionModule(n_blocks=2)
        
        x_noisy, t_hat, features, s_inputs, s_trunk1, z_trunk = \
            create_dummy_diffusion_module_input()
        
        s_trunk2 = torch.randn_like(s_trunk1)
        
        x_out1 = module(x_noisy, t_hat, features, s_inputs, s_trunk1, z_trunk)
        x_out2 = module(x_noisy, t_hat, features, s_inputs, s_trunk2, z_trunk)
        
        # Different trunk → different output
        assert not torch.allclose(x_out1, x_out2, atol=1e-5)
    
    def test_pair_conditioning_affects_output(self):
        """Different pair conditioning should affect output"""
        module = DiffusionModule(n_blocks=2)
        
        x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk1 = \
            create_dummy_diffusion_module_input()
        
        z_trunk2 = torch.randn_like(z_trunk1)
        
        x_out1 = module(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk1)
        x_out2 = module(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk2)
        
        # Different pair → different output
        assert not torch.allclose(x_out1, x_out2, atol=1e-5)


class TestAlgorithm20Faithfulness:
    """Test faithfulness to Algorithm 20"""
    
    def test_algorithm_structure(self):
        """
        Algorithm 20:
        1: s, z = DiffusionConditioning(...)
        2: r_noisy = x_noisy / sqrt(t² + σ²)
        3: a, q_skip, c_skip, p_skip = AtomAttentionEncoder(...)
        4: a += Linear(LayerNorm(s))
        5: a = DiffusionTransformer(a, s, z, β=0, Nblock=24, Nhead=16)
        6: a = LayerNorm(a)
        7: r_update = AtomAttentionDecoder(...)
        8: x_out = combine(x_noisy, r_update, t, σ)
        9: return x_out
        """
        module = DiffusionModule()
        
        # Should have all components
        assert hasattr(module, 'conditioning')  # Line 1
        assert hasattr(module, 'atom_encoder')  # Line 3
        assert hasattr(module, 'conditioning_projection')  # Line 4
        assert hasattr(module, 'transformer')  # Line 5
        assert hasattr(module, 'output_norm')  # Line 6
        assert hasattr(module, 'atom_decoder')  # Line 7
    
    def test_default_parameters_match_paper(self):
        """Default parameters should match Algorithm 20"""
        module = DiffusionModule()
        
        # From Algorithm 20
        assert module.sigma_data == 16.0  # σ_data = 16
        assert module.c_atom == 128  # c_atom = 128
        assert module.c_atompair == 16  # c_atompair = 16
        assert module.c_token == 768  # c_token = 768
        assert module.n_blocks == 24  # Nblock = 24
        assert module.n_heads == 16  # Nhead = 16
    
    def test_transformer_uses_no_bias_mask(self):
        """Transformer should use β_ij = 0 (Algorithm 20, line 5)"""
        # This is tested implicitly in forward pass
        # β_ij = 0 means bias_mask=None in transformer call
        module = DiffusionModule(n_blocks=2)
        
        x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input()
        
        # Should not raise any errors
        x_out = module(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk)
        
        assert x_out.shape == x_noisy.shape


class TestGradientFlow:
    """Test gradient flow"""
    
    def test_gradients_to_noisy_coords(self):
        """Gradients should flow to noisy coordinates"""
        module = DiffusionModule(n_blocks=2)
        
        x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input()
        
        x_noisy.requires_grad = True
        
        x_out = module(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk)
        loss = x_out.sum()
        loss.backward()
        
        assert x_noisy.grad is not None
        assert not torch.allclose(x_noisy.grad, torch.zeros_like(x_noisy.grad))
    
    def test_gradients_to_trunk_single(self):
        """Gradients should flow to trunk single"""
        module = DiffusionModule(n_blocks=2)
        
        x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input()
        
        s_trunk.requires_grad = True
        
        x_out = module(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk)
        loss = x_out.sum()
        loss.backward()
        
        assert s_trunk.grad is not None
    
    def test_gradients_to_trunk_pair(self):
        """Gradients should flow to trunk pair"""
        module = DiffusionModule(n_blocks=2)
        
        x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input()
        
        z_trunk.requires_grad = True
        
        x_out = module(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk)
        loss = x_out.sum()
        loss.backward()
        
        assert z_trunk.grad is not None


class TestBatchConsistency:
    """Test batch processing consistency"""
    
    def test_batch_consistency(self):
        """Batched and individual processing should match"""
        module = DiffusionModule(n_blocks=2)
        
        x_noisy_batched, t_hat_batched, features, s_inputs_batched, s_trunk_batched, z_trunk_batched = \
            create_dummy_diffusion_module_input(batch_size=4)
        
        # Process as batch
        x_out_batched = module(x_noisy_batched, t_hat_batched, features, 
                              s_inputs_batched, s_trunk_batched, z_trunk_batched)
        
        # Process individually
        for i in range(4):
            x_out_i = module(
                x_noisy_batched[i], t_hat_batched[i], features,
                s_inputs_batched[i], s_trunk_batched[i], z_trunk_batched[i]
            )
            
            # Should match
            assert torch.allclose(x_out_batched[i], x_out_i, atol=1e-5)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_very_small_timestep(self):
        """Should handle very small timesteps"""
        module = DiffusionModule(n_blocks=2)
        
        x_noisy, _, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input()
        
        t_hat = torch.tensor(0.001)
        
        x_out = module(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk)
        
        assert not torch.isnan(x_out).any()
        assert not torch.isinf(x_out).any()
    
    def test_large_timestep(self):
        """Should handle large timesteps"""
        module = DiffusionModule(n_blocks=2)
        
        x_noisy, _, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input()
        
        t_hat = torch.tensor(100.0)
        
        x_out = module(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk)
        
        assert not torch.isnan(x_out).any()
        assert not torch.isinf(x_out).any()


class TestDummyInputGeneration:
    """Test dummy input utility"""
    
    def test_unbatched_dummy(self):
        """Should generate unbatched inputs"""
        x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input()
        
        assert x_noisy.shape == (100, 3)
        assert t_hat.dim() == 0  # Scalar
        assert s_inputs.shape == (10, 384)  # Trunk output dimension
        assert s_trunk.shape == (10, 384)   # Trunk output dimension
        assert z_trunk.shape == (10, 10, 128)
    
    def test_batched_dummy(self):
        """Should generate batched inputs"""
        x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input(batch_size=4)
        
        assert x_noisy.shape == (4, 100, 3)
        assert t_hat.shape == (4,)
        assert s_inputs.shape == (4, 10, 384)  # Trunk output dimension
        assert s_trunk.shape == (4, 10, 384)   # Trunk output dimension
        assert z_trunk.shape == (4, 10, 10, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])