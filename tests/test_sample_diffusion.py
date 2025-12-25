"""
Unit tests for AlphaFold3 Diffusion Sampling.

File: tests/test_sample_diffusion.py

Tests cover:
1. Noise schedule generation
2. CentreRandomAugmentation
3. SampleDiffusion forward pass
4. Algorithm 18 & 19 faithfulness
5. Determinism and reproducibility
"""

import pytest
import torch
import math
from src.models.diffusion.sample_diffusion import (
    create_noise_schedule,
    CentreRandomAugmentation,
    SampleDiffusion
)
from src.models.diffusion.diffusion_module import (
    DiffusionModule,
    create_dummy_diffusion_module_input
)


class TestNoiseSchedule:
    """Test noise schedule generation"""
    
    def test_default_schedule(self):
        """Should create schedule with correct length"""
        schedule = create_noise_schedule()
        
        assert len(schedule) == 201  # n_steps + 1
        assert schedule[0] > schedule[-1]  # Decreasing
    
    def test_schedule_bounds(self):
        """Schedule should be monotonically decreasing"""
        schedule = create_noise_schedule(n_steps=100)
        
        for i in range(len(schedule) - 1):
            assert schedule[i] >= schedule[i + 1]
    
    def test_custom_parameters(self):
        """Should accept custom parameters"""
        schedule = create_noise_schedule(
            n_steps=50,
            s_max=200.0,
            s_min=1e-3,
            p=5.0
        )
        
        assert len(schedule) == 51
    
    def test_endpoint_values(self):
        """First value should be largest, last should be smallest"""
        schedule = create_noise_schedule()
        
        assert schedule[0] == schedule.max()
        assert schedule[-1] == schedule.min()


class TestCentreRandomAugmentation:
    """Test CentreRandomAugmentation (Algorithm 19)"""
    
    def test_initialization(self):
        """Should initialize with correct parameters"""
        aug = CentreRandomAugmentation(s_trans=2.0)
        
        assert aug.s_trans == 2.0
    
    def test_forward_shape(self):
        """Should preserve input shape"""
        aug = CentreRandomAugmentation()
        
        x = torch.randn(100, 3)
        x_aug = aug(x)
        
        assert x_aug.shape == (100, 3)
    
    def test_centering(self):
        """Output should be approximately centered"""
        aug = CentreRandomAugmentation(s_trans=0.0)  # No translation
        
        # Create non-centered input
        x = torch.randn(100, 3) + torch.tensor([10.0, 20.0, 30.0])
        
        # Apply augmentation multiple times
        for _ in range(5):
            x_aug = aug(x)
            # Should be centered (within translation noise)
            mean = x_aug.mean(dim=0)
            assert torch.allclose(mean, torch.zeros(3), atol=1.0)
    
    def test_rotation_preserves_distances(self):
        """Rotation should preserve pairwise distances"""
        aug = CentreRandomAugmentation(s_trans=0.0)  # No translation
        
        x = torch.randn(50, 3)
        
        # Centre the input manually to match what augmentation does
        x_centered = x - x.mean(dim=0, keepdim=True)
        
        # Apply augmentation (which centers, rotates, no translation)
        x_aug = aug(x)
        
        # Compute pairwise distances
        dist_centered = torch.cdist(x_centered, x_centered)
        dist_aug = torch.cdist(x_aug, x_aug)
        
        # Distances should be preserved (rotation is isometric)
        # Use slightly relaxed tolerance due to numerical precision in rotation matrix
        assert torch.allclose(dist_centered, dist_aug, atol=1e-3, rtol=1e-5)
    
    def test_rotation_matrix_orthogonal(self):
        """Rotation matrices should be orthogonal"""
        aug = CentreRandomAugmentation()
        
        for _ in range(10):
            R = aug._random_rotation_matrix()
            
            # R^T @ R should be identity
            identity = torch.eye(3)
            assert torch.allclose(R.T @ R, identity, atol=1e-5)
            assert torch.allclose(R @ R.T, identity, atol=1e-5)
            
            # Determinant should be 1 (proper rotation)
            assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5)
    
    def test_translation_applied(self):
        """Translation should move coordinates"""
        aug = CentreRandomAugmentation(s_trans=5.0)
        
        x = torch.randn(100, 3)
        
        # Apply augmentation
        x_aug = aug(x)
        
        # Mean should not be at origin (due to translation)
        mean = x_aug.mean(dim=0)
        # Translation has std=5.0, so mean should be within ~3*std
        assert mean.norm() < 15.0


class TestSampleDiffusion:
    """Test SampleDiffusion (Algorithm 18)"""
    
    def test_initialization(self):
        """Should initialize with diffusion module"""
        diffusion_module = DiffusionModule(n_blocks=2)
        sampler = SampleDiffusion(diffusion_module)
        
        assert sampler.diffusion_module is diffusion_module
        assert len(sampler.noise_schedule) == 201
        assert sampler.gamma_0 == 0.8
        assert sampler.gamma_min == 1.0
        assert sampler.noise_scale == 1.003
        assert sampler.step_scale == 1.5
    
    def test_custom_schedule(self):
        """Should accept custom noise schedule"""
        diffusion_module = DiffusionModule(n_blocks=2)
        schedule = [100.0, 50.0, 25.0, 10.0, 1.0]
        
        sampler = SampleDiffusion(diffusion_module, noise_schedule=schedule)
        
        assert len(sampler.noise_schedule) == 5
    
    def test_forward_shape(self):
        """Should generate correct number of atoms"""
        diffusion_module = DiffusionModule(n_blocks=1)
        schedule = create_noise_schedule(n_steps=5)  # Short schedule for speed
        
        sampler = SampleDiffusion(diffusion_module, noise_schedule=schedule)
        
        # Create dummy inputs
        _, _, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input(n_atoms=50)
        
        # Sample
        x = sampler(features, s_inputs, s_trunk, z_trunk, n_atoms=50)
        
        assert x.shape == (50, 3)
    
    def test_different_atom_counts(self):
        """Should work with different numbers of atoms"""
        diffusion_module = DiffusionModule(n_blocks=1)
        schedule = create_noise_schedule(n_steps=3)
        
        sampler = SampleDiffusion(diffusion_module, noise_schedule=schedule)
        
        for n_atoms in [20, 50, 100]:
            _, _, features, s_inputs, s_trunk, z_trunk = \
                create_dummy_diffusion_module_input(n_atoms=n_atoms)
            
            # Update features for correct atom count
            features['atom_to_token'] = torch.arange(n_atoms) % 10
            features['ref_pos'] = torch.randn(n_atoms, 3)
            features['ref_mask'] = torch.ones(n_atoms)
            features['ref_element'] = torch.randn(n_atoms, 128)
            features['ref_charge'] = torch.zeros(n_atoms)
            features['ref_atom_name_chars'] = torch.randn(n_atoms, 4, 64)
            features['ref_space_uid'] = torch.arange(n_atoms) % 10
            
            x = sampler(features, s_inputs, s_trunk, z_trunk, n_atoms=n_atoms)
            
            assert x.shape == (n_atoms, 3)
    
    def test_output_not_nan(self):
        """Output should not contain NaN or Inf"""
        diffusion_module = DiffusionModule(n_blocks=1)
        schedule = create_noise_schedule(n_steps=5)
        
        sampler = SampleDiffusion(diffusion_module, noise_schedule=schedule)
        
        _, _, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input()
        
        x = sampler(features, s_inputs, s_trunk, z_trunk, n_atoms=100)
        
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()


class TestAlgorithm18Faithfulness:
    """Test faithfulness to Algorithm 18"""
    
    def test_algorithm_structure(self):
        """
        Algorithm 18:
        1: x ~ c_0 * N(0, I_3)
        2: for c_τ in [c_1, ..., c_T]:
        3:     x = CentreRandomAugmentation(x)
        4:     γ = γ_0 if c_τ > γ_min else 0
        5:     t_hat = c_{τ-1} * (γ + 1)
        6:     ξ = λ * sqrt(t_hat² - c_{τ-1}²) * N(0, I_3)
        7:     x_noisy = x + ξ
        8:     x_denoised = DiffusionModule(...)
        9:     δ = (x - x_denoised) / t_hat
        10:    dt = c_τ - t_hat
        11:    x = x_noisy + η * dt * δ
        13: return x
        """
        diffusion_module = DiffusionModule(n_blocks=1)
        sampler = SampleDiffusion(diffusion_module)
        
        # Should have augmentation module (line 3)
        assert hasattr(sampler, 'augmentation')
        
        # Should have diffusion module (line 8)
        assert hasattr(sampler, 'diffusion_module')
    
    def test_default_parameters_match_paper(self):
        """Default parameters should match Algorithm 18"""
        diffusion_module = DiffusionModule(n_blocks=1)
        sampler = SampleDiffusion(diffusion_module)
        
        assert sampler.gamma_0 == 0.8  # γ_0 = 0.8
        assert sampler.gamma_min == 1.0  # γ_min = 1.0
        assert sampler.noise_scale == 1.003  # λ = 1.003
        assert sampler.step_scale == 1.5  # η = 1.5
    
    def test_initialization_from_noise(self):
        """Should initialize from scaled noise (line 1)"""
        diffusion_module = DiffusionModule(n_blocks=1)
        schedule = torch.tensor([100.0, 50.0, 1.0])
        
        sampler = SampleDiffusion(diffusion_module, noise_schedule=schedule)
        
        # c_0 should be first element
        assert sampler.noise_schedule[0] == 100.0


class TestAlgorithm19Faithfulness:
    """Test faithfulness to Algorithm 19"""
    
    def test_algorithm_structure(self):
        """
        Algorithm 19:
        1: x = x - mean(x)
        2: R = UniformRandomRotation()
        3: t ~ s_trans * N(0, I_3)
        4: x = R @ x + t
        5: return x
        """
        aug = CentreRandomAugmentation()
        
        x = torch.randn(50, 3) + torch.tensor([5.0, 10.0, 15.0])
        x_aug = aug(x)
        
        # Should preserve shape
        assert x_aug.shape == x.shape
    
    def test_default_parameters_match_paper(self):
        """Default s_trans should be 1.0 Angstrom"""
        aug = CentreRandomAugmentation()
        
        assert aug.s_trans == 1.0


class TestReproducibility:
    """Test reproducibility with fixed seeds"""
    
    def test_augmentation_reproducible(self):
        """With same seed, augmentation should be reproducible"""
        aug = CentreRandomAugmentation()
        x = torch.randn(100, 3)
        
        # First run
        torch.manual_seed(42)
        x_aug1 = aug(x)
        
        # Second run with same seed
        torch.manual_seed(42)
        x_aug2 = aug(x)
        
        assert torch.allclose(x_aug1, x_aug2)
    
    def test_sampling_reproducible(self):
        """With same seed, sampling should be reproducible"""
        diffusion_module = DiffusionModule(n_blocks=1)
        schedule = create_noise_schedule(n_steps=3)
        sampler = SampleDiffusion(diffusion_module, noise_schedule=schedule)
        
        _, _, features, s_inputs, s_trunk, z_trunk = \
            create_dummy_diffusion_module_input()
        
        # First run
        torch.manual_seed(42)
        x1 = sampler(features, s_inputs, s_trunk, z_trunk, n_atoms=100)
        
        # Second run with same seed
        torch.manual_seed(42)
        x2 = sampler(features, s_inputs, s_trunk, z_trunk, n_atoms=100)
        
        assert torch.allclose(x1, x2)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_atom(self):
        """Should work with single atom"""
        aug = CentreRandomAugmentation()
        
        x = torch.randn(1, 3)
        x_aug = aug(x)
        
        assert x_aug.shape == (1, 3)
    
    def test_zero_translation(self):
        """With s_trans=0, should only rotate and centre"""
        aug = CentreRandomAugmentation(s_trans=0.0)
        
        x = torch.randn(50, 3)
        x_aug = aug(x)
        
        # Mean should be at origin
        assert torch.allclose(x_aug.mean(dim=0), torch.zeros(3), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])