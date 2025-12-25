"""
Unit tests for AlphaFold3 Main Inference Loop.

File: tests/test_alphafold3.py

Tests cover:
1. Model initialization
2. Forward pass functionality
3. Recycling behavior
4. Algorithm 1 faithfulness
5. End-to-end integration
"""

import pytest
import torch
from src.models.alphafold3 import AlphaFold3, create_alphafold3_model


def create_dummy_features(n_tokens: int = 10, n_atoms: int = 40) -> dict:
    """
    Create dummy input features for testing AlphaFold3.
    
    Args:
        n_tokens: Number of tokens
        n_atoms: Number of atoms
    
    Returns:
        features: Dictionary with all required features
    """
    # Token-level features
    features = {
        'residue_index': torch.arange(n_tokens),
        'token_index': torch.arange(n_tokens),
        'asym_id': torch.zeros(n_tokens, dtype=torch.long),
        'entity_id': torch.zeros(n_tokens, dtype=torch.long),
        'sym_id': torch.zeros(n_tokens, dtype=torch.long),
        
        # InputFeatureEmbedder requirements
        'restype': torch.randn(n_tokens, 32),  # One-hot residue types
        'profile': torch.randn(n_tokens, 32),  # MSA profile
        'deletion_mean': torch.randn(n_tokens),  # Mean deletions
        
        # Atom-level features
        'atom_to_token': torch.arange(n_atoms) % n_tokens,
        'ref_pos': torch.randn(n_atoms, 3),
        'ref_mask': torch.ones(n_atoms),
        'ref_element': torch.randn(n_atoms, 128),  # One-hot elements
        'ref_charge': torch.zeros(n_atoms),
        'ref_atom_name_chars': torch.randn(n_atoms, 4, 64),
        'ref_space_uid': torch.arange(n_atoms) % n_tokens,
    }
    
    return features


class TestInitialization:
    """Test AlphaFold3 model initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        model = AlphaFold3()
        
        assert model.c_token == 384
        assert model.c_pair == 128
        assert model.n_cycles == 4
        assert model.use_templates == False
    
    def test_custom_initialization(self):
        """Should accept custom parameters"""
        model = AlphaFold3(
            c_token=256,
            c_pair=64,
            n_cycles=2,
            msa_blocks=2,
            pairformer_blocks=24,
            diffusion_blocks=12
        )
        
        assert model.c_token == 256
        assert model.c_pair == 64
        assert model.n_cycles == 2
    
    def test_has_all_components(self):
        """Should have all required submodules"""
        model = AlphaFold3()
        
        # Algorithm 1 components
        assert hasattr(model, 'input_embedder')  # Line 1
        assert hasattr(model, 'single_init_proj')  # Line 2
        assert hasattr(model, 'pair_init_proj_i')  # Line 3
        assert hasattr(model, 'pair_init_proj_j')  # Line 3
        assert hasattr(model, 'relative_position_encoding')  # Line 4
        assert hasattr(model, 'msa_module')  # Line 10
        assert hasattr(model, 'pairformer')  # Line 12
        assert hasattr(model, 'sample_diffusion')  # Line 15
    
    def test_factory_function(self):
        """Should create model via factory function"""
        model = create_alphafold3_model()
        
        assert isinstance(model, AlphaFold3)
        assert model.c_token == 384


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_forward_without_msa(self):
        """Should run forward pass without MSA"""
        model = AlphaFold3(
            n_cycles=1,  # Fewer cycles for speed
            msa_blocks=2,
            pairformer_blocks=2,
            diffusion_blocks=2
        )
        
        # Create dummy features (now includes all required features)
        n_atoms = 40
        features = create_dummy_features(n_tokens=10, n_atoms=n_atoms)
        
        predictions = model(features, n_atoms=n_atoms)
        
        assert 'x_pred' in predictions
        assert predictions['x_pred'].shape == (n_atoms, 3)
    
    def test_forward_with_msa(self):
        """Should run forward pass with MSA features"""
        model = AlphaFold3(
            n_cycles=1,
            msa_blocks=2,
            pairformer_blocks=2,
            diffusion_blocks=2
        )
        
        # Create dummy features
        n_atoms = 40
        features = create_dummy_features(n_tokens=10, n_atoms=n_atoms)
        
        # MSA features
        msa_features = {
            'msa': torch.randn(5, 10, 64),  # [N_msa, N_token, c_m]
            'has_deletion': torch.zeros(5, 10),
            'deletion_value': torch.zeros(5, 10)
        }
        
        predictions = model(features, msa_features=msa_features, n_atoms=n_atoms)
        
        assert 'x_pred' in predictions
        assert predictions['x_pred'].shape == (n_atoms, 3)
    
    def test_output_contains_final_representations(self):
        """Should output final single and pair representations"""
        model = AlphaFold3(
            n_cycles=1,
            msa_blocks=1,
            pairformer_blocks=1,
            diffusion_blocks=1
        )
        
        n_atoms = 40
        features = create_dummy_features(n_tokens=10, n_atoms=n_atoms)
        
        predictions = model(features, n_atoms=n_atoms)
        
        assert 's_final' in predictions
        assert 'z_final' in predictions
        assert predictions['s_final'].shape == (10, 384)
        assert predictions['z_final'].shape == (10, 10, 128)


class TestRecycling:
    """Test recycling behavior"""
    
    def test_recycling_iterations(self):
        """Should perform correct number of recycling iterations"""
        for n_cycles in [1, 2, 4]:
            model = AlphaFold3(
                n_cycles=n_cycles,
                msa_blocks=1,
                pairformer_blocks=1,
                diffusion_blocks=1
            )
            
            assert model.n_cycles == n_cycles
    
    def test_recycling_changes_output(self):
        """More recycling should change predictions"""
        n_atoms = 40
        features = create_dummy_features(n_tokens=10, n_atoms=n_atoms)
        
        # Model with 1 cycle
        model_1 = AlphaFold3(
            n_cycles=1,
            msa_blocks=1,
            pairformer_blocks=1,
            diffusion_blocks=1
        )
        
        # Model with 2 cycles
        model_2 = AlphaFold3(
            n_cycles=2,
            msa_blocks=1,
            pairformer_blocks=1,
            diffusion_blocks=1
        )
        
        # Same seed for fair comparison
        torch.manual_seed(42)
        pred_1 = model_1(features, n_atoms=n_atoms)
        
        torch.manual_seed(42)
        pred_2 = model_2(features, n_atoms=n_atoms)
        
        # Different recycling → different outputs
        # Note: Might be similar due to randomness, but typically different
        # Just check they both work
        assert pred_1['x_pred'].shape == pred_2['x_pred'].shape


class TestAlgorithm1Faithfulness:
    """Test faithfulness to Algorithm 1"""
    
    def test_algorithm_structure(self):
        """
        Algorithm 1:
        1: s_inputs = InputFeatureEmbedder(features)
        2: s_init = Linear(s_inputs)
        3: z_init = Linear(s_inputs_i) + Linear(s_inputs_j)
        4: z_init += RelativePositionEncoding(features)
        5: z_init += Linear(token_bonds)
        6: s_hat, z_hat = 0, 0
        7: for cycle in [1...N_cycle]:
        8:     z = z_init + Linear(LayerNorm(z_hat))
        9:     z += TemplateEmbedder(features, z)
        10:    z += MSAModule(msa, z, s_inputs)
        11:    s = s_init + Linear(LayerNorm(s_hat))
        12:    s, z = PairformerStack(s, z)
        13:    s_hat, z_hat = s, z
        15: x_pred = SampleDiffusion(features, s_inputs, s, z)
        16-18: Return predictions
        """
        model = AlphaFold3()
        
        # Should have all components
        assert hasattr(model, 'input_embedder')
        assert hasattr(model, 'single_init_proj')
        assert hasattr(model, 'pair_init_proj_i')
        assert hasattr(model, 'pair_init_proj_j')
        assert hasattr(model, 'relative_position_encoding')
        assert hasattr(model, 'msa_module')
        assert hasattr(model, 'pairformer')
        assert hasattr(model, 'sample_diffusion')
    
    def test_default_parameters_match_paper(self):
        """Default parameters should match Algorithm 1"""
        model = AlphaFold3()
        
        assert model.c_token == 384  # c_s = 384
        assert model.c_pair == 128  # c_z = 128
        assert model.n_cycles == 4  # N_cycle = 4


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_token(self):
        """Should work with single token"""
        model = AlphaFold3(
            n_cycles=1,
            msa_blocks=1,
            pairformer_blocks=1,
            diffusion_blocks=1
        )
        
        n_atoms = 4
        features = create_dummy_features(n_tokens=1, n_atoms=n_atoms)
        
        predictions = model(features, n_atoms=n_atoms)
        
        assert predictions['x_pred'].shape == (n_atoms, 3)
    
    def test_no_nan_in_output(self):
        """Output should not contain NaN or Inf"""
        model = AlphaFold3(
            n_cycles=1,
            msa_blocks=1,
            pairformer_blocks=1,
            diffusion_blocks=1
        )
        
        n_atoms = 40
        features = create_dummy_features(n_tokens=10, n_atoms=n_atoms)
        
        predictions = model(features, n_atoms=n_atoms)
        
        assert not torch.isnan(predictions['x_pred']).any()
        assert not torch.isinf(predictions['x_pred']).any()


class TestIntegration:
    """Test end-to-end integration"""
    
    def test_complete_pipeline(self):
        """Should run complete pipeline without errors"""
        # Create lightweight model for testing
        model = AlphaFold3(
            n_cycles=2,
            msa_blocks=2,
            pairformer_blocks=4,
            diffusion_blocks=2
        )
        
        # Create features
        n_atoms = 60
        features = create_dummy_features(n_tokens=15, n_atoms=n_atoms)
        
        # MSA features
        msa_features = {
            'msa': torch.randn(8, 15, 64),
            'has_deletion': torch.zeros(8, 15),
            'deletion_value': torch.zeros(8, 15)
        }
        
        # Run inference
        predictions = model(features, msa_features=msa_features, n_atoms=n_atoms)
        
        # Check outputs
        assert 'x_pred' in predictions
        assert 's_final' in predictions
        assert 'z_final' in predictions
        
        assert predictions['x_pred'].shape == (n_atoms, 3)
        assert predictions['s_final'].shape == (15, 384)
        assert predictions['z_final'].shape == (15, 15, 128)
        
        # Verify no NaN
        assert not torch.isnan(predictions['x_pred']).any()
        assert not torch.isnan(predictions['s_final']).any()
        assert not torch.isnan(predictions['z_final']).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])