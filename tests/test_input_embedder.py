"""
Unit tests for AlphaFold3 InputFeatureEmbedder.

Tests cover:
1. Basic initialization
2. Forward pass shape validation
3. Feature concatenation correctness
4. Compatibility with different input sizes
5. Algorithm 2 faithfulness
"""

import pytest
import torch
from src.models.embeddings.input_embedder import (
    InputFeatureEmbedder,
    create_dummy_input_features
)


class TestInitialization:
    """Test InputFeatureEmbedder initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        embedder = InputFeatureEmbedder()
        
        assert embedder.c_token == 384
        assert embedder.c_atom == 128
        assert embedder.c_atompair == 16
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        embedder = InputFeatureEmbedder(
            c_token=512,
            c_atom=256,
            c_atompair=32
        )
        
        assert embedder.c_token == 512
        assert embedder.c_atom == 256
        assert embedder.c_atompair == 32


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_basic_forward(self):
        """Forward pass should produce correct output shape"""
        embedder = InputFeatureEmbedder(c_token=384)
        features = create_dummy_input_features(n_tokens=10, n_atoms=40)
        
        output = embedder(features)
        
        # Output should be [N_token, c_token + 65]
        # 65 = 32 (restype) + 32 (profile) + 1 (deletion_mean)
        assert output.shape == (10, 384 + 65)
        assert output.shape == (10, 449)
    
    def test_output_dimension_method(self):
        """get_output_dim should return correct dimension"""
        embedder = InputFeatureEmbedder(c_token=384)
        
        assert embedder.get_output_dim() == 449  # 384 + 32 + 32 + 1
    
    def test_different_token_counts(self):
        """Should handle different numbers of tokens"""
        embedder = InputFeatureEmbedder(c_token=384)
        
        for n_tokens in [5, 10, 20, 50]:
            features = create_dummy_input_features(
                n_tokens=n_tokens,
                n_atoms=n_tokens * 4
            )
            output = embedder(features)
            
            assert output.shape[0] == n_tokens
            assert output.shape[1] == 449
    
    def test_batch_processing(self):
        """Should handle batched inputs"""
        embedder = InputFeatureEmbedder(c_token=384)
        
        # Create batch of 2 samples
        features1 = create_dummy_input_features(n_tokens=10, n_atoms=40)
        features2 = create_dummy_input_features(n_tokens=10, n_atoms=40)
        
        # Stack features
        batched_features = {
            key: torch.stack([features1[key], features2[key]])
            for key in features1.keys()
        }
        
        # Process batch (note: current implementation doesn't handle batches yet)
        # This test documents expected future behavior
        # output = embedder(batched_features)
        # assert output.shape == (2, 10, 449)


class TestFeatureConcatenation:
    """Test that features are concatenated correctly per Algorithm 2"""
    
    def test_concatenation_order(self):
        """Features should be concatenated in correct order"""
        embedder = InputFeatureEmbedder(c_token=384)
        features = create_dummy_input_features(n_tokens=5, n_atoms=20)
        
        output = embedder(features)
        
        # Output structure: [ai (384), restype (32), profile (32), deletion_mean (1)]
        # Total: 449
        
        # Check that we can extract each component
        ai_part = output[:, :384]
        restype_part = output[:, 384:416]
        profile_part = output[:, 416:448]
        deletion_mean_part = output[:, 448:449]
        
        assert ai_part.shape == (5, 384)
        assert restype_part.shape == (5, 32)
        assert profile_part.shape == (5, 32)
        assert deletion_mean_part.shape == (5, 1)
    
    def test_restype_preserved(self):
        """Restype values should be preserved in concatenation"""
        embedder = InputFeatureEmbedder(c_token=384)
        features = create_dummy_input_features(n_tokens=5, n_atoms=20)
        
        original_restype = features['restype'].clone()
        output = embedder(features)
        
        # Extract restype from output
        restype_part = output[:, 384:416]
        
        # Should match original
        assert torch.allclose(restype_part, original_restype)
    
    def test_profile_preserved(self):
        """Profile values should be preserved in concatenation"""
        embedder = InputFeatureEmbedder(c_token=384)
        features = create_dummy_input_features(n_tokens=5, n_atoms=20)
        
        original_profile = features['profile'].clone()
        output = embedder(features)
        
        # Extract profile from output
        profile_part = output[:, 416:448]
        
        # Should match original
        assert torch.allclose(profile_part, original_profile)
    
    def test_deletion_mean_preserved(self):
        """Deletion mean should be preserved and expanded"""
        embedder = InputFeatureEmbedder(c_token=384)
        features = create_dummy_input_features(n_tokens=5, n_atoms=20)
        
        original_deletion_mean = features['deletion_mean'].clone()
        output = embedder(features)
        
        # Extract deletion_mean from output (expanded to 2D)
        deletion_mean_part = output[:, 448:449].squeeze(-1)
        
        # Should match original
        assert torch.allclose(deletion_mean_part, original_deletion_mean)


class TestAlgorithm2Faithfulness:
    """Test faithfulness to Algorithm 2 specification"""
    
    def test_placeholder_atom_features(self):
        """
        Currently uses placeholder for atom features.
        This test documents the expected behavior once
        AtomAttentionEncoder is implemented.
        """
        embedder = InputFeatureEmbedder(c_token=384)
        features = create_dummy_input_features(n_tokens=5, n_atoms=20)
        
        output = embedder(features)
        
        # Placeholder ai should be zeros
        ai_part = output[:, :384]
        assert torch.allclose(ai_part, torch.zeros_like(ai_part))
        
        # Once AtomAttentionEncoder is implemented, this test should verify:
        # - ai is computed from atom features
        # - Aggregation from atoms to tokens is correct
        # - Permutation invariance holds
    
    def test_algorithm_2_structure(self):
        """
        Verify implementation follows Algorithm 2 structure:
        1: {ai}, _, _, _ = AtomAttentionEncoder({f*}, ∅, ∅, ∅, ...)
        2: si = concat(ai, f_restype_i, f_profile_i, f_deletion_mean_i)
        3: return {si}
        """
        embedder = InputFeatureEmbedder(c_token=384)
        features = create_dummy_input_features(n_tokens=10, n_atoms=40)
        
        # Step 1: AtomAttentionEncoder (currently placeholder)
        # ai shape: [N_token, c_token]
        
        # Step 2 & 3: Concatenation
        output = embedder(features)
        
        # Verify output structure matches Algorithm 2 line 2
        n_tokens = features['restype'].shape[0]
        expected_dim = embedder.c_token + 32 + 32 + 1
        
        assert output.shape == (n_tokens, expected_dim)


class TestDummyFeatureGeneration:
    """Test dummy feature generation utility"""
    
    def test_dummy_features_shape(self):
        """Dummy features should have correct shapes"""
        features = create_dummy_input_features(
            n_tokens=10,
            n_atoms=40,
            c_token=384
        )
        
        assert features['restype'].shape == (10, 32)
        assert features['profile'].shape == (10, 32)
        assert features['deletion_mean'].shape == (10,)
        assert features['ref_pos'].shape == (40, 3)
        assert features['ref_charge'].shape == (40,)
        assert features['ref_mask'].shape == (40,)
        assert features['ref_element'].shape == (40, 128)
        assert features['ref_atom_name_chars'].shape == (40, 4, 64)
        assert features['ref_space_uid'].shape == (40,)
    
    def test_dummy_features_types(self):
        """Dummy features should have correct dtypes"""
        features = create_dummy_input_features(n_tokens=5, n_atoms=20)
        
        # All should be torch tensors
        assert all(isinstance(v, torch.Tensor) for v in features.values())
        
        # ref_space_uid should be integer
        assert features['ref_space_uid'].dtype in [torch.int32, torch.int64]


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_token(self):
        """Should handle single token input"""
        embedder = InputFeatureEmbedder(c_token=384)
        features = create_dummy_input_features(n_tokens=1, n_atoms=4)
        
        output = embedder(features)
        
        assert output.shape == (1, 449)
    
    def test_many_tokens(self):
        """Should handle large number of tokens"""
        embedder = InputFeatureEmbedder(c_token=384)
        features = create_dummy_input_features(n_tokens=500, n_atoms=2000)
        
        output = embedder(features)
        
        assert output.shape == (500, 449)
    
    def test_custom_c_token(self):
        """Should work with non-default c_token"""
        embedder = InputFeatureEmbedder(c_token=512)
        features = create_dummy_input_features(n_tokens=10, n_atoms=40, c_token=512)
        
        output = embedder(features)
        
        # 512 (c_token) + 32 + 32 + 1 = 577
        assert output.shape == (10, 577)
        assert embedder.get_output_dim() == 577


if __name__ == "__main__":
    pytest.main([__file__, "-v"])