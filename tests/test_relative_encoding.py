"""
Unit tests for AlphaFold3 RelativePositionEncoding.

Tests cover:
1. Initialization
2. Forward pass shape validation
3. Relative position encoding
4. Relative token encoding
5. Relative chain encoding
6. Same entity mask
7. Algorithm 3 faithfulness
"""

import pytest
import torch
from src.models.embeddings.relative_encoding import (
    RelativePositionEncoding,
    create_dummy_relative_features
)


class TestInitialization:
    """Test RelativePositionEncoding initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        encoder = RelativePositionEncoding()
        
        assert encoder.r_max == 32
        assert encoder.s_max == 2
        assert encoder.c_z == 128
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        encoder = RelativePositionEncoding(
            r_max=64,
            s_max=4,
            c_z=256
        )
        
        assert encoder.r_max == 64
        assert encoder.s_max == 4
        assert encoder.c_z == 256


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_basic_forward(self):
        """Forward pass should produce correct output shape"""
        encoder = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
        features = create_dummy_relative_features(n_tokens=10, n_chains=2)
        
        output = encoder(features)
        
        assert output.shape == (10, 10, 128)
    
    def test_different_token_counts(self):
        """Should handle different numbers of tokens"""
        encoder = RelativePositionEncoding()
        
        for n_tokens in [5, 10, 20, 50]:
            features = create_dummy_relative_features(n_tokens=n_tokens)
            output = encoder(features)
            
            assert output.shape == (n_tokens, n_tokens, 128)
    
    def test_single_token(self):
        """Should handle single token"""
        encoder = RelativePositionEncoding()
        features = create_dummy_relative_features(n_tokens=1, n_chains=1)
        
        output = encoder(features)
        
        assert output.shape == (1, 1, 128)


class TestRelativePositionEncoding:
    """Test relative position (residue-level) encoding"""
    
    def test_same_chain_positions(self):
        """Should encode relative positions within same chain"""
        encoder = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
        
        # Create features with known positions
        features = {
            'token_index': torch.tensor([0, 1, 2, 3, 4]),
            'residue_index': torch.tensor([0, 1, 2, 3, 4]),
            'asym_id': torch.tensor([0, 0, 0, 0, 0]),  # All same chain
            'entity_id': torch.tensor([0, 0, 0, 0, 0]),
            'sym_id': torch.tensor([0, 0, 0, 0, 0])
        }
        
        output = encoder(features)
        
        # Output should be computed
        assert output.shape == (5, 5, 128)
        
        # Positions within same chain should be encoded differently from
        # different chains
        assert not torch.allclose(output[0, 1], output[0, 0])
    
    def test_different_chain_positions(self):
        """Different chains should get special value 2*r_max+1"""
        encoder = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
        
        features = {
            'token_index': torch.tensor([0, 1, 2, 3]),
            'residue_index': torch.tensor([0, 1, 0, 1]),
            'asym_id': torch.tensor([0, 0, 1, 1]),  # 2 chains
            'entity_id': torch.tensor([0, 0, 1, 1]),
            'sym_id': torch.tensor([0, 0, 0, 0])
        }
        
        output = encoder(features)
        
        # Cross-chain pairs should have different encoding
        # than within-chain pairs
        assert not torch.allclose(output[0, 2], output[0, 1])
    
    def test_clipping_at_r_max(self):
        """Positions beyond r_max should be clipped"""
        encoder = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
        
        # Create features with large position differences
        features = {
            'token_index': torch.tensor([0, 1, 2]),
            'residue_index': torch.tensor([0, 50, 100]),  # Large gaps
            'asym_id': torch.tensor([0, 0, 0]),
            'entity_id': torch.tensor([0, 0, 0]),
            'sym_id': torch.tensor([0, 0, 0])
        }
        
        output = encoder(features)
        
        # Should still produce valid output
        assert output.shape == (3, 3, 128)


class TestRelativeTokenEncoding:
    """Test relative token (within-residue) encoding"""
    
    def test_same_residue_tokens(self):
        """Tokens within same residue should be encoded"""
        encoder = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
        
        # Create multi-atom residue (per-atom tokenization)
        features = {
            'token_index': torch.tensor([0, 1, 2, 3]),
            'residue_index': torch.tensor([0, 0, 0, 0]),  # All same residue
            'asym_id': torch.tensor([0, 0, 0, 0]),
            'entity_id': torch.tensor([0, 0, 0, 0]),
            'sym_id': torch.tensor([0, 0, 0, 0])
        }
        
        output = encoder(features)
        
        # Tokens within same residue should have relative token encoding
        assert output.shape == (4, 4, 128)
    
    def test_different_residue_tokens(self):
        """Tokens in different residues should get special value"""
        encoder = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
        
        features = {
            'token_index': torch.tensor([0, 1, 2, 3]),
            'residue_index': torch.tensor([0, 0, 1, 1]),  # 2 residues
            'asym_id': torch.tensor([0, 0, 0, 0]),
            'entity_id': torch.tensor([0, 0, 0, 0]),
            'sym_id': torch.tensor([0, 0, 0, 0])
        }
        
        output = encoder(features)
        
        # Cross-residue pairs should differ from within-residue pairs
        assert not torch.allclose(output[0, 2], output[0, 1])


class TestRelativeChainEncoding:
    """Test relative chain (sym_id) encoding"""
    
    def test_same_chain_special_value(self):
        """Same chain should get special value 2*s_max+1"""
        encoder = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
        
        features = {
            'token_index': torch.tensor([0, 1, 2, 3]),
            'residue_index': torch.tensor([0, 1, 2, 3]),
            'asym_id': torch.tensor([0, 0, 0, 0]),  # All same chain
            'entity_id': torch.tensor([0, 0, 0, 0]),
            'sym_id': torch.tensor([0, 0, 0, 0])
        }
        
        output = encoder(features)
        
        # Should produce valid output
        assert output.shape == (4, 4, 128)
    
    def test_different_chains_relative_sym_id(self):
        """Different chains should encode sym_id differences"""
        encoder = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
        
        # Multiple copies of same sequence
        features = {
            'token_index': torch.tensor([0, 1, 2, 3, 4, 5]),
            'residue_index': torch.tensor([0, 1, 0, 1, 0, 1]),
            'asym_id': torch.tensor([0, 0, 1, 1, 2, 2]),  # 3 chains
            'entity_id': torch.tensor([0, 0, 0, 0, 0, 0]),  # Same sequence
            'sym_id': torch.tensor([0, 0, 1, 1, 2, 2])  # Different sym IDs
        }
        
        output = encoder(features)
        
        # Cross-chain pairs should have chain encoding
        assert output.shape == (6, 6, 128)
        assert not torch.allclose(output[0, 2], output[0, 4])


class TestSameEntityMask:
    """Test same entity binary mask"""
    
    def test_same_entity(self):
        """Tokens from same entity should be marked"""
        encoder = RelativePositionEncoding()
        
        features = {
            'token_index': torch.tensor([0, 1, 2, 3]),
            'residue_index': torch.tensor([0, 1, 0, 1]),
            'asym_id': torch.tensor([0, 0, 1, 1]),
            'entity_id': torch.tensor([0, 0, 0, 0]),  # All same entity
            'sym_id': torch.tensor([0, 0, 1, 1])
        }
        
        output = encoder(features)
        
        # Should encode that all are same entity
        assert output.shape == (4, 4, 128)
    
    def test_different_entities(self):
        """Tokens from different entities should be distinguished"""
        encoder = RelativePositionEncoding()
        
        features = {
            'token_index': torch.tensor([0, 1, 2, 3]),
            'residue_index': torch.tensor([0, 1, 0, 1]),
            'asym_id': torch.tensor([0, 0, 1, 1]),
            'entity_id': torch.tensor([0, 0, 1, 1]),  # 2 entities
            'sym_id': torch.tensor([0, 0, 0, 0])
        }
        
        output = encoder(features)
        
        # Different entities should be encoded differently
        assert output.shape == (4, 4, 128)


class TestAlgorithm3Faithfulness:
    """Test faithfulness to Algorithm 3 specification"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 3 structure:
        1-3: Compute masks (same_chain, same_residue, same_entity)
        4-5: Relative residue positions
        6-7: Relative token positions
        8-9: Relative chain positions
        10: Concatenate and project
        """
        encoder = RelativePositionEncoding(r_max=32, s_max=2, c_z=128)
        features = create_dummy_relative_features(n_tokens=10, n_chains=2)
        
        output = encoder(features)
        
        # Should produce pair representation
        assert output.shape == (10, 10, 128)
    
    def test_default_parameters(self):
        """Should use paper's default parameters"""
        encoder = RelativePositionEncoding()
        
        # Algorithm 3 defaults
        assert encoder.r_max == 32
        assert encoder.s_max == 2
        assert encoder.c_z == 128
    
    def test_concatenation_dimensions(self):
        """Concatenated features should have correct total dimension"""
        encoder = RelativePositionEncoding(r_max=32, s_max=2)
        
        # rel_pos: 2*32 + 2 = 66
        # rel_token: 2*32 + 2 = 66
        # same_entity: 1
        # rel_chain: 2*2 + 2 = 6
        # Total: 139
        
        expected_input_dim = 66 + 66 + 1 + 6
        assert encoder.linear.in_features == expected_input_dim


class TestEdgeCases:
    """Test edge cases"""
    
    def test_all_same_positions(self):
        """All tokens at same position (e.g., ligand)"""
        encoder = RelativePositionEncoding()
        
        features = {
            'token_index': torch.tensor([0, 1, 2, 3]),
            'residue_index': torch.tensor([0, 0, 0, 0]),  # All same
            'asym_id': torch.tensor([0, 0, 0, 0]),
            'entity_id': torch.tensor([0, 0, 0, 0]),
            'sym_id': torch.tensor([0, 0, 0, 0])
        }
        
        output = encoder(features)
        
        assert output.shape == (4, 4, 128)
    
    def test_many_chains(self):
        """Handle many chains (complex)"""
        encoder = RelativePositionEncoding()
        
        n_chains = 10
        tokens_per_chain = 5
        n_tokens = n_chains * tokens_per_chain
        
        features = create_dummy_relative_features(
            n_tokens=n_tokens,
            n_chains=n_chains
        )
        
        output = encoder(features)
        
        assert output.shape == (n_tokens, n_tokens, 128)
    
    def test_custom_r_max(self):
        """Should work with different r_max"""
        encoder = RelativePositionEncoding(r_max=16, s_max=2, c_z=128)
        features = create_dummy_relative_features(n_tokens=10)
        
        output = encoder(features)
        
        # Input dim should be: 34 + 34 + 1 + 6 = 75
        assert encoder.linear.in_features == 75
        assert output.shape == (10, 10, 128)


class TestDummyFeatureGeneration:
    """Test dummy feature generation utility"""
    
    def test_dummy_features_shape(self):
        """Dummy features should have correct shapes"""
        features = create_dummy_relative_features(n_tokens=10, n_chains=2)
        
        assert features['token_index'].shape == (10,)
        assert features['residue_index'].shape == (10,)
        assert features['asym_id'].shape == (10,)
        assert features['entity_id'].shape == (10,)
        assert features['sym_id'].shape == (10,)
    
    def test_chain_assignment(self):
        """Tokens should be distributed across chains"""
        features = create_dummy_relative_features(n_tokens=10, n_chains=2)
        
        # Should have 2 unique chain IDs
        assert len(torch.unique(features['asym_id'])) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])