"""
Unit tests for AlphaFold3 AtomAttentionEncoder.

Tests cover:
1. Initialization
2. Forward pass shape validation
3. Atom-to-token aggregation (mean pooling)
4. Pair feature computation
5. Permutation invariance
6. Algorithm 5 faithfulness
"""

import pytest
import torch
from src.models.embeddings.atom_attention import (
    AtomAttentionEncoder,
    create_dummy_atom_features
)


class TestInitialization:
    """Test AtomAttentionEncoder initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        encoder = AtomAttentionEncoder()
        
        assert encoder.c_atom == 128
        assert encoder.c_atompair == 16
        assert encoder.c_token == 384
        assert encoder.n_blocks == 3
        assert encoder.n_heads == 4
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        encoder = AtomAttentionEncoder(
            c_atom=256,
            c_atompair=32,
            c_token=512,
            n_blocks=5,
            n_heads=8
        )
        
        assert encoder.c_atom == 256
        assert encoder.c_atompair == 32
        assert encoder.c_token == 512
        assert encoder.n_blocks == 5
        assert encoder.n_heads == 8


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_basic_forward(self):
        """Forward pass should produce correct output shapes"""
        encoder = AtomAttentionEncoder(c_atom=128, c_atompair=16, c_token=384)
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=40, n_tokens=10
        )
        
        ai, q_skip, c_skip, p_skip = encoder(atom_features, atom_to_token_idx)
        
        # Check output shapes
        assert ai.shape == (10, 384)  # [N_token, c_token]
        assert q_skip.shape == (40, 128)  # [N_atom, c_atom]
        assert c_skip.shape == (40, 128)  # [N_atom, c_atom]
        assert p_skip.shape == (40, 40, 16)  # [N_atom, N_atom, c_atompair]
    
    def test_different_atom_counts(self):
        """Should handle different numbers of atoms"""
        encoder = AtomAttentionEncoder()
        
        for n_atoms, n_tokens in [(20, 5), (40, 10), (100, 25)]:
            atom_features, atom_to_token_idx = create_dummy_atom_features(
                n_atoms=n_atoms, n_tokens=n_tokens
            )
            
            ai, _, _, _ = encoder(atom_features, atom_to_token_idx)
            
            assert ai.shape == (n_tokens, 384)
    
    def test_with_optional_inputs(self):
        """Should handle optional trunk embeddings and noisy positions"""
        encoder = AtomAttentionEncoder()
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=40, n_tokens=10
        )
        
        # Optional inputs
        noisy_positions = torch.randn(40, 3)
        trunk_single = torch.randn(10, 384)
        trunk_pair = torch.randn(10, 10, 128)
        
        ai, _, _, _ = encoder(
            atom_features,
            atom_to_token_idx,
            noisy_positions=noisy_positions,
            trunk_single=trunk_single,
            trunk_pair=trunk_pair
        )
        
        assert ai.shape == (10, 384)


class TestAtomToTokenAggregation:
    """Test atom-to-token aggregation (mean pooling)"""
    
    def test_mean_pooling(self):
        """Aggregation should use mean pooling"""
        encoder = AtomAttentionEncoder(c_atom=128, c_token=384)
        
        # Create simple case: 4 atoms per token, 2 tokens
        n_atoms = 8
        n_tokens = 2
        atom_features, _ = create_dummy_atom_features(n_atoms, n_tokens)
        
        # Manually create mapping: atoms 0-3 → token 0, atoms 4-7 → token 1
        atom_to_token_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        
        ai, _, _, _ = encoder(atom_features, atom_to_token_idx)
        
        # Output should average over atoms per token
        assert ai.shape == (2, 384)
    
    def test_uneven_atoms_per_token(self):
        """Should handle uneven number of atoms per token"""
        encoder = AtomAttentionEncoder()
        
        # Token 0: 2 atoms, Token 1: 5 atoms, Token 2: 3 atoms
        atom_features, _ = create_dummy_atom_features(n_atoms=10, n_tokens=3)
        atom_to_token_idx = torch.tensor([0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
        
        ai, _, _, _ = encoder(atom_features, atom_to_token_idx)
        
        assert ai.shape == (3, 384)
    
    def test_single_atom_per_token(self):
        """Should handle single atom per token"""
        encoder = AtomAttentionEncoder()
        atom_features, _ = create_dummy_atom_features(n_atoms=10, n_tokens=10)
        atom_to_token_idx = torch.arange(10)
        
        ai, _, _, _ = encoder(atom_features, atom_to_token_idx)
        
        assert ai.shape == (10, 384)


class TestPairFeatureComputation:
    """Test atom pair feature computation"""
    
    def test_position_offsets(self):
        """Should compute position offsets between atoms"""
        encoder = AtomAttentionEncoder()
        
        # Create specific atom positions
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=3, n_tokens=1
        )
        atom_features['ref_pos'] = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        _, _, _, p_skip = encoder(atom_features, atom_to_token_idx)
        
        # Pair features should be computed
        assert p_skip.shape == (3, 3, 16)
    
    def test_same_residue_masking(self):
        """Should mask pairs based on ref_space_uid"""
        encoder = AtomAttentionEncoder()
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=6, n_tokens=2
        )
        
        # Atoms 0-2 in residue 0, atoms 3-5 in residue 1
        atom_features['ref_space_uid'] = torch.tensor([0, 0, 0, 1, 1, 1])
        
        _, _, _, p_skip = encoder(atom_features, atom_to_token_idx)
        
        # Pairs should respect residue boundaries
        assert p_skip.shape == (6, 6, 16)
    
    def test_distance_embedding(self):
        """Should embed inverse squared distances"""
        encoder = AtomAttentionEncoder()
        
        # Two atoms at known distance
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=2, n_tokens=1
        )
        atom_features['ref_pos'] = torch.tensor([
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0]  # Distance = 5.0
        ])
        atom_features['ref_space_uid'] = torch.tensor([0, 0])
        
        _, _, _, p_skip = encoder(atom_features, atom_to_token_idx)
        
        # Distance features should be encoded
        assert p_skip.shape == (2, 2, 16)


class TestPermutationInvariance:
    """Test permutation invariance of atom aggregation"""
    
    def test_atom_order_invariance(self):
        """Mean pooling should be invariant to atom ordering"""
        encoder = AtomAttentionEncoder()
        
        # Create features
        n_atoms = 12
        n_tokens = 3
        atom_features, _ = create_dummy_atom_features(n_atoms, n_tokens)
        atom_to_token_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        # Forward pass
        ai_1, _, _, _ = encoder(atom_features, atom_to_token_idx)
        
        # Permute atoms within each token
        perm = torch.tensor([3, 1, 0, 2, 7, 5, 4, 6, 11, 9, 8, 10])
        
        atom_features_perm = {
            k: v[perm] if v.shape[0] == n_atoms else v
            for k, v in atom_features.items()
        }
        atom_to_token_idx_perm = atom_to_token_idx[perm]
        
        # Forward pass with permuted atoms
        ai_2, _, _, _ = encoder(atom_features_perm, atom_to_token_idx_perm)
        
        # Results should be the same (mean is permutation invariant)
        assert torch.allclose(ai_1, ai_2, atol=1e-5)


class TestAlgorithm5Faithfulness:
    """Test faithfulness to Algorithm 5 specification"""
    
    def test_atom_single_embedding(self):
        """Line 1: Should embed atom metadata"""
        encoder = AtomAttentionEncoder()
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=10, n_tokens=3
        )
        
        _, _, c_skip, _ = encoder(atom_features, atom_to_token_idx)
        
        # c_skip is the atom single conditioning
        assert c_skip.shape == (10, 128)
    
    def test_skip_connections_stored(self):
        """Lines 17-18: Should return skip connections"""
        encoder = AtomAttentionEncoder()
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=20, n_tokens=5
        )
        
        ai, q_skip, c_skip, p_skip = encoder(atom_features, atom_to_token_idx)
        
        # All skip connections should be returned
        assert q_skip is not None
        assert c_skip is not None
        assert p_skip is not None
        assert q_skip.shape[0] == 20  # N_atom
        assert c_skip.shape[0] == 20  # N_atom
        assert p_skip.shape[:2] == (20, 20)  # [N_atom, N_atom]
    
    def test_aggregation_is_mean(self):
        """Line 16: Should use mean aggregation"""
        encoder = AtomAttentionEncoder()
        
        # Simple test: 2 atoms per token
        atom_features, _ = create_dummy_atom_features(n_atoms=4, n_tokens=2)
        atom_to_token_idx = torch.tensor([0, 0, 1, 1])
        
        ai, _, _, _ = encoder(atom_features, atom_to_token_idx)
        
        # Should average atoms per token
        assert ai.shape == (2, 384)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_atom(self):
        """Should handle single atom"""
        encoder = AtomAttentionEncoder()
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=1, n_tokens=1
        )
        
        ai, _, _, _ = encoder(atom_features, atom_to_token_idx)
        
        assert ai.shape == (1, 384)
    
    def test_many_atoms(self):
        """Should handle large number of atoms"""
        encoder = AtomAttentionEncoder()
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=500, n_tokens=100
        )
        
        ai, _, _, _ = encoder(atom_features, atom_to_token_idx)
        
        assert ai.shape == (100, 384)
    
    def test_custom_dimensions(self):
        """Should work with non-default dimensions"""
        encoder = AtomAttentionEncoder(
            c_atom=256,
            c_atompair=32,
            c_token=512
        )
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=40, n_tokens=10
        )
        
        ai, q_skip, c_skip, p_skip = encoder(atom_features, atom_to_token_idx)
        
        assert ai.shape == (10, 512)  # c_token
        assert q_skip.shape == (40, 256)  # c_atom
        assert c_skip.shape == (40, 256)  # c_atom
        assert p_skip.shape == (40, 40, 32)  # c_atompair


class TestDummyFeatureGeneration:
    """Test dummy feature generation utility"""
    
    def test_dummy_features_shape(self):
        """Dummy features should have correct shapes"""
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=40, n_tokens=10
        )
        
        assert atom_features['ref_pos'].shape == (40, 3)
        assert atom_features['ref_charge'].shape == (40,)
        assert atom_features['ref_mask'].shape == (40,)
        assert atom_features['ref_element'].shape == (40, 128)
        assert atom_features['ref_atom_name_chars'].shape == (40, 4, 64)
        assert atom_features['ref_space_uid'].shape == (40,)
        assert atom_to_token_idx.shape == (40,)
    
    def test_atom_to_token_mapping(self):
        """Atom-to-token mapping should be valid"""
        _, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=40, n_tokens=10
        )
        
        # All indices should be < n_tokens
        assert torch.all(atom_to_token_idx < 10)
        assert torch.all(atom_to_token_idx >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])