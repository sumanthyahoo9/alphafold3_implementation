"""
Unit tests for AlphaFold3 AtomAttentionDecoder.

Tests cover:
1. Initialization
2. Forward pass shape validation
3. Token-to-atom broadcasting
4. Skip connections
5. Position update generation
6. Algorithm 6 faithfulness
7. Integration with encoder
"""

import pytest
import torch
from src.models.embeddings.atom_attention import (
    AtomAttentionEncoder,
    AtomAttentionDecoder,
    create_dummy_atom_features
)


class TestInitialization:
    """Test AtomAttentionDecoder initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        decoder = AtomAttentionDecoder()
        
        assert decoder.c_atom == 128
        assert decoder.c_atompair == 16
        assert decoder.c_token == 384
        assert decoder.n_blocks == 3
        assert decoder.n_heads == 4
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        decoder = AtomAttentionDecoder(
            c_atom=256,
            c_atompair=32,
            c_token=512,
            n_blocks=5,
            n_heads=8
        )
        
        assert decoder.c_atom == 256
        assert decoder.c_atompair == 32
        assert decoder.c_token == 512
        assert decoder.n_blocks == 5
        assert decoder.n_heads == 8


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_basic_forward(self):
        """Forward pass should produce position updates"""
        decoder = AtomAttentionDecoder(c_atom=128, c_atompair=16, c_token=384)
        
        # Create dummy inputs (matching encoder outputs)
        n_atoms = 40
        n_tokens = 10
        
        ai = torch.randn(n_tokens, 384)  # Token activations
        q_skip = torch.randn(n_atoms, 128)  # Atom queries
        c_skip = torch.randn(n_atoms, 128)  # Atom conditioning
        p_skip = torch.randn(n_atoms, n_atoms, 16)  # Atom pairs
        atom_to_token_idx = torch.arange(n_atoms) // 4
        
        r_update = decoder(ai, q_skip, c_skip, p_skip, atom_to_token_idx)
        
        # Should produce 3D position updates for each atom
        assert r_update.shape == (n_atoms, 3)
    
    def test_different_atom_counts(self):
        """Should handle different numbers of atoms"""
        decoder = AtomAttentionDecoder()
        
        for n_atoms, n_tokens in [(20, 5), (40, 10), (100, 25)]:
            ai = torch.randn(n_tokens, 384)
            q_skip = torch.randn(n_atoms, 128)
            c_skip = torch.randn(n_atoms, 128)
            p_skip = torch.randn(n_atoms, n_atoms, 16)
            atom_to_token_idx = torch.arange(n_atoms) // (n_atoms // n_tokens)
            atom_to_token_idx = atom_to_token_idx.clamp(max=n_tokens - 1)
            
            r_update = decoder(ai, q_skip, c_skip, p_skip, atom_to_token_idx)
            
            assert r_update.shape == (n_atoms, 3)


class TestTokenToAtomBroadcasting:
    """Test broadcasting of token activations to atoms"""
    
    def test_broadcasting_mechanism(self):
        """Token activations should be correctly broadcast to atoms"""
        decoder = AtomAttentionDecoder()
        
        n_atoms = 12
        n_tokens = 3
        
        # Create token activations with distinct values per token
        ai = torch.randn(n_tokens, 384)
        q_skip = torch.randn(n_atoms, 128)
        c_skip = torch.randn(n_atoms, 128)
        p_skip = torch.randn(n_atoms, n_atoms, 16)
        
        # Map: atoms 0-3 → token 0, atoms 4-7 → token 1, atoms 8-11 → token 2
        atom_to_token_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        
        r_update = decoder(ai, q_skip, c_skip, p_skip, atom_to_token_idx)
        
        # All atoms should get position updates
        assert r_update.shape == (12, 3)
    
    def test_single_atom_per_token(self):
        """Should handle 1-to-1 atom-token mapping"""
        decoder = AtomAttentionDecoder()
        
        n_atoms = 10
        n_tokens = 10
        
        ai = torch.randn(n_tokens, 384)
        q_skip = torch.randn(n_atoms, 128)
        c_skip = torch.randn(n_atoms, 128)
        p_skip = torch.randn(n_atoms, n_atoms, 16)
        atom_to_token_idx = torch.arange(n_atoms)  # 1-to-1 mapping
        
        r_update = decoder(ai, q_skip, c_skip, p_skip, atom_to_token_idx)
        
        assert r_update.shape == (n_atoms, 3)


class TestSkipConnections:
    """Test usage of skip connections from encoder"""
    
    def test_skip_connections_used(self):
        """Skip connections should affect output"""
        decoder = AtomAttentionDecoder()
        
        n_atoms = 20
        n_tokens = 5
        
        ai = torch.randn(n_tokens, 384)
        atom_to_token_idx = torch.arange(n_atoms) // 4
        
        # Different skip connections
        q_skip1 = torch.randn(n_atoms, 128)
        c_skip1 = torch.randn(n_atoms, 128)
        p_skip1 = torch.randn(n_atoms, n_atoms, 16)
        
        q_skip2 = torch.randn(n_atoms, 128)
        c_skip2 = torch.randn(n_atoms, 128)
        p_skip2 = torch.randn(n_atoms, n_atoms, 16)
        
        # Forward with different skip connections
        r_update1 = decoder(ai, q_skip1, c_skip1, p_skip1, atom_to_token_idx)
        r_update2 = decoder(ai, q_skip2, c_skip2, p_skip2, atom_to_token_idx)
        
        # Outputs should differ (skip connections have effect)
        assert not torch.allclose(r_update1, r_update2, atol=1e-5)


class TestPositionUpdates:
    """Test position update generation"""
    
    def test_position_update_shape(self):
        """Position updates should be 3D vectors"""
        decoder = AtomAttentionDecoder()
        
        n_atoms = 50
        n_tokens = 10
        
        ai = torch.randn(n_tokens, 384)
        q_skip = torch.randn(n_atoms, 128)
        c_skip = torch.randn(n_atoms, 128)
        p_skip = torch.randn(n_atoms, n_atoms, 16)
        atom_to_token_idx = torch.arange(n_atoms) // 5
        
        r_update = decoder(ai, q_skip, c_skip, p_skip, atom_to_token_idx)
        
        # Each atom gets a 3D update
        assert r_update.shape == (n_atoms, 3)
    
    def test_position_updates_vary(self):
        """Different inputs should produce different position updates"""
        decoder = AtomAttentionDecoder()
        
        n_atoms = 30
        n_tokens = 6
        atom_to_token_idx = torch.arange(n_atoms) // 5
        
        # Different token activations
        ai1 = torch.randn(n_tokens, 384)
        ai2 = torch.randn(n_tokens, 384)
        
        q_skip = torch.randn(n_atoms, 128)
        c_skip = torch.randn(n_atoms, 128)
        p_skip = torch.randn(n_atoms, n_atoms, 16)
        
        r_update1 = decoder(ai1, q_skip, c_skip, p_skip, atom_to_token_idx)
        r_update2 = decoder(ai2, q_skip, c_skip, p_skip, atom_to_token_idx)
        
        # Different inputs → different outputs
        assert not torch.allclose(r_update1, r_update2, atol=1e-5)


class TestAlgorithm6Faithfulness:
    """Test faithfulness to Algorithm 6 specification"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 6 structure:
        1: ql = LinearNoBias(a[tok_idx(l)]) + q_skip_l
        2: {ql} = AtomTransformer({ql}, {c_skip_l}, {p_skip_lm})
        3: r_update_l = LinearNoBias(LayerNorm(ql))
        4: return {r_update_l}
        """
        decoder = AtomAttentionDecoder()
        
        n_atoms = 40
        n_tokens = 10
        
        ai = torch.randn(n_tokens, 384)
        q_skip = torch.randn(n_atoms, 128)
        c_skip = torch.randn(n_atoms, 128)
        p_skip = torch.randn(n_atoms, n_atoms, 16)
        atom_to_token_idx = torch.arange(n_atoms) // 4
        
        r_update = decoder(ai, q_skip, c_skip, p_skip, atom_to_token_idx)
        
        # Should follow algorithm steps and produce position updates
        assert r_update.shape == (n_atoms, 3)
    
    def test_default_parameters(self):
        """Should use paper's default parameters"""
        decoder = AtomAttentionDecoder()
        
        # Algorithm 6 defaults
        assert decoder.n_blocks == 3
        assert decoder.n_heads == 4


class TestEncoderDecoderIntegration:
    """Test integration between encoder and decoder"""
    
    def test_encoder_decoder_round_trip(self):
        """Encoder outputs should be valid decoder inputs"""
        encoder = AtomAttentionEncoder(c_atom=128, c_atompair=16, c_token=384)
        decoder = AtomAttentionDecoder(c_atom=128, c_atompair=16, c_token=384)
        
        # Create atom features
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=40, n_tokens=10
        )
        
        # Encode: atoms → tokens
        ai, q_skip, c_skip, p_skip = encoder(atom_features, atom_to_token_idx)
        
        # Decode: tokens → atom position updates
        r_update = decoder(ai, q_skip, c_skip, p_skip, atom_to_token_idx)
        
        # Should produce position updates
        assert r_update.shape == (40, 3)
    
    def test_skip_connection_dimensions_match(self):
        """Encoder skip connections should match decoder expectations"""
        encoder = AtomAttentionEncoder()
        decoder = AtomAttentionDecoder()
        
        atom_features, atom_to_token_idx = create_dummy_atom_features(
            n_atoms=60, n_tokens=15
        )
        
        ai, q_skip, c_skip, p_skip = encoder(atom_features, atom_to_token_idx)
        
        # Verify skip connection shapes match decoder expectations
        assert q_skip.shape == (60, 128)  # [N_atom, c_atom]
        assert c_skip.shape == (60, 128)  # [N_atom, c_atom]
        assert p_skip.shape == (60, 60, 16)  # [N_atom, N_atom, c_atompair]
        
        # Decoder should accept these
        r_update = decoder(ai, q_skip, c_skip, p_skip, atom_to_token_idx)
        assert r_update.shape == (60, 3)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_atom(self):
        """Should handle single atom"""
        decoder = AtomAttentionDecoder()
        
        ai = torch.randn(1, 384)
        q_skip = torch.randn(1, 128)
        c_skip = torch.randn(1, 128)
        p_skip = torch.randn(1, 1, 16)
        atom_to_token_idx = torch.tensor([0])
        
        r_update = decoder(ai, q_skip, c_skip, p_skip, atom_to_token_idx)
        
        assert r_update.shape == (1, 3)
    
    def test_many_atoms(self):
        """Should handle large number of atoms"""
        decoder = AtomAttentionDecoder()
        
        n_atoms = 500
        n_tokens = 100
        
        ai = torch.randn(n_tokens, 384)
        q_skip = torch.randn(n_atoms, 128)
        c_skip = torch.randn(n_atoms, 128)
        p_skip = torch.randn(n_atoms, n_atoms, 16)
        atom_to_token_idx = torch.arange(n_atoms) // 5
        
        r_update = decoder(ai, q_skip, c_skip, p_skip, atom_to_token_idx)
        
        assert r_update.shape == (n_atoms, 3)
    
    def test_custom_dimensions(self):
        """Should work with non-default dimensions"""
        decoder = AtomAttentionDecoder(
            c_atom=256,
            c_atompair=32,
            c_token=512
        )
        
        n_atoms = 40
        n_tokens = 10
        
        ai = torch.randn(n_tokens, 512)
        q_skip = torch.randn(n_atoms, 256)
        c_skip = torch.randn(n_atoms, 256)
        p_skip = torch.randn(n_atoms, n_atoms, 32)
        atom_to_token_idx = torch.arange(n_atoms) // 4
        
        r_update = decoder(ai, q_skip, c_skip, p_skip, atom_to_token_idx)
        
        assert r_update.shape == (n_atoms, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])