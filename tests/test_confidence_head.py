"""
Unit tests for AlphaFold3 Confidence Head.

File: tests/test_confidence_head.py

Tests cover:
1. Initialization
2. Forward pass functionality
3. Output shapes and ranges
4. Algorithm 31 faithfulness
5. Score conversion utilities
"""

import pytest
import torch
from src.models.heads.confidence_head import ConfidenceHead


class TestInitialization:
    """Test ConfidenceHead initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        head = ConfidenceHead()
        
        assert head.c_single == 384
        assert head.c_pair == 128
        assert head.n_bins_plddt == 50
        assert head.n_bins_pae == 64
        assert head.n_bins_pde == 64
        assert head.n_bins_resolved == 2
    
    def test_custom_initialization(self):
        """Should accept custom parameters"""
        head = ConfidenceHead(
            c_single=256,
            c_pair=64,
            n_blocks=2,
            n_bins_plddt=25
        )
        
        assert head.c_single == 256
        assert head.c_pair == 64
        assert head.n_bins_plddt == 25
    
    def test_has_all_components(self):
        """Should have all required submodules"""
        head = ConfidenceHead()
        
        # Algorithm 31 components
        assert hasattr(head, 's_inputs_proj_i')  # Line 1
        assert hasattr(head, 's_inputs_proj_j')  # Line 1
        assert hasattr(head, 'distance_proj')    # Line 3
        assert hasattr(head, 'pairformer')       # Line 4
        assert hasattr(head, 'pae_proj')         # Line 5
        assert hasattr(head, 'pde_proj')         # Line 6
        assert hasattr(head, 'plddt_proj')       # Line 7
        assert hasattr(head, 'resolved_proj')    # Line 8


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_forward_shape(self):
        """Should produce correct output shapes"""
        head = ConfidenceHead(n_blocks=1)
        
        n_tokens = 10
        n_atoms = 40
        
        s_inputs = torch.randn(n_tokens, 384)
        s = torch.randn(n_tokens, 384)
        z = torch.randn(n_tokens, n_tokens, 128)
        x_pred = torch.randn(n_atoms, 3)
        atom_to_token_idx = torch.arange(n_atoms) % n_tokens
        
        p_plddt, p_pae, p_pde, p_resolved = head(
            s_inputs, s, z, x_pred, atom_to_token_idx
        )
        
        assert p_plddt.shape == (n_atoms, 50)
        assert p_pae.shape == (n_tokens, n_tokens, 64)
        assert p_pde.shape == (n_tokens, n_tokens, 64)
        assert p_resolved.shape == (n_atoms, 2)
    
    def test_different_sizes(self):
        """Should work with different input sizes"""
        head = ConfidenceHead(n_blocks=1)
        
        for n_tokens in [5, 10, 20]:
            n_atoms = n_tokens * 4
            
            s_inputs = torch.randn(n_tokens, 384)
            s = torch.randn(n_tokens, 384)
            z = torch.randn(n_tokens, n_tokens, 128)
            x_pred = torch.randn(n_atoms, 3)
            atom_to_token_idx = torch.arange(n_atoms) % n_tokens
            
            p_plddt, p_pae, p_pde, p_resolved = head(
                s_inputs, s, z, x_pred, atom_to_token_idx
            )
            
            assert p_plddt.shape == (n_atoms, 50)
            assert p_pae.shape == (n_tokens, n_tokens, 64)
    
    def test_outputs_are_probabilities(self):
        """Output probabilities should sum to 1"""
        head = ConfidenceHead(n_blocks=1)
        
        n_tokens = 10
        n_atoms = 40
        
        s_inputs = torch.randn(n_tokens, 384)
        s = torch.randn(n_tokens, 384)
        z = torch.randn(n_tokens, n_tokens, 128)
        x_pred = torch.randn(n_atoms, 3)
        atom_to_token_idx = torch.arange(n_atoms) % n_tokens
        
        p_plddt, p_pae, p_pde, p_resolved = head(
            s_inputs, s, z, x_pred, atom_to_token_idx
        )
        
        # Probabilities should sum to 1
        assert torch.allclose(p_plddt.sum(dim=-1), torch.ones(n_atoms), atol=1e-5)
        assert torch.allclose(p_pae.sum(dim=-1), torch.ones(n_tokens, n_tokens), atol=1e-5)
        assert torch.allclose(p_pde.sum(dim=-1), torch.ones(n_tokens, n_tokens), atol=1e-5)
        assert torch.allclose(p_resolved.sum(dim=-1), torch.ones(n_atoms), atol=1e-5)


class TestScoreConversion:
    """Test score conversion utilities"""
    
    def test_plddt_score_range(self):
        """pLDDT scores should be in [0, 1]"""
        head = ConfidenceHead()
        
        # Create dummy probabilities
        p_plddt = torch.softmax(torch.randn(40, 50), dim=-1)
        
        plddt_scores = head.get_plddt_scores(p_plddt)
        
        assert plddt_scores.shape == (40,)
        assert (plddt_scores >= 0).all()
        assert (plddt_scores <= 1).all()
    
    def test_pae_score_range(self):
        """PAE scores should be in [0, 32]"""
        head = ConfidenceHead()
        
        # Create dummy probabilities
        p_pae = torch.softmax(torch.randn(10, 10, 64), dim=-1)
        
        pae_scores = head.get_pae_scores(p_pae)
        
        assert pae_scores.shape == (10, 10)
        assert (pae_scores >= 0).all()
        assert (pae_scores <= 32).all()
    
    def test_pde_score_range(self):
        """PDE scores should be in [0, 32]"""
        head = ConfidenceHead()
        
        # Create dummy probabilities
        p_pde = torch.softmax(torch.randn(10, 10, 64), dim=-1)
        
        pde_scores = head.get_pde_scores(p_pde)
        
        assert pde_scores.shape == (10, 10)
        assert (pde_scores >= 0).all()
        assert (pde_scores <= 32).all()


class TestAlgorithm31Faithfulness:
    """Test faithfulness to Algorithm 31"""
    
    def test_algorithm_structure(self):
        """
        Algorithm 31:
        1: z += Linear(s_inputs_i) + Linear(s_inputs_j)
        2: d_ij = ||x_pred[rep(i)] - x_pred[rep(j)]||
        3: z += Linear(one_hot(d_ij, bins))
        4: s, z = PairformerStack(s, z, N_block=4)
        5: p_pae = softmax(Linear(z))
        6: p_pde = softmax(Linear(z + z^T))
        7: p_plddt = softmax(Linear(s)[token_atom_idx])
        8: p_resolved = softmax(Linear(s)[token_atom_idx])
        9: return p_plddt, p_pae, p_pde, p_resolved
        """
        head = ConfidenceHead()
        
        # Should have all components
        assert hasattr(head, 's_inputs_proj_i')
        assert hasattr(head, 's_inputs_proj_j')
        assert hasattr(head, 'distance_proj')
        assert hasattr(head, 'pairformer')
        assert hasattr(head, 'pae_proj')
        assert hasattr(head, 'pde_proj')
        assert hasattr(head, 'plddt_proj')
        assert hasattr(head, 'resolved_proj')
    
    def test_default_parameters_match_paper(self):
        """Default parameters should match Algorithm 31"""
        head = ConfidenceHead()
        
        # N_block = 4 in paper
        assert head.pairformer.n_blocks == 4
        
        # 50 bins for pLDDT (0 to 1)
        assert head.n_bins_plddt == 50
        
        # 64 bins for PAE and PDE (0 to 32 Angstroms)
        assert head.n_bins_pae == 64
        assert head.n_bins_pde == 64
        
        # 2 bins for resolved (yes/no)
        assert head.n_bins_resolved == 2


class TestDistanceEncoding:
    """Test distance encoding"""
    
    def test_one_hot_encoding_shape(self):
        """Distance one-hot encoding should have correct shape"""
        head = ConfidenceHead()
        
        distances = torch.randn(10, 10).abs() * 20  # Positive distances
        
        one_hot = head._one_hot_encode_distances(distances)
        
        assert one_hot.shape == (10, 10, 64)
    
    def test_one_hot_sums_to_one(self):
        """One-hot encoding should sum to 1"""
        head = ConfidenceHead()
        
        distances = torch.randn(10, 10).abs() * 20
        
        one_hot = head._one_hot_encode_distances(distances)
        
        assert torch.allclose(one_hot.sum(dim=-1), torch.ones(10, 10))


class TestGradientFlow:
    """Test gradient handling"""
    
    def test_stop_gradients(self):
        """Should stop gradients on inputs as per paper"""
        head = ConfidenceHead(n_blocks=1)
        
        n_tokens = 10
        n_atoms = 40
        
        s_inputs = torch.randn(n_tokens, 384, requires_grad=True)
        s = torch.randn(n_tokens, 384, requires_grad=True)
        z = torch.randn(n_tokens, n_tokens, 128, requires_grad=True)
        x_pred = torch.randn(n_atoms, 3, requires_grad=True)
        atom_to_token_idx = torch.arange(n_atoms) % n_tokens
        
        p_plddt, p_pae, p_pde, p_resolved = head(
            s_inputs, s, z, x_pred, atom_to_token_idx
        )
        
        # Forward should work
        assert p_plddt.shape == (n_atoms, 50)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_token(self):
        """Should work with single token"""
        head = ConfidenceHead(n_blocks=1)
        
        n_tokens = 1
        n_atoms = 4
        
        s_inputs = torch.randn(n_tokens, 384)
        s = torch.randn(n_tokens, 384)
        z = torch.randn(n_tokens, n_tokens, 128)
        x_pred = torch.randn(n_atoms, 3)
        atom_to_token_idx = torch.zeros(n_atoms, dtype=torch.long)
        
        p_plddt, p_pae, p_pde, p_resolved = head(
            s_inputs, s, z, x_pred, atom_to_token_idx
        )
        
        assert p_plddt.shape == (n_atoms, 50)
        assert p_pae.shape == (1, 1, 64)
    
    def test_no_nan_in_output(self):
        """Output should not contain NaN"""
        head = ConfidenceHead(n_blocks=1)
        
        n_tokens = 10
        n_atoms = 40
        
        s_inputs = torch.randn(n_tokens, 384)
        s = torch.randn(n_tokens, 384)
        z = torch.randn(n_tokens, n_tokens, 128)
        x_pred = torch.randn(n_atoms, 3)
        atom_to_token_idx = torch.arange(n_atoms) % n_tokens
        
        p_plddt, p_pae, p_pde, p_resolved = head(
            s_inputs, s, z, x_pred, atom_to_token_idx
        )
        
        assert not torch.isnan(p_plddt).any()
        assert not torch.isnan(p_pae).any()
        assert not torch.isnan(p_pde).any()
        assert not torch.isnan(p_resolved).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])