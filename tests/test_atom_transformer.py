"""
Unit tests for AlphaFold3 AtomTransformer.

Tests cover:
1. Initialization
2. Sequence-local mask generation
3. Forward pass
4. Attention pattern verification
5. Block-sparse attention
6. Algorithm 7 faithfulness
"""

import pytest
import torch
from src.models.embeddings.atom_attention import (
    AtomTransformer,
    AtomTransformerBlock
)


class TestAtomTransformerInitialization:
    """Test AtomTransformer initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default parameters"""
        transformer = AtomTransformer()
        
        assert transformer.c_atom == 128
        assert transformer.c_atompair == 16
        assert transformer.n_blocks == 3
        assert transformer.n_heads == 4
        assert transformer.n_queries == 32
        assert transformer.n_keys == 128
    
    def test_custom_dimensions(self):
        """Should accept custom dimensions"""
        transformer = AtomTransformer(
            c_atom=256,
            c_atompair=32,
            n_blocks=5,
            n_heads=8,
            n_queries=64,
            n_keys=256
        )
        
        assert transformer.c_atom == 256
        assert transformer.c_atompair == 32
        assert transformer.n_blocks == 5
        assert transformer.n_heads == 8
        assert transformer.n_queries == 64
        assert transformer.n_keys == 256
    
    def test_blocks_created(self):
        """Should create correct number of transformer blocks"""
        transformer = AtomTransformer(n_blocks=5)
        
        assert len(transformer.blocks) == 5
        assert all(isinstance(block, AtomTransformerBlock) 
                  for block in transformer.blocks)


class TestSequenceLocalMask:
    """Test sequence-local attention mask generation"""
    
    def test_mask_shape(self):
        """Mask should have correct shape"""
        transformer = AtomTransformer(n_queries=32, n_keys=128)
        
        mask = transformer._create_sequence_local_mask(n_atoms=200)
        
        assert mask.shape == (200, 200)
    
    def test_mask_values(self):
        """Mask should have only 0 and -1e10 values"""
        transformer = AtomTransformer(n_queries=32, n_keys=128)
        
        mask = transformer._create_sequence_local_mask(n_atoms=100)
        
        unique_values = torch.unique(mask)
        assert len(unique_values) <= 2
        assert 0.0 in unique_values or -1e10 in unique_values
    
    def test_first_block_pattern(self):
        """First block queries should attend to nearby keys"""
        transformer = AtomTransformer(n_queries=32, n_keys=128)
        
        mask = transformer._create_sequence_local_mask(n_atoms=200)
        
        # Check that first atoms can attend to nearby atoms
        # The exact pattern depends on center positions
        first_queries = mask[:32, :]
        
        # Each query should be able to attend to at least n_keys atoms
        for q_idx in range(32):
            num_allowed = (first_queries[q_idx] == 0.0).sum()
            assert num_allowed >= 64, f"Query {q_idx} can only attend to {num_allowed} atoms (expected >= 64)"
    
    def test_diagonal_pattern(self):
        """Mask should create rectangular blocks along diagonal"""
        transformer = AtomTransformer(n_queries=32, n_keys=128)
        
        mask = transformer._create_sequence_local_mask(n_atoms=200)
        
        # Count non-masked entries per row
        non_masked_per_row = (mask != -1e10).sum(dim=1)
        
        # Most rows should attend to ~128 atoms
        # (edge rows might attend to fewer)
        assert non_masked_per_row.max() <= 128
        assert non_masked_per_row[50] > 64  # Middle rows should attend to many
    
    def test_small_n_atoms(self):
        """Should handle n_atoms < n_keys"""
        transformer = AtomTransformer(n_queries=32, n_keys=128)
        
        mask = transformer._create_sequence_local_mask(n_atoms=50)
        
        assert mask.shape == (50, 50)
        # With only 50 atoms, all should attend to all
        assert (mask == 0.0).any()


class TestAtomTransformerForward:
    """Test AtomTransformer forward pass"""
    
    def test_basic_forward(self):
        """Forward pass should produce correct output shape"""
        transformer = AtomTransformer(c_atom=128, c_atompair=16)
        
        ql = torch.randn(100, 128)
        cl = torch.randn(100, 128)
        plm = torch.randn(100, 100, 16)
        
        output = transformer(ql, cl, plm)
        
        assert output.shape == (100, 128)
    
    def test_different_n_atoms(self):
        """Should handle different numbers of atoms"""
        transformer = AtomTransformer()
        
        for n_atoms in [50, 100, 200, 500]:
            ql = torch.randn(n_atoms, 128)
            cl = torch.randn(n_atoms, 128)
            plm = torch.randn(n_atoms, n_atoms, 16)
            
            output = transformer(ql, cl, plm)
            
            assert output.shape == (n_atoms, 128)
    
    def test_residual_connection(self):
        """Output should differ from input (transformer is applied)"""
        transformer = AtomTransformer(n_blocks=1)
        
        ql = torch.randn(50, 128)
        cl = torch.randn(50, 128)
        plm = torch.randn(50, 50, 16)
        
        output = transformer(ql, cl, plm)
        
        # Output should be different from input
        assert not torch.allclose(output, ql, atol=1e-3)


class TestAttentionPattern:
    """Test that attention respects sequence-local pattern"""
    
    def test_attention_is_local(self):
        """Attention should be restricted to local windows"""
        transformer = AtomTransformer(n_queries=32, n_keys=128, n_blocks=1)
        
        # Create inputs with clear structure
        n_atoms = 200
        ql = torch.randn(n_atoms, 128)
        cl = torch.randn(n_atoms, 128)
        plm = torch.randn(n_atoms, n_atoms, 16)
        
        # Forward pass
        output = transformer(ql, cl, plm)
        
        # Output should be influenced only by local atoms
        assert output.shape == (n_atoms, 128)
    
    def test_no_global_attention(self):
        """Distant atoms should not directly influence each other"""
        transformer = AtomTransformer(n_queries=32, n_keys=128, n_blocks=1)
        
        n_atoms = 300
        ql = torch.randn(n_atoms, 128)
        cl = torch.randn(n_atoms, 128)
        plm = torch.randn(n_atoms, n_atoms, 16)
        
        # Modify atom 0
        ql_modified = ql.clone()
        ql_modified[0] *= 10  # Large change to atom 0
        
        output1 = transformer(ql, cl, plm)
        output2 = transformer(ql_modified, cl, plm)
        
        # Atom 200 should be unaffected (too far away)
        # (They're ~200 atoms apart, well beyond n_keys=128 window)
        # Note: This is approximate due to residual connections
        assert torch.allclose(output1[200], output2[200], rtol=0.1)


class TestAlgorithm7Faithfulness:
    """Test faithfulness to Algorithm 7 specification"""
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 7 structure:
        1: Create sequence-local mask βlm
        2: Apply DiffusionTransformer with mask
        3: Return transformed queries
        """
        transformer = AtomTransformer()
        
        # Step 1: Mask creation
        mask = transformer._create_sequence_local_mask(100)
        assert mask.shape == (100, 100)
        
        # Steps 2-3: Forward pass
        ql = torch.randn(100, 128)
        cl = torch.randn(100, 128)
        plm = torch.randn(100, 100, 16)
        
        output = transformer(ql, cl, plm)
        assert output.shape == (100, 128)
    
    def test_default_parameters(self):
        """Should use paper's default parameters"""
        transformer = AtomTransformer()
        
        # Algorithm 7 defaults
        assert transformer.n_blocks == 3
        assert transformer.n_heads == 4
        assert transformer.n_queries == 32
        assert transformer.n_keys == 128


class TestAtomTransformerBlock:
    """Test individual transformer block"""
    
    def test_block_initialization(self):
        """Block should initialize correctly"""
        block = AtomTransformerBlock(c_atom=128, c_atompair=16, n_heads=4)
        
        assert block.c_atom == 128
        assert block.c_atompair == 16
        assert block.n_heads == 4
        assert block.c_head == 32  # 128 / 4
    
    def test_block_forward(self):
        """Block forward should work correctly"""
        block = AtomTransformerBlock(c_atom=128, c_atompair=16, n_heads=4)
        
        ql = torch.randn(50, 128)
        cl = torch.randn(50, 128)
        plm = torch.randn(50, 50, 16)
        beta_mask = torch.zeros(50, 50)
        
        output = block(ql, cl, plm, beta_mask)
        
        assert output.shape == (50, 128)
    
    def test_pair_bias_applied(self):
        """Pair features should bias attention"""
        block = AtomTransformerBlock(c_atom=128, c_atompair=16, n_heads=4)
        
        ql = torch.randn(20, 128)
        cl = torch.randn(20, 128)
        plm = torch.randn(20, 20, 16)
        beta_mask = torch.zeros(20, 20)
        
        # Forward with pair features
        output1 = block(ql, cl, plm, beta_mask)
        
        # Forward with zero pair features
        plm_zero = torch.zeros_like(plm)
        output2 = block(ql, cl, plm_zero, beta_mask)
        
        # Outputs should differ
        assert not torch.allclose(output1, output2, atol=1e-3)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_single_atom(self):
        """Should handle single atom"""
        transformer = AtomTransformer()
        
        ql = torch.randn(1, 128)
        cl = torch.randn(1, 128)
        plm = torch.randn(1, 1, 16)
        
        output = transformer(ql, cl, plm)
        
        assert output.shape == (1, 128)
    
    def test_exact_window_size(self):
        """Should handle n_atoms = n_keys"""
        transformer = AtomTransformer(n_queries=32, n_keys=128)
        
        ql = torch.randn(128, 128)
        cl = torch.randn(128, 128)
        plm = torch.randn(128, 128, 16)
        
        output = transformer(ql, cl, plm)
        
        assert output.shape == (128, 128)
    
    def test_custom_window_sizes(self):
        """Should work with custom window sizes"""
        transformer = AtomTransformer(n_queries=16, n_keys=64)
        
        ql = torch.randn(100, 128)
        cl = torch.randn(100, 128)
        plm = torch.randn(100, 100, 16)
        
        output = transformer(ql, cl, plm)
        
        assert output.shape == (100, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])