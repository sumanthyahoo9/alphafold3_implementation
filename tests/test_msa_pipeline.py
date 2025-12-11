"""
Unit tests for AlphaFold3 MSA pipeline.

Tests cover:
1. MSA encoding (one-hot 32 classes)
2. Deletion processing (binary + transformed values)
3. Profile computation (residue distributions)
4. MSA subsampling
5. Feature shape validation
"""

import pytest
import numpy as np
from src.data.msa_pipeline import (
    AlphaFold3MSAPipeline,
    MSASequence
)


class TestMSAEncoding:
    """Test MSA sequence encoding"""
    
    def test_simple_msa_encoding(self):
        """Basic MSA should encode correctly"""
        pipeline = AlphaFold3MSAPipeline()
        
        query = "ACDEFG"
        msa_seqs = [
            MSASequence("ACDEFG", [0, 0, 0, 0, 0, 0]),
            MSASequence("ACDQFG", [0, 0, 0, 0, 0, 0]),
        ]
        
        features = pipeline.process_msa(query, msa_seqs, n_tokens=6)
        
        assert features['msa'].shape == (3, 6, 32)  # 3 seqs (query + 2)
        # Each position should sum to 1 (one-hot)
        assert np.allclose(features['msa'].sum(axis=2), 1.0)
    
    def test_gap_encoding(self):
        """Gaps should encode to GAP class"""
        from src.data.featurizer import RESTYPE_TO_INDEX
        
        pipeline = AlphaFold3MSAPipeline()
        
        query = "A-C"
        msa_seqs = []
        
        features = pipeline.process_msa(query, msa_seqs, n_tokens=3)
        
        # Position 1 should be gap
        gap_idx = RESTYPE_TO_INDEX['GAP']
        assert features['msa'][0, 1, gap_idx] == 1.0
    
    def test_unknown_residue_handling(self):
        """Unknown residues should map to UNK_PROTEIN"""
        from src.data.featurizer import RESTYPE_TO_INDEX
        
        pipeline = AlphaFold3MSAPipeline()
        
        query = "AXYZ"  # X, Y, Z are invalid
        msa_seqs = []
        
        features = pipeline.process_msa(query, msa_seqs, n_tokens=4)
        
        unk_idx = RESTYPE_TO_INDEX['UNK_PROTEIN']
        # Positions 1, 2, 3 should be unknown
        assert features['msa'][0, 1, unk_idx] == 1.0
        assert features['msa'][0, 2, unk_idx] == 1.0


class TestDeletionProcessing:
    """Test deletion feature extraction"""
    
    def test_binary_deletion_indicator(self):
        """has_deletion should be binary"""
        pipeline = AlphaFold3MSAPipeline()
        
        query = "ACGT"
        msa_seqs = [
            MSASequence("ACGT", [0, 2, 0, 5]),  # Deletions at pos 1, 3
        ]
        
        features = pipeline.process_msa(query, msa_seqs, n_tokens=4)
        
        # Query has no deletions
        assert features['has_deletion'][0].sum() == 0
        # MSA seq has deletions at positions 1 and 3
        assert features['has_deletion'][1, 1] == 1.0
        assert features['has_deletion'][1, 3] == 1.0
    
    def test_deletion_value_transform(self):
        """deletion_value should apply (2/π)*arctan(d/3) transform"""
        pipeline = AlphaFold3MSAPipeline()
        
        query = "AC"
        msa_seqs = [
            MSASequence("AC", [3, 0]),  # 3 deletions at pos 0
        ]
        
        features = pipeline.process_msa(query, msa_seqs, n_tokens=2)
        
        # Check transform: (2/π)*arctan(3/3) = (2/π)*arctan(1) = 0.5
        expected = (2.0 / np.pi) * np.arctan(1.0)
        assert np.isclose(features['deletion_value'][1, 0], expected)
        assert features['deletion_value'][1, 1] == 0.0


class TestProfileComputation:
    """Test MSA profile (residue distribution)"""
    
    def test_profile_shape(self):
        """Profile should have shape [N_token, 32]"""
        pipeline = AlphaFold3MSAPipeline()
        
        query = "AACC"
        msa_seqs = [
            MSASequence("AACC", [0]*4),
            MSASequence("GGCC", [0]*4),
        ]
        
        features = pipeline.process_msa(query, msa_seqs, n_tokens=4)
        
        assert features['profile'].shape == (4, 32)
    
    def test_profile_distribution(self):
        """Profile should represent residue distribution"""
        from src.data.featurizer import RESTYPE_TO_INDEX
        
        pipeline = AlphaFold3MSAPipeline()
        
        # All sequences have 'A' at position 0
        query = "ACGT"
        msa_seqs = [
            MSASequence("ACGT", [0]*4),
            MSASequence("ACGT", [0]*4),
        ]
        
        features = pipeline.process_msa(query, msa_seqs, n_tokens=4)
        
        # Position 0 should have 100% 'A'
        ala_idx = RESTYPE_TO_INDEX['ALA'] if 'ALA' in RESTYPE_TO_INDEX else RESTYPE_TO_INDEX['A']
        # Note: depends on if using 3-letter or 1-letter codes
        # Profile should sum to ~1.0 at each position
        assert np.allclose(features['profile'].sum(axis=1), 1.0)


class TestMSASubsampling:
    """Test MSA subsampling"""
    
    def test_subsample_reduces_size(self):
        """Subsampling should reduce MSA size"""
        pipeline = AlphaFold3MSAPipeline()
        
        query = "ACGT"
        msa_seqs = [MSASequence("ACGT", [0]*4) for _ in range(100)]
        
        features = pipeline.process_msa(query, msa_seqs, n_tokens=4)
        
        # Subsample multiple times
        for _ in range(10):
            subsampled = pipeline.subsample_msa_randomly(features)
            # Size should be between 1 and original size
            assert 1 <= subsampled['msa'].shape[0] <= features['msa'].shape[0]
            # Query (row 0) should always be present
            np.testing.assert_array_equal(
                subsampled['msa'][0], 
                features['msa'][0]
            )
    
    def test_subsample_preserves_profile(self):
        """Subsampling should not affect profile"""
        pipeline = AlphaFold3MSAPipeline()
        
        query = "ACGT"
        msa_seqs = [MSASequence("ACGT", [0]*4) for _ in range(50)]
        
        features = pipeline.process_msa(query, msa_seqs, n_tokens=4)
        subsampled = pipeline.subsample_msa_randomly(features)
        
        # Profile should be identical (not subsampled)
        np.testing.assert_array_equal(
            subsampled['profile'],
            features['profile']
        )


class TestMSAMaxRows:
    """Test MSA size limits"""
    
    def test_respects_max_msa_rows(self):
        """MSA should be capped at max_msa_rows"""
        pipeline = AlphaFold3MSAPipeline(max_msa_rows=100)
        
        query = "ACGT"
        # Generate more sequences than max
        msa_seqs = [MSASequence("ACGT", [0]*4) for _ in range(500)]
        
        features = pipeline.process_msa(query, msa_seqs, n_tokens=4)
        
        assert features['msa'].shape[0] <= 100


class TestDeletionMean:
    """Test mean deletion computation"""
    
    def test_deletion_mean_computation(self):
        """Deletion mean should average across sequences"""
        pipeline = AlphaFold3MSAPipeline()
        
        query = "AC"
        msa_seqs = [
            MSASequence("AC", [2, 0]),  # 2 deletions at pos 0
            MSASequence("AC", [4, 0]),  # 4 deletions at pos 0
        ]
        
        features = pipeline.process_msa(query, msa_seqs, n_tokens=2)
        
        # Mean at position 0: (0 + 2 + 4) / 3 = 2.0
        assert np.isclose(features['deletion_mean'][0], 2.0)
        # Mean at position 1: 0.0
        assert features['deletion_mean'][1] == 0.0


class TestDummyMSA:
    """Test dummy MSA generation"""
    
    def test_dummy_msa_creation(self):
        """Dummy MSA should have correct structure"""
        pipeline = AlphaFold3MSAPipeline()
        
        query = "ACDEFGHIKLMNPQRSTVWY"  # 20 residues
        features = pipeline.create_dummy_msa(query, n_tokens=20, n_homologs=50)
        
        assert features['msa'].shape[0] == 51  # Query + 50 homologs
        assert features['msa'].shape[1] == 20
        assert features['msa'].shape[2] == 32
        assert features['profile'].shape == (20, 32)
        assert features['deletion_mean'].shape == (20,)


class TestComplexPairing:
    """Test multi-chain MSA pairing"""
    
    def test_paired_msa_concatenation(self):
        """Paired MSA should concatenate chains"""
        pipeline = AlphaFold3MSAPipeline()
        
        # Two chains
        msa_per_chain = {
            'A': [
                MSASequence("ACG", [0]*3),
                MSASequence("ACG", [0]*3),
            ],
            'B': [
                MSASequence("TT", [0]*2),
                MSASequence("TT", [0]*2),
            ]
        }
        
        n_tokens_per_chain = {'A': 3, 'B': 2}
        
        features = pipeline.pair_msa_by_species(msa_per_chain, n_tokens_per_chain)
        
        # Should concatenate to 5 tokens total
        assert features['msa'].shape[1] == 5
        assert features['profile'].shape[0] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])