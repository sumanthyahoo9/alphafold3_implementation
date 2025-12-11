"""
Unit tests for AlphaFold3 featurizer.

Tests cover:
1. Restype one-hot encoding (32 classes)
2. Reference conformer features (positions, elements, charges)
3. Atom name character encoding
4. Reference space UID computation
5. Feature shape validation
"""

import pytest
import numpy as np
from src.data.featurizer import (
    AlphaFold3Featurizer, 
    Atom, 
    RESTYPE_TO_INDEX,
    RESTYPE_VOCAB
)


class TestRestypeEncoding:
    """Test residue type one-hot encoding"""
    
    def test_standard_amino_acid_encoding(self):
        """Standard amino acids should encode correctly"""
        featurizer = AlphaFold3Featurizer()
        restypes = ['ALA', 'GLY', 'VAL']
        
        encoding = featurizer._encode_restypes(restypes)
        
        assert encoding.shape == (3, 32)
        assert encoding[0, RESTYPE_TO_INDEX['ALA']] == 1.0
        assert encoding[1, RESTYPE_TO_INDEX['GLY']] == 1.0
        assert encoding[2, RESTYPE_TO_INDEX['VAL']] == 1.0
        assert encoding.sum(axis=1).tolist() == [1.0, 1.0, 1.0]
    
    def test_rna_encoding(self):
        """RNA bases should encode correctly"""
        featurizer = AlphaFold3Featurizer()
        restypes = ['A', 'U', 'G', 'C']
        
        encoding = featurizer._encode_restypes(restypes)
        
        assert encoding.shape == (4, 32)
        for i, base in enumerate(restypes):
            assert encoding[i, RESTYPE_TO_INDEX[base]] == 1.0
    
    def test_dna_encoding(self):
        """DNA bases should encode correctly"""
        featurizer = AlphaFold3Featurizer()
        restypes = ['DA', 'DT', 'DG', 'DC']
        
        encoding = featurizer._encode_restypes(restypes)
        
        assert encoding.shape == (4, 32)
        for i, base in enumerate(restypes):
            assert encoding[i, RESTYPE_TO_INDEX[base]] == 1.0
    
    def test_unknown_restype_handling(self):
        """Unknown restypes should map to UNK_PROTEIN"""
        featurizer = AlphaFold3Featurizer()
        restypes = ['UNKNOWN_LIG']
        
        encoding = featurizer._encode_restypes(restypes)
        
        assert encoding[0, RESTYPE_TO_INDEX['UNK_PROTEIN']] == 1.0
    
    def test_vocab_size(self):
        """Vocabulary should have exactly 32 classes"""
        assert len(RESTYPE_VOCAB) == 32


class TestReferenceFeatures:
    """Test reference conformer feature extraction"""
    
    def test_atom_positions(self):
        """Atom positions should be extracted correctly"""
        featurizer = AlphaFold3Featurizer()
        
        atoms = [
            Atom(0, 'CA', 'C', 6, np.array([1.0, 2.0, 3.0]), 0.0, 1.0),
            Atom(1, 'N', 'N', 7, np.array([4.0, 5.0, 6.0]), 0.0, 1.0)
        ]
        
        ref_features = featurizer._extract_reference_features(atoms)
        
        assert ref_features['positions'].shape == (2, 3)
        np.testing.assert_array_equal(ref_features['positions'][0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(ref_features['positions'][1], [4.0, 5.0, 6.0])
    
    def test_element_one_hot(self):
        """Element atomic numbers should be one-hot encoded"""
        featurizer = AlphaFold3Featurizer()
        
        atoms = [
            Atom(0, 'C', 'C', 6, np.zeros(3), 0.0, 1.0),   # Carbon
            Atom(1, 'N', 'N', 7, np.zeros(3), 0.0, 1.0),   # Nitrogen
            Atom(2, 'O', 'O', 8, np.zeros(3), 0.0, 1.0),   # Oxygen
        ]
        
        ref_features = featurizer._extract_reference_features(atoms)
        
        assert ref_features['element'].shape == (3, 128)
        assert ref_features['element'][0, 6] == 1.0  # Carbon
        assert ref_features['element'][1, 7] == 1.0  # Nitrogen  
        assert ref_features['element'][2, 8] == 1.0  # Oxygen
        assert ref_features['element'].sum(axis=1).tolist() == [1.0, 1.0, 1.0]
    
    def test_atom_mask(self):
        """Atom masks should indicate presence"""
        featurizer = AlphaFold3Featurizer()
        
        atoms = [
            Atom(0, 'CA', 'C', 6, np.zeros(3), 0.0, 1.0),  # Present
            Atom(1, 'N', 'N', 7, np.zeros(3), 0.0, 0.0),   # Missing
        ]
        
        ref_features = featurizer._extract_reference_features(atoms)
        
        assert ref_features['mask'][0] == 1.0
        assert ref_features['mask'][1] == 0.0
    
    def test_atom_charges(self):
        """Atom charges should be extracted"""
        featurizer = AlphaFold3Featurizer()
        
        atoms = [
            Atom(0, 'CA', 'C', 6, np.zeros(3), 0.5, 1.0),
            Atom(1, 'O', 'O', 8, np.zeros(3), -0.5, 1.0),
        ]
        
        ref_features = featurizer._extract_reference_features(atoms)
        
        assert ref_features['charge'][0] == 0.5
        assert ref_features['charge'][1] == -0.5


class TestAtomNameEncoding:
    """Test atom name character encoding"""
    
    def test_short_name_padding(self):
        """Short atom names should be padded to length 4"""
        featurizer = AlphaFold3Featurizer()
        
        encoding = featurizer._encode_atom_name('CA')
        
        assert encoding.shape == (4, 64)
        # First 2 chars should be encoded, last 2 should be spaces
    
    def test_long_name_truncation(self):
        """Long atom names should be truncated to length 4"""
        featurizer = AlphaFold3Featurizer()
        
        encoding = featurizer._encode_atom_name('VERYLONGNAME')
        
        assert encoding.shape == (4, 64)
        # Only first 4 characters encoded
    
    def test_character_encoding(self):
        """Characters should encode as ord(c) - 32"""
        featurizer = AlphaFold3Featurizer()
        
        encoding = featurizer._encode_atom_name('CA')
        
        # 'C' = ASCII 67, 67 - 32 = 35
        # 'A' = ASCII 65, 65 - 32 = 33
        assert encoding[0, 35] == 1.0  # C
        assert encoding[1, 33] == 1.0  # A


class TestRefSpaceUID:
    """Test reference space UID computation"""
    
    def test_unique_uids_per_residue(self):
        """Each (chain, residue) pair should get unique UID"""
        featurizer = AlphaFold3Featurizer()
        
        # 2 atoms in token 0, 2 atoms in token 1
        atom_to_token_map = np.array([0, 0, 1, 1])
        asym_ids = np.array([0, 0])  # Same chain
        residue_indices = np.array([1, 2])  # Different residues
        
        uids = featurizer._compute_ref_space_uid(
            atom_to_token_map, asym_ids, residue_indices
        )
        
        assert uids.shape == (4,)
        assert uids[0] == uids[1]  # Same residue
        assert uids[2] == uids[3]  # Same residue
        assert uids[0] != uids[2]  # Different residues
    
    def test_different_chains_different_uids(self):
        """Different chains should have different UIDs even with same residue index"""
        featurizer = AlphaFold3Featurizer()
        
        atom_to_token_map = np.array([0, 1])
        asym_ids = np.array([0, 1])  # Different chains
        residue_indices = np.array([1, 1])  # Same residue number
        
        uids = featurizer._compute_ref_space_uid(
            atom_to_token_map, asym_ids, residue_indices
        )
        
        assert uids[0] != uids[1]  # Different chains


class TestFeaturizationPipeline:
    """Test complete featurization pipeline"""
    
    def test_complete_feature_extraction(self):
        """Full pipeline should produce all required features"""
        featurizer = AlphaFold3Featurizer()
        
        # Minimal token features
        token_features = {
            'token_index': np.array([0, 1]),
            'residue_index': np.array([1, 2]),
            'asym_id': np.array([0, 0]),
            'entity_id': np.array([0, 0]),
            'sym_id': np.array([0, 0]),
            'is_protein': np.array([1.0, 1.0]),
            'is_rna': np.array([0.0, 0.0]),
            'is_dna': np.array([0.0, 0.0]),
            'is_ligand': np.array([0.0, 0.0]),
        }
        
        atoms = [
            Atom(0, 'CA', 'C', 6, np.array([1.0, 0.0, 0.0]), 0.0, 1.0),
            Atom(1, 'N', 'N', 7, np.array([0.0, 1.0, 0.0]), 0.0, 1.0),
        ]
        
        restypes = ['ALA', 'GLY']
        atom_to_token_map = np.array([0, 1])
        
        features = featurizer.featurize(
            token_features, atoms, restypes, atom_to_token_map
        )
        
        # Check all required features present
        assert 'token_index' in features
        assert 'residue_index' in features
        assert 'restype' in features
        assert 'ref_pos' in features
        assert 'ref_mask' in features
        assert 'ref_element' in features
        assert 'ref_charge' in features
        assert 'ref_space_uid' in features
    
    def test_feature_shapes(self):
        """All features should have correct shapes"""
        featurizer = AlphaFold3Featurizer()
        
        n_tokens = 3
        n_atoms = 5
        
        token_features = {
            'token_index': np.arange(n_tokens),
            'residue_index': np.arange(n_tokens),
            'asym_id': np.zeros(n_tokens),
            'entity_id': np.zeros(n_tokens),
            'sym_id': np.zeros(n_tokens),
            'is_protein': np.ones(n_tokens),
            'is_rna': np.zeros(n_tokens),
            'is_dna': np.zeros(n_tokens),
            'is_ligand': np.zeros(n_tokens),
        }
        
        atoms = [
            Atom(i, 'CA', 'C', 6, np.zeros(3), 0.0, 1.0)
            for i in range(n_atoms)
        ]
        
        restypes = ['ALA', 'GLY', 'VAL']
        atom_to_token_map = np.array([0, 0, 1, 1, 2])
        
        features = featurizer.featurize(
            token_features, atoms, restypes, atom_to_token_map
        )
        
        assert features['restype'].shape == (n_tokens, 32)
        assert features['ref_pos'].shape == (n_atoms, 3)
        assert features['ref_mask'].shape == (n_atoms,)
        assert features['ref_element'].shape == (n_atoms, 128)
        assert features['ref_charge'].shape == (n_atoms,)
        assert features['ref_atom_name_chars'].shape == (n_atoms, 4, 64)
        assert features['ref_space_uid'].shape == (n_atoms,)


class TestPlaceholderFeatures:
    """Test placeholder feature generation"""
    
    def test_placeholder_msa_shapes(self):
        """Placeholder MSA features should have correct shapes"""
        featurizer = AlphaFold3Featurizer()
        n_tokens = 10
        
        msa_features = featurizer.create_placeholder_msa_features(n_tokens)
        
        assert msa_features['msa'].shape[1] == n_tokens
        assert msa_features['profile'].shape == (n_tokens, 32)
        assert msa_features['deletion_mean'].shape == (n_tokens,)
    
    def test_placeholder_bond_shapes(self):
        """Placeholder bond features should be square matrix"""
        featurizer = AlphaFold3Featurizer()
        n_tokens = 10
        
        bond_features = featurizer.create_placeholder_bond_features(n_tokens)
        
        assert bond_features['token_bonds'].shape == (n_tokens, n_tokens)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])