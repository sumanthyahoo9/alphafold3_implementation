"""
Unit tests for AlphaFold3 tokenizer.

Tests cover:
1. Standard residue tokenization (1 residue = 1 token)
2. Per-atom tokenization for ligands/modified residues
3. Centre atom identification
4. Token feature extraction
5. Atom-to-token mapping
"""

import pytest
from src.data.tokenizer import AlphaFold3Tokenizer, MoleculeType


class TestStandardResidueTokenization:
    """Test standard amino acid and nucleotide tokenization"""
    
    def test_standard_protein_single_token(self):
        """Standard amino acid should create 1 token with all atoms"""
        tokenizer = AlphaFold3Tokenizer()
        
        residues = [{
            'restype': 'ALA',
            'residue_index': 1,
            'atoms': [
                {'atom_index': 0, 'atom_name': 'N', 'element': 'N'},
                {'atom_index': 1, 'atom_name': 'CA', 'element': 'C'},
                {'atom_index': 2, 'atom_name': 'C', 'element': 'C'},
                {'atom_index': 3, 'atom_name': 'O', 'element': 'O'},
            ]
        }]
        
        tokens = tokenizer.tokenize_chain(
            chain_id='A', asym_id=0, entity_id=0, sym_id=0,
            residues=residues, mol_type=MoleculeType.PROTEIN
        )
        
        assert len(tokens) == 1
        assert tokens[0].is_standard == True
        assert len(tokens[0].atom_indices) == 4
        assert tokens[0].centre_atom_index == 1  # CA atom
    
    def test_standard_rna_single_token(self):
        """Standard RNA nucleotide should create 1 token"""
        tokenizer = AlphaFold3Tokenizer()
        
        residues = [{
            'restype': 'A',
            'residue_index': 1,
            'atoms': [
                {'atom_index': 0, 'atom_name': "C1'", 'element': 'C'},
                {'atom_index': 1, 'atom_name': "C2'", 'element': 'C'},
                {'atom_index': 2, 'atom_name': "C3'", 'element': 'C'},
            ]
        }]
        
        tokens = tokenizer.tokenize_chain(
            chain_id='B', asym_id=1, entity_id=1, sym_id=0,
            residues=residues, mol_type=MoleculeType.RNA
        )
        
        assert len(tokens) == 1
        assert tokens[0].centre_atom_index == 0  # C1' atom


class TestPerAtomTokenization:
    """Test per-atom tokenization for ligands and modified residues"""
    
    def test_ligand_per_atom_tokens(self):
        """Ligand should create N tokens for N atoms"""
        tokenizer = AlphaFold3Tokenizer()
        
        residues = [{
            'restype': 'LIG',
            'residue_index': 1,
            'atoms': [
                {'atom_index': 0, 'atom_name': 'C1', 'element': 'C'},
                {'atom_index': 1, 'atom_name': 'O1', 'element': 'O'},
                {'atom_index': 2, 'atom_name': 'N1', 'element': 'N'},
            ]
        }]
        
        tokens = tokenizer.tokenize_chain(
            chain_id='L', asym_id=2, entity_id=2, sym_id=0,
            residues=residues, mol_type=MoleculeType.LIGAND
        )
        
        assert len(tokens) == 3  # 1 token per atom
        assert all(not t.is_standard for t in tokens)
        assert all(len(t.atom_indices) == 1 for t in tokens)
        # Each atom is its own centre
        assert tokens[0].centre_atom_index == 0
        assert tokens[1].centre_atom_index == 1
        assert tokens[2].centre_atom_index == 2
    
    def test_modified_residue_per_atom(self):
        """Modified amino acid should use per-atom tokenization"""
        tokenizer = AlphaFold3Tokenizer()
        
        residues = [{
            'restype': 'MSE',  # Selenomethionine (modified)
            'residue_index': 10,
            'atoms': [
                {'atom_index': 0, 'atom_name': 'N', 'element': 'N'},
                {'atom_index': 1, 'atom_name': 'CA', 'element': 'C'},
            ]
        }]
        
        tokens = tokenizer.tokenize_chain(
            chain_id='A', asym_id=0, entity_id=0, sym_id=0,
            residues=residues, mol_type=MoleculeType.PROTEIN
        )
        
        assert len(tokens) == 2  # Per-atom tokenization
        assert all(not t.is_standard for t in tokens)


class TestTokenIndexing:
    """Test token and atom indexing"""
    
    def test_monotonic_token_indices(self):
        """Token indices should increase monotonically"""
        tokenizer = AlphaFold3Tokenizer()
        
        residues = [
            {
                'restype': 'ALA',
                'residue_index': 1,
                'atoms': [{'atom_index': 0, 'atom_name': 'CA', 'element': 'C'}]
            },
            {
                'restype': 'GLY',
                'residue_index': 2,
                'atoms': [{'atom_index': 1, 'atom_name': 'CA', 'element': 'C'}]
            },
        ]
        
        tokens = tokenizer.tokenize_chain(
            chain_id='A', asym_id=0, entity_id=0, sym_id=0,
            residues=residues, mol_type=MoleculeType.PROTEIN
        )
        
        assert tokens[0].token_index == 0
        assert tokens[1].token_index == 1
    
    def test_atom_to_token_mapping(self):
        """Atom-to-token mapping should be correct"""
        tokenizer = AlphaFold3Tokenizer()
        
        residues = [{
            'restype': 'ALA',
            'residue_index': 1,
            'atoms': [
                {'atom_index': 10, 'atom_name': 'N', 'element': 'N'},
                {'atom_index': 11, 'atom_name': 'CA', 'element': 'C'},
            ]
        }]
        
        tokenizer.tokenize_chain(
            chain_id='A', asym_id=0, entity_id=0, sym_id=0,
            residues=residues, mol_type=MoleculeType.PROTEIN
        )
        
        mapping = tokenizer.get_atom_to_token_mapping()
        assert mapping[10] == 0  # Both atoms map to token 0
        assert mapping[11] == 0


class TestTokenFeatures:
    """Test feature extraction for model input"""
    
    def test_feature_shapes(self):
        """Feature arrays should have correct shapes"""
        tokenizer = AlphaFold3Tokenizer()
        
        residues = [
            {
                'restype': 'ALA',
                'residue_index': 1,
                'atoms': [{'atom_index': 0, 'atom_name': 'CA', 'element': 'C'}]
            },
            {
                'restype': 'GLY',
                'residue_index': 2,
                'atoms': [{'atom_index': 1, 'atom_name': 'CA', 'element': 'C'}]
            },
        ]
        
        tokenizer.tokenize_chain(
            chain_id='A', asym_id=0, entity_id=0, sym_id=0,
            residues=residues, mol_type=MoleculeType.PROTEIN
        )
        
        features = tokenizer.get_token_features()
        
        n_tokens = 2
        assert features['token_index'].shape == (n_tokens,)
        assert features['residue_index'].shape == (n_tokens,)
        assert features['asym_id'].shape == (n_tokens,)
        assert features['is_protein'].shape == (n_tokens,)
        assert features['is_ligand'].shape == (n_tokens,)
    
    def test_molecule_type_masks(self):
        """Molecule type masks should be correct"""
        tokenizer = AlphaFold3Tokenizer()
        
        # Add protein residue
        tokenizer.tokenize_chain(
            chain_id='A', asym_id=0, entity_id=0, sym_id=0,
            residues=[{
                'restype': 'ALA',
                'residue_index': 1,
                'atoms': [{'atom_index': 0, 'atom_name': 'CA', 'element': 'C'}]
            }],
            mol_type=MoleculeType.PROTEIN
        )
        
        # Add ligand
        tokenizer.tokenize_chain(
            chain_id='L', asym_id=1, entity_id=1, sym_id=0,
            residues=[{
                'restype': 'LIG',
                'residue_index': 1,
                'atoms': [{'atom_index': 1, 'atom_name': 'C1', 'element': 'C'}]
            }],
            mol_type=MoleculeType.LIGAND
        )
        
        features = tokenizer.get_token_features()
        
        assert features['is_protein'][0] == 1.0
        assert features['is_protein'][1] == 0.0
        assert features['is_ligand'][0] == 0.0
        assert features['is_ligand'][1] == 1.0


class TestMultiChain:
    """Test multi-chain tokenization"""
    
    def test_multiple_chains_correct_asym_ids(self):
        """Each chain should have unique asym_id"""
        tokenizer = AlphaFold3Tokenizer()
        
        residue_template = [{
            'restype': 'ALA',
            'residue_index': 1,
            'atoms': [{'atom_index': None, 'atom_name': 'CA', 'element': 'C'}]
        }]
        
        # Chain A
        res_a = residue_template.copy()
        res_a[0]['atoms'][0]['atom_index'] = 0
        tokenizer.tokenize_chain('A', asym_id=0, entity_id=0, sym_id=0, 
                                 residues=res_a, mol_type=MoleculeType.PROTEIN)
        
        # Chain B
        res_b = residue_template.copy()
        res_b[0]['atoms'][0]['atom_index'] = 1
        tokenizer.tokenize_chain('B', asym_id=1, entity_id=0, sym_id=1,
                                 residues=res_b, mol_type=MoleculeType.PROTEIN)
        
        features = tokenizer.get_token_features()
        assert features['asym_id'][0] == 0
        assert features['asym_id'][1] == 1
        assert features['sym_id'][0] == 0
        assert features['sym_id'][1] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])