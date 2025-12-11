"""
Unit tests for AlphaFold3 Dataset.

Tests cover:
1. End-to-end data pipeline
2. Feature integration (tokens + featurizer + MSA)
3. PyTorch tensor conversion
4. Batch collation with padding
5. Dummy structure generation
"""

import pytest
import torch
import numpy as np
from src.data.dataset import AlphaFold3Dataset, create_dummy_structure
from src.data.tokenizer import MoleculeType


class TestDatasetCreation:
    """Test dataset initialization and basic operations"""
    
    def test_dataset_initialization(self):
        """Dataset should initialize correctly"""
        structures = [create_dummy_structure(n_residues=5)]
        dataset = AlphaFold3Dataset(structures, use_msa=True)
        
        assert len(dataset) == 1
        assert dataset.use_msa == True
    
    def test_dataset_length(self):
        """Dataset length should match number of structures"""
        structures = [
            create_dummy_structure(n_residues=5),
            create_dummy_structure(n_residues=10),
            create_dummy_structure(n_residues=15)
        ]
        dataset = AlphaFold3Dataset(structures)
        
        assert len(dataset) == 3


class TestFeatureExtraction:
    """Test end-to-end feature extraction"""
    
    def test_single_chain_features(self):
        """Single chain should produce correct features"""
        structure = create_dummy_structure(n_residues=10, n_chains=1)
        dataset = AlphaFold3Dataset([structure], use_msa=True)
        
        batch = dataset[0]
        
        # Check tensor types
        assert isinstance(batch, dict)
        assert all(isinstance(v, torch.Tensor) for v in batch.values())
        
        # Check key features present
        assert 'token_index' in batch
        assert 'restype' in batch
        assert 'ref_pos' in batch
        assert 'msa' in batch
    
    def test_multi_chain_features(self):
        """Multi-chain structure should work"""
        structure = create_dummy_structure(n_residues=5, n_chains=2)
        dataset = AlphaFold3Dataset([structure], use_msa=True)
        
        batch = dataset[0]
        
        assert batch['token_index'].shape[0] == 10  # 5 residues * 2 chains
        # Each chain has unique asym_id
        unique_asym = torch.unique(batch['asym_id'])
        assert len(unique_asym) == 2


class TestTokenFeatures:
    """Test token-level feature extraction"""
    
    def test_token_indices_monotonic(self):
        """Token indices should be monotonically increasing"""
        structure = create_dummy_structure(n_residues=10)
        dataset = AlphaFold3Dataset([structure])
        
        batch = dataset[0]
        token_indices = batch['token_index']
        
        assert torch.all(token_indices[1:] >= token_indices[:-1])
    
    def test_molecule_type_masks(self):
        """Molecule type masks should be correct"""
        structure = create_dummy_structure(n_residues=5, n_chains=1)
        dataset = AlphaFold3Dataset([structure])
        
        batch = dataset[0]
        
        # All residues are protein
        assert torch.all(batch['is_protein'] == 1.0)
        assert torch.all(batch['is_rna'] == 0.0)
        assert torch.all(batch['is_dna'] == 0.0)
        assert torch.all(batch['is_ligand'] == 0.0)
    
    def test_restype_encoding(self):
        """Restype should be one-hot encoded"""
        structure = create_dummy_structure(n_residues=5)
        dataset = AlphaFold3Dataset([structure])
        
        batch = dataset[0]
        restype = batch['restype']
        
        assert restype.shape == (5, 32)  # 5 tokens, 32 classes
        # Each position should sum to 1 (one-hot)
        assert torch.allclose(restype.sum(dim=1), torch.ones(5))


class TestAtomFeatures:
    """Test atom-level feature extraction"""
    
    def test_atom_positions_shape(self):
        """Atom positions should have correct shape"""
        structure = create_dummy_structure(n_residues=10)
        dataset = AlphaFold3Dataset([structure])
        
        batch = dataset[0]
        ref_pos = batch['ref_pos']
        
        # 10 residues * 4 atoms per residue = 40 atoms
        assert ref_pos.shape == (40, 3)
    
    def test_atom_elements_one_hot(self):
        """Element encoding should be one-hot"""
        structure = create_dummy_structure(n_residues=5)
        dataset = AlphaFold3Dataset([structure])
        
        batch = dataset[0]
        ref_element = batch['ref_element']
        
        # Should be [N_atom, 128] one-hot
        assert ref_element.shape[1] == 128
        # Each atom should have exactly one element
        assert torch.all(ref_element.sum(dim=1) == 1.0)


class TestMSAFeatures:
    """Test MSA feature extraction"""
    
    def test_msa_shape_with_msa(self):
        """MSA features should have correct shape"""
        structure = create_dummy_structure(n_residues=10, n_msa=50)
        dataset = AlphaFold3Dataset([structure], use_msa=True, training=False)
        
        batch = dataset[0]
        msa = batch['msa']
        
        # Should have query + MSA sequences
        assert msa.shape[0] <= 51  # Query + 50 homologs
        assert msa.shape[1] == 10  # 10 tokens
        assert msa.shape[2] == 32  # 32 restype classes
    
    def test_msa_subsampling_in_training(self):
        """MSA should subsample in training mode"""
        structure = create_dummy_structure(n_residues=10, n_msa=100)
        dataset = AlphaFold3Dataset([structure], use_msa=True, training=True)
        
        # Get multiple samples - sizes should vary
        sizes = []
        for _ in range(10):
            batch = dataset[0]
            sizes.append(batch['msa'].shape[0])
        
        # Should have variation due to random subsampling
        assert len(set(sizes)) > 1
    
    def test_profile_shape(self):
        """Profile should have shape [N_token, 32]"""
        structure = create_dummy_structure(n_residues=10)
        dataset = AlphaFold3Dataset([structure], use_msa=True)
        
        batch = dataset[0]
        profile = batch['profile']
        
        assert profile.shape == (10, 32)
    
    def test_deletion_features(self):
        """Deletion features should be present"""
        structure = create_dummy_structure(n_residues=10)
        dataset = AlphaFold3Dataset([structure], use_msa=True)
        
        batch = dataset[0]
        
        assert 'has_deletion' in batch
        assert 'deletion_value' in batch
        assert 'deletion_mean' in batch


class TestBatchCollation:
    """Test batching with padding"""
    
    def test_collate_different_sizes(self):
        """Collate should pad to max size in batch"""
        structures = [
            create_dummy_structure(n_residues=5),
            create_dummy_structure(n_residues=10),
            create_dummy_structure(n_residues=15)
        ]
        dataset = AlphaFold3Dataset(structures, use_msa=False)
        
        batch_list = [dataset[i] for i in range(3)]
        collated = AlphaFold3Dataset.collate_fn(batch_list)
        
        # Should pad to max (15 tokens)
        assert collated['token_index'].shape == (3, 15)
        assert collated['restype'].shape == (3, 15, 32)
    
    def test_collate_padding_with_zeros(self):
        """Padded positions should be zero"""
        structures = [
            create_dummy_structure(n_residues=5),
            create_dummy_structure(n_residues=10)
        ]
        dataset = AlphaFold3Dataset(structures, use_msa=False)
        
        batch_list = [dataset[i] for i in range(2)]
        collated = AlphaFold3Dataset.collate_fn(batch_list)
        
        # First example padded from 5 to 10
        # Positions 5-9 should be zero
        assert torch.all(collated['token_index'][0, 5:] == 0)
    
    def test_collate_msa_features(self):
        """MSA features should batch correctly"""
        structures = [
            create_dummy_structure(n_residues=5, n_msa=20),
            create_dummy_structure(n_residues=8, n_msa=30)
        ]
        dataset = AlphaFold3Dataset(structures, use_msa=True, training=False)
        
        batch_list = [dataset[i] for i in range(2)]
        collated = AlphaFold3Dataset.collate_fn(batch_list)
        
        # Should pad to max dimensions
        assert collated['msa'].ndim == 4  # [batch, n_msa, n_token, 32]
        assert collated['msa'].shape[0] == 2  # Batch size


class TestDummyStructureGeneration:
    """Test dummy structure generation utility"""
    
    def test_dummy_structure_format(self):
        """Dummy structure should have correct format"""
        structure = create_dummy_structure(n_residues=5, n_chains=1)
        
        assert 'chains' in structure
        assert len(structure['chains']) == 1
        assert len(structure['chains'][0]['residues']) == 5
    
    def test_dummy_multi_chain(self):
        """Dummy multi-chain should work"""
        structure = create_dummy_structure(n_residues=5, n_chains=3)
        
        assert len(structure['chains']) == 3
        # Each chain should have unique chain_id
        chain_ids = [c['chain_id'] for c in structure['chains']]
        assert len(set(chain_ids)) == 3
    
    def test_dummy_msa_generation(self):
        """Dummy MSA should be generated"""
        structure = create_dummy_structure(n_residues=10, n_msa=50)
        
        msa = structure['chains'][0]['msa']
        assert len(msa) == 50


class TestEndToEndPipeline:
    """Test complete data pipeline"""
    
    def test_full_pipeline(self):
        """Complete pipeline should run without errors"""
        # Create dataset
        structures = [create_dummy_structure(n_residues=10, n_chains=2, n_msa=50)]
        dataset = AlphaFold3Dataset(structures, use_msa=True, training=False)
        
        # Get single example
        example = dataset[0]
        
        # Check all required features present
        required_keys = [
            'token_index', 'residue_index', 'asym_id', 'entity_id', 'sym_id',
            'restype', 'is_protein', 'is_rna', 'is_dna', 'is_ligand',
            'ref_pos', 'ref_mask', 'ref_element', 'ref_charge',
            'msa', 'has_deletion', 'deletion_value', 'profile', 'deletion_mean'
        ]
        
        for key in required_keys:
            assert key in example, f"Missing required key: {key}"
    
    def test_dataloader_compatibility(self):
        """Dataset should work with PyTorch DataLoader"""
        from torch.utils.data import DataLoader
        
        structures = [create_dummy_structure(n_residues=5) for _ in range(4)]
        dataset = AlphaFold3Dataset(structures, use_msa=False)
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=AlphaFold3Dataset.collate_fn
        )
        
        batch = next(iter(dataloader))
        
        assert batch['token_index'].shape[0] == 2  # Batch size
        assert isinstance(batch['token_index'], torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])