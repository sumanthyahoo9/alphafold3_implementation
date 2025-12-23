"""
AlphaFold3 Dataset

PyTorch Dataset that combines tokenization, featurization, and MSA processing
into a single pipeline for training/inference.

Integrates:
- Tokenizer: Molecular structure → tokens
- Featurizer: Tokens → model input features (Table 5)
- MSA Pipeline: Evolutionary sequences → MSA features
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.tokenizer import AlphaFold3Tokenizer, Token, MoleculeType
from src.data.featurizer import AlphaFold3Featurizer, Atom
from src.data.msa_pipeline import AlphaFold3MSAPipeline, MSASequence


class AlphaFold3Dataset(Dataset):
    """
    PyTorch Dataset for AlphaFold3 training/inference.
    
    Converts raw molecular structures into model-ready tensors.
    
    Usage:
        dataset = AlphaFold3Dataset(structures)
        batch = dataset[0]  # Returns dict of torch tensors
    """
    
    def __init__(
        self,
        structures: List[Dict],
        use_msa: bool = True,
        use_templates: bool = False,
        max_msa_rows: int = 16384,
        training: bool = True
    ):
        """
        Args:
            structures: List of structure dicts (see format below)
            use_msa: Whether to include MSA features
            use_templates: Whether to include template features
            max_msa_rows: Maximum MSA sequences
            training: Whether in training mode (enables MSA subsampling)
            
        Structure format:
        {
            'chains': [
                {
                    'chain_id': 'A',
                    'mol_type': MoleculeType.PROTEIN,
                    'sequence': 'ACDEFGHIKLM',
                    'residues': [
                        {
                            'restype': 'ALA',
                            'residue_index': 1,
                            'atoms': [
                                {
                                    'atom_index': 0,
                                    'atom_name': 'CA',
                                    'element': 'C',
                                    'atomic_number': 6,
                                    'position': [x, y, z],
                                    'charge': 0.0
                                },
                                ...
                            ]
                        },
                        ...
                    ],
                    'msa': [
                        MSASequence(sequence, deletions, species_id),
                        ...
                    ]
                },
                ...
            ]
        }
        """
        self.structures = structures
        self.use_msa = use_msa
        self.use_templates = use_templates
        self.max_msa_rows = max_msa_rows
        self.training = training
        
        # Initialize processors
        self.tokenizer = AlphaFold3Tokenizer()
        self.featurizer = AlphaFold3Featurizer()
        self.msa_pipeline = AlphaFold3MSAPipeline(max_msa_rows=max_msa_rows)
    
    def __len__(self) -> int:
        return len(self.structures)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example as dict of torch tensors.
        
        Returns:
            Dict with keys matching AF3 Table 5 input features
        """
        structure = self.structures[idx]
        
        # Reset tokenizer for new structure
        self.tokenizer.reset()
        
        # Process each chain
        all_atoms = []
        all_restypes = []
        
        entity_id_counter = 0
        chain_to_entity = {}
        
        for chain_idx, chain_data in enumerate(structure['chains']):
            chain_id = chain_data['chain_id']
            mol_type = chain_data['mol_type']
            sequence = chain_data['sequence']
            residues = chain_data['residues']
            
            # Assign entity_id (unique per sequence)
            if sequence not in chain_to_entity.values():
                entity_id = entity_id_counter
                chain_to_entity[sequence] = entity_id
                entity_id_counter += 1
            else:
                entity_id = chain_to_entity[sequence]
            
            # Compute sym_id (index within chains of same sequence)
            sym_id = sum(1 for s in list(chain_to_entity.values())[:chain_idx] if s == entity_id)
            
            # Tokenize chain
            tokens = self.tokenizer.tokenize_chain(
                chain_id=chain_id,
                asym_id=chain_idx,
                entity_id=entity_id,
                sym_id=sym_id,
                residues=residues,
                mol_type=mol_type
            )
            
            # Extract atoms
            for residue in residues:
                for atom_data in residue['atoms']:
                    atom = Atom(
                        atom_index=atom_data['atom_index'],
                        atom_name=atom_data['atom_name'],
                        element=atom_data['element'],
                        atomic_number=atom_data['atomic_number'],
                        position=np.array(atom_data['position']),
                        charge=atom_data.get('charge', 0.0),
                        mask=1.0  # Present
                    )
                    all_atoms.append(atom)
            
            # Collect restypes per token
            for token in tokens:
                all_restypes.append(token.restype)
        
        # Get token features
        token_features = self.tokenizer.get_token_features()
        atom_to_token_map = self.tokenizer.get_atom_to_token_mapping()
        
        # Featurize
        features = self.featurizer.featurize(
            token_features=token_features,
            atoms=all_atoms,
            restypes=all_restypes,
            atom_to_token_map=atom_to_token_map
        )
        
        # Add MSA features
        if self.use_msa:
            msa_features = self._process_msa(structure)
            features.update(msa_features)
        else:
            # Placeholder MSA
            n_tokens = len(token_features['token_index'])
            features.update(
                self.featurizer.create_placeholder_msa_features(n_tokens)
            )
        
        # Add template features
        if self.use_templates:
            # TODO: Implement template processing
            pass
        else:
            n_tokens = len(token_features['token_index'])
            features.update(
                self.featurizer.create_placeholder_template_features(n_tokens)
            )
        
        # Add bond features
        n_tokens = len(token_features['token_index'])
        features.update(
            self.featurizer.create_placeholder_bond_features(n_tokens)
        )
        
        # Convert to torch tensors
        tensors = self._to_torch(features)
        
        return tensors
    
    def _process_msa(self, structure: Dict) -> Dict[str, np.ndarray]:
        """Process MSA for all chains"""
        # Simple case: single chain
        if len(structure['chains']) == 1:
            chain = structure['chains'][0]
            sequence = chain['sequence']
            msa_seqs = chain.get('msa', [])
            n_tokens = len(self.tokenizer.tokens)
            
            features = self.msa_pipeline.process_msa(
                query_sequence=sequence,
                msa_sequences=msa_seqs,
                n_tokens=n_tokens
            )
            
            # Subsample during training
            if self.training:
                features = self.msa_pipeline.subsample_msa_randomly(features)
            
            return features
        
        # Multi-chain: pair MSAs
        else:
            msa_per_chain = {}
            n_tokens_per_chain = {}
            
            for chain in structure['chains']:
                chain_id = chain['chain_id']
                msa_per_chain[chain_id] = chain.get('msa', [])
                # Count tokens for this chain
                chain_tokens = [t for t in self.tokenizer.tokens if t.asym_id == chain['asym_id']]
                n_tokens_per_chain[chain_id] = len(chain_tokens)
            
            features = self.msa_pipeline.pair_msa_by_species(
                msa_per_chain,
                n_tokens_per_chain
            )
            
            if self.training:
                features = self.msa_pipeline.subsample_msa_randomly(features)
            
            return features
    
    def _to_torch(self, features: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to torch tensors"""
        tensors = {}
        
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                tensors[key] = torch.from_numpy(value)
            else:
                tensors[key] = torch.tensor(value)
        
        return tensors
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with padding.
        
        Pads sequences to max length in batch.
        """
        # Find max dimensions
        max_n_tokens = max(b['token_index'].shape[0] for b in batch)
        max_n_atoms = max(b['ref_pos'].shape[0] for b in batch)
        max_n_msa = max(b['msa'].shape[0] for b in batch)
        
        batch_size = len(batch)
        
        # Initialize padded tensors
        collated = {}
        
        for key in batch[0].keys():
            example_tensor = batch[0][key]
            
            # Determine shape based on feature type
            if key == 'token_bonds':
                # Special case: 2D token-token matrix [N_token, N_token]
                shape = (batch_size, max_n_tokens, max_n_tokens)
                padded = torch.zeros(shape, dtype=example_tensor.dtype)
                for i, b in enumerate(batch):
                    n = b[key].shape[0]
                    padded[i, :n, :n] = b[key]
                    
            elif 'token' in key or key in ['restype', 'is_protein', 'is_rna', 'is_dna', 'is_ligand', 'asym_id', 'entity_id', 'sym_id', 'residue_index']:
                # Token-level features [N_token, ...]
                shape = (batch_size, max_n_tokens) + example_tensor.shape[1:]
                padded = torch.zeros(shape, dtype=example_tensor.dtype)
                for i, b in enumerate(batch):
                    n = b[key].shape[0]
                    if example_tensor.ndim == 1:
                        padded[i, :n] = b[key]
                    elif example_tensor.ndim == 2:
                        padded[i, :n, :] = b[key]
                    else:
                        padded[i, :n] = b[key]
                    
            elif 'ref_' in key:
                # Atom-level features [N_atom, ...]
                shape = (batch_size, max_n_atoms) + example_tensor.shape[1:]
                padded = torch.zeros(shape, dtype=example_tensor.dtype)
                for i, b in enumerate(batch):
                    n = b[key].shape[0]
                    padded[i, :n] = b[key]
                    
            elif key in ['msa', 'has_deletion', 'deletion_value']:
                # MSA features [N_msa, N_token, ...]
                if example_tensor.ndim == 3:  # [N_msa, N_token, 32]
                    shape = (batch_size, max_n_msa, max_n_tokens) + example_tensor.shape[2:]
                else:  # [N_msa, N_token]
                    shape = (batch_size, max_n_msa, max_n_tokens)
                padded = torch.zeros(shape, dtype=example_tensor.dtype)
                for i, b in enumerate(batch):
                    tensor = b[key]
                    if tensor.ndim == 3:
                        n_msa, n_token = tensor.shape[:2]
                        padded[i, :n_msa, :n_token] = tensor
                    else:
                        n_msa, n_token = tensor.shape
                        padded[i, :n_msa, :n_token] = tensor
                        
            elif key in ['profile', 'deletion_mean']:
                # Profile features [N_token, ...]
                shape = (batch_size, max_n_tokens) + example_tensor.shape[1:]
                padded = torch.zeros(shape, dtype=example_tensor.dtype)
                for i, b in enumerate(batch):
                    n = b[key].shape[0]
                    padded[i, :n] = b[key]
                    
            elif 'template' in key:
                # Template features: [N_templ, N_token, ...]
                # Special case: when N_templ=0, shapes differ in token dim but that's OK
                n_templ_first = example_tensor.shape[0]
                
                if n_templ_first == 0:
                    # All empty templates - pad token dimensions anyway for consistency
                    if 'distogram' in key or 'unit_vector' in key:
                        # [0, N_token, N_token, ...]
                        extra_dims = example_tensor.shape[3:]
                        shape = (batch_size, 0, max_n_tokens, max_n_tokens) + extra_dims
                    elif example_tensor.ndim == 3:
                        # [0, N_token, features]
                        extra_dims = example_tensor.shape[2:]
                        shape = (batch_size, 0, max_n_tokens) + extra_dims
                    elif example_tensor.ndim == 2:
                        # [0, N_token]
                        shape = (batch_size, 0, max_n_tokens)
                    else:
                        # Unknown shape - try stacking
                        padded = torch.stack([b[key] for b in batch])
                        collated[key] = padded
                        continue
                    
                    # Create empty padded tensor
                    padded = torch.zeros(shape, dtype=example_tensor.dtype)
                    
                else:
                    # Non-empty templates - check if padding needed
                    shapes_match = all(b[key].shape == example_tensor.shape for b in batch)
                    
                    if shapes_match:
                        # All same shape - just stack
                        padded = torch.stack([b[key] for b in batch])
                    else:
                        # Different shapes - need padding
                        max_n_templ = max(b[key].shape[0] for b in batch)
                        
                        if 'distogram' in key or 'unit_vector' in key:
                            # [N_templ, N_token, N_token, ...]
                            extra_dims = example_tensor.shape[3:]
                            shape = (batch_size, max_n_templ, max_n_tokens, max_n_tokens) + extra_dims
                            padded = torch.zeros(shape, dtype=example_tensor.dtype)
                            for i, b in enumerate(batch):
                                n_templ = b[key].shape[0]
                                n_token = b['token_index'].shape[0]
                                if n_templ > 0:
                                    padded[i, :n_templ, :n_token, :n_token] = b[key][:, :n_token, :n_token]
                        elif example_tensor.ndim == 3:
                            # [N_templ, N_token, features]
                            extra_dims = example_tensor.shape[2:]
                            shape = (batch_size, max_n_templ, max_n_tokens) + extra_dims
                            padded = torch.zeros(shape, dtype=example_tensor.dtype)
                            for i, b in enumerate(batch):
                                n_templ = b[key].shape[0]
                                n_token = b['token_index'].shape[0]
                                if n_templ > 0:
                                    padded[i, :n_templ, :n_token] = b[key][:, :n_token]
                        elif example_tensor.ndim == 2:
                            # [N_templ, N_token]
                            shape = (batch_size, max_n_templ, max_n_tokens)
                            padded = torch.zeros(shape, dtype=example_tensor.dtype)
                            for i, b in enumerate(batch):
                                n_templ = b[key].shape[0]
                                n_token = b['token_index'].shape[0]
                                if n_templ > 0:
                                    padded[i, :n_templ, :n_token] = b[key][:, :n_token]
                        else:
                            # Fallback
                            padded = torch.stack([b[key] for b in batch])
                    
            else:
                # Default: try to stack (no padding)
                try:
                    padded = torch.stack([b[key] for b in batch])
                except RuntimeError:
                    # Fallback: pad to max_n_tokens
                    shape = (batch_size, max_n_tokens) + example_tensor.shape[1:]
                    padded = torch.zeros(shape, dtype=example_tensor.dtype)
                    for i, b in enumerate(batch):
                        n = b[key].shape[0]
                        padded[i, :n] = b[key]
            
            collated[key] = padded
        
        return collated


def create_dummy_structure(
    n_residues: int = 10,
    n_chains: int = 1,
    n_msa: int = 50
) -> Dict:
    """
    Create dummy structure for testing.
    
    Args:
        n_residues: Number of residues per chain
        n_chains: Number of chains
        n_msa: Number of MSA sequences
        
    Returns:
        Structure dict compatible with AlphaFold3Dataset
    """
    from src.data.featurizer import AMINO_ACIDS
    
    chains = []
    atom_counter = 0
    
    for chain_idx in range(n_chains):
        # Generate sequence using 3-letter codes (standard amino acids)
        sequence = ''.join(np.random.choice(AMINO_ACIDS, size=n_residues))
        
        residues = []
        for res_idx in range(n_residues):
            # Use 3-letter code so it's recognized as STANDARD amino acid
            restype = AMINO_ACIDS[res_idx % len(AMINO_ACIDS)]
            
            # Create dummy atoms (CA, N, C, O) - backbone atoms
            atoms = []
            for atom_name, element, atomic_num in [
                ('N', 'N', 7),
                ('CA', 'C', 6),
                ('C', 'C', 6),
                ('O', 'O', 8)
            ]:
                atoms.append({
                    'atom_index': atom_counter,
                    'atom_name': atom_name,
                    'element': element,
                    'atomic_number': atomic_num,
                    'position': np.random.randn(3).tolist(),
                    'charge': 0.0
                })
                atom_counter += 1
            
            residues.append({
                'restype': restype,  # 3-letter code (ALA, GLY, etc.)
                'residue_index': res_idx + 1,
                'atoms': atoms
            })
        
        # Create dummy MSA (use 3-letter codes)
        msa_sequences = []
        for _ in range(n_msa):
            msa_seq = ''.join(np.random.choice(AMINO_ACIDS, size=n_residues))
            deletions = [0] * n_residues
            msa_sequences.append(MSASequence(msa_seq, deletions))
        
        # Build sequence string for the chain (3-letter codes joined)
        chain_sequence = ''.join([residues[i]['restype'] for i in range(n_residues)])
        
        chains.append({
            'chain_id': chr(65 + chain_idx),  # A, B, C, ...
            'mol_type': MoleculeType.PROTEIN,
            'sequence': chain_sequence,
            'residues': residues,
            'msa': msa_sequences,
            'asym_id': chain_idx
        })
    
    return {'chains': chains}