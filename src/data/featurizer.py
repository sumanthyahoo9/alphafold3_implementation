"""
AlphaFold3 Featurizer

Extracts model input features according to AF3 Table 5:
- Token features: indices, chain IDs, molecule type masks
- Reference features: atom positions, elements, charges (from conformers)
- Restype encoding: one-hot encoding (32 classes)

Follows section 2.8 of AF3 supplementary materials.
"""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass


# Restype encoding: 32 classes
# 20 amino acids + unknown, 4 RNA + unknown, 4 DNA + unknown, gap
AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
RNA_BASES = ['A', 'C', 'G', 'U']
DNA_BASES = ['DA', 'DC', 'DG', 'DT']

# Build restype vocabulary (32 classes)
RESTYPE_VOCAB = (
    AMINO_ACIDS + ['UNK_PROTEIN'] +  # 21 protein classes
    RNA_BASES + ['UNK_RNA'] +        # 5 RNA classes  
    DNA_BASES + ['UNK_DNA'] +        # 5 DNA classes
    ['GAP']                           # 1 gap class
)  # Total: 32

RESTYPE_TO_INDEX = {res: i for i, res in enumerate(RESTYPE_VOCAB)}


@dataclass
class Atom:
    """Atom representation with all required features"""
    atom_index: int
    atom_name: str
    element: str
    atomic_number: int
    position: np.ndarray  # [3] coordinates in Angstroms
    charge: float
    mask: float  # 1.0 if atom exists, 0.0 otherwise


class AlphaFold3Featurizer:
    """
    Converts tokenized molecular structure into model input features.
    
    Generates features from AF3 Table 5:
    - Token-level features (indices, chain IDs, restypes)
    - Atom-level reference features (positions, elements, charges)
    """
    
    def __init__(self, max_atoms_per_token: int = 64):
        """
        Args:
            max_atoms_per_token: Maximum atoms per token for padding
        """
        self.max_atoms_per_token = max_atoms_per_token
        
    def featurize(
        self,
        token_features: Dict[str, np.ndarray],
        atoms: List[Atom],
        restypes: List[str],
        atom_to_token_map: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete feature dictionary for model input.
        
        Args:
            token_features: Output from tokenizer.get_token_features()
            atoms: List of Atom objects
            restypes: List of residue types (length = n_tokens)
            atom_to_token_map: Array mapping atom indices to token indices
            
        Returns:
            Dictionary with all features from AF3 Table 5
        """
        n_tokens = len(token_features['token_index'])
        n_atoms = len(atoms)
        
        features = {}
        
        # === Token features (from tokenizer) ===
        features['token_index'] = token_features['token_index']
        features['residue_index'] = token_features['residue_index']
        features['asym_id'] = token_features['asym_id']
        features['entity_id'] = token_features['entity_id']
        features['sym_id'] = token_features['sym_id']
        features['is_protein'] = token_features['is_protein']
        features['is_rna'] = token_features['is_rna']
        features['is_dna'] = token_features['is_dna']
        features['is_ligand'] = token_features['is_ligand']
        
        # === Restype one-hot encoding [N_token, 32] ===
        features['restype'] = self._encode_restypes(restypes)
        
        # === Reference conformer features (atom-level) ===
        ref_features = self._extract_reference_features(atoms)
        features['ref_pos'] = ref_features['positions']        # [N_atom, 3]
        features['ref_mask'] = ref_features['mask']            # [N_atom]
        features['ref_element'] = ref_features['element']      # [N_atom, 128]
        features['ref_charge'] = ref_features['charge']        # [N_atom]
        features['ref_atom_name_chars'] = ref_features['atom_name_chars']  # [N_atom, 4, 64]
        
        # === Reference space UID (for grouping atoms by residue) ===
        features['ref_space_uid'] = self._compute_ref_space_uid(
            atom_to_token_map, 
            token_features['asym_id'],
            token_features['residue_index']
        )
        
        return features
    
    def _encode_restypes(self, restypes: List[str]) -> np.ndarray:
        """
        One-hot encode residue types.
        
        Args:
            restypes: List of residue type strings (e.g., ['ALA', 'GLY', 'LIG'])
            
        Returns:
            One-hot array of shape [N_token, 32]
        """
        n_tokens = len(restypes)
        encoding = np.zeros((n_tokens, 32), dtype=np.float32)
        
        for i, restype in enumerate(restypes):
            if restype in RESTYPE_TO_INDEX:
                encoding[i, RESTYPE_TO_INDEX[restype]] = 1.0
            else:
                # Unknown types map to appropriate unknown class
                encoding[i, RESTYPE_TO_INDEX['UNK_PROTEIN']] = 1.0
                
        return encoding
    
    def _extract_reference_features(self, atoms: List[Atom]) -> Dict[str, np.ndarray]:
        """
        Extract atom-level reference conformer features.
        
        Returns dict with:
            - positions: [N_atom, 3] 
            - mask: [N_atom]
            - element: [N_atom, 128] one-hot atomic number
            - charge: [N_atom]
            - atom_name_chars: [N_atom, 4, 64] one-hot atom names
        """
        n_atoms = len(atoms)
        
        positions = np.zeros((n_atoms, 3), dtype=np.float32)
        mask = np.zeros(n_atoms, dtype=np.float32)
        element = np.zeros((n_atoms, 128), dtype=np.float32)
        charge = np.zeros(n_atoms, dtype=np.float32)
        atom_name_chars = np.zeros((n_atoms, 4, 64), dtype=np.float32)
        
        for i, atom in enumerate(atoms):
            positions[i] = atom.position
            mask[i] = atom.mask
            charge[i] = atom.charge
            
            # One-hot encode atomic number (up to 128)
            if 0 < atom.atomic_number < 128:
                element[i, atom.atomic_number] = 1.0
            
            # Encode atom name as characters (up to 4 chars)
            atom_name_chars[i] = self._encode_atom_name(atom.atom_name)
        
        return {
            'positions': positions,
            'mask': mask,
            'element': element,
            'charge': charge,
            'atom_name_chars': atom_name_chars
        }
    
    def _encode_atom_name(self, atom_name: str) -> np.ndarray:
        """
        Encode atom name as one-hot character encoding.
        
        Each character encoded as ord(c) - 32, padded to length 4.
        Returns array of shape [4, 64].
        """
        encoding = np.zeros((4, 64), dtype=np.float32)
        
        # Pad/truncate to length 4
        name = atom_name[:4].ljust(4)
        
        for i, char in enumerate(name):
            # Encode as ord(c) - 32 (ASCII printable chars start at 32)
            char_idx = ord(char) - 32
            if 0 <= char_idx < 64:
                encoding[i, char_idx] = 1.0
                
        return encoding
    
    def _compute_ref_space_uid(
        self, 
        atom_to_token_map: np.ndarray,
        asym_ids: np.ndarray,
        residue_indices: np.ndarray
    ) -> np.ndarray:
        """
        Compute reference space UID for each atom.
        
        Groups atoms by (chain_id, residue_index) tuple.
        Each unique tuple gets a unique integer ID.
        
        Args:
            atom_to_token_map: [N_atom] mapping atom -> token
            asym_ids: [N_token] chain IDs
            residue_indices: [N_token] residue indices
            
        Returns:
            Array of shape [N_atom] with space UIDs
        """
        n_atoms = len(atom_to_token_map)
        ref_space_uid = np.zeros(n_atoms, dtype=np.int32)
        
        # Build (asym_id, residue_index) -> uid mapping
        uid_map = {}
        uid_counter = 0
        
        for atom_idx in range(n_atoms):
            token_idx = atom_to_token_map[atom_idx]
            if token_idx < 0:  # Invalid mapping
                continue
                
            asym_id = asym_ids[token_idx]
            res_idx = residue_indices[token_idx]
            key = (asym_id, res_idx)
            
            if key not in uid_map:
                uid_map[key] = uid_counter
                uid_counter += 1
            
            ref_space_uid[atom_idx] = uid_map[key]
        
        return ref_space_uid
    
    def create_placeholder_msa_features(self, n_tokens: int) -> Dict[str, np.ndarray]:
        """
        Create placeholder MSA features (for testing without MSA generation).
        
        Args:
            n_tokens: Number of tokens
            
        Returns:
            Dict with placeholder MSA features
        """
        # Minimal MSA: just the query sequence
        n_msa = 1
        
        return {
            'msa': np.zeros((n_msa, n_tokens, 32), dtype=np.float32),
            'has_deletion': np.zeros((n_msa, n_tokens), dtype=np.float32),
            'deletion_value': np.zeros((n_msa, n_tokens), dtype=np.float32),
            'profile': np.zeros((n_tokens, 32), dtype=np.float32),
            'deletion_mean': np.zeros(n_tokens, dtype=np.float32)
        }
    
    def create_placeholder_template_features(self, n_tokens: int) -> Dict[str, np.ndarray]:
        """
        Create placeholder template features (for testing without templates).
        
        Args:
            n_tokens: Number of tokens
            
        Returns:
            Dict with placeholder template features
        """
        n_templates = 0  # No templates
        
        return {
            'template_restype': np.zeros((n_templates, n_tokens, 32), dtype=np.float32),
            'template_pseudo_beta_mask': np.zeros((n_templates, n_tokens), dtype=np.float32),
            'template_backbone_frame_mask': np.zeros((n_templates, n_tokens), dtype=np.float32),
            'template_distogram': np.zeros((n_templates, n_tokens, n_tokens, 39), dtype=np.float32),
            'template_unit_vector': np.zeros((n_templates, n_tokens, n_tokens, 3), dtype=np.float32)
        }
    
    def create_placeholder_bond_features(self, n_tokens: int) -> Dict[str, np.ndarray]:
        """
        Create placeholder bond features.
        
        Args:
            n_tokens: Number of tokens
            
        Returns:
            Dict with placeholder bond features
        """
        return {
            'token_bonds': np.zeros((n_tokens, n_tokens), dtype=np.float32)
        }