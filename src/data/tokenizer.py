"""
AlphaFold3 Tokenizer

Implements tokenization scheme from AF3 supplementary materials section 2.6:
- Standard amino acid residues → 1 token
- Standard nucleotide residues → 1 token  
- Modified amino acids/nucleotides → N tokens (per-atom)
- All ligands → N tokens (per-atom)

Token centre atoms:
- Cα for standard amino acids
- C1' for standard nucleotides
- First atom for per-atom tokens
"""

from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
import numpy as np


class MoleculeType(Enum):
    """Molecule type classification"""
    PROTEIN = "protein"
    RNA = "rna"
    DNA = "dna"
    LIGAND = "ligand"


# Standard 20 amino acids
STANDARD_AMINO_ACIDS = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

# Standard DNA nucleotides
STANDARD_DNA = {'DA', 'DC', 'DG', 'DT'}

# Standard RNA nucleotides  
STANDARD_RNA = {'A', 'C', 'G', 'U'}


@dataclass
class Token:
    """
    Represents a single token in AF3.
    
    Attributes:
        token_index: Global monotonic token number
        residue_index: Residue number in original chain
        asym_id: Unique chain identifier
        entity_id: Unique sequence identifier
        sym_id: Symmetry ID within chains of same sequence
        mol_type: Type of molecule (protein/rna/dna/ligand)
        restype: Residue type (e.g., 'ALA', 'G', ligand name)
        is_standard: Whether this is a standard residue
        atom_indices: List of atom indices belonging to this token
        centre_atom_index: Index of the token centre atom
    """
    token_index: int
    residue_index: int
    asym_id: int
    entity_id: int
    sym_id: int
    mol_type: MoleculeType
    restype: str
    is_standard: bool
    atom_indices: List[int]
    centre_atom_index: int


class AlphaFold3Tokenizer:
    """
    Tokenizes biomolecular structures according to AlphaFold3 scheme.
    
    Key logic:
    - Standard residues/nucleotides → single token (residue-level)
    - Modified residues & ligands → per-atom tokens
    - Maintains atom→token mapping for downstream processing
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.atom_to_token_map: Dict[int, int] = {}
        self.token_counter = 0
        
    def tokenize_chain(
        self, 
        chain_id: str,
        asym_id: int,
        entity_id: int,
        sym_id: int,
        residues: List[Dict],
        mol_type: MoleculeType
    ) -> List[Token]:
        """
        Tokenize a single chain.
        
        Args:
            chain_id: Chain identifier
            asym_id: Unique chain ID
            entity_id: Unique sequence ID
            sym_id: Symmetry ID for chains with same sequence
            residues: List of residue dicts with keys:
                - 'restype': residue name (e.g., 'ALA', 'G')
                - 'residue_index': residue number in chain
                - 'atoms': list of atom dicts with keys:
                    - 'atom_index': global atom index
                    - 'atom_name': atom name (e.g., 'CA', 'N')
                    - 'element': element symbol
            mol_type: Molecule type
            
        Returns:
            List of Token objects
        """
        chain_tokens = []
        
        for residue in residues:
            restype = residue['restype']
            residue_index = residue['residue_index']
            atoms = residue['atoms']
            
            # Determine if standard residue
            is_standard = self._is_standard_residue(restype, mol_type)
            
            if is_standard:
                # Standard residue → 1 token
                atom_indices = [atom['atom_index'] for atom in atoms]
                centre_atom_idx = self._find_centre_atom(atoms, restype, mol_type)
                
                token = Token(
                    token_index=self.token_counter,
                    residue_index=residue_index,
                    asym_id=asym_id,
                    entity_id=entity_id,
                    sym_id=sym_id,
                    mol_type=mol_type,
                    restype=restype,
                    is_standard=True,
                    atom_indices=atom_indices,
                    centre_atom_index=centre_atom_idx
                )
                
                # Map all atoms to this token
                for atom_idx in atom_indices:
                    self.atom_to_token_map[atom_idx] = self.token_counter
                    
                chain_tokens.append(token)
                self.token_counter += 1
                
            else:
                # Modified residue or ligand → per-atom tokens
                for atom in atoms:
                    atom_idx = atom['atom_index']
                    
                    token = Token(
                        token_index=self.token_counter,
                        residue_index=residue_index,
                        asym_id=asym_id,
                        entity_id=entity_id,
                        sym_id=sym_id,
                        mol_type=mol_type if mol_type != MoleculeType.LIGAND else MoleculeType.LIGAND,
                        restype=restype,
                        is_standard=False,
                        atom_indices=[atom_idx],
                        centre_atom_index=atom_idx  # Single atom is its own centre
                    )
                    
                    self.atom_to_token_map[atom_idx] = self.token_counter
                    chain_tokens.append(token)
                    self.token_counter += 1
        
        self.tokens.extend(chain_tokens)
        return chain_tokens
    
    def _is_standard_residue(self, restype: str, mol_type: MoleculeType) -> bool:
        """Check if residue is standard (not modified)"""
        if mol_type == MoleculeType.PROTEIN:
            return restype in STANDARD_AMINO_ACIDS
        elif mol_type == MoleculeType.RNA:
            return restype in STANDARD_RNA
        elif mol_type == MoleculeType.DNA:
            return restype in STANDARD_DNA
        else:  # LIGAND
            return False
    
    def _find_centre_atom(
        self, 
        atoms: List[Dict], 
        restype: str, 
        mol_type: MoleculeType
    ) -> int:
        """
        Find token centre atom index.
        
        Centre atoms (from AF3 paper):
        - Cα for standard amino acids
        - C1' for standard nucleotides
        - First atom for per-atom tokens
        """
        if mol_type == MoleculeType.PROTEIN:
            # Find CA (alpha carbon)
            for atom in atoms:
                if atom['atom_name'] == 'CA':
                    return atom['atom_index']
        
        elif mol_type in (MoleculeType.RNA, MoleculeType.DNA):
            # Find C1' atom
            for atom in atoms:
                if atom['atom_name'] == "C1'":
                    return atom['atom_index']
        
        # Fallback: return first atom
        return atoms[0]['atom_index']
    
    def get_token_features(self) -> Dict[str, np.ndarray]:
        """
        Extract token-level features for model input.
        
        Returns dict with keys matching Table 5 in AF3 supplementary:
            - token_index: [N_token]
            - residue_index: [N_token]
            - asym_id: [N_token]
            - entity_id: [N_token]
            - sym_id: [N_token]
            - is_protein/rna/dna/ligand: [N_token] masks
        """
        n_tokens = len(self.tokens)
        
        features = {
            'token_index': np.array([t.token_index for t in self.tokens], dtype=np.int32),
            'residue_index': np.array([t.residue_index for t in self.tokens], dtype=np.int32),
            'asym_id': np.array([t.asym_id for t in self.tokens], dtype=np.int32),
            'entity_id': np.array([t.entity_id for t in self.tokens], dtype=np.int32),
            'sym_id': np.array([t.sym_id for t in self.tokens], dtype=np.int32),
            
            # Molecule type masks
            'is_protein': np.array([t.mol_type == MoleculeType.PROTEIN for t in self.tokens], dtype=np.float32),
            'is_rna': np.array([t.mol_type == MoleculeType.RNA for t in self.tokens], dtype=np.float32),
            'is_dna': np.array([t.mol_type == MoleculeType.DNA for t in self.tokens], dtype=np.float32),
            'is_ligand': np.array([t.mol_type == MoleculeType.LIGAND for t in self.tokens], dtype=np.float32),
        }
        
        return features
    
    def get_atom_to_token_mapping(self) -> np.ndarray:
        """
        Get mapping from flat atom indices to token indices.
        
        Returns:
            Array of shape [N_atoms] where value at index i is the token ID
            that atom i belongs to
        """
        max_atom_idx = max(self.atom_to_token_map.keys())
        mapping = np.full(max_atom_idx + 1, -1, dtype=np.int32)
        
        for atom_idx, token_idx in self.atom_to_token_map.items():
            mapping[atom_idx] = token_idx
            
        return mapping
    
    def reset(self):
        """Reset tokenizer state"""
        self.tokens = []
        self.atom_to_token_map = {}
        self.token_counter = 0