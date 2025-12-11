"""
AlphaFold3 MSA Pipeline

Processes Multiple Sequence Alignments according to AF3 section 2.3:
- Constructs MSA with up to 16,384 rows
- First row is query sequence
- Next rows (up to 8,191) are paired by species  
- Remaining rows filled densely from original MSA
- Computes profile and deletion features

Simplified implementation for research purposes (no actual sequence search).
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class MSASequence:
    """Single MSA sequence entry"""
    sequence: str
    deletion_matrix: List[int]  # Deletions at each position
    species_id: Optional[str] = None


class AlphaFold3MSAPipeline:
    """
    Processes MSAs for AF3 model input.
    
    Key operations:
    1. Subsample MSA to max_msa_rows (16,384)
    2. Pair sequences by species for multi-chain complexes
    3. Compute profile (restype distribution)
    4. Compute deletion features
    """
    
    def __init__(self, max_msa_rows: int = 16384, max_paired_rows: int = 8191):
        """
        Args:
            max_msa_rows: Maximum total MSA rows (default 16,384)
            max_paired_rows: Maximum paired rows (default 8,191)
        """
        self.max_msa_rows = max_msa_rows
        self.max_paired_rows = max_paired_rows
    
    def process_msa(
        self,
        query_sequence: str,
        msa_sequences: List[MSASequence],
        n_tokens: int
    ) -> Dict[str, np.ndarray]:
        """
        Process raw MSA into model features.
        
        Args:
            query_sequence: Query sequence string
            msa_sequences: List of homologous sequences
            n_tokens: Number of tokens (for feature dimensions)
            
        Returns:
            Dict with MSA features:
                - msa: [N_msa, N_token, 32] one-hot encoded sequences
                - has_deletion: [N_msa, N_token] binary deletion indicator
                - deletion_value: [N_msa, N_token] transformed deletion counts
                - profile: [N_token, 32] residue type distribution
                - deletion_mean: [N_token] mean deletions per position
        """
        # Build MSA matrix
        all_sequences = [query_sequence] + [seq.sequence for seq in msa_sequences]
        all_deletions = [[0] * len(query_sequence)] + [seq.deletion_matrix for seq in msa_sequences]
        
        # Subsample to max_msa_rows
        if len(all_sequences) > self.max_msa_rows:
            # Keep query + random subset
            indices = [0] + list(np.random.choice(
                range(1, len(all_sequences)), 
                size=self.max_msa_rows - 1, 
                replace=False
            ))
            all_sequences = [all_sequences[i] for i in indices]
            all_deletions = [all_deletions[i] for i in indices]
        
        n_msa = len(all_sequences)
        
        # Encode MSA sequences
        msa_encoded = self._encode_msa_sequences(all_sequences, n_tokens)
        
        # Process deletions
        has_deletion, deletion_value = self._process_deletions(all_deletions, n_tokens)
        
        # Compute profile (before any subsampling in training)
        profile = self._compute_profile(msa_encoded)
        
        # Compute mean deletions
        deletion_mean = self._compute_deletion_mean(all_deletions, n_tokens)
        
        return {
            'msa': msa_encoded,
            'has_deletion': has_deletion,
            'deletion_value': deletion_value,
            'profile': profile,
            'deletion_mean': deletion_mean
        }
    
    def _encode_msa_sequences(
        self, 
        sequences: List[str], 
        n_tokens: int
    ) -> np.ndarray:
        """
        Encode MSA sequences as one-hot vectors.
        
        Uses same 32-class encoding as restype:
        20 amino acids + unknown, 4 RNA + unknown, 4 DNA + unknown, gap
        
        Args:
            sequences: List of sequence strings
            n_tokens: Number of tokens
            
        Returns:
            Array of shape [N_msa, N_token, 32]
        """
        from src.data.featurizer import RESTYPE_TO_INDEX
        
        n_msa = len(sequences)
        msa_encoded = np.zeros((n_msa, n_tokens, 32), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            for j, residue in enumerate(seq[:n_tokens]):
                if residue == '-':
                    # Gap character
                    msa_encoded[i, j, RESTYPE_TO_INDEX['GAP']] = 1.0
                elif residue in RESTYPE_TO_INDEX:
                    msa_encoded[i, j, RESTYPE_TO_INDEX[residue]] = 1.0
                else:
                    # Unknown -> map to UNK_PROTEIN
                    msa_encoded[i, j, RESTYPE_TO_INDEX['UNK_PROTEIN']] = 1.0
        
        return msa_encoded
    
    def _process_deletions(
        self, 
        deletion_matrices: List[List[int]], 
        n_tokens: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process deletion information.
        
        Args:
            deletion_matrices: List of deletion counts per position per sequence
            n_tokens: Number of tokens
            
        Returns:
            Tuple of (has_deletion, deletion_value):
                - has_deletion: [N_msa, N_token] binary indicator
                - deletion_value: [N_msa, N_token] transformed counts
        """
        n_msa = len(deletion_matrices)
        has_deletion = np.zeros((n_msa, n_tokens), dtype=np.float32)
        deletion_value = np.zeros((n_msa, n_tokens), dtype=np.float32)
        
        for i, deletions in enumerate(deletion_matrices):
            for j, count in enumerate(deletions[:n_tokens]):
                if count > 0:
                    has_deletion[i, j] = 1.0
                
                # Transform: (2/π) * arctan(d/3)
                deletion_value[i, j] = (2.0 / np.pi) * np.arctan(count / 3.0)
        
        return has_deletion, deletion_value
    
    def _compute_profile(self, msa_encoded: np.ndarray) -> np.ndarray:
        """
        Compute residue type distribution across MSA.
        
        Profile is the mean one-hot encoding across all MSA rows.
        
        Args:
            msa_encoded: [N_msa, N_token, 32] one-hot MSA
            
        Returns:
            Array of shape [N_token, 32] with type distributions
        """
        # Average across MSA rows
        profile = msa_encoded.mean(axis=0)
        return profile
    
    def _compute_deletion_mean(
        self, 
        deletion_matrices: List[List[int]], 
        n_tokens: int
    ) -> np.ndarray:
        """
        Compute mean number of deletions per position.
        
        Args:
            deletion_matrices: List of deletion counts
            n_tokens: Number of tokens
            
        Returns:
            Array of shape [N_token] with mean deletions
        """
        # Stack and average
        deletion_array = np.zeros((len(deletion_matrices), n_tokens), dtype=np.float32)
        
        for i, deletions in enumerate(deletion_matrices):
            deletion_array[i, :min(len(deletions), n_tokens)] = deletions[:n_tokens]
        
        deletion_mean = deletion_array.mean(axis=0)
        return deletion_mean
    
    def subsample_msa_randomly(
        self, 
        msa_features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Randomly subsample MSA during training (per recycling iteration).
        
        From AF3 section 2.2: "During training, the main MSA for each sequence 
        is subsampled from size n to size k = Uniform[1, n]"
        
        Args:
            msa_features: Dict with 'msa', 'has_deletion', 'deletion_value'
            
        Returns:
            Subsampled MSA features with same keys
        """
        n_msa = msa_features['msa'].shape[0]
        
        # Sample new size k ~ Uniform[1, n]
        k = np.random.randint(1, n_msa + 1)
        
        # Always keep query (row 0), sample remaining
        if k == 1:
            indices = [0]
        else:
            indices = [0] + list(np.random.choice(
                range(1, n_msa), 
                size=min(k - 1, n_msa - 1),
                replace=False
            ))
        
        return {
            'msa': msa_features['msa'][indices],
            'has_deletion': msa_features['has_deletion'][indices],
            'deletion_value': msa_features['deletion_value'][indices],
            'profile': msa_features['profile'],  # Not subsampled
            'deletion_mean': msa_features['deletion_mean']  # Not subsampled
        }
    
    def create_dummy_msa(
        self, 
        query_sequence: str, 
        n_tokens: int, 
        n_homologs: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Create dummy MSA for testing (random sequences).
        
        Args:
            query_sequence: Query sequence
            n_tokens: Number of tokens
            n_homologs: Number of synthetic homologs to generate
            
        Returns:
            MSA features dict
        """
        from src.data.featurizer import AMINO_ACIDS
        
        # Generate random homologous sequences
        msa_sequences = []
        for _ in range(n_homologs):
            # Random mutations of query
            seq = ''.join(np.random.choice(AMINO_ACIDS, size=len(query_sequence)))
            deletions = [0] * len(query_sequence)
            msa_sequences.append(MSASequence(seq, deletions))
        
        return self.process_msa(query_sequence, msa_sequences, n_tokens)
    
    def pair_msa_by_species(
        self,
        msa_sequences_per_chain: Dict[str, List[MSASequence]],
        n_tokens_per_chain: Dict[str, int]
    ) -> Dict[str, np.ndarray]:
        """
        Pair MSA sequences across chains by species (for complexes).
        
        Simplified implementation: groups by species_id and stacks.
        Full implementation would follow AF-Multimer pairing algorithm.
        
        Args:
            msa_sequences_per_chain: Dict mapping chain_id -> MSA sequences
            n_tokens_per_chain: Dict mapping chain_id -> num tokens
            
        Returns:
            Paired MSA features with concatenated sequences
        """
        # For now, just concatenate unpaired MSAs
        # Full pairing algorithm is complex and requires taxonomy data
        
        chain_ids = list(msa_sequences_per_chain.keys())
        total_tokens = sum(n_tokens_per_chain.values())
        
        # Process each chain separately
        chain_features = {}
        for chain_id in chain_ids:
            query = msa_sequences_per_chain[chain_id][0].sequence
            others = msa_sequences_per_chain[chain_id][1:]
            n_tokens = n_tokens_per_chain[chain_id]
            
            chain_features[chain_id] = self.process_msa(query, others, n_tokens)
        
        # Concatenate across chains (simplified)
        # Real implementation would pair by species first
        n_msa = min(feat['msa'].shape[0] for feat in chain_features.values())
        
        msa_concat = np.concatenate([
            chain_features[cid]['msa'][:n_msa] 
            for cid in chain_ids
        ], axis=1)
        
        has_del_concat = np.concatenate([
            chain_features[cid]['has_deletion'][:n_msa]
            for cid in chain_ids
        ], axis=1)
        
        del_val_concat = np.concatenate([
            chain_features[cid]['deletion_value'][:n_msa]
            for cid in chain_ids
        ], axis=1)
        
        profile_concat = np.concatenate([
            chain_features[cid]['profile']
            for cid in chain_ids
        ], axis=0)
        
        del_mean_concat = np.concatenate([
            chain_features[cid]['deletion_mean']
            for cid in chain_ids
        ], axis=0)
        
        return {
            'msa': msa_concat,
            'has_deletion': has_del_concat,
            'deletion_value': del_val_concat,
            'profile': profile_concat,
            'deletion_mean': del_mean_concat
        }