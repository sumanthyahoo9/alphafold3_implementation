#!/usr/bin/env python3
"""
AlphaFold3 Inference Script

Usage:
    python inference.py --input protein.fasta --output output_dir/

Features:
    - Load model checkpoint
    - Process input sequences
    - Run full AlphaFold3 pipeline
    - Generate 3D structures
    - Output PDB with confidence scores
    - Visualize predictions
"""
from pathlib import Path
import json
from typing import Dict, Optional
import time
import argparse
import torch
import numpy as np

from src.models.alphafold3 import AlphaFold3
from src.data.tokenizer import AlphaFold3Tokenizer, MoleculeType
from src.data.featurizer import AlphaFold3Featurizer, Atom
from src.models.heads.confidence_head import ConfidenceHead


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run AlphaFold3 inference on protein sequences'
    )
    
    # Input/Output
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input FASTA file or sequence string'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory for results'
    )
    
    # Model
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (optional - will use random init if not provided)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (cuda/cpu)'
    )
    
    # Sampling
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1,
        help='Number of structures to sample'
    )
    parser.add_argument(
        '--n_diffusion_steps',
        type=int,
        default=200,
        help='Number of diffusion steps (50-200, higher=better)'
    )
    parser.add_argument(
        '--n_recycles',
        type=int,
        default=4,
        help='Number of recycling iterations (1-4)'
    )
    
    # MSA (optional - for now use None)
    parser.add_argument(
        '--msa',
        type=str,
        default=None,
        help='Path to MSA file (optional)'
    )
    
    # Output options
    parser.add_argument(
        '--save_pdb',
        action='store_true',
        default=True,
        help='Save structure as PDB file'
    )
    parser.add_argument(
        '--save_confidence',
        action='store_true',
        default=True,
        help='Save confidence metrics as JSON'
    )
    
    return parser.parse_args()


def load_sequence(input_path: str) -> str:
    """
    Load sequence from FASTA file or use as direct input.
    
    Args:
        input_path: Path to FASTA or raw sequence string
    
    Returns:
        sequence: Amino acid sequence
    """
    # Check if it's a file
    if Path(input_path).exists():
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        # Parse FASTA
        sequence = ''
        for line in lines:
            line = line.strip()
            if not line.startswith('>'):
                sequence += line
        
        return sequence
    else:
        # Treat as direct sequence
        return input_path


def create_features_from_sequence(sequence: str, device: str) -> Dict[str, torch.Tensor]:
    """
    Create proper features from sequence using tokenizer and featurizer.
    
    Args:
        sequence: Amino acid sequence
        device: Device to create tensors on
    
    Returns:
        features: Dictionary of input features
    """
    # Initialize tokenizer and featurizer
    tokenizer = AlphaFold3Tokenizer()
    featurizer = AlphaFold3Featurizer()
    
    # Create mock residues (simple protein chain)
    residues = []
    atoms = []
    atom_counter = 0
    
    # Map 1-letter to 3-letter amino acid codes
    aa_map = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }
    
    for i, aa_letter in enumerate(sequence):
        # Convert to 3-letter code
        restype = aa_map.get(aa_letter.upper(), 'UNK')
        
        # Create minimal atoms for this residue (N, CA, C, O)
        residue_atoms = []
        for atom_name in ['N', 'CA', 'C', 'O']:
            atom = {
                'atom_index': atom_counter,
                'atom_name': atom_name,
                'element': 'C' if atom_name in ['CA', 'C'] else ('N' if atom_name == 'N' else 'O')
            }
            residue_atoms.append(atom)
            
            # Create Atom object for featurizer
            atomic_number = {'C': 6, 'N': 7, 'O': 8}[atom['element']]
            atoms.append(Atom(
                atom_index=atom_counter,
                atom_name=atom_name,
                element=atom['element'],
                atomic_number=atomic_number,
                position=np.random.randn(3) * 5.0 + i * 3.8,  # Rough CA spacing
                charge=0.0,
                mask=1.0
            ))
            
            atom_counter += 1
        
        # Create residue
        residues.append({
            'restype': restype,
            'residue_index': i,
            'atoms': residue_atoms
        })
    
    # Tokenize the chain
    chain_tokens = tokenizer.tokenize_chain(
        chain_id='A',
        asym_id=0,
        entity_id=0,
        sym_id=0,
        residues=residues,
        mol_type=MoleculeType.PROTEIN
    )
    
    # Get token features
    token_features = tokenizer.get_token_features()
    atom_to_token_map = tokenizer.get_atom_to_token_mapping()
    
    # Get restypes
    restypes = [token.restype for token in chain_tokens]
    
    # Featurize
    features_np = featurizer.featurize(
        token_features=token_features,
        atoms=atoms,
        restypes=restypes,
        atom_to_token_map=atom_to_token_map
    )
    
    # Add MSA features (placeholder)
    n_tokens = len(chain_tokens)
    msa_features = featurizer.create_placeholder_msa_features(n_tokens)
    features_np.update(msa_features)
    
    # Convert to torch tensors
    features = {}
    for key, value in features_np.items():
        if isinstance(value, np.ndarray):
            features[key] = torch.from_numpy(value).to(device)
        else:
            features[key] = value
    
    # Add atom_to_token mapping as tensor
    features['atom_to_token'] = torch.from_numpy(atom_to_token_map).to(device)
    
    return features


def run_inference(
    model: AlphaFold3,
    features: Dict[str, torch.Tensor],
    n_samples: int = 1,
    device: str = 'cuda'
) -> Dict:
    """
    Run AlphaFold3 inference.
    
    Args:
        model: AlphaFold3 model
        features: Input features
        n_samples: Number of structures to sample
        device: Device to run on
    
    Returns:
        results: Dictionary with predictions
    """
    model.eval()
    
    results = {
        'structures': [],
        'confidence': []
    }
    
    print(f"\nRunning inference with {n_samples} sample(s)...")
    
    with torch.no_grad():
        for i in range(n_samples):
            print(f"  Sample {i+1}/{n_samples}...")
            start = time.time()
            
            # Run model
            predictions = model(features)
            
            elapsed = time.time() - start
            print(f"    Time: {elapsed:.1f}s")
            
            # Extract results
            results['structures'].append(predictions['x_pred'].cpu())
            
            # Confidence metrics
            confidence = {
                'plddt': predictions['p_plddt'].cpu(),
                'pae': predictions['p_pae'].cpu(),
                'pde': predictions['p_pde'].cpu(),
                'resolved': predictions['p_resolved'].cpu(),
            }
            results['confidence'].append(confidence)
    
    return results


def save_pdb(
    coords: torch.Tensor,
    sequence: str,
    output_path: str,
    plddt_scores: Optional[torch.Tensor] = None
):
    """
    Save structure as PDB file.
    
    Args:
        coords: Atom coordinates [N_atoms, 3]
        sequence: Amino acid sequence
        output_path: Output PDB path
        plddt_scores: Per-atom confidence scores [N_atoms]
    """
    coords = coords.numpy()
    
    if plddt_scores is not None:
        plddt_scores = plddt_scores.numpy()
    else:
        plddt_scores = np.ones(len(coords)) * 100
    
    with open(output_path, 'w') as f:
        # Header
        f.write("REMARK   Generated by AlphaFold3\n")
        f.write(f"REMARK   Sequence: {sequence}\n")
        
        # Atoms
        for i, (coord, plddt) in enumerate(zip(coords, plddt_scores)):
            # Simple CA-only model
            aa = sequence[i] if i < len(sequence) else 'X'
            
            # PDB format
            line = (
                f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00{plddt:6.2f}           C  \n"
            )
            f.write(line)
        
        f.write("END\n")
    
    print(f"  Saved PDB: {output_path}")


def save_confidence_metrics(
    confidence: Dict,
    output_path: str
):
    """
    Save confidence metrics as JSON.
    
    Args:
        confidence: Confidence metrics dictionary
        output_path: Output JSON path
    """
    # Convert tensors to lists
    confidence_json = {}
    
    for key, value in confidence.items():
        if isinstance(value, torch.Tensor):
            confidence_json[key] = value.tolist()
    
    with open(output_path, 'w') as f:
        json.dump(confidence_json, f, indent=2)
    
    print(f"  Saved confidence: {output_path}")


def main():
    """Main inference pipeline."""
    args = parse_args()
    
    print("="*70)
    print("AlphaFold3 Inference")
    print("="*70)
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDevice: {device}")
    print(f"Output directory: {output_dir}")
    
    # Load sequence
    print(f"\nLoading sequence from: {args.input}")
    sequence = load_sequence(args.input)
    print(f"  Length: {len(sequence)} residues")
    print(f"  Sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
    
    # Create features
    print("\nCreating input features...")
    features = create_features_from_sequence(sequence, device)
    print(f"  Number of tokens: {len(features['restype'])}")
    
    # Initialize model
    print("\nInitializing AlphaFold3 model...")
    model = AlphaFold3(
        c_token=384,
        c_pair=128,
        n_cycles=args.n_recycles
    ).to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Checkpoint loaded!")
    else:
        print("  Using random initialization (no checkpoint provided)")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    
    # Run inference
    results = run_inference(
        model=model,
        features=features,
        n_samples=args.n_samples,
        device=device
    )
    
    # Save results
    print("\nSaving results...")
    
    for i, (structure, confidence) in enumerate(
        zip(results['structures'], results['confidence'])
    ):
        sample_dir = output_dir / f"sample_{i+1}"
        sample_dir.mkdir(exist_ok=True)
        
        # Get pLDDT scores
        head = ConfidenceHead()
        plddt_scores = head.get_plddt_scores(confidence['plddt'])
        
        # Save PDB
        if args.save_pdb:
            pdb_path = sample_dir / "structure.pdb"
            save_pdb(structure, sequence, str(pdb_path), plddt_scores)
        
        # Save confidence
        if args.save_confidence:
            conf_path = sample_dir / "confidence.json"
            save_confidence_metrics(confidence, str(conf_path))
        
        # Summary statistics
        mean_plddt = plddt_scores.mean().item()
        print(f"\n  Sample {i+1} Statistics:")
        print(f"    Mean pLDDT: {mean_plddt:.2f}")
        print(f"    Min pLDDT:  {plddt_scores.min().item():.2f}")
        print(f"    Max pLDDT:  {plddt_scores.max().item():.2f}")
    
    print("\n" + "="*70)
    print("Inference complete!")
    print("="*70)
    
    # Print summary
    print(f"\nResults saved to: {output_dir}")
    print(f"Number of samples: {args.n_samples}")
    
    if args.n_samples > 1:
        print("\nTo view results:")
        print(f"  PyMOL: pymol {output_dir}/sample_*/structure.pdb")
        print(f"  ChimeraX: chimerax {output_dir}/sample_*/structure.pdb")


if __name__ == '__main__':
    main()