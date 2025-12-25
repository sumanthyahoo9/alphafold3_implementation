"""
AlphaFold3 Diffusion Conditioning

File: src/models/diffusion/diffusion_conditioning.py

Implements Algorithm 21: Diffusion Conditioning

Prepares conditioning signals from trunk outputs for the diffusion module.
Combines trunk representations (single + pair) with timestep information
to create time-aware conditioning for the denoiser.

Key features:
- Pair conditioning: trunk pair + relative position encoding + transitions
- Single conditioning: trunk single + input single + timestep + transitions
- Separate processing paths for single vs pair
- Timestep encoded via FourierEmbedding

Mathematical intuition:
    The diffusion module needs to know:
    1. What structure we're denoising (from trunk)
    2. How noisy it is (from timestep)
    3. Geometric relationships (from relative positions)
    
    DiffusionConditioning combines all these signals.

Architecture:
    Pair path: concat → project → 2 transitions
    Single path: concat → project → add timestep → 2 transitions
    
Why separate processing:
- Pair captures geometric relationships (no time dependence)
- Single processes sequence (time-dependent via timestep embedding)
"""

import torch
import torch.nn as nn

from src.models.trunk.transition import Transition
from src.models.embeddings.relative_encoding import RelativePositionEncoding
from src.models.diffusion.fourier_embedding import FourierEmbedding


class DiffusionConditioning(nn.Module):
    """
    Diffusion Conditioning - prepares trunk outputs for diffusion.
    
    Implements Algorithm 21 from AF3 supplementary.
    
    Takes trunk representations and timestep, produces conditioned
    representations for the diffusion transformer. The pair conditioning
    includes geometric information, while the single conditioning includes
    timestep information.
    
    Args:
        c_token: Token dimension from trunk single (default: 384)
        c_pair_trunk: Pair dimension from trunk (default: 128)
        c_single: Output single dimension (default: 384)
        c_pair: Output pair dimension (default: 128)
        n_transition: Number of transition blocks (default: 2)
    """
    
    def __init__(
        self,
        c_token: int = 384,
        c_pair_trunk: int = 128,
        c_single: int = 384,
        c_pair: int = 128,
        n_transition: int = 2
    ):
        super().__init__()
        
        self.c_token = c_token
        self.c_pair_trunk = c_pair_trunk
        self.c_single = c_single
        self.c_pair = c_pair
        self.n_transition = n_transition
        
        # Algorithm 21, line 1: Relative position encoding
        self.relative_position_encoding = RelativePositionEncoding(
            r_max=32,
            s_max=2,
            c_z=c_pair_trunk  # Same as trunk pair dimension initially
        )
        
        # Algorithm 21, line 2: Pair projection
        # Input: trunk_pair (c_pair_trunk) + relative_pos (c_pair_trunk)
        # After concat: 2 * c_pair_trunk
        self.pair_projection = nn.Linear(2 * c_pair_trunk, c_pair, bias=False)
        self.pair_norm = nn.LayerNorm(2 * c_pair_trunk)
        
        # Algorithm 21, lines 3-5: Pair transitions
        self.pair_transitions = nn.ModuleList([
            Transition(c=c_pair, n=2)
            for _ in range(n_transition)
        ])
        
        # Algorithm 21, line 6: Single concatenation
        # Input: trunk_single (c_token) + inputs_single (c_token)
        # After concat: 2 * c_token
        
        # Algorithm 21, line 7: Single projection
        self.single_projection = nn.Linear(2 * c_token, c_single, bias=False)
        self.single_norm = nn.LayerNorm(2 * c_token)
        
        # Algorithm 21, line 8: Fourier embedding for timestep
        self.fourier_embedding = FourierEmbedding(dim=256)
        
        # Algorithm 21, line 9: Timestep projection
        self.timestep_projection = nn.Linear(256, c_single, bias=False)
        self.timestep_norm = nn.LayerNorm(256)
        
        # Algorithm 21, lines 10-12: Single transitions
        self.single_transitions = nn.ModuleList([
            Transition(c=c_single, n=2)
            for _ in range(n_transition)
        ])
    
    def forward(
        self,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        t: torch.Tensor,
        features: dict,
        sigma_data: float = 16.0
    ) -> tuple:
        """
        Prepare conditioning from trunk outputs and timestep.
        
        Args:
            s_inputs: Input single representation
                     Shape: [N_token, c_token] or [batch, N_token, c_token]
            s_trunk: Trunk single representation
                    Shape: [N_token, c_token] or [batch, N_token, c_token]
            z_trunk: Trunk pair representation
                    Shape: [N_token, N_token, c_pair_trunk] or [batch, N_token, N_token, c_pair_trunk]
            t: Diffusion timestep (t_hat)
               Shape: [] (scalar) or [batch]
            features: Dictionary with tokenization features
                     (residue_index, asym_id, entity_id, sym_id, token_index)
            sigma_data: Data variance constant (default: 16.0)
        
        Returns:
            (s_cond, z_cond): Conditioned single and pair representations
                             s_cond: [N_token, c_single] or [batch, N_token, c_single]
                             z_cond: [N_token, N_token, c_pair] or [batch, N_token, N_token, c_pair]
        
        Algorithm 21:
        # Pair conditioning
        1: z = concat([z_trunk, RelativePositionEncoding(features)])
        2: z ← LinearNoBias(LayerNorm(z))
        3-5: for b in [1,2]: z += Transition(z, n=2)
        
        # Single conditioning  
        6: s = concat([s_trunk, s_inputs])
        7: s ← LinearNoBias(LayerNorm(s))
        8: n = FourierEmbedding(0.25 * log(t/sigma_data), 256)
        9: s += LinearNoBias(LayerNorm(n))
        10-12: for b in [1,2]: s += Transition(s, n=2)
        
        13: return s, z
        """
        # Handle batching
        is_batched = s_trunk.dim() == 3
        if not is_batched:
            s_inputs = s_inputs.unsqueeze(0)
            s_trunk = s_trunk.unsqueeze(0)
            z_trunk = z_trunk.unsqueeze(0)
            if t.dim() == 0:
                t = t.unsqueeze(0)
        
        batch_size, n_token = s_trunk.shape[:2]
        
        # ============ PAIR CONDITIONING ============
        # Algorithm 21, line 1: Concatenate trunk pair + relative position encoding
        rel_pos = self.relative_position_encoding(features)  # [N_token, N_token, c_pair_trunk]
        
        # Expand rel_pos to match batch if needed
        if is_batched:
            # rel_pos is [N_token, N_token, c_pair_trunk], expand to [batch, N_token, N_token, c_pair_trunk]
            rel_pos = rel_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)
        else:
            # Unbatched: both are [1, N_token, N_token, c] after unsqueeze, expand rel_pos
            rel_pos = rel_pos.unsqueeze(0)  # [1, N_token, N_token, c_pair_trunk]
        
        # Concatenate
        z = torch.cat([z_trunk, rel_pos], dim=-1)  # [..., N_token, N_token, 2*c_pair_trunk]
        
        # Algorithm 21, line 2: Project to output dimension
        z = self.pair_projection(self.pair_norm(z))  # [..., N_token, N_token, c_pair]
        
        # Algorithm 21, lines 3-5: Apply transition blocks
        for transition in self.pair_transitions:
            z = z + transition(z)  # [..., N_token, N_token, c_pair]
        
        # ============ SINGLE CONDITIONING ============
        # Algorithm 21, line 6: Concatenate trunk single + input single
        s = torch.cat([s_trunk, s_inputs], dim=-1)  # [..., N_token, 2*c_token]
        
        # Algorithm 21, line 7: Project to output dimension
        s = self.single_projection(self.single_norm(s))  # [..., N_token, c_single]
        
        # Algorithm 21, line 8: Fourier embedding of timestep
        # Preprocess: 0.25 * log(t / sigma_data)
        t_preprocessed = 0.25 * torch.log(t / sigma_data)  # [batch] or []
        
        # Embed timestep
        n = self.fourier_embedding(t_preprocessed)  # [batch, 256] or [256]
        
        # Algorithm 21, line 9: Add timestep embedding to single
        # Project timestep to single dimension
        n_proj = self.timestep_projection(self.timestep_norm(n))  # [batch, c_single] or [c_single]
        
        # Broadcast to all tokens
        if n_proj.dim() == 1:
            # Unbatched: [c_single] → [1, 1, c_single] → [1, N_token, c_single]
            n_proj = n_proj.unsqueeze(0).unsqueeze(0).expand(1, n_token, -1)
        else:
            # Batched: [batch, c_single] → [batch, 1, c_single] → [batch, N_token, c_single]
            n_proj = n_proj.unsqueeze(1).expand(-1, n_token, -1)
        
        s = s + n_proj  # [..., N_token, c_single]
        
        # Algorithm 21, lines 10-12: Apply transition blocks
        for transition in self.single_transitions:
            s = s + transition(s)  # [..., N_token, c_single]
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            s = s.squeeze(0)
            z = z.squeeze(0)
        
        # Algorithm 21, line 13: Return conditioned representations
        return s, z


def create_dummy_diffusion_conditioning_input(
    batch_size: int = 0, # Change to None, to clear unit tests
    n_token: int = 10,
    c_token: int = 384,
    c_pair: int = 128,
    device: str = 'cpu'
) -> tuple:
    """
    Create dummy inputs for testing DiffusionConditioning.
    
    Args:
        batch_size: If None, returns unbatched. Otherwise, batched.
        n_token: Number of tokens
        c_token: Token dimension
        c_pair: Pair dimension
        device: Device for tensors
    
    Returns:
        (s_inputs, s_trunk, z_trunk, t, features): Tuple of inputs
    """
    if batch_size is None:
        s_inputs = torch.randn(n_token, c_token, device=device)
        s_trunk = torch.randn(n_token, c_token, device=device)
        z_trunk = torch.randn(n_token, n_token, c_pair, device=device)
        t = torch.tensor(5.0, device=device)  # Scalar timestep
    else:
        s_inputs = torch.randn(batch_size, n_token, c_token, device=device)
        s_trunk = torch.randn(batch_size, n_token, c_token, device=device)
        z_trunk = torch.randn(batch_size, n_token, n_token, c_pair, device=device)
        t = torch.rand(batch_size, device=device) * 10  # Batch of timesteps
    
    # Create dummy features for RelativePositionEncoding
    features = {
        'residue_index': torch.arange(n_token, device=device),
        'token_index': torch.arange(n_token, device=device),
        'asym_id': torch.zeros(n_token, dtype=torch.long, device=device),
        'entity_id': torch.zeros(n_token, dtype=torch.long, device=device),
        'sym_id': torch.zeros(n_token, dtype=torch.long, device=device),
    }
    
    return s_inputs, s_trunk, z_trunk, t, features