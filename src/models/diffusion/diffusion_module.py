"""
AlphaFold3 Diffusion Module

File: src/models/diffusion/diffusion_module.py

Implements Algorithm 20: Diffusion Module

The main denoiser network that removes noise from atomic coordinates.
Integrates all diffusion components:
- DiffusionConditioning (prepares trunk features)
- AtomAttentionEncoder (atoms → tokens)
- DiffusionTransformer (token-level processing)
- AtomAttentionDecoder (tokens → atoms)

Architecture:
    Noisy coords → Scale → Encoder → Transformer → Decoder → Rescale → Denoised coords
    
Key features:
- Two-level architecture (atoms → tokens → atoms)
- No geometric biases (just linear projections)
- Conditioning from trunk via single and pair representations
- Variance-preserving scaling
"""
from typing import Dict
import torch
import torch.nn as nn

from src.models.diffusion.diffusion_conditioning import DiffusionConditioning
from src.models.diffusion.diffusion_transformer import DiffusionTransformer
from src.models.embeddings.atom_attention import AtomAttentionEncoder, AtomAttentionDecoder


class DiffusionModule(nn.Module):
    """
    Diffusion Module - main denoiser for coordinate generation.
    
    Implements Algorithm 20 from AF3 supplementary.
    
    Takes noisy atomic coordinates and produces denoised coordinates
    through a two-level architecture (atoms → tokens → atoms).
    
    Args:
        sigma_data: Data variance constant (default: 16.0)
        c_atom: Atom channel dimension (default: 128)
        c_atompair: Atom pair channel dimension (default: 16)
        c_token: Token channel dimension (default: 768)
        c_pair: Pair channel dimension for conditioning (default: 128)
        c_s: Single conditioning dimension (default: 384)
        n_blocks: Number of transformer blocks (default: 24)
        n_heads: Number of attention heads (default: 16)
    """
    
    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_pair: int = 128,
        c_s: int = 384,
        n_blocks: int = 24,
        n_heads: int = 16
    ):
        super().__init__()
        
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_pair = c_pair
        self.c_s = c_s
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        
        # Algorithm 20, line 1: DiffusionConditioning
        # Note: DiffusionConditioning takes trunk outputs (c_token=384)
        # and outputs c_single=384, c_pair=128
        self.conditioning = DiffusionConditioning(
            c_token=384,  # Trunk single dimension (s_inputs, s_trunk)
            c_pair_trunk=c_pair,  # Trunk pair dimension (z_trunk)
            c_single=c_s,  # Output single conditioning (384)
            c_pair=c_pair,  # Output pair conditioning (128)
            n_transition=2
        )
        
        # Algorithm 20, line 3: AtomAttentionEncoder
        # Note: Encoder outputs c_token=768 as per Algorithm 20
        self.atom_encoder = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token  # This should be 768
        )
        
        # Algorithm 20, line 4: Conditioning projection
        self.conditioning_projection = nn.Linear(c_s, c_token, bias=False)
        self.conditioning_norm = nn.LayerNorm(c_s)
        
        # Algorithm 20, line 5: DiffusionTransformer
        self.transformer = DiffusionTransformer(
            c_token=c_token,
            c_pair=c_pair,
            c_s=c_s,
            n_blocks=n_blocks,
            n_heads=n_heads
        )
        
        # Algorithm 20, line 6: Output normalization
        self.output_norm = nn.LayerNorm(c_token)
        
        # Algorithm 20, line 7: AtomAttentionDecoder
        self.atom_decoder = AtomAttentionDecoder(
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair
        )
    
    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat: torch.Tensor,
        features: Dict[str, torch.Tensor],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor
    ) -> torch.Tensor:
        """
        Denoise noisy atomic coordinates.
        
        Args:
            x_noisy: Noisy atom positions
                    Shape: [N_atoms, 3] or [batch, N_atoms, 3]
            t_hat: Diffusion timestep (scaled)
                  Shape: [] (scalar) or [batch]
            features: Dictionary with tokenization features
                     (residue_index, asym_id, entity_id, sym_id, token_index, etc.)
            s_inputs: Input single representation
                     Shape: [N_token, 384] or [batch, N_token, 384]
            s_trunk: Trunk single representation
                    Shape: [N_token, 384] or [batch, N_token, 384]
            z_trunk: Trunk pair representation
                    Shape: [N_token, N_token, 128] or [batch, N_token, N_token, 128]
        
        Returns:
            x_out: Denoised atom positions
                  Shape: same as x_noisy
        
        Algorithm 20:
        1: s, z = DiffusionConditioning(t, features, s_inputs, s_trunk, z_trunk, σ)
        2: r_noisy = x_noisy / sqrt(t² + σ²)
        3: a, q_skip, c_skip, p_skip = AtomAttentionEncoder(features, r_noisy, s_trunk, z)
        4: a += Linear(LayerNorm(s))
        5: a = DiffusionTransformer(a, s, z, β=0)
        6: a = LayerNorm(a)
        7: r_update = AtomAttentionDecoder(a, q_skip, c_skip, p_skip)
        8: x_out = combine(x_noisy, r_update, t, σ)
        9: return x_out
        """
        # Check if batched
        is_batched = x_noisy.dim() == 3
        
        if is_batched:
            # Process each batch item separately
            batch_size = x_noisy.shape[0]
            x_outs = []
            
            for i in range(batch_size):
                x_out_i = self._forward_single(
                    x_noisy[i],
                    t_hat[i] if t_hat.dim() > 0 else t_hat,
                    features,
                    s_inputs[i],
                    s_trunk[i],
                    z_trunk[i]
                )
                x_outs.append(x_out_i)
            
            return torch.stack(x_outs, dim=0)
        else:
            return self._forward_single(x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk)
    
    def _forward_single(
        self,
        x_noisy: torch.Tensor,
        t_hat: torch.Tensor,
        features: Dict[str, torch.Tensor],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for a single (unbatched) example.
        
        All inputs are unbatched:
        - x_noisy: [N_atoms, 3]
        - t_hat: scalar
        - s_inputs, s_trunk: [N_token, 384]
        - z_trunk: [N_token, N_token, 128]
        """
        # Algorithm 20, line 1: Compute conditioning
        # Add batch dim for conditioning (which expects batched input)
        s_cond, z_cond = self.conditioning(
            s_inputs=s_inputs.unsqueeze(0),  # [1, N_token, 384]
            s_trunk=s_trunk.unsqueeze(0),    # [1, N_token, 384]
            z_trunk=z_trunk.unsqueeze(0),    # [1, N_token, N_token, 128]
            t=t_hat.unsqueeze(0) if t_hat.dim() == 0 else t_hat.unsqueeze(0),
            features=features,
            sigma_data=self.sigma_data
        )
        
        # Remove batch dim for encoder (which expects unbatched input)
        s_cond = s_cond.squeeze(0)  # [N_token, 384]
        z_cond = z_cond.squeeze(0)  # [N_token, N_token, 128]
        
        # Algorithm 20, line 2: Scale positions to unit variance
        # r_noisy = x_noisy / sqrt(t_hat² + sigma_data²)
        if t_hat.dim() == 0:  # Scalar timestep
            scale_factor = torch.sqrt(t_hat ** 2 + self.sigma_data ** 2)
        else:  # Batched timesteps
            scale_factor = torch.sqrt(t_hat ** 2 + self.sigma_data ** 2)
            # Reshape for broadcasting: [batch] -> [batch, 1, 1]
            scale_factor = scale_factor.view(-1, 1, 1)
        
        r_noisy = x_noisy / scale_factor  # [..., N_atoms, 3]
        
        # Algorithm 20, line 3: Encode atoms to tokens  
        a, q_skip, c_skip, p_skip = self.atom_encoder(
            atom_features=features,
            atom_to_token_idx=features['atom_to_token'],
            noisy_positions=r_noisy,
            trunk_single=s_trunk,
            trunk_pair=z_cond
        )
        
        # Algorithm 20, line 4: Add conditioning to token activations
        # a += Linear(LayerNorm(s))
        s_proj = self.conditioning_projection(self.conditioning_norm(s_cond))
        a = a + s_proj
        
        # Algorithm 20, line 5: Apply transformer
        # β_ij = 0 (no additional bias mask)
        a = self.transformer(a, s_cond, z_cond, bias_mask=None)
        
        # Algorithm 20, line 6: Output normalization
        a = self.output_norm(a)
        
        # Algorithm 20, line 7: Decode tokens to atom position updates
        r_update = self.atom_decoder(
            ai=a,
            q_skip=q_skip,
            c_skip=c_skip,
            p_skip=p_skip,
            atom_to_token_idx=features['atom_to_token']
        )
        
        # Algorithm 20, line 8: Rescale and combine with input
        # x_out = (σ²/(σ² + t²)) · x_noisy + (σ·t/sqrt(σ² + t²)) · r_update
        sigma_sq = self.sigma_data ** 2
        t_sq = t_hat ** 2
        
        if t_hat.dim() == 0:  # Scalar
            coef_noisy = sigma_sq / (sigma_sq + t_sq)
            coef_update = (self.sigma_data * t_hat) / torch.sqrt(sigma_sq + t_sq)
        else:  # Batched
            coef_noisy = sigma_sq / (sigma_sq + t_sq)
            coef_update = (self.sigma_data * t_hat) / torch.sqrt(sigma_sq + t_sq)
            # Reshape for broadcasting
            coef_noisy = coef_noisy.view(-1, 1, 1)
            coef_update = coef_update.view(-1, 1, 1)
        
        x_out = coef_noisy * x_noisy + coef_update * r_update
        
        # Algorithm 20, line 9: Return
        return x_out


def create_dummy_diffusion_module_input(
    batch_size: int = None,
    n_atoms: int = 100,
    n_tokens: int = 10,
    c_token: int = 768,  # Diffusion internal dimension
    c_pair: int = 128,
    device: str = 'cpu'
) -> tuple:
    """
    Create dummy inputs for testing DiffusionModule.
    
    Args:
        batch_size: If None, unbatched. Otherwise, batched.
        n_atoms: Number of atoms
        n_tokens: Number of tokens
        c_token: Token dimension FOR DIFFUSION (768)
        c_pair: Pair dimension
        device: Device for tensors
    
    Returns:
        (x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk): Tuple of inputs
    """
    # Note: s_inputs and s_trunk are TRUNK outputs (384), not diffusion dimension (768)
    c_trunk_token = 384  # Trunk outputs 384-dim single
    
    if batch_size is None:
        x_noisy = torch.randn(n_atoms, 3, device=device)
        t_hat = torch.tensor(5.0, device=device)
        s_inputs = torch.randn(n_tokens, c_trunk_token, device=device)  # 384, not 768
        s_trunk = torch.randn(n_tokens, c_trunk_token, device=device)   # 384, not 768
        z_trunk = torch.randn(n_tokens, n_tokens, c_pair, device=device)
    else:
        x_noisy = torch.randn(batch_size, n_atoms, 3, device=device)
        t_hat = torch.rand(batch_size, device=device) * 10
        s_inputs = torch.randn(batch_size, n_tokens, c_trunk_token, device=device)  # 384
        s_trunk = torch.randn(batch_size, n_tokens, c_trunk_token, device=device)   # 384
        z_trunk = torch.randn(batch_size, n_tokens, n_tokens, c_pair, device=device)
    
    # Create dummy features
    features = {
        'residue_index': torch.arange(n_tokens, device=device),
        'token_index': torch.arange(n_tokens, device=device),
        'asym_id': torch.zeros(n_tokens, dtype=torch.long, device=device),
        'entity_id': torch.zeros(n_tokens, dtype=torch.long, device=device),
        'sym_id': torch.zeros(n_tokens, dtype=torch.long, device=device),
        # Additional features needed by AtomAttentionEncoder
        'atom_to_token': torch.arange(n_atoms, device=device) % n_tokens,  # Map atoms to tokens
        'ref_pos': torch.randn(n_atoms, 3, device=device),  # Reference positions
        'ref_mask': torch.ones(n_atoms, device=device),  # Atom mask
        'ref_element': torch.randn(n_atoms, 128, device=device),  # Elements (one-hot)
        'ref_charge': torch.zeros(n_atoms, device=device),  # Charges
        'ref_atom_name_chars': torch.randn(n_atoms, 4, 64, device=device),  # Atom names
        'ref_space_uid': torch.arange(n_atoms, device=device) % n_tokens,  # Residue grouping
    }
    
    return x_noisy, t_hat, features, s_inputs, s_trunk, z_trunk