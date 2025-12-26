"""
AlphaFold3 Diffusion Sampling

File: src/models/diffusion/sample_diffusion.py

Implements Algorithm 18: Sample Diffusion
Implements Algorithm 19: Centre Random Augmentation

The reverse diffusion sampling loop that generates structures from noise.

Key features:
- Iterative denoising over ~200 timesteps
- Random augmentation at each step (rotation + translation)
- Euler integration for denoising trajectory
- Noise schedule from paper

Architecture:
    Random noise → [200 denoising steps] → Final structure
    
Each step:
    x → CentreRandomAugmentation → add noise → DiffusionModule → Euler update → x'
"""
from typing import Dict, List, Optional
import math
import torch
import torch.nn as nn

from src.models.diffusion.diffusion_module import DiffusionModule


def create_noise_schedule(
    n_steps: int = 200,
    s_max: float = 160.0,
    s_min: float = 4e-4,
    p: float = 7.0,
    sigma_data: float = 16.0
) -> torch.Tensor:
    """
    Create noise schedule for diffusion sampling.
    
    From supplementary page 24:
    t_hat = sigma_data * (s_max^(1/p) + t * (s_min^(1/p) - s_max^(1/p)))^p
    where t is uniformly distributed in [0, 1] with step size 1/n_steps
    
    Args:
        n_steps: Number of diffusion steps (default: 200)
        s_max: Maximum noise level (default: 160.0)
        s_min: Minimum noise level (default: 4e-4)
        p: Schedule power (default: 7.0)
        sigma_data: Data variance (default: 16.0)
    
    Returns:
        schedule: [n_steps+1] noise levels from c_0 (max) to c_T (min)
    """
    # t uniformly from 0 to 1
    t = torch.linspace(0, 1, n_steps + 1)
    
    # Compute schedule
    schedule = sigma_data * (
        s_max ** (1/p) + t * (s_min ** (1/p) - s_max ** (1/p))
    ) ** p
    
    return schedule


class CentreRandomAugmentation(nn.Module):
    """
    Centre and randomly augment coordinates.
    
    Implements Algorithm 19 from AF3 supplementary.
    
    Centers coordinates, applies random rotation, adds random translation.
    This data augmentation helps the model be invariant to global position/orientation.
    
    Args:
        s_trans: Translation noise scale in Angstroms (default: 1.0)
    """
    
    def __init__(self, s_trans: float = 1.0):
        super().__init__()
        self.s_trans = s_trans
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply centre and random augmentation.
        
        Args:
            x: Atom positions
               Shape: [N_atoms, 3]
        
        Returns:
            x: Augmented positions
               Shape: [N_atoms, 3]
        
        Algorithm 19:
        1: x = x - mean(x)                    # Centre
        2: R = UniformRandomRotation()        # Random rotation matrix
        3: t ~ s_trans * N(0, I_3)           # Random translation
        4: x = R @ x + t                      # Apply rotation + translation
        5: return x
        """
        # Line 1: Centre coordinates
        x_centered = x - x.mean(dim=0, keepdim=True)  # [N_atoms, 3]
        
        # Line 2: Generate random rotation matrix
        R = self._random_rotation_matrix(x.device)  # [3, 3]
        
        # Line 3: Generate random translation
        t = self.s_trans * torch.randn(3, device=x.device)  # [3]
        
        # Line 4: Apply rotation and translation
        # x_aug = (R @ x^T)^T + t = x @ R^T + t
        x_aug = x_centered @ R.T + t  # [N_atoms, 3]
        
        return x_aug
    
    def _random_rotation_matrix(self, device: str = 'cpu') -> torch.Tensor:
        """
        Generate uniform random 3D rotation matrix.
        
        Uses quaternion method for uniform sampling on SO(3).
        
        Args:
            device: Device for tensor
        
        Returns:
            R: [3, 3] rotation matrix
        """
        # Sample uniform quaternion (Shoemake method)
        u = torch.rand(3, device=device)
        
        q0 = torch.sqrt(1 - u[0]) * torch.sin(2 * math.pi * u[1])
        q1 = torch.sqrt(1 - u[0]) * torch.cos(2 * math.pi * u[1])
        q2 = torch.sqrt(u[0]) * torch.sin(2 * math.pi * u[2])
        q3 = torch.sqrt(u[0]) * torch.cos(2 * math.pi * u[2])
        
        # Convert quaternion to rotation matrix
        R = torch.tensor([
            [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
        ], device=device)
        
        return R


class SampleDiffusion(nn.Module):
    """
    Sample structures via reverse diffusion.
    
    Implements Algorithm 18 from AF3 supplementary.
    
    Iteratively denoises random noise to generate molecular structures.
    Uses DiffusionModule as the denoiser at each timestep.
    
    Args:
        diffusion_module: DiffusionModule for denoising
        noise_schedule: List of noise levels [c_0, c_1, ..., c_T]
        gamma_0: Noise injection scale (default: 0.8)
        gamma_min: Noise injection threshold (default: 1.0)
        noise_scale: Noise scale factor λ (default: 1.003)
        step_scale: Step scale factor η (default: 1.5)
        s_trans: Translation scale for augmentation (default: 1.0)
    """
    
    def __init__(
        self,
        diffusion_module: DiffusionModule,
        noise_schedule: Optional[List[float]] = None,
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        s_trans: float = 1.0
    ):
        super().__init__()
        
        self.diffusion_module = diffusion_module
        
        # Create default noise schedule if not provided
        if noise_schedule is None:
            noise_schedule = create_noise_schedule()
        
        # Convert to tensor if it's a list, otherwise clone if already tensor
        if isinstance(noise_schedule, list):
            self.register_buffer('noise_schedule', torch.tensor(noise_schedule))
        else:
            self.register_buffer('noise_schedule', noise_schedule.clone())
        
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale  # λ
        self.step_scale = step_scale    # η
        
        # Algorithm 19: Centre and random augmentation
        self.augmentation = CentreRandomAugmentation(s_trans=s_trans)
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        n_atoms: int
    ) -> torch.Tensor:
        """
        Sample structure via reverse diffusion.
        
        Args:
            features: Dictionary with tokenization features
            s_inputs: Input single representation [N_token, 384]
            s_trunk: Trunk single representation [N_token, 384]
            z_trunk: Trunk pair representation [N_token, N_token, 128]
            n_atoms: Number of atoms to generate
        
        Returns:
            x: Final atom positions [N_atoms, 3]
        
        Algorithm 18:
        1: x ~ c_0 * N(0, I_3)                          # Initialize from noise
        2: for c_τ in [c_1, ..., c_T]:
        3:     x = CentreRandomAugmentation(x)
        4:     γ = γ_0 if c_τ > γ_min else 0
        5:     t_hat = c_{τ-1} * (γ + 1)
        6:     ξ = λ * sqrt(t_hat² - c_{τ-1}²) * N(0, I_3)
        7:     x_noisy = x + ξ
        8:     x_denoised = DiffusionModule(x_noisy, t_hat, ...)
        9:     δ = (x - x_denoised) / t_hat
        10:    dt = c_τ - t_hat
        11:    x = x_noisy + η * dt * δ                # Euler step
        13: return x
        """
        device = s_inputs.device
        
        # Algorithm 18, line 1: Initialize from noise
        c_0 = self.noise_schedule[0]
        x = c_0 * torch.randn(n_atoms, 3, device=device)  # [N_atoms, 3]
        
        # Algorithm 18, line 2: Iterate through noise schedule
        for tau in range(1, len(self.noise_schedule)):
            c_tau = self.noise_schedule[tau]
            c_prev = self.noise_schedule[tau - 1]
            
            # Line 3: Apply random augmentation
            x = self.augmentation(x)
            
            # Line 4: Compute noise injection scale
            gamma = self.gamma_0 if c_tau > self.gamma_min else 0.0
            
            # Line 5: Compute adjusted timestep
            t_hat = c_prev * (gamma + 1)
            
            # Line 6: Generate additional noise
            # ξ = λ * sqrt(t_hat² - c_prev²) * N(0, I_3)
            noise_std = self.noise_scale * torch.sqrt(
                torch.clamp(t_hat ** 2 - c_prev ** 2, min=0.0)
            )
            xi = noise_std * torch.randn(n_atoms, 3, device=device)
            
            # Line 7: Add noise
            x_noisy = x + xi
            
            # Line 8: Denoise with DiffusionModule
            x_denoised = self.diffusion_module(
                x_noisy=x_noisy,
                t_hat=t_hat,
                features=features,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk
            )
            
            # Line 9: Compute error estimate
            delta = (x - x_denoised) / t_hat
            
            # Line 10: Compute timestep increment
            dt = c_tau - t_hat
            
            # Line 11: Euler integration step
            x = x_noisy + self.step_scale * dt * delta
        
        # Line 13: Return final structure
        return x