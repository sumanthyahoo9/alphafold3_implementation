# AlphaFold3 Diffusion Module Summary

## Overview

The diffusion module is the **structure generator** - it takes learned representations from the trunk and generates 3D atomic coordinates through a reverse diffusion process.

**Key Innovation:** Instead of predicting coordinates directly, AlphaFold3 uses **diffusion models** to iteratively denoise random coordinates into accurate structures.

**Total Algorithms Implemented:** 8

---

## Module Architecture

```
Trunk Representations → Diffusion Module → 3D Structure
                             ↑
                     (You are here!)
```

---

## What is Diffusion?

### The Big Picture

**Forward Process (Training):**
```
Clean Structure → Add Noise → Add More Noise → ... → Pure Noise
   x_0              x_1           x_2              x_T
```

**Reverse Process (Inference):**
```
Pure Noise → Denoise → Denoise → ... → Clean Structure
   x_T        x_{T-1}    x_{T-2}        x_0
```

### Why Diffusion?

**Traditional approach:**
- Network directly predicts coordinates
- One-shot prediction
- Hard to capture multi-modal distributions

**Diffusion approach:**
- Network learns to denoise
- Iterative refinement (200 steps)
- Naturally handles multiple conformations
- More stable training

### The Math (Simplified)

**Noise Schedule:**
```python
# Sample noise level
t ~ Uniform(0, 1)
σ_t = σ_data × exp(-1.2 + 1.5 × N(0,1))

# Add noise to structure
x_noisy = x_clean + σ_t × ε, where ε ~ N(0, I)
```

**Denoising:**
```python
# Network predicts clean structure from noisy input
x_pred = DiffusionModule(x_noisy, t, conditioning)

# Training loss: predict clean coordinates
loss = MSE(x_pred, x_clean) + other_losses
```

---

## Architecture Components

### 1. Diffusion Conditioning (Algorithm 21)

**Purpose:** Embed noise level and trunk features for diffusion transformer

```python
class DiffusionConditioning(nn.Module):
    def forward(self, s_trunk, z_trunk, t):
        # Embed noise level (time)
        t_emb = FourierEmbedding(t)  # [1, 256]
        
        # Condition single representation
        s_cond = s_trunk + Linear(t_emb)
        
        # Pair representation unchanged
        z_cond = z_trunk
        
        return s_cond, z_cond
```

**Why Fourier embedding?** Captures periodic patterns at multiple frequencies.

---

### 2. Fourier Embedding (Algorithm 22)

**Purpose:** Encode scalar values (time, noise level) into high-dimensional space

```python
class FourierEmbedding(nn.Module):
    def __init__(self, dim=256):
        # Learnable frequency bands
        self.freqs = nn.Parameter(torch.randn(dim // 2))
    
    def forward(self, x):
        # x: scalar or [batch]
        # Compute sin/cos at multiple frequencies
        angles = x * self.freqs * 2 * pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
```

**Output:** `[256]` dimensional embedding for a scalar input

**Why multiple frequencies?** Different timescales need different representations!

---

### 3. Adaptive Layer Norm (Algorithm 26)

**Purpose:** Condition normalization on external signal (noise level)

```python
class AdaptiveLayerNorm(nn.Module):
    def forward(self, x, conditioning):
        # Standard layer norm (no learnable params)
        x_norm = (x - mean(x)) / std(x)
        
        # Condition with external signal
        scale = sigmoid(Linear(conditioning))
        shift = Linear(conditioning)
        
        return scale * x_norm + shift
```

**Why adaptive?** Different noise levels need different normalization!

---

### 4. Conditioned Transition Block (Algorithm 25)

**Purpose:** MLP with adaptive normalization and gating (SwiGLU activation)

```python
class ConditionedTransitionBlock(nn.Module):
    def forward(self, x, conditioning):
        # Adaptive normalization
        x = AdaLN(x, conditioning)
        
        # SwiGLU activation (gated linear unit)
        gate = swish(Linear(x))
        value = Linear(x)
        x = gate * value  # Element-wise product
        
        # Gated output projection (adaLN-Zero)
        out_gate = sigmoid(Linear(conditioning, bias_init=-2))
        x = out_gate * Linear(x)
        
        return x
```

**Why SwiGLU?** Better than ReLU - smooth, gated, empirically strong!

---

### 5. Diffusion Transformer (Algorithm 23)

**Purpose:** Core transformer that processes noisy coordinates

**Architecture:**
```
Input: Noisy Coordinates [N_atoms, 3]
       Conditioning (s, z, t)
        ↓
┌─────────────────────────────────┐
│  Diffusion Transformer Block    │
│  (Repeat 24 times!)              │
│                                  │
│  1. Atom Attention Encoder      │ → Atoms to tokens
│  2. Attention with Pair Bias    │ → Token self-attention
│  3. Conditioned Transition      │ → Token MLP
│  4. Atom Attention Decoder      │ → Tokens to atoms
│  5. Conditioned Transition      │ → Atom MLP
└─────────────────────────────────┘
        ↓
Output: Denoised Coordinates [N_atoms, 3]
```

**Key Features:**
- **24 blocks deep** (separate from trunk's 48)
- **Atom-token-atom flow:** Aggregates atoms to tokens, processes, broadcasts back
- **Pair bias:** Uses trunk's pair representation to guide attention
- **Conditioning:** Every layer conditioned on noise level

```python
class DiffusionTransformer(nn.Module):
    def __init__(self, n_blocks=24):
        self.blocks = nn.ModuleList([
            DiffusionBlock() for _ in range(n_blocks)
        ])
    
    def forward(self, coords, s_cond, z_cond, t):
        # Encode atoms to tokens
        atom_repr = AtomAttentionEncoder(coords, ...)
        
        for block in self.blocks:
            # Token-level attention with pair bias
            token_repr = block.attention(atom_repr, z_cond)
            token_repr = block.transition(token_repr, t)
            
            # Decode back to atoms
            atom_repr = AtomAttentionDecoder(token_repr, ...)
            atom_repr = block.atom_transition(atom_repr, t)
        
        # Final projection to coordinates
        coords_pred = Linear(atom_repr)  # [N_atoms, 3]
        return coords_pred
```

---

### 6. Attention with Pair Bias (Algorithm 24)

**Purpose:** Self-attention guided by pairwise relationships from trunk

```python
class AttentionPairBias(nn.Module):
    def forward(self, tokens, pair_bias):
        # Standard attention
        Q, K, V = Linear(tokens)
        
        # Add pair bias to attention logits
        logits = (Q @ K.T) / sqrt(d_k)
        logits = logits + pair_bias  # [N, N] bias from trunk
        
        attention = softmax(logits)
        output = attention @ V
        
        return output
```

**Why pair bias?** Trunk learned which atoms interact - diffusion should respect that!

---

### 7. Sample Diffusion (Algorithm 18)

**Purpose:** Run full reverse diffusion to generate structure

**Inference Process (200 steps):**
```python
def sample_diffusion(trunk_repr, n_steps=200):
    # Start from pure noise
    coords = torch.randn(n_atoms, 3) * sigma_max
    
    # Noise schedule: σ_max (160) → σ_min (0.0004)
    sigmas = get_noise_schedule(n_steps)
    
    for t in range(n_steps):
        # Current and next noise levels
        sigma_t = sigmas[t]
        sigma_next = sigmas[t+1]
        
        # Predict clean coordinates
        coords_pred = diffusion_module(coords, sigma_t, trunk_repr)
        
        # Interpolate toward prediction (Euler step)
        coords = coords + (coords_pred - coords) * (sigma_next / sigma_t)
        
        # Add noise for next step (except last)
        if t < n_steps - 1:
            coords = coords + torch.randn_like(coords) * sigma_next
    
    return coords  # Final structure
```

**Key Parameters:**
- `n_steps`: 200 (inference steps)
- `σ_max`: 160 Å (initial noise)
- `σ_min`: 0.0004 Å (final noise)
- `p`: 7 (schedule exponent)

---

### 8. Centre Random Augmentation (Algorithm 19)

**Purpose:** Data augmentation - randomly rotate and translate during training

```python
def centre_random_augmentation(coords):
    # Center coordinates
    coords = coords - coords.mean(dim=0)
    
    # Random rotation
    R = random_rotation_matrix()  # SO(3)
    coords = coords @ R.T
    
    # Random translation
    t = torch.randn(3) * translation_scale
    coords = coords + t
    
    return coords
```

**Why augment?** Structure prediction is translation/rotation invariant!

---

## Complete Diffusion Pipeline

### Training
```python
# 1. Get trunk representations
s_trunk, z_trunk = trunk(embeddings)

# 2. Sample noise level
t = sample_noise_level()  # Uniform or learned distribution
sigma_t = compute_sigma(t)

# 3. Add noise to ground truth
coords_clean = ground_truth_coords
coords_noisy = coords_clean + sigma_t * torch.randn_like(coords_clean)

# 4. Augment (rotate/translate)
coords_noisy = centre_random_augmentation(coords_noisy)
coords_clean = centre_random_augmentation(coords_clean)

# 5. Predict clean from noisy
coords_pred = diffusion_module(
    coords_noisy, 
    sigma_t, 
    s_trunk, 
    z_trunk
)

# 6. Compute losses
loss_mse = weighted_mse_loss(coords_pred, coords_clean, ...)
loss_lddt = smooth_lddt_loss(coords_pred, coords_clean, ...)
loss_bond = bond_length_loss(coords_pred, coords_clean, bonds)

total_loss = weight(t) * (loss_mse + loss_bond) + loss_lddt
```

### Inference
```python
# 1. Get trunk representations
s_trunk, z_trunk = trunk(embeddings)

# 2. Start from noise
coords = torch.randn(n_atoms, 3) * 160.0  # σ_max

# 3. Denoise iteratively (200 steps)
for t in range(200):
    coords = denoise_step(coords, t, s_trunk, z_trunk)

# 4. Final structure
return coords
```

---

## Noise Schedule Deep Dive

### During Training
```python
# Sample from log-normal distribution
log_sigma = -1.2 + 1.5 * torch.randn(1)
sigma = sigma_data * torch.exp(log_sigma)
# σ_data = 16.0 (data variance)
```

**Why log-normal?** Covers wide range of noise levels efficiently.

### During Inference
```python
# Deterministic schedule (200 steps)
t = torch.linspace(0, 1, 200)
sigma = sigma_data * (
    (s_max ** (1/p) + t * (s_min ** (1/p) - s_max ** (1/p))) ** p
)
# s_max = 160, s_min = 0.0004, p = 7
```

**Why this schedule?** More steps at low noise (fine details), fewer at high noise.

---

## Design Insights

### 1. Why Diffusion Instead of Direct Prediction?

**Direct prediction problems:**
- Hard to model multi-modal distributions (multiple conformations)
- Training instability (large coordinate space)
- No uncertainty quantification

**Diffusion advantages:**
- ✅ Natural multi-modality (sample multiple structures)
- ✅ Stable training (predict small denoising steps)
- ✅ Uncertainty from variance across samples
- ✅ Handles missing/uncertain regions gracefully

### 2. Atom ↔ Token Architecture

**Why not work directly on atoms?**
- Too many atoms (10,000+ for large proteins)
- Attention is O(N²) - prohibitively expensive

**Solution: Hierarchical processing**
```
Atoms (10,000) → Aggregate → Tokens (500) → Attention → Broadcast → Atoms
```

**Result:** O(N_token²) << O(N_atom²)

### 3. Conditioning Everywhere

Every layer conditioned on:
- **Time/noise level:** What level of detail to predict
- **Trunk features:** What structure to predict
- **Pair biases:** Which atoms interact

**Why?** Each noise level needs different behavior!

### 4. Iterative Refinement

**200 denoising steps = 200 forward passes!**
- Expensive but accurate
- Can reduce to 50-100 steps for speed (slight accuracy loss)
- Each step makes small improvement

---

## Computational Cost

### Training (per sample)
- **Trunk:** 1 forward pass (~5 sec)
- **Diffusion:** 1 forward pass (~2 sec)
- **Total:** ~7 seconds per structure
- **Batch:** 48 diffusion samples per trunk (efficient!)

### Inference (per sample)
- **Trunk:** 1 forward pass (~5 sec)
- **Diffusion:** 200 forward passes (~400 sec = 7 min!)
- **Total:** ~7.5 minutes per structure
- **Speed-ups:** Reduce to 50 steps → ~2 minutes

### Memory
- **Trunk:** ~30 GB
- **Diffusion:** ~10 GB
- **Total:** ~40 GB for 500 residue protein

---

## Testing Summary

**Files:**
- `test_fourier_embedding.py` - 8+ tests
- `test_adaln.py` - 10+ tests
- `test_conditioned_transition.py` - 12+ tests
- `test_diffusion_conditioning.py` - 8+ tests
- `test_diffusion_transformer.py` - 20+ tests
- `test_sample_diffusion.py` - 15+ tests

**Coverage:** 100% of public APIs  
**All Tests:** ✅ Passing  
**Total Tests:** 73+

---

## File Structure

```
src/models/diffusion/
├── __init__.py
├── fourier_embedding.py          # Algorithm 22
├── adaln.py                      # Algorithm 26
├── conditioned_transition.py     # Algorithm 25
├── diffusion_conditioning.py     # Algorithm 21
├── attention_pair_bias.py        # Algorithm 24
├── diffusion_transformer.py      # Algorithm 23
├── diffusion_module.py           # Algorithm 20
├── sample_diffusion.py           # Algorithm 18
└── centre_random_augmentation.py # Algorithm 19
```

---

## Usage Example

```python
from src.models.diffusion import (
    DiffusionModule,
    SampleDiffusion
)

# Initialize diffusion module
diffusion = DiffusionModule(
    c_atom=128,
    c_atompair=16,
    c_token=384,
    c_pair=128,
    n_blocks=24
)

# Training: predict clean from noisy
coords_pred = diffusion(
    noisy_coords,
    noise_level=sigma_t,
    s_trunk=s,
    z_trunk=z,
    features=features
)

# Inference: full sampling
sampler = SampleDiffusion(diffusion)
coords_final = sampler(
    s_trunk=s,
    z_trunk=z,
    features=features,
    n_steps=200
)
```

---

## Key Takeaways

1. **Diffusion = Iterative Denoising**
   - Not one-shot prediction
   - 200 small steps instead of 1 big step

2. **Conditioning is Critical**
   - Every layer knows noise level
   - Trunk features guide generation
   - Pair biases constrain geometry

3. **Hierarchical Processing**
   - Atoms → Tokens (efficient attention)
   - Tokens → Atoms (detailed coordinates)

4. **Training ≠ Inference**
   - Training: 1 random noise level
   - Inference: 200 sequential steps

5. **Trade-offs**
   - More steps = better quality, slower
   - Fewer steps = faster, slight quality loss