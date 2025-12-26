## Overview

AlphaFold3 uses multiple loss functions to train the model to predict accurate 3D protein structures. Each loss focuses on different aspects of structural quality.

**Total Loss Functions Implemented:** 5

---

## 1. Smooth LDDT Loss (Algorithm 27)

### What It Measures
**Local Distance Difference Test (LDDT)** - Measures how well local pairwise distances are preserved between predicted and ground truth structures.

### Why It Matters
- More robust than global RMSD for local structure quality
- Standard metric in CASP protein structure competitions
- Evaluates if nearby atoms maintain correct relative positions

### How It Works
1. Compute all pairwise distances in prediction and ground truth
2. Calculate distance errors: `δ = |dist_pred - dist_gt|`
3. Apply smooth sigmoid scoring at 4 thresholds: 0.5Å, 1Å, 2Å, 4Å
4. Average scores over atom pairs within inclusion radius:
   - **Proteins:** 15Å radius
   - **Nucleotides (DNA/RNA):** 30Å radius
5. Return: `loss = 1 - LDDT` (range: 0-1, lower is better)

### Mathematical Formula
```
ε_lm = (1/4) × Σ sigmoid(threshold - δ_lm)
LDDT = mean(inclusion_mask × ε) / mean(inclusion_mask)
Loss = 1 - LDDT
```

### Simple Example
```python
from src.models.losses import SmoothLDDTLoss

loss_fn = SmoothLDDTLoss()

# Perfect prediction
x_pred = torch.randn(20, 3)
x_gt = x_pred.clone()
loss = loss_fn(x_pred, x_gt)
# Result: loss ≈ 0.2 (due to sigmoid saturation, not 0.0)

# Bad prediction
x_pred = torch.randn(20, 3)
x_gt = torch.randn(20, 3) * 10  # Very different
loss = loss_fn(x_pred, x_gt)
# Result: loss ≈ 0.9 (close to 1.0)
```

### Key Parameters
- `thresholds`: [0.5, 1.0, 2.0, 4.0] Angstroms
- `radius_protein`: 15.0 Angstroms
- `radius_nucleotide`: 30.0 Angstroms

---

## 2. Weighted Rigid Align (Algorithm 28)

### What It Measures
**Not a loss function** - This is a utility that finds optimal rotation and translation to align two structures.

### Why It Matters
- Required for computing alignment-invariant losses
- Uses weighted Kabsch algorithm for optimal superposition
- Ensures we measure structural similarity, not just coordinate differences

### How It Works
1. Compute weighted centroids of both structures
2. Center coordinates around centroids
3. Compute weighted covariance matrix: `H = x^T @ x_gt`
4. SVD decomposition: `H = U @ S @ V^T`
5. Compute rotation: `R = U @ V^T`
6. Handle reflections: if `det(R) < 0`, flip sign of last singular vector
7. Apply transformation: `x_aligned = R @ x_gt + translation`

### Mathematical Formula
```
μ = Σ(w × x) / Σ(w)
H = x_centered^T @ x_gt_centered
R = U @ V^T  (from SVD of H)
x_aligned = R @ x_gt_centered + μ
```

### Simple Example
```python
from src.models.losses import weighted_rigid_align

# Rotate structure by 90 degrees
angle = math.pi / 2
R = torch.tensor([
    [cos(angle), -sin(angle), 0],
    [sin(angle), cos(angle), 0],
    [0, 0, 1]
])

x_pred = torch.randn(10, 3)
x_gt = (R @ x_pred.T).T  # Rotated version

# Align x_gt back to x_pred's frame
x_aligned = weighted_rigid_align(x_pred, x_gt)

# x_aligned should now be very close to x_pred
rmsd = torch.sqrt(((x_aligned - x_pred)**2).mean())
# Result: rmsd ≈ 0.0001 (nearly perfect)
```

### Key Features
- Handles weighted atoms (upweight important molecules)
- Removes reflections (ensures proper rotations)
- Stops gradients (for training stability)

---

## 3. Weighted MSE Loss (Equations 3-4)

### What It Measures
**Mean Squared Error of coordinates** after optimal alignment, with molecule-specific upweighting.

### Why It Matters
- Direct measure of coordinate prediction accuracy
- Upweights challenging molecules (ligands 10x, nucleotides 5x)
- Forces model to prioritize difficult predictions

### How It Works
1. Compute per-atom weights:
   - Base weight: 1.0
   - DNA atoms: +5.0
   - RNA atoms: +5.0
   - Ligand atoms: +10.0
2. Align ground truth to prediction using `weighted_rigid_align`
3. Compute weighted squared distances
4. Average and divide by 3 (as per paper)

### Mathematical Formula
```
w = 1 + 5×is_dna + 5×is_rna + 10×is_ligand
x_gt_aligned = weighted_rigid_align(x_pred, x_gt, w)
L_MSE = (1/3) × mean(w × ||x_pred - x_gt_aligned||²)
```

### Simple Example
```python
from src.models.losses import WeightedMSELoss

loss_fn = WeightedMSELoss(alpha_dna=5.0, alpha_rna=5.0, alpha_ligand=10.0)

# 10 protein atoms + 5 ligand atoms
x_pred = torch.randn(15, 3)
x_gt = torch.randn(15, 3)

is_ligand = torch.zeros(15, dtype=torch.bool)
is_ligand[10:] = True  # Last 5 are ligands

loss = loss_fn(x_pred, x_gt, is_ligand=is_ligand)
# Ligand atoms contribute 10x more to loss!
```

### Molecule Weights
| Molecule Type | Weight Multiplier |
|--------------|-------------------|
| Protein      | 1.0               |
| DNA          | 6.0 (1 + 5)       |
| RNA          | 6.0 (1 + 5)       |
| Ligand       | 11.0 (1 + 10)     |

---

## 4. Bond Length Loss (Equation 5)

### What It Measures
**Bond length preservation** for bonded ligands (e.g., glycans covalently attached to proteins).

### Why It Matters
- Ensures chemical validity of predictions
- Critical during fine-tuning for bonded molecules
- Maintains correct covalent bond geometry

### How It Works
1. For each bond `(atom_i, atom_j)`:
   - Compute predicted bond length: `||x_pred[i] - x_pred[j]||`
   - Compute ground truth bond length: `||x_gt[i] - x_gt[j]||`
2. Calculate squared error: `(pred_length - gt_length)²`
3. Average over all bonds

### Mathematical Formula
```
L_bond = mean_{(i,j) ∈ Bonds} (||x_pred[i] - x_pred[j]|| - ||x_gt[i] - x_gt[j]||)²
```

### Simple Example
```python
from src.models.losses import BondLengthLoss

loss_fn = BondLengthLoss()

# Ground truth: atoms 0-1 are 1.0Å apart
x_gt = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

# Prediction: atoms 0-1 are 1.5Å apart (wrong!)
x_pred = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])

bonds = [(0, 1)]  # Bond between atoms 0 and 1

loss = loss_fn(x_pred, x_gt, bonds)
# Result: loss = (1.5 - 1.0)² = 0.25
```

### When It's Used
- **Training:** `α_bond = 0` (disabled)
- **Fine-tuning Stage 1:** `α_bond = 1` (enabled)
- **Fine-tuning Stage 2:** `α_bond = 1` (enabled)

---

## 5. Combined Diffusion Loss (Equation 6)

### What It Measures
**Total training loss** combining all components with time-dependent weighting.

### Why It Matters
- Complete loss function for diffusion model training
- Balances coordinate accuracy, bond geometry, and local structure
- Time-dependent weighting handles varying noise levels

### How It Works
1. Compute MSE loss (with molecule upweighting)
2. Compute bond length loss (if bonds present)
3. Compute smooth LDDT loss
4. Apply time-dependent weight to reconstruction losses
5. Combine all components

### Mathematical Formula
```
weight(t) = (t² + σ_data²) / (t + σ_data)²
L_diffusion = weight(t) × (L_MSE + α_bond × L_bond) + L_smooth_lddt

where:
- t: noise level (sampled during training)
- σ_data: data variance constant (16.0)
- α_bond: bond loss weight (0 or 1)
```

### Simple Example
```python
from src.models.losses import DiffusionLoss, SmoothLDDTLoss

# Initialize combined loss
loss_fn = DiffusionLoss(
    alpha_bond=1.0,      # Enable bond loss
    sigma_data=16.0,     # Data variance
    alpha_ligand=10.0    # Ligand upweighting
)

# Predictions and ground truth
x_pred = torch.randn(20, 3)
x_gt = torch.randn(20, 3)
noise_level = torch.tensor(5.0)

# Pre-compute LDDT loss
lddt_fn = SmoothLDDTLoss()
smooth_lddt = lddt_fn(x_pred, x_gt)

# Bonds for bonded ligands
bonds = [(0, 5), (10, 15)]

# Mark ligand atoms
is_ligand = torch.zeros(20, dtype=torch.bool)
is_ligand[15:] = True

# Compute total loss
loss = loss_fn(
    x_pred, x_gt, 
    noise_level, 
    smooth_lddt,
    is_ligand=is_ligand,
    bonds=bonds
)
# Result: weighted combination of all losses
```

### Loss Components

| Component | Weight | Purpose |
|-----------|--------|---------|
| MSE Loss | `weight(t)` | Coordinate accuracy |
| Bond Loss | `weight(t) × α_bond` | Bond geometry |
| LDDT Loss | 1.0 | Local structure |

### Time Weighting Function

The `weight(t)` function adapts to noise level:

```python
# Low noise (t=0.1): weight ≈ 0.004 (small gradient)
# Medium noise (t=5): weight ≈ 0.69 (balanced)
# High noise (t=50): weight ≈ 0.91 (large gradient)
```

**Why?** At high noise, the model needs strong supervision. At low noise, predictions are already good, so we reduce gradient magnitude.

---

## Loss Function Comparison

### When to Use Each Loss

| Loss | Training | Fine-tuning | Evaluation |
|------|----------|-------------|------------|
| Smooth LDDT | ✅ Always | ✅ Always | ✅ Yes |
| Weighted MSE | ✅ Always | ✅ Always | ❌ No |
| Bond Length | ❌ No | ✅ Yes | ❌ No |
| Combined | ✅ Training | ✅ Fine-tuning | ❌ No |

### Computational Complexity

| Loss | Complexity | Notes |
|------|-----------|-------|
| Smooth LDDT | O(N²) | All pairwise distances |
| Weighted Align | O(N) | SVD is O(1) for 3×3 |
| Weighted MSE | O(N) | After alignment |
| Bond Length | O(B) | B = number of bonds |
| Combined | O(N²) | Dominated by LDDT |

---

## Complete Training Example

```python
from src.models.losses import (
    DiffusionLoss,
    SmoothLDDTLoss
)

# Initialize loss functions
diffusion_loss = DiffusionLoss(
    alpha_bond=1.0,         # Fine-tuning mode
    sigma_data=16.0,
    alpha_dna=5.0,
    alpha_rna=5.0,
    alpha_ligand=10.0
)

lddt_loss_fn = SmoothLDDTLoss()

# Training step
def compute_loss(x_pred, x_gt, t, features):
    # Extract molecule types
    is_dna = features['is_dna']
    is_rna = features['is_rna']
    is_ligand = features['is_ligand']
    bonds = features.get('bonds', None)
    
    # Compute LDDT component
    smooth_lddt = lddt_loss_fn(
        x_pred, x_gt, 
        is_dna=is_dna, 
        is_rna=is_rna
    )
    
    # Compute total diffusion loss
    loss = diffusion_loss(
        x_pred, x_gt, 
        noise_level=t,
        smooth_lddt_loss=smooth_lddt,
        is_dna=is_dna,
        is_rna=is_rna,
        is_ligand=is_ligand,
        bonds=bonds
    )
    
    return loss

# Example usage
x_pred = model(noisy_coords, t, features)
loss = compute_loss(x_pred, x_gt, t, features)
loss.backward()
optimizer.step()
```

---

## Key Insights

### 1. Multi-Scale Evaluation
- **Global:** MSE measures overall coordinate accuracy
- **Local:** LDDT measures local structure quality
- **Chemical:** Bond loss ensures valid geometry

### 2. Hierarchical Weighting
- Time-dependent: Focus on different noise levels
- Molecule-specific: Prioritize challenging molecules
- Component-specific: Balance different objectives

### 3. Alignment Invariance
- All losses use `weighted_rigid_align` first
- Measures structural similarity, not coordinate differences
- Critical for rotation/translation invariance

### 4. Smooth & Differentiable
- Sigmoid functions instead of hard thresholds
- Enables gradient-based optimization
- Maintains stability during training

---

## References

1. **AlphaFold3 Paper:** Abramson et al., Nature 2024
2. **Supplementary Materials:** Algorithms 27-28, Equations 3-6
3. **Implementation:** `src/models/losses/`

---

## File Structure

```
src/models/losses/
├── __init__.py                    # Main exports
├── smooth_lddt_loss.py           # Algorithm 27
├── weighted_rigid_align.py       # Algorithm 28
└── mse_losses.py                 # Equations 3-6

tests/
├── test_smooth_lddt_loss.py      # 15+ tests
├── test_weighted_rigid_align.py  # 25+ tests
└── test_mse_losses.py            # 20+ tests
```

