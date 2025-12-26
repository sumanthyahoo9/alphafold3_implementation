# AlphaFold3 Heads Module Summary

## Overview

The heads module predicts **quality metrics** for generated structures. Instead of just outputting coordinates, AlphaFold3 also tells you **how confident it is** in each prediction.

**Why crucial?** Knowing which parts of a structure are reliable vs uncertain is essential for downstream applications!

**Total Algorithms Implemented:** 1 (Confidence Head with 4 outputs)

---

## Module Architecture

```
Trunk + Diffusion → Structure Coordinates
                         ↓
                  Confidence Head → Quality Metrics
                         ↑
                 (You are here!)
```

---

## Confidence Head (Algorithm 31)

### What It Does
Predicts 4 different quality metrics from structure and representations:

1. **pLDDT** - Per-atom confidence (0-1)
2. **PAE** - Predicted Aligned Error between token pairs (Å)
3. **PDE** - Predicted Distance Error between token pairs (Å)
4. **Resolved** - Whether atom is experimentally resolved (binary)

### Architecture (9 lines!)
```
Input: s_inputs, s_trunk, z_trunk, x_pred
        ↓
1. Update pair: z += Linear(s_inputs_i) + Linear(s_inputs_j)
2-3. Embed distances of representative atoms
4. Small Pairformer (4 blocks)
5. Project to pAE (64 bins)
6. Project to PDE (64 bins)
7. Project to pLDDT (50 bins per atom)
8. Project to Resolved (2 bins per atom)
9. Return all predictions
```

### Implementation
```python
class ConfidenceHead(nn.Module):
    def __init__(self, c_single=384, c_pair=128, n_blocks=4):
        # Algorithm 31, line 1
        self.s_inputs_proj_i = nn.Linear(c_single, c_pair, bias=False)
        self.s_inputs_proj_j = nn.Linear(c_single, c_pair, bias=False)
        
        # Algorithm 31, line 3
        self.distance_proj = nn.Linear(64, c_pair, bias=False)
        
        # Algorithm 31, line 4
        self.pairformer = PairformerStack(c_single, c_pair, n_blocks=4)
        
        # Algorithm 31, lines 5-8
        self.pae_proj = nn.Linear(c_pair, 64, bias=False)
        self.pde_proj = nn.Linear(c_pair, 64, bias=False)
        self.plddt_proj = nn.Linear(c_single, max_atoms * 50, bias=False)
        self.resolved_proj = nn.Linear(c_single, max_atoms * 2, bias=False)
    
    def forward(self, s_inputs, s, z, x_pred, atom_to_token_idx):
        # Stop gradients (no backprop to structure)
        s = s.detach()
        z = z.detach()
        x_pred = x_pred.detach()
        
        # Add s_inputs to pair
        z = z + self.s_inputs_proj_i(s_inputs).unsqueeze(1)
        z = z + self.s_inputs_proj_j(s_inputs).unsqueeze(0)
        
        # Embed pairwise distances
        distances = compute_pairwise_distances(x_pred, atom_to_token_idx)
        distance_one_hot = one_hot_encode(distances)
        z = z + self.distance_proj(distance_one_hot)
        
        # Refine with small Pairformer
        s, z = self.pairformer(s, z)
        
        # Predict all metrics
        p_pae = softmax(self.pae_proj(z))
        p_pde = softmax(self.pde_proj(z + z.T))  # Symmetrize
        p_plddt = softmax(self.plddt_proj(s))
        p_resolved = softmax(self.resolved_proj(s))
        
        return p_plddt, p_pae, p_pde, p_resolved
```

---

## 1. pLDDT - Per-Atom Confidence

### What It Measures
**Local Distance Difference Test** - How well local distances are preserved.

### Definition (Equation 8)
```
For each atom l:
lddtl = (1/4) × Σ_{m∈R} Σ_{c∈{0.5,1,2,4}} [d_lm < c]

Where:
- d_lm: distance between atoms l and m in prediction
- R: atoms within 15Å (protein) or 30Å (nucleotides) in ground truth
- Thresholds: 0.5Å, 1Å, 2Å, 4Å
```

### Prediction
```python
# 50 bins from 0 to 1
p_plddt = softmax(Linear(s)[atom_idx])  # [N_atoms, 50]

# Convert to score
plddt_score = (p_plddt * bin_centers).sum(dim=-1)  # [N_atoms]
# Range: 0-1, higher = better
```

### Interpretation
| pLDDT Score | Interpretation |
|-------------|----------------|
| > 90 | Very high confidence - experimental quality |
| 70-90 | High confidence - likely correct |
| 50-70 | Low confidence - backbone likely correct |
| < 50 | Very low confidence - likely disordered |

### Visualizations
```python
# Color structure by pLDDT
colors = {
    plddt > 90: 'blue',      # Very confident
    70 < plddt <= 90: 'cyan',
    50 < plddt <= 70: 'yellow',
    plddt <= 50: 'orange'    # Not confident
}
```

---

## 2. PAE - Predicted Aligned Error

### What It Measures
**Error when aligning frame i to frame j** - If I align the structure according to residue i, how wrong is residue j?

### Definition (Algorithm 30)
```python
def compute_alignment_error(x_pred, x_gt, frame_atoms):
    # Build frames from 3 atoms per token
    frame_i_pred = build_frame(x_pred[frame_atoms[i]])
    frame_i_gt = build_frame(x_gt[frame_atoms[i]])
    
    # Express token j in both frames
    x_j_in_i_pred = express_in_frame(x_pred[j], frame_i_pred)
    x_j_in_i_gt = express_in_frame(x_gt[j], frame_i_gt)
    
    # Error = distance after alignment
    error_ij = ||x_j_in_i_pred - x_j_in_i_gt||
    return error_ij
```

### Prediction
```python
# 64 bins from 0 to 32 Angstroms (0.5Å increments)
p_pae = softmax(Linear(z_ij))  # [N_tokens, N_tokens, 64]

# Convert to scores
pae_score = (p_pae * bin_centers).sum(dim=-1)  # [N_tokens, N_tokens]
# Range: 0-32 Å, lower = better
```

### Interpretation
**PAE Matrix** shows confidence in relative positioning:
```
       j →
    ┌─────────┐
  i │  Low    │ = Good! Confident in i-j relative position
  ↓ │         │
    │  High   │ = Bad! Uncertain about i-j relationship
    └─────────┘
```

**Example:**
- `PAE[10,50] = 2Å` → If aligned by residue 10, residue 50 has ~2Å error
- `PAE[10,50] = 20Å` → If aligned by residue 10, residue 50 is very uncertain

### Applications
1. **Domain detection:** Low PAE within domains, high between domains
2. **Interface confidence:** Low PAE = confident interface
3. **Multi-chain accuracy:** High PAE between chains = uncertain complex
4. **ipTM score:** Aggregate PAE into single metric

---

## 3. PDE - Predicted Distance Error

### What It Measures
**Direct distance error** - How wrong is the predicted distance between atoms i and j?

### Definition
```
error_ij = |distance_pred(i,j) - distance_gt(i,j)|
```

### Prediction
```python
# 64 bins from 0 to 32 Angstroms
# Symmetrize pair representation
z_sym = z_ij + z_ji

p_pde = softmax(Linear(z_sym))  # [N_tokens, N_tokens, 64]

# Convert to scores
pde_score = (p_pde * bin_centers).sum(dim=-1)  # [N_tokens, N_tokens]
# Range: 0-32 Å, lower = better
```

### Difference from PAE
| Metric | What It Measures | Use Case |
|--------|------------------|----------|
| **PAE** | Alignment error | Relative orientation |
| **PDE** | Distance error | Absolute distances |

**Example:**
- Two domains correctly positioned but wrong relative orientation:
  - PAE: High (bad alignment)
  - PDE: Low (distances correct)

---

## 4. Resolved - Experimental Resolution

### What It Measures
**Whether atom is visible in experimental structure** - Some atoms are disordered/missing in crystals.

### Prediction
```python
# 2 bins: [not resolved, resolved]
p_resolved = softmax(Linear(s)[atom_idx])  # [N_atoms, 2]

# Probability atom is resolved
prob_resolved = p_resolved[:, 1]  # [N_atoms]
# Range: 0-1, higher = more likely visible in experiment
```

### Why Predict This?
- **Flexibility:** Loops, termini often disordered
- **Confidence:** Low resolution prediction = uncertain region
- **Filtering:** Exclude disordered regions from analysis

---

## Confidence Head Design

### Why Small Pairformer?
**4 additional blocks** to refine representations specifically for confidence:
```python
# Trunk: 48 blocks for structure
# Confidence: +4 blocks for quality metrics

# Why separate? Different objectives:
# - Structure: Predict coordinates
# - Confidence: Predict accuracy
```

### Why Stop Gradients?
```python
s = s.detach()
z = z.detach()
x_pred = x_pred.detach()
```

**Reason:** Confidence head doesn't improve structure prediction during training - it only predicts quality.

### Distance Embedding
**Why embed predicted distances?**
- Confidence depends on geometry
- Close atoms = more information
- Distance patterns reveal errors

**Distance bins:** 3⅜Å to 21⅜Å (64 bins)
- Below 3⅜Å: Very close (bonded)
- Above 21⅜Å: Too far to influence confidence

---

## Training Confidence Head

### Losses (Equations 9, 10, 12, 14)

**1. pLDDT Loss:**
```python
# Cross-entropy with binned LDDT
loss_plddt = -mean(lddtb[l] * log(p_plddt[l]))
```

**2. PAE Loss:**
```python
# Cross-entropy with binned alignment errors
loss_pae = -mean(errorb[ij] * log(p_pae[ij]))
```

**3. PDE Loss:**
```python
# Cross-entropy with binned distance errors
loss_pde = -mean(errorb[ij] * log(p_pde[ij]))
```

**4. Resolved Loss:**
```python
# Binary cross-entropy
loss_resolved = -mean(resolved[l] * log(p_resolved[l]))
```

**Total:**
```python
loss_confidence = (
    loss_plddt + 
    loss_pae + 
    loss_pde + 
    loss_resolved
)
```

---

## Derived Metrics

### pTM Score
**Predicted TM-Score** - Global structure quality (0-1)

```python
def compute_pTM(pae_probs):
    # TM-Score from PAE probabilities
    d0 = 1.24 * (N_res - 15) ** (1/3) - 1.8  # Normalization
    
    scores = []
    for i, j in all_pairs:
        # Convert PAE bins to probability of being close
        prob_close = sum(pae_probs[i,j,bins < d0])
        scores.append(prob_close)
    
    pTM = mean(scores)
    return pTM  # Range: 0-1, higher = better
```

**Interpretation:**
- pTM > 0.8: High confidence in overall fold
- pTM 0.5-0.8: Medium confidence
- pTM < 0.5: Low confidence

### ipTM Score
**Interface pTM** - Confidence in protein-protein interfaces

```python
def compute_ipTM(pae_probs, chains):
    # Only consider inter-chain pairs
    pairs = [(i,j) for i,j in all_pairs 
             if chain[i] != chain[j]]
    
    # Compute pTM only on interface
    return compute_pTM_subset(pae_probs, pairs)
```

**Use cases:**
- Protein complexes
- Antibody-antigen
- Protein-DNA/RNA

---

## Practical Usage

### Example 1: Filter Low Confidence Regions
```python
# Get predictions
plddt, pae, pde, resolved = confidence_head(...)

# Convert to scores
plddt_scores = get_plddt_scores(plddt)  # [N_atoms]

# Filter
confident_atoms = plddt_scores > 70
filtered_structure = structure[confident_atoms]
```

### Example 2: Identify Domains
```python
# Get PAE matrix
pae_scores = get_pae_scores(pae)  # [N_tokens, N_tokens]

# Low PAE = same domain
# High PAE = different domains
domains = cluster_by_pae(pae_scores, threshold=5.0)
```

### Example 3: Rank Multiple Predictions
```python
# Generate 5 samples
structures = [sample_diffusion() for _ in range(5)]

# Compute confidence for each
ptm_scores = [
    compute_pTM(confidence_head(s)) 
    for s in structures
]

# Rank by confidence
best_structure = structures[argmax(ptm_scores)]
```

---

## Visualization

### pLDDT Coloring (PyMOL)
```python
# Color by confidence
for atom, plddt in zip(atoms, plddt_scores):
    if plddt > 90:
        color = 'blue'    # Very high
    elif plddt > 70:
        color = 'cyan'    # High
    elif plddt > 50:
        color = 'yellow'  # Medium
    else:
        color = 'orange'  # Low
    
    pymol.color(color, atom)
```

### PAE Heatmap
```python
import matplotlib.pyplot as plt

plt.imshow(pae_scores, cmap='Greens_r', vmin=0, vmax=30)
plt.colorbar(label='Expected position error (Å)')
plt.xlabel('Aligned residue')
plt.ylabel('Scored residue')
plt.title('Predicted Aligned Error')
```

---

## Testing Summary

**Files:**
- `test_confidence_head.py` - 25+ tests

**Coverage:** 100% of public APIs  
**All Tests:** ✅ Passing

**Test categories:**
- Initialization
- Forward pass shapes
- Probability constraints (sum to 1)
- Score conversion utilities
- Algorithm 31 faithfulness

---

## File Structure

```
src/models/heads/
├── __init__.py
└── confidence_head.py  # Algorithm 31
```

---

## Usage Example

```python
from src.models.heads import ConfidenceHead

# Initialize
confidence_head = ConfidenceHead(
    c_single=384,
    c_pair=128,
    n_blocks=4
)

# Predict confidence metrics
p_plddt, p_pae, p_pde, p_resolved = confidence_head(
    s_inputs=s_inputs,        # Input embeddings
    s=s_trunk,                # Final single repr
    z=z_trunk,                # Final pair repr
    x_pred=predicted_coords,  # Structure coordinates
    atom_to_token_idx=mapping
)

# Convert to scores
plddt_scores = confidence_head.get_plddt_scores(p_plddt)
pae_scores = confidence_head.get_pae_scores(p_pae)
pde_scores = confidence_head.get_pde_scores(p_pde)

# Compute aggregate metrics
ptm = compute_pTM(p_pae)
iptm = compute_ipTM(p_pae, chain_ids)

print(f"pTM: {ptm:.3f}")  # Overall confidence
print(f"ipTM: {iptm:.3f}")  # Interface confidence
print(f"Mean pLDDT: {plddt_scores.mean():.1f}")  # Per-atom confidence
```

---

## Key Takeaways

1. **Four Complementary Metrics**
   - pLDDT: Per-atom local confidence
   - PAE: Relative positioning confidence
   - PDE: Distance accuracy
   - Resolved: Experimental visibility

2. **Stop Gradients**
   - Confidence doesn't improve structure
   - Trained separately from structure prediction

3. **Binned Predictions**
   - All metrics predict probability distributions
   - Convert to scores via expectation
   - More informative than single values

4. **Critical for Downstream Use**
   - Filter unreliable regions
   - Identify domains and interfaces
   - Rank multiple predictions
   - Guide experimental validation

5. **Calibrated Predictions**
   - Trained on experimental structures
   - Learned correlations between features and quality
   - Generalizes to novel structures