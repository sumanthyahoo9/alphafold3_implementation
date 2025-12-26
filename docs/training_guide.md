# AlphaFold3 Training Quick Start Guide

## Overview

This guide covers distributed training of AlphaFold3 on multiple GPUs.

**Total Parameters:** ~400M  
**Minimum GPU Memory:** 40GB per GPU  
**Recommended Setup:** 8x A100 (80GB) or H100

---

## File Structure

```
alphafold3/
├── scripts/
│   ├── train.py              # Main training script
│   ├── inference.py          # Inference script
│   └── config.yaml           # Training configuration
├── src/
│   ├── models/
│   │   ├── alphafold3.py     # Main model
│   │   ├── embeddings/
│   │   ├── trunk/
│   │   ├── diffusion/
│   │   ├── heads/
│   │   └── losses/
│   └── data/
│       ├── tokenizer.py      # Tokenization
│       ├── featurizer.py     # Feature extraction
│       └── dataset.py        # Dataset loading
└── data/
    └── pdb/
        ├── train/            # Training data
        └── val/              # Validation data
```

---

## Installation

### 1. Create Environment

```bash
conda create -n alphafold3 python=3.10
conda activate alphafold3
```

### 2. Install Dependencies

```bash
# PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install numpy scipy pyyaml

# Optional: Logging
pip install wandb tensorboard

# Optional: Data processing
pip install biopython rdkit-pypi
```

---

## Data Preparation

### Option 1: Use Mock Data (for testing)

```bash
# Create mock dataset
python src/data/dataset.py

# This creates:
# - ./data/pdb/train (100 samples)
# - ./data/pdb/val (20 samples)
```

### Option 2: Real PDB Data (for production)

**TODO:** Implement full PDB preprocessing pipeline

Would include:
1. Download PDB structures
2. Generate MSAs (HHBlits, Jackhmmer)
3. Search templates
4. Generate reference conformers (RDKit)
5. Preprocess and cache features

---

## Training

### Single GPU Training

```bash
python scripts/train.py --config scripts/config.yaml
```

### Multi-GPU Training (Single Node)

```bash
# 8 GPUs
torchrun --nproc_per_node=8 scripts/train.py --config scripts/config.yaml
```

### Multi-Node Training (DDP)

**Node 0 (Master):**
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    scripts/train.py --config scripts/config.yaml
```

**Node 1-3 (Workers):**
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=<1,2,3> \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    scripts/train.py --config scripts/config.yaml
```

### Resume from Checkpoint

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --config scripts/config.yaml \
    --resume ./outputs/alphafold3_run1/checkpoints/latest.pt
```

---

## Configuration

Edit `scripts/config.yaml` to customize training:

### Key Parameters

**Model:**
```yaml
model:
  c_token: 384          # Increase for larger model
  n_cycles: 4           # Recycling iterations (1-4)
  pairformer_blocks: 48 # Trunk depth
```

**Training:**
```yaml
training:
  num_epochs: 100
  mixed_precision: true             # FP16/BF16
  gradient_accumulation_steps: 8    # Effective batch size multiplier
  gradient_clip: 1.0                # Gradient clipping
```

**Data:**
```yaml
data:
  batch_size: 1         # Per-GPU batch size
  max_tokens: 512       # Max sequence length
```

**Effective Batch Size:**
```
effective_batch = batch_size × num_gpus × grad_accum_steps
                = 1 × 8 × 8 = 64 structures
```

---

## Memory Optimization

### For Limited GPU Memory (<40GB)

1. **Reduce batch size:**
   ```yaml
   data:
     batch_size: 1  # Already minimum
   ```

2. **Increase gradient accumulation:**
   ```yaml
   training:
     gradient_accumulation_steps: 16  # Compensate for batch_size=1
   ```

3. **Enable gradient checkpointing:**
   ```yaml
   hardware:
     gradient_checkpointing: true  # Trades compute for memory
   ```

4. **Mixed precision:**
   ```yaml
   training:
     mixed_precision: true  # ~2x memory savings
   ```

### For Very Large Models

1. **Model parallelism** (not yet implemented):
   - Split model across GPUs
   - Use DeepSpeed or FSDP

2. **CPU offloading** (very slow):
   ```yaml
   hardware:
     cpu_offload: true
   ```

---

## Monitoring Training

### Console Logging

Training automatically logs to console:
```
Epoch 0 | Step 100 | Loss: 0.4523 | LDDT: 0.3245 | LR: 1.00e-04
```

### Weights & Biases

Enable W&B in config:
```yaml
logging:
  use_wandb: true
  wandb_project: alphafold3
  wandb_run_name: my_experiment
```

Then:
```bash
wandb login
torchrun --nproc_per_node=8 scripts/train.py --config scripts/config.yaml
```

View at: https://wandb.ai

---

## Checkpoints

Checkpoints saved to `./outputs/<run_name>/checkpoints/`:

- `latest.pt` - Latest checkpoint (overwritten each epoch)
- `best.pt` - Best validation loss
- `checkpoint_epoch_N.pt` - Periodic checkpoints

### Checkpoint Contents

```python
{
    'epoch': int,
    'global_step': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'scaler_state_dict': dict,
    'best_val_loss': float,
    'config': dict
}
```

---

## Inference

After training, run inference:

```bash
python scripts/inference.py \
    --input protein.fasta \
    --checkpoint ./outputs/alphafold3_run1/checkpoints/best.pt \
    --output ./predictions/ \
    --n_samples 5
```

See `INFERENCE_GUIDE.md` for details.

---

## Training Stages

AlphaFold3 training has 3 stages:

### Stage 1: Pre-training (Current)
- No bond loss (`alpha_bond: 0.0`)
- Focus on structure prediction
- Train for ~100 epochs

### Stage 2: Fine-tuning with Bonds
- Enable bond loss (`alpha_bond: 1.0`)
- Refine bonded ligand geometry
- Train for ~20 epochs

### Stage 3: Fine-tuning with Confidence
- Add confidence head loss
- Train confidence predictions
- Train for ~10 epochs

**Switch stages** by modifying `config.yaml`:
```yaml
losses:
  alpha_bond: 1.0  # Enable for stage 2+
```

---

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `batch_size` to 1
2. Increase `gradient_accumulation_steps`
3. Enable `gradient_checkpointing`
4. Reduce `max_tokens` (crop sequences)
5. Use fewer recycling iterations (`n_cycles: 2`)

### Slow Training

1. Check GPU utilization: `nvidia-smi`
2. Increase `num_workers` in data loading
3. Reduce diffusion steps during training (save for inference)
4. Profile with PyTorch profiler

### NaN Loss

1. Reduce learning rate
2. Enable gradient clipping
3. Check input data for NaNs
4. Use mixed precision carefully (switch to FP32 if issues)

### Multi-GPU Issues

1. Check NCCL: `python -c "import torch; print(torch.cuda.nccl.version())"`
2. Set environment variables:
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_IB_DISABLE=1  # If using InfiniBand
   ```
3. Verify all GPUs visible: `nvidia-smi`

---

## Performance Benchmarks

**Expected training speed (8x A100 80GB):**

| Metric | Value |
|--------|-------|
| Structures/sec | ~2-3 |
| Steps/sec | ~0.3 |
| Epoch time (100 samples) | ~10 min |
| Time to 100 epochs | ~17 hours |
| GPU memory per GPU | ~60-70 GB |

**With gradient accumulation (effective batch=64):**
- Train time: ~20 hours for 100 epochs
- Convergence: ~50-100 epochs