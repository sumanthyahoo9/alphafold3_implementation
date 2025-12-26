#!/usr/bin/env python3
"""
AlphaFold3 Distributed Training Script

Supports:
- Multi-GPU training with DistributedDataParallel (DDP)
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Checkpointing and resuming
- Weights & Biases logging
- Gradient clipping
- Learning rate scheduling

Usage:
    # Single GPU
    python train.py --config config.yaml
    
    # Multi-GPU (DDP)
    torchrun --nproc_per_node=8 train.py --config config.yaml
    
    # Multi-node
    torchrun --nproc_per_node=8 --nnodes=4 --node_rank=0 \
             --master_addr="192.168.1.1" --master_port=29500 \
             train.py --config config.yaml
"""

import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import yaml
import os
import time
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime

# Imports for model and data
from src.models.alphafold3 import AlphaFold3
from src.models.losses import (
    DiffusionLoss,
    SmoothLDDTLoss,
    WeightedMSELoss,
    BondLengthLoss
)

from src.data.dataset import AlphaFold3Dataset  # To be implemented
from src.data.tokenizer import AlphaFold3Tokenizer
from src.data.featurizer import AlphaFold3Featurizer


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set device
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class Trainer:
    """
    AlphaFold3 Distributed Trainer.
    
    Handles:
    - Multi-GPU training
    - Mixed precision
    - Gradient accumulation
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        config: Dict,
        rank: int,
        world_size: int,
        local_rank: int
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f'cuda:{local_rank}')
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Setup
        self._setup_directories()
        self._setup_model()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_functions()
        self._setup_dataloaders()
        self._setup_logging()
        
        # Mixed precision
        self.use_amp = config['training'].get('mixed_precision', True)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Gradient accumulation
        self.grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)
        
    def _setup_directories(self):
        """Create output directories."""
        self.output_dir = Path(self.config['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        
        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.log_dir.mkdir(exist_ok=True)
    
    def _setup_model(self):
        """Initialize AlphaFold3 model."""
        model_config = self.config['model']
        
        print(f"[Rank {self.rank}] Initializing AlphaFold3 model...")
        
        self.model = AlphaFold3(
            c_token=model_config.get('c_token', 384),
            c_pair=model_config.get('c_pair', 128),
            c_atom=model_config.get('c_atom', 128),
            c_atompair=model_config.get('c_atompair', 16),
            n_cycles=model_config.get('n_cycles', 4),
            # Add more config as needed
        ).to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if self.rank == 0:
            print(f"Total parameters: {n_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Model size: {n_params * 4 / 1e9:.2f} GB (FP32)")
        
        # Wrap with DDP
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,  # Set True if needed
                gradient_as_bucket_view=True   # Memory optimization
            )
            
            if self.rank == 0:
                print(f"Model wrapped with DDP across {self.world_size} GPUs")
    
    def _setup_optimizer(self):
        """Initialize optimizer."""
        opt_config = self.config['optimizer']
        
        # Get base model (unwrap DDP if needed)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        
        # Separate parameters for different learning rates (optional)
        # For now: single learning rate for all params
        params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer_type = opt_config.get('type', 'adam').lower()
        
        if optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                params,
                lr=opt_config.get('learning_rate', 1e-4),
                betas=opt_config.get('betas', (0.9, 0.999)),
                eps=opt_config.get('eps', 1e-8),
                weight_decay=opt_config.get('weight_decay', 0.0)
            )
        elif optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt_config.get('learning_rate', 1e-4),
                betas=opt_config.get('betas', (0.9, 0.999)),
                eps=opt_config.get('eps', 1e-8),
                weight_decay=opt_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        if self.rank == 0:
            print(f"Optimizer: {optimizer_type}")
            print(f"Learning rate: {opt_config.get('learning_rate', 1e-4)}")
    
    def _setup_scheduler(self):
        """Initialize learning rate scheduler."""
        sched_config = self.config.get('scheduler', {})
        sched_type = sched_config.get('type', 'cosine').lower()
        
        if sched_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('T_max', 100000),
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_type == 'warmup_cosine':
            # Custom warmup + cosine
            # TODO: Implement warmup scheduler
            self.scheduler = None
        else:
            self.scheduler = None
        
        if self.rank == 0:
            print(f"Scheduler: {sched_type if self.scheduler else 'None'}")
    
    def _setup_loss_functions(self):
        """Initialize loss functions."""
        loss_config = self.config.get('losses', {})
        
        # Diffusion loss (combines MSE + bond + LDDT)
        self.diffusion_loss = DiffusionLoss(
            alpha_bond=loss_config.get('alpha_bond', 0.0),  # 0 for training, 1 for fine-tuning
            sigma_data=loss_config.get('sigma_data', 16.0),
            alpha_dna=loss_config.get('alpha_dna', 5.0),
            alpha_rna=loss_config.get('alpha_rna', 5.0),
            alpha_ligand=loss_config.get('alpha_ligand', 10.0)
        )
        
        # LDDT loss (computed separately)
        self.lddt_loss = SmoothLDDTLoss(
            thresholds=loss_config.get('lddt_thresholds', [0.5, 1.0, 2.0, 4.0])
        )
        
        if self.rank == 0:
            print("Loss functions initialized:")
            print(f"  - Diffusion loss (α_bond={loss_config.get('alpha_bond', 0.0)})")
            print(f"  - Smooth LDDT loss")
    
    def _setup_dataloaders(self):
        """Initialize dataloaders."""
        data_config = self.config['data']
        
        # Create datasets
        # TODO: Implement AlphaFold3Dataset
        # For now: placeholder
        print(f"[Rank {self.rank}] Creating datasets...")
        
        # Training dataset
        train_dataset = AlphaFold3Dataset(
            data_dir=data_config['train_dir'],
            tokenizer=AlphaFold3Tokenizer(),
            featurizer=AlphaFold3Featurizer(),
            max_tokens=data_config.get('max_tokens', 512)
        )
        
        # Validation dataset
        val_dataset = AlphaFold3Dataset(
            data_dir=data_config['val_dir'],
            tokenizer=AlphaFold3Tokenizer(),
            featurizer=AlphaFold3Featurizer(),
            max_tokens=data_config.get('max_tokens', 512)
        )
        
        # Distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        ) if self.world_size > 1 else None
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False
        ) if self.world_size > 1 else None
        
        # Dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=data_config.get('batch_size', 1),
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=data_config.get('val_batch_size', 1),
            sampler=val_sampler,
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            drop_last=False
        )
        
        if self.rank == 0:
            print(f"Training samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            print(f"Batch size per GPU: {data_config.get('batch_size', 1)}")
            print(f"Effective batch size: {data_config.get('batch_size', 1) * self.world_size * self.grad_accum_steps}")
    
    def _setup_logging(self):
        """Initialize logging (Weights & Biases, TensorBoard, etc.)."""
        if self.rank == 0:
            log_config = self.config.get('logging', {})
            
            # Optional: W&B integration
            if log_config.get('use_wandb', False):
                try:
                    import wandb
                    wandb.init(
                        project=log_config.get('wandb_project', 'alphafold3'),
                        name=log_config.get('wandb_run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                        config=self.config
                    )
                    self.use_wandb = True
                    print("W&B logging enabled")
                except ImportError:
                    print("W&B not available, skipping")
                    self.use_wandb = False
            else:
                self.use_wandb = False
    
    def train_epoch(self) -> Dict:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'mse': 0.0,
            'lddt': 0.0,
            'bond': 0.0
        }
        
        # Set epoch for distributed sampler
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(self.epoch)
        
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                # Run model
                predictions = self.model(batch)
                
                # Compute LDDT loss
                lddt_loss = self.lddt_loss(
                    predictions['x_pred'],
                    batch['x_gt'],
                    is_dna=batch.get('is_dna'),
                    is_rna=batch.get('is_rna')
                )
                
                # Compute diffusion loss
                loss = self.diffusion_loss(
                    x_pred=predictions['x_pred'],
                    x_gt=batch['x_gt'],
                    noise_level=batch['noise_level'],
                    smooth_lddt_loss=lddt_loss,
                    is_dna=batch.get('is_dna'),
                    is_rna=batch.get('is_rna'),
                    is_ligand=batch.get('is_ligand'),
                    bonds=batch.get('bonds'),
                    mask=batch.get('mask')
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Update weights every grad_accum_steps
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Accumulate losses
            epoch_losses['total'] += loss.item() * self.grad_accum_steps
            epoch_losses['lddt'] += lddt_loss.item()
            
            # Logging
            if self.rank == 0 and self.global_step % self.config['logging'].get('log_interval', 100) == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {self.epoch} | Step {self.global_step} | "
                      f"Loss: {loss.item() * self.grad_accum_steps:.4f} | "
                      f"LDDT: {lddt_loss.item():.4f} | "
                      f"LR: {lr:.2e}")
                
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'train/loss': loss.item() * self.grad_accum_steps,
                        'train/lddt': lddt_loss.item(),
                        'train/lr': lr,
                        'epoch': self.epoch,
                        'step': self.global_step
                    })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict:
        """Run validation."""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'lddt': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        for batch in self.val_loader:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(batch)
            
            # Compute losses
            lddt_loss = self.lddt_loss(
                predictions['x_pred'],
                batch['x_gt'],
                is_dna=batch.get('is_dna'),
                is_rna=batch.get('is_rna')
            )
            
            loss = self.diffusion_loss(
                x_pred=predictions['x_pred'],
                x_gt=batch['x_gt'],
                noise_level=batch['noise_level'],
                smooth_lddt_loss=lddt_loss,
                is_dna=batch.get('is_dna'),
                is_rna=batch.get('is_rna'),
                is_ligand=batch.get('is_ligand'),
                bonds=batch.get('bonds'),
                mask=batch.get('mask')
            )
            
            val_losses['total'] += loss.item()
            val_losses['lddt'] += lddt_loss.item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Reduce across GPUs
        if self.world_size > 1:
            for key in val_losses:
                tensor = torch.tensor(val_losses[key], device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                val_losses[key] = tensor.item()
        
        return val_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return
        
        # Get model state (unwrap DDP)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")
        
        # Save periodic checkpoints
        if self.epoch % self.config['training'].get('save_every', 10) == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        print(f"[Rank {self.rank}] Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model (unwrap DDP)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']
        
        if self.rank == 0:
            print("\n" + "="*70)
            print("Starting Training")
            print("="*70)
            print(f"Total epochs: {num_epochs}")
            print(f"Starting from epoch: {self.epoch}")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            if self.rank == 0:
                print(f"\nEpoch {epoch}/{num_epochs}")
                print("-" * 70)
            
            # Train
            start_time = time.time()
            train_losses = self.train_epoch()
            train_time = time.time() - start_time
            
            # Validate
            start_time = time.time()
            val_losses = self.validate()
            val_time = time.time() - start_time
            
            # Logging
            if self.rank == 0:
                print(f"Train Loss: {train_losses['total']:.4f} | "
                      f"Val Loss: {val_losses['total']:.4f} | "
                      f"Train Time: {train_time:.1f}s | "
                      f"Val Time: {val_time:.1f}s")
                
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'val/loss': val_losses['total'],
                        'val/lddt': val_losses['lddt'],
                        'epoch': epoch
                    })
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            self.save_checkpoint(is_best=is_best)
        
        if self.rank == 0:
            print("\n" + "="*70)
            print("Training Complete!")
            print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AlphaFold3 Distributed Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    # Load config
    config = load_config(args.config)
    
    # Create trainer
    trainer = Trainer(config, rank, world_size, local_rank)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    try:
        # Train
        trainer.train()
    finally:
        # Cleanup
        cleanup_distributed()


if __name__ == '__main__':
    main()