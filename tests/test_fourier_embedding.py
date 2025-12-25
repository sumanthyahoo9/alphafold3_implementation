"""
Unit tests for AlphaFold3 Fourier Embedding.

File: tests/test_fourier_embedding.py

Tests cover:
1. Initialization
2. Forward pass shape validation
3. Fixed random weights (not learned)
4. Cosine encoding properties
5. Algorithm 22 faithfulness
6. Batch vs scalar inputs
"""

import pytest
import torch
from src.models.diffusion.fourier_embedding import (
    FourierEmbedding,
    create_dummy_timestep
)


class TestInitialization:
    """Test FourierEmbedding initialization"""
    
    def test_default_initialization(self):
        """Should initialize with default dimension"""
        embedding = FourierEmbedding()
        
        assert embedding.dim == 256
        assert embedding.w.shape == (256,)
        assert embedding.b.shape == (256,)
    
    def test_custom_dimension(self):
        """Should accept custom dimension"""
        embedding = FourierEmbedding(dim=128)
        
        assert embedding.dim == 128
        assert embedding.w.shape == (128,)
        assert embedding.b.shape == (128,)
    
    def test_weights_are_buffers(self):
        """Weights should be buffers (not parameters)"""
        embedding = FourierEmbedding()
        
        # Should NOT be in parameters (won't be updated by optimizer)
        param_names = [name for name, _ in embedding.named_parameters()]
        assert 'w' not in param_names
        assert 'b' not in param_names
        
        # Should be in buffers (saved with state_dict but not trained)
        buffer_names = [name for name, _ in embedding.named_buffers()]
        assert 'w' in buffer_names
        assert 'b' in buffer_names
    
    def test_weights_are_random(self):
        """Each instance should have different random weights"""
        emb1 = FourierEmbedding(dim=64)
        emb2 = FourierEmbedding(dim=64)
        
        # Different instances have different weights
        assert not torch.allclose(emb1.w, emb2.w)
        assert not torch.allclose(emb1.b, emb2.b)


class TestForwardPass:
    """Test forward pass functionality"""
    
    def test_scalar_timestep(self):
        """Should handle scalar timestep"""
        embedding = FourierEmbedding(dim=256)
        
        t = torch.tensor(0.5)
        
        out = embedding(t)
        
        assert out.shape == (256,)
    
    def test_batch_timesteps(self):
        """Should handle batch of timesteps"""
        embedding = FourierEmbedding(dim=256)
        
        t = torch.rand(16)  # Batch of 16
        
        out = embedding(t)
        
        assert out.shape == (16, 256)
    
    def test_different_batch_sizes(self):
        """Should work with different batch sizes"""
        embedding = FourierEmbedding(dim=128)
        
        for batch_size in [1, 4, 16, 64]:
            t = torch.rand(batch_size)
            out = embedding(t)
            assert out.shape == (batch_size, 128)
    
    def test_different_dimensions(self):
        """Should work with different embedding dimensions"""
        for dim in [64, 128, 256, 512]:
            embedding = FourierEmbedding(dim=dim)
            
            t = torch.tensor(0.5)
            out = embedding(t)
            
            assert out.shape == (dim,)


class TestFixedWeights:
    """Test that weights are fixed (not learned)"""
    
    def test_weights_dont_change(self):
        """Weights should not change during forward pass"""
        embedding = FourierEmbedding(dim=64)
        
        # Store original weights
        w_orig = embedding.w.clone()
        b_orig = embedding.b.clone()
        
        # Forward pass
        t = torch.rand(10, requires_grad=True)
        out = embedding(t)
        
        # Backward pass
        loss = out.sum()
        loss.backward()
        
        # Weights should be unchanged
        assert torch.allclose(embedding.w, w_orig)
        assert torch.allclose(embedding.b, b_orig)
    
    def test_no_gradients_for_weights(self):
        """Weights should have no gradients"""
        embedding = FourierEmbedding(dim=64)
        
        t = torch.rand(10, requires_grad=True)
        out = embedding(t)
        
        loss = out.sum()
        loss.backward()
        
        # Buffers don't have .grad attribute
        assert not hasattr(embedding.w, 'grad') or embedding.w.grad is None
        assert not hasattr(embedding.b, 'grad') or embedding.b.grad is None


class TestCosineEncoding:
    """Test cosine encoding properties"""
    
    def test_output_range(self):
        """Output should be in range [-1, 1] (cosine range)"""
        embedding = FourierEmbedding(dim=128)
        
        # Test multiple timesteps
        t = torch.rand(100) * 10  # Random timesteps in [0, 10]
        
        out = embedding(t)
        
        # Cosine output is in [-1, 1]
        assert out.min() >= -1.0
        assert out.max() <= 1.0
    
    def test_deterministic(self):
        """Same timestep should always produce same embedding"""
        embedding = FourierEmbedding(dim=64)
        
        t = torch.tensor(0.7)
        
        out1 = embedding(t)
        out2 = embedding(t)
        
        assert torch.allclose(out1, out2)
    
    def test_different_times_different_embeddings(self):
        """Different timesteps should produce different embeddings"""
        embedding = FourierEmbedding(dim=64)
        
        t1 = torch.tensor(0.3)
        t2 = torch.tensor(0.7)
        
        out1 = embedding(t1)
        out2 = embedding(t2)
        
        # Should be different
        assert not torch.allclose(out1, out2, atol=1e-5)


class TestAlgorithm22Faithfulness:
    """Test faithfulness to Algorithm 22"""
    
    def test_cosine_formula(self):
        """Should implement cos(2π(t·w + b))"""
        embedding = FourierEmbedding(dim=64)
        
        t = torch.tensor(0.5)
        
        # Compute expected output manually
        import math
        angle = 2 * math.pi * (t * embedding.w + embedding.b)
        expected = torch.cos(angle)
        
        # Compute actual output
        actual = embedding(t)
        
        # Should match
        assert torch.allclose(actual, expected, atol=1e-6)
    
    def test_paper_default_dimension(self):
        """Paper uses dim=256 in Algorithm 21"""
        embedding = FourierEmbedding()
        
        # Default should be 256
        assert embedding.dim == 256
    
    def test_algorithm_structure(self):
        """
        Verify implementation follows Algorithm 22:
        1: w, b ~ N(0, I_c)  # Random initialization
        2: return cos(2π(t·w + b))
        """
        embedding = FourierEmbedding(dim=128)
        
        t = torch.tensor(0.3)
        out = embedding(t)
        
        assert out.shape == (128,)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


class TestUsageInDiffusion:
    """Test usage patterns from Algorithm 21"""
    
    def test_log_scaled_timestep(self):
        """Algorithm 21 line 8: FourierEmbedding(0.25 * log(t/σ))"""
        embedding = FourierEmbedding(dim=256)
        
        # Simulate Algorithm 21 usage
        t_hat = torch.tensor(5.0)
        sigma_data = 16.0
        
        # Preprocess timestep as in Algorithm 21
        t_preprocessed = 0.25 * torch.log(t_hat / sigma_data)
        
        # Embed
        n = embedding(t_preprocessed)
        
        assert n.shape == (256,)
    
    def test_batch_diffusion_timesteps(self):
        """Should handle batch of diffusion timesteps"""
        embedding = FourierEmbedding(dim=256)
        
        # Simulate batch of structures at different diffusion times
        batch_size = 48  # Paper uses 48 parallel samples
        t_hat = torch.rand(batch_size) * 10  # Random timesteps
        sigma_data = 16.0
        
        # Preprocess
        t_preprocessed = 0.25 * torch.log(t_hat / sigma_data)
        
        # Embed
        n = embedding(t_preprocessed)
        
        assert n.shape == (48, 256)


class TestEdgeCases:
    """Test edge cases"""
    
    def test_zero_timestep(self):
        """Should handle t=0"""
        embedding = FourierEmbedding(dim=64)
        
        t = torch.tensor(0.0)
        
        out = embedding(t)
        
        assert out.shape == (64,)
        assert not torch.isnan(out).any()
    
    def test_large_timestep(self):
        """Should handle large timesteps"""
        embedding = FourierEmbedding(dim=64)
        
        t = torch.tensor(1000.0)
        
        out = embedding(t)
        
        assert out.shape == (64,)
        assert not torch.isnan(out).any()
    
    def test_negative_timestep(self):
        """Should handle negative timesteps (from log-scaling)"""
        embedding = FourierEmbedding(dim=64)
        
        t = torch.tensor(-2.0)
        
        out = embedding(t)
        
        assert out.shape == (64,)
        assert not torch.isnan(out).any()
    
    def test_single_dimension(self):
        """Should work with dim=1"""
        embedding = FourierEmbedding(dim=1)
        
        t = torch.tensor(0.5)
        out = embedding(t)
        
        assert out.shape == (1,)


class TestGradientFlow:
    """Test gradient flow"""
    
    def test_gradients_to_input(self):
        """Gradients should flow back to timestep input"""
        embedding = FourierEmbedding(dim=64)
        
        t = torch.tensor(0.5, requires_grad=True)
        
        out = embedding(t)
        loss = out.sum()
        loss.backward()
        
        # Timestep should have gradient
        assert t.grad is not None
        assert not torch.allclose(t.grad, torch.zeros_like(t.grad))


class TestStateDictSaving:
    """Test state dict saving/loading"""
    
    def test_save_load_state(self):
        """Saved weights should be restored correctly"""
        # Create and save
        emb1 = FourierEmbedding(dim=64)
        state = emb1.state_dict()
        
        # Create new instance and load
        emb2 = FourierEmbedding(dim=64)
        emb2.load_state_dict(state)
        
        # Weights should match
        assert torch.allclose(emb1.w, emb2.w)
        assert torch.allclose(emb1.b, emb2.b)
    
    def test_same_output_after_load(self):
        """Should produce same output after loading weights"""
        emb1 = FourierEmbedding(dim=64)
        
        t = torch.tensor(0.3)
        out1 = emb1(t)
        
        # Save and load
        state = emb1.state_dict()
        emb2 = FourierEmbedding(dim=64)
        emb2.load_state_dict(state)
        
        out2 = emb2(t)
        
        # Outputs should match
        assert torch.allclose(out1, out2)


class TestDummyTimestepGeneration:
    """Test dummy timestep utility"""
    
    def test_scalar_dummy(self):
        """Should generate scalar timestep"""
        t = create_dummy_timestep()
        
        assert t.dim() == 0 or (t.dim() == 1 and t.shape[0] == 1)
    
    def test_batch_dummy(self):
        """Should generate batch of timesteps"""
        t = create_dummy_timestep(batch_size=16)
        
        assert t.shape == (16,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])