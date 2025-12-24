"""
AlphaFold3 Triangle Attention

File: src/models/trunk/triangle_attention.py

Implements Algorithm 14: Triangular gated self-attention around starting node
Implements Algorithm 15: Triangular gated self-attention around ending node

These layers apply multi-head attention along different axes of the pair
representation, treating it as a graph where tokens are nodes and pair
features are edge features.

Key features:
- Multi-head attention along pair representation axes
- Gating mechanism for selective updates
- Pair bias added to attention logits
- Two perspectives: starting node vs ending node

Mathematical intuition:
    StartingNode: For edge i→j, attend over edges i→k (all edges from i)
    EndingNode: For edge i→j, attend over edges k→j (all edges to j)
    Both enforce geometric consistency through attention mechanism
"""

import torch
import torch.nn as nn
import math


class TriangleAttentionStartingNode(nn.Module):
    """
    Triangular gated self-attention around starting node.
    
    Implements Algorithm 14 from AF3 supplementary.
    
    For each edge i→j in the pair representation, attends over all edges
    that start from node i (i→k for all k). This allows information to
    flow between edges that share a starting node.
    
    Geometric interpretation:
        Updating edge i→j by attending over edges i→k
        Enforces: "If I know about edges from i to various k, 
                   what should i→j look like?"
    
    Args:
        c_pair: Pair representation channels (default: 128)
        c: Attention dimension per head (default: 32)
        n_heads: Number of attention heads (default: 4)
    """
    
    def __init__(
        self,
        c_pair: int = 128,
        c: int = 32,
        n_heads: int = 4
    ):
        super().__init__()
        
        self.c_pair = c_pair
        self.c = c
        self.n_heads = n_heads
        
        # Layer normalization
        self.norm = nn.LayerNorm(c_pair)
        
        # Line 2: QKV projections (one for each head)
        self.linear_q = nn.Linear(c_pair, c * n_heads, bias=False)
        self.linear_k = nn.Linear(c_pair, c * n_heads, bias=False)
        self.linear_v = nn.Linear(c_pair, c * n_heads, bias=False)
        
        # Line 3: Bias projection
        self.linear_b = nn.Linear(c_pair, n_heads, bias=False)
        
        # Line 4: Gating projection
        self.linear_g = nn.Linear(c_pair, c * n_heads, bias=False)
        
        # Line 7: Output projection
        self.linear_out = nn.Linear(c * n_heads, c_pair, bias=False)
    
    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """
        Apply starting node triangle attention.
        
        Args:
            pair: [N_token, N_token, c_pair] pair representation
                  Or [batch, N_token, N_token, c_pair] if batched
        
        Returns:
            pair_update: Same shape as input
        
        Algorithm 14:
        1: z_ij ← LayerNorm(z_ij)
        2: q^h_ij, k^h_ij, v^h_ij = LinearNoBias(z_ij)
        3: b^h_ij = LinearNoBias(z_ij)
        4: g^h_ij = sigmoid(LinearNoBias(z_ij))
        5: a^h_ijk = softmax_k(1/√c * q^h_ij^T k^h_ik + b^h_jk)
        6: o^h_ij = g^h_ij ⊙ Σ_k a^h_ijk v^h_ik
        7: z̃_ij = LinearNoBias(concat_h(o^h_ij))
        8: return {z̃_ij}
        """
        # Handle batching
        is_batched = pair.dim() == 4
        if not is_batched:
            pair = pair.unsqueeze(0)
        
        batch_size, n_token, _, _ = pair.shape
        
        # Line 1: Layer normalization
        pair_norm = self.norm(pair)  # [batch, N_token, N_token, c_pair]
        
        # Line 2: Project to Q, K, V
        q = self.linear_q(pair_norm)  # [batch, N_token, N_token, c*n_heads]
        k = self.linear_k(pair_norm)  # [batch, N_token, N_token, c*n_heads]
        v = self.linear_v(pair_norm)  # [batch, N_token, N_token, c*n_heads]
        
        # Reshape to separate heads: [batch, N_token, N_token, n_heads, c]
        q = q.view(batch_size, n_token, n_token, self.n_heads, self.c)
        k = k.view(batch_size, n_token, n_token, self.n_heads, self.c)
        v = v.view(batch_size, n_token, n_token, self.n_heads, self.c)
        
        # Line 3: Bias projection
        b = self.linear_b(pair_norm)  # [batch, N_token, N_token, n_heads]
        
        # Line 4: Gating projection
        g = torch.sigmoid(self.linear_g(pair_norm))  # [batch, N_token, N_token, c*n_heads]
        g = g.view(batch_size, n_token, n_token, self.n_heads, self.c)
        
        # Line 5: Compute attention
        # For edge (i,j), attend over edges (i,k) - same starting node i
        # q_ij: [batch, i, j, heads, c]
        # k_ik: [batch, i, k, heads, c]
        # We want: q[i,j] @ k[i,k] for all k
        
        # Rearrange for attention computation
        # q: [batch, i, j, heads, c] 
        # k: [batch, i, k, heads, c]
        # attention scores: [batch, i, j, k, heads]
        
        # Compute Q^T K using einsum: q[i,j,h,:] @ k[i,k,h,:].T
        # Result shape: [batch, i, j, k, heads]
        attn_logits = torch.einsum('bijhc,bikhc->bijkh', q, k)  
        attn_logits = attn_logits / math.sqrt(self.c)  # Scale
        
        # Add bias: b_jk
        # b: [batch, j, k, heads] → need to broadcast to [batch, i, j, k, heads]
        # Extract b_jk and broadcast over i dimension
        b_expanded = b.unsqueeze(1)  # [batch, 1, j, k, heads]
        attn_logits = attn_logits + b_expanded  # [batch, i, j, k, heads]
        
        # Softmax over k dimension (dimension 3)
        attn_weights = torch.softmax(attn_logits, dim=3)  # [batch, i, j, k, heads]
        
        # Line 6: Apply attention to values
        # v_ik: [batch, i, k, heads, c]
        # attn_weights: [batch, i, j, k, heads]
        # Result: [batch, i, j, heads, c]
        o = torch.einsum('bijkh,bikhc->bijhc', attn_weights, v)
        
        # Apply gating
        o = g * o  # [batch, N_token, N_token, n_heads, c]
        
        # Line 7: Concatenate heads and project to output
        o = o.reshape(batch_size, n_token, n_token, self.n_heads * self.c)
        pair_update = self.linear_out(o)  # [batch, N_token, N_token, c_pair]
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            pair_update = pair_update.squeeze(0)
        
        return pair_update


class TriangleAttentionEndingNode(nn.Module):
    """
    Triangular gated self-attention around ending node.
    
    Implements Algorithm 15 from AF3 supplementary.
    
    For each edge i→j in the pair representation, attends over all edges
    that end at node j (k→j for all k). This allows information to flow
    between edges that share an ending node.
    
    Geometric interpretation:
        Updating edge i→j by attending over edges k→j
        Enforces: "If I know about edges to j from various k,
                   what should i→j look like?"
    
    The only differences from Algorithm 14 are in lines 5-6:
        Line 5: attention uses k_kj instead of k_ik, bias is b_ki instead of b_jk
        Line 6: values use v_kj instead of v_ik
    
    Args:
        c_pair: Pair representation channels (default: 128)
        c: Attention dimension per head (default: 32)
        n_heads: Number of attention heads (default: 4)
    """
    
    def __init__(
        self,
        c_pair: int = 128,
        c: int = 32,
        n_heads: int = 4
    ):
        super().__init__()
        
        self.c_pair = c_pair
        self.c = c
        self.n_heads = n_heads
        
        # Layer normalization
        self.norm = nn.LayerNorm(c_pair)
        
        # Line 2: QKV projections
        self.linear_q = nn.Linear(c_pair, c * n_heads, bias=False)
        self.linear_k = nn.Linear(c_pair, c * n_heads, bias=False)
        self.linear_v = nn.Linear(c_pair, c * n_heads, bias=False)
        
        # Line 3: Bias projection
        self.linear_b = nn.Linear(c_pair, n_heads, bias=False)
        
        # Line 4: Gating projection
        self.linear_g = nn.Linear(c_pair, c * n_heads, bias=False)
        
        # Line 7: Output projection
        self.linear_out = nn.Linear(c * n_heads, c_pair, bias=False)
    
    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """
        Apply ending node triangle attention.
        
        Args:
            pair: [N_token, N_token, c_pair] pair representation
                  Or [batch, N_token, N_token, c_pair] if batched
        
        Returns:
            pair_update: Same shape as input
        
        Algorithm 15 (differences from 14 highlighted):
        1: z_ij ← LayerNorm(z_ij)
        2: q^h_ij, k^h_ij, v^h_ij = LinearNoBias(z_ij)
        3: b^h_ij = LinearNoBias(z_ij)
        4: g^h_ij = sigmoid(LinearNoBias(z_ij))
        5: a^h_ijk = softmax_k(1/√c * q^h_ij^T k^h_kj + b^h_ki)  ← different!
        6: o^h_ij = g^h_ij ⊙ Σ_k a^h_ijk v^h_kj  ← different!
        7: z̃_ij = LinearNoBias(concat_h(o^h_ij))
        8: return {z̃_ij}
        """
        # Handle batching
        is_batched = pair.dim() == 4
        if not is_batched:
            pair = pair.unsqueeze(0)
        
        batch_size, n_token, _, _ = pair.shape
        
        # Line 1: Layer normalization
        pair_norm = self.norm(pair)  # [batch, N_token, N_token, c_pair]
        
        # Line 2: Project to Q, K, V
        q = self.linear_q(pair_norm)  # [batch, N_token, N_token, c*n_heads]
        k = self.linear_k(pair_norm)  # [batch, N_token, N_token, c*n_heads]
        v = self.linear_v(pair_norm)  # [batch, N_token, N_token, c*n_heads]
        
        # Reshape to separate heads
        q = q.view(batch_size, n_token, n_token, self.n_heads, self.c)
        k = k.view(batch_size, n_token, n_token, self.n_heads, self.c)
        v = v.view(batch_size, n_token, n_token, self.n_heads, self.c)
        
        # Line 3: Bias projection
        b = self.linear_b(pair_norm)  # [batch, N_token, N_token, n_heads]
        
        # Line 4: Gating projection
        g = torch.sigmoid(self.linear_g(pair_norm))  # [batch, N_token, N_token, c*n_heads]
        g = g.view(batch_size, n_token, n_token, self.n_heads, self.c)
        
        # Line 5: Compute attention - ENDING NODE version
        # For edge (i,j), attend over edges (k,j) - same ending node j
        # q_ij: [batch, i, j, heads, c]
        # k_kj: [batch, k, j, heads, c]
        # We want: q[i,j] @ k[k,j] for all k
        
        # Compute Q^T K using einsum: q[i,j,h,:] @ k[k,j,h,:].T
        # Result shape: [batch, i, j, k, heads]
        attn_logits = torch.einsum('bijhc,bkjhc->bijkh', q, k)
        attn_logits = attn_logits / math.sqrt(self.c)  # Scale
        
        # Add bias: b_ki  (different from starting node!)
        # b: [batch, k, i, heads] → need to broadcast to [batch, i, j, k, heads]
        # Extract b_ki and broadcast over j dimension
        b_expanded = b.permute(0, 2, 1, 3).unsqueeze(2)  # [batch, i, 1, k, heads]
        attn_logits = attn_logits + b_expanded  # [batch, i, j, k, heads]
        
        # Softmax over k dimension
        attn_weights = torch.softmax(attn_logits, dim=3)  # [batch, i, j, k, heads]
        
        # Line 6: Apply attention to values - ENDING NODE version
        # v_kj: [batch, k, j, heads, c]
        # attn_weights: [batch, i, j, k, heads]
        # Result: [batch, i, j, heads, c]
        o = torch.einsum('bijkh,bkjhc->bijhc', attn_weights, v)
        
        # Apply gating
        o = g * o  # [batch, N_token, N_token, n_heads, c]
        
        # Line 7: Concatenate heads and project to output
        o = o.reshape(batch_size, n_token, n_token, self.n_heads * self.c)
        pair_update = self.linear_out(o)  # [batch, N_token, N_token, c_pair]
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            pair_update = pair_update.squeeze(0)
        
        return pair_update