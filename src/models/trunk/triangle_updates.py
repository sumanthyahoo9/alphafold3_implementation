"""
AlphaFold3 Triangle Multiplicative Updates

File: src/models/trunk/triangle_updates.py

Implements Algorithm 12: Triangular multiplicative update using "outgoing" edges
Implements Algorithm 13: Triangular multiplicative update using "incoming" edges

These layers enforce geometric consistency in the pair representation by
establishing communication between edges that connect 3 nodes. Interpreting
the pair representation as edge features of a fully connected graph (tokens
as nodes), these updates check triangle inequality constraints.

Key features:
- Enforces geometric consistency (triangle inequality)
- Two variants: outgoing edges vs incoming edges
- Gating mechanism for selective updates
- Critical for spatial reasoning

Mathematical intuition:
    If d(i,j)=5Å and d(j,k)=3Å, then 2Å ≤ d(i,k) ≤ 8Å
    Triangle updates propagate this constraint through the network.
"""

import torch
import torch.nn as nn


class TriangleMultiplicationOutgoing(nn.Module):
    """
    Triangular multiplicative update using "outgoing" edges.
    
    Implements Algorithm 12 from AF3 supplementary.
    
    Updates edge i→j by combining information from triangles formed by
    node i's outgoing edges (i→k) and the edges k→j.
    
    Geometric interpretation:
        For each intermediate node k, combines:
        - Edge from i to k (outgoing from i)
        - Edge from k to j
        
        This enforces: if i→k and k→j exist, then i→j should be consistent
        with the triangle inequality.
    
    Args:
        c_pair: Pair representation channels (default: 128)
        c: Intermediate projection dimension (default: 128)
    """
    
    def __init__(self, c_pair: int = 128, c: int = 128):
        super().__init__()
        
        self.c_pair = c_pair
        self.c = c
        
        # Layer normalization
        self.norm = nn.LayerNorm(c_pair)
        
        # Line 2: Project to intermediate dimension for left/right multiplications
        self.linear_a = nn.Linear(c_pair, c, bias=False)
        self.linear_b = nn.Linear(c_pair, c, bias=False)
        
        # Line 3: Gating projection
        self.linear_g = nn.Linear(c_pair, c_pair, bias=False)
        
        # Line 4: Combine intermediate results
        self.norm_intermediate = nn.LayerNorm(c)
        self.linear_out = nn.Linear(c, c_pair, bias=False)
    
    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """
        Apply outgoing edge triangle update.
        
        Args:
            pair: [N_token, N_token, c_pair] pair representation
                  Or [batch, N_token, N_token, c_pair] if batched
        
        Returns:
            pair_update: Same shape as input
        
        Algorithm 12:
        1: z_ij ← LayerNorm(z_ij)
        2: a_ij, b_ij = sigmoid(Linear(z_ij)) ⊙ Linear(z_ij)
        3: g_ij = sigmoid(Linear(z_ij))
        4: z̃_ij = g_ij ⊙ Linear(LayerNorm(Σ_k a_ik ⊙ b_jk))
        5: return {z̃_ij}
        """
        # Handle batching
        is_batched = pair.dim() == 4
        if not is_batched:
            pair = pair.unsqueeze(0)
        
        batch_size, n_token, _, _ = pair.shape
        
        # Line 1: Layer normalization
        pair_norm = self.norm(pair)  # [batch, N_token, N_token, c_pair]
        
        # Line 2: Project and gate
        # a represents left edges (i→k), b represents right edges (k→j)
        a_proj = self.linear_a(pair_norm)  # [batch, N_token, N_token, c]
        b_proj = self.linear_b(pair_norm)  # [batch, N_token, N_token, c]
        
        # Apply sigmoid gating
        a = torch.sigmoid(a_proj) * a_proj  # [batch, N_token, N_token, c]
        b = torch.sigmoid(b_proj) * b_proj  # [batch, N_token, N_token, c]
        
        # Line 3: Compute gate
        g = torch.sigmoid(self.linear_g(pair_norm))  # [batch, N_token, N_token, c_pair]
        
        # Line 4: Triangle multiplication - "outgoing"
        # Sum over intermediate nodes k: a_ik ⊙ b_jk
        # a: [batch, i, k, c]
        # b: [batch, j, k, c]  (need to swap indices)
        # Result: [batch, i, j, c]
        
        # Rearrange b for the multiplication: b_jk → b_kj → need b at position j
        # a[i, k, :] ⊙ b[j, k, :]  summed over k
        # Use einsum for clarity: "bikc,bjkc->bijc"
        intermediate = torch.einsum('bikc,bjkc->bijc', a, b)  # [batch, N_token, N_token, c]
        
        # Normalize and project
        intermediate = self.norm_intermediate(intermediate)
        intermediate = self.linear_out(intermediate)  # [batch, N_token, N_token, c_pair]
        
        # Apply gating
        pair_update = g * intermediate  # [batch, N_token, N_token, c_pair]
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            pair_update = pair_update.squeeze(0)
        
        return pair_update


class TriangleMultiplicationIncoming(nn.Module):
    """
    Triangular multiplicative update using "incoming" edges.
    
    Implements Algorithm 13 from AF3 supplementary.
    
    Updates edge i→j by combining information from triangles formed by
    node j's incoming edges (k→j) and the edges i→k.
    
    Geometric interpretation:
        For each intermediate node k, combines:
        - Edge from k to i (incoming to i)
        - Edge from k to j (incoming to j)
        
        This enforces: if k→i and k→j exist, then i→j should be consistent
        with the triangle inequality.
    
    The only difference from Algorithm 12 is in line 4:
        Outgoing: Σ_k a_ik ⊙ b_jk
        Incoming: Σ_k a_ki ⊙ b_kj
    
    Args:
        c_pair: Pair representation channels (default: 128)
        c: Intermediate projection dimension (default: 128)
    """
    
    def __init__(self, c_pair: int = 128, c: int = 128):
        super().__init__()
        
        self.c_pair = c_pair
        self.c = c
        
        # Layer normalization
        self.norm = nn.LayerNorm(c_pair)
        
        # Line 2: Project to intermediate dimension
        self.linear_a = nn.Linear(c_pair, c, bias=False)
        self.linear_b = nn.Linear(c_pair, c, bias=False)
        
        # Line 3: Gating projection
        self.linear_g = nn.Linear(c_pair, c_pair, bias=False)
        
        # Line 4: Combine intermediate results
        self.norm_intermediate = nn.LayerNorm(c)
        self.linear_out = nn.Linear(c, c_pair, bias=False)
    
    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """
        Apply incoming edge triangle update.
        
        Args:
            pair: [N_token, N_token, c_pair] pair representation
                  Or [batch, N_token, N_token, c_pair] if batched
        
        Returns:
            pair_update: Same shape as input
        
        Algorithm 13:
        1: z_ij ← LayerNorm(z_ij)
        2: a_ij, b_ij = sigmoid(Linear(z_ij)) ⊙ Linear(z_ij)
        3: g_ij = sigmoid(Linear(z_ij))
        4: z̃_ij = g_ij ⊙ Linear(LayerNorm(Σ_k a_ki ⊙ b_kj))  ← different from Alg 12
        5: return {z̃_ij}
        """
        # Handle batching
        is_batched = pair.dim() == 4
        if not is_batched:
            pair = pair.unsqueeze(0)
        
        batch_size, n_token, _, _ = pair.shape
        
        # Line 1: Layer normalization
        pair_norm = self.norm(pair)  # [batch, N_token, N_token, c_pair]
        
        # Line 2: Project and gate
        # a represents incoming edges to i (k→i), b represents incoming edges to j (k→j)
        a_proj = self.linear_a(pair_norm)  # [batch, N_token, N_token, c]
        b_proj = self.linear_b(pair_norm)  # [batch, N_token, N_token, c]
        
        # Apply sigmoid gating
        a = torch.sigmoid(a_proj) * a_proj  # [batch, N_token, N_token, c]
        b = torch.sigmoid(b_proj) * b_proj  # [batch, N_token, N_token, c]
        
        # Line 3: Compute gate
        g = torch.sigmoid(self.linear_g(pair_norm))  # [batch, N_token, N_token, c_pair]
        
        # Line 4: Triangle multiplication - "incoming"
        # Sum over intermediate nodes k: a_ki ⊙ b_kj
        # a: [batch, k, i, c] (need to swap)
        # b: [batch, k, j, c] (need to swap)
        # Result: [batch, i, j, c]
        
        # Use einsum: "bkic,bkjc->bijc"
        intermediate = torch.einsum('bkic,bkjc->bijc', a, b)  # [batch, N_token, N_token, c]
        
        # Normalize and project
        intermediate = self.norm_intermediate(intermediate)
        intermediate = self.linear_out(intermediate)  # [batch, N_token, N_token, c_pair]
        
        # Apply gating
        pair_update = g * intermediate  # [batch, N_token, N_token, c_pair]
        
        # Remove batch dimension if input was unbatched
        if not is_batched:
            pair_update = pair_update.squeeze(0)
        
        return pair_update


def create_dummy_pair_input(
    n_token: int = 10,
    c_pair: int = 128,
    batch_size: int = 0 # Change to None to pass the unittest
) -> torch.Tensor:
    """
    Create dummy pair representation for testing triangle updates.
    
    Args:
        n_token: Number of tokens
        c_pair: Pair representation channels
        batch_size: Optional batch size (if None, returns unbatched)
        
    Returns:
        pair: [N_token, N_token, c_pair] or [batch, N_token, N_token, c_pair]
    """
    if batch_size is None:
        return torch.randn(n_token, n_token, c_pair)
    return torch.randn(batch_size, n_token, n_token, c_pair)