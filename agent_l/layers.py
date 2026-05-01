"""
Foundational layers for Agent L.

All components have published provenance:
- RMSNorm: Zhang & Sennrich, 2019
- RoPE: Su et al., 2021 (LLaMA-style implementation)
- Loop-index embedding: Saunshi et al., ICLR 2025

References:
[1] Zhang & Sennrich "Root Mean Square Layer Normalization" NeurIPS 2019
[2] Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" 2021
[3] Saunshi et al. "Reasoning with Latent Thoughts" ICLR 2025
"""

import torch
import torch.nn as nn
from typing import Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization [1].
    
    Normalizes by RMS rather than mean+variance (LayerNorm).
    No bias term, only learned per-channel rescaling weight.
    More stable and efficient than LayerNorm for transformers.
    
    References:
        [1] Zhang & Sennrich, NeurIPS 2019
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: Feature dimension to normalize over
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., dim)
        Returns:
            RMS-normalized tensor, same shape
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        # Normalize and rescale
        return x * rms * self.weight


# ============================================================
# RoPE: Rotary Position Embeddings
# ============================================================

def precompute_rope_freqs(
    dim: int, 
    max_len: int, 
    theta: float = 500000.0
) -> torch.Tensor:
    """
    Precompute complex-valued RoPE rotation matrices [2].
    
    Each position m gets a complex phasor e^{i·m·θ_k} for each frequency pair k.
    Stored as complex tensor so rotation is a single pointwise multiply.
    
    Args:
        dim: Head dimension (must be even)
        max_len: Maximum sequence length to precompute
        theta: RoPE base frequency (LLaMA-3 uses 500K for long context)
    
    Returns:
        Complex tensor of shape (max_len, dim//2)
    
    References:
        [2] Su et al., 2021 - RoFormer
    """
    # Frequency bands: θ_k = θ^{-2k/d} for k = 0, 1, ..., d/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # Position indices
    t = torch.arange(max_len, dtype=torch.float32)
    # Outer product: angles[m, k] = m * θ_k
    freqs = torch.outer(t, freqs)
    # Convert to complex: e^{i·angle}
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(
    x: torch.Tensor, 
    freqs_cis: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to Q or K tensors [2].
    
    Interprets each pair of adjacent features as a 2D complex number,
    multiplies by precomputed phasor for that position, rotating in complex plane.
    Rotation preserves norm while encoding position information.
    
    Args:
        x: Tensor of shape (B, T, H, head_dim); head_dim must be even
        freqs_cis: Precomputed complex frequencies, shape (T, head_dim//2)
            Already sliced to correct positions (caller handles start_pos offset)
    
    Returns:
        Rotated tensor, same shape and dtype as x
    
    References:
        [2] Su et al., 2021 - RoFormer
    """
    # View as complex: (B, T, H, head_dim//2, 2) → (B, T, H, head_dim//2) complex
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Rotate: multiply by complex phasor
    # freqs_cis: (T, head_dim//2) → broadcast to (1, T, 1, head_dim//2)
    rotated = xc * freqs_cis.unsqueeze(0).unsqueeze(2)
    # Convert back to real: (B, T, H, head_dim//2, 2) → (B, T, H, head_dim)
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)


# ============================================================
# Loop-Index Embedding (Saunshi et al., ICLR 2025)
# ============================================================

def loop_index_embedding(
    h: torch.Tensor, 
    loop_t: int, 
    loop_dim: int, 
    theta: float = 10000.0
) -> torch.Tensor:
    """
    Inject sinusoidal loop-index signal into hidden states [3].
    
    Analogous to RoPE for sequence position, but over recurrence depth.
    Without this, shared recurrent weights must handle both early-stage 
    pattern-matching and late-stage refinement with no signal distinguishing
    which loop iteration. Adding loop index lets same parameters implement
    functionally distinct operations per iteration.
    
    Args:
        h: Hidden state tensor of shape (B, T, dim)
        loop_t: Current loop iteration index (0-based)
        loop_dim: Number of leading channels to receive embedding (must be even)
        theta: Sinusoidal base frequency
    
    Returns:
        h with sinusoidal bias added to first loop_dim channels; same shape
    
    References:
        [3] Saunshi et al., ICLR 2025 - "Reasoning with Latent Thoughts"
           Proves loop-index embedding improves depth extrapolation.
    """
    # Frequency bands
    freqs = 1.0 / (
        theta ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    # Angles for this loop iteration
    angles = loop_t * freqs  # (loop_dim//2,)
    # Sinusoidal embedding: [sin, cos] concatenated
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    # Pad to full dimension
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    # Add to hidden state
    return h + emb_full.unsqueeze(0).unsqueeze(0)


# ============================================================
# Causal Mask
# ============================================================

def causal_mask(
    seq_len: int, 
    device: torch.device, 
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Build additive causal mask: 0 on/below diagonal, -inf above.
    
    Args:
        seq_len: Sequence length
        device: Target device
        dtype: Tensor dtype (must match activation dtype for correct matmul)
    
    Returns:
        Tensor of shape (1, 1, seq_len, seq_len), broadcastable over (B, H, T, S)
    """
    mask = torch.full(
        (1, 1, seq_len, seq_len), 
        float("-inf"), 
        device=device, 
        dtype=dtype
    )
    return torch.triu(mask, diagonal=1)
