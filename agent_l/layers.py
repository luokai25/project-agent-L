"""
Foundational layers for Agent L.

This module implements:
- RMSNorm: Root Mean Square Layer Normalization
- RoPE: Rotary Positional Embeddings
- Loop Index Embedding: Sinusoidal signal for differentiating loop iterations
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes by the RMS of the input rather than mean+variance, with a
    learned per-channel rescaling weight. No bias term. More efficient than
    LayerNorm and commonly used in modern LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., dim)
        Returns:
            RMS-normalized tensor of the same shape, rescaled by weight
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# -----------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# -----------------------------------------------------------------------------


def precompute_rope_freqs(dim: int, max_len: int, theta: float = 500000.0) -> torch.Tensor:
    """
    Precompute complex-valued RoPE rotation matrices for positions 0..max_len-1.

    Each position gets a complex phasor e^{i·m·θ_k} for each frequency pair k.
    Stored as a complex tensor so that rotation is a single pointwise multiply.

    Args:
        dim: Head dimension (must be even); frequencies computed for dim//2 pairs
        max_len: Maximum sequence length to precompute
        theta: RoPE base frequency (higher = slower frequency decay)

    Returns:
        Complex tensor of shape (max_len, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to query or key tensors.

    Interprets each pair of adjacent features as a 2D complex number and
    multiplies by the precomputed phasor for that position, rotating the
    representation in the complex plane without changing its norm.

    Args:
        x: Tensor of shape (B, T, H, head_dim); head_dim must be even
        freqs_cis: Precomputed complex frequencies of shape (T, head_dim//2),
                   already sliced to the positions being processed

    Returns:
        Rotated tensor of the same shape and dtype as x
    """
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return (
        torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2))
        .flatten(-2)
        .to(x.dtype)
    )


# -----------------------------------------------------------------------------
# Loop Index Embedding
# -----------------------------------------------------------------------------


def loop_index_embedding(
    h: torch.Tensor,
    loop_t: int,
    loop_dim: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    Inject a sinusoidal loop-index signal into the first loop_dim channels of h.

    Analogous to RoPE for sequence position, but applied over recurrence depth
    instead of token position. Without this, the shared recurrent block weights
    must handle both early-stage pattern-matching and late-stage refinement with
    no signal distinguishing which loop they are on.

    Args:
        h: Hidden state tensor of shape (B, T, dim)
        loop_t: Current loop iteration index (0-based)
        loop_dim: Number of leading channels to receive the embedding (must be even)
        theta: Sinusoidal base frequency

    Returns:
        h with a sinusoidal bias added to its first loop_dim channels
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs  # (loop_dim//2,)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)
