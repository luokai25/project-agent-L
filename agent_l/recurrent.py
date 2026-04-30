"""
Recurrent block components for Agent L.

This module implements:
- LoRAAdapter: Depth-wise LoRA per loop iteration
- LTIInjection: Linear Time-Invariant injection for stability
- ACTHalting: Adaptive Computation Time halting mechanism
- RecurrentBlock: The core looped transformer block
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AgentConfig
from .layers import RMSNorm, loop_index_embedding
from .attention import GQAttention, MLAttention
from .moe import MoEFFN


class LoRAAdapter(nn.Module):
    """
    Depth-wise LoRA adaptation for the recurrent block (Bae et al., 2024).

    Pure weight-tying (identical weights every loop) limits expressiveness;
    fully distinct weights per loop eliminate parameter savings. This adapter
    sits in between: shared low-rank projections with a per-loop scale vector.

    delta(x, t) = (down(x) * scale[t]) @ B
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)  # shared A: dim → rank
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)  # shared B: rank → dim
        self.scale = nn.Embedding(max_loops, rank)  # per-loop element-wise scale

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, dim)
            loop_t: Current loop index for scale lookup

        Returns:
            Delta tensor of shape (B, T, dim) to add to block output
        """
        # Clamp for depth extrapolation at inference
        max_t = self.scale.num_embeddings - 1
        t_idx = loop_t if loop_t <= max_t else max_t
        s = self.scale.weight[t_idx]  # (rank,)
        return self.down(x) * s @ self.B


class LTIInjection(nn.Module):
    """
    Linear Time-Invariant injection for stable recurrent updates.

    Implements h_{t+1} = A·h_t + B·e + transformer_out with guaranteed
    spectral radius ρ(A) < 1 via log-space parameterization.

    This prevents hidden state drift across arbitrary loop depth, keeping
    the original input signal e alive throughout recurrence.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Log-space parameters for numerical stability
        self.log_dt = nn.Parameter(torch.zeros(()))
        self.log_A = nn.Parameter(torch.zeros(dim))

    def get_A(self) -> torch.Tensor:
        """
        Compute A = exp(-exp(log_dt + log_A)) with ρ(A) < 1 guaranteed.
        """
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute h_{t+1} = A·h_t + B·e + transformer_out.

        Args:
            h: Current hidden state (B, T, dim)
            e: Encoded input from Prelude, frozen across loops (B, T, dim)
            transformer_out: Output of recurrent TransformerBlock (B, T, dim)

        Returns:
            Updated hidden state (B, T, dim)
        """
        A = self.get_A()
        B = 1.0 - A  # Ensures unity gain on the e signal
        return A * h + B * e + transformer_out


class ACTHalting(nn.Module):
    """
    Adaptive Computation Time halting mechanism (Graves, 2016).

    Learns a per-position halting probability at each loop iteration. Positions
    where the hidden state has converged stop accumulating updates, while
    positions still being refined continue. This enables variable compute per
    token within the same batch.

    Also makes the model Turing-complete under certain assumptions about
    transformer block expressiveness.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict per-position halting probability from hidden state.

        Args:
            h: Hidden state of shape (B, T, dim)

        Returns:
            Halting probability of shape (B, T), values in (0, 1)
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)


class TransformerBlock(nn.Module):
    """
    Standard transformer block: Attention + FFN with residual connections.

    When use_moe=True, uses MoE FFN; otherwise uses simple SwiGLU FFN.
    """

    def __init__(self, cfg: AgentConfig, use_moe: bool = False):
        super().__init__()
        self.attn = GQAttention(cfg) if cfg.attn_type == "gqa" else MLAttention(cfg)
        self.attn_norm = RMSNorm(cfg.dim)

        if use_moe:
            self.ffn = MoEFFN(cfg)
        else:
            # Simple SwiGLU FFN for prelude/coda
            hidden_dim = cfg.dim * 4
            self.ffn = nn.Sequential(
                nn.Linear(cfg.dim, hidden_dim, bias=False),
                nn.SiLU(),
                nn.Linear(hidden_dim, cfg.dim, bias=False),
            )

        self.ffn_norm = RMSNorm(cfg.dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, T, dim)
            freqs_cis: RoPE frequencies
            mask: Causal mask or None
            kv_cache: KV cache dict
            cache_key: Cache key for this block

        Returns:
            Output of shape (B, T, dim)
        """
        # Attention with residual
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key)
        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RecurrentBlock(nn.Module):
    """
    The core recurrent block of Agent L — a single TransformerBlock looped T times.

    At each loop iteration t, the hidden state h is updated via:
        1. loop_index_embedding: Inject sinusoidal loop-index signal into h
        2. TransformerBlock: Compute attention + MoE FFN on normalized (h + e)
        3. LoRAAdapter: Apply depth-wise LoRA delta to transformer output
        4. LTIInjection: Stable update h = A·h + B·e + transformer_out
        5. ACTHalting: Accumulate halting probabilities; converged positions stop

    The encoded input e (output of Prelude) is injected at every step to keep
    the original input signal alive, preventing drift.

    More loop iterations at inference = deeper reasoning chains (depth extrapolation).
    """

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = cfg.dim // 8  # fraction of channels for loop-index embedding

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Run the recurrent loop for up to n_loops iterations with ACT early exit.

        Args:
            h: Initial hidden state from Prelude (B, T, dim)
            e: Encoded input frozen for injection each step (B, T, dim)
            freqs_cis: Precomputed RoPE frequencies
            mask: Additive causal mask or None
            n_loops: Number of loop iterations; defaults to cfg.max_loop_iters
            kv_cache: Cache dict passed to inner TransformerBlock

        Returns:
            ACT-weighted sum of hidden states across iterations (B, T, dim)
        """
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            # Inject loop-index signal
            h_loop = loop_index_embedding(h, t, self.loop_dim)

            # Transformer block on (h + e)
            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)

            # Apply LoRA adapter
            trans_out = trans_out + self.lora(trans_out, t)

            # LTI-stable hidden state update
            h = self.injection(h, e, trans_out)

            # Compute halting probability
            p = self.act(h)  # (B, T)
            still_running = ~halted

            # ACT remainder trick: assign remaining probability mass at threshold
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()

            # Accumulate weighted hidden state
            h_out = h_out + weight.unsqueeze(-1) * h

            # Update cumulative probability and halt status
            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            # Short-circuit only when no KV cache (cache needs all loop keys populated)
            if halted.all() and kv_cache is None:
                break

        return h_out
