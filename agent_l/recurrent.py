"""
Recurrent block components for Agent L.

This module implements:
- LoRAAdapter: Depth-wise LoRA per loop iteration
- LTIInjection: Linear Time-Invariant injection for stability
- ACTHalting: Adaptive Computation Time halting (Graves 2016)
- RecurrentBlock: Core looped transformer block

Key properties:
- Depth extrapolation: train on T loops, test on T+k for harder problems
- Variable compute: ACT halting allocates more loops to harder tokens
- Stability: LTI injection guarantees spectral radius < 1

References:
[1] Graves "Adaptive Computation Time for RNNs" arXiv 2016
[2] Saunshi et al. "Reasoning with Latent Thoughts" ICLR 2025
[3] Bae et al. "LoRA" ICLR 2024
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AgentConfig
from .layers import RMSNorm, loop_index_embedding
from .attention import GQAttention, MLAttention
from .moe import MoEFFN, DenseFFN


# ============================================================
# LoRA Depth Adapter
# ============================================================

class LoRAAdapter(nn.Module):
    """
    Depth-wise LoRA adaptation for recurrent block [3].
    
    Pure weight-tying (identical weights every loop) limits expressiveness.
    Fully distinct weights per loop eliminate parameter savings.
    
    This adapter sits in between: shared low-rank projections with a
    small per-loop scale vector that shifts the effective transformation
    at each depth without significant parameter overhead.
    
    delta(x, t) = down(x) * scale[t] @ B
    
    References:
        [3] Bae et al., ICLR 2024 - LoRA
    """
    
    def __init__(self, dim: int, rank: int, max_loops: int):
        """
        Args:
            dim: Model hidden dimension
            rank: Low-rank bottleneck dimension
            max_loops: Maximum loop iterations (embedding table size)
        """
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.scale = nn.Embedding(max_loops, rank)
    
    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, T, dim)
            loop_t: Current loop index (0-based)
        
        Returns:
            Delta tensor of shape (B, T, dim) to add to block output
        """
        # Clamp for depth extrapolation (reuse last scale beyond training max)
        max_t = self.scale.num_embeddings - 1
        t_idx = min(loop_t, max_t)
        
        s = self.scale(torch.tensor([t_idx], device=x.device))  # (1, rank)
        down = self.down(x)  # (B, T, rank)
        return (down * s).unsqueeze(1) @ self.B.unsqueeze(0)  # (B, T, dim)


# ============================================================
# LTI Injection (Stability)
# ============================================================

class LTIInjection(nn.Module):
    """
    Linear Time-Invariant injection for stable recurrence.
    
    The recurrent update: h_{t+1} = A·h_t + B·e + transformer_out
    
    For stability, we need ||A|| < 1 (spectral radius < 1) to prevent
    unbounded growth over many loop iterations.
    
    Solution: Parameterize A as A = alpha * tanh(W) / ||W||
    This guarantees ||A|| <= alpha < 1 when alpha < 1.
    
    The encoded input e is injected at every step to keep the original
    input signal alive across arbitrary loop depth, preventing drift.
    """
    
    def __init__(self, dim: int, alpha: float = 0.9):
        """
        Args:
            dim: Model hidden dimension
            alpha: Spectral radius upper bound (< 1 for stability)
        """
        super().__init__()
        self.alpha = alpha
        self.W_h = nn.Linear(dim, dim, bias=False)
        self.W_e = nn.Linear(dim, dim, bias=False)
    
    def forward(
        self, 
        h: torch.Tensor, 
        e: torch.Tensor, 
        delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: Current hidden state, shape (B, T, dim)
            e: Encoded input (frozen), shape (B, T, dim)
            delta: Transformer output to inject, shape (B, T, dim)
        
        Returns:
            Updated hidden state, shape (B, T, dim)
        """
        # Normalize W to control spectral radius
        W = self.W_h.weight
        W_norm = W.norm()
        if W_norm > 0:
            A = self.alpha * torch.tanh(W / W_norm)
        else:
            A = self.alpha * torch.tanh(W)
        
        # Apply A via functional linear (use normalized weight)
        h_update = F.linear(h, A, None)
        
        return h_update + self.W_e(e) + delta


# ============================================================
# ACT Halting (Graves 2016)
# ============================================================

class ACTHalting(nn.Module):
    """
    Adaptive Computation Time halting [1].
    
    Each position accumulates a halting probability across loop iterations.
    When cumulative probability >= threshold, that position "halts" and
    stops contributing to further iterations.
    
    The final output is a weighted sum of hidden states across iterations,
    where weights reflect when each position converged.
    
    ACT enables variable compute per position within a batch - harder tokens
    get more iterations, easier tokens halt early.
    
    References:
        [1] Graves, arXiv 2016 - "Adaptive Computation Time for RNNs"
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1, bias=False)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden state, shape (B, T, dim)
        
        Returns:
            Halting probability per position, shape (B, T)
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ============================================================
# Transformer Block (single layer)
# ============================================================

class TransformerBlock(nn.Module):
    """
    Single transformer block: Attention + FFN.
    
    Can use either:
    - MoE FFN (for recurrent block)
    - Dense FFN (for prelude/coda)
    """
    
    def __init__(self, cfg: AgentConfig, use_moe: bool = False):
        super().__init__()
        
        # Attention
        if cfg.attn_type == "mla":
            self.attn = MLAttention(cfg)
        else:
            self.attn = GQAttention(cfg)
        
        # FFN
        if use_moe:
            self.ffn = MoEFFN(cfg)
        else:
            self.ffn = DenseFFN(cfg.dim)
        
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Pre-norm architecture: norm → attention → residual → norm → ffn → residual
        """
        # Attention
        x = x + self.attn(self.norm1(x), freqs_cis, mask, kv_cache, cache_key)
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Recurrent Block (looped transformer)
# ============================================================

class RecurrentBlock(nn.Module):
    """
    Core recurrent block with ACT halting [2].
    
    At each loop iteration t:
    1. Loop-index embedding: inject sinusoidal signal for depth awareness
    2. TransformerBlock: compute attention + MoE FFN
    3. LoRAAdapter: apply depth-wise delta
    4. LTIInjection: stable update h = A·h + B·e + transformer_out
    5. ACTHalting: accumulate halting probabilities
    
    The encoded input e (from Prelude) is injected at every step to prevent
    hidden state drift across arbitrary depth.
    
    Properties:
    - Same weights, more loops → deeper reasoning (no parameter growth)
    - Depth extrapolation: train on T, test on T+k
    - Variable compute: ACT allocates more iterations to harder tokens
    
    References:
        [2] Saunshi et al., ICLR 2025 - "Reasoning with Latent Thoughts"
    """
    
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        
        # Core transformer block with MoE
        self.block = TransformerBlock(cfg, use_moe=True)
        
        # Stability and adaptation
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        
        # Fraction of channels receiving loop-index embedding
        self.loop_dim = cfg.dim // 8
    
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
        Run recurrent loop with ACT early exit.
        
        Args:
            h: Initial hidden state from Prelude, shape (B, T, dim)
            e: Encoded input frozen for injection, shape (B, T, dim)
            freqs_cis: Precomputed RoPE frequencies
            mask: Additive causal mask or None
            n_loops: Number of loop iterations (default: cfg.max_loop_iters)
            kv_cache: Cache dict for each loop iteration
        
        Returns:
            ACT-weighted sum of hidden states, shape (B, T, dim)
        """
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape
        
        # ACT state
        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)
        
        for t in range(n_loops):
            # 1. Loop-index embedding
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            
            # 2. Transformer block
            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, cache_key)
            
            # 3. LoRA depth adapter
            trans_out = trans_out + self.lora(trans_out, t)
            
            # 4. LTI-stable update
            h = self.injection(h, e, trans_out)
            
            # 5. ACT halting
            p = self.act(h)  # (B, T)
            still_running = ~halted
            
            # ACT remainder trick: assign remaining probability mass as final weight
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h
            
            # Update state
            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)
            
            # Early exit only when no KV cache (cache requires all depths to run)
            if halted.all() and kv_cache is None:
                break
        
        return h_out
