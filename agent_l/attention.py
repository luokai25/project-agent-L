"""
Attention mechanisms for Agent L.

This module implements:
- GQAttention: Grouped Query Attention with KV cache
- MLAttention: Multi-Latent Attention (DeepSeek-V2 style)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, apply_rope
from .config import AgentConfig

# Flash Attention support (optional)
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


class GQAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023) with Flash Attention 2.

    Uses fewer KV heads than Q heads (n_kv_heads < n_heads). Each KV head is
    shared across n_heads // n_kv_heads query heads, reducing KV cache size
    by that factor while keeping full query expressiveness.

    When flash-attn is installed, uses flash_attn_func which handles GQA natively.
    Falls back to manual scaled dot-product attention otherwise.
    """

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

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
            freqs_cis: RoPE frequencies for head_dim, shape (T, head_dim//2)
            mask: Additive causal mask of shape (1, 1, T, S) or None
            kv_cache: Dict mutated in-place; stores {"k": ..., "v": ...} per cache_key
            cache_key: Unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # KV caching for autoregressive generation
        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        if _HAS_FLASH_ATTN:
            # Flash Attention handles GQA natively
            orig_dtype = q.dtype
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            dropout_p = self.dropout_p if self.training else 0.0
            out = flash_attn_func(
                q, k, v, dropout_p=dropout_p, causal=(mask is not None)
            )
            out = out.to(orig_dtype).contiguous().view(B, T, -1)
        else:
            # Fallback: manual attention with KV head expansion
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
            q = q.transpose(1, 2)  # (B, H, T, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            attn = F.dropout(F.softmax(attn, dim=-1), p=self.dropout_p, training=self.training)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(out)


class MLAttention(nn.Module):
    """
    Multi-Latent Attention (DeepSeek-V2, 2024).

    Instead of caching full K and V tensors, MLA compresses the KV path through
    a low-rank latent c_kv and only caches that plus the RoPE keys. K_nope and V
    are reconstructed from c_kv at each decoding step, trading a cheap linear
    projection for dramatically smaller cache memory (~10-20x reduction).

    Architecture:
        Q path:
            x → q_down (dim→q_lora_rank) → q_norm
              → q_up_nope (no RoPE) + q_up_rope (RoPE applied)
            q = cat(q_nope, q_rope) per head

        KV path:
            x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
              splits into c_kv (latent, cached) and k_rope_raw
            k_rope = RoPE(expand(k_rope_raw))
            c_kv → kv_norm → kv_up → [k_nope | v]
            k = cat(k_nope, k_rope) per head
    """

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        # Q compression path
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False)
        self.q_up_rope = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False)

        # KV compression path: output is [c_kv | k_rope_raw] concatenated
        self.kv_down = nn.Linear(
            cfg.dim,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            bias=False,
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

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
            freqs_cis: RoPE frequencies for qk_rope_head_dim, shape (T, qk_rope_dim//2)
            mask: Additive causal mask or None
            kv_cache: Dict for KV caching
            cache_key: Unique key for this layer's cache

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape

        # Q path with compression
        q_latent = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(q_latent).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(q_latent).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, n_heads, q_head_dim)

        # KV path with compression
        kv_down_out = self.kv_down(x)
        c_kv = kv_down_out[..., : self.kv_lora_rank]  # (B, T, kv_lora_rank)
        k_rope_raw = kv_down_out[..., self.kv_lora_rank:]  # (B, T, qk_rope_dim)

        # Expand k_rope_raw to all heads
        k_rope = k_rope_raw.unsqueeze(2).expand(-1, -1, self.n_heads, -1)  # (B, T, n_heads, qk_rope_dim)
        k_rope = apply_rope(k_rope, freqs_cis)

        # KV caching (store compressed latents)
        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}

        # Reconstruct K_nope and V from c_kv
        kv_up_out = self.kv_up(self.kv_norm(c_kv))  # (B, T, n_heads * (qk_nope_dim + v_dim))
        kv_up_out = kv_up_out.view(B, -1, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv_up_out[..., : self.qk_nope_dim]
        v = kv_up_out[..., self.qk_nope_dim:]

        # Assemble full K
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, T', n_heads, q_head_dim)

        # Attention computation
        q = q.transpose(1, 2)  # (B, n_heads, T, q_head_dim)
        k = k.transpose(1, 2)  # (B, n_heads, T', q_head_dim)
        v = v.transpose(1, 2)  # (B, n_heads, T', v_dim)

        scale = self.q_head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn + mask

        attn = F.dropout(F.softmax(attn, dim=-1), p=self.dropout_p, training=self.training)
        out = torch.matmul(attn, v)  # (B, n_heads, T, v_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(out)
