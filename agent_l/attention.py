"""
Attention mechanisms for Agent L.

Two implementations with published provenance:

1. GQAttention: Grouped Query Attention (Ainslie et al., 2023)
   - Standard, widely deployed
   - KV cache: n_kv_heads × head_dim per token
   
2. MLAttention: Multi-Latent Attention (DeepSeek-V2, 2024)
   - 93% KV cache reduction
   - 5.76× throughput improvement
   - Production-proven in DeepSeek-V2/V3

References:
[1] Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models" 2023
[2] DeepSeek-AI "DeepSeek-V2" arXiv 2024 - MLA architecture
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, apply_rope
from .config import AgentConfig

# Flash Attention 2 support (optional, requires CUDA)
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


# ============================================================
# Grouped Query Attention (GQA)
# ============================================================

class GQAttention(nn.Module):
    """
    Grouped Query Attention [1].
    
    Uses fewer KV heads than Q heads (n_kv_heads < n_heads).
    Each KV head is shared across n_heads // n_kv_heads query heads,
    reducing KV cache size by that factor while keeping full query expressiveness.
    
    When flash-attn is installed, uses flash_attn_func which handles GQA natively
    (no KV head expansion needed) and is IO-optimal.
    
    References:
        [1] Ainslie et al., 2023 - GQA paper
    """
    
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads
        
        # Projections
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
            cache_key: Unique key for this layer in cache dict
        
        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape
        
        # Project to Q, K, V
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
        
        # Update KV cache
        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}
        
        # Compute attention
        if _HAS_FLASH_ATTN:
            # Flash Attention handles GQA natively
            orig_dtype = q.dtype
            out = flash_attn_func(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=(mask is not None)
            )
            out = out.to(orig_dtype).contiguous().view(B, T, -1)
        else:
            # Fallback: manual attention with KV expansion
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
            
            q = q.transpose(1, 2)  # (B, H, T, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            if mask is not None:
                attn = attn + mask
            
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout_p, training=self.training)
            
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.wo(out)


# ============================================================
# Multi-Latent Attention (MLA) - DeepSeek-V2
# ============================================================

class MLAttention(nn.Module):
    """
    Multi-Latent Attention (MLA) from DeepSeek-V2 [2].
    
    Key innovation: Instead of caching full K and V tensors (n_heads × head_dim per token),
    MLA compresses the KV path through a low-rank latent c_kv and only caches that
    plus the RoPE keys. K_nope and V are reconstructed from c_kv at each decode step.
    
    This trades a cheap linear projection for dramatically smaller cache memory:
    - Standard GQA KV cache: 2 × n_kv_heads × head_dim × seq_len × batch
    - MLA KV cache: kv_lora_rank + qk_rope_head_dim per token
    
    Results from DeepSeek-V2:
    - 93.3% KV cache reduction
    - 5.76× maximum generation throughput
    
    Architecture:
    
    Q path:
        x → q_down (dim→q_lora_rank) → q_norm
          → q_up_nope (q_lora_rank → n_heads×qk_nope_head_dim)  [no RoPE]
          → q_up_rope (q_lora_rank → n_heads×qk_rope_head_dim)  [RoPE applied]
        q = cat(q_nope, q_rope) per head
    
    KV path:
        x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
          splits into:
          - c_kv (latent, cached): kv_lora_rank dims
          - k_rope_raw (shared across heads): qk_rope_head_dim dims
        k_rope = RoPE(k_rope_raw)  [same for all heads]
        At decode: reconstruct k_nope = c_kv @ kv_up_nope, v = c_kv @ kv_up_v
    
    References:
        [2] DeepSeek-AI, "DeepSeek-V2" arXiv 2024
            https://arxiv.org/abs/2405.04434
    """
    
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.q_lora_rank = cfg.q_lora_rank
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.head_dim = cfg.qk_rope_head_dim + cfg.qk_nope_head_dim
        self.dropout_p = cfg.dropout
        
        # Q path: down-project to latent, then up-project with split for RoPE/nope
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False)
        self.q_up_rope = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False)
        
        # KV path: down-project to latent + rope key, up-project to reconstruct
        self.kv_down = nn.Linear(cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False)
        self.kv_up_nope = nn.Linear(cfg.kv_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False)
        self.kv_up_v = nn.Linear(cfg.kv_lora_rank, cfg.n_heads * cfg.v_head_dim, bias=False)
        
        # Output projection
        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
    
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
            freqs_cis: RoPE frequencies for qk_rope_head_dim, shape (T, qk_rope_head_dim//2)
            mask: Additive causal mask or None
            kv_cache: Dict for KV caching
            cache_key: Key for this layer in cache
        
        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape
        
        # ========== Q Path ==========
        q_down = self.q_down(x)  # (B, T, q_lora_rank)
        q_down = self.q_norm(q_down)
        
        q_nope = self.q_up_nope(q_down).view(B, T, self.n_heads, self.qk_nope_head_dim)
        q_rope = self.q_up_rope(q_down).view(B, T, self.n_heads, self.qk_rope_head_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, n_heads, head_dim)
        
        # ========== KV Path ==========
        kv_down = self.kv_down(x)  # (B, T, kv_lora_rank + qk_rope_head_dim)
        c_kv = kv_down[..., :self.kv_lora_rank]  # (B, T, kv_lora_rank)
        k_rope_raw = kv_down[..., self.kv_lora_rank:]  # (B, T, qk_rope_head_dim)
        
        # Apply RoPE to the shared rope key
        k_rope = apply_rope(
            k_rope_raw.unsqueeze(2),  # (B, T, 1, qk_rope_head_dim)
            freqs_cis
        ).squeeze(2)  # (B, T, qk_rope_head_dim)
        
        # Update cache
        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            kv_cache[cache_key] = {
                "c_kv": c_kv.detach(),
                "k_rope": k_rope.detach()
            }
        
        # Reconstruct K_nope and V from latent
        k_nope = self.kv_up_nope(c_kv).view(B, -1, self.n_heads, self.qk_nope_head_dim)
        v = self.kv_up_v(c_kv).view(B, -1, self.n_heads, self.v_head_dim)
        
        # Expand k_rope to all heads
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, self.n_heads, -1)  # (B, T, n_heads, qk_rope_head_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, T, n_heads, head_dim)
        
        # ========== Attention ==========
        if _HAS_FLASH_ATTN:
            orig_dtype = q.dtype
            out = flash_attn_func(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=(mask is not None)
            )
            out = out.to(orig_dtype).contiguous().view(B, T, -1)
        else:
            # Manual attention
            q = q.transpose(1, 2)  # (B, H, T, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            if mask is not None:
                attn = attn + mask
            
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout_p, training=self.training)
            
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.wo(out)
