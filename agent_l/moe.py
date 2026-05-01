"""
Mixture of Experts (MoE) for Agent L.

Implementation of DeepSeekMoE architecture with:
- Fine-grained expert segmentation
- Shared experts for common knowledge
- Aux-loss-free load balancing (DeepSeek-V3)

Key results from DeepSeekMoE:
- 2B model matches GShard 2.9B with 1.5× fewer parameters
- 16B model matches LLaMA2-7B with 40% compute
- 145B model matches DeepSeek 67B with 28.5% compute

References:
[1] Dai et al. "DeepSeekMoE: Towards Ultimate Expert Specialization" ACL 2024
[2] DeepSeek-AI "DeepSeek-V3 Technical Report" arXiv 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .config import AgentConfig


# ============================================================
# SwiGLU Expert
# ============================================================

class Expert(nn.Module):
    """
    Single expert FFN with SwiGLU activation.
    
    Architecture: x → gate(x) * up(x) → down(...)
    
    SwiGLU: gate uses SiLU (Swish) activation
    """
    
    def __init__(self, dim: int, hidden_dim: int):
        """
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (typically 4× dim or expert_dim)
        """
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (..., dim)
        Returns:
            Output of shape (..., dim)
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ============================================================
# DeepSeekMoE FFN
# ============================================================

class MoEFFN(nn.Module):
    """
    DeepSeekMoE Mixture of Experts [1][2].
    
    Two key innovations:
    
    1. Fine-grained expert segmentation:
       Instead of top-K out of N experts, DeepSeekMoE splits experts into
       m×N smaller sub-experts and activates m×K of them. This enables more
       flexible combinations without linearly increasing computation.
       
       In this implementation: we directly use n_experts fine-grained experts
       and select n_experts_per_tok (top-K) per token.
    
    2. Shared experts:
       A subset of experts (n_shared_experts) are always active for every token.
       They capture common, broadly useful knowledge and reduce redundancy
       across routed experts.
    
    3. Aux-loss-free load balancing (DeepSeek-V3):
       Instead of an auxiliary loss, we use a learnable bias per expert that
       shifts the routing selection without affecting gradients. Underused
       experts get higher bias, encouraging selection without gradient penalty.
    
    References:
        [1] Dai et al., ACL 2024 - DeepSeekMoE
        [2] DeepSeek-V3 Technical Report, 2024
    """
    
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok
        
        # Router: produces logits for each expert
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        
        # Load-balancing bias (not a gradient parameter)
        # Adjusted externally during training; shifts selection without
        # affecting the gating weights in the gradient
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))
        
        # Routed experts (fine-grained)
        self.routed_experts = nn.ModuleList([
            Expert(cfg.dim, cfg.expert_dim) 
            for _ in range(cfg.n_experts)
        ])
        
        # Shared experts (always active)
        # Each shared expert is larger: handles common knowledge
        self.shared_experts = nn.ModuleList([
            Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
            for _ in range(self.n_shared)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, T, dim)
        
        Returns:
            Output of shape (B, T, dim)
            Shared expert outputs summed on top of weighted routed outputs
        """
        B, T, D = x.shape
        flat = x.view(B * T, D)  # Flatten for token-level routing
        
        # ========== Routing ==========
        # Unbiased logits for gating weights
        logits = self.router(flat)  # (B*T, n_experts)
        scores = F.softmax(logits, dim=-1)
        
        # Bias-shifted logits for selection (aux-loss-free load balancing)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        
        # Gather scores for selected experts
        topk_scores = scores.gather(-1, topk_idx)  # (B*T, topk)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # Renormalize
        
        # ========== Routed Expert Dispatch ==========
        out = torch.zeros_like(flat)
        
        # Dispatch each selected expert
        for i in range(self.topk):
            expert_ids = topk_idx[:, i]  # (B*T,) - which expert for each token
            token_scores = topk_scores[:, i].unsqueeze(-1)  # (B*T, 1)
            
            # Process each expert
            for eid in range(self.n_experts):
                mask = expert_ids == eid
                if not mask.any():
                    continue
                # Apply expert to matching tokens
                out[mask] += token_scores[mask] * self.routed_experts[eid](flat[mask])
        
        # ========== Shared Experts ==========
        # Always active for every token
        for shared in self.shared_experts:
            out = out + shared(flat)
        
        return out.view(B, T, D)


# ============================================================
# Dense FFN (for non-MoE layers)
# ============================================================

class DenseFFN(nn.Module):
    """
    Standard dense FFN with SwiGLU activation.
    
    Used in Prelude and Coda layers (non-recurrent).
    """
    
    def __init__(self, dim: int, hidden_dim: int = None):
        """
        Args:
            dim: Input/output dimension
            hidden_dim: Hidden dimension (default: 4 × dim)
        """
        super().__init__()
        hidden_dim = hidden_dim or (4 * dim)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
