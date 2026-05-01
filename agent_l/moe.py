"""
Mixture of Experts (MoE) for Agent L.

Implementation of DeepSeekMoE architecture with:
- Fine-grained expert segmentation
- Shared experts for common knowledge
- Aux-loss-free load balancing (DeepSeek-V3)
- Vectorized computation for efficiency

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
    
    4. Vectorized computation:
       Uses einsum for batched expert computation, avoiding slow Python loops.
    
    References:
        [1] Dai et al., ACL 2024 - DeepSeekMoE
        [2] DeepSeek-V3 Technical Report, 2024
    """
    
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok
        self.dim = cfg.dim
        self.expert_dim = cfg.expert_dim
        
        # Router: produces logits for each expert
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        
        # Load-balancing bias (not a gradient parameter)
        # Adjusted externally during training; shifts selection without
        # affecting the gating weights in the gradient
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))
        
        # Fused expert weights for efficient vectorized computation
        # Shape: (n_experts, dim, expert_dim)
        self.up_proj = nn.Parameter(torch.empty(cfg.n_experts, cfg.dim, cfg.expert_dim))
        self.gate_proj = nn.Parameter(torch.empty(cfg.n_experts, cfg.dim, cfg.expert_dim))
        self.down_proj = nn.Parameter(torch.empty(cfg.n_experts, cfg.expert_dim, cfg.dim))
        
        # Initialize fused weights
        for i in range(cfg.n_experts):
            nn.init.normal_(self.up_proj[i], std=0.02)
            nn.init.normal_(self.gate_proj[i], std=0.02)
            nn.init.zeros_(self.down_proj[i])
        
        # Shared experts (always active)
        # Each shared expert is larger: handles common knowledge
        self.shared_experts = nn.ModuleList([
            Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
            for _ in range(self.n_shared)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized forward pass using einsum for batched expert computation.
        
        Args:
            x: Input of shape (B, T, dim)
        
        Returns:
            Output of shape (B, T, dim)
        """
        B, T, D = x.shape
        flat = x.view(B * T, D)  # (N, D) where N = B*T
        N = flat.shape[0]
        
        # ========== Routing ==========
        # Unbiased logits for gating weights
        logits = self.router(flat)  # (N, n_experts)
        scores = F.softmax(logits, dim=-1)
        
        # Bias-shifted logits for selection (aux-loss-free load balancing)
        topk_scores, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = F.softmax(topk_scores, dim=-1)  # Renormalize
        
        # ========== Vectorized Expert Computation ==========
        # Gather weights for selected experts
        # up_proj[topk_idx]: (N, K, D, E) where E = expert_dim
        up_weights = self.up_proj[topk_idx]
        gate_weights = self.gate_proj[topk_idx]
        down_weights = self.down_proj[topk_idx]
        
        # Expand input for each top-k position
        x_expanded = flat.unsqueeze(1).expand(-1, self.topk, -1)  # (N, K, D)
        
        # Compute expert outputs via batched einsum
        # x @ W: (N, K, D) @ (N, K, D, E) -> (N, K, E)
        up_out = torch.einsum('nkd,nkde->nke', x_expanded, up_weights)
        gate_out = torch.einsum('nkd,nkde->nke', x_expanded, gate_weights)
        hidden = F.silu(gate_out) * up_out  # SwiGLU: gate * up
        
        # hidden @ W_down: (N, K, E) @ (N, K, E, D) -> (N, K, D)
        expert_out = torch.einsum('nke,nked->nkd', hidden, down_weights)
        
        # Weight by routing scores and sum over top-k
        out = (expert_out * topk_scores.unsqueeze(-1)).sum(dim=1)  # (N, D)
        
        # ========== Shared Experts ==========
        # Always active for every token
        for shared in self.shared_experts:
            out = out + shared(flat)
        
        return out.view(B, T, D)
    
    def forward_python_loop(self, x: torch.Tensor) -> torch.Tensor:
        """
        Legacy forward pass using Python loops (slower but more explicit).
        Useful for debugging and verification.
        
        Args:
            x: Input of shape (B, T, dim)
        
        Returns:
            Output of shape (B, T, dim)
        """
        B, T, D = x.shape
        flat = x.view(B * T, D)
        
        # Routing
        logits = self.router(flat)
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        
        # Dispatch using F.linear with fused weights
        out = torch.zeros_like(flat)
        for i in range(self.topk):
            expert_ids = topk_idx[:, i]
            token_scores = topk_scores[:, i].unsqueeze(-1)
            for eid in range(self.n_experts):
                mask = expert_ids == eid
                if not mask.any():
                    continue
                expert_input = flat[mask]
                # F.linear expects weight shape (out_features, in_features)
                # Our weights are (dim, expert_dim), need transpose
                up_out = F.linear(expert_input, self.up_proj[eid].T)
                gate_out = F.linear(expert_input, self.gate_proj[eid].T)
                hidden = F.silu(gate_out) * up_out
                expert_out = F.linear(hidden, self.down_proj[eid].T)
                out[mask] += token_scores[mask] * expert_out
        
        # Shared experts
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
