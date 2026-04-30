"""
Mixture of Experts (MoE) for Agent L.

This module implements fine-grained MoE with shared experts,
following the DeepSeekMoE architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AgentConfig


class Expert(nn.Module):
    """
    Single expert FFN with SwiGLU activation.

    Architecture: x → up(x) * gate(x) → down(...)
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (..., dim)
        Returns:
            Output of shape (..., dim)
        """
        return self.down(F.silu(self.up(x)) * self.gate(x))


class MoEFFN(nn.Module):
    """
    Fine-grained Mixture-of-Experts FFN (DeepSeekMoE, Dai et al., 2024).

    Two classes of experts:
    - Routed experts: n_experts small FFNs; each token activates top-K via router
    - Shared experts: n_shared_experts larger FFNs always activated for every token

    Shared experts absorb common cross-domain patterns (syntax, basic reasoning)
    that would otherwise be redundantly learned by many routed experts.

    Load balancing is achieved via an aux-loss-free bias mechanism (DeepSeek-V3):
    the bias shifts only expert selection, not gating weights, so it never affects
    gradients while still encouraging balanced expert utilization.
    """

    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok

        # Router for expert selection
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        # Load-balancing bias (adjusted externally during training, not a gradient param)
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

        # Routed experts (fine-grained, smaller)
        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )

        # Shared experts (larger, always active)
        self.shared_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok) for _ in range(self.n_shared)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, T, dim)
        Returns:
            Output of shape (B, T, dim); shared expert outputs summed on top
            of weighted routed expert outputs
        """
        B, T, D = x.shape
        flat = x.view(B * T, D)

        # Compute routing scores (unbiased for gradients)
        logits = self.router(flat)  # (B*T, n_experts)
        scores = F.softmax(logits, dim=-1)

        # Select top-K experts using biased logits for selection only
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # renormalize

        # Dispatch to routed experts (token-level scatter)
        out = torch.zeros_like(flat)
        for i in range(self.topk):
            expert_ids = topk_idx[:, i]
            token_scores = topk_scores[:, i].unsqueeze(-1)
            for eid in range(self.n_experts):
                mask = expert_ids == eid
                if not mask.any():
                    continue
                out[mask] += token_scores[mask] * self.routed_experts[eid](flat[mask])

        # Shared experts always fire for every token
        for shared in self.shared_experts:
            out = out + shared(flat)

        return out.view(B, T, D)
