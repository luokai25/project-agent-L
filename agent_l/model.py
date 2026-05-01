"""
Agent L: Recurrent-Depth Transformer Language Model.

A research implementation combining proven components from published papers:

Architecture:
    Input tokens
         ↓
    [Prelude]          — prelude_layers standard transformer blocks, run once
         ↓
    [Recurrent Block]  — one transformer block looped T times with input injection
         ↑_______↓     h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
         ↓
    [Coda]             — coda_layers standard transformer blocks, run once
         ↓
    Output logits

Key properties:
- Same weights, more loops → deeper reasoning (no parameter growth)
- Depth extrapolation: train on N loops, test on N+k loops (emergent)
- ACT halting: variable compute per position within a batch
- MoE FFN in recurrent block: breadth across domains
- LTI-stable injection: spectral radius < 1 guaranteed
- Supports both GQA and MLA attention

Provenance:
- Looped transformers: Saunshi et al., ICLR 2025
- MLA: DeepSeek-V2, 2024
- DeepSeekMoE: Dai et al., ACL 2024
- ACT: Graves, 2016

IMPORTANT: This is NOT Claude's architecture. Anthropic has not published
Claude's architecture. This is a novel combination of proven components.

References:
[1] Saunshi et al. "Reasoning with Latent Thoughts" ICLR 2025
[2] DeepSeek-AI "DeepSeek-V2" arXiv 2024
[3] Dai et al. "DeepSeekMoE" ACL 2024
[4] Graves "Adaptive Computation Time for RNNs" arXiv 2016
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AgentConfig
from .layers import RMSNorm, precompute_rope_freqs, causal_mask
from .recurrent import RecurrentBlock, TransformerBlock


class AgentL(nn.Module):
    """
    Agent L Recurrent-Depth Transformer.
    
    Combines published research components into a unified architecture:
    
    - Prelude: Standard transformer layers for initial encoding
    - Recurrent Block: Looped transformer with MoE + ACT halting
    - Coda: Standard transformer layers for final processing
    
    The recurrent block can be run with more iterations at inference time
    for harder problems (depth extrapolation, proven in [1]).
    
    Example:
        >>> from agent_l import AgentL, agent_3b
        >>> cfg = agent_3b()
        >>> model = AgentL(cfg)
        >>> 
        >>> # Forward pass
        >>> tokens = torch.randint(0, cfg.vocab_size, (1, 32))
        >>> logits = model(tokens, n_loops=8)
        >>> 
        >>> # Generation
        >>> output = model.generate(tokens, max_new_tokens=64)
    """
    
    def __init__(self, cfg: AgentConfig):
        """
        Args:
            cfg: AgentConfig specifying all hyperparameters
        """
        super().__init__()
        self.cfg = cfg
        
        # Token embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        
        # RoPE frequencies
        # GQA uses full head_dim for RoPE; MLA uses qk_rope_head_dim
        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, 
            cfg.max_seq_len, 
            cfg.rope_theta
        )
        self.register_buffer("freqs_cis", freqs)
        
        freqs_mla = precompute_rope_freqs(
            cfg.qk_rope_head_dim, 
            cfg.max_seq_len, 
            cfg.rope_theta
        )
        self.register_buffer("freqs_cis_mla", freqs_mla)
        
        # Prelude: standard transformer blocks
        self.prelude = nn.ModuleList([
            TransformerBlock(cfg, use_moe=False) 
            for _ in range(cfg.prelude_layers)
        ])
        
        # Recurrent block (looped)
        self.recurrent = RecurrentBlock(cfg)
        
        # Coda: standard transformer blocks
        self.coda = nn.ModuleList([
            TransformerBlock(cfg, use_moe=False) 
            for _ in range(cfg.coda_layers)
        ])
        
        # Output
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        
        # Weight tying (reduces parameters, improves generalization)
        self.head.weight = self.embed.weight
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights with N(0, 0.02) - standard for transformers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass: Prelude → Recurrent Block → Coda.
        
        Args:
            input_ids: Token indices of shape (B, T)
            n_loops: Recurrent loop depth (default: cfg.max_loop_iters)
                Can be increased at inference for deeper reasoning
            kv_cache: Dict for autoregressive KV caching
            start_pos: Position offset for RoPE during incremental decoding
        
        Returns:
            Logits of shape (B, T, vocab_size)
        """
        T = input_ids.shape[1]
        device = input_ids.device
        
        # Embed tokens
        x = self.embed(input_ids)
        
        # Select RoPE frequencies based on attention type
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[start_pos : start_pos + T]
        
        # Causal mask (only for prefill, not single-token decode)
        mask = causal_mask(T, device, x.dtype) if T > 1 else None
        
        # Prelude
        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")
        
        # Recurrent block
        e = x  # Encoded input, frozen for injection each loop
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)
        
        # Coda
        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")
        
        # Output logits
        return self.head(self.norm(x))
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV caching.
        
        On step 0: process full prompt
        On subsequent steps: process only last token, retrieve K/V from cache
        
        Args:
            input_ids: Prompt tokens, shape (B, T)
            max_new_tokens: Number of tokens to generate
            n_loops: Recurrent depth per decode step
            temperature: Sampling temperature (lower = more greedy)
            top_k: Restrict to top-K logits (0 = disabled)
        
        Returns:
            Generated tokens, shape (B, T + max_new_tokens)
        """
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]
        
        for step in range(max_new_tokens):
            # Prefill on first step, then decode single token
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
            
            # Forward pass
            logits = self.forward(
                cur_ids, 
                n_loops=n_loops, 
                kv_cache=kv_cache, 
                start_pos=start_pos
            )
            
            # Sample next token
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> dict:
        """Count total and active parameters."""
        total = sum(p.numel() for p in self.parameters())
        
        # Estimate active parameters for MoE
        cfg = self.cfg
        
        # Embedding + output head
        embed_params = cfg.vocab_size * cfg.dim
        
        # Prelude + Coda (dense)
        prelude_params = cfg.prelude_layers * (
            # Attention (approximate for MLA)
            cfg.dim * cfg.dim * 2 +  # rough estimate
            cfg.dim * cfg.dim * 4    # FFN
        )
        coda_params = cfg.coda_layers * prelude_params / cfg.prelude_layers
        
        # Recurrent block
        # - Attention (shared)
        recurrent_attn = cfg.dim * cfg.dim * 2
        # - MoE: only topk experts active
        expert_params = cfg.dim * cfg.expert_dim * 2 + cfg.expert_dim * cfg.dim
        routed_active = cfg.n_experts_per_tok * expert_params
        shared_active = cfg.n_shared_experts * expert_params * cfg.n_experts_per_tok
        moe_active = routed_active + shared_active
        
        active = int(embed_params + prelude_params + coda_params + recurrent_attn + moe_active)
        
        return {
            "total": total,
            "active_estimate": active,
            "total_billions": total / 1e9,
            "active_billions": active / 1e9,
        }
