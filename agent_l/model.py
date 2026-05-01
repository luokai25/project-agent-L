"""
Agent L: Recurrent-Depth Transformer Language Model.

A theoretical reconstruction of the Claude Mythos architecture as a
Recurrent-Depth Transformer (RDT), implementing:

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
- Same weights, more loops → deeper reasoning, no parameter growth
- Depth extrapolation: train on N loops, test on N+k loops (emergent)
- ACT halting: variable compute per position within a batch
- MoE FFN in the recurrent block: breadth across domains
- LTI-stable injection: spectral radius < 1 guaranteed by construction
- Supports both GQA and MLA attention (set via cfg.attn_type)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AgentConfig
from .layers import RMSNorm, precompute_rope_freqs
from .recurrent import TransformerBlock, RecurrentBlock


class AgentL(nn.Module):
    """
    Agent L — Recurrent-Depth Transformer language model.

    Implements the hypothesized Claude Mythos architecture as a Recurrent-Depth
    Transformer (RDT). The model divides computation into three functional blocks:

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
    - Same weights, more loops → deeper reasoning, no parameter growth
    - Depth extrapolation: train on N loops, test on N+k loops (emergent)
    - ACT halting: variable compute per position within a batch
    - MoE FFN in the recurrent block: breadth across domains
    - LTI-stable injection: spectral radius < 1 guaranteed by construction
    - Supports both GQA and MLA attention (set via cfg.attn_type)
    """

    def __init__(self, cfg: AgentConfig):
        """
        Args:
            cfg: AgentConfig specifying all architecture hyperparameters
        """
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # Precompute RoPE frequencies
        # GQA uses full head_dim for RoPE; MLA uses only qk_rope_head_dim
        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis", freqs)

        freqs_mla = precompute_rope_freqs(
            cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis_mla", freqs_mla)

        # Prelude: standard transformer blocks before the loop
        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )

        # Recurrent block: looped T times
        self.recurrent = RecurrentBlock(cfg)

        # Coda: standard transformer blocks after the loop
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )

        # Final normalization and output projection
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Weight tying between embedding and output head
        self.head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all linear and embedding weights with N(0, 0.02)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Build an additive causal mask: 0 on and below diagonal, -inf above.

        Args:
            seq_len: Sequence length
            device: Target device
            dtype: Tensor dtype (must match activation dtype for correct addition)

        Returns:
            Tensor of shape (1, 1, seq_len, seq_len) broadcastable over (B, H, T, S)
        """
        mask = torch.full(
            (1, 1, seq_len, seq_len),
            float("-inf"),
            device=device,
            dtype=dtype,
        )
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass through Prelude → Recurrent Block → Coda.

        Args:
            input_ids: Token indices of shape (B, T)
            n_loops: Recurrent loop depth; defaults to cfg.max_loop_iters.
                     Increase at inference to extrapolate to harder problems.
            kv_cache: Dict mutated in-place for autoregressive KV caching;
                      pass an empty dict {} and reuse across decode steps
            start_pos: Index of the first token in input_ids within the full
                       sequence; used to select correct RoPE frequencies
                       during incremental decoding

        Returns:
            Logits of shape (B, T, vocab_size)
        """
        T = input_ids.shape[1]
        device = input_ids.device

        # Token embedding
        x = self.embed(input_ids)

        # Select correct RoPE frequencies
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[start_pos : start_pos + T]

        # Build causal mask (only for sequences > 1 token)
        mask = self._causal_mask(T, device, x.dtype) if T > 1 else None

        # Prelude: standard transformer layers
        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

        # Encoded input frozen for injection every loop
        e = x

        # Recurrent block: looped T times with ACT halting
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)

        # Coda: standard transformer layers
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
        Autoregressive token generation with KV caching.

        On step 0 the full prompt is processed. On subsequent steps only the
        last generated token is passed, with all previous keys and values
        retrieved from kv_cache. This keeps decode cost proportional to one
        token per step rather than the full growing sequence.

        n_loops can be set higher than training value for depth extrapolation
        to harder problems at inference time.

        Args:
            input_ids: Prompt token indices of shape (B, T)
            max_new_tokens: Number of tokens to generate
            n_loops: Recurrent loop depth for each decode step
            temperature: Softmax temperature; lower = more greedy
            top_k: Restrict sampling to top-K logits (0 = disabled)

        Returns:
            Token indices of shape (B, T + max_new_tokens)
        """
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]

        for step in range(max_new_tokens):
            if step == 0:
                # First step: process full prompt
                cur_ids = input_ids
                start_pos = 0
            else:
                # Subsequent steps: process only last token
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1

            # Forward pass with KV cache
            logits = self.forward(
                cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos
            )

            # Sample next token from last position
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

        return input_ids

    def count_parameters(self) -> dict:
        """
        Count model parameters by category.

        Returns:
            Dict with 'total', 'embed', 'prelude', 'recurrent', 'coda', 'head'
        """
        total = sum(p.numel() for p in self.parameters())
        embed = sum(p.numel() for p in self.embed.parameters())
        prelude = sum(p.numel() for p in self.prelude.parameters())
        recurrent = sum(p.numel() for p in self.recurrent.parameters())
        coda = sum(p.numel() for p in self.coda.parameters())
        head = sum(p.numel() for p in self.head.parameters())

        return {
            "total": total,
            "embed": embed,
            "prelude": prelude,
            "recurrent": recurrent,
            "coda": coda,
            "head": head,
        }

    def get_spectral_radius(self) -> float:
        """
        Get the maximum spectral radius of LTI injection A matrix.
        Should be < 1 for stability.
        """
        A = self.recurrent.injection.get_A()
        return A.max().item()
