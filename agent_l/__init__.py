"""
Agent L: A Recurrent-Depth Transformer implementation.

This package provides a research implementation combining proven components
from published papers into a unified architecture.

Architecture:
    Input → [Prelude] → [Recurrent Block × T loops] → [Coda] → Output

Provenance of components:
- Looped transformers: Saunshi et al., ICLR 2025 (proven depth extrapolation)
- MLA attention: DeepSeek-V2, 2024 (93% KV cache reduction, production-proven)
- DeepSeekMoE: Dai et al., ACL 2024 (fine-grained + shared experts)
- ACT halting: Graves, 2016 (adaptive computation time)

IMPORTANT: This is NOT Claude's architecture. Anthropic has not published
Claude's architecture. This is a novel combination of proven components.

Quick Start:
    >>> from agent_l import AgentL, AgentConfig, agent_3b
    >>> 
    >>> # Use pre-configured model
    >>> cfg = agent_3b()
    >>> model = AgentL(cfg)
    >>> 
    >>> # Run forward pass
    >>> import torch
    >>> tokens = torch.randint(0, cfg.vocab_size, (1, 32))
    >>> logits = model(tokens, n_loops=8)
    >>> 
    >>> # Generate tokens
    >>> output = model.generate(tokens, max_new_tokens=64, n_loops=8)
    >>> 
    >>> # Check parameter count
    >>> params = model.count_parameters()
    >>> print(f"Total: {params['total_billions']:.2f}B, Active: {params['active_billions']:.2f}B")

References:
[1] Saunshi et al. "Reasoning with Latent Thoughts" ICLR 2025
[2] DeepSeek-AI "DeepSeek-V2" arXiv 2024
[3] Dai et al. "DeepSeekMoE" ACL 2024
[4] Graves "Adaptive Computation Time for RNNs" arXiv 2016
"""

# Configuration
from .config import (
    AgentConfig,
    agent_1b,
    agent_3b,
    agent_10b,
    agent_50b,
    agent_100b,
)

# Layers
from .layers import (
    RMSNorm,
    precompute_rope_freqs,
    loop_index_embedding,
)

# Attention
from .attention import (
    GQAttention,
    MLAttention,
)

# MoE
from .moe import (
    Expert,
    MoEFFN,
    DenseFFN,
)

# Recurrent components
from .recurrent import (
    LoRAAdapter,
    LTIInjection,
    ACTHalting,
    TransformerBlock,
    RecurrentBlock,
)

# Main model
from .model import AgentL

__all__ = [
    # Config
    "AgentConfig",
    "agent_1b",
    "agent_3b",
    "agent_10b",
    "agent_50b",
    "agent_100b",
    # Layers
    "RMSNorm",
    "precompute_rope_freqs",
    "loop_index_embedding",
    # Attention
    "GQAttention",
    "MLAttention",
    # MoE
    "Expert",
    "MoEFFN",
    "DenseFFN",
    # Recurrent
    "LoRAAdapter",
    "LTIInjection",
    "ACTHalting",
    "TransformerBlock",
    "RecurrentBlock",
    # Model
    "AgentL",
]

__version__ = "0.2.0"
