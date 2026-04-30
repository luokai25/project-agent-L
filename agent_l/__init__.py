"""
Agent L: A Recurrent-Depth Transformer implementation.

This package provides a clean implementation of the hypothesized Claude Mythos
architecture as a Recurrent-Depth Transformer (RDT).

Quick Start:
    from agent_l import AgentL, AgentConfig, agent_3b

    # Use a pre-configured model size
    cfg = agent_3b()
    model = AgentL(cfg)

    # Or configure manually
    cfg = AgentConfig(
        vocab_size=1000,
        dim=256,
        n_heads=8,
        max_seq_len=128,
        max_loop_iters=4,
        attn_type="mla",
        n_experts=8,
        n_shared_experts=1,
    )
    model = AgentL(cfg)

Architecture Overview:
    Input tokens → [Prelude] → [Recurrent Block (looped T times)] → [Coda] → Output

Key Features:
    - Recurrent depth: More loops = deeper reasoning, no parameter growth
    - Depth extrapolation: Train on N loops, test on N+k loops
    - ACT halting: Variable compute per position
    - MoE FFN: Mixture of Experts for domain breadth
    - LTI-stable injection: Guaranteed spectral radius < 1
    - Dual attention: GQA or MLA (Multi-Latent Attention)
"""

from .config import (
    AgentConfig,
    agent_1b,
    agent_3b,
    agent_10b,
    agent_50b,
    agent_100b,
)

from .model import AgentL

from .layers import (
    RMSNorm,
    precompute_rope_freqs,
    loop_index_embedding,
)

from .attention import GQAttention, MLAttention

from .moe import Expert, MoEFFN

from .recurrent import (
    TransformerBlock,
    RecurrentBlock,
    LoRAAdapter,
    LTIInjection,
    ACTHalting,
)

__version__ = "0.1.0"

__all__ = [
    # Main model
    "AgentL",
    "AgentConfig",
    # Pre-configured variants
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
    # Recurrent components
    "TransformerBlock",
    "RecurrentBlock",
    "LoRAAdapter",
    "LTIInjection",
    "ACTHalting",
]
