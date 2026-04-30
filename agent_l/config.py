"""
Configuration for Agent L models.

The AgentConfig dataclass defines all hyperparameters for the Recurrent-Depth Transformer.
"""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """
    Hyperparameter configuration for Agent L Recurrent-Depth Transformer.

    Core Architecture:
        vocab_size: Token vocabulary size
        dim: Model hidden dimension
        n_heads: Number of query attention heads
        n_kv_heads: Number of key/value heads (GQA; ignored by MLA)
        max_seq_len: Maximum sequence length for RoPE precomputation
        max_loop_iters: Default recurrent loop depth T at inference
        prelude_layers: Standard transformer layers before the loop
        coda_layers: Standard transformer layers after the loop

    Attention (attn_type selects between):
        attn_type: "gqa" for Grouped Query Attention, "mla" for Multi-Latent Attention
        kv_lora_rank: [MLA] Compressed KV latent dimension stored in cache
        q_lora_rank: [MLA] Compressed Q latent dimension
        qk_rope_head_dim: [MLA] Per-head dims that receive RoPE
        qk_nope_head_dim: [MLA] Per-head dims without positional encoding
        v_head_dim: [MLA] Per-head value dimension

    MoE FFN (used inside the recurrent block):
        n_experts: Total number of routed expert FFNs
        n_shared_experts: Number of always-active shared experts
        n_experts_per_tok: Top-K experts selected per token by router
        expert_dim: Hidden dimension inside each fine-grained expert

    Other:
        act_threshold: ACT halting threshold (cumulative probability to stop)
        rope_theta: RoPE base frequency
        lora_rank: Rank of the per-loop depth-wise LoRA adapter
        dropout: Dropout probability (0.0 = disabled)
    """

    # Core dimensions
    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads than Q heads
    max_seq_len: int = 4096
    max_loop_iters: int = 16  # T — recurrent depth at inference
    prelude_layers: int = 2
    coda_layers: int = 2

    # Attention type: "gqa" | "mla"
    attn_type: str = "mla"

    # MLA-specific params (only used when attn_type="mla")
    kv_lora_rank: int = 512  # compressed KV latent cached instead of full K/V
    q_lora_rank: int = 1536  # compressed Q latent dim
    qk_rope_head_dim: int = 64  # per-head dims that receive RoPE
    qk_nope_head_dim: int = 128  # per-head dims without RoPE
    v_head_dim: int = 128  # per-head value dim

    # MoE configuration
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # top-K routed experts per token
    expert_dim: int = 512  # hidden dim inside each fine-grained expert

    # ACT halting
    act_threshold: float = 0.99

    # RoPE
    rope_theta: float = 500000.0

    # Depth-wise LoRA adaptation
    lora_rank: int = 16

    # Generation
    max_output_tokens: int = 4096

    # Dropout
    dropout: float = 0.0


# -----------------------------------------------------------------------------
# Pre-configured Model Variants
# -----------------------------------------------------------------------------


def agent_1b() -> AgentConfig:
    """1B parameter config. Small research/fine-tuning model."""
    return AgentConfig(
        vocab_size=32000,
        dim=2048,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=4096,
        max_loop_iters=16,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=256,
        q_lora_rank=512,
        qk_rope_head_dim=32,
        qk_nope_head_dim=64,
        v_head_dim=64,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=2048,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def agent_3b() -> AgentConfig:
    """3B parameter config. Compact inference model."""
    return AgentConfig(
        vocab_size=32000,
        dim=3072,
        n_heads=24,
        n_kv_heads=6,
        max_seq_len=4096,
        max_loop_iters=16,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=384,
        q_lora_rank=768,
        qk_rope_head_dim=32,
        qk_nope_head_dim=96,
        v_head_dim=96,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=4096,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def agent_10b() -> AgentConfig:
    """10B parameter config. Mid-scale general model."""
    return AgentConfig(
        vocab_size=32000,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=24,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=128,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=5632,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=16,
    )


def agent_50b() -> AgentConfig:
    """50B parameter config. Large reasoning model."""
    return AgentConfig(
        vocab_size=32000,
        dim=6144,
        n_heads=48,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=32,
        prelude_layers=3,
        coda_layers=3,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=4,
        expert_dim=9728,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=32,
    )


def agent_100b() -> AgentConfig:
    """100B parameter config. Frontier-class model with 1M context."""
    return AgentConfig(
        vocab_size=32000,
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        max_seq_len=1000000,
        max_loop_iters=32,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=2048,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=8,
        expert_dim=13568,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=64,
        max_output_tokens=131072,
    )
