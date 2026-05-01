"""
Configuration for Agent L models.

Model variants based on proven scaling laws and published architectures.

References:
- DeepSeek-V2: 236B total, 21B active, 128K context
- Saunshi et al.: recurrence-equivalence φ ≈ 0.46
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AgentConfig:
    """
    Hyperparameter configuration for Agent L Recurrent-Depth Transformer.
    
    Architecture:
        Prelude → Recurrent Block (looped T times) → Coda
    
    Each component has published provenance:
    - Looped transformers: Saunshi et al., ICLR 2025
    - MLA attention: DeepSeek-V2, 2024
    - DeepSeekMoE: Dai et al., ACL 2024
    - ACT halting: Graves, 2016
    """
    
    # ============================================================
    # Core Architecture
    # ============================================================
    
    vocab_size: int = 32000
    """Token vocabulary size. DeepSeek-V2 uses 100K for large models."""
    
    dim: int = 2048
    """Model hidden dimension."""
    
    n_heads: int = 16
    """Number of query attention heads."""
    
    n_kv_heads: int = 4
    """Number of key/value heads for GQA. Must divide n_heads evenly."""
    
    max_seq_len: int = 4096
    """Maximum sequence length for RoPE precomputation."""
    
    max_loop_iters: int = 16
    """Recurrent loop depth T at inference. Can be increased for harder problems (depth extrapolation)."""
    
    prelude_layers: int = 2
    """Number of standard transformer layers before the recurrent block."""
    
    coda_layers: int = 2
    """Number of standard transformer layers after the recurrent block."""
    
    # ============================================================
    # Attention: GQA vs MLA (DeepSeek-V2)
    # ============================================================
    
    attn_type: Literal["gqa", "mla"] = "mla"
    """
    Attention type:
    - "gqa": Grouped Query Attention (standard, simpler)
    - "mla": Multi-Latent Attention (DeepSeek-V2, 93% KV cache reduction)
    
    MLA compresses KV into latent vector, dramatically reducing memory bandwidth.
    See DeepSeek-V2 paper for details.
    """
    
    # MLA-specific parameters (DeepSeek-V2 Table 1)
    kv_lora_rank: int = 512
    """MLA: compressed KV latent dimension cached during inference."""
    
    q_lora_rank: int = 1536
    """MLA: compressed Q latent dimension."""
    
    qk_rope_head_dim: int = 64
    """MLA: per-head dims that receive RoPE (decoupled from main attention)."""
    
    qk_nope_head_dim: int = 128
    """MLA: per-head dims without positional encoding."""
    
    v_head_dim: int = 128
    """MLA: per-head value dimension."""
    
    # ============================================================
    # Mixture of Experts (DeepSeekMoE)
    # ============================================================
    
    n_experts: int = 64
    """Total number of routed expert FFNs."""
    
    n_shared_experts: int = 2
    """
    Number of always-active shared experts.
    DeepSeekMoE: shared experts capture common knowledge, reducing redundancy.
    """
    
    n_experts_per_tok: int = 4
    """Top-K experts selected per token by the router."""
    
    expert_dim: int = 512
    """Hidden dimension inside each fine-grained expert."""
    
    # ============================================================
    # ACT Halting (Graves 2016)
    # ============================================================
    
    act_threshold: float = 0.99
    """
    ACT halting threshold.
    Positions stop looping when cumulative halting probability >= threshold.
    """
    
    # ============================================================
    # RoPE (Rotary Position Embeddings)
    # ============================================================
    
    rope_theta: float = 500000.0
    """
    RoPE base frequency.
    LLaMA-3 uses 500K for long context. Higher = slower frequency decay.
    """
    
    # ============================================================
    # LoRA Depth Adapter
    # ============================================================
    
    lora_rank: int = 16
    """
    Rank of per-loop depth-wise LoRA adapter.
    Allows shared recurrent block to behave differently at each loop iteration.
    """
    
    # ============================================================
    # Training
    # ============================================================
    
    dropout: float = 0.0
    """Dropout probability. Set 0.0 for inference, 0.1 for pretraining."""
    
    max_output_tokens: int = 4096
    """Maximum tokens to generate per forward pass."""


# ============================================================
# Model Variants
# ============================================================

def agent_debug() -> AgentConfig:
    """
    Debug config for testing.
    
    Minimal model for rapid iteration (~100K params).
    """
    return AgentConfig(
        vocab_size=1000,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=64,
        lora_rank=4,
    )


def agent_tiny() -> AgentConfig:
    """
    Tiny config for quick experiments.
    
    ~10M parameter model for rapid prototyping.
    """
    return AgentConfig(
        vocab_size=10000,
        dim=256,
        n_heads=8,
        n_kv_heads=2,
        max_seq_len=512,
        max_loop_iters=8,
        prelude_layers=1,
        coda_layers=1,
        attn_type="mla",
        kv_lora_rank=64,
        q_lora_rank=128,
        qk_rope_head_dim=16,
        qk_nope_head_dim=32,
        v_head_dim=32,
        n_experts=16,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=256,
        lora_rank=8,
    )


def agent_small() -> AgentConfig:
    """Alias for agent_1b."""
    return agent_1b()


def agent_medium() -> AgentConfig:
    """Alias for agent_3b."""
    return agent_3b()


def agent_large() -> AgentConfig:
    """Alias for agent_10b."""
    return agent_10b()


def agent_xl() -> AgentConfig:
    """Alias for agent_50b."""
    return agent_50b()


def agent_xxl() -> AgentConfig:
    """Alias for agent_100b."""
    return agent_100b()


def agent_1b() -> AgentConfig:
    """
    1B parameter config.
    
    Small research/fine-tuning model.
    Based on DeepSeekMoE scaling: matches dense 1B with MoE efficiency.
    """
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
    """
    3B parameter config.
    
    Compact inference model.
    Based on DeepSeekMoE: 16B matches LLaMA2-7B with 40% compute.
    """
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
    """
    10B parameter config.
    
    Mid-scale general model.
    Scales to DeepSeek-V2 style dimensions.
    """
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
    """
    50B parameter config.
    
    Large reasoning model with extended context.
    Approaches DeepSeek-V2 scale (236B total, 21B active).
    """
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
    """
    100B parameter config.
    
    Frontier-class model with 128K context.
    Based on DeepSeek-V2 proven scaling.
    """
    return AgentConfig(
        vocab_size=100000,
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        max_seq_len=131072,
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
