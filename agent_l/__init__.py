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
    agent_debug,
    agent_tiny,
    agent_1b,
    agent_3b,
    agent_10b,
    agent_50b,
    agent_100b,
    agent_small,
    agent_medium,
    agent_large,
    agent_xl,
    agent_xxl,
)

from .model import AgentL

from .layers import (
    RMSNorm,
    precompute_rope_freqs,
    apply_rope,
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

from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    export_model,
    load_pretrained,
)

from .profiling import (
    ProfileResult,
    ExpertStats,
    ACTStats,
    profile_forward,
    profile_generation,
    analyze_expert_utilization,
    analyze_act_halting,
    benchmark_depth_scaling,
    print_profile_report,
    print_expert_report,
)

from .generation import (
    GenerationConfig,
    generate_advanced,
    beam_search,
    generate_with_depth_schedule,
    sample_next_token,
    apply_top_p_filtering,
    apply_repetition_penalty,
)

from .logging_utils import (
    get_logger,
    setup_logger,
    TrainingLogger,
    InferenceLogger,
    log_model_summary,
    log_memory_usage,
)

from .utils import (
    # Visualization
    VisualizationData,
    AttentionCapture,
    ExpertRoutingTracker,
    ACTHaltingTracker,
    visualize_attention,
    visualize_expert_routing,
    visualize_act_halting,
    visualize_hidden_evolution,
    create_model_report,
    # Quantization
    QuantizationConfig,
    quantize_model_dynamic,
    quantize_model_static,
    quantize_kv_cache,
    dequantize_kv_cache,
    get_model_size_mb,
    benchmark_quantization,
    # Distributed
    DistributedConfig,
    is_distributed,
    get_world_size,
    get_rank,
    is_main_process,
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    wrap_model_fsdp,
    GradientCheckpointing,
    get_mixed_precision_scaler,
    get_mixed_precision_context,
    create_distributed_dataloader,
    all_reduce_tensor,
    all_gather_tensors,
    sync_context,
    print_on_main,
    log_on_main,
)

from .utils.onnx_export import (
    ONNXExportConfig,
    export_to_onnx,
    load_onnx_model,
    benchmark_onnx_inference,
)

from .utils.speculative import (
    SpeculativeConfig,
    SpeculativeDecoder,
    create_speculative_decoder,
)

from .initialization import (
    InitializationConfig,
    init_weights_small,
    init_weights_megatron,
    init_weights_deepseek,
    get_initialization_fn,
    reinitialize_layer,
)

__version__ = "0.1.0"

__all__ = [
    # Main model
    "AgentL",
    "AgentConfig",
    # Pre-configured variants
    "agent_debug",
    "agent_tiny",
    "agent_1b",
    "agent_3b",
    "agent_10b",
    "agent_50b",
    "agent_100b",
    "agent_small",
    "agent_medium",
    "agent_large",
    "agent_xl",
    "agent_xxl",
    # Layers
    "RMSNorm",
    "precompute_rope_freqs",
    "apply_rope",
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
    # Checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "export_model",
    "load_pretrained",
    # Profiling
    "ProfileResult",
    "ExpertStats",
    "ACTStats",
    "profile_forward",
    "profile_generation",
    "analyze_expert_utilization",
    "analyze_act_halting",
    "benchmark_depth_scaling",
    "print_profile_report",
    "print_expert_report",
    # Generation
    "GenerationConfig",
    "generate_advanced",
    "beam_search",
    "generate_with_depth_schedule",
    "sample_next_token",
    "apply_top_p_filtering",
    "apply_repetition_penalty",
    # Logging
    "get_logger",
    "setup_logger",
    "TrainingLogger",
    "InferenceLogger",
    "log_model_summary",
    "log_memory_usage",
    # Utils - Visualization
    "VisualizationData",
    "AttentionCapture",
    "ExpertRoutingTracker",
    "ACTHaltingTracker",
    "visualize_attention",
    "visualize_expert_routing",
    "visualize_act_halting",
    "visualize_hidden_evolution",
    "create_model_report",
    # Utils - Quantization
    "QuantizationConfig",
    "quantize_model_dynamic",
    "quantize_model_static",
    "quantize_kv_cache",
    "dequantize_kv_cache",
    "get_model_size_mb",
    "benchmark_quantization",
    # Utils - Distributed
    "DistributedConfig",
    "is_distributed",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "setup_distributed",
    "cleanup_distributed",
    "wrap_model_ddp",
    "wrap_model_fsdp",
    "GradientCheckpointing",
    "get_mixed_precision_scaler",
    "get_mixed_precision_context",
    "create_distributed_dataloader",
    "all_reduce_tensor",
    "all_gather_tensors",
    "sync_context",
    "print_on_main",
    "log_on_main",
    # ONNX Export
    "ONNXExportConfig",
    "export_to_onnx",
    "load_onnx_model",
    "benchmark_onnx_inference",
    # Speculative Decoding
    "SpeculativeConfig",
    "SpeculativeDecoder",
    "create_speculative_decoder",
]
