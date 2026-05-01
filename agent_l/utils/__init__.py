"""
Utilities for Agent L.

This package provides:
- visualization: Attention, expert routing, ACT halting visualization
- quantization: Dynamic and static quantization for inference
- distributed: DDP and FSDP utilities for multi-GPU training
"""

from .visualization import (
    VisualizationData,
    AttentionCapture,
    ExpertRoutingTracker,
    ACTHaltingTracker,
    visualize_attention,
    visualize_expert_routing,
    visualize_act_halting,
    visualize_hidden_evolution,
    create_model_report,
)

from .quantization import (
    QuantizationConfig,
    quantize_model_dynamic,
    quantize_model_static,
    quantize_kv_cache,
    dequantize_kv_cache,
    get_model_size_mb,
    benchmark_quantization,
)

from .distributed import (
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

__all__ = [
    # Visualization
    "VisualizationData",
    "AttentionCapture",
    "ExpertRoutingTracker",
    "ACTHaltingTracker",
    "visualize_attention",
    "visualize_expert_routing",
    "visualize_act_halting",
    "visualize_hidden_evolution",
    "create_model_report",
    # Quantization
    "QuantizationConfig",
    "quantize_model_dynamic",
    "quantize_model_static",
    "quantize_kv_cache",
    "dequantize_kv_cache",
    "get_model_size_mb",
    "benchmark_quantization",
    # Distributed
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
]
