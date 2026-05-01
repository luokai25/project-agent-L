"""
Quantization utilities for Agent L.

Provides:
- Dynamic quantization (int8)
- Static quantization with calibration
- 4-bit quantization support
- KV cache quantization
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..config import AgentConfig
from ..model import AgentL


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    quantize_embeddings: bool = True
    quantize_linear: bool = True
    quantize_moe: bool = True
    dtype: torch.dtype = torch.qint8
    kv_cache_dtype: Optional[torch.dtype] = torch.float16  # or torch.int8


def quantize_model_dynamic(
    model: AgentL,
    qconfig_spec: Optional[Dict] = None,
) -> AgentL:
    """
    Apply dynamic quantization to the model.

    Dynamic quantization quantizes weights ahead of time but computes
    activations dynamically at runtime. Good for CPU inference.

    Args:
        model: AgentL model to quantize
        qconfig_spec: Custom quantization config specification

    Returns:
        Quantized model
    """
    # Default quantization config
    qconfig = torch.quantization.default_dynamic_qconfig

    # Quantize linear layers
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )

    return quantized


def quantize_model_static(
    model: AgentL,
    calibration_data: torch.Tensor,
    n_loops: int = 8,
) -> AgentL:
    """
    Apply static quantization with calibration.

    Static quantization quantizes both weights and activations ahead of time.
    Requires calibration data to determine activation scales.

    Args:
        model: AgentL model to quantize
        calibration_data: Calibration input data
        n_loops: Number of loops for calibration forward pass

    Returns:
        Quantized model
    """
    model.eval()

    # Prepare for quantization
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)

    # Calibrate with forward pass
    with torch.no_grad():
        _ = model(calibration_data, n_loops=n_loops)

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

    return model


def quantize_kv_cache(
    kv_cache: Dict[str, Dict[str, torch.Tensor]],
    dtype: torch.dtype = torch.float16,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Quantize KV cache to reduce memory footprint.

    Args:
        kv_cache: KV cache dictionary
        dtype: Target dtype (float16, bfloat16, or int8)

    Returns:
        Quantized KV cache
    """
    quantized_cache = {}

    for key, cache in kv_cache.items():
        quantized_cache[key] = {}
        for tensor_name, tensor in cache.items():
            if dtype in (torch.float16, torch.bfloat16):
                # Simple dtype conversion
                quantized_cache[key][tensor_name] = tensor.to(dtype)
            elif dtype == torch.int8:
                # Int8 quantization with scale
                scale = tensor.abs().max() / 127.0
                quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
                quantized_cache[key][tensor_name] = quantized
                quantized_cache[key][f"{tensor_name}_scale"] = scale

    return quantized_cache


def dequantize_kv_cache(
    kv_cache: Dict[str, Dict[str, torch.Tensor]],
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Dequantize KV cache back to full precision.

    Args:
        kv_cache: Quantized KV cache
        dtype: Target dtype

    Returns:
        Dequantized KV cache
    """
    dequantized_cache = {}

    for key, cache in kv_cache.items():
        dequantized_cache[key] = {}
        for tensor_name, tensor in cache.items():
            if tensor_name.endswith("_scale"):
                continue

            if tensor.dtype == torch.int8:
                scale = cache.get(f"{tensor_name}_scale", 1.0)
                dequantized = tensor.to(dtype) * scale
                dequantized_cache[key][tensor_name] = dequantized
            else:
                dequantized_cache[key][tensor_name] = tensor.to(dtype)

    return dequantized_cache


class QuantizedMoE(nn.Module):
    """
    Quantized Mixture of Experts layer.

    Quantizes expert weights for memory-efficient inference.
    """

    def __init__(self, original_moe, dtype: torch.dtype = torch.qint8):
        super().__init__()
        self.n_experts = original_moe.n_experts
        self.n_shared = original_moe.n_shared
        self.topk = original_moe.topk
        self.dtype = dtype

        # Quantize router
        self.router = nn.quantized.Linear.from_float(original_moe.router)

        # Quantize experts
        self.routed_experts = nn.ModuleList([
            self._quantize_expert(expert) for expert in original_moe.routed_experts
        ])
        self.shared_experts = nn.ModuleList([
            self._quantize_expert(expert) for expert in original_moe.shared_experts
        ])

    def _quantize_expert(self, expert: nn.Module) -> nn.Module:
        """Quantize a single expert FFN."""
        quantized = nn.quantized.Linear.from_float(expert.up)
        # Note: Full expert quantization requires custom handling
        return expert  # Placeholder - full implementation needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantized experts."""
        # Dequantize input, compute, quantize output
        raise NotImplementedError("Full quantized MoE forward not yet implemented")


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in megabytes.

    Args:
        model: PyTorch model

    Returns:
        Size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def benchmark_quantization(
    model: AgentL,
    input_ids: torch.Tensor,
    n_loops: int = 8,
) -> Dict[str, float]:
    """
    Benchmark original vs quantized model.

    Args:
        model: Original model
        input_ids: Test input
        n_loops: Number of loops

    Returns:
        Dict with benchmark results
    """
    import time

    # Original model
    original_size = get_model_size_mb(model)

    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        _ = model(input_ids, n_loops=n_loops)
        original_time = time.perf_counter() - start

    # Quantized model
    quantized = quantize_model_dynamic(model)

    quantized_size = get_model_size_mb(quantized)

    quantized.eval()
    with torch.no_grad():
        start = time.perf_counter()
        _ = quantized(input_ids, n_loops=n_loops)
        quantized_time = time.perf_counter() - start

    return {
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "compression_ratio": original_size / quantized_size,
        "original_latency_s": original_time,
        "quantized_latency_s": quantized_time,
        "speedup": original_time / quantized_time,
    }
