"""
ONNX export utilities for Agent L.

Provides:
- Model export to ONNX format
- ONNX runtime inference
- Optimization passes
- Quantization support for ONNX models
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

from ..config import AgentConfig
from ..model import AgentL


@dataclass
class ONNXExportConfig:
    """Configuration for ONNX export."""
    
    opset_version: int = 17
    dynamic_batch: bool = True
    dynamic_sequence: bool = True
    simplify: bool = True
    optimize: bool = True
    output_names: List[str] = None
    
    def __post_init__(self):
        if self.output_names is None:
            self.output_names = ["logits"]


def export_to_onnx(
    model: AgentL,
    output_path: str,
    config: ONNXExportConfig = None,
    sample_input: torch.Tensor = None,
    n_loops: int = 8,
) -> Dict[str, Any]:
    """
    Export Agent L model to ONNX format.
    
    Args:
        model: AgentL model instance
        output_path: Path to save ONNX model
        config: Export configuration
        sample_input: Sample input for tracing (auto-generated if None)
        n_loops: Number of recurrent loops for export
    
    Returns:
        Dict with export metadata
    """
    config = config or ONNXExportConfig()
    model.eval()
    
    # Create sample input if not provided
    if sample_input is None:
        sample_input = torch.randint(
            0, model.cfg.vocab_size, (1, 16)
        )
    
    # Dynamic axes configuration
    dynamic_axes = {}
    if config.dynamic_batch:
        dynamic_axes["input_ids"] = {0: "batch_size"}
        dynamic_axes["logits"] = {0: "batch_size"}
    if config.dynamic_sequence:
        if "input_ids" not in dynamic_axes:
            dynamic_axes["input_ids"] = {}
        dynamic_axes["input_ids"][1] = "sequence_length"
        if "logits" not in dynamic_axes:
            dynamic_axes["logits"] = {}
        dynamic_axes["logits"][1] = "sequence_length"
    
    # Export
    torch.onnx.export(
        model,
        (sample_input, n_loops),
        output_path,
        input_names=["input_ids"],
        output_names=config.output_names,
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        opset_version=config.opset_version,
        do_constant_folding=True,
        export_params=True,
    )
    
    result = {
        "path": output_path,
        "opset_version": config.opset_version,
        "dynamic_batch": config.dynamic_batch,
        "dynamic_sequence": config.dynamic_sequence,
    }
    
    # Optional: simplify and optimize
    if config.simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            onnx_model = onnx.load(output_path)
            simplified, check = onnx_simplify(onnx_model)
            onnx.save(simplified, output_path)
            result["simplified"] = True
        except ImportError:
            result["simplified"] = False
    
    return result


def load_onnx_model(
    model_path: str,
    provider: str = "CUDAExecutionProvider",
) -> Any:
    """
    Load ONNX model for inference.
    
    Args:
        model_path: Path to ONNX model
        provider: Execution provider ("CUDAExecutionProvider" or "CPUExecutionProvider")
    
    Returns:
        ONNX Runtime inference session
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ONNX inference. "
            "Install with: pip install onnxruntime-gpu"
        )
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=[provider],
    )
    
    return session


def benchmark_onnx_inference(
    session: Any,
    input_ids: torch.Tensor,
    n_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """
    Benchmark ONNX model inference.
    
    Args:
        session: ONNX Runtime session
        input_ids: Input tensor
        n_runs: Number of benchmark runs
        warmup: Number of warmup runs
    
    Returns:
        Dict with timing statistics
    """
    import time
    
    input_numpy = input_ids.numpy()
    
    # Warmup
    for _ in range(warmup):
        session.run(None, {"input_ids": input_numpy})
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {"input_ids": input_numpy})
        end = time.perf_counter()
        times.append(end - start)
    
    import numpy as np
    times = np.array(times)
    
    return {
        "mean_ms": times.mean() * 1000,
        "std_ms": times.std() * 1000,
        "p50_ms": np.percentile(times, 50) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000,
        "p99_ms": np.percentile(times, 99) * 1000,
    }


__all__ = [
    "ONNXExportConfig",
    "export_to_onnx",
    "load_onnx_model",
    "benchmark_onnx_inference",
]
