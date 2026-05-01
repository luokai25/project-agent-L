"""
Profiling and benchmarking utilities for Agent L.

Provides tools for:
- Memory usage profiling
- Throughput benchmarking
- Expert utilization analysis
- ACT halting statistics
- Inference latency measurement
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .model import AgentL
from .config import AgentConfig


@dataclass
class ProfileResult:
    """Results from profiling a model forward pass."""

    batch_size: int
    seq_len: int
    n_loops: int
    latency_ms: float
    memory_mb: float
    tokens_per_second: float
    device: str


@dataclass
class ExpertStats:
    """Statistics for MoE expert utilization."""

    expert_id: int
    num_activations: int
    total_tokens: int
    utilization: float
    avg_gate_score: float


@dataclass
class ACTStats:
    """Statistics for ACT halting behavior."""

    avg_loops: float
    min_loops: int
    max_loops: int
    early_halt_ratio: float


def profile_forward(
    model: AgentL,
    batch_size: int = 1,
    seq_len: int = 128,
    n_loops: int = 8,
    warmup_steps: int = 3,
    measure_steps: int = 10,
    device: Optional[torch.device] = None,
) -> ProfileResult:
    """
    Profile a forward pass for latency and memory.

    Args:
        model: AgentL model instance
        batch_size: Batch size for profiling
        seq_len: Sequence length for profiling
        n_loops: Number of recurrent loops
        warmup_steps: Number of warmup iterations
        measure_steps: Number of measurement iterations
        device: Device to run on

    Returns:
        ProfileResult with timing and memory stats
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Clear cache and measure baseline memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device) / 1024**2
    else:
        baseline_memory = 0.0

    input_ids = torch.randint(0, model.cfg.vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(input_ids, n_loops=n_loops)

    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(measure_steps):
            start = time.perf_counter()
            _ = model(input_ids, n_loops=n_loops)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    avg_latency = sum(latencies) / len(latencies)

    # Measure peak memory
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2 - baseline_memory
    else:
        peak_memory = 0.0

    tokens_per_second = (batch_size * seq_len) / (avg_latency / 1000)

    return ProfileResult(
        batch_size=batch_size,
        seq_len=seq_len,
        n_loops=n_loops,
        latency_ms=avg_latency,
        memory_mb=peak_memory,
        tokens_per_second=tokens_per_second,
        device=str(device),
    )


def profile_generation(
    model: AgentL,
    prompt_len: int = 32,
    gen_len: int = 64,
    batch_size: int = 1,
    n_loops: int = 8,
    warmup_steps: int = 1,
    measure_steps: int = 3,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Profile autoregressive generation.

    Args:
        model: AgentL model instance
        prompt_len: Prompt length
        gen_len: Number of tokens to generate
        batch_size: Batch size
        n_loops: Number of recurrent loops per decode step
        warmup_steps: Warmup iterations
        measure_steps: Measurement iterations
        device: Device to run on

    Returns:
        Dict with generation stats
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    input_ids = torch.randint(0, model.cfg.vocab_size, (batch_size, prompt_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model.generate(input_ids, max_new_tokens=gen_len, n_loops=n_loops)

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(measure_steps):
            start = time.perf_counter()
            _ = model.generate(input_ids, max_new_tokens=gen_len, n_loops=n_loops)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    avg_latency = sum(latencies) / len(latencies)
    tokens_per_second = (batch_size * gen_len) / (avg_latency / 1000)
    ms_per_token = avg_latency / gen_len

    return {
        "total_latency_ms": avg_latency,
        "ms_per_token": ms_per_token,
        "tokens_per_second": tokens_per_second,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "batch_size": batch_size,
    }


def analyze_expert_utilization(
    model: AgentL,
    input_ids: torch.Tensor,
    n_loops: int = 8,
) -> List[ExpertStats]:
    """
    Analyze MoE expert utilization for a forward pass.

    This requires hooking into the MoE router to track which experts
    are selected and their gating scores.

    Args:
        model: AgentL model instance
        input_ids: Input token IDs
        n_loops: Number of recurrent loops

    Returns:
        List of ExpertStats for each expert
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    expert_activations: Dict[int, int] = defaultdict(int)
    expert_scores: Dict[int, List[float]] = defaultdict(list)
    total_tokens = 0

    def hook_fn(module, inp, out):
        nonlocal total_tokens
        # Get router logits
        x = inp[0]
        B, T, D = x.shape
        total_tokens += B * T

        # Get routing decisions (this is approximate since we can't easily
        # access the exact routing from outside)
        logits = module.router(x.view(-1, D))
        scores = torch.softmax(logits, dim=-1)
        topk_idx = (logits + module.router_bias).topk(module.topk, dim=-1).indices

        for i in range(module.topk):
            for eid in range(module.n_experts):
                mask = topk_idx[:, i] == eid
                if mask.any():
                    expert_activations[eid] += mask.sum().item()
                    expert_scores[eid].append(scores[mask, eid].mean().item())

    # Register hook on MoE layer
    moe = model.recurrent.block.ffn
    handle = moe.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(input_ids, n_loops=n_loops)

    handle.remove()

    stats = []
    for eid in range(model.cfg.n_experts):
        stats.append(
            ExpertStats(
                expert_id=eid,
                num_activations=expert_activations.get(eid, 0),
                total_tokens=total_tokens,
                utilization=expert_activations.get(eid, 0) / max(total_tokens, 1),
                avg_gate_score=sum(expert_scores.get(eid, [0])) / max(len(expert_scores.get(eid, [])), 1),
            )
        )

    return stats


def analyze_act_halting(
    model: AgentL,
    input_ids: torch.Tensor,
    n_loops: int = 8,
) -> ACTStats:
    """
    Analyze ACT halting behavior.

    Requires modifying the recurrent block to track halting statistics.
    This is a simplified version that estimates based on the model state.

    Args:
        model: AgentL model instance
        input_ids: Input token IDs
        n_loops: Number of recurrent loops

    Returns:
        ACTStats with halting statistics
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Track actual loops per position
    actual_loops = []

    # We need to modify the recurrent block to track this
    # For now, return an estimate
    with torch.no_grad():
        # Run forward pass and track halting
        # This requires access to the recurrent block internals
        # For a proper implementation, we'd add a return_halting_stats parameter
        _ = model(input_ids, n_loops=n_loops)

    # Placeholder - in practice you'd track this during the forward pass
    return ACTStats(
        avg_loops=float(n_loops),
        min_loops=n_loops,
        max_loops=n_loops,
        early_halt_ratio=0.0,
    )


def benchmark_depth_scaling(
    model: AgentL,
    batch_size: int = 1,
    seq_len: int = 128,
    loop_counts: List[int] = [1, 2, 4, 8, 16],
    device: Optional[torch.device] = None,
) -> Dict[int, ProfileResult]:
    """
    Benchmark latency across different recurrent depths.

    Args:
        model: AgentL model instance
        batch_size: Batch size
        seq_len: Sequence length
        loop_counts: List of loop counts to benchmark
        device: Device to run on

    Returns:
        Dict mapping loop count to ProfileResult
    """
    results = {}
    for n_loops in loop_counts:
        results[n_loops] = profile_forward(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            n_loops=n_loops,
            device=device,
        )
    return results


def print_profile_report(result: ProfileResult) -> None:
    """Print a formatted profiling report."""
    print("=" * 60)
    print("Agent L Profile Report")
    print("=" * 60)
    print(f"Batch size:      {result.batch_size}")
    print(f"Sequence length: {result.seq_len}")
    print(f"Loop depth:      {result.n_loops}")
    print(f"Device:          {result.device}")
    print("-" * 60)
    print(f"Latency:         {result.latency_ms:.2f} ms")
    print(f"Peak memory:     {result.memory_mb:.2f} MB")
    print(f"Throughput:      {result.tokens_per_second:.1f} tokens/sec")
    print("=" * 60)


def print_expert_report(stats: List[ExpertStats], top_k: int = 10) -> None:
    """Print a formatted expert utilization report."""
    print("=" * 60)
    print("MoE Expert Utilization Report")
    print("=" * 60)

    # Sort by utilization
    sorted_stats = sorted(stats, key=lambda s: s.utilization, reverse=True)

    print(f"\nTop {top_k} most utilized experts:")
    print(f"{'Expert':>8} {'Activations':>12} {'Utilization':>12} {'Avg Score':>12}")
    print("-" * 52)
    for s in sorted_stats[:top_k]:
        print(f"{s.expert_id:>8} {s.num_activations:>12} {s.utilization:>12.2%} {s.avg_gate_score:>12.4f}")

    # Check for load imbalance
    utilizations = [s.utilization for s in stats]
    avg_util = sum(utilizations) / len(utilizations)
    max_util = max(utilizations)
    min_util = min(utilizations)

    print(f"\nLoad balance summary:")
    print(f"  Average utilization: {avg_util:.2%}")
    print(f"  Max utilization:     {max_util:.2%}")
    print(f"  Min utilization:     {min_util:.2%}")
    print(f"  Ratio (max/min):     {max_util / max(min_util, 1e-6):.2f}x")
    print("=" * 60)
