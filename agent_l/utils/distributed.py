"""
Distributed training utilities for Agent L.

Provides:
- Distributed Data Parallel (DDP) setup
- Fully Sharded Data Parallel (FSDP) configuration
- Gradient checkpointing
- Mixed precision training
- Distributed sampling
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ..config import AgentConfig
from ..model import AgentL


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    backend: str = "nccl"  # or "gloo" for CPU
    init_method: str = "env://"
    world_size: int = -1  # Set from environment
    rank: int = -1  # Set from environment
    local_rank: int = -1  # Set from environment

    # FSDP settings
    use_fsdp: bool = False
    fsdp_min_params: int = 1_000_000
    fsdp_mixed_precision: bool = True

    # Gradient checkpointing
    gradient_checkpointing: bool = True

    # Mixed precision
    mixed_precision: bool = True
    fp16: bool = False
    bf16: bool = True

    # Gradient accumulation
    gradient_accumulation_steps: int = 1


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get world size (number of processes)."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get current process rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return get_rank() == 0


def setup_distributed(cfg: DistributedConfig) -> None:
    """
    Initialize distributed training.

    Args:
        cfg: Distributed configuration
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.local_rank = int(os.environ["LOCAL_RANK"])

    if cfg.world_size == -1:
        return  # Not distributed

    dist.init_process_group(
        backend=cfg.backend,
        init_method=cfg.init_method,
        world_size=cfg.world_size,
        rank=cfg.rank,
    )

    torch.cuda.set_device(cfg.local_rank)


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: AgentL,
    cfg: DistributedConfig,
    find_unused_parameters: bool = False,
) -> DDP:
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: AgentL model
        cfg: Distributed configuration
        find_unused_parameters: Whether to find unused parameters

    Returns:
        DDP-wrapped model
    """
    model = model.to(torch.device(f"cuda:{cfg.local_rank}"))

    return DDP(
        model,
        device_ids=[cfg.local_rank],
        output_device=cfg.local_rank,
        find_unused_parameters=find_unused_parameters,
    )


def wrap_model_fsdp(
    model: AgentL,
    cfg: DistributedConfig,
) -> "FSDP":
    """
    Wrap model with Fully Sharded Data Parallel.

    Args:
        model: AgentL model
        cfg: Distributed configuration

    Returns:
        FSDP-wrapped model
    """
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
            CPUOffload,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    except ImportError:
        raise ImportError("FSDP requires PyTorch 2.0+")

    # Define mixed precision policy
    if cfg.fsdp_mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
    else:
        mp_policy = None

    # Auto-wrap policy for transformer blocks
    from ..recurrent import TransformerBlock, RecurrentBlock

    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_set={TransformerBlock, RecurrentBlock},
        min_num_params=cfg.fsdp_min_params,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device(),
    )

    return model


class GradientCheckpointing:
    """
    Gradient checkpointing for memory-efficient training.

    Usage:
        with GradientCheckpointing(model, enabled=True):
            output = model(input_ids)
    """

    def __init__(self, model: nn.Module, enabled: bool = True):
        self.model = model
        self.enabled = enabled
        self._original_forward = None

    def __enter__(self):
        if self.enabled and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        return self

    def __exit__(self, *args):
        if self.enabled and hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()


def get_mixed_precision_scaler(
    cfg: DistributedConfig,
) -> Optional[torch.cuda.amp.GradScaler]:
    """
    Get gradient scaler for mixed precision training.

    Args:
        cfg: Distributed configuration

    Returns:
        GradScaler or None
    """
    if not cfg.mixed_precision:
        return None

    if cfg.fp16:
        return torch.cuda.amp.GradScaler()
    elif cfg.bf16:
        # BF16 doesn't need scaling
        return None

    return None


def get_mixed_precision_context(
    cfg: DistributedConfig,
) -> contextmanager:
    """
    Get mixed precision context manager.

    Args:
        cfg: Distributed configuration

    Returns:
        Context manager for mixed precision
    """
    if cfg.bf16:
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif cfg.fp16:
        return torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        return torch.cuda.amp.autocast(enabled=False)


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    cfg: DistributedConfig,
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader with distributed sampler.

    Args:
        dataset: Dataset instance
        batch_size: Per-GPU batch size
        cfg: Distributed configuration
        **kwargs: Additional DataLoader args

    Returns:
        DataLoader with distributed sampler
    """
    if is_distributed():
        sampler = DistributedSampler(
            dataset,
            num_replicas=cfg.world_size,
            rank=cfg.rank,
            shuffle=kwargs.get("shuffle", True),
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = kwargs.get("shuffle", True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=kwargs.get("num_workers", 4),
        pin_memory=kwargs.get("pin_memory", True),
    )


def all_reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor across all processes.

    Args:
        tensor: Tensor to reduce

    Returns:
        Reduced tensor
    """
    if not is_distributed():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(get_world_size())
    return tensor


def all_gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Gather tensors from all processes.

    Args:
        tensor: Tensor to gather

    Returns:
        List of tensors from all processes
    """
    if not is_distributed():
        return [tensor]

    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)
    return gathered


@contextmanager
def sync_context():
    """
    Context manager for synchronized operations.

    Ensures all processes reach this point before continuing.
    """
    if is_distributed():
        dist.barrier()
    yield
    if is_distributed():
        dist.barrier()


def print_on_main(msg: str) -> None:
    """Print only on main process."""
    if is_main_process():
        print(msg)


def log_on_main(logger, msg: str, level: str = "info") -> None:
    """Log only on main process."""
    if is_main_process():
        getattr(logger, level)(msg)
