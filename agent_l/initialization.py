"""
Weight initialization strategies for Agent L.

Provides various initialization schemes optimized for:
- Recurrent depth architectures
- Mixture of Experts
- Multi-head attention
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Optional, List

from .config import AgentConfig


@dataclass
class InitializationConfig:
    """Configuration for weight initialization."""
    method: str = "small"  # "small", "megatron", "deepseek", "kaiming", "xavier"
    std: float = 0.02
    hidden_std: Optional[float] = None  # Defaults to std if None
    embedding_std: Optional[float] = None
    moe_std: Optional[float] = None
    attention_std: Optional[float] = None
    lora_init: str = "kaiming"  # For LoRA adapters
    rezero_init: float = 0.0  # For ReZero-style residual scaling


def init_weights_small(model: nn.Module, std: float = 0.02) -> None:
    """
    Small initialization (GPT-2 style).
    
    Initializes all linear and embedding weights from N(0, std^2).
    Good for small to medium models.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)


def init_weights_megatron(model: nn.Module, config: AgentConfig) -> None:
    """
    Megatron-style initialization (larger std for embeddings).
    
    Uses larger std for embeddings and smaller std for hidden layers.
    Better for large models with deep architectures.
    """
    embed_std = config.dim ** -0.5  # ~0.022 for dim=2048
    hidden_std = config.dim ** -0.5 / math.sqrt(2 * config.prelude_layers + config.max_loop_iters)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "embed" in name or "head" in name:
                nn.init.normal_(module.weight, mean=0.0, std=embed_std)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=hidden_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=embed_std)


def init_weights_deepseek(model: nn.Module, config: AgentConfig) -> None:
    """
    DeepSeek-style initialization for MoE models.
    
    Uses different std for expert weights vs dense layers.
    """
    base_std = 0.02
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Expert weights get smaller initialization
            if "routed_experts" in name or "shared_experts" in name:
                std = base_std / math.sqrt(config.n_experts_per_tok)
            else:
                std = base_std
            
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=base_std)
    
    # Special initialization for LoRA adapters
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            if "down" in name or "up" in name:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))


def get_initialization_fn(
    method: str = "small",
    std: float = 0.02,
    config: Optional[AgentConfig] = None,
) -> Callable[[nn.Module], None]:
    """
    Get an initialization function by name.
    
    Args:
        method: One of "small", "megatron", "deepseek", "kaiming", "xavier"
        std: Standard deviation for small initialization
        config: Required for megatron and deepseek methods
    
    Returns:
        Initialization function that modifies model in-place
    """
    if method == "small":
        return lambda m: init_weights_small(m, std)
    elif method == "megatron":
        if config is None:
            raise ValueError("config required for megatron initialization")
        return lambda m: init_weights_megatron(m, config)
    elif method == "deepseek":
        if config is None:
            raise ValueError("config required for deepseek initialization")
        return lambda m: init_weights_deepseek(m, config)
    elif method == "kaiming":
        return lambda m: _init_kaiming(m)
    elif method == "xavier":
        return lambda m: _init_xavier(m)
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def _init_kaiming(model: nn.Module) -> None:
    """Kaiming/He initialization for ReLU/SiLU activations."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _init_xavier(model: nn.Module) -> None:
    """Xavier/Glorot initialization for tanh/sigmoid."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def reinitialize_layer(
    model: nn.Module,
    layer_name: str,
    method: str = "small",
    std: float = 0.02,
) -> None:
    """
    Reinitialize a specific layer.
    
    Useful for transfer learning or fixing bad initialization.
    
    Args:
        model: The model
        layer_name: Name of layer to reinitialize (e.g., "recurrent.block.ffn")
        method: Initialization method
        std: Standard deviation
    """
    # Navigate to the layer
    parts = layer_name.split(".")
    layer = model
    for part in parts:
        if hasattr(layer, part):
            layer = getattr(layer, part)
        elif part.isdigit():
            layer = layer[int(part)]
        else:
            raise ValueError(f"Layer {layer_name} not found")
    
    # Reinitialize
    init_fn = get_initialization_fn(method, std)
    init_fn(layer)


def init_recurrent_block(
    model: nn.Module,
    lora_scale: float = 0.01,
    injection_scale: float = 0.1,
) -> None:
    """
    Special initialization for recurrent components.
    
    - LoRA adapters: Small scale to start close to identity
    - LTI injection: Initialized for slow decay (stable recurrence)
    """
    for name, param in model.named_parameters():
        # LoRA: Start small so delta is near zero initially
        if "lora" in name.lower():
            if "B" in name:
                nn.init.normal_(param, mean=0.0, std=lora_scale)
            elif "scale" in name:
                nn.init.ones_(param)
        
        # LTI injection: Initialize for slow decay
        elif "injection.log_A" in name:
            nn.init.constant_(param, math.log(1.0 / injection_scale - 1.0))
        elif "injection.log_dt" in name:
            nn.init.constant_(param, 0.0)


__all__ = [
    "InitializationConfig",
    "init_weights_small",
    "init_weights_megatron",
    "init_weights_deepseek",
    "get_initialization_fn",
    "reinitialize_layer",
    "init_recurrent_block",
]
