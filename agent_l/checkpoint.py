"""
Checkpointing utilities for Agent L.

Provides save/load functionality with:
- Model state dict serialization
- Optimizer state preservation
- Training metadata (step, epoch, loss)
- Config serialization
- Automatic versioning
"""

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .config import AgentConfig
from .model import AgentL


def save_checkpoint(
    model: AgentL,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    step: int = 0,
    epoch: int = 0,
    loss: float = 0.0,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a training checkpoint.

    Args:
        model: AgentL model instance
        optimizer: Optional optimizer to save state
        path: Directory path for checkpoint
        step: Current training step
        epoch: Current epoch
        loss: Current loss value
        extra_metadata: Additional metadata to store

    Returns:
        Path to saved checkpoint file
    """
    os.makedirs(path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"checkpoint_{step}_{timestamp}.pt"
    checkpoint_path = os.path.join(path, filename)

    checkpoint = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "config": asdict(model.cfg),
        "timestamp": timestamp,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if extra_metadata is not None:
        checkpoint["metadata"] = extra_metadata

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str,
    model: Optional[AgentL] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    load_model: bool = True,
) -> Tuple[Optional[AgentL], Optional[torch.optim.Optimizer], Dict[str, Any]]:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint file
        model: Existing model to load weights into (optional)
        optimizer: Existing optimizer to load state into (optional)
        device: Device to load tensors to
        load_model: Whether to load model weights

    Returns:
        Tuple of (model, optimizer, metadata dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)

    metadata = {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0.0),
        "timestamp": checkpoint.get("timestamp", ""),
    }

    if load_model:
        if model is None:
            config = AgentConfig(**checkpoint["config"])
            model = AgentL(config)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, metadata


def export_model(
    model: AgentL,
    path: str,
    format: str = "safetensors",
) -> str:
    """
    Export model weights for inference.

    Args:
        model: AgentL model instance
        path: Output directory path
        format: Export format ("safetensors" or "pt")

    Returns:
        Path to exported model file
    """
    os.makedirs(path, exist_ok=True)

    if format == "safetensors":
        try:
            from safetensors.torch import save_file

            filename = "model.safetensors"
            filepath = os.path.join(path, filename)
            save_file(model.state_dict(), filepath)
        except ImportError:
            print("safetensors not installed, falling back to .pt format")
            format = "pt"

    if format == "pt":
        filename = "model.pt"
        filepath = os.path.join(path, filename)
        torch.save(model.state_dict(), filepath)

    # Save config alongside
    config_path = os.path.join(path, "config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(model.cfg), f, indent=2)

    return filepath


def load_pretrained(
    path: str,
    device: Optional[torch.device] = None,
) -> AgentL:
    """
    Load a pretrained model from directory.

    Args:
        path: Directory containing model weights and config.json
        device: Device to load model to

    Returns:
        AgentL model instance with loaded weights
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config_path = os.path.join(path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = AgentConfig(**config_dict)

    # Create model
    model = AgentL(config)

    # Load weights
    safetensors_path = os.path.join(path, "model.safetensors")
    pt_path = os.path.join(path, "model.pt")

    if os.path.exists(safetensors_path):
        try:
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path)
        except ImportError:
            raise ImportError("safetensors required to load .safetensors files")
    elif os.path.exists(pt_path):
        state_dict = torch.load(pt_path, map_location=device)
    else:
        raise FileNotFoundError(f"No model weights found in {path}")

    model.load_state_dict(state_dict)
    model.to(device)

    return model
