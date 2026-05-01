"""
Logging utilities for Agent L.

Provides configurable logging with:
- Console and file handlers
- Different log levels per module
- Structured JSON logging option
- Progress tracking for training
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Module-level logger
_logger: Optional[logging.Logger] = None


def get_logger(name: str = "agent_l") -> logging.Logger:
    """Get or create a logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logger(name)
    return _logger


def setup_logger(
    name: str = "agent_l",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Setup a configurable logger.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        json_format: Use JSON structured logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class JsonFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "data"):
            log_data["data"] = record.data

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TrainingLogger:
    """
    Context-aware logger for training loops.

    Tracks step, epoch, loss, and other metrics with optional
    progress bar-style output.
    """

    def __init__(
        self,
        name: str = "agent_l.training",
        log_interval: int = 10,
        log_file: Optional[str] = None,
    ):
        self.logger = setup_logger(name, log_file=log_file)
        self.log_interval = log_interval
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.metrics: Dict[str, float] = {}

    def log_step(
        self,
        loss: float,
        lr: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log a training step."""
        self.step += 1

        if self.step % self.log_interval == 0:
            msg = f"Step {self.step} | Loss: {loss:.4f} | LR: {lr:.2e}"
            if metrics:
                for k, v in metrics.items():
                    msg += f" | {k}: {v:.4f}"
            self.logger.info(msg)

            if metrics:
                self.metrics.update(metrics)

    def log_epoch(
        self,
        avg_loss: float,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log epoch completion."""
        self.epoch += 1

        msg = f"Epoch {self.epoch} | Avg Loss: {avg_loss:.4f}"
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            msg += " | Best!"

        self.logger.info(msg)

    def log_checkpoint(self, path: str, step: int) -> None:
        """Log checkpoint save."""
        self.logger.info(f"Checkpoint saved: {path} (step {step})")

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration."""
        self.logger.info("Configuration:")
        for k, v in config.items():
            self.logger.info(f"  {k}: {v}")


class InferenceLogger:
    """Logger for inference and generation."""

    def __init__(self, name: str = "agent_l.inference"):
        self.logger = setup_logger(name)

    def log_generation_start(
        self,
        prompt_len: int,
        max_tokens: int,
        n_loops: int,
        temperature: float,
    ) -> None:
        """Log generation start."""
        self.logger.info(
            f"Generating: prompt={prompt_len}, max={max_tokens}, "
            f"loops={n_loops}, temp={temperature}"
        )

    def log_generation_end(
        self,
        total_tokens: int,
        latency_ms: float,
        tokens_per_sec: float,
    ) -> None:
        """Log generation completion."""
        self.logger.info(
            f"Generated {total_tokens} tokens in {latency_ms:.1f}ms "
            f"({tokens_per_sec:.1f} tok/s)"
        )

    def log_depth_change(self, step: int, n_loops: int) -> None:
        """Log depth scheduling change."""
        self.logger.debug(f"Step {step}: depth = {n_loops} loops")


def log_model_summary(model) -> None:
    """Log model architecture summary."""
    logger = get_logger()

    params = model.count_parameters()
    spectral_radius = model.get_spectral_radius()

    logger.info("Model Summary:")
    logger.info(f"  Total parameters: {params['total']:,}")
    logger.info(f"  Embedding: {params['embed']:,}")
    logger.info(f"  Prelude: {params['prelude']:,}")
    logger.info(f"  Recurrent: {params['recurrent']:,}")
    logger.info(f"  Coda: {params['coda']:,}")
    logger.info(f"  Spectral radius: {spectral_radius:.4f}")

    if spectral_radius < 1.0:
        logger.info("  ✓ Model is stable (ρ < 1)")
    else:
        logger.warning("  ⚠ Model may be unstable (ρ >= 1)")


def log_memory_usage() -> None:
    """Log current GPU memory usage."""
    logger = get_logger()

    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        else:
            logger.debug("CUDA not available")
    except ImportError:
        logger.debug("PyTorch not available for memory logging")


def set_log_level(level: int) -> None:
    """Set the global log level."""
    logger = get_logger()
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
