"""
Advanced generation strategies for Agent L.

Provides:
- Top-p (nucleus) sampling
- Beam search
- Repetition penalty
- Temperature scheduling
- Stopping criteria
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import AgentL


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 64
    n_loops: int = 8
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    early_stopping: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 2
    min_length: int = 0
    no_repeat_ngram_size: int = 0


class StoppingCriteria(ABC):
    """Base class for stopping criteria."""

    @abstractmethod
    def __call__(self, generated_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        """Return True if generation should stop."""
        pass


class EOSStopping(StoppingCriteria):
    """Stop when all sequences have generated EOS token."""

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def __call__(self, generated_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        return (generated_ids == self.eos_token_id).any(dim=-1).all()


class MaxLengthStopping(StoppingCriteria):
    """Stop when reaching max length."""

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, generated_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        return generated_ids.shape[-1] >= self.max_length


class StoppingCriteriaList:
    """Collection of stopping criteria."""

    def __init__(self, criteria: List[StoppingCriteria]):
        self.criteria = criteria

    def __call__(self, generated_ids: torch.Tensor, scores: torch.Tensor) -> bool:
        return any(criterion(generated_ids, scores) for criterion in self.criteria)


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits.

    Args:
        logits: Logits tensor (B, vocab_size)
        generated_ids: Previously generated token IDs (B, T)
        penalty: Penalty factor (>1.0 penalizes repetition, <1.0 encourages)

    Returns:
        Modified logits
    """
    if penalty == 1.0:
        return logits

    for i in range(generated_ids.shape[0]):
        prev_tokens = generated_ids[i].unique()
        if penalty > 1.0:
            logits[i, prev_tokens] /= penalty
        else:
            logits[i, prev_tokens] *= penalty

    return logits


def apply_top_p_filtering(
    logits: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """
    Apply nucleus (top-p) filtering to logits.

    Args:
        logits: Logits tensor (B, vocab_size)
        top_p: Cumulative probability threshold

    Returns:
        Filtered logits with low-probability tokens set to -inf
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generated_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample next token from logits.

    Args:
        logits: Logits tensor (B, vocab_size)
        temperature: Sampling temperature
        top_k: Number of top tokens to consider (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: Repetition penalty factor
        generated_ids: Previously generated tokens for repetition penalty

    Returns:
        Sampled token IDs (B, 1)
    """
    # Apply temperature
    logits = logits / temperature

    # Apply repetition penalty
    if repetition_penalty != 1.0 and generated_ids is not None:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Apply top-k filtering
    if top_k > 0:
        v, _ = logits.topk(top_k)
        logits[logits < v[:, -1:]] = float("-inf")

    # Apply top-p filtering
    if top_p < 1.0:
        logits = apply_top_p_filtering(logits, top_p)

    # Sample
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


@torch.no_grad()
def generate_advanced(
    model: AgentL,
    input_ids: torch.Tensor,
    config: Optional[GenerationConfig] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    stream_callback: Optional[Callable[[torch.Tensor], None]] = None,
) -> torch.Tensor:
    """
    Advanced generation with multiple decoding strategies.

    Args:
        model: AgentL model instance
        input_ids: Prompt token IDs (B, T)
        config: GenerationConfig with generation parameters
        stopping_criteria: Optional stopping criteria
        stream_callback: Optional callback for streaming generation

    Returns:
        Generated token IDs (B, T + max_new_tokens)
    """
    if config is None:
        config = GenerationConfig()

    device = input_ids.device
    kv_cache: dict = {}
    prompt_len = input_ids.shape[1]

    # Setup stopping criteria
    if stopping_criteria is None:
        stopping_criteria = StoppingCriteriaList([
            MaxLengthStopping(prompt_len + config.max_new_tokens),
            EOSStopping(config.eos_token_id),
        ])

    generated_ids = input_ids.clone()

    for step in range(config.max_new_tokens):
        if step == 0:
            cur_ids = input_ids
            start_pos = 0
        else:
            cur_ids = generated_ids[:, -1:]
            start_pos = prompt_len + step - 1

        # Forward pass
        logits = model.forward(
            cur_ids,
            n_loops=config.n_loops,
            kv_cache=kv_cache,
            start_pos=start_pos,
        )
        logits = logits[:, -1, :]

        # Sample or greedy decode
        if config.do_sample:
            next_token = sample_next_token(
                logits,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                generated_ids=generated_ids,
            )
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # Stream callback
        if stream_callback is not None:
            stream_callback(next_token)

        # Check stopping criteria
        if stopping_criteria(generated_ids, logits):
            break

    return generated_ids


@torch.no_grad()
def beam_search(
    model: AgentL,
    input_ids: torch.Tensor,
    num_beams: int = 4,
    max_new_tokens: int = 64,
    n_loops: int = 8,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    eos_token_id: int = 2,
) -> torch.Tensor:
    """
    Beam search decoding.

    Args:
        model: AgentL model instance
        input_ids: Prompt token IDs (1, T)
        num_beams: Number of beams
        max_new_tokens: Maximum tokens to generate
        n_loops: Number of recurrent loops
        length_penalty: Length penalty for scoring
        early_stopping: Stop when all beams have generated EOS
        eos_token_id: End-of-sequence token ID

    Returns:
        Best generated sequence (1, T + generated_len)
    """
    if input_ids.shape[0] != 1:
        raise ValueError("Beam search expects batch_size=1")

    device = input_ids.device
    batch_size = num_beams

    # Expand input for all beams
    input_ids = input_ids.expand(num_beams, -1)
    prompt_len = input_ids.shape[1]

    # Initialize beam scores
    beam_scores = torch.zeros(num_beams, device=device)
    beam_scores[1:] = float("-inf")  # Only first beam is active initially

    kv_cache: dict = {}
    generated_ids = input_ids.clone()
    done_beams = torch.zeros(num_beams, dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        if step == 0:
            cur_ids = generated_ids
            start_pos = 0
        else:
            cur_ids = generated_ids[:, -1:]
            start_pos = prompt_len + step - 1

        # Forward pass
        logits = model.forward(
            cur_ids,
            n_loops=n_loops,
            kv_cache=kv_cache,
            start_pos=start_pos,
        )
        logits = logits[:, -1, :]

        # Apply length penalty
        current_length = generated_ids.shape[-1]
        scores = F.log_softmax(logits, dim=-1) / (current_length**length_penalty)

        # Add beam scores
        scores = scores + beam_scores.unsqueeze(-1)

        # Flatten and get top-k
        vocab_size = scores.shape[-1]
        scores_flat = scores.view(-1)
        top_scores, top_indices = scores_flat.topk(num_beams)

        # Update beams
        beam_ids = top_indices // vocab_size
        token_ids = top_indices % vocab_size

        generated_ids = torch.cat([
            generated_ids[beam_ids],
            token_ids.unsqueeze(-1),
        ], dim=-1)
        beam_scores = top_scores

        # Check for EOS
        done_beams = done_beams | (token_ids == eos_token_id)
        if early_stopping and done_beams.all():
            break

    # Return best beam
    best_idx = beam_scores.argmax()
    return generated_ids[best_idx:best_idx+1]


@torch.no_grad()
def generate_with_depth_schedule(
    model: AgentL,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    depth_schedule: str = "constant",
    base_loops: int = 8,
    max_loops: int = 32,
    **kwargs,
) -> torch.Tensor:
    """
    Generate with depth scheduling (curriculum over loop iterations).

    Args:
        model: AgentL model instance
        input_ids: Prompt token IDs (B, T)
        max_new_tokens: Maximum tokens to generate
        depth_schedule: "constant" | "linear" | "cosine" | "exponential"
        base_loops: Starting number of loops
        max_loops: Maximum number of loops
        **kwargs: Additional generation config parameters

    Returns:
        Generated token IDs
    """
    kv_cache: dict = {}
    prompt_len = input_ids.shape[1]
    generated_ids = input_ids.clone()

    for step in range(max_new_tokens):
        # Compute current depth
        progress = step / max_new_tokens

        if depth_schedule == "constant":
            n_loops = base_loops
        elif depth_schedule == "linear":
            n_loops = int(base_loops + progress * (max_loops - base_loops))
        elif depth_schedule == "cosine":
            n_loops = int(base_loops + (max_loops - base_loops) * (1 - (1 + progress * 3.14159) / 2))
        elif depth_schedule == "exponential":
            n_loops = int(base_loops * ((max_loops / base_loops) ** progress))
        else:
            n_loops = base_loops

        if step == 0:
            cur_ids = generated_ids
            start_pos = 0
        else:
            cur_ids = generated_ids[:, -1:]
            start_pos = prompt_len + step - 1

        logits = model.forward(
            cur_ids,
            n_loops=n_loops,
            kv_cache=kv_cache,
            start_pos=start_pos,
        )
        logits = logits[:, -1, :]

        # Sample
        config = GenerationConfig(**kwargs) if kwargs else GenerationConfig()
        next_token = sample_next_token(
            logits,
            temperature=kwargs.get("temperature", 1.0),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 1.0),
        )

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    return generated_ids
