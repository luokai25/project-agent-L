# Agent L

A Recurrent-Depth Transformer implementation based on the Claude Mythos architecture hypothesis.

## Architecture

Agent L implements a **Recurrent-Depth Transformer (RDT)** - a transformer architecture where a subset of layers is recycled and run through multiple times per forward pass. This enables deeper reasoning without parameter explosion.

```
Input tokens
     ↓
[Prelude]          — standard transformer layers, run once
     ↓
[Recurrent Block]  — one transformer block looped T times with input injection
     ↑_______↓     h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
     ↓
[Coda]             — standard transformer layers, run once
     ↓
Output logits
```

## Key Features

- **Recurrent Depth**: Same weights, more loops → deeper reasoning, no parameter growth
- **Depth Extrapolation**: Train on N loops, test on N+k loops (emergent capability)
- **ACT Halting**: Variable compute per position within a batch
- **MoE FFN**: Mixture of Experts in the recurrent block for breadth across domains
- **LTI-Stable Injection**: Spectral radius < 1 guaranteed by construction
- **Dual Attention**: Supports both GQA and MLA (Multi-Latent Attention)

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import torch
from agent_l import AgentL, AgentConfig

# Create a small model for testing
cfg = AgentConfig(
    vocab_size=1000,
    dim=256,
    n_heads=8,
    max_seq_len=128,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=8,
    attn_type="mla",
)

model = AgentL(cfg)

# Forward pass
input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
logits = model(input_ids, n_loops=4)
print(f"Logits shape: {logits.shape}")

# Generate
output = model.generate(input_ids, max_new_tokens=8, n_loops=8)
print(f"Generated shape: {output.shape}")
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vocab_size` | Token vocabulary size | 32000 |
| `dim` | Model hidden dimension | 2048 |
| `n_heads` | Number of query attention heads | 16 |
| `n_kv_heads` | Number of key/value heads (GQA) | 4 |
| `max_seq_len` | Maximum sequence length | 4096 |
| `max_loop_iters` | Default recurrent loop depth | 16 |
| `prelude_layers` | Layers before the loop | 2 |
| `coda_layers` | Layers after the loop | 2 |
| `attn_type` | "gqa" or "mla" | "mla" |
| `n_experts` | Number of routed experts | 64 |
| `n_shared_experts` | Always-active shared experts | 2 |
| `n_experts_per_tok` | Top-K experts per token | 4 |
| `expert_dim` | Expert hidden dimension | 512 |
| `act_threshold` | ACT halting threshold | 0.99 |
| `lora_rank` | Depth-wise LoRA rank | 16 |

## Model Variants

Pre-configured model sizes:

- `agent_1b()` - 1B parameters (dim=2048, 64 experts)
- `agent_3b()` - 3B parameters (dim=3072, 64 experts)
- `agent_10b()` - 10B parameters (dim=4096, 128 experts)
- `agent_50b()` - 50B parameters (dim=6144, 256 experts)
- `agent_100b()` - 100B parameters (dim=8192, 256 experts, 1M context)

## Research Background

This implementation is based on the hypothesis that Claude Mythos uses a Recurrent-Depth Transformer architecture. Key papers:

- [Loop, Think, & Generalize](https://arxiv.org/pdf/2604.07822) - Implicit reasoning in RDTs
- [Parcae: Scaling Laws for Stable Looped Language Models](https://arxiv.org/abs/2604.12946)
- [Reasoning with Latent Thoughts](https://arxiv.org/abs/2502.17416) - Power of looped transformers
- [DeepSeek-V2](https://arxiv.org/abs/2401.06066) - MoE with shared experts

## License

MIT License
