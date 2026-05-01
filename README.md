# Agent L

A Recurrent-Depth Transformer implementation based on published research.

## Architecture Overview

```
Input → [Prelude] → [Recurrent Block × T loops] → [Coda] → Output
                          ↓
              MLA + DeepSeekMoE + ACT Halting
```

## Provenance: What's Proven vs. Hypothesized

| Component | Source | Status |
|-----------|--------|--------|
| **Looped Transformers** | Saunshi et al., ICLR 2025 [1] | ✅ Proven - k×L ≈ kL layers on reasoning |
| **Multi-Latent Attention (MLA)** | DeepSeek-V2, 2024 [2] | ✅ Proven - 93% KV cache reduction |
| **DeepSeekMoE** | DeepSeekMoE, ACL 2024 [3] | ✅ Proven - fine-grained + shared experts |
| **ACT Halting** | Graves, 2016 [4] | ✅ Proven - differentiable adaptive compute |
| **Loop-index embedding** | Saunshi et al., 2025 [1] | ✅ Proven - distinguishes loop iterations |
| **LTI Injection** | OpenMythos [5] | ⚠️ Hypothesized - stability mechanism |
| **Combined architecture** | This implementation | ⚠️ Novel combination of proven parts |

**Important**: This is NOT Claude's architecture. Anthropic has not published Claude's architecture. This is a research implementation combining proven components from published papers.

## References

1. Saunshi et al. "Reasoning with Latent Thoughts: On the Power of Looped Transformers" ICLR 2025
2. DeepSeek-AI "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" arXiv 2024
3. Dai et al. "DeepSeekMoE: Towards Ultimate Expert Specialization in MoE LLMs" ACL 2024
4. Graves "Adaptive Computation Time for Recurrent Neural Networks" arXiv 2016
5. Gomez "OpenMythos" GitHub 2024 - hypothesis reconstruction

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from agent_l import AgentL, AgentConfig, agent_3b

# Use pre-configured model
cfg = agent_3b()
model = AgentL(cfg)

# Run forward pass
import torch
tokens = torch.randint(0, cfg.vocab_size, (1, 32))
logits = model(tokens, n_loops=8)

# Generate
output = model.generate(tokens, max_new_tokens=64, n_loops=8)
```

## Key Properties

### Depth Extrapolation (Saunshi et al.)
- Train with T loops, test with T+k loops for harder problems
- Recurrence-equivalence exponent φ ≈ 0.46

### MLA Efficiency (DeepSeek-V2)
- KV cache compressed to latent vector
- 93% cache reduction, 5.76× throughput gain

### DeepSeekMoE Routing
- Fine-grained experts: split N into m×N sub-experts, activate m×K
- Shared experts: Ks always-active for common knowledge
- 16B model matches LLaMA2-7B with 40% compute

### ACT Halting (Graves)
- Per-position halting probabilities
- Variable compute per token

## Project Structure

```
agent_l/
├── __init__.py          # Package exports
├── config.py            # Configuration + model variants
├── model.py             # AgentL main model
├── layers.py            # RMSNorm, RoPE, loop-index embedding
├── attention.py         # GQA + MLA (Multi-Latent Attention)
├── moe.py               # DeepSeekMoE (fine-grained + shared)
└── recurrent.py         # RecurrentBlock, LTI injection, ACT halting
```

## Training

This is an architecture skeleton with random weights. To make it useful:

1. Pretrain on large text corpus (see `training/`)
2. Apply MLA + MoE efficiency techniques
3. Fine-tune with SFT + RLHF

## License

MIT

## Acknowledgments

- DeepSeek-AI for MLA and DeepSeekMoE architectures
- Saunshi et al. for looped transformer research
- Alex Graves for ACT
- Kye Gomez for OpenMythos hypothesis
