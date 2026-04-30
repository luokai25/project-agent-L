"""
Basic usage example for Agent L.

This demonstrates:
1. Creating a small model configuration
2. Running a forward pass
3. Generating tokens autoregressively
4. Checking model stability (spectral radius)
"""

import torch
from agent_l import AgentL, AgentConfig


def main():
    # Create a small model configuration for testing
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
        attn_type="mla",  # or "gqa"
    )

    print("=" * 60)
    print("Agent L: Recurrent-Depth Transformer")
    print("=" * 60)

    # Initialize model
    model = AgentL(cfg)

    # Count parameters
    params = model.count_parameters()
    print(f"\nModel Configuration:")
    print(f"  Hidden dim: {cfg.dim}")
    print(f"  Attention: {cfg.attn_type.upper()}")
    print(f"  Max loop iterations: {cfg.max_loop_iters}")
    print(f"  Experts: {cfg.n_experts} routed + {cfg.n_shared_experts} shared")
    print(f"\nParameter Counts:")
    print(f"  Total: {params['total']:,}")
    print(f"  Embedding: {params['embed']:,}")
    print(f"  Prelude: {params['prelude']:,}")
    print(f"  Recurrent: {params['recurrent']:,}")
    print(f"  Coda: {params['coda']:,}")

    # Create random input
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    print(f"\nInput shape: {input_ids.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    logits = model(input_ids, n_loops=4)
    print(f"Output logits shape: {logits.shape}")

    # Generate tokens
    print("\nGenerating tokens...")
    output = model.generate(input_ids, max_new_tokens=8, n_loops=8, temperature=0.8)
    print(f"Generated sequence shape: {output.shape}")

    # Check spectral radius (should be < 1 for stability)
    spectral_radius = model.get_spectral_radius()
    print(f"\nSpectral radius ρ(A): {spectral_radius:.4f}")
    if spectral_radius < 1.0:
        print("  ✓ Model is stable (ρ < 1)")
    else:
        print("  ✗ Warning: Model may be unstable (ρ >= 1)")

    # Test with GQA attention
    print("\n" + "=" * 60)
    print("Testing with GQA attention...")
    print("=" * 60)

    cfg_gqa = AgentConfig(
        vocab_size=1000,
        dim=256,
        n_heads=8,
        n_kv_heads=2,  # GQA: fewer KV heads
        max_seq_len=128,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=64,
        lora_rank=8,
        attn_type="gqa",
    )

    model_gqa = AgentL(cfg_gqa)
    logits_gqa = model_gqa(input_ids, n_loops=4)
    print(f"GQA output logits shape: {logits_gqa.shape}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
