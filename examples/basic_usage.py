"""
Basic usage example for Agent L.

This demonstrates:
1. Creating model configurations
2. Running forward passes
3. Generating tokens autoregressively
4. Checking parameter counts
5. Depth extrapolation (more loops = deeper reasoning)
"""

import torch
from agent_l import (
    AgentL, 
    AgentConfig, 
    agent_1b, 
    agent_3b,
    RMSNorm,
)


def main():
    print("=" * 60)
    print("Agent L: Recurrent-Depth Transformer")
    print("=" * 60)
    
    # ============================================================
    # 1. Configuration
    # ============================================================
    print("\n[1] Creating model configuration...")
    
    # Option A: Use pre-configured variant
    cfg = agent_1b()
    print(f"    Using agent_1b() config: dim={cfg.dim}, experts={cfg.n_experts}, loops={cfg.max_loop_iters}")
    
    # Option B: Custom configuration
    # cfg = AgentConfig(
    #     vocab_size=1000,
    #     dim=256,
    #     n_heads=8,
    #     max_loop_iters=4,
    #     n_experts=16,
    #     n_shared_experts=2,
    # )
    
    # ============================================================
    # 2. Create Model
    # ============================================================
    print("\n[2] Creating model...")
    model = AgentL(cfg)
    
    params = model.count_parameters()
    print(f"    Total parameters: {params['total_billions']:.3f}B")
    print(f"    Active parameters (estimate): {params['active_billions']:.3f}B")
    
    # ============================================================
    # 3. Forward Pass
    # ============================================================
    print("\n[3] Running forward pass...")
    
    batch_size = 2
    seq_len = 16
    
    # Random input tokens
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    print(f"    Input shape: {input_ids.shape}")
    
    # Forward with different loop counts
    for n_loops in [4, 8, 16]:
        logits = model(input_ids, n_loops=n_loops)
        print(f"    Output logits (n_loops={n_loops}): shape={logits.shape}")
    
    # ============================================================
    # 4. Generation
    # ============================================================
    print("\n[4] Generating tokens...")
    
    # Single prompt
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    print(f"    Prompt shape: {prompt.shape}")
    
    # Generate with depth extrapolation (more loops than training)
    output = model.generate(
        prompt, 
        max_new_tokens=16, 
        n_loops=8,  # Can use more at inference for harder problems
        temperature=1.0,
        top_k=50
    )
    print(f"    Generated shape: {output.shape}")
    print(f"    Generated {output.shape[1] - prompt.shape[1]} new tokens")
    
    # ============================================================
    # 5. Depth Extrapolation Demo
    # ============================================================
    print("\n[5] Depth extrapolation...")
    print("    Key property: train on T loops, test on T+k loops")
    print("    More iterations = deeper reasoning chains")
    print("    (Proven in Saunshi et al., ICLR 2025)")
    
    # Test with increasing loop counts
    test_input = torch.randint(0, cfg.vocab_size, (1, 4))
    for n_loops in [4, 8, 16, 32]:
        with torch.no_grad():
            logits = model(test_input, n_loops=n_loops)
            probs = torch.softmax(logits[0, -1], dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        print(f"    n_loops={n_loops:2d}: output entropy={entropy:.3f}")
    
    # ============================================================
    # 6. Component Tests
    # ============================================================
    print("\n[6] Testing individual components...")
    
    # RMSNorm
    norm = RMSNorm(cfg.dim)
    x = torch.randn(2, 4, cfg.dim)
    y = norm(x)
    print(f"    RMSNorm: input={x.shape} → output={y.shape}")
    assert y.shape == x.shape
    
    # MLA Attention
    from agent_l import MLAttention
    attn = MLAttention(cfg)
    print(f"    MLAttention: kv_lora_rank={cfg.kv_lora_rank}, qk_rope_head_dim={cfg.qk_rope_head_dim}")
    
    # MoE
    from agent_l import MoEFFN
    moe = MoEFFN(cfg)
    print(f"    MoEFFN: {cfg.n_experts} experts, {cfg.n_shared_experts} shared, top-{cfg.n_experts_per_tok}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. Agent L combines proven components from published papers")
    print("2. MLA reduces KV cache by ~93% (DeepSeek-V2)")
    print("3. MoE enables large capacity with sparse activation")
    print("4. Looped transformers enable depth extrapolation")
    print("5. ACT halting provides variable compute per token")
    print("\nNext steps:")
    print("- Pretrain on large text corpus")
    print("- Fine-tune with SFT + RLHF")
    print("- Experiment with loop counts for different tasks")


if __name__ == "__main__":
    main()
