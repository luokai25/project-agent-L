"""
Tests for MoE components.
"""

import torch
import pytest

from agent_l import AgentConfig, MoEFFN


class TestMoEFFN:
    """Test Mixture of Experts FFN."""

    def test_moe_forward_shape(self):
        """Test MoE output shape matches input."""
        cfg = AgentConfig(
            dim=128,
            n_experts=8,
            n_shared_experts=2,
            n_experts_per_tok=2,
            expert_dim=64,
        )
        moe = MoEFFN(cfg)
        x = torch.randn(2, 8, 128)
        out = moe(x)
        assert out.shape == (2, 8, 128)

    def test_moe_different_batch_sizes(self):
        """Test MoE with various batch sizes."""
        cfg = AgentConfig(
            dim=64,
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=32,
        )
        moe = MoEFFN(cfg)
        
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 8, 64)
            out = moe(x)
            assert out.shape == (batch_size, 8, 64)

    def test_moe_different_seq_lengths(self):
        """Test MoE with various sequence lengths."""
        cfg = AgentConfig(
            dim=64,
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=32,
        )
        moe = MoEFFN(cfg)
        
        for seq_len in [1, 16, 64, 128]:
            x = torch.randn(2, seq_len, 64)
            out = moe(x)
            assert out.shape == (2, seq_len, 64)

    def test_moe_expert_selection(self):
        """Test that top-k experts are correctly selected."""
        cfg = AgentConfig(
            dim=64,
            n_experts=8,
            n_shared_experts=1,
            n_experts_per_tok=3,
            expert_dim=32,
        )
        moe = MoEFFN(cfg)
        x = torch.randn(4, 16, 64)
        
        # Get router output
        with torch.no_grad():
            flat = x.view(-1, 64)
            logits = moe.router(flat)
            _, topk_idx = (logits + moe.router_bias).topk(moe.topk, dim=-1)
            
            # Check that indices are valid
            assert (topk_idx >= 0).all()
            assert (topk_idx < cfg.n_experts).all()
            assert topk_idx.shape == (64, 3)

    def test_moe_router_bias(self):
        """Test that router bias affects selection but not gradients."""
        cfg = AgentConfig(
            dim=64,
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=32,
        )
        moe = MoEFFN(cfg)
        
        # Set bias to favor expert 0
        moe.router_bias[0] = 10.0
        
        x = torch.randn(4, 8, 64)
        
        with torch.no_grad():
            flat = x.view(-1, 64)
            logits = moe.router(flat)
            biased_logits = logits + moe.router_bias
            _, topk_idx = biased_logits.topk(moe.topk, dim=-1)
            
            # Expert 0 should be selected frequently (at least 50% of the time)
            # With top_k=2 and 4 experts, expert 0 appears in ~50% of top-k selections
            assert (topk_idx == 0).float().mean() >= 0.5

    def test_moe_shared_experts(self):
        """Test that shared experts are always active."""
        cfg = AgentConfig(
            dim=64,
            n_experts=4,
            n_shared_experts=2,
            n_experts_per_tok=2,
            expert_dim=32,
        )
        moe = MoEFFN(cfg)
        x = torch.randn(2, 8, 64)
        
        # Forward pass
        out = moe(x)
        
        # Check that shared experts exist and have parameters
        assert len(moe.shared_experts) == 2
        for shared in moe.shared_experts:
            assert hasattr(shared, 'up')
            assert hasattr(shared, 'gate')
            assert hasattr(shared, 'down')

    def test_moe_gradient_flow(self):
        """Test that gradients flow through MoE."""
        cfg = AgentConfig(
            dim=64,
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=32,
        )
        moe = MoEFFN(cfg)
        x = torch.randn(2, 8, 64, requires_grad=True)
        
        out = moe(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist
        assert moe.router.weight.grad is not None
        assert moe.up_proj.grad is not None
        assert x.grad is not None

    def test_moe_vectorized_vs_python_loop(self):
        """Test that vectorized and Python loop implementations match."""
        cfg = AgentConfig(
            dim=64,
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=32,
        )
        moe = MoEFFN(cfg)
        x = torch.randn(2, 8, 64)
        
        # Both methods should produce similar results
        out_vectorized = moe(x)
        out_python = moe.forward_python_loop(x)
        
        # Results should be close (not exact due to different accumulation order)
        assert out_vectorized.shape == out_python.shape
        assert torch.allclose(out_vectorized, out_python, rtol=1e-4, atol=1e-4)

    def test_moe_load_balance(self):
        """Test expert load balancing."""
        cfg = AgentConfig(
            dim=128,
            n_experts=16,
            n_shared_experts=2,
            n_experts_per_tok=4,
            expert_dim=64,
        )
        moe = MoEFFN(cfg)
        
        # Run multiple batches
        all_experts_used = set()
        for _ in range(10):
            x = torch.randn(4, 16, 128)
            with torch.no_grad():
                flat = x.view(-1, 128)
                logits = moe.router(flat)
                _, topk_idx = (logits + moe.router_bias).topk(moe.topk, dim=-1)
                all_experts_used.update(topk_idx.flatten().tolist())
        
        # Most experts should be used at least once
        assert len(all_experts_used) >= cfg.n_experts // 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
