"""
Tests for recurrent components.
"""

import torch
import pytest

from agent_l import (
    AgentConfig,
    LoRAAdapter,
    LTIInjection,
    ACTHalting,
    RecurrentBlock,
)


class TestLoRAAdapter:
    """Test LoRA adapter for depth-wise adaptation."""

    def test_forward_shape(self):
        """Test LoRA output shape."""
        lora = LoRAAdapter(dim=128, rank=8, max_loops=16)
        x = torch.randn(2, 8, 128)
        
        for t in range(5):
            out = lora(x, loop_t=t)
            assert out.shape == (2, 8, 128)

    def test_different_loop_indices(self):
        """Test that different loop indices produce different outputs."""
        lora = LoRAAdapter(dim=128, rank=8, max_loops=16)
        x = torch.randn(2, 8, 128)
        
        outputs = [lora(x, loop_t=t) for t in range(4)]
        
        # All outputs should be different
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j])

    def test_depth_extrapolation(self):
        """Test that loop indices beyond max_loops are handled."""
        lora = LoRAAdapter(dim=128, rank=8, max_loops=4)
        x = torch.randn(2, 8, 128)
        
        # Should not crash with loop_t > max_loops
        out = lora(x, loop_t=10)
        assert out.shape == (2, 8, 128)

    def test_gradient_flow(self):
        """Test gradient flow through LoRA."""
        lora = LoRAAdapter(dim=128, rank=8, max_loops=16)
        x = torch.randn(2, 8, 128, requires_grad=True)
        
        out = lora(x, loop_t=0)
        loss = out.sum()
        loss.backward()
        
        assert lora.down.weight.grad is not None
        assert lora.B.grad is not None
        assert lora.scale.weight.grad is not None
        assert x.grad is not None


class TestLTIInjection:
    """Test LTI injection for stable recurrent updates."""

    def test_spectral_radius(self):
        """Test that spectral radius is always < 1."""
        lti = LTIInjection(128)
        
        # Initialize and check
        A = lti.get_A()
        assert A.max() < 1.0
        assert A.min() >= 0.0

    def test_forward_shape(self):
        """Test LTI output shape."""
        lti = LTIInjection(128)
        h = torch.randn(2, 8, 128)
        e = torch.randn(2, 8, 128)
        trans_out = torch.randn(2, 8, 128)
        
        out = lti(h, e, trans_out)
        assert out.shape == (2, 8, 128)

    def test_stability_over_iterations(self):
        """Test that hidden state remains stable over many iterations."""
        lti = LTIInjection(128)
        h = torch.randn(2, 8, 128)
        e = torch.randn(2, 8, 128)
        
        # Simulate many recurrent iterations
        for _ in range(100):
            trans_out = torch.randn(2, 8, 128) * 0.1
            h = lti(h, e, trans_out)
        
        # Hidden state should not explode
        assert torch.isfinite(h).all()
        assert h.abs().max() < 100

    def test_input_preservation(self):
        """Test that input signal e is preserved through LTI."""
        lti = LTIInjection(128)
        h = torch.zeros(2, 8, 128)
        e = torch.randn(2, 8, 128)
        trans_out = torch.zeros(2, 8, 128)
        
        # First iteration should blend e into h
        h_new = lti(h, e, trans_out)
        
        # h_new should be close to e (since h=0 and trans_out=0)
        # With B = 1 - A, h_new = B * e ≈ e
        correlation = (h_new * e).sum() / (e * e).sum()
        assert correlation > 0.5

    def test_gradient_flow(self):
        """Test gradient flow through LTI."""
        lti = LTIInjection(128)
        h = torch.randn(2, 8, 128, requires_grad=True)
        e = torch.randn(2, 8, 128)
        trans_out = torch.randn(2, 8, 128)
        
        out = lti(h, e, trans_out)
        loss = out.sum()
        loss.backward()
        
        assert lti.log_dt.grad is not None
        assert lti.log_A.grad is not None
        assert h.grad is not None


class TestACTHalting:
    """Test ACT halting mechanism."""

    def test_forward_shape(self):
        """Test ACT output shape."""
        act = ACTHalting(128)
        h = torch.randn(2, 8, 128)
        
        p = act(h)
        assert p.shape == (2, 8)

    def test_probability_range(self):
        """Test that halting probability is in (0, 1)."""
        act = ACTHalting(128)
        h = torch.randn(2, 8, 128)
        
        p = act(h)
        assert (p > 0).all()
        assert (p < 1).all()

    def test_different_inputs_different_probs(self):
        """Test that different inputs produce different probabilities."""
        act = ACTHalting(128)
        h1 = torch.randn(2, 8, 128)
        h2 = torch.randn(2, 8, 128) * 10  # Different scale
        
        p1 = act(h1)
        p2 = act(h2)
        
        # Probabilities should differ
        assert not torch.allclose(p1, p2)

    def test_gradient_flow(self):
        """Test gradient flow through ACT."""
        act = ACTHalting(128)
        h = torch.randn(2, 8, 128, requires_grad=True)
        
        p = act(h)
        loss = p.sum()
        loss.backward()
        
        assert act.halt.weight.grad is not None
        assert h.grad is not None


class TestRecurrentBlock:
    """Test the full recurrent block."""

    def test_forward_shape(self):
        """Test recurrent block output shape."""
        cfg = AgentConfig(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            max_loop_iters=4,
            attn_type="gqa",
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=32,
            lora_rank=4,
        )
        recurrent = RecurrentBlock(cfg)
        
        h = torch.randn(2, 8, 64)
        e = torch.randn(2, 8, 64)
        # freqs_cis shape should be (seq_len, head_dim // 2) for GQA
        head_dim = cfg.dim // cfg.n_heads
        freqs = torch.randn(8, head_dim // 2, dtype=torch.complex64)
        
        out = recurrent(h, e, freqs, n_loops=2)
        assert out.shape == (2, 8, 64)

    def test_different_loop_counts(self):
        """Test with different numbers of loops."""
        cfg = AgentConfig(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            max_loop_iters=8,
            attn_type="gqa",
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=32,
            lora_rank=4,
        )
        recurrent = RecurrentBlock(cfg)
        
        h = torch.randn(2, 8, 64)
        e = torch.randn(2, 8, 64)
        head_dim = cfg.dim // cfg.n_heads
        freqs = torch.randn(8, head_dim // 2, dtype=torch.complex64)
        
        for n_loops in [1, 2, 4, 8]:
            out = recurrent(h, e, freqs, n_loops=n_loops)
            assert out.shape == (2, 8, 64)

    def test_kv_cache_compatibility(self):
        """Test that recurrent block works with KV caching."""
        cfg = AgentConfig(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            max_loop_iters=4,
            attn_type="gqa",
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=32,
            lora_rank=4,
        )
        recurrent = RecurrentBlock(cfg)
        
        h = torch.randn(2, 8, 64)
        e = torch.randn(2, 8, 64)
        head_dim = cfg.dim // cfg.n_heads
        freqs = torch.randn(8, head_dim // 2, dtype=torch.complex64)
        kv_cache = {}
        
        out = recurrent(h, e, freqs, n_loops=2, kv_cache=kv_cache)
        
        # Cache should be populated
        assert len(kv_cache) > 0

    def test_act_early_exit(self):
        """Test that ACT can cause early exit from loops."""
        cfg = AgentConfig(
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            max_loop_iters=16,
            act_threshold=0.5,  # Low threshold for early halting
            attn_type="gqa",
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=32,
            lora_rank=4,
        )
        recurrent = RecurrentBlock(cfg)
        
        h = torch.randn(2, 8, 64)
        e = torch.randn(2, 8, 64)
        head_dim = cfg.dim // cfg.n_heads
        freqs = torch.randn(8, head_dim // 2, dtype=torch.complex64)
        
        # Should complete without error even with low threshold
        out = recurrent(h, e, freqs, n_loops=16)
        assert out.shape == (2, 8, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
