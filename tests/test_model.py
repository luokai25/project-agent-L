"""
Tests for Agent L model components.
"""

import torch
import pytest

from agent_l import (
    AgentL,
    AgentConfig,
    RMSNorm,
    GQAttention,
    MLAttention,
    MoEFFN,
    RecurrentBlock,
    LTIInjection,
    ACTHalting,
    agent_1b,
    agent_3b,
)


class TestAgentConfig:
    """Test configuration."""

    def test_default_config(self):
        cfg = AgentConfig()
        assert cfg.dim == 2048
        assert cfg.n_heads == 16
        assert cfg.max_loop_iters == 16
        assert cfg.attn_type == "mla"

    def test_custom_config(self):
        cfg = AgentConfig(dim=512, n_heads=8, max_loop_iters=8)
        assert cfg.dim == 512
        assert cfg.n_heads == 8
        assert cfg.max_loop_iters == 8

    def test_preconfigured_variants(self):
        cfg_1b = agent_1b()
        assert cfg_1b.dim == 2048

        cfg_3b = agent_3b()
        assert cfg_3b.dim == 3072


class TestRMSNorm:
    """Test RMSNorm layer."""

    def test_forward_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        # Check that output is normalized (roughly unit variance)
        rms = out.pow(2).mean(-1).sqrt()
        assert (rms - 1.0).abs().mean() < 0.1


class TestAttention:
    """Test attention mechanisms."""

    def test_gqa_forward(self):
        cfg = AgentConfig(
            dim=128,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            attn_type="gqa",
        )
        attn = GQAttention(cfg)
        x = torch.randn(2, 8, 128)
        freqs = torch.randn(8, 16, dtype=torch.complex64)
        out = attn(x, freqs)
        assert out.shape == (2, 8, 128)

    def test_mla_forward(self):
        cfg = AgentConfig(
            dim=128,
            n_heads=4,
            max_seq_len=32,
            attn_type="mla",
            kv_lora_rank=32,
            q_lora_rank=64,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=16,
        )
        attn = MLAttention(cfg)
        x = torch.randn(2, 8, 128)
        freqs = torch.randn(8, 8, dtype=torch.complex64)
        out = attn(x, freqs)
        assert out.shape == (2, 8, 128)


class TestMoE:
    """Test Mixture of Experts."""

    def test_moe_forward(self):
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


class TestLTIInjection:
    """Test LTI injection for stability."""

    def test_spectral_radius(self):
        lti = LTIInjection(128)
        A = lti.get_A()
        assert A.max() < 1.0, "Spectral radius must be < 1 for stability"

    def test_forward_shape(self):
        lti = LTIInjection(128)
        h = torch.randn(2, 8, 128)
        e = torch.randn(2, 8, 128)
        trans_out = torch.randn(2, 8, 128)
        out = lti(h, e, trans_out)
        assert out.shape == (2, 8, 128)


class TestACTHalting:
    """Test ACT halting mechanism."""

    def test_forward_shape(self):
        act = ACTHalting(128)
        h = torch.randn(2, 8, 128)
        p = act(h)
        assert p.shape == (2, 8)
        assert (p >= 0).all() and (p <= 1).all()


class TestAgentL:
    """Test full model."""

    def test_forward_shape(self):
        cfg = AgentConfig(
            vocab_size=100,
            dim=64,
            n_heads=4,
            max_seq_len=32,
            max_loop_iters=2,
            prelude_layers=1,
            coda_layers=1,
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=32,
            lora_rank=4,
        )
        model = AgentL(cfg)
        input_ids = torch.randint(0, 100, (2, 8))
        logits = model(input_ids, n_loops=2)
        assert logits.shape == (2, 8, 100)

    def test_generate_shape(self):
        cfg = AgentConfig(
            vocab_size=100,
            dim=64,
            n_heads=4,
            max_seq_len=32,
            max_loop_iters=2,
            prelude_layers=1,
            coda_layers=1,
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=32,
            lora_rank=4,
        )
        model = AgentL(cfg)
        input_ids = torch.randint(0, 100, (2, 4))
        output = model.generate(input_ids, max_new_tokens=4, n_loops=2)
        assert output.shape == (2, 8)

    def test_spectral_radius(self):
        cfg = AgentConfig(
            vocab_size=100,
            dim=64,
            n_heads=4,
            max_seq_len=32,
            max_loop_iters=2,
            prelude_layers=1,
            coda_layers=1,
            n_experts=4,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=32,
            lora_rank=4,
        )
        model = AgentL(cfg)
        rho = model.get_spectral_radius()
        assert rho < 1.0, "Model must have spectral radius < 1 for stability"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
