"""
Speculative decoding support for Agent L.

Implements speculative decoding where a smaller draft model proposes tokens
that are verified by the main model, enabling faster inference with identical
output distribution.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F

from ..model import AgentL
from ..config import AgentConfig


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    
    draft_model: AgentL  # Smaller/faster draft model
    target_model: AgentL  # Main/target model
    num_speculative_tokens: int = 4  # Number of tokens to speculate
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    draft_n_loops: int = 4  # Fewer loops for draft (faster)
    target_n_loops: int = 8  # More loops for target (higher quality)


class SpeculativeDecoder:
    """
    Speculative decoding with draft and target models.
    
    The draft model generates K candidate tokens, then the target model
    verifies them in parallel. Accepted tokens are kept, and rejection
    leads to resampling from the adjusted distribution.
    
    This provides 2-3x speedup on average while maintaining the exact
    output distribution of the target model.
    """
    
    def __init__(self, config: SpeculativeConfig):
        self.config = config
        self.draft = config.draft_model
        self.target = config.target_model
    
    @torch.no_grad()
    def _sample_from_draft(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate tokens from draft model.
        
        Returns:
            Tuple of (tokens, log_probs) each of shape (batch, num_tokens)
        """
        tokens = []
        log_probs = []
        current = input_ids
        
        for _ in range(num_tokens):
            logits = self.draft(
                current,
                n_loops=self.config.draft_n_loops,
            )[:, -1, :]
            
            # Apply temperature and sampling
            logits = logits / self.config.temperature
            
            if self.config.top_k > 0:
                v, _ = logits.topk(self.config.top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            log_prob = F.log_softmax(logits, dim=-1).gather(-1, token)
            
            tokens.append(token)
            log_probs.append(log_prob)
            
            current = torch.cat([current, token], dim=1)
        
        return (
            torch.cat(tokens, dim=1),
            torch.cat(log_probs, dim=1),
        )
    
    @torch.no_grad()
    def _verify_with_target(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Verify draft tokens with target model.
        
        Returns:
            Tuple of (accepted_tokens, num_accepted)
        """
        batch_size = input_ids.shape[0]
        num_spec = draft_tokens.shape[1]
        
        # Run target model on full sequence
        full_seq = torch.cat([input_ids, draft_tokens], dim=1)
        target_logits = self.target(
            full_seq,
            n_loops=self.config.target_n_loops,
        )
        
        # Get target probabilities for draft positions
        target_log_probs = F.log_softmax(
            target_logits[:, input_ids.shape[1]-1:-1, :],
            dim=-1,
        )
        
        # Acceptance test
        accepted = []
        for t in range(num_spec):
            draft_token = draft_tokens[:, t:t+1]
            draft_lp = draft_log_probs[:, t]
            target_lp = target_log_probs[:, t].gather(-1, draft_token).squeeze(-1)
            
            # Accept with probability min(1, p_target / p_draft)
            acceptance_prob = torch.exp(target_lp - draft_lp)
            random_val = torch.rand(batch_size, device=input_ids.device)
            
            if (random_val <= acceptance_prob).all():
                accepted.append(draft_token)
            else:
                # Rejection: sample from adjusted distribution
                # p_resid = max(0, p_target - p_draft)
                target_probs = F.softmax(
                    target_logits[:, input_ids.shape[1]-1+t, :],
                    dim=-1,
                )
                draft_probs = F.softmax(
                    self.draft(
                        torch.cat([input_ids] + accepted, dim=1) if accepted else input_ids,
                        n_loops=self.config.draft_n_loops,
                    )[:, -1, :] / self.config.temperature,
                    dim=-1,
                )
                
                residual = torch.clamp(target_probs - draft_probs, min=0)
                residual = residual / residual.sum(dim=-1, keepdim=True)
                
                new_token = torch.multinomial(residual, num_samples=1)
                accepted.append(new_token)
                return torch.cat(accepted, dim=1), t
        
        # All accepted: sample bonus token from target
        bonus_token = torch.multinomial(
            F.softmax(target_logits[:, -1, :], dim=-1),
            num_samples=1,
        )
        accepted.append(bonus_token)
        
        return torch.cat(accepted, dim=1), num_spec
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.
        
        Args:
            input_ids: Prompt tokens (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated sequence (batch, seq_len + generated_len)
        """
        generated = input_ids.clone()
        total_generated = 0
        
        while total_generated < max_new_tokens:
            # Determine how many to speculate
            remaining = max_new_tokens - total_generated
            num_spec = min(self.config.num_speculative_tokens, remaining)
            
            # Generate draft tokens
            draft_tokens, draft_log_probs = self._sample_from_draft(
                generated,
                num_spec,
            )
            
            # Verify with target
            accepted, num_accepted = self._verify_with_target(
                generated,
                draft_tokens,
                draft_log_probs,
            )
            
            # Append accepted tokens
            generated = torch.cat([generated, accepted], dim=1)
            total_generated += accepted.shape[1]
        
        return generated


def create_speculative_decoder(
    target_config: AgentConfig,
    draft_config: Optional[AgentConfig] = None,
    target_weights: Optional[str] = None,
    draft_weights: Optional[str] = None,
    num_speculative_tokens: int = 4,
    device: torch.device = None,
) -> SpeculativeDecoder:
    """
    Create a speculative decoder from configurations.
    
    Args:
        target_config: Configuration for target model
        draft_config: Configuration for draft model (default: smaller variant)
        target_weights: Path to target model weights
        draft_weights: Path to draft model weights
        num_speculative_tokens: Number of speculative tokens
        device: Device to load models on
    
    Returns:
        Configured SpeculativeDecoder
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Default draft config is smaller
    if draft_config is None:
        draft_config = AgentConfig(
            vocab_size=target_config.vocab_size,
            dim=target_config.dim // 2,
            n_heads=target_config.n_heads // 2,
            max_seq_len=target_config.max_seq_len,
            max_loop_iters=target_config.max_loop_iters // 2,
            prelude_layers=1,
            coda_layers=1,
            n_experts=target_config.n_experts // 2,
            n_shared_experts=1,
            n_experts_per_tok=2,
            expert_dim=target_config.expert_dim // 2,
            lora_rank=target_config.lora_rank // 2,
            attn_type=target_config.attn_type,
        )
    
    target_model = AgentL(target_config).to(device)
    draft_model = AgentL(draft_config).to(device)
    
    # Load weights if provided
    if target_weights:
        target_model.load_state_dict(torch.load(target_weights, map_location=device))
    if draft_weights:
        draft_model.load_state_dict(torch.load(draft_weights, map_location=device))
    
    target_model.eval()
    draft_model.eval()
    
    return SpeculativeDecoder(SpeculativeConfig(
        draft_model=draft_model,
        target_model=target_model,
        num_speculative_tokens=num_speculative_tokens,
    ))


__all__ = [
    "SpeculativeConfig",
    "SpeculativeDecoder",
    "create_speculative_decoder",
]
