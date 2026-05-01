"""
Visualization utilities for Agent L.

Provides tools for:
- Attention pattern visualization
- Expert routing heatmaps
- ACT halting progress visualization
- Hidden state evolution across loops
- Spectral radius monitoring
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AgentConfig
from ..model import AgentL


@dataclass
class VisualizationData:
    """Container for visualization data."""

    attention_weights: Optional[torch.Tensor] = None  # (layers, heads, seq, seq)
    expert_routing: Optional[torch.Tensor] = None  # (seq, experts)
    act_halting_probs: Optional[List[torch.Tensor]] = None  # [(batch, seq)] per loop
    hidden_states: Optional[List[torch.Tensor]] = None  # [(batch, seq, dim)] per loop
    spectral_radius: Optional[float] = None
    loop_iterations_used: Optional[int] = None


class AttentionCapture:
    """
    Hook-based attention weight capture for visualization.

    Usage:
        capture = AttentionCapture(model)
        with capture.capture():
            output = model(input_ids)
        weights = capture.get_weights()  # Dict[layer_name, weights]
    """

    def __init__(self, model: AgentL):
        self.model = model
        self.weights: Dict[str, torch.Tensor] = {}
        self.handles: List = []

    def _make_hook(self, name: str):
        def hook(module, input, output):
            # Store attention weights if available
            if hasattr(module, "_last_attention_weights"):
                self.weights[name] = module._last_attention_weights.detach().cpu()

        return hook

    def capture(self):
        """Context manager for capturing attention weights."""
        self.weights = {}
        self.handles = []

        # Register hooks on attention layers
        for name, module in self.model.named_modules():
            if "attn" in name.lower():
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get captured attention weights."""
        return self.weights


class ExpertRoutingTracker:
    """
    Track expert routing decisions during forward pass.

    Usage:
        tracker = ExpertRoutingTracker(model)
        routing = tracker.track(input_ids)  # (seq_len, n_experts)
    """

    def __init__(self, model: AgentL):
        self.model = model
        self.routing_scores: Optional[torch.Tensor] = None

    def track(self, input_ids: torch.Tensor, n_loops: int = 8) -> torch.Tensor:
        """
        Track expert routing for a single sequence.

        Returns:
            Tensor of shape (seq_len, n_experts) with routing probabilities
        """
        self.model.eval()

        # Hook to capture routing scores
        routing_scores = []

        def hook(module, input, output):
            if hasattr(module, "router"):
                logits = module.router(input[0])  # (B*T, n_experts)
                scores = F.softmax(logits, dim=-1)
                routing_scores.append(scores.detach().cpu())

        # Register hook on MoE layer
        handle = self.model.recurrent.block.ffn.register_forward_hook(hook)

        with torch.no_grad():
            _ = self.model(input_ids, n_loops=n_loops)

        handle.remove()

        if routing_scores:
            # Average routing scores across all tokens
            return torch.cat(routing_scores, dim=0)
        return torch.zeros(0)


class ACTHaltingTracker:
    """
    Track ACT halting probabilities across loop iterations.

    Usage:
        tracker = ACTHaltingTracker(model)
        probs = tracker.track(input_ids)  # List[(batch, seq)] per loop
    """

    def __init__(self, model: AgentL):
        self.model = model
        self.halting_probs: List[torch.Tensor] = []

    def track(
        self,
        input_ids: torch.Tensor,
        n_loops: int = 8,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Track halting probabilities per loop iteration.

        Returns:
            Tuple of (list of probs per loop, iterations used)
        """
        self.model.eval()
        self.halting_probs = []

        # Store original forward
        original_act_forward = self.model.recurrent.act.forward

        def tracked_forward(h):
            p = original_act_forward(h)
            self.halting_probs.append(p.detach().cpu())
            return p

        # Temporarily replace forward
        self.model.recurrent.act.forward = tracked_forward

        with torch.no_grad():
            _ = self.model(input_ids, n_loops=n_loops)

        # Restore original
        self.model.recurrent.act.forward = original_act_forward

        return self.halting_probs, len(self.halting_probs)


def visualize_attention(
    weights: torch.Tensor,
    layer_idx: int = 0,
    head_idx: Optional[int] = None,
    tokens: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Visualize attention weights as a heatmap.

    Args:
        weights: Attention weights of shape (layers, heads, seq, seq) or (heads, seq, seq)
        layer_idx: Which layer to visualize
        head_idx: Which head to visualize (None = average all heads)
        tokens: Optional token labels for axes
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("matplotlib and seaborn required for visualization")

    # Handle different input shapes
    if weights.dim() == 4:
        weights = weights[layer_idx]

    if head_idx is not None:
        attn = weights[head_idx].numpy()
        title = f"Layer {layer_idx}, Head {head_idx}"
    else:
        attn = weights.mean(dim=0).numpy()
        title = f"Layer {layer_idx}, Average Attention"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        attn,
        cmap="viridis",
        xticklabels=tokens if tokens else False,
        yticklabels=tokens if tokens else False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_expert_routing(
    routing: torch.Tensor,
    top_k: int = 10,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Visualize expert routing as a heatmap.

    Args:
        routing: Routing probabilities of shape (seq_len, n_experts)
        top_k: Number of top experts to show
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("matplotlib and seaborn required for visualization")

    # Get top-k experts by total usage
    expert_usage = routing.sum(dim=0)
    top_experts = torch.topk(expert_usage, min(top_k, routing.shape[1]))
    top_indices = top_experts.indices

    # Subset routing matrix
    routing_top = routing[:, top_indices].numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        routing_top.T,
        cmap="Blues",
        xticklabels=False,
        yticklabels=[f"Expert {i.item()}" for i in top_indices],
        ax=ax,
    )
    ax.set_title("Expert Routing Distribution (Top-K Experts)")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Expert")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_act_halting(
    halting_probs: List[torch.Tensor],
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Visualize ACT halting progression across loops.

    Args:
        halting_probs: List of (batch, seq) tensors per loop iteration
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    n_loops = len(halting_probs)
    seq_len = halting_probs[0].shape[-1]

    # Accumulate probabilities
    cumulative = torch.zeros(seq_len)
    for t, p in enumerate(halting_probs):
        cumulative = cumulative + p[0]  # Take first batch item

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Halting probability per loop
    ax = axes[0]
    for t, p in enumerate(halting_probs[:min(10, n_loops)]):
        ax.plot(p[0].numpy(), label=f"Loop {t}", alpha=0.7)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Halting Probability")
    ax.set_title("Halting Probability per Loop Iteration")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Right: Cumulative halting
    ax = axes[1]
    cumulative_matrix = torch.zeros(min(10, n_loops), seq_len)
    running_sum = torch.zeros(seq_len)
    for t, p in enumerate(halting_probs[:min(10, n_loops)]):
        running_sum = running_sum + p[0]
        cumulative_matrix[t] = running_sum

    im = ax.imshow(cumulative_matrix.numpy(), aspect="auto", cmap="Reds")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Loop Iteration")
    ax.set_title("Cumulative Halting Probability")
    plt.colorbar(im, ax=ax, label="Cumulative P(halt)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_hidden_evolution(
    hidden_states: List[torch.Tensor],
    sample_idx: int = 0,
    n_components: int = 2,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Visualize hidden state evolution across loops using PCA.

    Args:
        hidden_states: List of (batch, seq, dim) tensors per loop
        sample_idx: Which batch sample to visualize
        n_components: Number of PCA components (2 or 3)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("matplotlib and scikit-learn required for visualization")

    # Stack hidden states: (loops, seq, dim)
    hidden_stack = torch.stack([h[sample_idx] for h in hidden_states], dim=0)
    n_loops, seq_len, dim = hidden_stack.shape

    # Reshape for PCA: (loops * seq, dim)
    hidden_flat = hidden_stack.reshape(-1, dim).numpy()

    # Apply PCA
    pca = PCA(n_components=n_components)
    hidden_pca = pca.fit_transform(hidden_flat)  # (loops * seq, n_components)

    # Reshape back
    hidden_pca = hidden_pca.reshape(n_loops, seq_len, n_components)

    # Create visualization
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot trajectory with color gradient
        for t in range(n_loops):
            color = plt.cm.viridis(t / max(n_loops - 1, 1))
            ax.scatter(
                hidden_pca[t, :, 0],
                hidden_pca[t, :, 1],
                c=[color],
                alpha=0.5,
                label=f"Loop {t}" if t % max(1, n_loops // 5) == 0 else "",
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Hidden State Evolution Across Loops (PCA)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        for t in range(n_loops):
            color = plt.cm.viridis(t / max(n_loops - 1, 1))
            ax.scatter(
                hidden_pca[t, :, 0],
                hidden_pca[t, :, 1],
                hidden_pca[t, :, 2],
                c=[color],
                alpha=0.5,
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("Hidden State Evolution Across Loops (3D PCA)")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_model_report(
    model: AgentL,
    input_ids: torch.Tensor,
    n_loops: int = 8,
    save_dir: Optional[str] = None,
) -> Dict[str, any]:
    """
    Create comprehensive visualization report for a model.

    Args:
        model: AgentL model instance
        input_ids: Input token indices
        n_loops: Number of loop iterations
        save_dir: Directory to save visualizations

    Returns:
        Dict with visualization data and file paths
    """
    import os

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    report = {
        "model_config": model.cfg.__dict__,
        "n_loops": n_loops,
        "input_shape": tuple(input_ids.shape),
    }

    # Track expert routing
    routing_tracker = ExpertRoutingTracker(model)
    routing = routing_tracker.track(input_ids, n_loops)
    report["expert_routing"] = routing

    if save_dir and routing.numel() > 0:
        fig = visualize_expert_routing(routing, save_path=f"{save_dir}/expert_routing.png")
        plt.close(fig)
        report["expert_routing_plot"] = f"{save_dir}/expert_routing.png"

    # Track ACT halting
    act_tracker = ACTHaltingTracker(model)
    halting_probs, loops_used = act_tracker.track(input_ids, n_loops)
    report["act_halting_probs"] = halting_probs
    report["loops_used"] = loops_used

    if save_dir and halting_probs:
        fig = visualize_act_halting(halting_probs, save_path=f"{save_dir}/act_halting.png")
        plt.close(fig)
        report["act_halting_plot"] = f"{save_dir}/act_halting.png"

    # Spectral radius
    report["spectral_radius"] = model.get_spectral_radius()

    return report


# Import matplotlib conditionally for functions that need it
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
