"""Decoder-only language model implementations and architecture registry.

References:
    - RoPE on attention queries/keys follows RoFormer (Su et al., 2021).
    - RMSNorm follows Zhang and Sennrich (2019).
    - SwiGLU follows Shazeer (2020).
    - The GPT-2-style ablation bundle uses learned absolute position embeddings,
      LayerNorm, and GELU to create a coherent "old baseline" comparator instead
      of changing one primitive at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.utils import ArchitectureConfig, ExperimentConfig, extract_architecture_config, safe_torch_load


@dataclass(frozen=True, slots=True)
class ArchitectureRecipe:
    """Resolved implementation details for one architecture family."""

    name: str
    use_rope: bool
    use_learned_positions: bool
    norm_type: str
    mlp_type: str
    bias: bool


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize activations using their root mean square."""

        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def build_modern_recipe(config: ArchitectureConfig) -> ArchitectureRecipe:
    """Resolve the modern RoPE + RMSNorm + SwiGLU bundle."""

    return ArchitectureRecipe(
        name=config.name,
        use_rope=True,
        use_learned_positions=False,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        bias=False if config.bias is None else config.bias,
    )


def build_gpt2_recipe(config: ArchitectureConfig) -> ArchitectureRecipe:
    """Resolve the GPT-2-style learned-position + LayerNorm + GELU bundle."""

    return ArchitectureRecipe(
        name=config.name,
        use_rope=False,
        use_learned_positions=True,
        norm_type="layernorm",
        mlp_type="gelu",
        bias=True if config.bias is None else config.bias,
    )


ARCHITECTURE_REGISTRY: dict[str, Callable[[ArchitectureConfig], ArchitectureRecipe]] = {
    "modern_decoder": build_modern_recipe,
    "gpt2_decoder": build_gpt2_recipe,
    # Keeping registry-based stubs makes it obvious where future MTP variants plug in.
    # "mtp_decoder": build_mtp_recipe,
}


def resolve_architecture_recipe(config: ArchitectureConfig) -> ArchitectureRecipe:
    """Resolve an architecture name to a concrete implementation recipe."""

    try:
        return ARCHITECTURE_REGISTRY[config.name](config)
    except KeyError as exc:
        supported = ", ".join(sorted(ARCHITECTURE_REGISTRY))
        raise ValueError(f"Unsupported architecture '{config.name}'. Supported: {supported}") from exc


def make_norm(norm_type: str, dim: int) -> nn.Module:
    """Create the requested normalization module."""

    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unsupported norm type: {norm_type}")


class RotaryEmbedding(nn.Module):
    """Rotary position embedding cache for decoder self-attention."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head dimension, got {head_dim}.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_seq_len: int = 0
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None

    def get_cos_sin(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached cosine and sine tables for a sequence length."""

        if (
            self._cached_cos is None
            or self._cached_sin is None
            or self._cached_seq_len < seq_len
            or self._cached_cos.device != device
            or self._cached_cos.dtype != dtype
        ):
            positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            angles = torch.outer(positions, self.inv_freq.to(device=device))
            cos = torch.cos(angles).to(dtype=dtype)
            sin = torch.sin(angles).to(dtype=dtype)
            # We cache in broadcast-ready shape so the forward pass avoids extra reshapes.
            self._cached_cos = cos[None, None, :, :]
            self._cached_sin = sin[None, None, :, :]
            self._cached_seq_len = seq_len
        return self._cached_cos[:, :, :seq_len, :], self._cached_sin[:, :, :seq_len, :]


def apply_rope(x: torch.Tensor, *, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to a multi-head tensor."""

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    # Interleaving the rotated pairs keeps the head dimension unchanged after RoPE.
    rotated = torch.stack(
        (
            x_even * cos - x_odd * sin,
            x_even * sin + x_odd * cos,
        ),
        dim=-1,
    )
    return rotated.flatten(start_dim=-2)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, config: ArchitectureConfig, *, bias: bool) -> None:
        super().__init__()
        hidden_dim = int(round(config.ffn_multiplier * config.d_model))
        # SwiGLU uses two parallel projections, so the multiplier is lower than GPT-2's 4x MLP.
        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated MLP transformation."""

        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class GeluMLP(nn.Module):
    """GPT-2-style feed-forward block."""

    def __init__(self, config: ArchitectureConfig, *, bias: bool) -> None:
        super().__init__()
        hidden_dim = int(round(config.ffn_multiplier * config.d_model))
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the GELU MLP transformation."""

        hidden = F.gelu(self.up_proj(x), approximate="tanh")
        return self.dropout(self.down_proj(hidden))


def make_mlp(config: ArchitectureConfig, recipe: ArchitectureRecipe) -> nn.Module:
    """Create the MLP block requested by the architecture recipe."""

    if recipe.mlp_type == "swiglu":
        return SwiGLU(config, bias=recipe.bias)
    if recipe.mlp_type == "gelu":
        return GeluMLP(config, bias=recipe.bias)
    raise ValueError(f"Unsupported MLP type: {recipe.mlp_type}")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional RoPE."""

    def __init__(self, config: ArchitectureConfig, recipe: ArchitectureRecipe) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.use_rope = recipe.use_rope
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=recipe.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=recipe.bias)
        self.rotary = RotaryEmbedding(self.head_dim, base=config.rope_base) if recipe.use_rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention to a sequence."""

        batch_size, seq_len, channels = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
            # We move the head axis forward because scaled_dot_product_attention expects
            # tensors shaped like (batch, heads, time, head_dim).
            return tensor.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        if self.use_rope:
            assert self.rotary is not None
            # RoPE is applied only to Q and K; V stays untouched in modern decoder recipes.
            cos, sin = self.rotary.get_cos_sin(seq_len, device=x.device, dtype=q.dtype)
            q = apply_rope(q, cos=cos, sin=sin)
            k = apply_rope(k, cos=cos, sin=sin)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        # We transpose back so the residual path sees (batch, time, channels) again.
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.out_proj(attn_out)


class DecoderBlock(nn.Module):
    """Pre-norm decoder block with attention and configurable MLP."""

    def __init__(self, config: ArchitectureConfig, recipe: ArchitectureRecipe) -> None:
        super().__init__()
        self.attn_norm = make_norm(recipe.norm_type, config.d_model)
        self.ffn_norm = make_norm(recipe.norm_type, config.d_model)
        self.attn = CausalSelfAttention(config, recipe)
        self.mlp = make_mlp(config, recipe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single transformer block."""

        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class DecoderLanguageModel(nn.Module):
    """Generic decoder-only LM assembled from an architecture recipe."""

    def __init__(self, config: ArchitectureConfig, recipe: ArchitectureRecipe) -> None:
        super().__init__()
        self.config = config
        self.recipe = recipe
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = (
            nn.Embedding(config.context_len, config.d_model)
            if recipe.use_learned_positions
            else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config, recipe) for _ in range(config.n_layer)])
        self.final_norm = make_norm(recipe.norm_type, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        if config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable activation checkpointing for memory-constrained training."""

        self.gradient_checkpointing = enabled

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize module weights with GPT-style defaults."""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to vocabulary logits."""

        _, seq_len = input_ids.shape
        if seq_len > self.config.context_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds configured context "
                f"window {self.config.context_len}."
            )

        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            # GPT-2-style ablations keep learned absolute position embeddings explicit.
            positions = torch.arange(seq_len, device=input_ids.device)
            x = x + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


class GPT(nn.Module):
    """Submission-facing wrapper that resolves architectures via the registry."""

    def __init__(self, config: ArchitectureConfig | Mapping[str, Any]) -> None:
        super().__init__()
        if not isinstance(config, ArchitectureConfig):
            config = ArchitectureConfig.from_dict(config)
        self.config = config
        self.recipe = resolve_architecture_recipe(config)
        self.impl = DecoderLanguageModel(config, self.recipe)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Forward the training-only checkpointing switch to the implementation."""

        self.impl.set_gradient_checkpointing(enabled)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Delegate the forward pass to the resolved implementation."""

        return self.impl(input_ids)


def build_model(config: ExperimentConfig | ArchitectureConfig | Mapping[str, Any]) -> GPT:
    """Construct a model from either a full experiment config or architecture config."""

    if isinstance(config, ExperimentConfig):
        return GPT(config.architecture)
    return GPT(config)


def load_model(checkpoint_path: str, device: str = "cuda") -> nn.Module:
    """Load a trained model from a submission directory."""

    state = safe_torch_load(checkpoint_path, map_location=device)
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path.parent / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config: Mapping[str, Any] = json.load(handle)
        config = extract_architecture_config(raw_config)
    elif isinstance(state, Mapping) and "config" in state:
        config = extract_architecture_config(state["config"])
    else:
        raise FileNotFoundError(
            f"Expected config.json next to checkpoint, but did not find {config_path}, "
            "and the checkpoint did not embed a config payload."
        )

    model = GPT(config)
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
