"""Modern decoder-only GPT baseline used for training and submission.

References:
    - RoPE on attention queries/keys follows RoFormer (Su et al., 2021).
    - RMSNorm follows Zhang and Sennrich (2019); the stack uses pre-norm
      residual blocks as in modern decoder-only transformers.
    - SwiGLU follows Shazeer (2020), with the inner width controlled by a
      multiplier to keep parameter budgeting explicit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import ModelConfig, extract_model_config, safe_torch_load


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
            self._cached_cos = cos[None, None, :, :]
            self._cached_sin = sin[None, None, :, :]
            self._cached_seq_len = seq_len
        return self._cached_cos[:, :, :seq_len, :], self._cached_sin[:, :, :seq_len, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap and sign-flip even/odd features for RoPE."""

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


def apply_rope(
    x: torch.Tensor,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding to a multi-head tensor."""

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
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

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = int(round(config.ffn_multiplier * config.d_model))
        # SwiGLU follows Shazeer (2020): two parallel projections, one gated by SiLU.
        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated MLP transformation."""

        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and Flash SDP attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.rotary = RotaryEmbedding(self.head_dim, base=config.rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal self-attention to a sequence."""

        batch_size, seq_len, channels = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # RoPE is applied only to Q and K, matching RoFormer and LLaMA-style decoders.
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
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.out_proj(attn_out)


class Block(nn.Module):
    """Pre-norm decoder block with attention and SwiGLU MLP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single transformer block."""

        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    """Configurable decoder-only GPT model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.pos_encoding.lower() != "rope":
            raise ValueError("This baseline implements only RoPE positional encoding.")
        if config.norm.lower() != "rmsnorm":
            raise ValueError("This baseline implements only RMSNorm.")
        if config.activation.lower() != "swiglu":
            raise ValueError("This baseline implements only SwiGLU.")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize module weights with GPT-style defaults."""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to vocabulary logits.

        Args:
            input_ids: LongTensor of shape ``(batch_size, sequence_length)``.

        Returns:
            FloatTensor of shape ``(batch_size, sequence_length, vocab_size)``.
        """

        _, seq_len = input_ids.shape
        if seq_len > self.config.context_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds configured context "
                f"window {self.config.context_len}."
            )

        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


def load_model(checkpoint_path: str, device: str = "cuda") -> nn.Module:
    """Load a trained model from a submission directory.

    Args:
        checkpoint_path: Path to ``checkpoint.pt``.
        device: Target device, usually ``"cuda"`` or ``"cpu"``.

    Returns:
        A model in evaluation mode.

    Raises:
        FileNotFoundError: If ``config.json`` is missing next to the checkpoint.
    """

    state = safe_torch_load(checkpoint_path, map_location=device)
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path.parent / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw_config: Mapping[str, Any] = json.load(handle)
        config = extract_model_config(raw_config)
    elif isinstance(state, Mapping) and "config" in state:
        config = extract_model_config(state["config"])
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
