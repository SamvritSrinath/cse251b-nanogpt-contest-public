"""Utilities for configs, checkpointing, logging, and submission export.

References:
    - RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position
      Embedding" (2021), arXiv:2104.09864.
    - RMSNorm: Zhang and Sennrich, "Root Mean Square Layer Normalization"
      (2019), arXiv:1910.07467.
    - SwiGLU: Shazeer, "GLU Variants Improve Transformer" (2020),
      arXiv:2002.05202.
    - Muon: modded-nanogpt writeup and Muon repository lineage from Keller
      Jordan and collaborators.
    - Cosine decay with warmup is standard transformer practice following
      Vaswani et al. (2017) and later large-scale LM training recipes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


PARAMETER_BUDGET: int = 100_000_000


@dataclass(slots=True)
class ModelConfig:
    """Model architecture settings."""

    n_layer: int
    d_model: int
    n_heads: int
    ffn_multiplier: float
    context_len: int
    vocab_size: int
    weight_tying: bool = True
    pos_encoding: str = "rope"
    norm: str = "rmsnorm"
    activation: str = "swiglu"
    dropout: float = 0.0
    bias: bool = False
    rope_base: float = 10_000.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        """Build a model config from a plain mapping."""

        return cls(
            n_layer=int(data["n_layer"]),
            d_model=int(data["d_model"]),
            n_heads=int(data["n_heads"]),
            ffn_multiplier=float(data["ffn_multiplier"]),
            context_len=int(data["context_len"]),
            vocab_size=int(data["vocab_size"]),
            weight_tying=bool(data.get("weight_tying", True)),
            pos_encoding=str(data.get("pos_encoding", "rope")),
            norm=str(data.get("norm", "rmsnorm")),
            activation=str(data.get("activation", "swiglu")),
            dropout=float(data.get("dropout", 0.0)),
            bias=bool(data.get("bias", False)),
            rope_base=float(data.get("rope_base", 10_000.0)),
        )


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer and schedule settings."""

    name: str = "adamw"
    adamw_lr: float = 3e-4
    muon_lr: float = 2e-2
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    momentum: float = 0.95
    nesterov: bool = True
    muon_ns_steps: int = 5

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OptimizerConfig":
        """Build an optimizer config from a plain mapping."""

        betas_raw = data.get("betas", (0.9, 0.95))
        betas = (float(betas_raw[0]), float(betas_raw[1]))
        return cls(
            name=str(data.get("name", "adamw")),
            adamw_lr=float(data.get("adamw_lr", 3e-4)),
            muon_lr=float(data.get("muon_lr", 2e-2)),
            weight_decay=float(data.get("weight_decay", 0.1)),
            betas=betas,
            eps=float(data.get("eps", 1e-8)),
            momentum=float(data.get("momentum", 0.95)),
            nesterov=bool(data.get("nesterov", True)),
            muon_ns_steps=int(data.get("muon_ns_steps", 5)),
        )


@dataclass(slots=True)
class DataConfig:
    """Training and validation data settings."""

    train_data_dir: str
    train_glob: str
    val_data_path: str
    data_fraction: str = "100%"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataConfig":
        """Build a data config from a plain mapping."""

        return cls(
            train_data_dir=str(data["train_data_dir"]),
            train_glob=str(data.get("train_glob", "*.bin")),
            val_data_path=str(data["val_data_path"]),
            data_fraction=str(data.get("data_fraction", "100%")),
        )


@dataclass(slots=True)
class TrainConfig:
    """Training loop settings."""

    max_steps: int
    batch_size: int
    grad_accum_steps: int
    grad_clip: float
    log_interval: int
    eval_interval: int
    save_interval: int
    eval_batch_size: int
    eval_batches: int | None
    seed: int = 1337
    device: str = "cuda"
    compile: bool = False
    warmup_steps: int | None = None
    min_lr_ratio: float = 0.1
    checkpoint_dir: str = "checkpoints"
    submission_dir: str = "submission"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        """Build a train config from a plain mapping."""

        eval_batches_raw = data.get("eval_batches", 16)
        return cls(
            max_steps=int(data["max_steps"]),
            batch_size=int(data["batch_size"]),
            grad_accum_steps=int(data.get("grad_accum_steps", 1)),
            grad_clip=float(data.get("grad_clip", 1.0)),
            log_interval=int(data.get("log_interval", 10)),
            eval_interval=int(data.get("eval_interval", 100)),
            save_interval=int(data.get("save_interval", 1_000)),
            eval_batch_size=int(data.get("eval_batch_size", data.get("batch_size", 1))),
            eval_batches=None if eval_batches_raw is None else int(eval_batches_raw),
            seed=int(data.get("seed", 1337)),
            device=str(data.get("device", "cuda")),
            compile=bool(data.get("compile", False)),
            warmup_steps=None if data.get("warmup_steps") is None else int(data["warmup_steps"]),
            min_lr_ratio=float(data.get("min_lr_ratio", 0.1)),
            checkpoint_dir=str(data.get("checkpoint_dir", "checkpoints")),
            submission_dir=str(data.get("submission_dir", "submission")),
        )


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str
    model: ModelConfig
    optimizer: OptimizerConfig
    data: DataConfig
    train: TrainConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], default_name: str) -> "ExperimentConfig":
        """Build an experiment config from a plain mapping."""

        return cls(
            name=str(data.get("name", default_name)),
            model=ModelConfig.from_dict(data["model"]),
            optimizer=OptimizerConfig.from_dict(data["optimizer"]),
            data=DataConfig.from_dict(data["data"]),
            train=TrainConfig.from_dict(data["train"]),
        )


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config into typed dataclasses.

    Args:
        config_path: Path to a YAML config file.

    Returns:
        The typed experiment configuration.
    """

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig.from_dict(raw, default_name=config_path.stem)


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize a typed config back to a plain JSON-compatible mapping."""

    return asdict(config)


def extract_model_config(raw_config: Mapping[str, Any]) -> ModelConfig:
    """Extract the model subsection from either a full or model-only config."""

    if "model" in raw_config:
        return ModelConfig.from_dict(raw_config["model"])
    return ModelConfig.from_dict(raw_config)


def ensure_parent_dir(path: str | Path) -> None:
    """Create a file's parent directory if needed."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file with stable formatting."""

    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def safe_torch_load(path: str | Path, map_location: str | torch.device) -> Any:
    """Load a torch object while tolerating older torch versions.

    Args:
        path: Checkpoint path to read.
        map_location: Device mapping passed to ``torch.load``.

    Returns:
        The deserialized torch object.
    """

    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters in a model."""

    return sum(parameter.numel() for parameter in model.parameters())


def assert_parameter_budget(model: torch.nn.Module, limit: int = PARAMETER_BUDGET) -> int:
    """Assert that a model fits under the competition parameter budget.

    Args:
        model: The model to inspect.
        limit: Maximum allowed parameter count.

    Returns:
        The total parameter count.

    Raises:
        ValueError: If the model exceeds the limit.
    """

    total = count_parameters(model)
    if total > limit:
        raise ValueError(f"Model exceeds parameter budget: {total:,} > {limit:,}")
    return total


def compute_warmup_steps(train_config: TrainConfig) -> int:
    """Resolve warmup steps, defaulting to 1% of total steps."""

    if train_config.warmup_steps is not None:
        return train_config.warmup_steps
    return max(1, train_config.max_steps // 100)


def cosine_with_warmup(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Compute a warmup-plus-cosine schedule with a non-zero floor."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step <= warmup_steps:
        return max_lr * float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def resolve_device(device: str) -> str:
    """Resolve a requested device string against local availability."""

    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when ``torch.compile`` wraps the model."""

    return getattr(model, "_orig_mod", model)


def format_tokens_per_second(tokens: int, elapsed_seconds: float) -> float:
    """Compute throughput as tokens per second."""

    if elapsed_seconds <= 0.0:
        return 0.0
    return float(tokens) / elapsed_seconds


def hash_experiment_config(config: ExperimentConfig, notes: str = "") -> str:
    """Hash a config and optional notes string for result tracking."""

    payload = {"config": experiment_config_to_dict(config), "notes": notes}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:10]


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer_state: Mapping[str, Any],
    config: ExperimentConfig,
    step: int,
    val_ppl: float,
) -> None:
    """Save a wrapped training checkpoint with optimizer state and config."""

    ensure_parent_dir(path)
    checkpoint_dir = Path(path).parent
    save_json(checkpoint_dir / "config.json", experiment_config_to_dict(config))
    torch.save(
        {
            "step": step,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer_state,
            "config": experiment_config_to_dict(config),
            "val_ppl": val_ppl,
        },
        path,
    )


def export_submission_bundle(
    model: torch.nn.Module,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> None:
    """Export a submission-ready directory for ``evaluate.py``.

    The bundle includes ``checkpoint.pt``, ``config.json``, the root-level
    ``model.py`` shim, and the ``src`` package used by that shim.
    """

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(unwrap_model(model).state_dict(), output_dir / "checkpoint.pt")
    save_json(output_dir / "config.json", experiment_config_to_dict(config))
    shutil.copy2(repo_root / "model.py", output_dir / "model.py")

    src_target = output_dir / "src"
    if src_target.exists():
        shutil.rmtree(src_target)
    shutil.copytree(
        repo_root / "src",
        src_target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def append_experiment_result(
    results_path: str | Path,
    *,
    run_id: str,
    config_name: str,
    config_hash: str,
    data_fraction: str,
    steps: int,
    val_ppl: float,
    notes: str,
) -> None:
    """Append a markdown table row to the experiment log."""

    def escape(cell: str) -> str:
        return cell.replace("|", "/")

    row = (
        f"| {escape(run_id)} | {escape(config_name)} | {escape(config_hash)} | "
        f"{escape(data_fraction)} | {steps} | {val_ppl:.4f} | {escape(notes or '-') } |\n"
    )
    ensure_parent_dir(results_path)
    with Path(results_path).open("a", encoding="utf-8") as handle:
        handle.write(row)


def make_run_id() -> str:
    """Generate a compact wall-clock-based run identifier."""

    return time.strftime("%Y%m%d-%H%M%S")


def absolutize_from_cwd(path: str | Path) -> Path:
    """Resolve a path relative to the current working directory."""

    return Path(path).expanduser().resolve()


def maybe_enable_tf32() -> None:
    """Enable TF32 matmul kernels when CUDA is available."""

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def set_optimizer_lr(optimizer: Any, *, adamw_lr: float, muon_lr: float | None = None) -> None:
    """Set learning rates on either a plain AdamW or the hybrid optimizer."""

    if hasattr(optimizer, "set_learning_rates"):
        optimizer.set_learning_rates(
            adamw_lr=adamw_lr,
            muon_lr=adamw_lr if muon_lr is None else muon_lr,
        )
        return
    for param_group in optimizer.param_groups:
        param_group["lr"] = adamw_lr


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return the resolved path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
