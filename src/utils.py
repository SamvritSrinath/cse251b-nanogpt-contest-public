"""Utilities for configs, checkpointing, search studies, and submission export.

References:
    - RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position
      Embedding" (2021), arXiv:2104.09864.
    - RMSNorm: Zhang and Sennrich, "Root Mean Square Layer Normalization"
      (2019), arXiv:1910.07467.
    - SwiGLU: Shazeer, "GLU Variants Improve Transformer" (2020),
      arXiv:2002.05202.
    - Muon: modded-nanogpt writeup and Muon repository lineage from Keller
      Jordan and collaborators.
    - FineWeb / FineWeb-Edu: Hugging Face FineData team, dataset cards and
      FineWeb papers documenting large-scale web curation.
    - Optuna: Akiba et al., "Optuna: A Next-generation Hyperparameter
      Optimization Framework" (KDD 2019).
    - Cosine decay with warmup is standard transformer practice following
      Vaswani et al. (2017) and later large-scale LM training recipes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


PARAMETER_BUDGET: int = 100_000_000
RESULTS_HEADER: str = (
    "| Run | Config | Config Hash | Arch | Optimizer | Batch Tokens | Context | "
    "Data Mix | Steps | Params | Val PPL | Notes |\n"
    "|-----|--------|-------------|------|-----------|-------------|---------|"
    "----------|-------|--------|---------|-------|\n"
)


@dataclass(slots=True)
class ArchitectureConfig:
    """Architecture settings for a decoder-only language model."""

    name: str
    n_layer: int
    d_model: int
    embedding_dim: int | None
    n_heads: int
    ffn_multiplier: float
    context_len: int
    vocab_size: int
    weight_tying: bool = True
    dropout: float = 0.0
    bias: bool | None = None
    rope_base: float = 10_000.0
    qk_norm: bool = False
    residual_init: str = "default"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArchitectureConfig":
        """Build an architecture config from a plain mapping."""

        residual_init = str(data.get("residual_init", "default")).lower()
        if residual_init not in {"default", "scaled"}:
            raise ValueError(
                "architecture.residual_init must be 'default' or 'scaled'."
            )
        return cls(
            name=str(data.get("name", "modern_decoder")),
            n_layer=int(data["n_layer"]),
            d_model=int(data["d_model"]),
            embedding_dim=None
            if data.get("embedding_dim") is None
            else int(data["embedding_dim"]),
            n_heads=int(data["n_heads"]),
            ffn_multiplier=float(data["ffn_multiplier"]),
            context_len=int(data["context_len"]),
            vocab_size=int(data["vocab_size"]),
            weight_tying=bool(data.get("weight_tying", True)),
            dropout=float(data.get("dropout", 0.0)),
            bias=None if data.get("bias") is None else bool(data["bias"]),
            rope_base=float(data.get("rope_base", 10_000.0)),
            qk_norm=bool(data.get("qk_norm", False)),
            residual_init=residual_init,
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
class DataSourceConfig:
    """One weighted training source in a mixture."""

    name: str
    path: str
    glob: str = "**/*.bin"
    weight: float = 1.0
    notes: str = ""
    sample_policy: str = "random_window"
    prepare: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataSourceConfig":
        """Build a source config from a plain mapping."""

        sample_policy = str(data.get("sample_policy", "random_window"))
        if sample_policy not in {
            "random_window",
            "document_window",
            "section_window",
            "packed_short_docs",
        }:
            raise ValueError(
                "data.sources[].sample_policy must be one of: random_window, "
                "document_window, section_window, packed_short_docs."
            )
        prepare = None
        if data.get("prepare") is not None:
            prepare = dict(data["prepare"])
        return cls(
            name=str(data["name"]),
            path=str(data["path"]),
            glob=str(data.get("glob", "**/*.bin")),
            weight=float(data.get("weight", 1.0)),
            notes=str(data.get("notes", "")),
            sample_policy=sample_policy,
            prepare=prepare,
        )


@dataclass(slots=True)
class DataConfig:
    """Training and validation data settings."""

    sources: list[DataSourceConfig]
    val_data_path: str
    data_fraction: str = "100%"
    manifest_dir: str = "data/manifests"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataConfig":
        """Build a data config from a plain mapping.

        This accepts both the new mixture-aware schema and the older single
        source fields so existing configs remain readable during the refactor.
        """

        if "sources" in data:
            sources = [DataSourceConfig.from_dict(source) for source in data["sources"]]
        else:
            # Backwards compatibility: upgrade the old one-source fields in place.
            sources = [
                DataSourceConfig(
                    name=str(data.get("train_data_name", "fineweb_edu")),
                    path=str(data["train_data_dir"]),
                    glob=str(data.get("train_glob", "**/*.bin")),
                    weight=float(data.get("train_weight", 1.0)),
                    notes=str(data.get("train_notes", "")),
                )
            ]
        return cls(
            sources=sources,
            val_data_path=str(data["val_data_path"]),
            data_fraction=str(data.get("data_fraction", "100%")),
            manifest_dir=str(data.get("manifest_dir", "data/manifests")),
        )


@dataclass(slots=True)
class ContextScheduleConfig:
    """Training-time context length schedule.

    ``fixed`` keeps one context length for all steps.
    ``ramp`` holds ``start_context_len`` until ``switch_step`` and then linearly
    increases to ``end_context_len`` over ``ramp_steps`` steps.
    """

    name: str = "fixed"
    start_context_len: int | None = None
    end_context_len: int | None = None
    switch_step: int = 0
    ramp_steps: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ContextScheduleConfig":
        """Build a schedule config from a plain mapping."""

        if data is None:
            return cls()
        return cls(
            name=str(data.get("name", "fixed")),
            start_context_len=None
            if data.get("start_context_len") is None
            else int(data["start_context_len"]),
            end_context_len=None
            if data.get("end_context_len") is None
            else int(data["end_context_len"]),
            switch_step=int(data.get("switch_step", 0)),
            ramp_steps=int(data.get("ramp_steps", 0)),
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
    precision: str = "fp32"
    warmup_steps: int | None = None
    min_lr_ratio: float = 0.1
    checkpoint_dir: str = "checkpoints"
    submission_dir: str = "submission"
    resume_from: str | None = None
    resume_initial_step: int | None = None
    resume_initial_val_ppl: float | None = None
    lr_schedule_origin: str = "absolute"
    log_cuda_memory: bool = False
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_eval: bool = True
    ema_device: str = "cuda"
    use_teacher: bool = False
    teacher_model_name: str = "openai-community/gpt2-large"
    teacher_weight: float = 0.0
    teacher_temperature: float = 2.0
    teacher_device: str = "cuda"
    context_schedule: ContextScheduleConfig = field(default_factory=ContextScheduleConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        """Build a train config from a plain mapping."""

        eval_batches_raw = data.get("eval_batches", 16)
        precision = str(data.get("precision", "fp32")).lower()
        if precision not in {"fp32", "bf16"}:
            raise ValueError(
                f"Unsupported train.precision '{precision}'. Supported values: fp32, bf16."
            )
        lr_schedule_origin = str(data.get("lr_schedule_origin", "absolute")).lower()
        if lr_schedule_origin not in {"absolute", "resume"}:
            raise ValueError("train.lr_schedule_origin must be 'absolute' or 'resume'.")
        ema_device = str(data.get("ema_device", "cuda")).lower()
        if ema_device not in {"cuda", "cpu"}:
            raise ValueError("train.ema_device must be 'cuda' or 'cpu'.")
        ema_decay = float(data.get("ema_decay", 0.999))
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("train.ema_decay must satisfy 0.0 <= ema_decay < 1.0.")
        teacher_device = str(data.get("teacher_device", "cuda")).lower()
        if teacher_device not in {"cuda", "cpu"}:
            raise ValueError("train.teacher_device must be 'cuda' or 'cpu'.")
        teacher_weight = float(data.get("teacher_weight", 0.0))
        if not 0.0 <= teacher_weight <= 1.0:
            raise ValueError("train.teacher_weight must satisfy 0.0 <= teacher_weight <= 1.0.")
        teacher_temperature = float(data.get("teacher_temperature", 2.0))
        if teacher_temperature <= 0.0:
            raise ValueError("train.teacher_temperature must be positive.")
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
            precision=precision,
            warmup_steps=None if data.get("warmup_steps") is None else int(data["warmup_steps"]),
            min_lr_ratio=float(data.get("min_lr_ratio", 0.1)),
            checkpoint_dir=str(data.get("checkpoint_dir", "checkpoints")),
            submission_dir=str(data.get("submission_dir", "submission")),
            resume_from=None if data.get("resume_from") is None else str(data["resume_from"]),
            resume_initial_step=None
            if data.get("resume_initial_step") is None
            else int(data["resume_initial_step"]),
            resume_initial_val_ppl=None
            if data.get("resume_initial_val_ppl") is None
            else float(data["resume_initial_val_ppl"]),
            lr_schedule_origin=lr_schedule_origin,
            log_cuda_memory=bool(data.get("log_cuda_memory", False)),
            use_ema=bool(data.get("use_ema", False)),
            ema_decay=ema_decay,
            ema_eval=bool(data.get("ema_eval", True)),
            ema_device=ema_device,
            use_teacher=bool(data.get("use_teacher", False)),
            teacher_model_name=str(data.get("teacher_model_name", "openai-community/gpt2-large")),
            teacher_weight=teacher_weight,
            teacher_temperature=teacher_temperature,
            teacher_device=teacher_device,
            context_schedule=ContextScheduleConfig.from_dict(data.get("context_schedule")),
        )


@dataclass(slots=True)
class TrialMetadata:
    """Optional study metadata attached to a single training run."""

    study_name: str | None = None
    trial_id: str | None = None
    trial_index: int | None = None


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str
    architecture: ArchitectureConfig
    optimizer: OptimizerConfig
    data: DataConfig
    train: TrainConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], default_name: str) -> "ExperimentConfig":
        """Build an experiment config from a plain mapping."""

        architecture_data = data.get("architecture", data.get("model"))
        if architecture_data is None:
            raise KeyError("Expected an 'architecture' section in the experiment config.")
        return cls(
            name=str(data.get("name", default_name)),
            architecture=ArchitectureConfig.from_dict(architecture_data),
            optimizer=OptimizerConfig.from_dict(data["optimizer"]),
            data=DataConfig.from_dict(data["data"]),
            train=TrainConfig.from_dict(data["train"]),
        )


@dataclass(slots=True)
class SearchParameterConfig:
    """One search-space dimension for grid or Optuna studies."""

    path: str
    type: str
    values: list[Any] | None = None
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    conditions: dict[str, list[Any]] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SearchParameterConfig":
        """Build a search parameter from a plain mapping."""

        return cls(
            path=str(data["path"]),
            type=str(data["type"]),
            values=None if data.get("values") is None else list(data["values"]),
            low=data.get("low"),
            high=data.get("high"),
            step=data.get("step"),
            log=bool(data.get("log", False)),
            conditions=None
            if data.get("conditions") is None
            else {
                str(path): list(values if isinstance(values, list) else [values])
                for path, values in data["conditions"].items()
            },
        )


@dataclass(slots=True)
class StudyObjectiveConfig:
    """Objective metadata for a search study."""

    metric: str = "val_ppl"
    direction: str = "minimize"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "StudyObjectiveConfig":
        """Build an objective config from a plain mapping."""

        if data is None:
            return cls()
        return cls(
            metric=str(data.get("metric", "val_ppl")),
            direction=str(data.get("direction", "minimize")),
        )


@dataclass(slots=True)
class StudyPrunerConfig:
    """Pruning configuration for Optuna studies."""

    name: str = "median"
    n_startup_trials: int = 1
    n_warmup_steps: int = 1
    interval_steps: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "StudyPrunerConfig":
        """Build a pruner config from a plain mapping."""

        if data is None:
            return cls()
        return cls(
            name=str(data.get("name", "median")),
            n_startup_trials=int(data.get("n_startup_trials", 1)),
            n_warmup_steps=int(data.get("n_warmup_steps", 1)),
            interval_steps=int(data.get("interval_steps", 1)),
        )


@dataclass(slots=True)
class StudyConfig:
    """Top-level config for a grid or Optuna search study."""

    name: str
    mode: str
    base_config: str
    output_dir: str = "experiments/studies"
    objective: StudyObjectiveConfig = field(default_factory=StudyObjectiveConfig)
    pruner: StudyPrunerConfig = field(default_factory=StudyPrunerConfig)
    storage: str | None = None
    max_trials: int | None = None
    parallelism: int = 1
    resume: bool = True
    search_space: list[SearchParameterConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], default_name: str) -> "StudyConfig":
        """Build a study config from a plain mapping."""

        return cls(
            name=str(data.get("name", default_name)),
            mode=str(data["mode"]),
            base_config=str(data["base_config"]),
            output_dir=str(data.get("output_dir", "experiments/studies")),
            objective=StudyObjectiveConfig.from_dict(data.get("objective")),
            pruner=StudyPrunerConfig.from_dict(data.get("pruner")),
            storage=None if data.get("storage") is None else str(data["storage"]),
            max_trials=None if data.get("max_trials") is None else int(data["max_trials"]),
            parallelism=int(data.get("parallelism", 1)),
            resume=bool(data.get("resume", True)),
            search_space=[
                SearchParameterConfig.from_dict(parameter)
                for parameter in data.get("search_space", [])
            ],
        )


@dataclass(slots=True)
class TrainingSummary:
    """Structured summary returned by the trainer and consumed by search."""

    run_id: str
    config_name: str
    config_hash: str
    parameter_count: int
    best_val_ppl: float
    best_val_loss: float
    steps_completed: int
    architecture_name: str
    optimizer_name: str
    context_schedule: str
    data_mixture: str
    effective_batch_tokens: int
    checkpoint_dir: str
    final_submission_dir: str
    best_submission_dir: str | None
    notes: str
    status: str = "completed"
    study_name: str | None = None
    trial_id: str | None = None
    trial_index: int | None = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain Python mapping."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def save_yaml(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a YAML file with stable formatting."""

    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment config into typed dataclasses."""

    config_path = Path(config_path)
    raw = load_yaml(config_path)
    return ExperimentConfig.from_dict(raw, default_name=config_path.stem)


def load_study_config(config_path: str | Path) -> StudyConfig:
    """Load a YAML study config into typed dataclasses."""

    config_path = Path(config_path)
    raw = load_yaml(config_path)
    return StudyConfig.from_dict(raw, default_name=config_path.stem)


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize a typed experiment config to a plain JSON-compatible mapping."""

    return asdict(config)


def training_summary_to_dict(summary: TrainingSummary) -> dict[str, Any]:
    """Serialize a typed training summary to a plain mapping."""

    return asdict(summary)


def extract_architecture_config(raw_config: Mapping[str, Any]) -> ArchitectureConfig:
    """Extract the architecture subsection from either a full or model-only config."""

    if "architecture" in raw_config:
        return ArchitectureConfig.from_dict(raw_config["architecture"])
    if "model" in raw_config:
        return ArchitectureConfig.from_dict(raw_config["model"])
    return ArchitectureConfig.from_dict(raw_config)


def ensure_parent_dir(path: str | Path) -> None:
    """Create a file's parent directory if needed."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return the resolved path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON file with stable formatting."""

    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a plain mapping."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_torch_load(path: str | Path, map_location: str | torch.device) -> Any:
    """Load a torch object while tolerating older torch versions."""

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
    """Assert that a model fits under the competition parameter budget."""

    total = count_parameters(model)
    if total > limit:
        raise ValueError(f"Model exceeds parameter budget: {total:,} > {limit:,}")
    return total


def compute_warmup_steps(train_config: TrainConfig) -> int:
    """Resolve warmup steps, defaulting to 1% of total steps."""

    if train_config.warmup_steps is not None:
        return train_config.warmup_steps
    return max(1, train_config.max_steps // 100)


def resolve_warmup_steps(
    train_config: TrainConfig,
    *,
    initial_step: int,
    resumed: bool,
) -> int:
    """Resolve warmup steps for this run, disabling warmup on checkpoint resume.

    Continuation runs load weights that already passed warmup. Re-applying warmup
    would drop LR below the checkpoint's effective rate and waste steps.
    """

    if resumed and train_config.lr_schedule_origin == "resume":
        return compute_warmup_steps(train_config)

    if resumed or initial_step > 0:
        return 0
    return compute_warmup_steps(train_config)


def resolve_context_length(
    schedule: ContextScheduleConfig,
    *,
    max_context_len: int,
    step: int,
) -> int:
    """Resolve the training-time context length for the current step."""

    if schedule.name == "fixed":
        return int(schedule.end_context_len or schedule.start_context_len or max_context_len)
    if schedule.name != "ramp":
        raise ValueError(f"Unsupported context schedule: {schedule.name}")

    start_len = int(schedule.start_context_len or max_context_len)
    end_len = int(schedule.end_context_len or max_context_len)
    if step <= schedule.switch_step:
        return start_len
    if schedule.ramp_steps <= 0:
        return end_len

    # The linear ramp keeps the schedule explicit and easy to reason about in logs.
    progress = min(1.0, (step - schedule.switch_step) / float(schedule.ramp_steps))
    current_len = round(start_len + (end_len - start_len) * progress)
    return int(max(1, min(current_len, max_context_len)))


def format_context_schedule(schedule: ContextScheduleConfig, *, max_context_len: int) -> str:
    """Format a schedule for logs and results tables."""

    if schedule.name == "fixed":
        fixed_len = schedule.end_context_len or schedule.start_context_len or max_context_len
        return f"fixed@{fixed_len}"
    start_len = schedule.start_context_len or max_context_len
    end_len = schedule.end_context_len or max_context_len
    return f"ramp:{start_len}->{end_len}@{schedule.switch_step}+{schedule.ramp_steps}"


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


def format_data_mixture(data_config: DataConfig) -> str:
    """Summarize the weighted source mixture for logs and study tables."""

    return ",".join(f"{source.name}:{source.weight:g}" for source in data_config.sources)


def effective_batch_tokens(
    architecture_config: ArchitectureConfig,
    train_config: TrainConfig,
) -> int:
    """Compute full-length tokens processed per optimizer step."""

    return (
        architecture_config.context_len
        * train_config.batch_size
        * train_config.grad_accum_steps
    )


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer_state: Mapping[str, Any],
    config: ExperimentConfig,
    step: int,
    val_ppl: float,
    ema_model: torch.nn.Module | None = None,
    ema_val_ppl: float | None = None,
) -> None:
    """Save a wrapped training checkpoint with optimizer state and config."""

    ensure_parent_dir(path)
    checkpoint_dir = Path(path).parent
    save_json(checkpoint_dir / "config.json", experiment_config_to_dict(config))
    payload: dict[str, Any] = {
        "step": step,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer_state,
        "config": experiment_config_to_dict(config),
        "val_ppl": val_ppl,
    }
    if ema_model is not None:
        payload["ema_model_state_dict"] = unwrap_model(ema_model).state_dict()
        if ema_val_ppl is not None:
            payload["ema_val_ppl"] = ema_val_ppl
    torch.save(payload, path)


def export_submission_bundle(
    model: torch.nn.Module,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> None:
    """Export a submission-ready directory for ``evaluate.py``."""

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


def ensure_results_table(results_path: str | Path) -> None:
    """Create the experiment results table header if it does not exist."""

    results_path = Path(results_path)
    if results_path.exists() and results_path.stat().st_size > 0:
        return
    ensure_parent_dir(results_path)
    results_path.write_text(RESULTS_HEADER, encoding="utf-8")


def append_experiment_result(
    results_path: str | Path,
    *,
    summary: TrainingSummary,
) -> None:
    """Append a markdown table row to the experiment log."""

    def escape(cell: str) -> str:
        return cell.replace("|", "/")

    ensure_results_table(results_path)
    row = (
        f"| {escape(summary.run_id)} | {escape(summary.config_name)} | "
        f"{escape(summary.config_hash)} | {escape(summary.architecture_name)} | "
        f"{escape(summary.optimizer_name)} | {summary.effective_batch_tokens} | "
        f"{escape(summary.context_schedule)} | {escape(summary.data_mixture)} | "
        f"{summary.steps_completed} | {summary.parameter_count:,} | "
        f"{summary.best_val_ppl:.4f} | {escape(summary.notes or '-')} |\n"
    )
    with Path(results_path).open("a", encoding="utf-8") as handle:
        handle.write(row)


def make_run_id() -> str:
    """Generate a compact wall-clock-based run identifier."""

    return time.strftime("%Y%m%d-%H%M%S")


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


def get_by_path(payload: Mapping[str, Any], path: str) -> Any:
    """Read a dotted path from a nested mapping."""

    current: Any = payload
    for part in path.split("."):
        current = current[part]
    return current


def set_by_path(payload: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path inside a nested mapping, creating dicts as needed."""

    current: dict[str, Any] = payload
    parts = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def materialize_experiment_config(
    base_config: ExperimentConfig,
    overrides: Mapping[str, Any],
    *,
    run_name: str | None = None,
) -> ExperimentConfig:
    """Clone an experiment config and apply dotted-path overrides."""

    payload = experiment_config_to_dict(base_config)
    if run_name is not None:
        payload["name"] = run_name
    for path, value in overrides.items():
        set_by_path(payload, path, value)
    return ExperimentConfig.from_dict(payload, default_name=payload["name"])


def materialize_experiment_payload(
    base_config: ExperimentConfig,
    overrides: Mapping[str, Any],
    *,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Materialize a plain payload from a config and overrides."""

    return experiment_config_to_dict(
        materialize_experiment_config(base_config, overrides, run_name=run_name)
    )


def conditions_match(
    conditions: Mapping[str, list[Any]] | None,
    *,
    base_payload: Mapping[str, Any],
    current_overrides: Mapping[str, Any],
) -> bool:
    """Return whether a conditional search parameter is currently active."""

    if not conditions:
        return True
    for path, allowed_values in conditions.items():
        current_value = current_overrides.get(path, get_by_path(base_payload, path))
        if current_value not in allowed_values:
            return False
    return True


def absolutize_from_cwd(path: str | Path) -> Path:
    """Resolve a path relative to the current working directory."""

    return Path(path).expanduser().resolve()


def clone_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-clone a mapping using ``copy.deepcopy``."""

    return copy.deepcopy(dict(payload))
