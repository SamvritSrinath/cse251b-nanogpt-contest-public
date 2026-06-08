"""Main training loop for iterative experiments and study-driven trials."""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.data import WeightedShardSampler, load_source_manifests, iter_validation_batches, summarize_manifests
from src.model import build_model
from src.optimizer import build_optimizer
from src.utils import (
    ExperimentConfig,
    TrainingSummary,
    TrialMetadata,
    append_experiment_result,
    assert_parameter_budget,
    cosine_with_warmup,
    resolve_warmup_steps,
    effective_batch_tokens,
    ensure_directory,
    experiment_config_to_dict,
    export_submission_bundle,
    format_context_schedule,
    format_data_mixture,
    format_tokens_per_second,
    hash_experiment_config,
    load_experiment_config,
    make_run_id,
    maybe_enable_tf32,
    resolve_context_length,
    resolve_device,
    safe_torch_load,
    save_training_checkpoint,
    save_yaml,
    set_optimizer_lr,
    set_seed,
    unwrap_model,
)


MetricCallback = Callable[[dict[str, Any]], None]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""

    parser = argparse.ArgumentParser(description="Train the Part 3 NanoGPT baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Short note recorded in experiments/results.md.",
    )
    parser.add_argument("--study-name", type=str, default=None, help="Optional study name.")
    parser.add_argument("--trial-id", type=str, default=None, help="Optional trial identifier.")
    parser.add_argument("--trial-index", type=int, default=None, help="Optional trial index.")
    return parser.parse_args()


@torch.no_grad()
def evaluate_validation_loss(
    model: torch.nn.Module,
    *,
    data_path: str,
    context_len: int,
    batch_size: int,
    device: str,
    max_batches: int | None,
    autocast_enabled: bool = False,
    autocast_device_type: str = "cuda",
) -> tuple[float, float]:
    """Compute validation loss and perplexity using ``evaluate.py``-compatible slicing."""

    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in iter_validation_batches(
        data_path,
        context_len=context_len,
        batch_size=batch_size,
        device=device,
        max_batches=max_batches,
    ):
        with torch.autocast(
            device_type=autocast_device_type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
        total_loss += loss.item()
        total_tokens += targets.numel()

    if model_was_training:
        model.train()

    avg_loss = total_loss / max(1, total_tokens)
    return avg_loss, math.exp(avg_loss)


def maybe_compile(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    """Optionally compile the model with ``torch.compile``."""

    if not enabled or not hasattr(torch, "compile"):
        return model
    return torch.compile(model)


def create_ema_model(model: torch.nn.Module, *, device: str) -> torch.nn.Module:
    """Create a frozen eval-mode EMA copy from the unwrapped model."""

    ema_model = copy.deepcopy(unwrap_model(model)).to(device)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    *,
    decay: float,
) -> None:
    """Update EMA parameters and copy non-floating state from the current model."""

    current_state = unwrap_model(model).state_dict()
    for name, ema_value in ema_model.state_dict().items():
        current_value = current_state[name].detach().to(device=ema_value.device)
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(current_value.to(dtype=ema_value.dtype), alpha=1.0 - decay)
        else:
            ema_value.copy_(current_value)


def load_teacher_model(
    *,
    model_name: str,
    device: str,
    use_bf16: bool,
) -> torch.nn.Module:
    """Load a frozen GPT-2-family teacher model for optional distillation."""

    try:
        from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "train.use_teacher=true requires transformers. Install dependencies from "
            "requirements.txt before enabling teacher distillation."
        ) from exc

    dtype = torch.bfloat16 if use_bf16 and device.startswith("cuda") else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    vocab_size = int(getattr(getattr(teacher, "config", object()), "vocab_size", 0))
    if vocab_size != 50257:
        raise ValueError(
            f"Teacher model {model_name} has vocab_size={vocab_size}; "
            "GPT-2-token distillation requires vocab_size=50257."
        )
    return teacher


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Compute token-mean KL loss from teacher probabilities to student logits."""

    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"Teacher logits shape {tuple(teacher_logits.shape)} does not match "
            f"student logits shape {tuple(student_logits.shape)}."
        )
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
    token_count = max(1, student_logits.shape[0] * student_logits.shape[1])
    return (
        F.kl_div(student_log_probs, teacher_probs, reduction="sum")
        * (temperature * temperature)
        / token_count
    )


def log_cuda_memory_stats(label: str) -> None:
    """Print a short CUDA memory snapshot when instrumentation is enabled."""

    allocated_gb = torch.cuda.memory_allocated() / 1e9
    reserved_gb = torch.cuda.memory_reserved() / 1e9
    max_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"{label} "
        f"vram_allocated_gb={allocated_gb:.2f} "
        f"vram_reserved_gb={reserved_gb:.2f} "
        f"vram_max_allocated_gb={max_allocated_gb:.2f}"
    )


def run_training(
    config: ExperimentConfig,
    *,
    notes: str = "",
    trial_metadata: TrialMetadata | None = None,
    metric_callback: MetricCallback | None = None,
    append_results: bool = True,
) -> TrainingSummary:
    """Run the training loop and return a structured summary."""

    trial_metadata = trial_metadata or TrialMetadata()
    run_id = make_run_id()
    config_hash = hash_experiment_config(config, notes=notes)

    maybe_enable_tf32()
    device = resolve_device(config.train.device)
    set_seed(config.train.seed)
    device_is_cuda = device.startswith("cuda")
    use_bf16 = device_is_cuda and config.train.precision == "bf16"
    autocast_device_type = "cuda" if device_is_cuda else "cpu"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("train.precision=bf16 requires CUDA BF16 support on the selected device.")

    model = build_model(config).to(device)
    parameter_count = assert_parameter_budget(model)
    print(f"Model parameters: {parameter_count:,}")

    optimizer = build_optimizer(model, config.optimizer)
    initial_step = 0
    resumed = config.train.resume_from is not None
    best_val_ppl = float("inf")
    best_val_loss = float("inf")
    best_ema_val_ppl = float("inf")
    best_ema_val_loss = float("inf")
    if resumed:
        assert config.train.resume_from is not None
        resume_path = Path(config.train.resume_from)
        checkpoint_path = resume_path / "checkpoint.pt" if resume_path.is_dir() else resume_path
        state = safe_torch_load(checkpoint_path, map_location=device)
        if not isinstance(state, Mapping):
            raise ValueError(f"Expected a checkpoint mapping at {checkpoint_path}.")

        has_training_model = "model_state_dict" in state
        has_training_optimizer = "optimizer_state_dict" in state
        if has_training_model or has_training_optimizer:
            if not has_training_model or not has_training_optimizer:
                raise ValueError(
                    f"Expected training checkpoint at {checkpoint_path} to contain both "
                    "model_state_dict and optimizer_state_dict."
                )
            if "step" not in state:
                raise ValueError(f"Expected training checkpoint at {checkpoint_path} to record step.")
            model.load_state_dict(state["model_state_dict"], strict=True)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            initial_step = int(state["step"])
            resumed_val_ppl = float(state.get("val_ppl", float("inf")))
            best_val_ppl = resumed_val_ppl
            if math.isfinite(resumed_val_ppl) and resumed_val_ppl > 0.0:
                best_val_loss = math.log(resumed_val_ppl)
            resume_kind = "training checkpoint"
        else:
            if config.train.resume_initial_step is None:
                raise ValueError(
                    "resume_initial_step is required when resuming from a model-only checkpoint."
                )
            model.load_state_dict(state, strict=True)
            initial_step = int(config.train.resume_initial_step)
            resume_kind = "model-only checkpoint"

        if config.train.resume_initial_val_ppl is not None:
            best_val_ppl = float(config.train.resume_initial_val_ppl)
            if not math.isfinite(best_val_ppl) or best_val_ppl <= 0.0:
                raise ValueError("train.resume_initial_val_ppl must be a positive finite value.")
            best_val_loss = math.log(best_val_ppl)
        print(
            f"Resumed {resume_kind}: path={checkpoint_path} step={initial_step} "
            f"best_val_ppl={best_val_ppl:.4f}"
        )
    model = maybe_compile(model, config.train.compile)
    ema_device = resolve_device(config.train.ema_device)
    ema_model: torch.nn.Module | None = None
    if config.train.use_ema:
        ema_model = create_ema_model(model, device=ema_device)
        if resumed and "ema_model_state_dict" in state:
            ema_model.load_state_dict(state["ema_model_state_dict"], strict=True)
        print(
            "EMA enabled: "
            f"decay={config.train.ema_decay:g} "
            f"device={ema_device} "
            f"eval={'on' if config.train.ema_eval else 'off'}"
        )
    teacher_device = resolve_device(config.train.teacher_device)
    teacher_model: torch.nn.Module | None = None
    if config.train.use_teacher:
        if config.train.teacher_weight <= 0.0:
            print(
                "Teacher distillation configured but inactive: "
                "train.teacher_weight is 0.0."
            )
        else:
            teacher_model = load_teacher_model(
                model_name=config.train.teacher_model_name,
                device=teacher_device,
                use_bf16=use_bf16,
            )
            print(
                "Teacher distillation enabled: "
                f"model={config.train.teacher_model_name} "
                f"weight={config.train.teacher_weight:g} "
                f"temperature={config.train.teacher_temperature:g} "
                f"device={teacher_device}"
            )

    manifests = load_source_manifests(config.data)
    print(f"Data sources: {summarize_manifests(manifests)}")
    train_loader = WeightedShardSampler(
        manifests,
        batch_size=config.train.batch_size,
        seed=config.train.seed,
    )

    warmup_steps = resolve_warmup_steps(
        config.train,
        initial_step=initial_step,
        resumed=resumed,
    )

    def lr_schedule_index(step: int) -> tuple[int, int]:
        if resumed and config.train.lr_schedule_origin == "resume":
            return (
                max(1, step - initial_step),
                max(1, config.train.max_steps - initial_step),
            )
        return max(1, step), config.train.max_steps

    if resumed:
        if initial_step <= 0:
            print(
                "Warning: checkpoint has step=0; LR schedule starts at step 1. "
                "Save checkpoints with train.py so step is recorded for smooth continuation."
            )
        schedule_step, schedule_total = lr_schedule_index(initial_step + 1)
        resume_adamw_lr = cosine_with_warmup(
            schedule_step,
            warmup_steps=warmup_steps,
            total_steps=schedule_total,
            max_lr=config.optimizer.adamw_lr,
            min_lr_ratio=config.train.min_lr_ratio,
        )
        resume_muon_lr = cosine_with_warmup(
            schedule_step,
            warmup_steps=warmup_steps,
            total_steps=schedule_total,
            max_lr=config.optimizer.muon_lr,
            min_lr_ratio=config.train.min_lr_ratio,
        )
        set_optimizer_lr(optimizer, adamw_lr=resume_adamw_lr, muon_lr=resume_muon_lr)
        if warmup_steps == 0 and config.train.warmup_steps not in (None, 0):
            print(
                f"Resume LR schedule: warmup disabled (checkpoint step={initial_step}), "
                f"adamw_lr={resume_adamw_lr:.6g} muon_lr={resume_muon_lr:.6g}"
            )
    checkpoint_dir = ensure_directory(Path(config.train.checkpoint_dir) / run_id)
    submission_root = ensure_directory(Path(config.train.submission_dir) / run_id)
    # Saving the resolved YAML next to checkpoints makes every trial reproducible on its own.
    save_yaml(
        checkpoint_dir / "resolved_config.yaml",
        {
            "config_hash": config_hash,
            **experiment_config_to_dict(config),
        },
    )

    if initial_step >= config.train.max_steps:
        raise ValueError(
            f"resume_from step {initial_step} is already at or beyond max_steps "
            f"{config.train.max_steps}."
        )
    best_submission_dir: Path | None = None
    best_ema_submission_dir: Path | None = None
    tokens_since_log = 0
    python_last_log_time = time.time()
    logged_first_step_memory = False
    if config.train.log_cuda_memory and device_is_cuda:
        torch.cuda.reset_peak_memory_stats()
        log_cuda_memory_stats("startup")

    context_schedule_label = format_context_schedule(
        config.train.context_schedule,
        max_context_len=config.architecture.context_len,
    )
    model.train()
    for step in range(initial_step + 1, config.train.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_ce_loss = 0.0
        train_teacher_loss = 0.0
        train_context_len = resolve_context_length(
            config.train.context_schedule,
            max_context_len=config.architecture.context_len,
            step=step,
        )

        schedule_step, schedule_total = lr_schedule_index(step)
        current_adamw_lr = cosine_with_warmup(
            schedule_step,
            warmup_steps=warmup_steps,
            total_steps=schedule_total,
            max_lr=config.optimizer.adamw_lr,
            min_lr_ratio=config.train.min_lr_ratio,
        )
        current_muon_lr = cosine_with_warmup(
            schedule_step,
            warmup_steps=warmup_steps,
            total_steps=schedule_total,
            max_lr=config.optimizer.muon_lr,
            min_lr_ratio=config.train.min_lr_ratio,
        )
        set_optimizer_lr(optimizer, adamw_lr=current_adamw_lr, muon_lr=current_muon_lr)

        for _ in range(config.train.grad_accum_steps):
            # Validation always uses the full configured context, but training batches
            # can follow a curriculum schedule to make early updates cheaper.
            inputs, targets = train_loader.next_batch(device, context_len=train_context_len)
            with torch.autocast(
                device_type=autocast_device_type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(inputs)
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                loss = ce_loss
                if teacher_model is not None:
                    teacher_inputs = inputs if teacher_device == device else inputs.to(teacher_device)
                    with torch.no_grad():
                        teacher_outputs = teacher_model(teacher_inputs)
                        teacher_logits = teacher_outputs.logits.to(device=logits.device)
                    teacher_loss = distillation_kl_loss(
                        logits,
                        teacher_logits,
                        temperature=config.train.teacher_temperature,
                    )
                    loss = (
                        (1.0 - config.train.teacher_weight) * ce_loss
                        + config.train.teacher_weight * teacher_loss
                    )
                    train_teacher_loss += teacher_loss.item()
            train_loss += loss.item()
            train_ce_loss += ce_loss.item()
            (loss / config.train.grad_accum_steps).backward()
            tokens_since_log += inputs.numel()

        if config.train.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()
        if ema_model is not None:
            update_ema_model(ema_model, model, decay=config.train.ema_decay)
        if config.train.log_cuda_memory and device_is_cuda and not logged_first_step_memory:
            log_cuda_memory_stats("after_first_step")
            logged_first_step_memory = True

        if step % config.train.log_interval == 0 or step == initial_step + 1:
            now = time.time()
            elapsed = max(1e-6, now - python_last_log_time)
            tokens_per_second = format_tokens_per_second(tokens_since_log, elapsed)
            train_loss /= config.train.grad_accum_steps
            log_parts = [
                f"step={step:06d} "
                f"train_loss={train_loss:.4f} "
                f"tokens_per_second={tokens_per_second:.1f} "
                f"lr={max(current_adamw_lr, current_muon_lr):.6g} "
                f"train_context_len={train_context_len}",
            ]
            if teacher_model is not None:
                log_parts.append(
                    f"ce_loss={train_ce_loss / config.train.grad_accum_steps:.4f} "
                    f"teacher_kl={train_teacher_loss / config.train.grad_accum_steps:.4f}"
                )
            print(" ".join(log_parts))
            python_last_log_time = now
            tokens_since_log = 0

        should_eval = step % config.train.eval_interval == 0 or step == config.train.max_steps
        val_loss = best_val_loss
        val_ppl = best_val_ppl
        if should_eval:
            val_loss, val_ppl = evaluate_validation_loss(
                model,
                data_path=config.data.val_data_path,
                context_len=config.architecture.context_len,
                batch_size=config.train.eval_batch_size,
                device=device,
                max_batches=config.train.eval_batches,
                autocast_enabled=False,
                autocast_device_type=autocast_device_type,
            )
            print(
                f"step={step:06d} "
                f"val_loss={val_loss:.4f} "
                f"val_ppl={val_ppl:.4f}"
            )
            ema_val_loss: float | None = None
            ema_val_ppl: float | None = None
            if ema_model is not None and config.train.ema_eval:
                ema_val_loss, ema_val_ppl = evaluate_validation_loss(
                    ema_model,
                    data_path=config.data.val_data_path,
                    context_len=config.architecture.context_len,
                    batch_size=config.train.eval_batch_size,
                    device=ema_device,
                    max_batches=config.train.eval_batches,
                    autocast_enabled=False,
                    autocast_device_type="cuda" if ema_device.startswith("cuda") else "cpu",
                )
                print(
                    f"step={step:06d} "
                    f"ema_val_loss={ema_val_loss:.4f} "
                    f"ema_val_ppl={ema_val_ppl:.4f}"
                )
        else:
            ema_val_ppl = None

        should_save = (
            step % config.train.save_interval == 0
            or should_eval
            or step == config.train.max_steps
        )
        if should_save:
            latest_checkpoint = checkpoint_dir / "latest.pt"
            save_training_checkpoint(
                latest_checkpoint,
                model=model,
                optimizer_state=optimizer.state_dict(),
                config=config,
                step=step,
                val_ppl=val_ppl,
                ema_model=ema_model,
                ema_val_ppl=ema_val_ppl,
            )
            step_checkpoint = checkpoint_dir / f"ckpt_step{step:07d}.pt"
            save_training_checkpoint(
                step_checkpoint,
                model=model,
                optimizer_state=optimizer.state_dict(),
                config=config,
                step=step,
                val_ppl=val_ppl,
                ema_model=ema_model,
                ema_val_ppl=ema_val_ppl,
            )

        if should_eval and val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_val_loss = val_loss
            save_training_checkpoint(
                checkpoint_dir / "best.pt",
                model=model,
                optimizer_state=optimizer.state_dict(),
                config=config,
                step=step,
                val_ppl=val_ppl,
                ema_model=ema_model,
                ema_val_ppl=ema_val_ppl,
            )
            best_submission_dir = submission_root / "best"
            export_submission_bundle(model, config, best_submission_dir)

        if (
            should_eval
            and ema_model is not None
            and ema_val_ppl is not None
            and ema_val_loss is not None
            and ema_val_ppl < best_ema_val_ppl
        ):
            best_ema_val_ppl = ema_val_ppl
            best_ema_val_loss = ema_val_loss
            save_training_checkpoint(
                checkpoint_dir / "best_ema.pt",
                model=model,
                optimizer_state=optimizer.state_dict(),
                config=config,
                step=step,
                val_ppl=val_ppl,
                ema_model=ema_model,
                ema_val_ppl=ema_val_ppl,
            )
            best_ema_submission_dir = submission_root / "best_ema"
            export_submission_bundle(ema_model, config, best_ema_submission_dir)
            print(
                f"EMA best submission bundle saved to: {best_ema_submission_dir} "
                f"ema_val_ppl={best_ema_val_ppl:.4f}"
            )

        if should_eval and metric_callback is not None:
            metric_callback(
                {
                    "step": step,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "train_context_len": train_context_len,
                    "parameter_count": parameter_count,
                }
            )

    final_submission_dir = submission_root / "final"
    export_submission_bundle(model, config, final_submission_dir)
    ema_final_submission_dir: Path | None = None
    if ema_model is not None:
        ema_final_submission_dir = submission_root / "final_ema"
        export_submission_bundle(ema_model, config, ema_final_submission_dir)
        print(f"EMA final submission bundle saved to: {ema_final_submission_dir}")

    summary = TrainingSummary(
        run_id=run_id,
        config_name=config.name,
        config_hash=config_hash,
        parameter_count=parameter_count,
        best_val_ppl=best_val_ppl,
        best_val_loss=best_val_loss,
        steps_completed=config.train.max_steps,
        architecture_name=config.architecture.name,
        optimizer_name=config.optimizer.name,
        context_schedule=context_schedule_label,
        data_mixture=format_data_mixture(config.data),
        effective_batch_tokens=effective_batch_tokens(config.architecture, config.train),
        checkpoint_dir=str(checkpoint_dir),
        final_submission_dir=str(final_submission_dir),
        best_submission_dir=None if best_submission_dir is None else str(best_submission_dir),
        notes=notes,
        study_name=trial_metadata.study_name,
        trial_id=trial_metadata.trial_id,
        trial_index=trial_metadata.trial_index,
    )

    if append_results:
        append_experiment_result(Path("experiments") / "results.md", summary=summary)

    print(f"Best validation perplexity: {best_val_ppl:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    if best_submission_dir is not None:
        print(f"Best submission bundle saved to: {best_submission_dir}")
    if best_ema_submission_dir is not None:
        print(
            f"Best EMA validation perplexity: {best_ema_val_ppl:.4f} "
            f"bundle={best_ema_submission_dir}"
        )
    print(f"Final submission bundle saved to: {final_submission_dir}")
    if ema_model is not None:
        print("EMA export enabled: raw bundles were preserved and EMA bundles were exported separately.")
    return summary


def train_from_cli() -> None:
    """Parse CLI args, load config, and run training."""

    args = parse_args()
    config = load_experiment_config(args.config)
    run_training(
        config,
        notes=args.notes,
        trial_metadata=TrialMetadata(
            study_name=args.study_name,
            trial_id=args.trial_id,
            trial_index=args.trial_index,
        ),
    )


if __name__ == "__main__":
    train_from_cli()
