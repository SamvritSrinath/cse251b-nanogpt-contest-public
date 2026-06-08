"""Main training loop for iterative experiments and study-driven trials."""

from __future__ import annotations

import argparse
import math
import sys
import time
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
    compute_warmup_steps,
    cosine_with_warmup,
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
    save_training_checkpoint,
    save_yaml,
    set_optimizer_lr,
    set_seed,
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


def maybe_enable_gradient_checkpointing(model: torch.nn.Module, enabled: bool) -> None:
    """Enable model-supported activation checkpointing for large T4 runs."""

    if not enabled:
        return
    setter = getattr(model, "set_gradient_checkpointing", None)
    if setter is None:
        raise ValueError("Configured gradient_checkpointing=true, but model does not support it.")
    setter(True)


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

    model = build_model(config).to(device)
    maybe_enable_gradient_checkpointing(model, config.train.gradient_checkpointing)
    parameter_count = assert_parameter_budget(model)
    print(f"Model parameters: {parameter_count:,}")

    optimizer = build_optimizer(model, config.optimizer)
    model = maybe_compile(model, config.train.compile)

    manifests = load_source_manifests(config.data)
    print(f"Data sources: {summarize_manifests(manifests)}")
    train_loader = WeightedShardSampler(
        manifests,
        batch_size=config.train.batch_size,
        seed=config.train.seed,
    )

    warmup_steps = compute_warmup_steps(config.train)
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

    best_val_ppl = float("inf")
    best_val_loss = float("inf")
    best_submission_dir: Path | None = None
    tokens_since_log = 0
    python_last_log_time = time.time()

    context_schedule_label = format_context_schedule(
        config.train.context_schedule,
        max_context_len=config.architecture.context_len,
    )
    model.train()
    for step in range(1, config.train.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        train_context_len = resolve_context_length(
            config.train.context_schedule,
            max_context_len=config.architecture.context_len,
            step=step,
        )

        current_adamw_lr = cosine_with_warmup(
            step,
            warmup_steps=warmup_steps,
            total_steps=config.train.max_steps,
            max_lr=config.optimizer.adamw_lr,
            min_lr_ratio=config.train.min_lr_ratio,
        )
        current_muon_lr = cosine_with_warmup(
            step,
            warmup_steps=warmup_steps,
            total_steps=config.train.max_steps,
            max_lr=config.optimizer.muon_lr,
            min_lr_ratio=config.train.min_lr_ratio,
        )
        set_optimizer_lr(optimizer, adamw_lr=current_adamw_lr, muon_lr=current_muon_lr)

        for _ in range(config.train.grad_accum_steps):
            # Validation always uses the full configured context, but training batches
            # can follow a curriculum schedule to make early updates cheaper.
            inputs, targets = train_loader.next_batch(device, context_len=train_context_len)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            train_loss += loss.item()
            (loss / config.train.grad_accum_steps).backward()
            tokens_since_log += inputs.numel()

        if config.train.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()

        if step % config.train.log_interval == 0 or step == 1:
            now = time.time()
            elapsed = max(1e-6, now - python_last_log_time)
            tokens_per_second = format_tokens_per_second(tokens_since_log, elapsed)
            train_loss /= config.train.grad_accum_steps
            print(
                f"step={step:06d} "
                f"train_loss={train_loss:.4f} "
                f"tokens_per_second={tokens_per_second:.1f} "
                f"lr={max(current_adamw_lr, current_muon_lr):.6g} "
                f"train_context_len={train_context_len}"
            )
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
            )
            print(
                f"step={step:06d} "
                f"val_loss={val_loss:.4f} "
                f"val_ppl={val_ppl:.4f}"
            )

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
            )
            step_checkpoint = checkpoint_dir / f"ckpt_step{step:07d}.pt"
            save_training_checkpoint(
                step_checkpoint,
                model=model,
                optimizer_state=optimizer.state_dict(),
                config=config,
                step=step,
                val_ppl=val_ppl,
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
            )
            best_submission_dir = submission_root / "best"
            export_submission_bundle(model, config, best_submission_dir)

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
    print(f"Final submission bundle saved to: {final_submission_dir}")
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
