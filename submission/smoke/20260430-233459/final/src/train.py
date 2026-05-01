"""Main training loop for the contest baseline."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from src.data import ShardedTokenLoader, discover_training_shards, iter_validation_batches
from src.model import GPT
from src.optimizer import build_optimizer
from src.utils import (
    append_experiment_result,
    assert_parameter_budget,
    compute_warmup_steps,
    cosine_with_warmup,
    ensure_directory,
    export_submission_bundle,
    format_tokens_per_second,
    hash_experiment_config,
    load_experiment_config,
    make_run_id,
    maybe_enable_tf32,
    resolve_device,
    save_training_checkpoint,
    set_optimizer_lr,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""

    parser = argparse.ArgumentParser(description="Train the Parts 1–2 NanoGPT baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Short note recorded in experiments/results.md.",
    )
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
    """Compute validation loss and perplexity using evaluate.py-compatible slicing."""

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
    """Optionally compile the model with torch.compile."""

    if not enabled or not hasattr(torch, "compile"):
        return model
    return torch.compile(model)


def train() -> None:
    """Run the training loop described in the config file."""

    args = parse_args()
    config = load_experiment_config(args.config)
    run_id = make_run_id()
    config_hash = hash_experiment_config(config, notes=args.notes)

    maybe_enable_tf32()
    device = resolve_device(config.train.device)
    set_seed(config.train.seed)

    model = GPT(config.model).to(device)
    total_params = assert_parameter_budget(model)
    print(f"Model parameters: {total_params:,}")

    optimizer = build_optimizer(model, config.optimizer)
    model = maybe_compile(model, config.train.compile)

    shard_paths = discover_training_shards(config.data)
    train_loader = ShardedTokenLoader(
        shard_paths,
        context_len=config.model.context_len,
        batch_size=config.train.batch_size,
        seed=config.train.seed,
    )

    warmup_steps = compute_warmup_steps(config.train)
    checkpoint_dir = ensure_directory(Path(config.train.checkpoint_dir) / run_id)
    submission_root = ensure_directory(Path(config.train.submission_dir) / run_id)

    best_val_ppl = float("inf")
    best_val_loss = float("inf")
    tokens_since_log = 0
    python_last_log_time = time.time()

    model.train()
    for step in range(1, config.train.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0

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
            inputs, targets = train_loader.next_batch(device)
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
                f"lr={max(current_adamw_lr, current_muon_lr):.6g}"
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
                context_len=config.model.context_len,
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
            export_submission_bundle(model, config, submission_root / "best")

    export_submission_bundle(model, config, submission_root / "final")
    append_experiment_result(
        Path("experiments") / "results.md",
        run_id=run_id,
        config_name=config.name,
        config_hash=config_hash,
        data_fraction=config.data.data_fraction,
        steps=config.train.max_steps,
        val_ppl=best_val_ppl,
        notes=args.notes,
    )
    print(f"Best validation perplexity: {best_val_ppl:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    best_submission_dir = submission_root / "best"
    final_submission_dir = submission_root / "final"
    if best_submission_dir.exists():
        print(f"Best submission bundle saved to: {best_submission_dir}")
    print(f"Final submission bundle saved to: {final_submission_dir}")


if __name__ == "__main__":
    train()
