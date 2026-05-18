"""Search harness for deterministic grid studies and Optuna-based tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
from optuna.pruners import MedianPruner

from src.train import run_training
from src.utils import (
    StudyConfig,
    TrialMetadata,
    TrainingSummary,
    conditions_match,
    ensure_directory,
    experiment_config_to_dict,
    load_experiment_config,
    load_json,
    load_study_config,
    materialize_experiment_config,
    save_json,
    save_yaml,
    training_summary_to_dict,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the search harness."""

    parser = argparse.ArgumentParser(description="Run grid or Optuna studies.")
    parser.add_argument("--study", type=str, required=True, help="Path to a study YAML file.")
    return parser.parse_args()


def parameter_values_for_grid(parameter: Any) -> list[Any]:
    """Expand one search parameter into deterministic grid values."""

    if parameter.type == "categorical":
        assert parameter.values is not None
        return list(parameter.values)
    if parameter.type == "int":
        if parameter.step is None:
            raise ValueError(f"Grid int parameter {parameter.path} requires 'step'.")
        return list(range(int(parameter.low), int(parameter.high) + 1, int(parameter.step)))
    if parameter.type == "float":
        if parameter.step is None:
            raise ValueError(f"Grid float parameter {parameter.path} requires 'step'.")
        values: list[float] = []
        current = float(parameter.low)
        while current <= float(parameter.high) + 1e-12:
            values.append(round(current, 10))
            current += float(parameter.step)
        return values
    raise ValueError(f"Unsupported grid parameter type: {parameter.type}")


def suggest_optuna_value(trial: optuna.Trial, parameter: Any) -> Any:
    """Sample one parameter value from an Optuna trial."""

    suggestion_name = parameter.path
    if parameter.conditions:
        digest = hashlib.sha1(
            json.dumps(parameter.conditions, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        # Optuna requires one stable distribution per suggestion name, so
        # conditionally active parameters get disambiguated trial keys.
        suggestion_name = f"{parameter.path}__{digest}"

    if parameter.type == "categorical":
        assert parameter.values is not None
        return trial.suggest_categorical(suggestion_name, parameter.values)
    if parameter.type == "int":
        return trial.suggest_int(
            suggestion_name,
            int(parameter.low),
            int(parameter.high),
            step=1 if parameter.step is None else int(parameter.step),
            log=parameter.log,
        )
    if parameter.type == "float":
        return trial.suggest_float(
            suggestion_name,
            float(parameter.low),
            float(parameter.high),
            step=None if parameter.log else parameter.step,
            log=parameter.log,
        )
    raise ValueError(f"Unsupported Optuna parameter type: {parameter.type}")


def enumerate_grid_overrides(study_config: StudyConfig, base_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate deterministic override combinations while respecting conditions."""

    overrides: list[dict[str, Any]] = []

    def recurse(index: int, current: dict[str, Any]) -> None:
        if index >= len(study_config.search_space):
            overrides.append(dict(current))
            return
        parameter = study_config.search_space[index]
        if not conditions_match(parameter.conditions, base_payload=base_payload, current_overrides=current):
            recurse(index + 1, current)
            return
        for value in parameter_values_for_grid(parameter):
            current[parameter.path] = value
            recurse(index + 1, current)
        current.pop(parameter.path, None)

    recurse(0, {})
    return overrides


def trial_output_dir(study_root: Path, trial_id: str) -> Path:
    """Resolve the output directory for one trial."""

    return study_root / "trials" / trial_id


def save_trial_summary(trial_dir: Path, summary: TrainingSummary) -> None:
    """Persist a per-trial summary in JSON and markdown form."""

    save_json(trial_dir / "summary.json", training_summary_to_dict(summary))
    markdown = (
        f"# {summary.trial_id or summary.run_id}\n\n"
        f"- Status: {summary.status}\n"
        f"- Config: {summary.config_name}\n"
        f"- Architecture: {summary.architecture_name}\n"
        f"- Optimizer: {summary.optimizer_name}\n"
        f"- Best val PPL: {summary.best_val_ppl:.4f}\n"
        f"- Params: {summary.parameter_count:,}\n"
        f"- Context schedule: {summary.context_schedule}\n"
        f"- Data mixture: {summary.data_mixture}\n"
        f"- Notes: {summary.notes or '-'}\n"
    )
    (trial_dir / "summary.md").write_text(markdown, encoding="utf-8")


def summarize_study(study_root: Path, study_config: StudyConfig, trial_summaries: list[TrainingSummary]) -> None:
    """Write machine-readable and markdown summaries for the full study."""

    if not trial_summaries:
        return
    best_summary = min(trial_summaries, key=lambda summary: summary.best_val_ppl)
    payload = {
        "name": study_config.name,
        "mode": study_config.mode,
        "best_trial": training_summary_to_dict(best_summary),
        "trials": [training_summary_to_dict(summary) for summary in trial_summaries],
    }
    save_json(study_root / "study_summary.json", payload)

    lines = [
        f"# Study: {study_config.name}",
        "",
        f"- Mode: {study_config.mode}",
        f"- Best trial: {best_summary.trial_id or best_summary.run_id}",
        f"- Best val PPL: {best_summary.best_val_ppl:.4f}",
        "",
        "| Trial | Arch | Optimizer | Context | Data Mix | Params | Val PPL | Status |",
        "|-------|------|-----------|---------|----------|--------|---------|--------|",
    ]
    for summary in trial_summaries:
        lines.append(
            f"| {summary.trial_id or summary.run_id} | {summary.architecture_name} | "
            f"{summary.optimizer_name} | {summary.context_schedule} | {summary.data_mixture} | "
            f"{summary.parameter_count:,} | {summary.best_val_ppl:.4f} | {summary.status} |"
        )
    (study_root / "study_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_grid_study(study_config: StudyConfig) -> None:
    """Run a deterministic grid study."""

    base_config = load_experiment_config(study_config.base_config)
    base_payload = experiment_config_to_dict(base_config)
    study_root = ensure_directory(Path(study_config.output_dir) / study_config.name)
    overrides_list = enumerate_grid_overrides(study_config, base_payload)
    trial_summaries: list[TrainingSummary] = []

    for trial_index, overrides in enumerate(overrides_list):
        trial_id = f"trial-{trial_index:04d}"
        trial_dir = ensure_directory(trial_output_dir(study_root, trial_id))
        summary_path = trial_dir / "summary.json"
        if study_config.resume and summary_path.exists():
            payload = load_json(summary_path)
            trial_summaries.append(TrainingSummary(**payload))
            continue

        resolved = materialize_experiment_config(
            base_config,
            overrides,
            run_name=f"{study_config.name}-{trial_id}",
        )
        resolved.train.checkpoint_dir = str(trial_dir / "checkpoints")
        resolved.train.submission_dir = str(trial_dir / "submission")
        save_yaml(trial_dir / "resolved_config.yaml", experiment_config_to_dict(resolved))

        summary = run_training(
            resolved,
            notes=f"study={study_config.name} trial={trial_id}",
            trial_metadata=TrialMetadata(
                study_name=study_config.name,
                trial_id=trial_id,
                trial_index=trial_index,
            ),
        )
        save_trial_summary(trial_dir, summary)
        trial_summaries.append(summary)

    summarize_study(study_root, study_config, trial_summaries)


def build_optuna_pruner(study_config: StudyConfig) -> optuna.pruners.BasePruner:
    """Build the configured Optuna pruner."""

    if study_config.pruner.name != "median":
        raise ValueError(f"Unsupported pruner: {study_config.pruner.name}")
    return MedianPruner(
        n_startup_trials=study_config.pruner.n_startup_trials,
        n_warmup_steps=study_config.pruner.n_warmup_steps,
        interval_steps=study_config.pruner.interval_steps,
    )


def run_optuna_study(study_config: StudyConfig) -> None:
    """Run an Optuna-backed search study."""

    base_config = load_experiment_config(study_config.base_config)
    base_payload = experiment_config_to_dict(base_config)
    study_root = ensure_directory(Path(study_config.output_dir) / study_config.name)
    storage = study_config.storage or f"sqlite:///{(study_root / 'optuna_study.db').resolve()}"
    direction = "minimize" if study_config.objective.direction == "minimize" else "maximize"

    study = optuna.create_study(
        study_name=study_config.name,
        storage=storage,
        direction=direction,
        pruner=build_optuna_pruner(study_config),
        load_if_exists=study_config.resume,
    )

    def objective(trial: optuna.Trial) -> float:
        overrides: dict[str, Any] = {}
        for parameter in study_config.search_space:
            if not conditions_match(parameter.conditions, base_payload=base_payload, current_overrides=overrides):
                continue
            overrides[parameter.path] = suggest_optuna_value(trial, parameter)

        trial_id = f"trial-{trial.number:04d}"
        trial_dir = ensure_directory(trial_output_dir(study_root, trial_id))
        resolved = materialize_experiment_config(
            base_config,
            overrides,
            run_name=f"{study_config.name}-{trial_id}",
        )
        resolved.train.checkpoint_dir = str(trial_dir / "checkpoints")
        resolved.train.submission_dir = str(trial_dir / "submission")
        save_yaml(trial_dir / "resolved_config.yaml", experiment_config_to_dict(resolved))

        def metric_callback(metrics: dict[str, Any]) -> None:
            # Optuna's median pruner expects intermediate reports keyed by training step.
            trial.report(float(metrics["val_ppl"]), step=int(metrics["step"]))
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            summary = run_training(
                resolved,
                notes=f"study={study_config.name} trial={trial_id}",
                trial_metadata=TrialMetadata(
                    study_name=study_config.name,
                    trial_id=trial_id,
                    trial_index=trial.number,
                ),
                metric_callback=metric_callback,
            )
            save_trial_summary(trial_dir, summary)
            trial.set_user_attr("summary_path", str(trial_dir / "summary.json"))
            trial.set_user_attr("best_val_ppl", summary.best_val_ppl)
            return summary.best_val_ppl
        except optuna.TrialPruned:
            pruned_summary = TrainingSummary(
                run_id=trial_id,
                config_name=resolved.name,
                config_hash="pruned",
                parameter_count=0,
                best_val_ppl=math.inf,
                best_val_loss=math.inf,
                steps_completed=0,
                architecture_name=resolved.architecture.name,
                optimizer_name=resolved.optimizer.name,
                context_schedule="pruned",
                data_mixture="pruned",
                effective_batch_tokens=0,
                checkpoint_dir=str(trial_dir / "checkpoints"),
                final_submission_dir=str(trial_dir / "submission"),
                best_submission_dir=None,
                notes=f"study={study_config.name} trial={trial_id}",
                status="pruned",
                study_name=study_config.name,
                trial_id=trial_id,
                trial_index=trial.number,
            )
            save_trial_summary(trial_dir, pruned_summary)
            trial.set_user_attr("summary_path", str(trial_dir / "summary.json"))
            raise

    study.optimize(
        objective,
        n_trials=study_config.max_trials,
        n_jobs=study_config.parallelism,
    )

    trial_summaries: list[TrainingSummary] = []
    for trial in study.trials:
        summary_path = trial.user_attrs.get("summary_path")
        if summary_path and Path(summary_path).exists():
            trial_summaries.append(TrainingSummary(**load_json(summary_path)))
    summarize_study(study_root, study_config, trial_summaries)


def main() -> None:
    """Dispatch to the requested study mode."""

    args = parse_args()
    study_config = load_study_config(args.study)
    if study_config.mode == "grid":
        run_grid_study(study_config)
        return
    if study_config.mode == "optuna":
        run_optuna_study(study_config)
        return
    raise ValueError(f"Unsupported study mode: {study_config.mode}")


if __name__ == "__main__":
    main()
