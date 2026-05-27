#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def safe_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def step_of(path: Path) -> int:
    match = re.search(r"step(\d+)", path.name)
    return int(match.group(1)) if match else -1


def extract_state_dict(state: Any, path: Path) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError(f"Expected checkpoint mapping at {path}.")
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Expected model state dict mapping at {path}.")
    return state_dict


def extract_config(state: Any, path: Path) -> dict[str, Any] | None:
    if isinstance(state, Mapping) and isinstance(state.get("config"), dict):
        return dict(state["config"])
    config_path = path.parent / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Average training checkpoints into a submission bundle.")
    parser.add_argument("--glob", required=True, help="Example: 'checkpoints/RUN/ckpt_step*.pt'")
    parser.add_argument("--last", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ckpts = sorted((Path(path) for path in glob.glob(args.glob)), key=step_of)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matched {args.glob}")
    if args.last <= 0:
        raise ValueError("--last must be positive")
    ckpts = ckpts[-args.last :]

    avg: dict[str, torch.Tensor] | None = None
    expected_keys: set[str] | None = None
    config: dict[str, Any] | None = None

    for ckpt_path in ckpts:
        state = safe_load(ckpt_path)
        state_dict = extract_state_dict(state, ckpt_path)
        keys = set(state_dict.keys())
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            missing = sorted(expected_keys - keys)[:5]
            extra = sorted(keys - expected_keys)[:5]
            raise ValueError(
                f"Checkpoint key mismatch at {ckpt_path}; missing={missing} extra={extra}"
            )
        if config is None:
            config = extract_config(state, ckpt_path)

        if avg is None:
            avg = {}
            for key, value in state_dict.items():
                detached = value.detach()
                avg[key] = detached.float().clone() if torch.is_floating_point(detached) else detached.clone()
        else:
            for key, value in state_dict.items():
                detached = value.detach()
                if torch.is_floating_point(detached):
                    avg[key].add_(detached.float())

    assert avg is not None
    for value in avg.values():
        if torch.is_floating_point(value):
            value.div_(len(ckpts))

    if config is None:
        raise ValueError("Could not find config payload in checkpoints or config.json.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(avg, out / "checkpoint.pt")
    (out / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "averaged_checkpoints.txt").write_text(
        "\n".join(str(path) for path in ckpts) + "\n",
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parent.parent
    shutil.copy2(repo_root / "model.py", out / "model.py")
    target_src = out / "src"
    if target_src.exists():
        shutil.rmtree(target_src)
    shutil.copytree(
        repo_root / "src",
        target_src,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


if __name__ == "__main__":
    main()
