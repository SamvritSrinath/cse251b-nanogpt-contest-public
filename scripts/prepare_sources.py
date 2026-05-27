#!/usr/bin/env python3
"""Prepare data.sources entries that declare a config-native prepare block."""

from __future__ import annotations

import argparse
import os
import random
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare configured parquet-backed data sources.")
    parser.add_argument("--config", required=True, help="Experiment YAML with data.sources[].prepare.")
    parser.add_argument("--source", default=None, help="Only prepare the named source.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading or tokenizing.")
    parser.add_argument("--force-local", action="store_true", help="Skip GCS reuse even when GCS_DATA_ROOT is set.")
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def source_relpath(source: dict[str, Any]) -> str:
    path = Path(str(source["path"]))
    return str(path) if not path.is_absolute() else str(path.relative_to(REPO_ROOT))


def sources_with_prepare(config_path: Path, selected_source: str | None) -> list[dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    sources = raw.get("data", {}).get("sources", [])
    out = []
    for source in sources:
        if selected_source is not None and source.get("name") != selected_source:
            continue
        if source.get("prepare") is not None:
            out.append(dict(source))
    if selected_source is not None and not out:
        raise SystemExit(f"source has no prepare block or does not exist: {selected_source}")
    return out


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(f"prepare-sources: {shlex.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def gcloud_available() -> bool:
    return shutil.which("gcloud") is not None


def gcs_uri_for_source(source: dict[str, Any]) -> str:
    root = os.environ["GCS_DATA_ROOT"].rstrip("/")
    return f"{root}/{source_relpath(source).lstrip('/')}"


def gcs_has_bins(uri: str) -> bool:
    if not gcloud_available():
        return False
    for suffix in ("**", ""):
        result = subprocess.run(
            ["gcloud", "storage", "ls", f"{uri.rstrip('/')}/{suffix}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and any(line.strip().endswith(".bin") for line in result.stdout.splitlines()):
            return True
    return False


def try_pull_gcs(source: dict[str, Any], *, dry_run: bool, force_local: bool) -> bool:
    if force_local or not os.environ.get("GCS_DATA_ROOT"):
        return False
    uri = gcs_uri_for_source(source)
    if dry_run:
        print(f"prepare-sources: would check GCS for {uri}")
        return False
    if not gcs_has_bins(uri):
        print(f"prepare-sources: no token shards found at {uri}; preparing locally")
        return False
    out_dir = repo_path(str(source["path"]))
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(["gcloud", "storage", "cp", "--recursive", uri, str(out_dir.parent)], dry_run=False)
    return True


def hf_cli() -> str:
    for candidate in ("hf", "huggingface-cli"):
        found = shutil.which(candidate)
        if found is not None:
            return found
    raise SystemExit("prepare-sources: neither 'hf' nor 'huggingface-cli' is available on PATH")


def include_args(include: str | list[str] | None) -> list[str]:
    if include is None:
        return []
    values = include if isinstance(include, list) else [include]
    args: list[str] = []
    for value in values:
        args.extend(["--include", str(value)])
    return args


def reset_dir(path: Path) -> None:
    if not str(path.resolve()).startswith(str((REPO_ROOT / "data").resolve())):
        raise SystemExit(f"refusing to reset non-data directory: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parquet_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def maybe_select_parquets(
    root: Path,
    *,
    source_name: str,
    max_files: int | None,
    seed: int | None,
    dry_run: bool,
) -> Path:
    if max_files is None:
        return root
    files = parquet_files(root)
    if seed is None:
        selected = files[:max_files]
    else:
        rng = random.Random(seed)
        selected = sorted(rng.sample(files, min(max_files, len(files))))
    selected_root = REPO_ROOT / "data" / "selected-parquets" / source_name
    if dry_run:
        print(
            f"prepare-sources: would select {len(selected)} of {len(files)} parquet file(s) "
            f"into {selected_root}"
        )
        return selected_root
    reset_dir(selected_root)
    for path in selected:
        relative = path.relative_to(root)
        target = selected_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    return selected_root


def download_hf_parquets(
    source: dict[str, Any],
    prepare: dict[str, Any],
    *,
    dry_run: bool,
) -> Path:
    dataset = str(prepare.get("dataset", ""))
    if not dataset:
        raise SystemExit(f"{source['name']}: prepare.dataset is required")
    raw_dir = REPO_ROOT / "data" / "raw-parquets" / str(source["name"])
    cmd = [
        hf_cli() if not dry_run else "hf",
        "download",
        dataset,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(raw_dir),
        "--revision",
        str(prepare.get("revision", "main")),
    ]
    cmd.extend(include_args(prepare.get("include")))
    run_command(cmd, dry_run=dry_run)
    return raw_dir


def local_parquet_root(prepare: dict[str, Any]) -> Path:
    dataset = prepare.get("dataset")
    if dataset is None:
        raise SystemExit("local_parquet prepare.kind requires dataset to point at a local directory")
    root = repo_path(str(dataset))
    if not root.is_dir():
        raise SystemExit(f"local parquet directory not found: {root}")
    return root


def maybe_filter_parquets(
    source: dict[str, Any],
    prepare: dict[str, Any],
    input_root: Path,
    *,
    dry_run: bool,
) -> tuple[Path, str]:
    text_column = str(prepare.get("text_column", "text"))
    profile = prepare.get("filter_profile")
    if profile is None:
        return input_root, text_column

    filtered_root = REPO_ROOT / "data" / "filtered-parquets" / str(source["name"])
    if not dry_run:
        reset_dir(filtered_root)
    run_command(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "filter_parquet.py"),
            "--profile",
            str(profile),
            "--input-dir",
            str(input_root),
            "--output-dir",
            str(filtered_root),
            "--text-column",
            text_column,
        ],
        dry_run=dry_run,
    )
    return filtered_root, "text"


def run_corpus_prep(
    source: dict[str, Any],
    prepare: dict[str, Any],
    parquet_root: Path,
    *,
    text_column: str,
    dry_run: bool,
) -> None:
    out_dir = repo_path(str(source["path"]))
    shard_prefix = str(prepare.get("shard_prefix", str(source["name"]).replace("-", "_")))
    cmd = [
        str(REPO_ROOT / "scripts" / "corpus_prep.sh"),
        "--local-parquet-dir",
        str(parquet_root),
        "--out",
        str(out_dir),
        "--shard-prefix",
        shard_prefix,
        "--text-column",
        text_column,
        "--split-mode",
        str(prepare.get("split_mode", "train-only")),
        "--source-name",
        str(source["name"]),
    ]
    if bool(prepare.get("emit_doc_index", False)):
        cmd.append("--emit-doc-index")
        if prepare.get("doc_id_column") is not None or prepare.get("filter_profile") is not None:
            cmd.extend(["--doc-id-column", str(prepare.get("doc_id_column", "doc_id"))])
        if prepare.get("title_column") is not None or prepare.get("filter_profile") is not None:
            cmd.extend(["--title-column", str(prepare.get("title_column", "title"))])
        if prepare.get("section_column") is not None or prepare.get("filter_profile") == "s2orc_sections":
            cmd.extend(["--section-column", str(prepare.get("section_column", "section"))])
    if prepare.get("max_parquet_files") is not None and prepare.get("file_selection_seed") is None:
        cmd.extend(["--max-parquet-files", str(prepare["max_parquet_files"])])
    run_command(cmd, dry_run=dry_run)


def prepare_source(source: dict[str, Any], *, dry_run: bool, force_local: bool) -> None:
    prepare = dict(source["prepare"])
    if try_pull_gcs(source, dry_run=dry_run, force_local=force_local):
        return

    kind = str(prepare.get("kind", "hf_parquet"))
    if kind == "hf_parquet":
        parquet_root = download_hf_parquets(source, prepare, dry_run=dry_run)
    elif kind == "local_parquet":
        parquet_root = local_parquet_root(prepare)
    else:
        raise SystemExit(f"{source['name']}: unsupported prepare.kind '{kind}'")

    max_files = None if prepare.get("max_parquet_files") is None else int(prepare["max_parquet_files"])
    seed = None if prepare.get("file_selection_seed") is None else int(prepare["file_selection_seed"])
    parquet_root = maybe_select_parquets(
        parquet_root,
        source_name=str(source["name"]),
        max_files=max_files,
        seed=seed,
        dry_run=dry_run,
    )
    parquet_root, text_column = maybe_filter_parquets(source, prepare, parquet_root, dry_run=dry_run)
    run_corpus_prep(source, prepare, parquet_root, text_column=text_column, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    config_path = repo_path(args.config)
    sources = sources_with_prepare(config_path, args.source)
    if not sources:
        print(f"prepare-sources: no data.sources[].prepare blocks in {config_path}")
        return
    for source in sources:
        print(f"prepare-sources: === {source['name']} ===")
        prepare_source(source, dry_run=args.dry_run, force_local=args.force_local)


if __name__ == "__main__":
    main()
