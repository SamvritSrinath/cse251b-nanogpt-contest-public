#!/usr/bin/env python3
"""Prepare data.sources entries that declare a config-native prepare block."""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
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


def source_shards(source: dict[str, Any]) -> list[Path]:
    out_dir = repo_path(str(source["path"]))
    glob = str(source.get("glob", "**/*.bin"))
    return sorted(path for path in out_dir.glob(glob) if path.is_file())


def source_success_marker(source: dict[str, Any]) -> Path:
    return repo_path(str(source["path"])) / "_PREPARE_SUCCESS"


def tokenized_shards_exist(source: dict[str, Any]) -> bool:
    return source_success_marker(source).is_file() and any(
        "_train_" in path.name and path.suffix == ".bin" for path in source_shards(source)
    )


def write_success_marker(source: dict[str, Any]) -> None:
    marker = source_success_marker(source)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")


def assert_no_val_shards(source: dict[str, Any]) -> None:
    bad = [
        path
        for path in source_shards(source)
        if path.name == "val.bin" or "_val_" in path.name
    ]
    if bad:
        joined = "\n  ".join(str(path) for path in bad[:20])
        raise SystemExit(f"{source['name']}: validation shard(s) found in source output:\n  {joined}")


def print_disk_usage(label: str, *paths: Path) -> None:
    print(f"prepare-sources: disk usage ({label})")
    subprocess.run(["df", "-h", "/"], check=False)
    for path in paths:
        subprocess.run(["du", "-sh", str(path)], check=False)


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
    assert_no_val_shards(source)
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


def pattern_values(values: str | list[str] | None) -> list[str]:
    if values is None:
        return []
    return [str(value) for value in (values if isinstance(values, list) else [values])]


def matches_any(path: str, patterns: list[str]) -> bool:
    return not patterns or any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def pattern_args(flag: str, values: str | list[str] | None) -> list[str]:
    if values is None:
        return []
    items = values if isinstance(values, list) else [values]
    args: list[str] = []
    for value in items:
        args.extend([flag, str(value)])
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


def cleanup_generated_parquet_dirs(source_name: str, *, keep_raw_parquet: bool) -> None:
    if keep_raw_parquet:
        return
    for base in ("raw-parquets", "filtered-parquets", "selected-parquets", "stream-json-parquets"):
        path = REPO_ROOT / "data" / base / source_name
        if path.exists():
            print(f"prepare-sources: removing temporary parquet cache: {path}")
            shutil.rmtree(path)


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
    max_files = None if prepare.get("max_parquet_files") is None else int(prepare["max_parquet_files"])
    if max_files is not None:
        if dry_run:
            print(
                f"prepare-sources: would list {dataset} and download up to {max_files} parquet file(s) "
                f"into {raw_dir}"
            )
            return raw_dir
        from huggingface_hub import list_repo_files  # type: ignore[import-not-found]

        include = pattern_values(prepare.get("include"))
        exclude = pattern_values(prepare.get("exclude"))
        files = sorted(
            path
            for path in list_repo_files(dataset, repo_type="dataset", revision=str(prepare.get("revision", "main")))
            if path.endswith(".parquet")
            and matches_any(path, include)
            and not any(fnmatch.fnmatch(path, pattern) for pattern in exclude)
        )
        if not files:
            raise SystemExit(f"{source['name']}: no parquet files matched include/exclude patterns")
        selected = files[:max_files]
        raw_dir.mkdir(parents=True, exist_ok=True)
        for relpath in selected:
            enforce_max_download_gb(raw_dir, prepare, str(source["name"]))
            run_command(
                [
                    hf_cli(),
                    "download",
                    dataset,
                    relpath,
                    "--repo-type",
                    "dataset",
                    "--local-dir",
                    str(raw_dir),
                    "--revision",
                    str(prepare.get("revision", "main")),
                ],
                dry_run=False,
            )
            enforce_max_download_gb(raw_dir, prepare, str(source["name"]))
        return raw_dir

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
    cmd.extend(pattern_args("--exclude", prepare.get("exclude")))
    run_command(cmd, dry_run=dry_run)
    return raw_dir


def hf_json_relpaths(prepare: dict[str, Any]) -> list[str]:
    from huggingface_hub import list_repo_files  # type: ignore[import-not-found]

    dataset = str(prepare["dataset"])
    revision = str(prepare.get("revision", "main"))
    include = pattern_values(prepare.get("include"))
    exclude = pattern_values(prepare.get("exclude"))
    files = sorted(
        path
        for path in list_repo_files(dataset, repo_type="dataset", revision=revision)
        if (path.endswith(".json") or path.endswith(".jsonl") or path.endswith(".json.gz") or path.endswith(".jsonl.gz"))
        and matches_any(path, include)
        and not any(fnmatch.fnmatch(path, pattern) for pattern in exclude)
    )
    if not files:
        raise SystemExit(f"{dataset}: no JSON/JSONL files matched include/exclude patterns")
    max_files = prepare.get("max_source_files")
    if max_files is not None:
        files = files[: int(max_files)]
    return files


def iter_json_objects_from_bytes(raw: Any, *, gzip_compressed: bool, label: str) -> Any:
    stream = gzip.GzipFile(fileobj=raw) if gzip_compressed else raw
    bad_lines = 0
    for line_number, raw_line in enumerate(stream, start=1):
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            if bad_lines <= 5:
                print(f"prepare-sources: skipping malformed JSON line {label}:{line_number}", flush=True)
            continue
        if isinstance(payload, dict):
            yield payload
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item
    if bad_lines:
        print(f"prepare-sources: skipped {bad_lines} malformed JSON line(s) in {label}", flush=True)


def iter_hf_json_rows(prepare: dict[str, Any]) -> Any:
    from huggingface_hub import HfFileSystem  # type: ignore[import-not-found]

    dataset = str(prepare["dataset"])
    revision = str(prepare.get("revision", "main"))
    fs = HfFileSystem()
    for relpath in hf_json_relpaths(prepare):
        hf_path = f"datasets/{dataset}@{revision}/{relpath}"
        print(f"prepare-sources: streaming {hf_path}", flush=True)
        try:
            with fs.open(hf_path, "rb") as handle:
                yield from iter_json_objects_from_bytes(
                    handle,
                    gzip_compressed=relpath.endswith(".gz"),
                    label=relpath,
                )
        except Exception as exc:
            if bool(prepare.get("skip_bad_json_files", True)):
                print(f"prepare-sources: skipping unreadable JSON file {relpath}: {exc}", flush=True)
                continue
            raise


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
    cmd = [
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
    ]
    for key, flag in (
        ("min_chars", "--min-chars"),
        ("max_chars", "--max-chars"),
        ("line_quality_threshold", "--line-quality-threshold"),
    ):
        if prepare.get(key) is not None:
            cmd.extend([flag, str(prepare[key])])
    if bool(prepare.get("rewrite_low_quality_lines", False)):
        cmd.append("--rewrite-low-quality-lines")
    run_command(cmd, dry_run=dry_run)
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


def normalize_stream_row(row: dict[str, Any], *, source_name: str, row_index: int) -> dict[str, str] | None:
    text = str(row.get("text") or "").strip()
    if not text:
        return None
    doc_id = row.get("doc_id") or row.get("id") or row.get("url") or f"{source_name}:{row_index}"
    title = row.get("title") or row.get("name") or ""
    section = row.get("section") or row.get("source") or source_name
    return {
        "text": text,
        "doc_id": str(doc_id),
        "title": "" if title is None else str(title),
        "source": source_name,
        "section": "" if section is None else str(section),
    }


def write_normalized_parquet(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def temp_tree_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def enforce_max_temp_gb(path: Path, prepare: dict[str, Any], source_name: str) -> None:
    if prepare.get("max_temp_gb") is None:
        return
    limit_gb = float(prepare["max_temp_gb"])
    if temp_tree_size_bytes(path) > limit_gb * 1024**3:
        raise SystemExit(f"{source_name}: temporary parquet data exceeded max_temp_gb={limit_gb}")


def enforce_max_download_gb(path: Path, prepare: dict[str, Any], source_name: str) -> None:
    if prepare.get("max_download_gb") is None:
        return
    limit_gb = float(prepare["max_download_gb"])
    if temp_tree_size_bytes(path) > limit_gb * 1024**3:
        raise SystemExit(f"{source_name}: downloaded raw data exceeded max_download_gb={limit_gb}")


def prepare_hf_stream_json(source: dict[str, Any], prepare: dict[str, Any], *, dry_run: bool) -> None:
    dataset = str(prepare.get("dataset", ""))
    if not dataset:
        raise SystemExit(f"{source['name']}: prepare.dataset is required")

    docs_per_shard = int(prepare.get("docs_per_temp_shard", 50_000))
    max_docs = None if prepare.get("max_docs") is None else int(prepare["max_docs"])
    max_bytes = None if prepare.get("max_bytes") is None else int(prepare["max_bytes"])
    max_temp_gb = None if prepare.get("max_temp_gb") is None else float(prepare["max_temp_gb"])
    subset = prepare.get("subset", prepare.get("hf_subset"))
    source_name = str(source["name"])
    temp_root = REPO_ROOT / "data" / "stream-json-parquets" / source_name
    keep_raw = bool(prepare.get("keep_raw_parquet", False))

    if dry_run:
        sample = [
            normalize_stream_row({"text": "dry run text", "id": "dry-1"}, source_name=source_name, row_index=0)
        ]
        print(
            f"prepare-sources: would stream {dataset} split=train subset={subset or '<default>'} "
            f"into {docs_per_shard} doc parquet batches under {temp_root}"
        )
        print(f"prepare-sources: dry-run normalized example: {sample[0]}")
        return

    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    if subset is not None:
        print(f"prepare-sources: hf_stream_json ignores subset/hf_subset for file-backed JSON repos: {subset}")
    stream = iter_hf_json_rows(prepare)
    batch: list[dict[str, str]] = []
    total_docs = 0
    total_text_bytes = 0
    shard_index = 0
    def flush_batch() -> None:
        nonlocal batch, shard_index
        if not batch:
            return
        batch_dir = temp_root / f"batch_{shard_index:06}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = batch_dir / f"{source_name}_{shard_index:06}.parquet"
        write_normalized_parquet(batch, parquet_path)
        if max_temp_gb is not None and temp_tree_size_bytes(temp_root) > max_temp_gb * 1024**3:
            raise SystemExit(f"{source_name}: temp parquet cache exceeded max_temp_gb={max_temp_gb}")
        batch_prepare = dict(prepare)
        base_prefix = str(prepare.get("shard_prefix", source_name.replace("-", "_")))
        batch_prepare["shard_prefix"] = f"{base_prefix}_{shard_index:06}"
        run_corpus_prep(source, batch_prepare, batch_dir, text_column="text", dry_run=False)
        if not keep_raw:
            shutil.rmtree(batch_dir)
        batch = []
        shard_index += 1
        print_disk_usage(f"after {source_name} temp shard {shard_index}", temp_root, repo_path(str(source["path"])))

    for row_index, row in enumerate(stream):
        normalized = normalize_stream_row(dict(row), source_name=source_name, row_index=row_index)
        if normalized is None:
            continue
        text_bytes = len(normalized["text"].encode("utf-8"))
        if max_docs is not None and total_docs >= max_docs:
            break
        if max_bytes is not None and total_text_bytes + text_bytes > max_bytes:
            break
        batch.append(normalized)
        total_docs += 1
        total_text_bytes += text_bytes
        if len(batch) >= docs_per_shard:
            flush_batch()

    flush_batch()
    if not keep_raw and temp_root.exists():
        shutil.rmtree(temp_root)
    print_disk_usage(f"after {source_name}", repo_path(str(source["path"])), REPO_ROOT / "data")


def prepare_source(source: dict[str, Any], *, dry_run: bool, force_local: bool) -> None:
    prepare = dict(source["prepare"])
    if tokenized_shards_exist(source):
        print(f"prepare-sources: {source['name']} already has tokenized train shards; skipping")
        assert_no_val_shards(source)
        return
    if try_pull_gcs(source, dry_run=dry_run, force_local=force_local):
        return

    kind = str(prepare.get("kind", "hf_parquet"))
    source_name = str(source["name"])
    print_disk_usage(f"before {source_name}", REPO_ROOT / "data")
    try:
        if kind == "hf_stream_json":
            prepare_hf_stream_json(source, prepare, dry_run=dry_run)
            assert_no_val_shards(source)
            if not dry_run:
                write_success_marker(source)
            return
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
            source_name=source_name,
            max_files=max_files,
            seed=seed,
            dry_run=dry_run,
        )
        if not dry_run:
            enforce_max_temp_gb(parquet_root, prepare, source_name)
        parquet_root, text_column = maybe_filter_parquets(source, prepare, parquet_root, dry_run=dry_run)
        if not dry_run:
            enforce_max_temp_gb(parquet_root, prepare, source_name)
        run_corpus_prep(source, prepare, parquet_root, text_column=text_column, dry_run=dry_run)
        assert_no_val_shards(source)
        if not dry_run:
            write_success_marker(source)
    finally:
        cleanup_generated_parquet_dirs(
            source_name,
            keep_raw_parquet=bool(prepare.get("keep_raw_parquet", False)) or dry_run or kind == "local_parquet",
        )
        print_disk_usage(f"after {source_name}", repo_path(str(source["path"])), REPO_ROOT / "data")


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
