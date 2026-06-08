"""Mixture-aware data loading and manifest caching for tokenized shard corpora.

References:
    - FineWeb and FineWeb-Edu dataset cards from Hugging Face document the
      public source data and curation process used as the default corpus here.
    - Karpathy's build-nanogpt project provides the GPT-2-tokenized shard format
      this repository expects under ``data/fineweb-edu``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from src.utils import DataConfig, DataSourceConfig, absolutize_from_cwd, ensure_directory, load_experiment_config

GPT2_EOT = 50256


@dataclass(slots=True)
class ShardInfo:
    """Metadata for one token shard."""

    path: str
    num_tokens: int
    size_bytes: int
    mtime_ns: int
    doc_index_path: str | None = None
    doc_index_size_bytes: int | None = None
    doc_index_mtime_ns: int | None = None


@dataclass(slots=True)
class SourceManifest:
    """Cached shard manifest for one weighted source."""

    source_name: str
    source_path: str
    glob: str
    weight: float
    notes: str
    sample_policy: str
    shards: list[ShardInfo]


@dataclass(frozen=True, slots=True)
class TokenRange:
    """A document or section span inside a shard."""

    start: int
    end: int
    doc_id: str
    title: str | None = None
    section: str | None = None
    continued_from_previous: bool = False
    continues_to_next: bool = False

    @property
    def length(self) -> int:
        return self.end - self.start


def manifest_cache_key(source: DataSourceConfig) -> str:
    """Create a stable cache key for a source manifest."""

    digest = hashlib.sha256(
        json.dumps(
            {
                "name": source.name,
                "path": str(absolutize_from_cwd(source.path)),
                "glob": source.glob,
                "weight": source.weight,
                "sample_policy": source.sample_policy,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    return f"{source.name}-{digest}.json"


def count_tokens_in_shard(path: Path) -> int:
    """Count tokens in a GPT-2-tokenized uint16 shard without loading it into RAM."""

    return path.stat().st_size // np.dtype(np.uint16).itemsize


def doc_index_path_for_shard(path: Path) -> Path:
    """Return the conventional sidecar path for a shard."""

    return Path(f"{path}.docs.json")


def is_validation_shard(path: Path, *, val_path: Path) -> bool:
    """Return true for public validation shards that must not enter training."""

    name = path.name
    return path.resolve() == val_path.resolve() or name == "val.bin" or "_val_" in name


def shard_info_from_path(path: Path) -> ShardInfo:
    """Build a ``ShardInfo`` record from the current filesystem state."""

    stat = path.stat()
    doc_index = doc_index_path_for_shard(path)
    doc_stat = doc_index.stat() if doc_index.is_file() else None
    return ShardInfo(
        path=str(path.resolve()),
        num_tokens=stat.st_size // np.dtype(np.uint16).itemsize,
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        doc_index_path=str(doc_index.resolve()) if doc_index.is_file() else None,
        doc_index_size_bytes=None if doc_stat is None else doc_stat.st_size,
        doc_index_mtime_ns=None if doc_stat is None else doc_stat.st_mtime_ns,
    )


def discover_source_shards(source: DataSourceConfig, *, val_path: Path) -> list[ShardInfo]:
    """Discover shard metadata for one source while excluding validation shards."""

    source_root = absolutize_from_cwd(source.path)
    shard_paths = [
        shard
        for shard in sorted(source_root.glob(source.glob))
        if shard.is_file() and not is_validation_shard(shard, val_path=val_path)
    ]
    if not shard_paths:
        raise FileNotFoundError(
            f"No training shards found for source '{source.name}' in {source_root} "
            f"matching glob {source.glob}."
        )
    return [shard_info_from_path(path) for path in shard_paths]


def shard_from_payload(payload: dict[str, object]) -> ShardInfo:
    """Load shard metadata from a manifest payload, tolerating old cache files."""

    path = str(payload["path"])
    num_tokens = int(payload["num_tokens"])
    size_bytes = int(payload.get("size_bytes", num_tokens * np.dtype(np.uint16).itemsize))
    mtime_ns = int(payload.get("mtime_ns", 0))
    raw_doc_index_path = payload.get("doc_index_path")
    doc_index_path = None if raw_doc_index_path is None else str(raw_doc_index_path)
    raw_doc_index_size_bytes = payload.get("doc_index_size_bytes")
    raw_doc_index_mtime_ns = payload.get("doc_index_mtime_ns")
    return ShardInfo(
        path=path,
        num_tokens=num_tokens,
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        doc_index_path=doc_index_path,
        doc_index_size_bytes=None if raw_doc_index_size_bytes is None else int(raw_doc_index_size_bytes),
        doc_index_mtime_ns=None if raw_doc_index_mtime_ns is None else int(raw_doc_index_mtime_ns),
    )


def manifest_from_payload(payload: dict[str, object]) -> SourceManifest:
    """Load a source manifest from JSON-compatible data."""

    return SourceManifest(
        source_name=str(payload["source_name"]),
        source_path=str(payload["source_path"]),
        glob=str(payload["glob"]),
        weight=float(payload["weight"]),
        notes=str(payload.get("notes", "")),
        sample_policy=str(payload.get("sample_policy", "random_window")),
        shards=[shard_from_payload(dict(shard)) for shard in payload["shards"]],  # type: ignore[index]
    )


def manifest_matches_source(
    manifest: SourceManifest,
    source: DataSourceConfig,
    *,
    val_path: Path,
) -> bool:
    """Return true when a cached manifest still matches the current shard tree."""

    if manifest.source_name != source.name:
        return False
    if manifest.source_path != str(absolutize_from_cwd(source.path)):
        return False
    if manifest.glob != source.glob:
        return False
    if manifest.weight != source.weight:
        return False
    if manifest.sample_policy != source.sample_policy:
        return False

    current = discover_source_shards(source, val_path=val_path)
    cached_fingerprint = [
        (
            shard.path,
            shard.num_tokens,
            shard.size_bytes,
            shard.mtime_ns,
            shard.doc_index_path,
            shard.doc_index_size_bytes,
            shard.doc_index_mtime_ns,
        )
        for shard in manifest.shards
    ]
    current_fingerprint = [
        (
            shard.path,
            shard.num_tokens,
            shard.size_bytes,
            shard.mtime_ns,
            shard.doc_index_path,
            shard.doc_index_size_bytes,
            shard.doc_index_mtime_ns,
        )
        for shard in current
    ]
    return cached_fingerprint == current_fingerprint


def build_or_load_manifest(
    source: DataSourceConfig,
    *,
    manifest_dir: str | Path,
    val_path: Path,
    rebuild: bool = False,
) -> SourceManifest:
    """Load a cached manifest or build it by scanning the source directory."""

    manifest_dir = ensure_directory(manifest_dir)
    manifest_path = manifest_dir / manifest_cache_key(source)
    if manifest_path.exists() and not rebuild:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = manifest_from_payload(payload)
        try:
            if manifest_matches_source(manifest, source, val_path=val_path):
                return manifest
        except FileNotFoundError:
            pass

    manifest = SourceManifest(
        source_name=source.name,
        source_path=str(absolutize_from_cwd(source.path)),
        glob=source.glob,
        weight=source.weight,
        notes=source.notes,
        sample_policy=source.sample_policy,
        shards=discover_source_shards(source, val_path=val_path),
    )
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_source_manifests(data_config: DataConfig, *, rebuild: bool = False) -> list[SourceManifest]:
    """Load manifests for every configured source in the mixture."""

    val_path = absolutize_from_cwd(data_config.val_data_path)
    return [
        build_or_load_manifest(
            source,
            manifest_dir=absolutize_from_cwd(data_config.manifest_dir),
            val_path=val_path,
            rebuild=rebuild,
        )
        for source in data_config.sources
    ]


def summarize_manifests(manifests: list[SourceManifest]) -> str:
    """Build a short human-readable summary for logging and docs."""

    parts = []
    for manifest in manifests:
        token_count = sum(shard.num_tokens for shard in manifest.shards)
        parts.append(
            f"{manifest.source_name} weight={manifest.weight:g} "
            f"policy={manifest.sample_policy} shards={len(manifest.shards)} "
            f"tokens={token_count:,}"
        )
    return "; ".join(parts)


class WeightedShardSampler:
    """Batch sampler over a weighted mixture of shard sources."""

    def __init__(
        self,
        manifests: list[SourceManifest],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if not manifests:
            raise ValueError("Expected at least one source manifest.")
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._source_weights = np.asarray([manifest.weight for manifest in manifests], dtype=np.float64)
        if self._source_weights.sum() <= 0:
            raise ValueError("Source weights must sum to a positive value.")
        self._source_weights = self._source_weights / self._source_weights.sum()
        self._sources = [
            {
                "manifest": manifest,
                # Sampling shards proportional to token count avoids oversampling tiny files.
                "shard_probs": self._normalize_lengths([shard.num_tokens for shard in manifest.shards]),
            }
            for manifest in manifests
        ]
        self._memmaps: dict[str, np.memmap] = {}
        self._doc_ranges: dict[str, list[TokenRange]] = {}

    @staticmethod
    def _normalize_lengths(lengths: list[int] | list[float]) -> np.ndarray:
        """Normalize positive lengths into probabilities."""

        values = np.asarray(lengths, dtype=np.float64)
        if values.sum() <= 0:
            raise ValueError("Lengths must sum to a positive value.")
        return values / values.sum()

    def _get_tokens(self, shard_path: str) -> np.memmap:
        """Reuse memmaps so repeated batches do not reopen the same shard."""

        if shard_path not in self._memmaps:
            self._memmaps[shard_path] = np.memmap(shard_path, dtype=np.uint16, mode="r")
        return self._memmaps[shard_path]

    def _sample_source(self) -> dict[str, object]:
        """Sample one source according to the configured mixture weights."""

        source_index = int(self._rng.choice(len(self._sources), p=self._source_weights))
        return self._sources[source_index]

    def _load_doc_ranges(self, shard: ShardInfo) -> list[TokenRange]:
        """Load and validate document/section ranges for a shard sidecar."""

        if shard.path in self._doc_ranges:
            return self._doc_ranges[shard.path]
        if shard.doc_index_path is None:
            raise FileNotFoundError(
                f"Source policy requires a document sidecar, but {shard.path} has none."
            )
        payload = json.loads(Path(shard.doc_index_path).read_text(encoding="utf-8"))
        token_count = int(payload.get("token_count", shard.num_tokens))
        if token_count != shard.num_tokens:
            raise ValueError(
                f"Document sidecar token_count mismatch for {shard.path}: "
                f"{token_count} != {shard.num_tokens}."
            )

        raw_ranges = payload.get("ranges", payload.get("documents", []))
        ranges = []
        for raw_range in raw_ranges:
            item = dict(raw_range)
            start = int(item["start"])
            end = int(item["end"])
            if start < 0 or end <= start or end > shard.num_tokens:
                raise ValueError(f"Invalid document range {start}:{end} in {shard.doc_index_path}.")
            ranges.append(
                TokenRange(
                    start=start,
                    end=end,
                    doc_id=str(item.get("doc_id", "")),
                    title=None if item.get("title") is None else str(item.get("title")),
                    section=None if item.get("section") is None else str(item.get("section")),
                    continued_from_previous=bool(item.get("continued_from_previous", False)),
                    continues_to_next=bool(item.get("continues_to_next", False)),
                )
            )
        if not ranges:
            raise ValueError(f"Document sidecar has no ranges: {shard.doc_index_path}.")
        self._doc_ranges[shard.path] = ranges
        return ranges

    def _eligible_doc_shards(
        self,
        manifest: SourceManifest,
        *,
        context_len: int,
        policy: str,
    ) -> list[tuple[ShardInfo, list[TokenRange], int]]:
        """Return shards with ranges usable by the requested document-aware policy."""

        min_tokens = context_len + 1
        eligible = []
        for shard in manifest.shards:
            if shard.doc_index_path is None:
                expected = doc_index_path_for_shard(Path(shard.path))
                raise FileNotFoundError(
                    f"sample_policy={policy} for source '{manifest.source_name}' requires "
                    f"a document sidecar for every shard. Missing {expected}. "
                    "Prepare this source with emit_doc_index: true."
                )
            ranges = self._load_doc_ranges(shard)
            if policy == "section_window":
                section_ranges = [item for item in ranges if item.section]
                if section_ranges:
                    ranges = section_ranges
            if policy == "packed_short_docs":
                usable = [item for item in ranges if item.length > 0]
                total_weight = sum(item.length for item in usable)
            else:
                usable = [item for item in ranges if item.length >= min_tokens]
                total_weight = sum(item.length - context_len for item in usable)
            if usable and total_weight > 0:
                eligible.append((shard, usable, total_weight))
        if not eligible:
            raise ValueError(
                f"No document ranges in source '{manifest.source_name}' are usable "
                f"for context length {context_len} with policy {policy}."
            )
        return eligible

    def _sample_random_windows(
        self,
        manifest: SourceManifest,
        shard_probs: np.ndarray,
        *,
        context_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample ordinary random windows from one randomly selected shard."""

        shard_index = int(self._rng.choice(len(manifest.shards), p=shard_probs))
        shard = manifest.shards[shard_index]
        tokens = self._get_tokens(shard.path)
        if len(tokens) <= context_len:
            raise ValueError(f"Shard {shard.path} is too short for context length {context_len}.")

        max_start = len(tokens) - context_len - 1
        starts = self._rng.integers(0, max_start + 1, size=self.batch_size, endpoint=False)
        inputs = np.stack([tokens[start : start + context_len] for start in starts]).astype(np.int64)
        targets = np.stack(
            [tokens[start + 1 : start + context_len + 1] for start in starts]
        ).astype(np.int64)
        return inputs, targets

    def _sample_bounded_windows(
        self,
        manifest: SourceManifest,
        *,
        context_len: int,
        policy: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample windows that remain inside a selected document or section range."""

        eligible = self._eligible_doc_shards(manifest, context_len=context_len, policy=policy)
        shard_probs = self._normalize_lengths([item[2] for item in eligible])
        shard_index = int(self._rng.choice(len(eligible), p=shard_probs))
        shard, ranges, _ = eligible[shard_index]
        range_weights = [item.length - context_len for item in ranges]
        range_probs = self._normalize_lengths(range_weights)
        tokens = self._get_tokens(shard.path)

        inputs = []
        targets = []
        for _ in range(self.batch_size):
            range_index = int(self._rng.choice(len(ranges), p=range_probs))
            token_range = ranges[range_index]
            max_start = token_range.end - context_len - 1
            start = int(self._rng.integers(token_range.start, max_start + 1, endpoint=False))
            inputs.append(tokens[start : start + context_len])
            targets.append(tokens[start + 1 : start + context_len + 1])
        return np.stack(inputs).astype(np.int64), np.stack(targets).astype(np.int64)

    def _sample_packed_short_docs(
        self,
        manifest: SourceManifest,
        *,
        context_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pack whole or partial document ranges into each training example."""

        eligible = self._eligible_doc_shards(
            manifest,
            context_len=context_len,
            policy="packed_short_docs",
        )
        shard_probs = self._normalize_lengths([item[2] for item in eligible])

        packed = []
        needed = context_len + 1
        for _ in range(self.batch_size):
            pieces: list[np.ndarray] = []
            total = 0
            while total < needed:
                if pieces and total < needed:
                    pieces.append(np.asarray([GPT2_EOT], dtype=np.uint16))
                    total += 1
                    if total >= needed:
                        break
                shard_index = int(self._rng.choice(len(eligible), p=shard_probs))
                shard, ranges, _ = eligible[shard_index]
                range_probs = self._normalize_lengths([item.length for item in ranges])
                range_index = int(self._rng.choice(len(ranges), p=range_probs))
                token_range = ranges[range_index]
                take = min(needed - total, token_range.length)
                tokens = self._get_tokens(shard.path)
                piece = np.asarray(tokens[token_range.start : token_range.start + take], dtype=np.uint16)
                pieces.append(piece)
                total += len(piece)
            packed.append(np.concatenate(pieces)[:needed])

        examples = np.stack(packed).astype(np.int64)
        return examples[:, :-1], examples[:, 1:]

    def next_batch(self, device: str, *, context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random training batch from the weighted source mixture."""

        source_record = self._sample_source()
        manifest = source_record["manifest"]
        shard_probs = source_record["shard_probs"]
        assert isinstance(manifest, SourceManifest)
        assert isinstance(shard_probs, np.ndarray)

        if manifest.sample_policy == "random_window":
            inputs, targets = self._sample_random_windows(
                manifest,
                shard_probs,
                context_len=context_len,
            )
        elif manifest.sample_policy in {"document_window", "section_window"}:
            inputs, targets = self._sample_bounded_windows(
                manifest,
                context_len=context_len,
                policy=manifest.sample_policy,
            )
        elif manifest.sample_policy == "packed_short_docs":
            inputs, targets = self._sample_packed_short_docs(manifest, context_len=context_len)
        else:
            raise ValueError(f"Unsupported sample policy: {manifest.sample_policy}.")

        return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)


def iter_validation_batches(
    data_path: str | Path,
    *,
    context_len: int,
    batch_size: int,
    device: str,
    max_batches: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield validation batches aligned with ``evaluate.py`` semantics."""

    data = np.memmap(absolutize_from_cwd(data_path), dtype=np.uint16, mode="r")
    n_chunks = (len(data) - 1) // context_len
    n_chunks = (n_chunks // batch_size) * batch_size
    yielded_batches = 0

    for chunk_start in range(0, n_chunks, batch_size):
        if max_batches is not None and yielded_batches >= max_batches:
            return
        inputs = np.stack(
            [
                data[index * context_len : index * context_len + context_len]
                for index in range(chunk_start, chunk_start + batch_size)
            ]
        ).astype(np.int64)
        targets = np.stack(
            [
                data[index * context_len + 1 : index * context_len + context_len + 1]
                for index in range(chunk_start, chunk_start + batch_size)
            ]
        ).astype(np.int64)
        yielded_batches += 1
        yield torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for manifest creation."""

    parser = argparse.ArgumentParser(description="Build or inspect data manifests.")
    parser.add_argument("--config", type=str, required=True, help="Experiment config path.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore cached manifests and rescan current shard files.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the manifest summary after building/loading cached manifests.",
    )
    return parser.parse_args()


def main() -> None:
    """Build manifests for the configured data sources and print a summary."""

    args = parse_args()
    config = load_experiment_config(args.config)
    manifests = load_source_manifests(config.data, rebuild=args.rebuild)
    print(summarize_manifests(manifests))


if __name__ == "__main__":
    main()
