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


@dataclass(slots=True)
class ShardInfo:
    """Metadata for one token shard."""

    path: str
    num_tokens: int


@dataclass(slots=True)
class SourceManifest:
    """Cached shard manifest for one weighted source."""

    source_name: str
    source_path: str
    glob: str
    weight: float
    notes: str
    shards: list[ShardInfo]


def manifest_cache_key(source: DataSourceConfig) -> str:
    """Create a stable cache key for a source manifest."""

    digest = hashlib.sha256(
        json.dumps(
            {
                "name": source.name,
                "path": str(absolutize_from_cwd(source.path)),
                "glob": source.glob,
                "weight": source.weight,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:10]
    return f"{source.name}-{digest}.json"


def count_tokens_in_shard(path: Path) -> int:
    """Count tokens in a GPT-2-tokenized uint16 shard without loading it into RAM."""

    return path.stat().st_size // np.dtype(np.uint16).itemsize


def discover_source_shards(source: DataSourceConfig, *, val_path: Path) -> list[ShardInfo]:
    """Discover shard metadata for one source while excluding ``val.bin``."""

    source_root = absolutize_from_cwd(source.path)
    shard_paths = [
        shard
        for shard in sorted(source_root.glob(source.glob))
        if shard.is_file() and shard.resolve() != val_path and shard.name != "val.bin"
    ]
    if not shard_paths:
        raise FileNotFoundError(
            f"No training shards found for source '{source.name}' in {source_root} "
            f"matching glob {source.glob}."
        )
    return [ShardInfo(path=str(path.resolve()), num_tokens=count_tokens_in_shard(path)) for path in shard_paths]


def build_or_load_manifest(
    source: DataSourceConfig,
    *,
    manifest_dir: str | Path,
    val_path: Path,
) -> SourceManifest:
    """Load a cached manifest or build it by scanning the source directory."""

    manifest_dir = ensure_directory(manifest_dir)
    manifest_path = manifest_dir / manifest_cache_key(source)
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return SourceManifest(
            source_name=str(payload["source_name"]),
            source_path=str(payload["source_path"]),
            glob=str(payload["glob"]),
            weight=float(payload["weight"]),
            notes=str(payload.get("notes", "")),
            shards=[ShardInfo(**shard) for shard in payload["shards"]],
        )

    manifest = SourceManifest(
        source_name=source.name,
        source_path=str(absolutize_from_cwd(source.path)),
        glob=source.glob,
        weight=source.weight,
        notes=source.notes,
        shards=discover_source_shards(source, val_path=val_path),
    )
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_source_manifests(data_config: DataConfig) -> list[SourceManifest]:
    """Load manifests for every configured source in the mixture."""

    val_path = absolutize_from_cwd(data_config.val_data_path)
    return [
        build_or_load_manifest(
            source,
            manifest_dir=absolutize_from_cwd(data_config.manifest_dir),
            val_path=val_path,
        )
        for source in data_config.sources
    ]


def summarize_manifests(manifests: list[SourceManifest]) -> str:
    """Build a short human-readable summary for logging and docs."""

    parts = []
    for manifest in manifests:
        token_count = sum(shard.num_tokens for shard in manifest.shards)
        parts.append(
            f"{manifest.source_name} weight={manifest.weight:g} shards={len(manifest.shards)} "
            f"tokens={token_count:,}"
        )
    return "; ".join(parts)


class WeightedShardSampler:
    """Random-window batch sampler over a weighted mixture of shard sources."""

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

    @staticmethod
    def _normalize_lengths(lengths: list[int]) -> np.ndarray:
        """Normalize positive lengths into probabilities."""

        values = np.asarray(lengths, dtype=np.float64)
        if values.sum() <= 0:
            raise ValueError("Shard lengths must sum to a positive value.")
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

    def next_batch(self, device: str, *, context_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random training batch from the weighted source mixture."""

        source_record = self._sample_source()
        manifest = source_record["manifest"]
        shard_probs = source_record["shard_probs"]
        assert isinstance(manifest, SourceManifest)
        assert isinstance(shard_probs, np.ndarray)

        shard_index = int(self._rng.choice(len(manifest.shards), p=shard_probs))
        shard = manifest.shards[shard_index]
        tokens = self._get_tokens(shard.path)
        if len(tokens) <= context_len:
            raise ValueError(
                f"Shard {shard.path} is too short for context length {context_len}."
            )

        max_start = len(tokens) - context_len - 1
        starts = self._rng.integers(0, max_start + 1, size=self.batch_size, endpoint=False)
        inputs = np.stack([tokens[start : start + context_len] for start in starts]).astype(np.int64)
        targets = np.stack(
            [tokens[start + 1 : start + context_len + 1] for start in starts]
        ).astype(np.int64)
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
        "--print-only",
        action="store_true",
        help="Print the manifest summary after building/loading cached manifests.",
    )
    return parser.parse_args()


def main() -> None:
    """Build manifests for the configured data sources and print a summary."""

    args = parse_args()
    config = load_experiment_config(args.config)
    manifests = load_source_manifests(config.data)
    print(summarize_manifests(manifests))


if __name__ == "__main__":
    main()
