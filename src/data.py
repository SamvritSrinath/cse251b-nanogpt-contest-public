"""Data loading utilities for tokenized NanoGPT-style binary shards.

The training loader uses lazy ``numpy.memmap`` reads so the project can scale to
many shards without pulling the entire dataset into RAM. Validation iteration is
kept deliberately aligned with ``evaluate.py``: non-overlapping windows of
``context_len`` tokens, shifted by one token for the targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from src.utils import DataConfig


def discover_training_shards(data_config: DataConfig) -> list[Path]:
    """Find train shards while explicitly excluding the validation split.

    Args:
        data_config: Data settings describing the shard directory and glob.

    Returns:
        Sorted list of training shard paths.

    Raises:
        FileNotFoundError: If no training shards match the requested pattern.
    """

    train_dir = Path(data_config.train_data_dir).expanduser().resolve()
    val_path = Path(data_config.val_data_path).expanduser().resolve()
    shard_paths = [
        shard
        for shard in sorted(train_dir.glob(data_config.train_glob))
        if shard.resolve() != val_path and shard.name != "val.bin"
    ]
    if not shard_paths:
        raise FileNotFoundError(
            f"No training shards found in {train_dir} matching {data_config.train_glob}."
        )
    return shard_paths


class ShardedTokenLoader:
    """Random-window batch sampler over a set of token shards."""

    def __init__(
        self,
        shard_paths: list[Path],
        *,
        context_len: int,
        batch_size: int,
        seed: int,
    ) -> None:
        self.context_len = context_len
        self.batch_size = batch_size
        self._rng = np.random.default_rng(seed)
        self._epoch_rng = np.random.default_rng(seed + 1)
        self._ordered_paths = list(shard_paths)
        self._current_index = -1
        self._current_tokens: np.memmap | None = None
        self._batches_seen_in_shard = 0
        self._batches_per_shard = 1
        self._shuffle_for_new_epoch()
        self._advance_shard()

    def _shuffle_for_new_epoch(self) -> None:
        """Shuffle shard order for the next pass through the dataset."""

        permutation = self._epoch_rng.permutation(len(self._ordered_paths))
        self._ordered_paths = [self._ordered_paths[index] for index in permutation]

    def _advance_shard(self) -> None:
        """Open the next shard lazily via memmap."""

        self._current_index += 1
        if self._current_index >= len(self._ordered_paths):
            self._current_index = 0
            self._shuffle_for_new_epoch()
        shard_path = self._ordered_paths[self._current_index]
        tokens = np.memmap(shard_path, dtype=np.uint16, mode="r")
        if len(tokens) <= self.context_len:
            raise ValueError(
                f"Shard {shard_path} is too short for context length {self.context_len}."
            )
        self._current_tokens = tokens
        approx_batches = len(tokens) // max(1, self.batch_size * self.context_len)
        self._batches_per_shard = max(1, approx_batches)
        self._batches_seen_in_shard = 0

    def next_batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of overlapping training windows."""

        if self._current_tokens is None:
            self._advance_shard()
        if self._batches_seen_in_shard >= self._batches_per_shard:
            self._advance_shard()
        assert self._current_tokens is not None

        max_start = len(self._current_tokens) - self.context_len - 1
        starts = self._rng.integers(0, max_start + 1, size=self.batch_size, endpoint=False)
        inputs = np.stack(
            [self._current_tokens[start : start + self.context_len] for start in starts]
        ).astype(np.int64)
        targets = np.stack(
            [self._current_tokens[start + 1 : start + self.context_len + 1] for start in starts]
        ).astype(np.int64)
        self._batches_seen_in_shard += 1
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

    data = np.memmap(Path(data_path).expanduser().resolve(), dtype=np.uint16, mode="r")
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
