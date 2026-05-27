from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data import (
    ShardInfo,
    SourceManifest,
    WeightedShardSampler,
    load_source_manifests,
)
from src.utils import DataConfig, DataSourceConfig


class DataManifestTests(unittest.TestCase):
    def write_tokens(self, path: Path, count: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.arange(count, dtype=np.uint16).tofile(path)

    def data_config(self, root: Path, source_dir: Path) -> DataConfig:
        return DataConfig(
            sources=[
                DataSourceConfig(
                    name="tiny",
                    path=str(source_dir),
                    glob="**/*.bin",
                    weight=1.0,
                )
            ],
            val_data_path=str(root / "val.bin"),
            manifest_dir=str(root / "manifests"),
        )

    def test_manifest_excludes_val_shards_and_records_file_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            self.write_tokens(root / "val.bin", 8)
            self.write_tokens(source_dir / "tiny_train_000000.bin", 16)
            self.write_tokens(source_dir / "tiny_val_000000.bin", 16)
            self.write_tokens(source_dir / "val.bin", 16)

            manifests = load_source_manifests(self.data_config(root, source_dir), rebuild=True)

            self.assertEqual(len(manifests), 1)
            self.assertEqual(len(manifests[0].shards), 1)
            shard = manifests[0].shards[0]
            self.assertTrue(shard.path.endswith("tiny_train_000000.bin"))
            self.assertEqual(shard.num_tokens, 16)
            self.assertEqual(shard.size_bytes, 32)
            self.assertGreater(shard.mtime_ns, 0)

    def test_stale_manifest_rebuilds_when_shards_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            self.write_tokens(root / "val.bin", 8)
            self.write_tokens(source_dir / "tiny_train_000000.bin", 16)
            config = self.data_config(root, source_dir)

            first = load_source_manifests(config, rebuild=True)[0]
            self.assertEqual(len(first.shards), 1)

            self.write_tokens(source_dir / "tiny_train_000001.bin", 12)
            second = load_source_manifests(config)[0]
            self.assertEqual(len(second.shards), 2)
            self.assertEqual(sum(shard.num_tokens for shard in second.shards), 28)


class SamplerPolicyTests(unittest.TestCase):
    def write_sidecar(self, shard: Path, ranges: list[dict[str, object]]) -> Path:
        sidecar = Path(f"{shard}.docs.json")
        sidecar.write_text(
            json.dumps(
                {
                    "version": 1,
                    "shard": shard.name,
                    "source": "tiny",
                    "token_count": 50,
                    "ranges": ranges,
                }
            ),
            encoding="utf-8",
        )
        return sidecar

    def make_manifest(self, shard: Path, policy: str, sidecar: Path | None) -> SourceManifest:
        stat = shard.stat()
        return SourceManifest(
            source_name="tiny",
            source_path=str(shard.parent),
            glob="*.bin",
            weight=1.0,
            notes="",
            sample_policy=policy,
            shards=[
                ShardInfo(
                    path=str(shard),
                    num_tokens=50,
                    size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    doc_index_path=None if sidecar is None else str(sidecar),
                )
            ],
        )

    def assert_window_inside(self, start: int, length: int, ranges: list[dict[str, object]]) -> None:
        end = start + length
        self.assertTrue(
            any(int(item["start"]) <= start and end <= int(item["end"]) for item in ranges),
            f"{start}:{end} crosses configured ranges",
        )

    def test_all_sampler_policies_on_tiny_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = root / "tiny_train_000000.bin"
            np.arange(50, dtype=np.uint16).tofile(shard)
            ranges = [
                {"doc_id": "a", "start": 0, "end": 10},
                {"doc_id": "b", "section": "intro", "start": 10, "end": 30},
                {"doc_id": "c", "start": 30, "end": 35},
                {"doc_id": "d", "section": "methods", "start": 35, "end": 50},
            ]
            sidecar = self.write_sidecar(shard, ranges)

            random_sampler = WeightedShardSampler(
                [self.make_manifest(shard, "random_window", None)],
                batch_size=4,
                seed=1,
            )
            x, y = random_sampler.next_batch("cpu", context_len=6)
            self.assertEqual(tuple(x.shape), (4, 6))
            self.assertEqual(tuple(y.shape), (4, 6))

            doc_sampler = WeightedShardSampler(
                [self.make_manifest(shard, "document_window", sidecar)],
                batch_size=8,
                seed=2,
            )
            x, y = doc_sampler.next_batch("cpu", context_len=5)
            for row_x, row_y in zip(x.numpy(), y.numpy()):
                self.assertTrue(np.array_equal(row_x[1:], row_y[:-1]))
                self.assert_window_inside(int(row_x[0]), 6, ranges)

            section_sampler = WeightedShardSampler(
                [self.make_manifest(shard, "section_window", sidecar)],
                batch_size=8,
                seed=3,
            )
            x, y = section_sampler.next_batch("cpu", context_len=5)
            section_ranges = [item for item in ranges if item.get("section")]
            for row_x, row_y in zip(x.numpy(), y.numpy()):
                self.assertTrue(np.array_equal(row_x[1:], row_y[:-1]))
                self.assert_window_inside(int(row_x[0]), 6, section_ranges)

            packed_sampler = WeightedShardSampler(
                [self.make_manifest(shard, "packed_short_docs", sidecar)],
                batch_size=4,
                seed=4,
            )
            x, y = packed_sampler.next_batch("cpu", context_len=12)
            self.assertEqual(tuple(x.shape), (4, 12))
            self.assertEqual(tuple(y.shape), (4, 12))
            self.assertTrue(np.all((0 <= x.numpy()) & (x.numpy() < 50)))


if __name__ == "__main__":
    unittest.main()
