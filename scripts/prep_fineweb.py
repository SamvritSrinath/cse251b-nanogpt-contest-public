#!/usr/bin/env python3
"""Download FineWeb-Edu (sample-10BT) and write GPT-2 token shards as raw uint16 .bin files.

Matches Karpathy ``build-nanogpt/fineweb.py`` token layout (EOT between documents, same
shard boundaries and val/train naming), but:

- Writes **raw** ``.bin`` (``tofile``) so ``np.memmap(..., dtype=uint16)`` in training works.
- Optional **streaming** Hugging Face load to reduce time-to-first-token and peak cache use.
- Configurable ``imap`` **chunksize** (upstream uses 16; larger values cut IPC overhead).
- Worker **initializer** so each process loads tiktoken once (spawn-safe).

For the canonical dataset-agnostic pipeline (HF snapshot or local parquet → GPT-2 ``.bin``),
see ``./scripts/ingest_data.sh`` and ``./scripts/corpus_prep.sh``.

Note: ``datasets.map`` on a **streaming** iterable does not parallelize tokenization the way
Gemini-style snippets suggest; batched ``map`` returns one row per batch shape, not one mega-list.
This script keeps Karpathy's proven consumer loop and speeds up the edges that matter here.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

_enc: tiktoken.Encoding | None = None
_eot: int | None = None


def _init_worker() -> None:
    global _enc, _eot
    _enc = tiktoken.get_encoding("gpt2")
    assert _enc is not None
    _eot = int(_enc._special_tokens["<|endoftext|>"])


def _tokenize_doc(doc: dict) -> np.ndarray:
    assert _enc is not None and _eot is not None
    tokens = [_eot]
    tokens.extend(_enc.encode_ordinary(doc["text"]))
    arr = np.asarray(tokens, dtype=np.int32)
    if (arr < 0).any() or (arr >= 2**16).any():
        raise ValueError("token ids out of uint16 range")
    return arr.astype(np.uint16)


def _write_shard_bin(path_no_ext: Path, tokens: np.ndarray) -> None:
    """Write raw uint16 bytes (no numpy header)."""

    path = path_no_ext.with_suffix(".bin")
    tokens.tofile(path)


def run(
    *,
    out_dir: Path,
    remote_name: str,
    shard_size: int,
    num_proc: int,
    chunksize: int,
    streaming: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=remote_name,
        split="train",
        streaming=streaming,
    )

    nprocs = max(1, num_proc)
    with mp.Pool(nprocs, initializer=_init_worker) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar: tqdm | None = None

        for tokens in pool.imap(_tokenize_doc, fw, chunksize=chunksize):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size,
                        unit="tok",
                        desc=f"shard {shard_index}",
                        leave=False,
                    )
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                path_base = out_dir / f"edufineweb_{split}_{shard_index:06d}"
                remainder = shard_size - token_count
                if progress_bar is not None:
                    progress_bar.update(remainder)
                    progress_bar.close()
                    progress_bar = None
                all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                _write_shard_bin(path_base, all_tokens_np)
                shard_index += 1
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if progress_bar is not None:
            progress_bar.close()

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            path_base = out_dir / f"edufineweb_{split}_{shard_index:06d}"
            _write_shard_bin(path_base, all_tokens_np[:token_count])


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu GPT-2 uint16 shards.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/fineweb-edu"),
        help="Output directory for edufineweb_{split}_*.bin shards.",
    )
    parser.add_argument(
        "--remote-name",
        default="sample-10BT",
        help="Hugging Face config name (default: sample-10BT).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=int(1e8),
        help="Tokens per shard before rolling (default: 1e8).",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Worker processes for tokenization.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=128,
        help="imap chunksize; increase (e.g. 256) if docs are short to cut IPC overhead.",
    )
    parser.add_argument(
        "--streaming/--no-streaming",
        default=True,
        help="Stream dataset instead of full local cache first (default: stream).",
    )
    args = parser.parse_args()

    mp.freeze_support()
    run(
        out_dir=args.out.resolve(),
        remote_name=args.remote_name,
        shard_size=args.shard_size,
        num_proc=args.num_proc,
        chunksize=max(1, args.chunksize),
        streaming=args.streaming,
    )
    print(f"Wrote shards under {args.out.resolve()}")


if __name__ == "__main__":
    main()
