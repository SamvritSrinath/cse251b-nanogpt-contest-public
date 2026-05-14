# Data Sourcing

The default training corpus in this repository is **FineWeb-Edu**, sourced from the public Hugging Face dataset [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). FineWeb-Edu is a filtered educational subset of FineWeb curated by the Hugging Face FineData team; the filtering and large-scale web curation details are documented in the FineWeb and FineWeb-Edu papers and dataset cards.

## How data is prepared

1. The canonical one-shot ingest entrypoint is `./scripts/ingest_data.sh`.
2. In HF mode it downloads a dataset snapshot locally with the HF CLI, recursively discovers `*.parquet`, tokenizes them with the Rust `corpus-prep` binary, and by default deletes the temporary parquet cache after success.
3. In local mode it skips the download step and tokenizes an existing recursive parquet tree.
4. `./scripts/prep_data.sh` is the config-driven wrapper that resolves `data.sources` and delegates actual local builds to `ingest_data.sh`.
5. `scripts/prep_fineweb.py` remains as a legacy Python fallback for FineWeb-Edu only when explicitly requested.

These shards are then read lazily with `numpy.memmap`, so training never loads the full corpus into RAM.

## Why `.bin` instead of `.npy`

This repository standardizes on raw `uint16` `.bin` shards rather than `.npy`.

- `.bin` is the simplest representation for `numpy.memmap` and for non-Python tools such as the Rust `corpus-prep` binary.
- `.npy` is self-describing and slightly safer for ad hoc inspection, but it adds a NumPy-specific header and does not buy much for this training path.
- For this codebase, `.bin` is the better fit because the loader only needs flat GPT-2 token IDs and already knows the dtype.

## GCP workflow

For fresh GCP VMs, use the quick-start helpers in [`docs/gcp_quickstart.md`](./gcp_quickstart.md):

1. mount the attached disk with `./scripts/gcp_mount_disk.sh`,
2. bootstrap a `data` or `train` environment with `./scripts/setup_env.sh`,
3. materialize FineWeb-Edu or another parquet-backed HF dataset with `./scripts/ingest_data.sh` when preparing a data VM.

## How the training loader uses the data

- Training sources are declared explicitly in each experiment config under `data.sources`.
- Each source has a `name`, `path`, `glob`, `weight`, and optional `notes`.
- Source manifests are cached under `data/manifests/` so study runs do not repeatedly rescan directories.
- Weighted source mixing samples:
  1. a source by configured mixture weight,
  2. a shard within that source proportional to shard token count,
  3. a random overlapping training window from that shard.

## Validation safety

`val.bin` is evaluation-only. The manifest builder and training shard discovery both exclude `val.bin`, even if it is colocated with training data.

## Adding secondary sources

To mix in a second corpus for domain-generalization ablations:

1. tokenize it into GPT-2 `.bin` shards,
2. place the shards in a dedicated directory such as `data/openwebtext/`,
3. add a second entry to `data.sources` with a smaller weight, for example `0.1`,
4. rebuild or reuse manifests via `python -m src.data --config <experiment-config>`.
