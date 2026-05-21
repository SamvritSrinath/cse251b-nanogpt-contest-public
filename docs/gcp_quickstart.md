# GCP Quickstart

This repository now includes three GCP-oriented helper scripts:

- `./scripts/gcp_mount_disk.sh`
- `./scripts/setup_env.sh`
- `./scripts/ingest_data.sh`

Convenience wrappers:

- `./scripts/corpus_prep.sh` for invoking the built Rust binary without remembering `target/release/...`
- `./scripts/gcp_prep_fineweb.sh` as a FineWeb-Edu preset over `ingest_data.sh`
- `./scripts/gcp_setup_env.sh` as a compatibility alias for `setup_env.sh`

## Data-prep VM

Mount an attached persistent disk:

```bash
./scripts/gcp_mount_disk.sh \
  --device /dev/disk/by-id/google-nanodata \
  --mount-point /mnt/disks/nano-data-parse \
  --persist
```

If the disk is brand new and empty, add `--format-if-needed` once.

Bootstrap the data-prep environment:

```bash
./scripts/setup_env.sh data
source .workspace-env.sh
source .venv/bin/activate
```

Prepare FineWeb-Edu directly onto the mounted disk:

```bash
./scripts/ingest_data.sh \
  --hf-dataset HuggingFaceFW/fineweb-edu \
  --include "sample/10BT/*.parquet" \
  --source-name fineweb-edu \
  --shard-prefix edufineweb \
  --data-root /mnt/disks/nano-data-parse
```

Optionally upload the final shards to GCS:

```bash
./scripts/ingest_data.sh \
  --hf-dataset HuggingFaceFW/fineweb-edu \
  --include "sample/10BT/*.parquet" \
  --source-name fineweb-edu \
  --shard-prefix edufineweb \
  --data-root /mnt/disks/nano-data-parse \
  --gcs-uri gs://your-bucket/
```

## Training VM

Bootstrap the training environment (no Rust toolchain required):

```bash
./scripts/setup_env.sh train \
  --torch-index-url https://download.pytorch.org/whl/cu121
source .workspace-env.sh
source .venv/bin/activate
```

If you only need the Python workspace and will pull tokenized shards from GCS (no local
tokenization), skip rustup and `corpus-prep` entirely:

```bash
./scripts/setup_env.sh data --skip-rust
# or, for training-only VMs:
./scripts/setup_env.sh train --skip-rust \
  --torch-index-url https://download.pytorch.org/whl/cu121
```

`--skip-rust` implies `--skip-corpus-build` and omits `build-essential` / rustup on the
data role.

If you already have a mounted disk with `fineweb-edu/` on it, symlink it into the repo:

```bash
mkdir -p data
ln -sfn /mnt/disks/nano-data-parse/fineweb-edu data/fineweb-edu
```

Then run a smoke test:

```bash
./scripts/run_experiment.sh configs/small.yaml --notes "gcp smoke test"
```

## Notes

- `./scripts/ingest_data.sh` is the canonical repo entry point for dataset download + parquet discovery + tokenization.
- `./scripts/ingest_data.sh --dry-run ...` is the clean way to inspect a future HF dataset before pulling it for real.
- The setup script writes `.workspace-env.sh` so mutable caches and tool homes can live on the mounted disk.
- The helper scripts assume Debian/Ubuntu-style images with `apt-get`.
