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

## 400 GB Boot-Disk-Only Workflow

Use this when you do not have a persistent disk. Keep Hugging Face caches under the home directory, process one source at a time, and delete raw/cache material after every successful source.

```bash
export HF_HOME="$HOME/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
df -h /
du -sh data checkpoints submission "$HF_HOME" 2>/dev/null || true
```

Prepare one configured source at a time:

```bash
python scripts/prepare_sources.py \
  --config configs/endgame/full_v19_dclm_commonpile_boot400.yaml \
  --source fineweb_edu_hi

rm -rf "$HF_HOME"
df -h /
du -sh data checkpoints submission "$HOME/hf-cache" 2>/dev/null || true
```

Repeat for the next source. If `GCS_DATA_ROOT=gs://bucket/prefix` is set, `prepare_sources.py` first tries to pull already-tokenized shards before doing local prep. Upload final tokenized source directories to GCS when available:

```bash
gcloud storage cp --recursive data/fineweb-edu-hi "$GCS_DATA_ROOT/data/"
```

For direct single-source ingest, require free space up front and emit train-only shards:

```bash
./scripts/ingest_data.sh \
  --hf-dataset HuggingFaceFW/fineweb-edu \
  --include "sample/10BT/*.parquet" \
  --source-name fineweb-edu-hi \
  --split-mode train-only \
  --min-free-gb 80
```

Avoid trying to fully materialize large DCLM, Dolma, RedPajama, S2ORC, or similar full snapshots on a 400 GB boot disk. Use filtered subsets, `max_parquet_files`, `max_docs`, `max_bytes`, `max_temp_gb`, and GCS reuse instead.
