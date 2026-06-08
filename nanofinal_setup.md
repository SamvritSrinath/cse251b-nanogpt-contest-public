# Nano Final 400 GB Boot-Disk Setup

Run these on the GCP VM that will prepare data. This repo should not require a persistent disk.

## Environment

```bash
./scripts/setup_env.sh data
source .workspace-env.sh
source .venv/bin/activate
export HF_HOME="$HOME/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
df -h /
du -sh data checkpoints submission "$HF_HOME" 2>/dev/null || true
```

## Local Verification

```bash
cargo test --manifest-path tools/corpus-prep/Cargo.toml
python -m py_compile src/data.py src/utils.py scripts/prepare_sources.py scripts/filter_parquet.py
pytest tests/test_data.py tests/test_filter_parquet.py tests/test_prepare_sources.py
./scripts/corpus_prep.sh --help | grep -E "split-mode|emit-doc-index"
```

After shards exist, rebuild manifests:

```bash
python -m src.data --config configs/endgame/full_v19_dclm_commonpile_boot400.yaml --rebuild
```

## Prepare Sources One At A Time

Set this only if you already have tokenized shards in GCS:

```bash
export GCS_DATA_ROOT=gs://YOUR_BUCKET/YOUR_PREFIX
```

Run each source separately:

```bash
python scripts/prepare_sources.py --config configs/endgame/full_v19_dclm_commonpile_boot400.yaml --source fineweb_edu_hi
rm -rf "$HF_HOME"
df -h /
du -sh data checkpoints submission "$HOME/hf-cache" 2>/dev/null || true

python scripts/prepare_sources.py --config configs/endgame/full_v19_dclm_commonpile_boot400.yaml --source dclm_edu_hi
rm -rf "$HF_HOME"
df -h /
du -sh data checkpoints submission "$HOME/hf-cache" 2>/dev/null || true

python scripts/prepare_sources.py --config configs/endgame/full_v19_dclm_commonpile_boot400.yaml --source commonpile_wikimedia
rm -rf "$HF_HOME"
df -h /
du -sh data checkpoints submission "$HOME/hf-cache" 2>/dev/null || true

python scripts/prepare_sources.py --config configs/endgame/full_v19_dclm_commonpile_boot400.yaml --source commonpile_stackexchange
rm -rf "$HF_HOME"
df -h /
du -sh data checkpoints submission "$HOME/hf-cache" 2>/dev/null || true

python scripts/prepare_sources.py --config configs/endgame/full_v19_dclm_commonpile_boot400.yaml --source commonpile_gutenberg
rm -rf "$HF_HOME"
df -h /
du -sh data checkpoints submission "$HOME/hf-cache" 2>/dev/null || true

python scripts/prepare_sources.py --config configs/endgame/full_v19_dclm_commonpile_boot400.yaml --source commonpile_arxiv_abstracts
rm -rf "$HF_HOME"
df -h /
du -sh data checkpoints submission "$HOME/hf-cache" 2>/dev/null || true

python scripts/prepare_sources.py --config configs/endgame/full_v19_dclm_commonpile_boot400.yaml --source commonpile_youtube
rm -rf "$HF_HOME"
df -h /
du -sh data checkpoints submission "$HOME/hf-cache" 2>/dev/null || true

python scripts/prepare_sources.py --config configs/endgame/full_v19_dclm_commonpile_boot400.yaml --source openwebtext
rm -rf "$HF_HOME"
df -h /
du -sh data checkpoints submission "$HOME/hf-cache" 2>/dev/null || true

python scripts/prepare_sources.py --config configs/endgame/full_v18_commonpile_boot400.yaml --source pg19_books

```

Optional upload after each source:

```bash
gcloud storage cp --recursive data/fineweb-edu-hi "$GCS_DATA_ROOT/data/"
gcloud storage cp --recursive data/dclm-edu-hi "$GCS_DATA_ROOT/data/"
gcloud storage cp --recursive data/commonpile-wikimedia "$GCS_DATA_ROOT/data/"
gcloud storage cp --recursive data/commonpile-stackexchange "$GCS_DATA_ROOT/data/"
gcloud storage cp --recursive data/commonpile-gutenberg "$GCS_DATA_ROOT/data/"
gcloud storage cp --recursive data/commonpile-arxiv-abstracts "$GCS_DATA_ROOT/data/"
gcloud storage cp --recursive data/commonpile-youtube "$GCS_DATA_ROOT/data/"
gcloud storage cp --recursive data/openwebtext "$GCS_DATA_ROOT/data/"
```

## Final Checks

```bash
find data -name '*_val_*.bin' -print -quit | grep -q . && exit 1 || true
python -m src.data --config configs/endgame/full_v19_dclm_commonpile_boot400.yaml --rebuild
./scripts/run_experiment.sh configs/endgame/full_v19_dclm_commonpile_boot400.yaml --notes "v19 dclm commonpile boot400"
```

Avoid full DCLM, Dolma, RedPajama, and S2ORC snapshots on the boot disk. Use the configured filtered sources, streaming Common-Pile presets, and GCS reuse.
