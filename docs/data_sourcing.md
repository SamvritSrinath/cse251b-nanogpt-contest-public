# Data Sourcing

The default training corpus in this repository is **FineWeb-Edu**, sourced from the public Hugging Face dataset [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). FineWeb-Edu is a filtered educational subset of FineWeb curated by the Hugging Face FineData team; the filtering and large-scale web curation details are documented in the FineWeb and FineWeb-Edu papers and dataset cards.

## How data is prepared

1. The canonical one-shot ingest entrypoint is `./scripts/ingest_data.sh`.
2. In HF mode it downloads a dataset snapshot locally with the HF CLI, recursively discovers `*.parquet`, tokenizes them with the Rust `corpus-prep` binary, and by default deletes the temporary parquet cache after success.
3. In local mode it skips the download step and tokenizes an existing recursive parquet tree.
4. `./scripts/prep_data.sh` is the compatibility config wrapper. When a config declares `data.sources[].prepare`, it routes to `python scripts/prepare_sources.py --config <yaml>`; otherwise it keeps the older source-resolution behavior.
5. `scripts/prep_fineweb.py` remains as a legacy Python fallback for FineWeb-Edu only when explicitly requested.

These shards are then read lazily with `numpy.memmap`, so training never loads the full corpus into RAM.

## Config-native source prep

New data sprint configs can put a `prepare` block directly on each `data.sources[]` entry:

```yaml
data:
  sources:
    - name: dclm_clean
      path: data/dclm-clean
      glob: "**/*_train_*.bin"
      weight: 0.35
      sample_policy: random_window
      prepare:
        kind: hf_parquet
        dataset: mlfoundations/dclm-baseline-1.0-parquet
        include: "**/*.parquet"
        revision: main
        text_column: text
        shard_prefix: dclm_clean
        filter_profile: dclm_clean
        split_mode: train-only
        emit_doc_index: false
```

Use:

```bash
python scripts/prepare_sources.py --config configs/data_sprints/probe_b_clean_web.yaml --dry-run
python scripts/prepare_sources.py --config configs/data_sprints/probe_b_clean_web.yaml
```

If `GCS_DATA_ROOT=gs://bucket/prefix` is set, `prepare_sources.py` first checks for existing token shards at the matching repo-relative source path and pulls them unless `--force-local` is used. Local preparation downloads or reads parquet, applies the optional filter profile, and then invokes `scripts/corpus_prep.sh` with `--split-mode train-only`.

Available filter profiles are:

- `fineweb_edu_hi`: keeps high-scoring English FineWeb-Edu rows.
- `dclm_edu_hi`: keeps high-scoring English rows from `HuggingFaceTB/dclm-edu`.
- `dclm_clean`: keeps English DCLM parquet rows with basic length and optional quality thresholds.
- `pg19_books`: strips Project Gutenberg boilerplate and normalizes book rows.
- `s2orc_sections`: emits section-level rows and requires section metadata unless using nested `body_text`.
- `openwebmath`: keeps non-trivial math documents.
- `finerweb_line_quality`: keeps rows whose mean `line_quality` clears the threshold, or rewrites rows with low-quality lines removed when requested.
- `conservative_v1`: opt-in heuristic cleanup that drops only extremely short documents, very low alphanumeric text, obvious cookie/JavaScript boilerplate, repeated-line spam, and repeated-ngram spam.

Filtered parquet rows are normalized to `text`, `doc_id`, `title`, `source`, and optional `section`.
`filter_parquet.py` prints kept rows and the top drop reasons. No heuristic filter is applied unless a source explicitly sets `prepare.filter_profile`.

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

### 400 GB boot disk notes

For a boot-disk-only VM, set:

```bash
export HF_HOME="$HOME/hf-cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
```

Then process one source at a time with `python scripts/prepare_sources.py --config <yaml> --source <name>`. The prep script skips sources that already have tokenized train shards, tries `GCS_DATA_ROOT` first when set, deletes generated raw/filter/temp parquet caches after each source unless `keep_raw_parquet: true`, and hard-fails if a source output contains `*_val_*.bin`.

Monitor space between sources:

```bash
df -h /
du -sh data checkpoints submission "$HOME/hf-cache" 2>/dev/null || true
rm -rf "$HOME/hf-cache"
```

Keep final training data as raw `uint16` `*_train_*.bin` shards plus optional `<shard>.bin.docs.json` sidecars. Do not train on `val.bin` or `*_val_*.bin`; configs intended for boot-disk prep use `**/*_train_*.bin` globs.

Do not attempt full DCLM, Dolma, RedPajama, or S2ORC snapshot materialization on a 400 GB boot disk. Use filtered profiles, Common-Pile streaming JSON presets, per-source `max_docs`, `max_bytes`, `max_temp_gb`, and upload/reuse tokenized shards from GCS.

## How the training loader uses the data

- Training sources are declared explicitly in each experiment config under `data.sources`.
- Each source has a `name`, `path`, `glob`, `weight`, optional `notes`, optional `prepare`, and `sample_policy`.
- `sample_policy` defaults to `random_window`. Other options are `document_window`, `section_window`, and `packed_short_docs`.
- Source manifests are cached under `data/manifests/` so study runs do not repeatedly rescan directories. The cache records shard names, sizes, mtimes, and doc-index sidecar paths, and is rebuilt automatically when that fingerprint changes or when `python -m src.data --rebuild` is used.
- Weighted source mixing with `random_window` samples:
  1. a source by configured mixture weight,
  2. a shard within that source proportional to shard token count,
  3. a random overlapping training window from that shard.
- Document-aware policies use `<shard>.bin.docs.json` sidecars emitted by `corpus-prep --emit-doc-index`. `document_window` and `section_window` never cross a selected range. `packed_short_docs` can combine short document ranges to fill one example.
- `packed_short_docs` requires every matched `*.bin` shard to have a `<shard>.bin.docs.json` sidecar. Missing sidecars are a hard error; prepare those sources with `emit_doc_index: true` and use train-only globs.

Minimal `packed_short_docs` source shape:

```yaml
data:
  sources:
    - name: short_docs
      path: data/short-docs
      glob: "**/*_train_*.bin"
      weight: 0.05
      sample_policy: packed_short_docs
      prepare:
        kind: hf_parquet
        dataset: owner/dataset
        text_column: text
        split_mode: train-only
        emit_doc_index: true
        doc_id_column: doc_id
        title_column: title
```

## Validation safety

`val.bin` is evaluation-only. The manifest builder and training shard discovery exclude `val.bin` and `*_val_*.bin`, even if a config glob is broad. Configs under `configs/` now use `**/*_train_*.bin` by default.

## Adding secondary sources

To mix in a second corpus for domain-generalization ablations:

1. tokenize it into GPT-2 `.bin` shards,
2. place the shards in a dedicated directory such as `data/openwebtext/`,
3. add a second entry to `data.sources` with a smaller weight, for example `0.1`,
4. rebuild or reuse manifests via `python -m src.data --config <experiment-config>`.

The first sprint probe configs live in `configs/data_sprints/`:

- `probe_a_fineweb_edu.yaml`: FineWeb-Edu baseline.
- `probe_b_clean_web.yaml`: high-quality FineWeb-Edu plus DCLM parquet.
- `probe_c_documents_math.yaml`: PG19, S2ORC sections, and OpenWebMath with doc-aware sampling.
- `continuation_d_current_best.yaml`: continuation from `submission/20260520-095531/best/checkpoint.pt`.

FinerWeb-10BT remains a candidate later source, but it is Arrow-format rather than the first sprint's parquet path.
