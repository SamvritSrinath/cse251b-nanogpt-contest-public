# Data Sourcing

The default training corpus in this repository is **FineWeb-Edu**, sourced from the public Hugging Face dataset [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). FineWeb-Edu is a filtered educational subset of FineWeb curated by the Hugging Face FineData team; the filtering and large-scale web curation details are documented in the FineWeb and FineWeb-Edu papers and dataset cards.

## How data is prepared

1. Run `./scripts/prep_data.sh`.
2. The script clones Karpathy's `build-nanogpt` repository if needed.
3. `fineweb.py` downloads and tokenizes FineWeb-Edu using the GPT-2 tokenizer.
4. The resulting `.bin` shards are copied into `data/fineweb-edu/`.

These shards are then read lazily with `numpy.memmap`, so training never loads the full corpus into RAM.

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
