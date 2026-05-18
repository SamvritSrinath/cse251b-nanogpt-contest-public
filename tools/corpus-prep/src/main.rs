//! Generic **text → GPT-2 uint16 `.bin` shards** for numpy memmap loaders (same layout as
//! ``scripts/prep_fineweb.py`` / Karpathy ``fineweb.py``): one ``<|endoftext|>`` before each
//! document, Karpathy-style first shard labeled ``val`` then ``train``.
//!
//! **Sources** (pick one):
//! - **Hugging Face Hub**: parquet files under a dataset config folder (tree API + ``hf-hub`` download).
//! - **Local disk**: recursive ``*.parquet`` under a directory, sorted by path (stable order).
//!
//! **Tokenizer**: ``tiktoken-rs`` ``r50k_base`` (Python ``tiktoken`` ``gpt2`` / 50 257-way logits).

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use arrow_array::{Array, LargeStringArray, RecordBatch, StringArray};
use clap::Parser;
use hf_hub::api::sync::Api;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde::Deserialize;
use tiktoken_rs::{r50k_base, CoreBPE, Rank};

/// ``<|endoftext|>`` rank for ``r50k_base`` / GPT-2.
const GPT2_EOT: Rank = 50256;

#[derive(Parser, Debug)]
#[command(name = "corpus-prep")]
struct Args {
    /// Output directory for ``{shard_prefix}_{val|train}_{nnnnnn}.bin``.
    #[arg(long, default_value = "data/out")]
    out: PathBuf,

    /// Parquet column (Utf8 or LargeUtf8) with one document per row.
    #[arg(long, default_value = "text")]
    text_column: String,

    /// Filename stem before ``_{split}_{index}.bin`` (e.g. ``edufineweb`` for FineWeb-Edu).
    #[arg(long, default_value = "corpus")]
    shard_prefix: String,

    /// Tokens per shard (default 100M).
    #[arg(long, default_value_t = 100_000_000usize)]
    shard_size: usize,

    /// Rayon parallelism within each Arrow batch (0 = Rayon default thread count).
    #[arg(long, default_value_t = 0usize)]
    batch_threads: usize,

    /// When set, read recursive ``*.parquet`` here and **skip** Hugging Face.
    #[arg(long)]
    local_parquet_dir: Option<PathBuf>,

    /// Hugging Face **dataset** id (ignored if ``--local-parquet-dir`` is set).
    #[arg(long, default_value = "HuggingFaceFW/fineweb-edu")]
    hf_dataset: String,

    /// Hub **config** folder name (tree ``path=`` scope under the dataset repo).
    #[arg(long, default_value = "sample-10BT")]
    hf_subset: String,

    /// Hub revision (branch, tag, or commit).
    #[arg(long, default_value = "main")]
    hf_revision: String,

    /// Process only the first N parquet files (sorted order); smoke tests.
    #[arg(long)]
    max_parquet_files: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct TreeEntry {
    path: String,
    #[serde(rename = "type")]
    entry_type: String,
}

fn hf_parquet_rel_paths(dataset_id: &str, subset: &str, revision: &str) -> Result<Vec<String>> {
    let enc = urlencoding::encode(subset);
    let url = format!(
        "https://huggingface.co/api/datasets/{dataset_id}/tree/{revision}?recursive=true&path={enc}"
    );
    let entries: Vec<TreeEntry> = ureq::get(&url)
        .call()
        .with_context(|| format!("HF tree request failed: {url}"))?
        .into_json()
        .with_context(|| "HF tree JSON parse failed")?;

    let root = format!("{subset}/");
    let mut paths: Vec<String> = entries
        .into_iter()
        .filter(|e| e.entry_type == "file" && e.path.ends_with(".parquet"))
        .map(|e| format!("{root}{}", e.path))
        .collect();
    paths.sort();
    if paths.is_empty() {
        anyhow::bail!(
            "no parquet under Hub subset '{subset}' for {dataset_id}@{revision}; check --hf-subset / --hf-dataset"
        );
    }
    Ok(paths)
}

fn collect_local_parquets(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    fn walk(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for ent in std::fs::read_dir(root).with_context(|| format!("read_dir {}", root.display()))? {
            let ent = ent?;
            let path = ent.path();
            let ft = ent.file_type().with_context(|| format!("file_type {}", path.display()))?;
            if ft.is_dir() {
                walk(&path, out)?;
            } else if path.extension().and_then(|e| e.to_str()) == Some("parquet") {
                out.push(path);
            }
        }
        Ok(())
    }
    walk(dir, &mut out)?;
    out.sort();
    if out.is_empty() {
        anyhow::bail!("no *.parquet under {}", dir.display());
    }
    Ok(out)
}

fn tokenize_one(bpe: &CoreBPE, text: &str) -> Result<Vec<u16>> {
    let mut out = Vec::with_capacity(text.len() / 4 + 2);
    out.push(GPT2_EOT as u16);
    for id in bpe.encode_ordinary(text) {
        let v = u16::try_from(id).map_err(|_| anyhow!("token id {id} does not fit uint16"))?;
        out.push(v);
    }
    Ok(out)
}

enum TextColumn<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
}

impl<'a> TextColumn<'a> {
    fn from_batch(batch: &'a RecordBatch, name: &str) -> Result<Self> {
        let idx = column_index(batch, name)?;
        let col = batch.column(idx);
        if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
            return Ok(Self::Utf8(a));
        }
        if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
            return Ok(Self::LargeUtf8(a));
        }
        anyhow::bail!(
            "column '{name}' must be Utf8 or LargeUtf8 (Arrow string types), got {:?}",
            col.data_type()
        );
    }

    fn len(&self) -> usize {
        match self {
            Self::Utf8(a) => a.len(),
            Self::LargeUtf8(a) => a.len(),
        }
    }

    fn value(&self, i: usize) -> &str {
        match self {
            Self::Utf8(a) => a.value(i),
            Self::LargeUtf8(a) => a.value(i),
        }
    }
}

struct ShardWriter {
    out_dir: PathBuf,
    shard_size: usize,
    shard_index: usize,
    buf: Vec<u16>,
    token_count: usize,
    shard_prefix: String,
}

impl ShardWriter {
    fn new(out_dir: PathBuf, shard_size: usize, shard_prefix: String) -> Result<Self> {
        std::fs::create_dir_all(&out_dir)?;
        Ok(Self {
            out_dir,
            shard_size,
            shard_index: 0,
            buf: vec![0u16; shard_size],
            token_count: 0,
            shard_prefix,
        })
    }

    fn flush_shard(&mut self, path_stem: PathBuf, full: bool) -> Result<()> {
        let path = path_stem.with_extension("bin");
        let slice = if full {
            &self.buf[..]
        } else {
            &self.buf[..self.token_count]
        };
        let mut f = File::create(&path).with_context(|| format!("create {}", path.display()))?;
        f.write_all(bytemuck::cast_slice(slice))
            .with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    fn push_tokens(&mut self, tokens: &[u16]) -> Result<()> {
        let mut offset = 0usize;
        while offset < tokens.len() {
            let room = self.shard_size - self.token_count;
            let take = (tokens.len() - offset).min(room);
            self.buf[self.token_count..self.token_count + take]
                .copy_from_slice(&tokens[offset..offset + take]);
            self.token_count += take;
            offset += take;

            if self.token_count == self.shard_size {
                let split: &str = if self.shard_index == 0 { "val" } else { "train" };
                let stem = self
                    .out_dir
                    .join(format!("{}_{split}_{:06}", self.shard_prefix, self.shard_index));
                self.flush_shard(stem, true)?;
                self.shard_index += 1;
                self.token_count = 0;
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        if self.token_count > 0 {
            let split: &str = if self.shard_index == 0 { "val" } else { "train" };
            let stem = self
                .out_dir
                .join(format!("{}_{split}_{:06}", self.shard_prefix, self.shard_index));
            self.flush_shard(stem, false)?;
        }
        Ok(())
    }
}

fn column_index(batch: &RecordBatch, name: &str) -> Result<usize> {
    batch
        .schema()
        .fields()
        .iter()
        .position(|f| f.name() == name)
        .ok_or_else(|| anyhow!("parquet batch: no column '{name}'"))
}

fn process_batch(
    bpe: &CoreBPE,
    texts: &TextColumn<'_>,
    batch_threads: usize,
) -> Result<Vec<Vec<u16>>> {
    let n = texts.len();
    let pool_threads = if batch_threads == 0 {
        rayon::current_num_threads()
    } else {
        batch_threads
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_threads.min(n.max(1)))
        .build()?;

    let mut indexed: Vec<(usize, Result<Vec<u16>>)> = pool.install(|| {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let s = texts.value(i);
                (i, tokenize_one(bpe, s))
            })
            .collect()
    });
    indexed.sort_by_key(|(i, _)| *i);
    let mut out = Vec::with_capacity(n);
    for (i, r) in indexed {
        out.push(r.with_context(|| format!("tokenize row {i}"))?);
    }
    Ok(out)
}

fn run_parquet_file(
    path: &Path,
    bpe: &CoreBPE,
    text_column: &str,
    writer: &mut ShardWriter,
    batch_threads: usize,
) -> Result<()> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("parquet reader {}", path.display()))?;
    let mut reader = builder.build()?;

    while let Some(batch) = reader.next() {
        let batch = batch?;
        let texts = TextColumn::from_batch(&batch, text_column)?;
        for doc_tokens in process_batch(bpe, &texts, batch_threads)? {
            writer.push_tokens(&doc_tokens)?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bpe = r50k_base().context("r50k_base (gpt2)")?;
    let mut writer = ShardWriter::new(
        args.out.clone(),
        args.shard_size,
        args.shard_prefix.clone(),
    )?;

    if let Some(ref dir) = args.local_parquet_dir {
        let mut paths = collect_local_parquets(dir)?;
        if let Some(n) = args.max_parquet_files {
            paths.truncate(n);
            eprintln!("corpus-prep: local: using first {n} parquet file(s)");
        }
        eprintln!(
            "corpus-prep: local {} parquet file(s) from {}",
            paths.len(),
            dir.display()
        );
        for p in paths {
            eprintln!("  → {}", p.display());
            run_parquet_file(&p, &bpe, &args.text_column, &mut writer, args.batch_threads)?;
        }
    } else {
        let mut rels = hf_parquet_rel_paths(&args.hf_dataset, &args.hf_subset, &args.hf_revision)?;
        if let Some(n) = args.max_parquet_files {
            rels.truncate(n);
            eprintln!("corpus-prep: Hub: using first {n} parquet file(s)");
        }
        eprintln!(
            "corpus-prep: Hub {} / {} @{} — {} parquet file(s)",
            args.hf_dataset,
            args.hf_subset,
            args.hf_revision,
            rels.len()
        );
        let api = Api::new().context("hf-hub Api::new")?;
        let repo = api.dataset(args.hf_dataset.clone());
        for rel in rels {
            eprintln!("  ↓ {}", rel);
            let local = repo
                .download(&rel)
                .with_context(|| format!("download {rel}"))?;
            run_parquet_file(&local, &bpe, &args.text_column, &mut writer, args.batch_threads)?;
        }
    }

    writer.finish()?;
    eprintln!("corpus-prep: done → {}", args.out.display());
    Ok(())
}
