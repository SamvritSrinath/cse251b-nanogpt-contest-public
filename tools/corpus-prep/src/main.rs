//! Generic **text → GPT-2 uint16 `.bin` shards** for numpy memmap loaders (same layout as
//! ``scripts/prep_fineweb.py`` / Karpathy ``fineweb.py``): one ``<|endoftext|>`` before each
//! document, Karpathy-style first shard labeled ``val`` then ``train`` by default.
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
use arrow_array::{
    Array, Float64Array, Int32Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
    UInt32Array, UInt64Array,
};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::Api;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
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

    /// Shard split naming mode: Karpathy first-val shard or train-only shards.
    #[arg(long, value_enum, default_value = "karpathy")]
    split_mode: SplitMode,

    /// Emit one ``<shard>.docs.json`` sidecar with document/section token ranges.
    #[arg(long)]
    emit_doc_index: bool,

    /// Logical source name written into doc-index sidecars.
    #[arg(long, default_value = "corpus")]
    source_name: String,

    /// Optional parquet column for stable document ids in doc-index sidecars.
    #[arg(long)]
    doc_id_column: Option<String>,

    /// Optional parquet column for document titles in doc-index sidecars.
    #[arg(long)]
    title_column: Option<String>,

    /// Optional parquet column for section labels in doc-index sidecars.
    #[arg(long)]
    section_column: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, ValueEnum)]
enum SplitMode {
    Karpathy,
    TrainOnly,
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
        for ent in
            std::fs::read_dir(root).with_context(|| format!("read_dir {}", root.display()))?
        {
            let ent = ent?;
            let path = ent.path();
            let ft = ent
                .file_type()
                .with_context(|| format!("file_type {}", path.display()))?;
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

enum MetadataColumn<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    UInt32(&'a UInt32Array),
    UInt64(&'a UInt64Array),
    Float64(&'a Float64Array),
}

impl<'a> MetadataColumn<'a> {
    fn optional_from_batch(batch: &'a RecordBatch, name: &Option<String>) -> Result<Option<Self>> {
        let Some(name) = name else {
            return Ok(None);
        };
        let idx = column_index(batch, name)?;
        let col = batch.column(idx);
        if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
            return Ok(Some(Self::Utf8(a)));
        }
        if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
            return Ok(Some(Self::LargeUtf8(a)));
        }
        if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
            return Ok(Some(Self::Int32(a)));
        }
        if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
            return Ok(Some(Self::Int64(a)));
        }
        if let Some(a) = col.as_any().downcast_ref::<UInt32Array>() {
            return Ok(Some(Self::UInt32(a)));
        }
        if let Some(a) = col.as_any().downcast_ref::<UInt64Array>() {
            return Ok(Some(Self::UInt64(a)));
        }
        if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
            return Ok(Some(Self::Float64(a)));
        }
        anyhow::bail!(
            "metadata column '{name}' must be string or scalar numeric, got {:?}",
            col.data_type()
        );
    }

    fn value(&self, i: usize) -> Option<String> {
        match self {
            Self::Utf8(a) => (!a.is_null(i)).then(|| a.value(i).to_string()),
            Self::LargeUtf8(a) => (!a.is_null(i)).then(|| a.value(i).to_string()),
            Self::Int32(a) => (!a.is_null(i)).then(|| a.value(i).to_string()),
            Self::Int64(a) => (!a.is_null(i)).then(|| a.value(i).to_string()),
            Self::UInt32(a) => (!a.is_null(i)).then(|| a.value(i).to_string()),
            Self::UInt64(a) => (!a.is_null(i)).then(|| a.value(i).to_string()),
            Self::Float64(a) => (!a.is_null(i)).then(|| a.value(i).to_string()),
        }
    }
}

struct MetadataColumns<'a> {
    doc_id: Option<MetadataColumn<'a>>,
    title: Option<MetadataColumn<'a>>,
    section: Option<MetadataColumn<'a>>,
}

impl<'a> MetadataColumns<'a> {
    fn from_batch(batch: &'a RecordBatch, args: &Args) -> Result<Self> {
        Ok(Self {
            doc_id: MetadataColumn::optional_from_batch(batch, &args.doc_id_column)?,
            title: MetadataColumn::optional_from_batch(batch, &args.title_column)?,
            section: MetadataColumn::optional_from_batch(batch, &args.section_column)?,
        })
    }

    fn meta_for_row(&self, path: &Path, row_index: usize, row_in_batch: usize) -> DocMeta {
        let doc_id = self
            .doc_id
            .as_ref()
            .and_then(|col| col.value(row_in_batch))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| format!("{}:{row_index}", path.display()));
        DocMeta {
            doc_id,
            title: self.title.as_ref().and_then(|col| col.value(row_in_batch)),
            section: self
                .section
                .as_ref()
                .and_then(|col| col.value(row_in_batch)),
        }
    }
}

#[derive(Clone, Debug)]
struct DocMeta {
    doc_id: String,
    title: Option<String>,
    section: Option<String>,
}

#[derive(Debug)]
struct TokenizedDoc {
    tokens: Vec<u16>,
    meta: DocMeta,
}

#[derive(Debug, Serialize)]
struct DocRange {
    doc_id: String,
    title: Option<String>,
    section: Option<String>,
    start: usize,
    end: usize,
    continued_from_previous: bool,
    continues_to_next: bool,
}

#[derive(Debug, Serialize)]
struct ShardDocIndex {
    version: u8,
    shard: String,
    source: String,
    token_count: usize,
    ranges: Vec<DocRange>,
}

struct ShardWriter {
    out_dir: PathBuf,
    shard_size: usize,
    shard_index: usize,
    buf: Vec<u16>,
    token_count: usize,
    shard_prefix: String,
    split_mode: SplitMode,
    emit_doc_index: bool,
    source_name: String,
    ranges: Vec<DocRange>,
}

impl ShardWriter {
    fn new(
        out_dir: PathBuf,
        shard_size: usize,
        shard_prefix: String,
        split_mode: SplitMode,
        emit_doc_index: bool,
        source_name: String,
    ) -> Result<Self> {
        std::fs::create_dir_all(&out_dir)?;
        Ok(Self {
            out_dir,
            shard_size,
            shard_index: 0,
            buf: vec![0u16; shard_size],
            token_count: 0,
            shard_prefix,
            split_mode,
            emit_doc_index,
            source_name,
            ranges: Vec::new(),
        })
    }

    fn split_label(&self) -> &str {
        match self.split_mode {
            SplitMode::Karpathy if self.shard_index == 0 => "val",
            _ => "train",
        }
    }

    fn shard_stem(&self) -> PathBuf {
        self.out_dir.join(format!(
            "{}_{}_{:06}",
            self.shard_prefix,
            self.split_label(),
            self.shard_index
        ))
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
        if self.emit_doc_index {
            let sidecar_path = PathBuf::from(format!("{}.docs.json", path.display()));
            let shard_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string();
            let index = ShardDocIndex {
                version: 1,
                shard: shard_name,
                source: self.source_name.clone(),
                token_count: slice.len(),
                ranges: std::mem::take(&mut self.ranges),
            };
            let mut sidecar = File::create(&sidecar_path)
                .with_context(|| format!("create {}", sidecar_path.display()))?;
            serde_json::to_writer_pretty(&mut sidecar, &index)
                .with_context(|| format!("write {}", sidecar_path.display()))?;
            sidecar.write_all(b"\n")?;
        } else {
            self.ranges.clear();
        }
        Ok(())
    }

    fn push_document(&mut self, tokens: &[u16], meta: &DocMeta) -> Result<()> {
        let mut offset = 0usize;
        while offset < tokens.len() {
            let room = self.shard_size - self.token_count;
            let take = (tokens.len() - offset).min(room);
            let start = self.token_count;
            self.buf[self.token_count..self.token_count + take]
                .copy_from_slice(&tokens[offset..offset + take]);
            self.token_count += take;
            if self.emit_doc_index {
                self.ranges.push(DocRange {
                    doc_id: meta.doc_id.clone(),
                    title: meta.title.clone(),
                    section: meta.section.clone(),
                    start,
                    end: self.token_count,
                    continued_from_previous: offset > 0,
                    continues_to_next: offset + take < tokens.len(),
                });
            }
            offset += take;

            if self.token_count == self.shard_size {
                let stem = self.shard_stem();
                self.flush_shard(stem, true)?;
                self.shard_index += 1;
                self.token_count = 0;
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        if self.token_count > 0 {
            let stem = self.shard_stem();
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
    metadata: &MetadataColumns<'_>,
    path: &Path,
    row_base: usize,
    batch_threads: usize,
) -> Result<Vec<TokenizedDoc>> {
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
        out.push(TokenizedDoc {
            tokens: r.with_context(|| format!("tokenize row {}", row_base + i))?,
            meta: metadata.meta_for_row(path, row_base + i, i),
        });
    }
    Ok(out)
}

fn run_parquet_file(
    path: &Path,
    bpe: &CoreBPE,
    text_column: &str,
    args: &Args,
    writer: &mut ShardWriter,
    batch_threads: usize,
) -> Result<()> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("parquet reader {}", path.display()))?;
    let mut reader = builder.build()?;

    let mut row_base = 0usize;
    while let Some(batch) = reader.next() {
        let batch = batch?;
        let texts = TextColumn::from_batch(&batch, text_column)?;
        let metadata = MetadataColumns::from_batch(&batch, args)?;
        for doc in process_batch(bpe, &texts, &metadata, path, row_base, batch_threads)? {
            writer.push_document(&doc.tokens, &doc.meta)?;
        }
        row_base += texts.len();
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
        args.split_mode.clone(),
        args.emit_doc_index,
        args.source_name.clone(),
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
            run_parquet_file(
                &p,
                &bpe,
                &args.text_column,
                &args,
                &mut writer,
                args.batch_threads,
            )?;
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
            run_parquet_file(
                &local,
                &bpe,
                &args.text_column,
                &args,
                &mut writer,
                args.batch_threads,
            )?;
        }
    }

    writer.finish()?;
    eprintln!("corpus-prep: done → {}", args.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_out(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("corpus-prep-{name}-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn meta(doc_id: &str) -> DocMeta {
        DocMeta {
            doc_id: doc_id.to_string(),
            title: None,
            section: None,
        }
    }

    fn read_sidecar(path: &Path) -> Value {
        let text = fs::read_to_string(path).unwrap();
        serde_json::from_str(&text).unwrap()
    }

    #[test]
    fn train_only_names_all_shards_train() {
        let out = temp_out("train-only");
        let mut writer = ShardWriter::new(
            out.clone(),
            4,
            "tiny".to_string(),
            SplitMode::TrainOnly,
            false,
            "unit".to_string(),
        )
        .unwrap();
        writer.push_document(&[1, 2, 3, 4, 5], &meta("a")).unwrap();
        writer.finish().unwrap();

        assert!(out.join("tiny_train_000000.bin").is_file());
        assert!(out.join("tiny_train_000001.bin").is_file());
        assert!(!out.join("tiny_val_000000.bin").exists());
        fs::remove_dir_all(out).unwrap();
    }

    #[test]
    fn doc_index_records_ranges_and_metadata() {
        let out = temp_out("ranges");
        let mut writer = ShardWriter::new(
            out.clone(),
            16,
            "tiny".to_string(),
            SplitMode::TrainOnly,
            true,
            "unit_source".to_string(),
        )
        .unwrap();
        let first = DocMeta {
            doc_id: "doc-a".to_string(),
            title: Some("A title".to_string()),
            section: None,
        };
        let second = DocMeta {
            doc_id: "doc-b".to_string(),
            title: None,
            section: Some("Methods".to_string()),
        };
        writer.push_document(&[1, 2, 3], &first).unwrap();
        writer.push_document(&[4, 5, 6, 7], &second).unwrap();
        writer.finish().unwrap();

        let sidecar = read_sidecar(&out.join("tiny_train_000000.bin.docs.json"));
        assert_eq!(sidecar["version"], 1);
        assert_eq!(sidecar["source"], "unit_source");
        assert_eq!(sidecar["token_count"], 7);
        assert_eq!(sidecar["ranges"][0]["doc_id"], "doc-a");
        assert_eq!(sidecar["ranges"][0]["title"], "A title");
        assert_eq!(sidecar["ranges"][0]["start"], 0);
        assert_eq!(sidecar["ranges"][0]["end"], 3);
        assert_eq!(sidecar["ranges"][1]["doc_id"], "doc-b");
        assert_eq!(sidecar["ranges"][1]["section"], "Methods");
        assert_eq!(sidecar["ranges"][1]["start"], 3);
        assert_eq!(sidecar["ranges"][1]["end"], 7);
        fs::remove_dir_all(out).unwrap();
    }

    #[test]
    fn long_document_ranges_split_at_shard_boundaries() {
        let out = temp_out("long-doc");
        let mut writer = ShardWriter::new(
            out.clone(),
            4,
            "tiny".to_string(),
            SplitMode::TrainOnly,
            true,
            "unit".to_string(),
        )
        .unwrap();
        writer
            .push_document(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &meta("long"))
            .unwrap();
        writer.finish().unwrap();

        let first = read_sidecar(&out.join("tiny_train_000000.bin.docs.json"));
        let second = read_sidecar(&out.join("tiny_train_000001.bin.docs.json"));
        let third = read_sidecar(&out.join("tiny_train_000002.bin.docs.json"));
        assert_eq!(first["ranges"][0]["start"], 0);
        assert_eq!(first["ranges"][0]["end"], 4);
        assert_eq!(first["ranges"][0]["continued_from_previous"], false);
        assert_eq!(first["ranges"][0]["continues_to_next"], true);
        assert_eq!(second["ranges"][0]["continued_from_previous"], true);
        assert_eq!(second["ranges"][0]["continues_to_next"], true);
        assert_eq!(third["ranges"][0]["start"], 0);
        assert_eq!(third["ranges"][0]["end"], 1);
        assert_eq!(third["ranges"][0]["continued_from_previous"], true);
        assert_eq!(third["ranges"][0]["continues_to_next"], false);
        fs::remove_dir_all(out).unwrap();
    }
}
