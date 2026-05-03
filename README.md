# CSE 251B Spring 2026 — NanoGPT Competition

Train the best language model you can. Lowest perplexity on our hidden test set wins.

## Overview

This competition challenges you to train a GPT-style language model from scratch (or near-scratch) and achieve the lowest possible perplexity on a held-out evaluation set. You have freedom to choose your architecture, optimizer, training data, and training procedure — the only hard constraint is on model size.

This competition is inspired by the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt) community and [OpenAI's Parameter Golf](https://github.com/openai/parameter-golf), adapted for a course setting.

## Rules

### The One Hard Rule

**Your submitted model must have ≤ 100M total parameters.**

We verify this at submission time. Models exceeding 100M parameters will not be evaluated.

### Everything Else Is Open

- **Architecture:** Any architecture is allowed — standard Transformer, state-space model, RNN, hybrid, whatever you want — as long as the total parameter count is ≤ 100M and the model satisfies the interface described below.
- **Training data:** You may use any publicly available training data. We recommend [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`) as a starting point, which is the standard dataset used in the NanoGPT speedrun community. You are free to supplement or replace it with other data sources.
- **Training procedure:** Any optimizer, learning rate schedule, regularization, data augmentation, curriculum, or other training technique is fair game.
- **Pretrained components:** You may use pretrained tokenizers, pretrained embeddings, or distillation from larger models, as long as your final submitted model is ≤ 100M parameters and you document what you used in your report.
- **Compute:** The competition is designed so that competitive results are achievable with approximately $20 of GPU compute (e.g., ~65 hours on a rented RTX 4090). You are welcome to use more or less. Please document your approximate compute usage in your report.

### What You Cannot Do

- Submit a model with > 100M parameters.
- Tamper with the evaluation script or submit fabricated scores.
- Train on the public validation split (`val.bin`). This data is for evaluation only.

## Evaluation

### Metric

**Perplexity (PPL)** on a held-out test set. Lower is better.

Perplexity is defined as `exp(average cross-entropy loss)` where the average is taken per-token over the evaluation data. The evaluation data is tokenized with the [GPT-2 BPE tokenizer](https://github.com/openai/tiktoken) (encoding name: `gpt2`, vocab size: 50257).

### Public Validation Set

We provide `val.bin` in this repository — a tokenized evaluation split of approximately 5 million tokens. Use this to track your progress during development. Compute your val PPL using the provided `evaluate.py` script:

```bash
# Evaluate from a local directory (during development)
python evaluate.py --model_dir /path/to/your/submission/ --data val.bin

# Evaluate from your HuggingFace submission (to verify before deadline)
python evaluate.py --hf_repo your-username/cse251b-group-XX --data val.bin
```

The `--hf_repo` mode downloads your model from HuggingFace and evaluates it in exactly the same way the TAs will. **Use this to verify your submission works before the deadline.**

### Wall-Clock Time Limit

Each submission is given a maximum of **5 minutes (300 seconds)** of wall-clock time to run inference on the evaluation split. This is measured from when your model is loaded until the final perplexity is computed — download time does not count. A standard 100M parameter model completes in roughly 50 seconds, so this limit is generous. Models that exceed the limit are disqualified from that evaluation run.

### Hidden Test Set

Your final ranking is determined by perplexity on a **hidden test set** that is never released to students. The test set is drawn from the same distribution as the validation set. At the submission deadline, TAs will download your model and evaluate it against the hidden test set. The test set is a mix of domains designed to reward models that generalize well — not just models that memorize one particular data source.

### Leaderboards

There are three leaderboards. All use the same single submission link — see [Submission](#submission) below.

| Leaderboard | Split | Source | Purpose |
|---|---|---|---|
| **Unofficial (Val)** | Public val | Self-reported PPL | High-frequency, for fun — not used for grades |
| **Official (Val)** | Public val | TA-run eval, weekly | Tracks progress; not used for grades |
| **Official (Test)** | Hidden test | TA-run eval, after May 31 | Determines contest ranking and grade |

- **View unofficial leaderboard:** [Google Sheet](https://docs.google.com/spreadsheets/d/1mDsizxbzSE6RirQ-WyFqZfj5uvNRPpmZsg7sggsylfU/edit?usp=sharing) (self-reported, updated whenever you submit)
- **View official leaderboard:** [GitHub Pages site](https://matt-seb-ho.github.io/cse251b-nanogpt-contest-public/) (TA-run, updated weekly)

The unofficial leaderboard is for motivation and frequent self-tracking. Only the hidden test set evaluation after the submission deadline determines your grade.

## Submission

### What to Submit

At the competition deadline, each group submits a **HuggingFace repository** containing:

1. **`checkpoint.pt`** — Your trained model weights (a PyTorch state dict).
2. **`model.py`** — Your model class definition, including a `load_model()` function (see interface below).
3. **Any config files** your `model.py` needs to instantiate the model (e.g., `config.json`, `config.py`, etc.).

### How to Submit

There is a **single submission form** for everything — registering your team, joining the leaderboards, and submitting your final model. Fill it out once and update it as needed.

**[→ Submit here](https://forms.gle/p99o5vr26DLdY1X47)**

The form asks for: team name, member info, HuggingFace model repo ID, and an optional self-reported val PPL for the unofficial leaderboard.

1. Create a free account on [huggingface.co](https://huggingface.co) if you don't have one.
2. Create a **public** HuggingFace model repository. (Public is strongly preferred — private repos require adding the TA team as collaborators, which creates extra overhead. See note below.)
3. Upload your files:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload your-username/cse251b-group-XX ./checkpoint.pt ./model.py ./config.json
   ```
4. Fill out the submission form with your repo ID.

> **Private repo?** If you must keep your repo private, add the TA team as collaborators before the evaluation deadline. Our HuggingFace usernames are `msho` and `alexnrojas5`. See the [HuggingFace docs](https://huggingface.co/docs/hub/organizations-managing) for instructions.

### Required Model Interface

Your `model.py` must contain a function with this exact signature:

```python
def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load your trained model from a checkpoint.

    Args:
        checkpoint_path: Path to your checkpoint.pt file
        device: Device string ("cuda" or "cpu")

    Returns:
        A PyTorch nn.Module in eval mode where:
            model(input_ids) -> logits
            - input_ids: LongTensor of shape (batch_size, sequence_length)
            - logits: FloatTensor of shape (batch_size, sequence_length, 50257)
    """
```

See `model_example.py` in this repo for a complete working example.

**Important:** Your model must use vocab size **50257** (the GPT-2 BPE vocabulary). The eval script tokenizes the evaluation data with this tokenizer and expects logits over exactly 50257 tokens.

## Getting Started

This repository now includes a config-driven training stack in `src/`, helper
scripts in `scripts/`, experiment configs in `configs/`, and exported submission
bundles under `submission/`.

### 1. Environment Setup

In a Studio, do not create a repo-local virtual environment with
`python3 -m venv .venv`. Studios allow one default conda environment, so use the
existing Python environment instead.

Check the active Python:

```bash
python --version
which python
```

Install dependencies into the active environment:

```bash
python -m pip install -r requirements.txt
```

If `pip` fails on an old local build path such as `package @ file:///home/...`,
filter those local-only entries and install the cleaned requirements file:

```bash
grep -v ' @ file:' requirements.txt > /tmp/requirements.clean.txt
python -m pip install -r /tmp/requirements.clean.txt
```

The helper scripts resolve Python in this order:

1. `PYTHON_BIN`, if you set it.
2. `/home/zeus/miniconda3/envs/cloudspace/bin/python`, the default Studio conda Python.
3. `.venv/bin/python`, for non-Studio local clones.
4. `python` from `PATH`.

To force a specific interpreter:

```bash
PYTHON_BIN="$(which python)" ./scripts/run_experiment.sh configs/baseline.yaml
```

### 2. Project Layout

- `src/model.py`: decoder-only language model implementations and architecture registry.
- `src/train.py`: main training loop, validation, checkpointing, and submission export.
- `src/data.py`: weighted shard sampler and validation batching for GPT-2-tokenized `.bin` files.
- `src/optimizer.py`: AdamW and Muon/AdamW hybrid optimizers.
- `src/search.py`: grid and Optuna study runner.
- `src/utils.py`: config dataclasses, LR schedules, checkpoint helpers, and export utilities.
- `configs/`: ready-to-run experiment configs.
- `configs/studies/`: search study configs.
- `scripts/`: shell wrappers for data prep, training, studies, and HuggingFace upload.
- `evaluate.py`: contest-compatible local/HuggingFace perplexity evaluator.
- `model.py`: submission entrypoint shim that exposes `load_model()`.

### 3. Training Data

`val.bin` is provided for validation only. Do not train on it.

Training expects GPT-2-tokenized uint16 `.bin` shards under paths configured in
the YAML files, usually `data/fineweb-edu`. To download and tokenize FineWeb-Edu
through Karpathy's `build-nanogpt` flow, run:

```bash
./scripts/prep_data.sh
```

The script clones `build-nanogpt` if needed, runs `fineweb.py`, and copies `.bin`
shards into `data/fineweb-edu`.

You can inspect or build cached shard manifests without starting training:

```bash
python -m src.data --config configs/baseline.yaml --print-only
```

To add another corpus, place its `.bin` shards under `data/<source-name>/` and add
another entry under `data.sources` in a config file.

### 4. Train an Experiment

Run the smoke baseline:

```bash
./scripts/run_experiment.sh configs/baseline.yaml --notes "smoke test"
```

Equivalent direct command:

```bash
python src/train.py --config configs/baseline.yaml --notes "smoke test"
```

Useful configs:

- `configs/baseline.yaml`: short smoke run.
- `configs/small.yaml`: small model and short training run.
- `configs/medium.yaml`: medium-scale Muon hybrid run.
- `configs/full.yaml`: larger under-100M model.
- `configs/control.yaml`: longer control run from the smoke-study settings.

Training writes:

- `checkpoints/<run-id>/latest.pt`
- `checkpoints/<run-id>/best.pt`
- `checkpoints/<run-id>/ckpt_step*.pt`
- `submission/<run-id>/best/`
- `submission/<run-id>/final/`
- `experiments/results.md`

The `submission/<run-id>/best/` and `submission/<run-id>/final/` folders contain
`checkpoint.pt`, `config.json`, `model.py`, and `src/`, so they can be evaluated
or uploaded directly.

### 5. Evaluate Locally

Evaluate the best exported bundle:

```bash
python evaluate.py --model_dir submission/<run-id>/best --data val.bin
```

Use CPU only if CUDA is unavailable or you are debugging a small run:

```bash
python evaluate.py --model_dir submission/<run-id>/best --data val.bin --device cpu
```

The evaluator checks the submission interface, parameter count, output vocab size
`50257`, and reports perplexity.

### 6. Run Search Studies

Run a deterministic grid smoke study:

```bash
./scripts/run_study.sh configs/studies/grid_smoke.yaml
```

Run an Optuna smoke study:

```bash
./scripts/run_study.sh configs/studies/optuna_smoke.yaml
```

Study results are written under `experiments/studies/<study-name>/`, including
per-trial resolved configs, summaries, checkpoints, and submission bundles.

### 7. Upload a Submission Bundle

After choosing a `best` or `final` bundle, upload it to HuggingFace:

```bash
huggingface-cli login
./scripts/submit_hf.sh your-username/cse251b-group-XX submission/<run-id>/best
```

Verify the HuggingFace repo exactly as the TAs will load it:

```bash
python evaluate.py --hf_repo your-username/cse251b-group-XX --data val.bin
```

### 8. Current Implementation Notes

The model code supports two architecture recipes:

- `modern_decoder`: RoPE, RMSNorm, SwiGLU, no bias by default.
- `gpt2_decoder`: learned absolute positions, LayerNorm, GELU, bias by default.

The optimizer registry supports:

- `adamw`
- `muon_hybrid`, which applies Muon to hidden-layer matrices and AdamW to
  embeddings, output head, and vector parameters.

Training supports fixed or ramped context length schedules. Validation always
uses the configured full context length, typically 1024.

### 9. Troubleshooting

**`Venv creation is not allowed`**

Use the Studio's default conda environment. Do not run `python3 -m venv .venv` in
this Studio.

**`No training shards found`**

Run `./scripts/prep_data.sh`, or update `data.sources[*].path` in your config to
point at a directory containing GPT-2-tokenized `.bin` shards.

**`ModuleNotFoundError` or missing packages**

Install requirements into the active environment:

```bash
python -m pip install -r requirements.txt
```

If a local `@ file:` path appears, use the cleaned install command from
[Environment Setup](#1-environment-setup).

**CUDA is unavailable**

The training code falls back to CPU when `device: cuda` is requested but CUDA is
not available. CPU training and evaluation will be much slower.

## Timeline

| Week | Milestone |
|---|---|
| Week 2 | Competition released. Start forming groups. |
| Week 3 | Groups finalized. Run a baseline model. |
| Week 4–6 | Experiment with architectures, optimizers, data, etc. |
| Week 7 | **Milestone report due** — baseline results, ≥2 ablations, plan for remaining work. |
| Week 8–9 | Final push. Refine your best approach. |
| Week 9 | **Final submission deadline** — HuggingFace repo + leaderboard score. |
| Week 10 | **Presentations.** |
| Exam week | **Final report due (4 pages).** |

## Grading

The competition contributes **40%** of your course grade:

| Component | Weight | What we're looking for |
|---|---|---|
| Milestone report | 10% | Baseline results, ≥2 modifications with ablations, clear plan |
| Final report (4 pages) | 10% | Thorough description of approach, ablation studies, analysis of what worked and didn't, references to relevant literature |
| Presentation | 10% | Clear explanation, demo, insightful Q&A |
| Team ranking | 10% | Based on hidden test PPL. Tiered: top 20% → full marks, top 40% → 90%, top 60% → 80%, bottom 40% → 70% |

**Note:** No group receives zero for ranking if they submit a working model. The ranking curve is generous — what matters most is that you engage seriously with the problem and write a thoughtful report.

## Resources

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Recommended starting codebase
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — Reference for advanced techniques (RoPE, Muon, etc.)
- [NanoGPT Speedrun Leaderboard](https://app.primeintellect.ai/speedrun/nanogpt) — See what techniques top speedrunners use
- [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) — Similar competition from OpenAI
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — Recommended training data
- [tiktoken](https://github.com/openai/tiktoken) — The GPT-2 tokenizer library

## FAQ

**Q: Can I use multiple GPUs?**
A: Yes, but the competition is designed so a single GPU is sufficient. DDP across multiple GPUs is fine if you have access.

**Q: Can I fine-tune a pretrained model instead of training from scratch?**
A: Yes, as long as the final model is ≤ 100M parameters and you document your approach.

**Q: What if my model uses a different tokenizer internally?**
A: The eval script feeds your model GPT-2 token IDs and reads logits over the 50257-token GPT-2 vocabulary. Your model must accept this input format. If your internal architecture uses a different tokenization, you need to handle the mapping yourself.

**Q: What context length should my model support?**
A: The eval script uses a context window of 1024 tokens (matching GPT-2). Your model's forward pass must handle input sequences of length 1024.

**Q: Can I train on the validation set?**
A: No. The validation set is for evaluation only. We will check for suspiciously low val PPL coupled with high test PPL, which would indicate val-set overfitting.
