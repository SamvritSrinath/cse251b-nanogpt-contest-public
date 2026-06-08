# Phase 1 Ablations + Projected-Width Architecture Plan

## Current Baseline

- Best known local result: `20260515-104853`, `full_v2_mixture_template`, `29.5006` validation PPL.
- Gold reference target: `19.02` PPL.
- Submission constraints from `README.md`: `<=100M` total parameters, GPT-2 token IDs in, logits over exactly `50257` tokens out, `1024` token eval context, and no training on `val.bin`.
- Local checkout only has FineWeb-Edu data, but these configs assume OpenWebText, Wikimedia, arXiv, and optional StackExchange shards are present on the remote training machine.

## Track A: Continue The 29.5 Checkpoint

Run this on L4, not T4.

1. `configs/continue_29p5_generalize.yaml`
   - Resumes from `checkpoints/20260515-104853/best.pt`.
   - Uses `82/10/7/1` FineWeb-Edu/OpenWebText/Wikimedia/arXiv.
   - Lowers LR to `adamw_lr=0.0001`, `muon_lr=0.006`.
   - Runs to `max_steps=22000`.

2. `configs/continue_29p5_polish.yaml`
   - Resume path is intentionally `checkpoints/TODO_A1_BEST/best.pt`; replace it with the best A1 checkpoint before running.
   - Uses `95/5` FineWeb-Edu/OpenWebText.
   - Lowers LR again to `adamw_lr=0.00005`, `muon_lr=0.003`.
   - Runs to `max_steps=26000`, roughly 4k more steps if A1 ended at 22k.

Keep A1 as the hidden-test-safer candidate unless A2 clearly wins official `evaluate.py`.

## Track B: No-Code Phase 1 Ablations

Base config: `configs/small_v2_mix_ablation.yaml`.

Data mixture hand configs:

| ID | Config | Mixture | Purpose |
| --- | --- | --- | --- |
| D0 | `configs/ablations/data_mix_d0.yaml` | `75/15/9/1` | current winner control |
| D1 | `configs/ablations/data_mix_d1.yaml` | `85/10/4/1` | safer generalization |
| D2 | `configs/ablations/data_mix_d2.yaml` | `95/5/0/0` | public-val polish |
| D3 | `configs/ablations/data_mix_d3.yaml` | `65/20/14/1` | broader hidden-test hedge |
| D4 | `configs/ablations/data_mix_d4.yaml` | `75/15/5/0/5` | StackExchange hedge |
| D5 | `configs/ablations/data_mix_d5.yaml` | `70/10/15/5` | academic/factual heavy |

Study configs:

- `configs/studies/small_schedule_grid.yaml`
- `configs/studies/small_muon_grid.yaml`
- `configs/studies/small_reg_grid.yaml`

Use:

```bash
./scripts/run_study.sh configs/studies/small_schedule_grid.yaml
./scripts/run_study.sh configs/studies/small_muon_grid.yaml
./scripts/run_study.sh configs/studies/small_reg_grid.yaml
```

For hand configs, use the normal experiment runner:

```bash
./scripts/run_experiment.sh configs/ablations/data_mix_d0.yaml --notes "phase1 data mixture D0"
```

## Track C: Projected-Width 8-Layer Architecture

New architecture: `projected_modern_decoder`.

Default config: `configs/wide_projected_8l_192_768.yaml`.

Architecture:

- Token embedding: `50257 x 192`.
- `input_proj`: `192 -> 768`.
- Transformer: 8 modern decoder blocks, `d_model=768`, `n_heads=12`, head dim `64`.
- MLP: SwiGLU, default `ffn_multiplier=2.15`.
- `output_proj`: `768 -> 192`.
- LM head tied to token embedding, so output logits are `(batch, seq, 50257)` while staying around `59M` params.

Study configs:

- `configs/studies/wide_projected_ffn_grid.yaml`: tests `ffn_multiplier` `[2.0, 2.1, 2.15]`.
- `configs/studies/wide_projected_lr_grid.yaml`: tests `muon_lr`, warmup, and LR floor.

Run order:

1. Instantiate and forward-test the model locally.
2. Run a 100-200 step smoke on the training machine.
3. Run `wide_projected_ffn_grid`.
4. Run `wide_projected_lr_grid` only after the FFN grid is stable.
5. Promote best projected config to 3000 steps if it beats the 8x384 baseline by at least `0.05` validation loss at 1200 steps.

## Promotion Policy

- 1200-step run: promote if validation loss improves by `>=0.05`.
- 1200-step run with `0.03-0.05` gain: rerun at 3000 steps.
- 3000-step run: promote if `>=0.03` gain survives.
- Medium/full run: must survive unchanged `evaluate.py`.
- Never replace the protected 29.5 candidate unless official evaluation improves or hidden-test plausibility is clearly better.

## Completion Checklist

- [x] Add projected architecture schema and implementation.
- [x] Add Phase 1 and projected architecture configs.
- [x] Add persistent status tracker at `.codex/ablation_status.md`.
- [ ] Run remote data manifest preflight.
- [ ] Run projected architecture smoke train.
- [ ] Run D0-D5 mixture ablations.
- [ ] Run schedule, Muon, and regularization grids.
- [ ] Run projected FFN grid.
- [ ] Run projected LR grid.
- [ ] Promote best candidate to 3000 steps.
- [ ] Run unchanged `evaluate.py` on promoted submission bundle.
