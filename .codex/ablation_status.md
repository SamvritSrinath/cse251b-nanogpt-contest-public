# Ablation Status

last_updated: 2026-05-16

current_best_public_val_ppl: 29.5006
current_best_run: 20260515-104853
current_best_config: full_v2_mixture_template
gold_reference_ppl: 19.02
parameter_limit: 100000000

| Item | Config | Status | Run ID | Best Val PPL | Best Val Loss | Params | Official Eval PPL | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continue_29p5_generalize | `configs/continue_29p5_generalize.yaml` | pending |  |  |  |  |  | Run on L4 |
| continue_29p5_polish | `configs/continue_29p5_polish.yaml` | blocked |  |  |  |  |  | Replace `resume_from` with A1 best checkpoint |
| data_mix_d0 | `configs/ablations/data_mix_d0.yaml` | pending |  |  |  |  |  | 1200-step control |
| data_mix_d1 | `configs/ablations/data_mix_d1.yaml` | pending |  |  |  |  |  | 1200-step small ablation |
| data_mix_d2 | `configs/ablations/data_mix_d2.yaml` | pending |  |  |  |  |  | 1200-step public-val polish test |
| data_mix_d3 | `configs/ablations/data_mix_d3.yaml` | pending |  |  |  |  |  | 1200-step hidden-test hedge |
| data_mix_d4 | `configs/ablations/data_mix_d4.yaml` | blocked |  |  |  |  |  | Requires `data/stackexchange` shards |
| data_mix_d5 | `configs/ablations/data_mix_d5.yaml` | pending |  |  |  |  |  | 1200-step factual/academic mix |
| small_schedule_grid | `configs/studies/small_schedule_grid.yaml` | pending |  |  |  |  |  | Run after data mix control |
| small_muon_grid | `configs/studies/small_muon_grid.yaml` | pending |  |  |  |  |  | Run after schedule grid |
| small_reg_grid | `configs/studies/small_reg_grid.yaml` | pending |  |  |  |  |  | Run after Muon grid |
| projected_arch_smoke | `configs/wide_projected_8l_192_768.yaml` | pending |  |  |  |  |  | 100-200 training steps on GPU |
| wide_projected_ffn_grid | `configs/studies/wide_projected_ffn_grid.yaml` | pending |  |  |  |  |  | Promote if >=0.05 val-loss gain |
| wide_projected_lr_grid | `configs/studies/wide_projected_lr_grid.yaml` | pending |  |  |  |  |  | Run after FFN grid stability |

## Update Procedure

After each run, fill in `Run ID`, `Best Val PPL`, `Best Val Loss`, `Params`, and `Decision`.

Promotion rules:

- `>=0.05` validation loss gain at 1200 steps: promote.
- `0.03-0.05` validation loss gain at 1200 steps: rerun at 3000 steps.
- `<0.03` gain: ignore unless repeated.
- Official `evaluate.py` must be run before replacing the protected `29.5006` candidate.
