| Run | Config | Config Hash | Arch | Optimizer | Batch Tokens | Context | Data Mix | Steps | Params | Val PPL | Notes |
|-----|--------|-------------|------|-----------|-------------|---------|----------|-------|--------|---------|-------|
| 20260501-001526 | part3-smoke | 1e2ec7c24d | modern_decoder | adamw | 1024 | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 2 | 6,827,264 | 51088.4316 | part3 smoke |
| 20260501-001555 | part3-grid-smoke-trial-0000 | 5d4987cd59 | modern_decoder | adamw | 1024 | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 2 | 6,827,264 | 51088.4316 | study=part3-grid-smoke trial=trial-0000 |
| 20260501-001556 | part3-optuna-smoke-trial-0000 | be098aaaea | modern_decoder | muon_hybrid | 1024 | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 2 | 5,046,336 | 51698.1887 | study=part3-optuna-smoke trial=trial-0000 |
| 20260501-001559 | part3-grid-smoke-trial-0001 | a7bb1e2260 | gpt2_decoder | adamw | 1024 | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 2 | 6,960,768 | 51650.4852 | study=part3-grid-smoke trial=trial-0001 |
| 20260501-001605 | part3-grid-smoke-trial-0002 | 848fb649cb | modern_decoder | muon_hybrid | 1024 | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 2 | 6,827,264 | 51099.6388 | study=part3-grid-smoke trial=trial-0002 |
| 20260501-001614 | part3-grid-smoke-trial-0003 | c1d78b85fc | gpt2_decoder | muon_hybrid | 1024 | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 2 | 6,960,768 | 51594.6571 | study=part3-grid-smoke trial=trial-0003 |
| 20260501-001704 | part3-optuna-smoke-b-trial-0000 | 45081f499f | modern_decoder | muon_hybrid | 1024 | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 2 | 5,046,336 | 51698.1887 | study=part3-optuna-smoke-b trial=trial-0000 |
| 20260501-001708 | part3-optuna-smoke-b-trial-0001 | 1127da8b3b | modern_decoder | adamw | 1024 | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 2 | 6,827,264 | 51088.4316 | study=part3-optuna-smoke-b trial=trial-0001 |
| 20260501-180319 | baseline | 6c2d8012e5 | modern_decoder | adamw | 2048 | fixed@1024 | fineweb_edu:1 | 50 | 16,017,920 | 6824.4863 | post-prep sanity run |
| 20260501-180420 | control_from_optuna_trial_0001 | c0600f1079 | modern_decoder | adamw | 16384 | ramp:256->1024@500+500 | fineweb_edu:1 | 3000 | 6,827,264 | 459.1894 | non open text sanity run |
