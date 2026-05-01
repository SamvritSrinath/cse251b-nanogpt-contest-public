# Study: part3-optuna-smoke-b

- Mode: optuna
- Best trial: trial-0001
- Best val PPL: 51088.4316

| Trial | Arch | Optimizer | Context | Data Mix | Params | Val PPL | Status |
|-------|------|-----------|---------|----------|--------|---------|--------|
| trial-0000 | modern_decoder | muon_hybrid | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 5,046,336 | 51698.1887 | completed |
| trial-0001 | modern_decoder | adamw | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 6,827,264 | 51088.4316 | completed |
