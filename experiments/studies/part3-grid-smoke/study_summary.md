# Study: part3-grid-smoke

- Mode: grid
- Best trial: trial-0000
- Best val PPL: 51088.4316

| Trial | Arch | Optimizer | Context | Data Mix | Params | Val PPL | Status |
|-------|------|-----------|---------|----------|--------|---------|--------|
| trial-0000 | modern_decoder | adamw | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 6,827,264 | 51088.4316 | completed |
| trial-0001 | gpt2_decoder | adamw | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 6,960,768 | 51650.4852 | completed |
| trial-0002 | modern_decoder | muon_hybrid | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 6,827,264 | 51099.6388 | completed |
| trial-0003 | gpt2_decoder | muon_hybrid | ramp:256->1024@1+1 | fineweb_edu:0.9,general_web:0.1 | 6,960,768 | 51594.6571 | completed |
