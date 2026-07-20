# Exp4 Final v3 Freeze

Final mitigation version for the current paper draft.

- Prompt profile: `domain_aware_v3`
- Invalid split: `outputs/linguistic_blindness/exp3_invalid_instruction_clean_600/benchmark.jsonl`
- Normal split: `outputs/linguistic_blindness/exp4_normal_instruction_clean_600/benchmark.jsonl`
- OpenVLA actions: reused from Exp3 action-inhibition run; OpenVLA was not rerun for Exp4.
- Invalid Qwen schema v3: `outputs/linguistic_blindness/qwen25vl7b_schema_exp4_invalid_clean_1800_v3/`
- Normal Qwen schema v3: `outputs/linguistic_blindness/qwen25vl7b_schema_exp4_normal_clean_600_v3/`
- Final Exp4 v3 output: `outputs/linguistic_blindness/exp4_checker_gate_mitigation_clean_600_v3_unified/`

Main results:

| Method | Invalid Blind ↓ | Safe Deferral ↑ | Normal Pass ↑ | False Block ↓ |
|---|---:|---:|---:|---:|
| OpenVLA | 0.927 | 0.000 | 1.000 | 0.000 |
| All-stop baseline | 0.000 | 1.000 | 0.000 | 1.000 |
| Checker + Gate v3 | 0.028 | 0.972 | 0.940 | 0.060 |

Per invalid perturbation:

| Perturbation | N | OpenVLA Blind ↓ | Gated Blind ↓ | Reduction ↑ | Safe Deferral ↑ |
|---|---:|---:|---:|---:|---:|
| blank_instruction | 600 | 0.910 | 0.083 | 0.827 | 0.915 |
| impossible_instruction | 600 | 0.940 | 0.000 | 0.940 | 1.000 |
| negation | 600 | 0.932 | 0.000 | 0.932 | 1.000 |
| overall | 1800 | 0.927 | 0.028 | 0.899 | 0.972 |

Notes:

- Exp3 remains the diagnosis experiment using the original Exp3 Qwen schema outputs.
- Exp4A-v3 is the mitigation rerun on the same invalid split using the final domain-aware schema prompt.
- Normal preservation uses real Qwen normal schemas under the same `domain_aware_v3` prompt.
- Remaining caveats: blank residual default execution and negation blocked-action incompleteness.
