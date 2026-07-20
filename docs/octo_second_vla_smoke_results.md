# Octo Second-VLA Architecture Smoke Results

This run evaluates a fully different VLA architecture, Octo, on the same LIBERO target-swap diagnostic protocol. It is a native action-output sensitivity probe, not LIBERO task success and not a calibrated LIBERO controller evaluation.

## Setup

- VLA architecture: Octo
- Checkpoint: `hf://rail-berkeley/octo-small-1.5`
- Examples: 100 LIBERO target-swap examples
- Schema probe: Qwen2.5-VL-7B predictions from Exp2
- Output: `outputs/linguistic_blindness/exp2_octo_small_target_swap_100/`
- Control threshold: paraphrase-control P95

## Results

| Metric | Value |
|---|---:|
| N | 100 |
| VLM schema sensitivity | 1.000 |
| Octo action sensitivity | 0.420 |
| Semantic-action gap | 0.580 |
| Low sensitivity rate | 0.580 |
| Mean normalized action delta | 1.981 |
| Mean paraphrase-control delta | 0.875 |
| Action dim | 7 |

Target-pair breakdown:

| Target Pair | N | Octo Action Sens. | Gap Case Rate |
|---|---:|---:|---:|
| black bowl -> cookie box | 33 | 0.545 | 0.455 |
| black bowl -> ramekin | 48 | 0.479 | 0.521 |
| middle drawer -> top drawer | 19 | 0.053 | 0.947 |

## Interpretation

Octo is more action-sensitive than OpenVLA under this protocol, but still substantially below schema sensitivity. This gives architecture-level evidence for a semantic-action gap:

- Schema sensitivity remains 1.000.
- Octo action sensitivity is 0.420.
- The gap remains 0.580 on the 100-example smoke set.

Careful wording:

> As a fully different VLA architecture, Octo shows higher target-swap action sensitivity than OpenVLA, but it still leaves a large gap relative to schema-level target sensitivity. This suggests that semantic-action bottlenecks are not exclusive to OpenVLA, although their severity differs across action policies.

Do not overclaim:

- This is a 100-example smoke run, not the final 600-example full run.
- Octo is evaluated zero-shot on LIBERO-style observations, so the result should be treated as a native action-sensitivity probe rather than a LIBERO success benchmark.

## Full 600 Run

Attempted command:

```bash
MAX_EXAMPLES=600 \
OUT_DIR=outputs/linguistic_blindness/exp2_octo_small_target_swap_600 \
sbatch slurm/lb_octo_action_sensitivity.sbatch
```

Submission failed because Slurm controller was unreachable. Re-run when Slurm is stable.
