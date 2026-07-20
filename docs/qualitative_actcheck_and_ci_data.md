# Qualitative ActCheck Figures and Bootstrap CI Data

This document collects the exact data needed for two remaining paper-polish items:

1. qualitative ActCheck / trajectory-style case panels;
2. bootstrap confidence intervals for the main supplemental tables.

The goal is to make the remaining figure/table work reproducible without rerunning VLA or VLM inference.

## 1. Qualitative ActCheck / Trajectory Figures

### What the figure should show

Each case panel should contain:

- scene image;
- original instruction;
- counterfactual instruction;
- schema target from Qwen/VISA;
- original target and counterfactual target;
- native policy action direction;
- intended target vector and wrong/original target vector;
- ActCheck label: `target-aligned`, `wrong-target`, or `ambiguous`;
- optional normalized action delta and action sensitivity.

Recommended caption language:

> In these cases, the VLM schema probe correctly grounds the counterfactual target, but the native action is either more aligned with the wrong/original target or ambiguous under the schema target. VISA-ActCheck makes this semantic-action inconsistency observable.

### Generated data packet

Directory:

```text
outputs/linguistic_blindness/qualitative_ci_packet/
```

Files:

```text
outputs/linguistic_blindness/qualitative_ci_packet/qualitative_actcheck_case_candidates.csv
outputs/linguistic_blindness/qualitative_ci_packet/recommended_qualitative_cases.csv
```

`qualitative_actcheck_case_candidates.csv` contains all candidate cases where:

- Qwen relation grounding is correct;
- the policy action is wrong-target or ambiguous according to ActCheck.

`recommended_qualitative_cases.csv` contains 6 recommended cases:

- 3 OpenVLA-LIBERO90 cases;
- 3 Octo-small-1.5 cases;
- prioritizing high target-angle examples and wrong-target cases.

### Recommended cases

| Policy | Example ID | Original instruction | Counterfactual instruction | Schema target | ActCheck label |
|---|---|---|---|---|---|
| OpenVLA-LIBERO90 | `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo:demo_31:102::visual_relation_swap` | pick up the object to the left of the ramekin | pick up the object to the right of the black bowl | ramekin | wrong-target |
| OpenVLA-LIBERO90 | `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo:demo_45:124::visual_relation_swap` | pick up the object to the left of the cookie box | pick up the object to the right of the black bowl | cookie box | wrong-target |
| OpenVLA-LIBERO90 | `open_the_middle_drawer_of_the_cabinet_demo:demo_42:28::visual_relation_swap` | pick up the object to the behind the top drawer | pick up the object to the in front of the middle drawer | top drawer | wrong-target |
| Octo-small-1.5 | `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo:demo_8:70::visual_relation_swap` | pick up the object to the left of the ramekin | pick up the object to the right of the black bowl | ramekin | ambiguous |
| Octo-small-1.5 | `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo:demo_45:47::visual_relation_swap` | pick up the object to the left of the cookie box | pick up the object to the right of the black bowl | cookie box | ambiguous |
| Octo-small-1.5 | `open_the_middle_drawer_of_the_cabinet_demo:demo_42:6::visual_relation_swap` | pick up the object to the behind the top drawer | pick up the object to the in front of the middle drawer | top drawer | ambiguous |

The full CSV includes:

- `obs_ptr`: HDF5 image pointer;
- `original_instruction`;
- `counterfactual_instruction`;
- `original_target`;
- `counterfactual_target`;
- `qwen_pred_target`;
- `qwen_reason`;
- `target_angle_deg`;
- `target_distance`;
- `actcheck_label`;
- cosine alignment with counterfactual target;
- cosine alignment with original target;
- `semantic_action_consistency_score`;
- `action_sensitive`;
- `normalized_action_delta`.

### Source files used to build the case packet

Benchmark and images:

```text
outputs/linguistic_blindness/visual_relation_counterfactual_600/benchmark.jsonl
```

Qwen relation schema predictions:

```text
outputs/linguistic_blindness/qwen25vl7b_visual_relation_schema_600/predictions.jsonl
```

OpenVLA ActCheck predictions:

```text
outputs/linguistic_blindness/visual_relation_actcheck/actcheck_predictions.jsonl
```

Octo ActCheck predictions:

```text
outputs/linguistic_blindness/octo_visual_relation_analysis/relation_actcheck_predictions.jsonl
```

### How to load a scene image from `obs_ptr`

The `obs_ptr` format is:

```text
data/libero_raw/...hdf5#/data/demo_X/obs/agentview_rgb/FRAME
```

Existing loader:

```python
from scripts.lb_run_openvla_action import load_hdf5_image_pointer

image = load_hdf5_image_pointer(obs_ptr)
image.save("case.png")
```

Use this to export the scene image for each selected case.

### What to visualize

For each selected case, draw:

- a green arrow: end-effector to schema/counterfactual target;
- a red arrow: end-effector to original/wrong target;
- a blue arrow: native policy action direction projected into image/scene coordinates if available;
- label: `wrong-target` or `ambiguous`.

If 2D projection is too time-consuming, use the raw image plus a small side table of cosine scores:

```text
cos(action, schema target)
cos(action, original target)
semantic-action consistency = cos_schema - cos_original
```

This is enough for an appendix figure.

## 2. Bootstrap CI / Uncertainty

### Generated CI table

Main CI file:

```text
outputs/linguistic_blindness/qualitative_ci_packet/bootstrap_ci_core_supplement.csv
```

This file contains bootstrap 95% intervals for:

- OpenVLA relation action sensitivity;
- OpenVLA relation ActCheck rates;
- Octo relation action sensitivity;
- Octo relation ActCheck rates;
- Qwen pixel-grounded relation accuracy;
- short-horizon approach rates.

Bootstrap settings:

- resampling unit: example;
- bootstrap samples: 10,000;
- random seed: 13.

### CI values to report

| Table | Policy | Metric | N | Mean | 95% CI |
|---|---|---|---:|---:|---:|
| OpenVLA relation | OpenVLA-LIBERO90 | action sensitivity | 600 | 0.077 | [0.057, 0.098] |
| OpenVLA relation | OpenVLA-LIBERO90 | target-aligned action | 600 | 0.545 | [0.505, 0.585] |
| OpenVLA relation | OpenVLA-LIBERO90 | wrong-target action | 600 | 0.285 | [0.250, 0.322] |
| OpenVLA relation | OpenVLA-LIBERO90 | ambiguous alignment | 600 | 0.170 | [0.140, 0.200] |
| OpenVLA relation | OpenVLA-LIBERO90 | ActCheck flag | 600 | 0.455 | [0.415, 0.495] |
| Octo relation | Octo-small-1.5 | action sensitivity | 600 | 0.375 | [0.338, 0.415] |
| Octo relation | Octo-small-1.5 | target-aligned action | 600 | 0.572 | [0.532, 0.610] |
| Octo relation | Octo-small-1.5 | wrong-target action | 600 | 0.393 | [0.353, 0.432] |
| Octo relation | Octo-small-1.5 | ambiguous alignment | 600 | 0.035 | [0.022, 0.050] |
| Octo relation | Octo-small-1.5 | ActCheck flag | 600 | 0.428 | [0.388, 0.468] |
| Pixel-grounded relation | Qwen2.5-VL-7B | grounding accuracy | 600 | 0.958 | [0.942, 0.973] |
| Short horizon | OpenVLA-LIBERO90 | counterfactual instruction, CF approach | 25 | 0.560 | [0.360, 0.760] |
| Short horizon | OpenVLA-LIBERO90 | original instruction, CF approach | 25 | 0.760 | [0.600, 0.920] |

### Source files used for CI

OpenVLA relation ActCheck:

```text
outputs/linguistic_blindness/visual_relation_actcheck/actcheck_predictions.jsonl
```

Octo relation ActCheck:

```text
outputs/linguistic_blindness/octo_visual_relation_analysis/relation_actcheck_predictions.jsonl
```

Qwen relation grounding:

```text
outputs/linguistic_blindness/qwen25vl7b_visual_relation_schema_600/predictions.jsonl
```

Short-horizon rollouts:

```text
outputs/linguistic_blindness/short_horizon_target_approach_50/short_horizon_rollouts.jsonl
```

### Suggested writing

For the main/appendix table:

> We report bootstrap 95% confidence intervals over examples. The relation-defined split shows that Qwen-VL grounds the target with 95.8% accuracy [94.2, 97.3], while OpenVLA action sensitivity remains 7.7% [5.7, 9.8] and Octo action sensitivity is 37.5% [33.8, 41.5]. Thus, Octo reduces but does not eliminate the semantic-action gap.

For ActCheck:

> VISA-ActCheck flags 45.5% [41.5, 49.5] of OpenVLA relation-swap actions and 42.8% [38.8, 46.8] of Octo actions as wrong-target or ambiguous, showing that action sensitivity and semantic-action consistency are complementary diagnostics.

For short horizon:

> In a 25-pair K=10 short-horizon diagnostic, counterfactual instructions do not increase approach toward the counterfactual target relative to original instructions, indicating that the effect is not merely a one-step artifact. We keep this as a diagnostic rather than a task-success claim.

