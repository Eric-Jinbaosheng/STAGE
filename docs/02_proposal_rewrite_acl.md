# Language-Conditioned Interaction Schemas

## Title

Language-Conditioned Interaction Schemas for Fixing Instruction Blindness in LIBERO and Enabling Verifiable Recovery Under Uncertainty

## Motivation

Vision-language systems often appear to accept natural-language instructions, yet in practice they can become instruction-blind: replacing, shuffling, or blanking the instruction changes behavior only slightly because the model relies on visual shortcuts. At the same time, interactive manipulation is partially observable and failure-prone. Occlusion, ambiguous contact, hesitation, withdrawal, and non-release create situations where end-to-end policies are difficult to audit and unsafe when uncertain.

This project aims to build a system that is:

- language-sensitive: instruction changes must change the internal goal representation
- verifiable: outputs must be machine-checkable
- robust under uncertainty: unknown or conflicting evidence should trigger conservative recovery behavior

## Key Idea

We map sparse language into a dense, constrained intermediate representation called an **Interaction Schema**. Instead of directly predicting actions from `(image, instruction)`, we first predict a structured schema, then use verification rules and uncertainty signals to constrain downstream decisions.

The core claim is: if the schema is explicitly conditioned on language and forced to change under counterfactual instruction perturbations, then the system becomes less instruction-blind and easier to audit.

## System Overview

1. Structured perceptual summaries  
From RGB observations, object state proxies, proprioception, gripper width, and contact proxies, compute short-window evidence features and relation scores.

2. Schema predictor (current implemented baseline)  
Train a multimodal schema predictor on frozen weak labels. The current implementation is a small multimodal classifier (image + text + numeric features) trained from scratch; the later extension is to replace this with a pretrained VLM backbone and fine-tune it.

3. Checker / safety shield  
Apply deterministic checks over the schema fields and action feasibility set. The checker validates structure and removes unsafe actions, but it does not hard-code the policy.

4. Calibrator (planned extension)  
Estimate calibrated probabilities such as `secure_prob`, `release_prob`, `occlusion_prob`, and `conflict_score` to enable abstention and explicit `UNK` states.

5. Optional schema-conditioned policy (planned extension)  
Condition downstream action prediction on both observation and schema, to test whether improved schema-language coupling improves action-language coupling.

## Minimal Interaction Schema

To keep training stable, we fix a small, machine-checkable output space:

- `phase` (enum)
  - handover: `reach, align, contact, transfer, secure, retract, failure`
  - LIBERO-style manipulation: `reach, grasp, manipulate, place, done`
- `target_object` or `goal_id` (categorical)
- `state` (tri-valued)
  - `contact_confirmed in {T, F, UNK}`
  - `object_secured in {T, F, UNK}`
  - `occluded in {T, F, UNK}`
  - handover-only: `human_released in {T, F, UNK}`
- `affordance_mask` over fixed primitives
  - `HOLD, BACKOFF_SMALL, VIEWPOINT_CHANGE, REALIGN, CLOSE_GENTLE, RETRACT, PROMPT`
- `conflict_or_uncertainty`
  - scalar probability or discrete level such as `low / medium / high`

This schema is intentionally robot-agnostic and does not depend on joint-token counts.

## Training Objective

The main mechanism for fixing instruction blindness is **counterfactual instruction training**.

For each sample `(x, instr, y_schema)`, construct a perturbed instruction:

- `swap`: replace the referenced object with another object present in the same scene
- `shuffle`: randomly pair instructions across a batch
- `blank`: remove or replace the instruction with meaningless text

Training includes:

1. Supervised schema loss  
Standard CE/BCE losses for:
- `target_object`
- `phase`
- tri-valued state fields
- `affordance_mask`

2. Counterfactual schema loss  
For the same observation `x`, changing the instruction should change the schema in the correct direction.

Examples:
- `target_object(x, instr)` should match the intended target
- `target_object(x, instr_wrong)` should not remain equal to the original target, or should move to `UNK / high_conflict`

This explicitly penalizes instruction-invariant behavior.

3. Calibration loss (planned extension)  
Use weak supervision and future-consistency signals to calibrate `secure_prob` and `conflict_score`.

## Data Plan

The project uses different datasets for different roles.

- **LIBERO (primary language-sensitivity benchmark)**  
Use as the main benchmark for instruction grounding. Evaluate how schema predictions change under `correct / blank / shuffle / swap` instructions.

- **GenH2R or Handover-Sim (large-scale simulation + controllable anomalies)**  
Use for handover summaries, controllable unexpected events, and robustness experiments under `withdrawal`, `non-release`, `occlusion`, and `ambiguous contact`.

- **DexH2R (real-world validation)**  
Use to test whether schema predictions and uncertainty behavior generalize under real sensor noise and embodiment mismatch.

- **KTH-RPL or similar multimodal handover datasets (auxiliary event timing)**  
Use to improve phase/event segmentation and release timing cues when needed.

## Evaluation

We evaluate two axes: language sensitivity and robustness under uncertainty.

### A. Language Sensitivity

- task success under:
  - `correct instruction`
  - `blank instruction`
  - `shuffled instruction`
  - `swapped instruction`
- `Language Sensitivity Score`:
  - `SR(correct) - SR(perturbed)`
- schema-only sensitivity metrics:
  - `Target Flip Rate` under swap
  - `Schema Change Rate` under blank/shuffle
  - `target_object` grounding accuracy

### B. Reliability and Recovery

- schema validity rate
- safety invariant violation rate after the checker
- recovery success under injected anomalies
- uncertainty calibration metrics (`ECE`, `Brier`, optionally calibration curves)

## Current Implemented Baseline vs Planned Extension

### Current implemented baseline

- frozen weak labels for `target_object` and `phase`
- multimodal baseline model trained from scratch
- image + text + numeric inputs
- batch evaluation pipeline
- episode-level train / val / test splitting

### Planned next extension

- explicit counterfactual instruction training (`blank / shuffle / swap`)
- schema-level contrastive or consistency loss
- checker + uncertainty calibrator
- optional replacement of the current image-text backbone with a pretrained VLM for true fine-tuning

## Expected Contributions

1. A language-conditioned schema interface that turns sparse language into dense, checkable control signals.
2. A counterfactual training recipe that explicitly reduces instruction blindness at the schema level.
3. A verifiable safety shield and uncertainty-aware recovery framework for partially observable manipulation.
4. A practical baseline-to-extension path: from a trainable multimodal schema predictor to a stronger schema-VLM system.
