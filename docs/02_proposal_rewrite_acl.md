# Interaction Schema for Robust Human-to-Robot Handover

## Title

Interaction Schema for Robust Human-to-Robot Handover: Perception-First Structured Summaries, MLLM Semantic Parsing, and Calibrated Uncertainty

## Problem

Human-to-robot handover remains brittle under partial observability and diverse human behaviors (hesitation, withdrawal, non-release, occlusion). Policy-only approaches, including scalable simulation-trained methods, often miss a reusable and machine-checkable interaction representation for safe failure detection and recovery.

## Key Idea

We introduce an **Interaction Schema**: a fixed, interpretable, machine-checkable structured meaning representation that explicitly encodes:

- phase progression
- constraints
- affordance set
- calibrated uncertainty

The system uses a **perception-first (B2) pipeline**: structured perceptual summaries are produced first, then an MLLM performs constrained semantic parsing to schema JSON.

## Method

1. Perception + Tracking -> Structured Summaries  
Track hand/object/gripper and compute short-window kinematics, visibility/occlusion proxies, and optional force/torque feedback. Output includes:
- entity table
- relation candidates with continuous scores:
  - approaching
  - aligned
  - contact_possible
  - contact_confirmed
  - held_by_human
  - held_by_robot
  - human_released
  - occluded

2. Learned Conflict and Uncertainty Calibrator  
A lightweight temporal model maps summaries to calibrated probabilities:
- `release_prob`
- `secure_prob`
- `conflict_score`
- `occlusion_prob`
- `phase_regress_prob`

Training signals:
- time-consistency self-supervision ("future contradiction")
- weak labels from geometry and proprioceptive signals

3. MLLM Schema Generator  
Given summaries (+ optional keyframes) and calibrated scores, a pretrained MLLM outputs constrained JSON schema instances.  
Schema includes:
- phase set: `{offer, reach, align, contact, transfer, secure, retract, failure}`
- constraints
- scored affordances
- uncertainty fields (`contact_confirmed`, `human_released`, `object_secured`, `occluded`)

4. Schema Checking and Repair as Safety Shield  
Apply hard checks + semantic consistency rules to validate outputs and trigger targeted repair.  
The checker enforces safety invariants and constrains feasible actions, instead of hard-coding a full policy.

## Data Plan

- Large-scale training + controllable OOD: GenH2R and/or Handover-Sim  
Inject `not_releasing`, `withdrawal`, `occlusion`, `ambiguous_contact`.
- Real-world validation: DexH2R  
Evaluate sim-to-real generalization and failure-aware calibration.
- Auxiliary timing priors: KTH-RPL and/or HOH  
Improve phase/event boundaries and intent cues.

## Evaluation

Schema-level metrics:
- schema validity rate (hard checks)
- phase accuracy / F1
- constraint consistency
- uncertainty calibration (ECE, Brier)
- conflict detection AUC

Task-level metrics:
- handover success rate
- time-to-secure
- safety violations (near-hand speed, force/torque peaks if available)
- recovery success under unexpected events

OOD protocol:
- object shift
- human behavior shift
- scene and visibility shift

## Expected Contributions

1. A reusable H2R Interaction Schema with explicit uncertainty.
2. A perception-first MLLM semantic parsing pipeline with machine-checkable validation.
3. Failure-aware recovery improvements in safety and OOD robustness.
