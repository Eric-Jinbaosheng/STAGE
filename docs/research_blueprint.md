# Research Blueprint: VISA for Language-to-Action Alignment

## Working Title

**VISA: Verifiable Interaction Schema Alignment for Diagnosing Language-to-Action Bottlenecks in Embodied Agents**

A more descriptive title is:

**Verifiable Interaction Schemas for Language-to-Action Alignment in Embodied VLA Agents**

## Central Claim

Embodied VLA agents may understand instructions at the VLM level, yet fail to preserve that understanding at the action-decision interface.

The paper should not claim that VLMs simply do not understand language. The stronger and more precise claim is that language understanding can exist in the backbone or schema probe, while the action head fails to faithfully use that semantic signal.

## Problem Reframing

Old framing:

```text
VLM/VLA does not understand language, so behavior is wrong.
```

New framing:

```text
The VLM backbone may understand the instruction, but the language semantics are not stably transferred into action decisions.
```

This makes linguistic blindness an **agent-level** failure, not necessarily a pure VLM-backbone failure.

## Method: VISA

VISA replaces an implicit, hard-to-check action interface with an explicit, verifiable semantic interface.

Standard VLA pipeline:

```text
Image + Instruction
  -> VLM backbone
  -> hidden states / KV cache
  -> action head
  -> robot action
```

VISA pipeline:

```text
Image + Instruction
  -> VLM schema probe or structured parser
  -> explicit interaction schema
  -> deterministic consistency checker
  -> safety / uncertainty gate
  -> allowed symbolic action or safe deferral
  -> low-level controller or action adapter
```

The key design choice is not to assume that the action head faithfully extracts target, phase, prohibition, safety, and uncertainty information from hidden states. VISA externalizes that information into a checkable schema.

## Module 1: Semantic Schema Probe

Purpose: test whether the VLM-level representation captures the instruction semantics.

Input:

- image or observation if available
- robot/human state if available
- natural-language instruction

Output schema:

```json
{
  "target_object": "bowl",
  "target_exists": true,
  "phase": "reach",
  "human_contact": null,
  "human_released": null,
  "robot_contact": false,
  "robot_grasp_stable": false,
  "allowed_actions": ["MOVE_TO", "GRASP", "ASK"],
  "blocked_actions": ["RETRACT"],
  "next_action": "MOVE_TO",
  "reason": "The instruction asks for the bowl.",
  "confidence": 0.82
}
```

This module is a diagnostic probe first. It asks whether VLM-level semantics change correctly under counterfactual instruction changes.

## Module 2: Consistency Checker

Purpose: deterministically check whether a schema is valid, instruction-sensitive, and safe.

Core rules:

- Target swap: the target should change to the counterfactual target.
- Blank instruction: task-specific actions should be blocked or deferred.
- Impossible target: nonexistent targets should produce rejection or deferral.
- Safety conflict: if `human_released=false`, `RETRACT`, `TRANSFER`, and `PICK` should not execute.
- Negation: prohibited actions should appear in `blocked_actions` and not be recommended.
- Constraint consistency: `next_action` must not be in `blocked_actions`; if `allowed_actions` is non-empty, `next_action` should be allowed.
- Phase consistency: predicted phase should be compatible with observed contact/release state.

## Module 3: Safety / Uncertainty Gate

Purpose: prevent invalid or unsafe schema outputs from becoming action recommendations.

If schema is valid, safe, allowed, not blocked, and confident enough, the gate preserves `next_action`.

If checker flags invalidity, uncertainty, instruction conflict, or safety risk, the gate replaces the action with one of:

```text
ASK, PROMPT, WAIT, HOLD, ABORT, BACKOFF, TARGET_NOT_FOUND
```

Example:

```json
{
  "human_released": false,
  "next_action": "RETRACT"
}
```

Checker flags `safety_violation=true`; gate returns `HOLD`.

## Failure Levels

VISA distinguishes where the failure happens:

| Failure Level | Definition |
|---|---|
| VLM Semantic Failure | VLM schema probe is wrong. |
| Semantic-to-Action Transfer Failure | VLM schema probe is correct, but action head/output is wrong. |
| Verification / Safety Failure | Schema or action violates constraints or safety rules. |
| Execution Bias | Agent executes under blank, impossible, contradictory, or prohibited instructions. |

The most important category for the revised paper is **Semantic-to-Action Transfer Failure**: VLM knows, action head fails.

## Core Metrics

### VLM Schema Sensitivity

Whether the VLM schema probe changes correctly under counterfactual instructions.

### Action Sensitivity

Whether the final action or action-head output changes correctly under counterfactual instructions.

### Semantic-Action Sensitivity Gap

```text
SASG = VLM Schema Sensitivity - Action Sensitivity
```

A large positive gap supports the language-to-action bottleneck claim.

### Instruction Sensitivity Gap

```text
ISG = Normal Instruction Accuracy - Counterfactual Sensitivity
```

A large positive gap means standard task/action success overestimates instruction following.

## Paper Story

1. Standard embodied evaluation often rewards task/action success.
2. Task success can hide instruction-following failures because vision, object salience, or default policies may be enough.
3. Fixed-observation counterfactual instructions isolate whether language changes affect action-relevant semantics.
4. Linguistic blindness is an agent-level failure: the action-relevant semantics or final action fail to change appropriately when language changes.
5. VLM backbones may still recover the correct semantics when probed explicitly.
6. Therefore, a key failure mode is the language-to-action bottleneck.
7. VISA externalizes interaction semantics into a verifiable schema.
8. Checker and gate reveal and reduce invalid, blind, and unsafe recommendations.
9. The paper is about language-conditioned semantic decision and verification, not low-level robot control.

## Minimum Viable Paper

A complete minimal version needs:

- 500-1000 counterfactual examples.
- Qwen2.5-VL-7B schema probe.
- One VLA/action-head or action-label baseline.
- Semantic-action gap analysis.
- Parser/checker/gate baselines.
- Main results, mitigation table, ablation table, and failure taxonomy.

The minimum core finding is:

```text
VLM Schema Sensitivity > Action Sensitivity
```

This supports the claim that language semantics can exist at the VLM/schema level while being lost at the action interface.

## Strong Version

A stronger version adds:

- Multiple VLMs or VLAs, such as Qwen2.5-VL, InternVL/LLaVA, and OpenVLA-style action heads.
- Multiple domains, such as manipulation and handover/contact states.
- 2000+ counterfactual pairs.
- Human validation of a subset of schema labels.
- Replay-based safety evaluation.
- Small rollout sanity check.
- Qualitative semantic-action transfer failure cases.
