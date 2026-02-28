# Data-to-Schema Alignment (H2R)

This document defines the core alignment table between datasets, frame-level signals, relation candidates, and schema fields.

## 1. Unified Frame Record

Each frame in an episode should expose:

- `timestamp`
- `human_hand_pose` (position + orientation)
- `object_pose`
- `gripper_pose`
- `hand_object_distance`
- `gripper_object_distance`
- `hand_velocity`
- `gripper_velocity`
- `object_velocity`
- `visibility_score` (or binary visible/occluded proxy)
- `contact_signal_hand_object` (if available)
- `contact_signal_gripper_object` (if available)
- `gripper_opening`
- `force_torque` (optional)

## 2. Entities

The schema entity table should include:

- `human_hand_pose`
- `object_pose`
- `gripper_pose`
- `visibility_state`
- `contact_state`
- `grasp_state_human`
- `grasp_state_robot`

## 3. Relation Candidates and Scoring Signals

Use short temporal windows (`w=5~15` frames) to compute relation scores in `[0, 1]`.

1. `approaching(hand, object)`
- Score up when distance decreases and hand velocity points toward object.

2. `approaching(gripper, object)`
- Score up when gripper-object distance decreases and direction is consistent.

3. `aligned(gripper, object)`
- Score from orientation difference and grasp-axis alignment.

4. `contact_possible`
- High if distance below threshold and relative speed allows plausible contact.

5. `contact_confirmed`
- High if contact sensors fire, or geometry + velocity discontinuity indicate impact.

6. `held_by_human`
- High if hand-object relative transform is near-rigid over time.

7. `held_by_robot`
- High if gripper-object relative transform is near-rigid and gripper is closed.

8. `human_released`
- Event-like score when `held_by_human` drops rapidly from high to low.

9. `occluded(object)`
- High when mask visibility drops, line-of-sight is blocked, or confidence degrades.

## 4. Schema Fields Supervised by Datasets

## 4.1 Simulation Training (GenH2R / Handover-Sim)

Primary use:
- Large-scale normal trajectories
- Controlled OOD anomaly injection

Directly supervisable:
- `phase in {offer, reach, align, contact, transfer, secure, retract, failure}`
- `contact_confirmed`
- `held_by_human`, `held_by_robot`
- `human_released`
- `occluded`
- Safety invariants tied to affordance feasibility

Best synthetic unexpected cases:
- `not_releasing`
- `withdrawal` (human retreats/hesitates)
- `ambiguous_contact` (near-contact without stable contact)
- `occlusion`

## 4.2 Real Validation (DexH2R)

Primary use:
- Sim-to-real generalization checks
- Uncertainty calibration under real perception noise and dynamics

Strong signals for:
- Phase progression quality in real settings
- Outcome-grounded supervision for `object_secured` and failure
- Unknown/abstain behavior under occlusion and ambiguous grasp events

Note:
- Different embodiment is acceptable and useful for morphology-agnostic schema validation.

## 4.3 Auxiliary Event Timing (KTH-RPL / HOH)

Primary use:
- Better temporal boundaries for interaction events

Useful supervision:
- Event boundaries around `reach -> align -> contact -> release -> secure`
- Hesitation/withdrawal timing patterns
- Visibility/occlusion proxies (especially with multi-view + segmentation in HOH)

## 5. Week-1 Output Contract

For each episode, produce:

- `episode_id`
- `frame_records` (unified fields above)
- `relation_scores` (9 candidates per frame)
- `weak_phase_label` (if available)
- `event_markers` (`human_released`, `contact_confirmed`)

Recommended files:

- `data/unified/*.parquet` or `*.jsonl`
- `artifacts/summaries/*.json`
- `artifacts/plots/*.png` (distance, score, phase timelines)
