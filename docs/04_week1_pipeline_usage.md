# Week 1 Pipeline Usage

## Input format

Input should be JSONL, one frame per line, matching `FrameRecord` fields in:
- `src/h2r_schema/types.py`

Mandatory per-frame keys:
- `timestamp`
- `human_hand_pose.position_xyz`, `human_hand_pose.orientation_xyzw`
- `object_pose.position_xyz`, `object_pose.orientation_xyzw`
- `gripper_pose.position_xyz`, `gripper_pose.orientation_xyzw`
- `hand_object_distance`, `gripper_object_distance`
- `hand_velocity`, `gripper_velocity`, `object_velocity`
- `visibility_score`
- `contact_signal_hand_object`, `contact_signal_gripper_object`
- `gripper_opening`

## Build summaries

```powershell
$env:PYTHONPATH="src"
python scripts/build_summaries.py `
  --input data/examples/sample_episode.jsonl `
  --output artifacts/summaries/sample_summary.jsonl `
  --episode-id sample_episode `
  --window 10
```

## Output fields

Per frame output:
- `episode_id`
- `timestamp`
- `entities`
- `relations` (9 core candidates + 2 diagnostic speeds)
- `signals`

Core relation keys:
- `approaching_hand_object`
- `approaching_gripper_object`
- `aligned_gripper_object`
- `contact_possible`
- `contact_confirmed`
- `held_by_human`
- `held_by_robot`
- `human_released`
- `occluded_object`

## Adapting to GenH2R / Handover-Sim / DexH2R

Add dataset adapters that convert native annotations to this input JSONL schema first, then call `build_summaries.py`. This keeps a single downstream scoring and schema-generation path.

## Handover-Sim adapter

Convert one file or a directory:

```powershell
$env:PYTHONPATH="src"
python scripts/convert_handover_sim.py `
  --input data/examples/handover_sim_raw_sample.json `
  --output-dir data/unified/handover_sim `
  --mapping configs/handover_sim_mapping.example.json
```

Then build summaries (single file):

```powershell
$env:PYTHONPATH="src"
python scripts/build_summaries.py `
  --input data/unified/handover_sim/handover_sim_raw_sample.jsonl `
  --output artifacts/summaries/handover_sim_raw_sample.summary.jsonl `
  --window 10
```

Batch summaries:

```powershell
$env:PYTHONPATH="src"
python scripts/build_summaries_batch.py `
  --input-dir data/unified/handover_sim `
  --output-dir artifacts/summaries/handover_sim `
  --window 10
```

## GenH2R adapter

```powershell
$env:PYTHONPATH="src"
python scripts/convert_genh2r.py `
  --input data/examples/genh2r_raw_sample.json `
  --output-dir data/unified/genh2r `
  --mapping configs/genh2r_mapping.example.json
```
