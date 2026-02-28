# Week1 Checklist Status

This file documents the concrete Week1 outputs and commands.

## Required outputs

- `data/raw/handoversim/*.json or *.jsonl` from simulator export
- `data/proc/episodes.parquet` + `data/proc/meta.json` (fallback `episodes.jsonl` if parquet unavailable)
- `summaries/*.jsonl` + `summaries/summaries.parquet` (fallback `summaries.jsonl`)
- `viz/curves/*.png` + `viz/keyframes/*/keyframes.json`

## Repro commands (3-command version)

0. pull-ready cache -> raw (10 episodes)
```powershell
$env:PYTHONPATH="src"
python scripts/export_raw_from_dexycb_cache.py `
  --cache-dir third_party/handover-sim/handover/data/dex-ycb-cache `
  --out-dir data/raw/handoversim `
  --num-episodes 10 `
  --start-index 0 `
  --dt 0.01 `
  --max-frames 600
```

1. raw -> proc
```powershell
$env:PYTHONPATH="src"
python scripts/build_proc_dataset.py `
  --raw-dir data/raw/handoversim `
  --proc-dir data/proc `
  --mapping configs/handover_sim_mapping.example.json
```

2. proc -> summaries
```powershell
$env:PYTHONPATH="src"
python scripts/build_summaries_dataset.py `
  --proc-episodes-dir data/proc/episodes `
  --summaries-dir summaries `
  --window 10 `
  --config configs/summaries.yaml
```

3. summaries -> viz
```powershell
$env:PYTHONPATH="src"
python scripts/make_week1_viz.py `
  --summaries-dir summaries `
  --out-dir viz `
  --num-episodes 3
```

## Notes

- The pipeline is ready for real Handover-Sim dumps.
- If parquet libraries are missing in environment, scripts auto-fallback to JSONL and keep `meta.json` for reproducibility.
