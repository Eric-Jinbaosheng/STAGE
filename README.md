# NLP Final Project: H2R Interaction Schema

Week1-ready pipeline:
- raw sim export -> unified proc dataset (`episodes.parquet` + `meta.json`)
- proc episodes -> structured summaries (`summaries.parquet` + `meta.json`)
- sanity-check visualization for 3 episodes (curve SVG + keyframe index JSON)

## Key docs

- `docs/07_week1_checklist_status.md`
- `docs/08_week2_usage.md`
- `docs/09_unified_data_pipeline.md`
- `docs/10_training_machine_migration.md`
- `docs/11_github_repo_data_and_finetune_setup.md`
- `docs/12_finetune_baseline_skeleton.md`
- `docs/13_single_sample_inference.md`
- `docs/14_batch_evaluation.md`
- `docs/15_slurm_torch_training_workflow.md`
- `docs/04_week1_pipeline_usage.md`
- `docs/05_week2_schema_baseline.md`
- `docs/06_week3_ood_injection.md`

## Week1 reproduce (3 commands)

1) raw -> proc

```powershell
$env:PYTHONPATH="src"
python scripts/build_proc_dataset.py `
  --raw-dir data/raw/handoversim `
  --proc-dir data/proc `
  --mapping configs/handover_sim_mapping.example.json
```

2) proc -> summaries

```powershell
$env:PYTHONPATH="src"
python scripts/build_summaries_dataset.py `
  --proc-episodes-dir data/proc/episodes `
  --summaries-dir summaries `
  --window 10 `
  --config configs/summaries.yaml
```

3) summaries -> viz

```powershell
$env:PYTHONPATH="src"
python scripts/make_week1_viz.py `
  --summaries-dir summaries `
  --out-dir viz `
  --num-episodes 3
```

## Notes

- `configs/summaries.yaml` is frozen for Week1 threshold reproducibility.
- `data/processed/meta.json` stores the frozen schema label ruleset version for training-machine transfer checks.
- If `matplotlib` is unavailable, viz script auto-exports curve `.svg` instead of `.png`.
