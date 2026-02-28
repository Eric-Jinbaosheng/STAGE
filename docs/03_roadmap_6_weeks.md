# 6-Week Execution Roadmap

## Week 1: Unified Data Format + Summaries Extractor

Goals:
- Run at least one full episode from GenH2R and/or Handover-Sim.
- Export unified frame records (`episode.jsonl` or parquet).
- Implement `summaries_builder` for 9 relation candidate scores.

Deliverables:
- 100 episodes of summaries
- plots for key trajectories and relation scores
- sample keyframe visualizations

## Week 2: MLLM-to-Schema Baseline + Minimal Checker/Repair

Goals:
- Freeze schema JSON format (phase, constraints, affordances, uncertainty).
- Add constrained decoding and JSON validation.
- Implement minimal hard checks and targeted repair.

Deliverables:
- schema validity rate
- first phase prediction baseline (GT or weak labels)

## Week 3: Build Simulated Unexpected OOD Set

Goals:
- Inject four anomaly families:
  - not releasing
  - withdrawal
  - occlusion
  - ambiguous contact
- Define recovery action set:
  - hold
  - backoff
  - prompt
  - viewpoint change
  - realign
  - abort

Deliverables:
- OOD split package
- failure-aware benchmark protocol

## Week 4: Train Calibrator v1

Goals:
- Train temporal calibrator for:
  - `release_prob`
  - `secure_prob`
  - `conflict_score`
- Use future-outcome weak supervision (`secured`, drop, still-held-by-human).

Deliverables:
- calibrator checkpoint
- calibration curves (ECE/Brier)
- ablation: without calibrator vs with calibrator

## Week 5: Real Validation (DexH2R) + Auxiliary Timing (KTH-RPL/HOH)

Goals:
- Run sim-to-real validation on available DexH2R modalities.
- Transfer event timing priors from KTH-RPL/HOH (at least release/transfer timing).

Deliverables:
- sim-to-real metrics
- OOD performance drop report
- 2-3 case studies

## Week 6: Paper Writing and Finalization

Goals:
- Final tables:
  - success
  - safety violation
  - recovery success
  - ECE
- Core ablations:
  - remove checker
  - remove uncertainty fields
  - remove calibrator
- Build explanation figures:
  - evidence -> schema -> action chain

Deliverables:
- camera-ready draft
- appendix with schema and checks
