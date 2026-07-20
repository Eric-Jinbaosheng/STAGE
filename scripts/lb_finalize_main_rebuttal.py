#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: List[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def analysis_python() -> str:
    env_python = Path(".conda_vlm/bin/python")
    return str(env_python) if env_python.exists() else sys.executable


def first_existing(paths: List[str]) -> Path | None:
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    return None


def fmt(x: Any) -> str:
    try:
        if x is None or x == "":
            return "NA"
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/linguistic_blindness/main_rebuttal_final")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sat_actions = first_existing(
        [
            "outputs/linguistic_blindness/satbenchpp_v1/openvla_action_sensitivity_1000/predictions.jsonl",
            "outputs/linguistic_blindness/satbenchpp_v1/openvla_action_sensitivity_1000_alt1091/predictions.jsonl",
        ]
    )
    if sat_actions:
        run(
            [
                analysis_python(),
                "scripts/lb_analyze_satbenchpp.py",
                "--benchmark",
                "outputs/linguistic_blindness/satbenchpp_v1/benchmark.jsonl",
                "--actions",
                str(sat_actions),
                "--out-dir",
                "outputs/linguistic_blindness/satbenchpp_v1/analysis",
            ]
        )

    k50_rollouts = first_existing(
        [
            "outputs/linguistic_blindness/short_horizon_target_approach_k50_100/short_horizon_rollouts.jsonl",
            "outputs/linguistic_blindness/short_horizon_target_approach_k50_100_alt1091/short_horizon_rollouts.jsonl",
        ]
    )
    if k50_rollouts:
        run(
            [
                analysis_python(),
                "scripts/lb_analyze_actcheck_predicts_rollout.py",
                "--rollouts",
                str(k50_rollouts),
                "--out-dir",
                "outputs/linguistic_blindness/actcheck_predicts_short_horizon_k50_100",
            ]
        )

    sat = read_csv(Path("outputs/linguistic_blindness/satbenchpp_v1/analysis/satbenchpp_summary.csv"))
    k50 = read_csv(Path("outputs/linguistic_blindness/actcheck_predicts_short_horizon_k50_100/actcheck_rollout_group_summary.csv"))
    visa_r_metrics = read_json(
        first_existing(
            [
                "outputs/linguistic_blindness/visa_r_prompt_rewrite_target_swap_600_concise/metrics.json",
                "outputs/linguistic_blindness/visa_r_prompt_rewrite_target_swap_600_concise_alt1091/metrics.json",
            ]
        )
        or Path("__missing__")
    )

    md = ["# Main Rebuttal Final Tables", ""]
    md.append("## SAT-Bench++")
    if sat:
        md += [
            "",
            "| Split | N | Sem. recovery | Native ActSens | SAG | Native ActCheck | VISA-R handoff | AUC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in sat:
            md.append(
                f"| {r['split']} | {r['n']} | {fmt(r['semantic_recovery'])} | {fmt(r['native_action_sensitivity'])} "
                f"| {fmt(r['semantic_action_gap'])} | {fmt(r['native_actcheck_aligned'])} "
                f"| {fmt(r['visa_r_handoff_actcheck_aligned'])} | {fmt(r['target_vs_control_auc'])} |"
            )
    else:
        md.append("\nPending: SAT-Bench++ OpenVLA action predictions not found yet.")

    md.append("\n## K=50 ActCheck Predicts Rollout")
    if k50:
        md += [
            "",
            "| First-step ActCheck | N | CF approach | Original approach | Final pref | Integrated pref |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in k50:
            md.append(
                f"| {r['group']} | {r['n']} | {fmt(r['counterfactual_approach_rate'])} "
                f"| {fmt(r['original_approach_rate'])} | {fmt(r['mean_final_target_preference'])} "
                f"| {fmt(r['mean_integrated_target_preference'])} |"
            )
    else:
        md.append("\nPending: K=50 rollout predictions not found yet.")

    md.append("\n## VISA-R Prompt Rewrite")
    if visa_r_metrics:
        md += [
            "",
            f"- N: {visa_r_metrics.get('num_examples')}",
            f"- Raw counterfactual action sensitivity: {fmt(visa_r_metrics.get('raw_counterfactual_action_sensitivity'))}",
            f"- Schema-rewrite action sensitivity: {fmt(visa_r_metrics.get('schema_rewrite_action_sensitivity'))}",
            f"- Rewrite sensitivity gain: {fmt(visa_r_metrics.get('rewrite_sensitivity_gain'))}",
            f"- Rewrite improves over raw rate: {fmt(visa_r_metrics.get('rewrite_improves_over_raw_rate'))}",
        ]
    else:
        md.append("\nPending: VISA-R prompt-rewrite metrics not found yet.")

    (out / "main_rebuttal_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "satbenchpp_ready": bool(sat), "k50_ready": bool(k50), "visa_r_ready": bool(visa_r_metrics)}, indent=2))


if __name__ == "__main__":
    main()
