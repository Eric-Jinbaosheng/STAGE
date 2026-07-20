#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def fnum(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


def mean(xs: Iterable[Any]) -> float | None:
    vals = [fnum(x) for x in xs]
    vals = [x for x in vals if x is not None]
    return None if not vals else float(sum(vals) / len(vals))


def rate(xs: Iterable[Any]) -> float | None:
    vals = [x for x in xs if x is not None]
    return None if not vals else float(sum(bool(x) for x in vals) / len(vals))


def auc(pos: Iterable[Any], neg: Iterable[Any]) -> float | None:
    ps = [fnum(x) for x in pos]
    ns = [fnum(x) for x in neg]
    ps = [x for x in ps if x is not None]
    ns = [x for x in ns if x is not None]
    if not ps or not ns:
        return None
    wins = ties = 0
    for p in ps:
        for n in ns:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return float((wins + 0.5 * ties) / (len(ps) * len(ns)))


def vec3(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)[:3]


def unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n < 1e-12 else v / n


def cos(a: np.ndarray, b: np.ndarray) -> float | None:
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None:
        return None
    return float(np.dot(ua, ub))


def actcheck(row: Dict[str, Any], margin: float) -> Dict[str, Any]:
    obs = row.get("observation_state") or {}
    ee = vec3(obs.get("ee_pos"))
    po = vec3(obs.get("original_target_pos"))
    pc = vec3(obs.get("counterfactual_target_pos"))
    ao = vec3(row.get("openvla_action_original"))
    ac = vec3(row.get("openvla_action_counterfactual"))
    orig_to_orig = cos(ao, po - ee)
    orig_to_cf = cos(ao, pc - ee)
    cf_to_cf = cos(ac, pc - ee)
    cf_to_orig = cos(ac, po - ee)
    aligned = cf_to_cf is not None and cf_to_orig is not None and cf_to_cf > cf_to_orig + margin
    wrong = cf_to_cf is not None and cf_to_orig is not None and cf_to_orig >= cf_to_cf + margin
    ambiguous = cf_to_cf is not None and cf_to_orig is not None and abs(cf_to_cf - cf_to_orig) <= margin
    return {
        "orig_action_cos_to_orig_target": orig_to_orig,
        "orig_action_cos_to_cf_target": orig_to_cf,
        "cf_action_cos_to_cf_target": cf_to_cf,
        "cf_action_cos_to_orig_target": cf_to_orig,
        "cf_target_aligned_action": aligned,
        "cf_wrong_target_aligned_action": wrong,
        "cf_ambiguous_target_alignment": ambiguous,
        "actcheck_flag": bool(wrong or ambiguous),
        "actcheck_label": "aligned" if aligned else ("wrong-target" if wrong else ("ambiguous" if ambiguous else "unknown")),
        "semantic_action_consistency_score": None if cf_to_cf is None or cf_to_orig is None else cf_to_cf - cf_to_orig,
        "visa_r_handoff_target_aligned": True if (wrong or ambiguous) else aligned,
        "visa_r_intervention": bool(wrong or ambiguous),
    }


def summarize(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    sem = rate(r.get("qwen_schema_sensitive") for r in rows)
    action = rate(r.get("openvla_action_sensitive") for r in rows)
    return {
        "split": label,
        "n": len(rows),
        "semantic_recovery": sem,
        "native_action_sensitivity": action,
        "semantic_action_gap": None if sem is None or action is None else sem - action,
        "native_actcheck_aligned": rate(r.get("cf_target_aligned_action") for r in rows),
        "native_wrong_or_ambiguous": rate((r.get("cf_wrong_target_aligned_action") or r.get("cf_ambiguous_target_alignment")) for r in rows),
        "visa_r_handoff_actcheck_aligned": rate(r.get("visa_r_handoff_target_aligned") for r in rows),
        "visa_r_intervention_rate": rate(r.get("visa_r_intervention") for r in rows),
        "target_vs_control_auc": auc([r.get("normalized_action_delta") for r in rows], [r.get("control_normalized_action_delta") for r in rows]),
        "mean_normalized_delta": mean(r.get("normalized_action_delta") for r in rows),
        "mean_control_delta": mean(r.get("control_normalized_action_delta") for r in rows),
        "mean_semantic_action_consistency": mean(r.get("semantic_action_consistency_score") for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="outputs/linguistic_blindness/satbenchpp_v1/benchmark.jsonl")
    ap.add_argument("--actions", required=True)
    ap.add_argument("--out-dir", default="outputs/linguistic_blindness/satbenchpp_v1/analysis")
    ap.add_argument("--alignment-margin", type=float, default=0.05)
    args = ap.parse_args()

    bench = {r["example_id"]: r for r in read_jsonl(args.benchmark)}
    action_rows = read_jsonl(args.actions)
    rows: List[Dict[str, Any]] = []
    for a in action_rows:
        b = bench.get(a.get("example_id"))
        if not b:
            continue
        row = {**b, **a}
        row.update(actcheck(row, args.alignment_margin))
        rows.append(row)

    summaries = [summarize(rows, "overall")]
    for split in sorted({r.get("satbenchpp_split") for r in rows}):
        summaries.append(summarize([r for r in rows if r.get("satbenchpp_split") == split], str(split)))
    for fam in sorted({r.get("satbenchpp_family") for r in rows}):
        summaries.append(summarize([r for r in rows if r.get("satbenchpp_family") == fam], str(fam)))

    counts = Counter(r.get("actcheck_label") for r in rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "satbenchpp_predictions_augmented.jsonl", rows)
    write_csv(out / "satbenchpp_summary.csv", summaries)
    (out / "actcheck_label_counts.json").write_text(json.dumps(dict(counts), indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# SAT-Bench++ Summary",
        "",
        "| Split | N | Sem. recovery | Native ActSens | SAG | Native ActCheck | VISA-R handoff | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        md.append(
            f"| {s['split']} | {s['n']} | {s['semantic_recovery']:.3f} | {s['native_action_sensitivity']:.3f} "
            f"| {s['semantic_action_gap']:.3f} | {s['native_actcheck_aligned']:.3f} "
            f"| {s['visa_r_handoff_actcheck_aligned']:.3f} | {s['target_vs_control_auc']:.3f} |"
        )
    md += [
        "",
        "VISA-R handoff is an action-boundary repair proxy: wrong/ambiguous native actions are routed to a schema-conditioned target-direction handoff candidate. It is not counted as full low-level control success.",
    ]
    (out / "satbenchpp_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "num_rows": len(rows), "actcheck_labels": dict(counts)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
