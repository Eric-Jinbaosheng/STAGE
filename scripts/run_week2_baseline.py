import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bool_to_tristate(v):
    if v is True:
        return True
    if v is False:
        return False
    return "unknown"


def weak_labels_from_summary(s: Dict) -> Dict:
    rel = s.get("relations", {})
    return {
        "contact_confirmed": bool_to_tristate(float(rel.get("contact_confirmed", 0.0)) > 0.6),
        "human_released": bool_to_tristate(float(rel.get("human_released", 0.0)) > 0.6),
        "object_secured": bool_to_tristate(float(rel.get("held_by_robot", 0.0)) > 0.7),
    }


def weak_phase_from_summary(s: Dict) -> str:
    rel = s.get("relations", {})
    if float(rel.get("held_by_robot", 0.0)) > 0.75 and float(rel.get("human_released", 0.0)) > 0.4:
        return "secure"
    if float(rel.get("contact_confirmed", 0.0)) > 0.7 and float(rel.get("human_released", 0.0)) < 0.4:
        return "transfer"
    if float(rel.get("contact_confirmed", 0.0)) > 0.6:
        return "contact"
    if float(rel.get("aligned_gripper_object", 0.0)) > 0.65:
        return "align"
    if float(rel.get("approaching_gripper_object", 0.0)) > 0.5:
        return "reach"
    return "offer"


def state_match_rate(summary_rows: List[Dict], schema_rows: List[Dict], key: str) -> float:
    matched = 0
    total = min(len(summary_rows), len(schema_rows))
    for s, g in zip(summary_rows[:total], schema_rows[:total]):
        w = weak_labels_from_summary(s)[key]
        p = g.get("state", {}).get(key, "unknown")
        if p == w:
            matched += 1
    return matched / max(1, total)


def phase_match_rate(summary_rows: List[Dict], schema_rows: List[Dict]) -> float:
    matched = 0
    total = min(len(summary_rows), len(schema_rows))
    for s, g in zip(summary_rows[:total], schema_rows[:total]):
        if weak_phase_from_summary(s) == g.get("phase", "failure"):
            matched += 1
    return matched / max(1, total)


def make_case_svg(summary_rows: List[Dict], schema_rows: List[Dict], out_path: Path) -> None:
    w, h, pad = 980, 420, 40
    n = min(len(summary_rows), len(schema_rows))
    t = [float(summary_rows[i].get("timestamp", i)) for i in range(n)]
    contact = [float(summary_rows[i].get("relations", {}).get("contact_confirmed", 0.0)) for i in range(n)]
    held_h = [float(summary_rows[i].get("relations", {}).get("held_by_human", 0.0)) for i in range(n)]
    held_r = [float(summary_rows[i].get("relations", {}).get("held_by_robot", 0.0)) for i in range(n)]
    released = [float(summary_rows[i].get("relations", {}).get("human_released", 0.0)) for i in range(n)]
    action_idx = [i for i in range(n) if schema_rows[i].get("next_action", {}).get("chosen") == "PROMPT_HUMAN_RELEASE"]

    y_min, y_max = 0.0, 1.0

    def sx(i: int) -> float:
        return pad + (w - 2 * pad) * (i / max(1, n - 1))

    def sy(v: float) -> float:
        return h - pad - (h - 2 * pad) * ((v - y_min) / (y_max - y_min))

    def poly(vals: List[float], color: str) -> str:
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    parts.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#444"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#444"/>')
    parts.append(poly(contact, "#d62728"))
    parts.append(poly(held_h, "#2ca02c"))
    parts.append(poly(held_r, "#1f77b4"))
    parts.append(poly(released, "#9467bd"))
    for i in action_idx[:20]:
        x = sx(i)
        parts.append(f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{h-pad}" stroke="#ff7f0e" stroke-width="1"/>')
    parts.append(f'<text x="{w-230}" y="20" font-size="12">red=contact, green=held_h, blue=held_r, purple=released</text>')
    parts.append(f'<text x="{w-230}" y="38" font-size="12">orange lines=PROMPT_HUMAN_RELEASE</text>')
    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Week2 baseline and generate report.")
    p.add_argument("--summaries-dir", default="summaries")
    p.add_argument("--schema-spec", default="schema_spec/interaction_schema.json")
    p.add_argument("--out-dir", default="artifacts/week2")
    p.add_argument("--report", default="reports/week2_baseline.md")
    p.add_argument("--sample-frames", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summaries_dir = Path(args.summaries_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Day2-style random 50-frame baseline run.
    sample_out = out_dir / "schema_sample_50.jsonl"
    cmd = [
        "python",
        "src/schema_gen/run_schema_gen.py",
        "--input",
        str(summaries_dir),
        "--output",
        str(sample_out),
        "--schema-spec",
        str(args.schema_spec),
        "--max-retries",
        "2",
        "--sample-frames",
        str(args.sample_frames),
        "--seed",
        "7",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(cmd, check=True, env=env)

    # 2) Full run for report metrics.
    full_out = out_dir / "schema_all.jsonl"
    cmd_full = [
        "python",
        "src/schema_gen/run_schema_gen.py",
        "--input",
        str(summaries_dir),
        "--output",
        str(full_out),
        "--schema-spec",
        str(args.schema_spec),
        "--max-retries",
        "2",
    ]
    subprocess.run(cmd_full, check=True, env=env)

    schema_rows = read_jsonl(full_out)

    # Build aligned summary rows.
    summary_rows: List[Dict] = []
    for f in sorted(summaries_dir.glob("ep_*.jsonl")):
        summary_rows.extend(read_jsonl(f))

    n = min(len(schema_rows), len(summary_rows))
    schema_rows = schema_rows[:n]
    summary_rows = summary_rows[:n]

    first_valid = sum(1 for r in schema_rows if r.get("_metrics", {}).get("first_pass_valid", False))
    final_valid = sum(1 for r in schema_rows if r.get("_metrics", {}).get("final_valid", False))
    repair_mean = sum(float(r.get("_metrics", {}).get("repair_count", 0)) for r in schema_rows) / max(1, n)

    contact_match = state_match_rate(summary_rows, schema_rows, "contact_confirmed")
    release_match = state_match_rate(summary_rows, schema_rows, "human_released")
    secure_match = state_match_rate(summary_rows, schema_rows, "object_secured")
    phase_match = phase_match_rate(summary_rows, schema_rows)

    # Case figures: normal / not_release proxy / challenging
    case_dir = out_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)

    ep_to_rows_sum: Dict[str, List[Dict]] = {}
    ep_to_rows_schema: Dict[str, List[Dict]] = {}
    for s in summary_rows:
        ep_to_rows_sum.setdefault(s.get("episode_id", ""), []).append(s)
    for s in schema_rows:
        ep_to_rows_schema.setdefault(s.get("meta", {}).get("episode_id", ""), []).append(s)

    episodes = sorted(ep_to_rows_sum.keys())[:3]
    case_labels = ["normal", "not_release_proxy", "hard_case"]
    case_paths = []
    for ep, label in zip(episodes, case_labels):
        out_svg = case_dir / f"{label}_{ep}.svg"
        make_case_svg(ep_to_rows_sum[ep], ep_to_rows_schema.get(ep, []), out_svg)
        case_paths.append(out_svg)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Week2 Baseline Report")
    lines.append("")
    lines.append("## Core Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| JSON validity (first pass) | {first_valid}/{n} ({first_valid/max(1,n):.4f}) |")
    lines.append(f"| After-repair validity | {final_valid}/{n} ({final_valid/max(1,n):.4f}) |")
    lines.append(f"| Mean repair count | {repair_mean:.4f} |")
    lines.append(f"| Phase coarse match (weak) | {phase_match:.4f} |")
    lines.append(f"| contact_confirmed match (weak) | {contact_match:.4f} |")
    lines.append(f"| human_released match (weak) | {release_match:.4f} |")
    lines.append(f"| object_secured match (weak) | {secure_match:.4f} |")
    lines.append("")
    lines.append("## Case Figures")
    lines.append("")
    for p in case_paths:
        lines.append(f"- {p.as_posix()}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Weak labels are derived from summaries thresholds.")
    lines.append("- Repair loop uses at most 2 retries and minimal field edits.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
