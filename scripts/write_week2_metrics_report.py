import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write a small Week2 metrics markdown report from batch eval JSON.")
    p.add_argument("--eval-json", default="artifacts/batch_eval/result.json")
    p.add_argument("--out", default="reports/week2_metrics.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    cfg = payload.get("config", {})
    qual_cases = payload.get("qual_cases", [])

    lines = [
        "# Week2 Metrics",
        "",
        "## Main Table",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| target_acc | {metrics.get('target_object_accuracy', 0.0):.6f} |",
        f"| phase_acc | {metrics.get('phase_accuracy', 0.0):.6f} |",
        "",
        "## Language Sensitivity",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| SSS_target | {metrics.get('schema_sensitivity_score_target', 0.0):.6f} |",
        f"| TFR | {metrics.get('target_flip_rate_swap', 0.0):.6f} |",
        "",
        "## Eval Config",
        "",
        f"- checkpoint: `{cfg.get('checkpoint', '')}`",
        f"- dataset_filter: `{cfg.get('dataset_filter', '')}`",
        f"- split: `{cfg.get('split', '')}`",
        f"- evaluated_samples: `{cfg.get('evaluated_samples', 0)}`",
        "",
        "## Qual Cases",
        "",
        f"- stored in eval JSON: `{len(qual_cases)}` cases",
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
