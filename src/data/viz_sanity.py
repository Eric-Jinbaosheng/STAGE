import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _to_svg(x: List[float], ys: Dict[str, List[float]], out_path: Path, title: str) -> None:
    w, h, pad = 980, 440, 40
    y_min = min(min(v) for v in ys.values()) if ys else 0.0
    y_max = max(max(v) for v in ys.values()) if ys else 1.0
    if y_max - y_min < 1e-8:
        y_max = y_min + 1.0

    def sx(i: int) -> float:
        return pad + (w - 2 * pad) * (i / max(1, len(x) - 1))

    def sy(v: float) -> float:
        return h - pad - (h - 2 * pad) * ((v - y_min) / (y_max - y_min))

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    parts.append(f'<text x="{pad}" y="20" font-size="16">{title}</text>')
    parts.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#444"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#444"/>')
    legend_y = 24
    for ci, (name, vals) in enumerate(ys.items()):
        c = colors[ci % len(colors)]
        points = " ".join(f"{sx(i):.1f},{sy(float(v)):.1f}" for i, v in enumerate(vals))
        parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{points}"/>')
        parts.append(f'<rect x="{w-250}" y="{legend_y-10}" width="12" height="12" fill="{c}"/>')
        parts.append(f'<text x="{w-232}" y="{legend_y}" font-size="12">{name}</text>')
        legend_y += 18
    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week1 sanity visualization and report.")
    p.add_argument("--libero-index", default="data/processed/libero_index.parquet")
    p.add_argument("--handover-index", default="data/processed/handover_index.parquet")
    p.add_argument("--schema-labels", default="data/processed/schema_labels.parquet")
    p.add_argument("--report", default="reports/week1_sanity.md")
    p.add_argument("--out-dir", default="reports/week1_assets")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = ["# Week1 Sanity Report", ""]

    libero_exists = Path(args.libero_index).exists()
    hand_exists = Path(args.handover_index).exists()
    labels_exists = Path(args.schema_labels).exists()
    labels = pd.read_parquet(args.schema_labels) if labels_exists else pd.DataFrame()

    if libero_exists:
        li = pd.read_parquet(args.libero_index)
        lines.append(f"- LIBERO rows: {len(li)}")
        if len(li) > 0:
            eps = li["episode_id"].drop_duplicates().tolist()[:20]
            lines.append(f"- LIBERO sampled episodes: {len(eps)}")
            for ep in eps[:3]:
                sub = li[li["episode_id"] == ep]
                inst = str(sub.iloc[0].get("instruction", ""))
                lines.append(f"  - `{ep}` instruction: `{inst}`")
                if len(labels) > 0:
                    lsub = labels[(labels["dataset"] == "libero") & (labels["episode_id"] == ep)]
                    if len(lsub) > 0:
                        x = lsub["frame_id"].astype(float).tolist()
                        phase_num = [hash(p) % 7 for p in lsub["phase"].tolist()]
                        path = out_dir / f"libero_phase_{ep}.svg"
                        _to_svg(x, {"phase_hash": phase_num}, path, f"LIBERO phase proxy: {ep}")
                        lines.append(f"  - phase plot: `{path.as_posix()}`")
        else:
            lines.append("- LIBERO index is empty (no raw files were found).")
    else:
        lines.append("- LIBERO index missing.")

    if hand_exists:
        hi = pd.read_parquet(args.handover_index)
        lines.append("")
        lines.append(f"- Handover rows: {len(hi)}")
        episodes = hi["episode_id"].drop_duplicates().tolist()
        random.seed(7)
        sampled = episodes[:10] if len(episodes) <= 10 else random.sample(episodes, 10)
        lines.append(f"- Handover sampled episodes: {len(sampled)}")
        for ep in sampled[:3]:
            sub = hi[hi["episode_id"] == ep].sort_values("frame_id")
            x = sub["frame_id"].astype(float).tolist()
            ys = {
                "d_grip_obj": sub["d_grip_obj"].astype(float).tolist(),
                "contact_flag": sub["contact_flag"].astype(float).tolist(),
                "gripper_width": sub["gripper_width"].astype(float).tolist(),
            }
            p = out_dir / f"handover_curve_{ep}.svg"
            _to_svg(x, ys, p, f"Handover sanity: {ep}")
            lines.append(f"  - `{ep}` curve: `{p.as_posix()}`")
    else:
        lines.append("")
        lines.append("- Handover index missing.")

    if labels_exists:
        lines.append("")
        lines.append(f"- Schema labels rows: {len(labels)}")
        if len(labels) > 0:
            lines.append(f"- target_object vocab in labels: {sorted(labels['target_object'].drop_duplicates().tolist())}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

