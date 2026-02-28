import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Week1 sanity-check visualizations.")
    parser.add_argument("--summaries-dir", required=True, help="Directory with per-episode summaries JSONL")
    parser.add_argument("--out-dir", required=True, help="Visualization output directory")
    parser.add_argument("--num-episodes", type=int, default=3, help="How many episodes to visualize")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_keyframes(rows: List[Dict]) -> Dict[str, int]:
    idx_reach = 0
    idx_contact = 0
    idx_transfer = 0
    idx_release = 0

    max_app = -1.0
    max_contact = -1.0
    max_robot = -1.0
    max_release = -1.0
    for i, r in enumerate(rows):
        rel = r.get("relations", {})
        app = float(rel.get("approaching_gripper_object", 0.0))
        ct = float(rel.get("contact_confirmed", 0.0))
        rb = float(rel.get("held_by_robot", 0.0))
        hr = float(rel.get("human_released", 0.0))
        if app > max_app:
            max_app = app
            idx_reach = i
        if ct > max_contact:
            max_contact = ct
            idx_contact = i
        if rb > max_robot:
            max_robot = rb
            idx_transfer = i
        if hr > max_release:
            max_release = hr
            idx_release = i

    return {
        "reach": idx_reach,
        "contact": idx_contact,
        "transfer": idx_transfer,
        "release": idx_release,
    }


def make_plot(rows: List[Dict], out_png: Path) -> None:
    t = [float(r.get("timestamp", i)) for i, r in enumerate(rows)]
    d_grip = [float(r.get("signals", {}).get("d_grip_obj", 0.0)) for r in rows]
    contact = [float(r.get("relations", {}).get("contact_confirmed", 0.0)) for r in rows]
    held_h = [float(r.get("relations", {}).get("held_by_human", 0.0)) for r in rows]
    held_r = [float(r.get("relations", {}).get("held_by_robot", 0.0)) for r in rows]
    released = [float(r.get("relations", {}).get("human_released", 0.0)) for r in rows]
    try:
        import matplotlib.pyplot as plt  # type: ignore
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, d_grip, label="d_grip_obj")
        ax.plot(t, contact, label="contact_confirmed")
        ax.plot(t, held_h, label="held_by_human")
        ax.plot(t, held_r, label="held_by_robot")
        ax.plot(t, released, label="human_released")
        ax.set_xlabel("t")
        ax.set_ylabel("score / distance")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        return
    except Exception:
        pass

    out_svg = out_png.with_suffix(".svg")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    series = {
        "d_grip_obj": d_grip,
        "contact_confirmed": contact,
        "held_by_human": held_h,
        "held_by_robot": held_r,
        "human_released": released,
    }
    colors = {
        "d_grip_obj": "#1f77b4",
        "contact_confirmed": "#d62728",
        "held_by_human": "#2ca02c",
        "held_by_robot": "#ff7f0e",
        "human_released": "#9467bd",
    }
    w, h, pad = 980, 460, 40
    y_min = min(min(v) for v in series.values())
    y_max = max(max(v) for v in series.values())
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0

    def sx(i: int) -> float:
        return pad + (w - 2 * pad) * (i / max(1, len(t) - 1))

    def sy(v: float) -> float:
        return h - pad - (h - 2 * pad) * ((v - y_min) / (y_max - y_min))

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">']
    parts.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>')
    parts.append(f'<line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#444"/>')
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h-pad}" stroke="#444"/>')
    y_mid = (y_min + y_max) / 2.0
    parts.append(f'<text x="8" y="{pad}" font-size="12">{y_max:.3f}</text>')
    parts.append(f'<text x="8" y="{h/2:.0f}" font-size="12">{y_mid:.3f}</text>')
    parts.append(f'<text x="8" y="{h-pad}" font-size="12">{y_min:.3f}</text>')

    legend_y = 16
    for name, vals in series.items():
        points = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
        parts.append(
            f'<polyline fill="none" stroke="{colors[name]}" stroke-width="2" points="{points}"/>'
        )
        parts.append(f'<rect x="{w-240}" y="{legend_y-9}" width="12" height="12" fill="{colors[name]}"/>')
        parts.append(f'<text x="{w-222}" y="{legend_y+1}" font-size="12">{name}</text>')
        legend_y += 18
    parts.append("</svg>")
    out_svg.write_text("\n".join(parts), encoding="utf-8")


def maybe_copy_keyframes(rows: List[Dict], keyframe_idx: Dict[str, int], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for phase, idx in keyframe_idx.items():
        row = rows[idx]
        image_path = row.get("meta", {}).get("rgb_path")
        if image_path and Path(image_path).exists():
            src = Path(image_path)
            dst = out_dir / f"{phase}_{src.name}"
            shutil.copy2(src, dst)
            mapping[phase] = {"index": idx, "timestamp": row.get("timestamp", 0.0), "image": str(dst)}
        else:
            mapping[phase] = {"index": idx, "timestamp": row.get("timestamp", 0.0), "image": ""}

    (out_dir / "keyframes.json").write_text(json.dumps(mapping, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summaries_dir = Path(args.summaries_dir)
    out_dir = Path(args.out_dir)
    files = [p for p in sorted(summaries_dir.rglob("*.jsonl")) if p.name != "summaries.jsonl"]
    if not files:
        raise ValueError(f"No episode summary files under: {summaries_dir}")

    chosen = files[: max(1, args.num_episodes)]
    for file in chosen:
        rows = read_jsonl(file)
        ep_id = file.stem
        make_plot(rows, out_dir / "curves" / f"{ep_id}.png")
        kf = select_keyframes(rows)
        maybe_copy_keyframes(rows, kf, out_dir / "keyframes" / ep_id)
        print(f"Viz done: {ep_id}")


if __name__ == "__main__":
    main()
