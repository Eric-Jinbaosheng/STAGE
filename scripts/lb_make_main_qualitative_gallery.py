#!/usr/bin/env python
import argparse
import csv
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_font(size=22, bold=False):
    paths = [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def wrap(text, width):
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/linguistic_blindness/qualitative_ci_packet/scene_images/manifest.csv")
    ap.add_argument("--out-dir", default="outputs/linguistic_blindness/main_qualitative_gallery")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.manifest)))[:6]
    W, H = 620, 520
    pad = 24
    title_font = load_font(23, True)
    small_font = load_font(17)
    mono_font = load_font(16)
    label_font = load_font(19, True)
    panels = []
    for i, r in enumerate(rows, 1):
        img_path = Path(r["annotated_image_file"] or r["image_file"])
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((W - 2*pad, 255))
        panel = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(panel)
        # colored top stripe by failure label
        color = (190, 55, 45) if r["actcheck_label"] == "wrong-target" else (210, 140, 35)
        draw.rectangle([0, 0, W, 12], fill=color)
        label = f"({chr(96+i)}) {r['policy']} | {r['actcheck_label']}"
        draw.text((pad, 22), label, font=title_font, fill=(25,25,25))
        x = (W - img.width)//2
        panel.paste(img, (x, 62))
        y = 72 + img.height
        lines = [
            f"Orig: {r['original_instruction']}",
            f"CF: {r['counterfactual_instruction']}",
            f"Schema target: {r['qwen_pred_target']} | intended: {r['counterfactual_target']}",
            f"Action: aligned={r['target_aligned']} wrong={r['wrong_target']} ambiguous={r['ambiguous']}",
            f"cos(target)={float(r['cf_action_cos_to_cf_target']):.2f}, cos(orig)={float(r['cf_action_cos_to_orig_target']):.2f}, Δ={float(r['normalized_action_delta']):.2f}",
        ]
        for j, line in enumerate(lines):
            fnt = label_font if j == 2 else small_font
            draw.multiline_text((pad, y), wrap(line, 58), font=fnt, fill=(20,20,20), spacing=2)
            y += 42 if j != 2 else 48
        panels.append(panel)
    sheet = Image.new("RGB", (3*W, 2*H), (245,245,242))
    for i, p in enumerate(panels):
        sheet.paste(p, ((i % 3)*W, (i // 3)*H))
    sheet.save(out / "figure_main_actcheck_gallery.png")
    # Also write a concise markdown caption for paper use.
    md = ["# Main qualitative ActCheck gallery", "", "Generated figure: `figure_main_actcheck_gallery.png`", "", "Each panel shows a case where Qwen/VISA recovers the counterfactual schema target, but the native VLA action is wrong-target or ambiguous under ActCheck.", ""]
    for i, r in enumerate(rows, 1):
        md.append(f"{i}. `{r['policy']}` `{r['actcheck_label']}`: original target `{r['original_target']}`, counterfactual/schema target `{r['counterfactual_target']}`, action-sensitive={r['action_sensitive']}, normalized delta={float(r['normalized_action_delta']):.3f}.")
    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(out / "figure_main_actcheck_gallery.png")

if __name__ == "__main__":
    main()
