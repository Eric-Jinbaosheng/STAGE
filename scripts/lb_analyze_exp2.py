#!/usr/bin/env python3
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    import csv
    keys = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fbool(x: Any) -> float:
    return 1.0 if bool(x) else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    vals = np.asarray([float(x) for x in values if x is not None and not math.isnan(float(x))], dtype=float)
    if vals.size == 0:
        return float('nan')
    return float(np.percentile(vals, q))


def mean(values: Iterable[float]) -> float:
    vals = np.asarray([float(x) for x in values if x is not None and not math.isnan(float(x))], dtype=float)
    if vals.size == 0:
        return float('nan')
    return float(np.mean(vals))


def std(values: Iterable[float]) -> float:
    vals = np.asarray([float(x) for x in values if x is not None and not math.isnan(float(x))], dtype=float)
    if vals.size == 0:
        return float('nan')
    return float(np.std(vals))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    vlm = mean(fbool(r.get('qwen_schema_sensitive')) for r in rows)
    action = mean(fbool(r.get('openvla_action_sensitive')) for r in rows)
    low = mean(fbool(r.get('low_action_sensitivity')) for r in rows)
    deltas = [r.get('normalized_action_delta') for r in rows]
    control = [r.get('control_normalized_action_delta') for r in rows]
    return {
        'n': n,
        'vlm_schema_sensitivity': vlm,
        'openvla_action_sensitivity': action,
        'semantic_action_gap': vlm - action if not math.isnan(vlm) and not math.isnan(action) else float('nan'),
        'low_sensitivity_rate': low,
        'mean_normalized_delta': mean(deltas),
        'std_normalized_delta': std(deltas),
        'p25_normalized_delta': percentile(deltas, 25),
        'median_normalized_delta': percentile(deltas, 50),
        'p75_normalized_delta': percentile(deltas, 75),
        'p90_normalized_delta': percentile(deltas, 90),
        'p95_normalized_delta': percentile(deltas, 95),
        'mean_control_normalized_delta': mean(control),
        'mean_delta_pos': mean(r.get('delta_pos') for r in rows),
        'mean_delta_rot': mean(r.get('delta_rot') for r in rows),
        'mean_delta_gripper': mean(r.get('delta_gripper') for r in rows),
        'mean_action_cosine': mean(r.get('action_cosine') for r in rows),
    }


def distribution_summary(rows: List[Dict[str, Any]], column: str, name: str, threshold: float | None) -> Dict[str, Any]:
    vals = [r.get(column) for r in rows if r.get(column) is not None]
    out = {
        'distribution': name,
        'n': len(vals),
        'mean': mean(vals),
        'std': std(vals),
        'p25': percentile(vals, 25),
        'median': percentile(vals, 50),
        'p75': percentile(vals, 75),
        'p90': percentile(vals, 90),
        'p95': percentile(vals, 95),
        'min': percentile(vals, 0),
        'max': percentile(vals, 100),
        'threshold': threshold if threshold is not None else '',
    }
    if threshold is not None and vals:
        out['fraction_above_threshold'] = float(np.mean(np.asarray(vals, dtype=float) >= threshold))
    else:
        out['fraction_above_threshold'] = ''
    return out


def latex_table(rows: List[Dict[str, Any]], columns: List[str], caption: str = '') -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if math.isnan(v):
                return '--'
            return f'{v:.3f}'
        return str(v)
    lines = ['\\begin{tabular}{' + 'l' + 'r' * (len(columns) - 1) + '}']
    lines.append(' & '.join(columns).replace('_', ' ') + ' \\\\')
    lines.append('\\hline')
    for row in rows:
        lines.append(' & '.join(fmt(row.get(c, '')) for c in columns) + ' \\\\')
    lines.append('\\end{tabular}')
    if caption:
        lines.append(f'% {caption}')
    return '\n'.join(lines) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser(description='Analyze Exp2 OpenVLA/Qwen semantic-action gap results.')
    ap.add_argument('--predictions', default='outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/predictions.jsonl')
    ap.add_argument('--out-dir', default='')
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    out_dir = Path(args.out_dir) if args.out_dir else pred_path.parent / 'analysis'
    rows = read_jsonl(pred_path)
    for r in rows:
        r['target_pair'] = f"{r.get('original_target_object')} -> {r.get('counterfactual_target_object')}"

    threshold_values = [r.get('action_sensitivity_threshold') for r in rows if r.get('action_sensitivity_threshold') is not None]
    threshold = float(threshold_values[0]) if threshold_values else None

    pair_rows = []
    for pair in sorted({r['target_pair'] for r in rows}):
        group = [r for r in rows if r['target_pair'] == pair]
        row = {'target_pair': pair, **summarize(group)}
        pair_rows.append(row)
    pair_rows.append({'target_pair': 'Overall', **summarize(rows)})

    dist_rows = [distribution_summary(rows, 'normalized_action_delta', 'target_swap', threshold)]
    if any(r.get('control_normalized_action_delta') is not None for r in rows):
        dist_rows.append(distribution_summary(rows, 'control_normalized_action_delta', 'paraphrase_control', threshold))

    non_drawer = [r for r in rows if 'drawer' not in str(r.get('target_pair', '')).lower()]
    drawer = [r for r in rows if 'drawer' in str(r.get('target_pair', '')).lower()]
    proxy_rows = [
        {'subset': 'all', 'rule': 'all target swaps', **summarize(rows)},
        {'subset': 'non_drawer_object_swaps', 'rule': 'black bowl -> ramekin/cookie box; proxy for visually distinct object swaps without geometry labels', **summarize(non_drawer)},
        {'subset': 'drawer_swaps', 'rule': 'middle drawer -> top drawer; likely shared early approach motion', **summarize(drawer)},
    ]

    has_spatial = all(
        any(r.get(col) is not None for r in rows)
        for col in ['ee_pos', 'original_target_pos', 'counterfactual_target_pos', 'target_distance', 'target_angle_deg']
    )
    spatial_rows = []
    if has_spatial:
        dist_vals = [float(r['target_distance']) for r in rows if r.get('target_distance') is not None]
        angle_vals = [float(r['target_angle_deg']) for r in rows if r.get('target_angle_deg') is not None]
        dist_thr = percentile(dist_vals, 50)
        angle_thr = percentile(angle_vals, 50)
        spatial_defs = [
            ('all', 'all target swaps', rows),
            ('distance_ge_median', f'target_distance >= median ({dist_thr:.4f})', [r for r in rows if r.get('target_distance') is not None and float(r['target_distance']) >= dist_thr]),
            ('angle_ge_median', f'target_angle_deg >= median ({angle_thr:.2f})', [r for r in rows if r.get('target_angle_deg') is not None and float(r['target_angle_deg']) >= angle_thr]),
            ('distance_and_angle_ge_median', f'distance >= median and angle >= median', [r for r in rows if r.get('target_distance') is not None and r.get('target_angle_deg') is not None and float(r['target_distance']) >= dist_thr and float(r['target_angle_deg']) >= angle_thr]),
            ('angle_ge_30', 'target_angle_deg >= 30', [r for r in rows if r.get('target_angle_deg') is not None and float(r['target_angle_deg']) >= 30.0]),
            ('angle_ge_45', 'target_angle_deg >= 45', [r for r in rows if r.get('target_angle_deg') is not None and float(r['target_angle_deg']) >= 45.0]),
        ]
        for subset, rule, group in spatial_defs:
            if group:
                spatial_rows.append({
                    'subset': subset,
                    'rule': rule,
                    'distance_threshold': dist_thr if 'distance' in subset or subset == 'all' else '',
                    'angle_threshold': angle_thr if 'angle' in subset or subset == 'all' else '',
                    'mean_target_distance': mean(r.get('target_distance') for r in group),
                    'median_target_distance': percentile([r.get('target_distance') for r in group], 50),
                    'mean_target_angle_deg': mean(r.get('target_angle_deg') for r in group),
                    'median_target_angle_deg': percentile([r.get('target_angle_deg') for r in group], 50),
                    **summarize(group),
                })
        spatial_status = {
            'status': 'available',
            'reason': 'MuJoCo object/site positions are present in predictions.',
            'distance_median': dist_thr,
            'angle_median_deg': angle_thr,
            'required_fields_for_true_spatial_subset': ['ee_pos', 'original_target_pos', 'counterfactual_target_pos'],
        }
    else:
        spatial_status = {
            'status': 'unavailable',
            'reason': 'predictions do not contain ee_pos/object_positions/bounding boxes, so no true geometric spatial subset can be computed.',
            'available_proxy': 'non_drawer_object_swaps vs drawer_swaps',
            'required_fields_for_true_spatial_subset': ['ee_pos', 'original_target_pos', 'counterfactual_target_pos'],
        }

    write_csv(out_dir / 'target_pair_breakdown.csv', pair_rows)
    write_csv(out_dir / 'delta_distribution_summary.csv', dist_rows)
    write_csv(out_dir / 'spatial_proxy_subset_summary.csv', proxy_rows)
    if spatial_rows:
        write_csv(out_dir / 'spatial_subset_summary.csv', spatial_rows)
    write_json(out_dir / 'spatial_subset_status.json', spatial_status)
    write_json(out_dir / 'analysis_summary.json', {
        'predictions': str(pred_path),
        'num_examples': len(rows),
        'threshold': threshold,
        'target_pairs': Counter(r['target_pair'] for r in rows),
        'overall': summarize(rows),
        'spatial_subset_status': spatial_status,
    })

    pair_cols = ['target_pair', 'n', 'vlm_schema_sensitivity', 'openvla_action_sensitivity', 'semantic_action_gap', 'low_sensitivity_rate', 'mean_normalized_delta', 'median_normalized_delta', 'p75_normalized_delta', 'p95_normalized_delta']
    dist_cols = ['distribution', 'n', 'mean', 'std', 'p25', 'median', 'p75', 'p90', 'p95', 'threshold', 'fraction_above_threshold']
    proxy_cols = ['subset', 'n', 'vlm_schema_sensitivity', 'openvla_action_sensitivity', 'semantic_action_gap', 'low_sensitivity_rate', 'mean_normalized_delta', 'median_normalized_delta']
    spatial_cols = ['subset', 'n', 'mean_target_distance', 'median_target_distance', 'mean_target_angle_deg', 'median_target_angle_deg', 'vlm_schema_sensitivity', 'openvla_action_sensitivity', 'semantic_action_gap', 'low_sensitivity_rate']
    (out_dir / 'target_pair_breakdown.tex').write_text(latex_table(pair_rows, pair_cols), encoding='utf-8')
    (out_dir / 'delta_distribution_summary.tex').write_text(latex_table(dist_rows, dist_cols), encoding='utf-8')
    (out_dir / 'spatial_proxy_subset_summary.tex').write_text(latex_table(proxy_rows, proxy_cols), encoding='utf-8')
    if spatial_rows:
        (out_dir / 'spatial_subset_summary.tex').write_text(latex_table(spatial_rows, spatial_cols), encoding='utf-8')

    print(json.dumps({'out_dir': str(out_dir), 'num_examples': len(rows), 'threshold': threshold}, indent=2))


if __name__ == '__main__':
    main()
