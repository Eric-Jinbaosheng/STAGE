#!/usr/bin/env python
import csv,json
from pathlib import Path

def write_csv(path, rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def read_metrics(d):
    p=Path(d)/'metrics.json'
    if not p.exists(): return None
    m=json.loads(p.read_text())
    return {
        'condition': m.get('feature_condition'),
        'n': m['num_examples'],
        'test_examples': m['num_test_examples'],
        'target_accuracy': m['hidden_probe_target_accuracy']['mean'],
        'target_accuracy_ci95': m['hidden_probe_target_accuracy'].get('ci95'),
        'schema_sensitivity': m['hidden_probe_schema_sensitivity']['mean'],
        'schema_sensitivity_ci95': m['hidden_probe_schema_sensitivity'].get('ci95'),
        'action_sensitivity_test_split': m['openvla_action_sensitivity']['mean'],
        'hidden_action_gap': m['hidden_probe_action_gap']['mean'],
        'majority_baseline': m['majority_baseline_accuracy']['mean'],
        'random_labels': m.get('random_labels', False),
        'source_dir': str(d),
    }

def main():
    out=Path('outputs/linguistic_blindness/pixel_relation_hidden_state_probe')
    specs=[
        ('full image + relation instruction','outputs/linguistic_blindness/hprobe_openvla_pixel_relation_full_600'),
        ('language-only / no image','outputs/linguistic_blindness/hprobe_openvla_pixel_relation_language_only_600'),
        ('shuffled-image','outputs/linguistic_blindness/hprobe_openvla_pixel_relation_shuffled_image_600'),
        ('anchor-masked','outputs/linguistic_blindness/hprobe_openvla_pixel_relation_anchor_masked_600'),
        ('relation-masked','outputs/linguistic_blindness/hprobe_openvla_pixel_relation_relation_masked_600'),
        ('random-label','outputs/linguistic_blindness/hprobe_openvla_pixel_relation_random_label_600'),
    ]
    rows=[]
    for label,d in specs:
        r=read_metrics(d)
        if r:
            r['display_condition']=label; rows.append(r)
    write_csv(out/'pixel_relation_hidden_probe_summary.csv', rows)
    lines=['# Pixel-relation OpenVLA hidden-state probe','', 'This uses 600 pixel-grounded relation examples where the instruction does not name the target directly (e.g., object to the right of the black bowl).', '', '| Condition | Target Acc | Schema Sens. | Action Sens. | Gap |', '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['display_condition']} | {float(r['target_accuracy']):.3f} | {float(r['schema_sensitivity']):.3f} | {float(r['action_sensitivity_test_split']):.3f} | {float(r['hidden_action_gap']):.3f} |")
    lines += ['', 'Interpretation: full image+relation recovers the relation-defined target perfectly on the held-out test split, while language-only and shuffled-image controls drop substantially. Native OpenVLA action sensitivity on the same held-out split remains low, supporting a grounded semantic-to-action bottleneck rather than only lexical target-word extraction.']
    out.mkdir(parents=True,exist_ok=True)
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(out)
if __name__=='__main__': main()
