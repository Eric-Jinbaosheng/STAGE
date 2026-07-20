#!/usr/bin/env python3
from __future__ import annotations
import json, csv
from pathlib import Path
DIRS=[
 ('full','outputs/linguistic_blindness/exp6_openvla_hidden_state_probe_600'),
 ('language_only','outputs/linguistic_blindness/exp6_openvla_hidden_state_probe_600_language_only'),
 ('mask_target','outputs/linguistic_blindness/exp6_openvla_hidden_state_probe_600_mask_target'),
 ('image_only','outputs/linguistic_blindness/exp6_openvla_hidden_state_probe_600_image_only'),
]
out=Path('outputs/linguistic_blindness/supp_hidden_state_probe_ablations'); out.mkdir(parents=True,exist_ok=True)
rows=[]
for name,d in DIRS:
    m=json.loads((Path(d)/'metrics.json').read_text())
    rows.append({
        'condition':name,
        'num_test_examples':m['num_test_examples'],
        'target_accuracy':m['hidden_probe_target_accuracy']['mean'],
        'target_accuracy_ci95':m['hidden_probe_target_accuracy'].get('ci95'),
        'schema_sensitivity':m['hidden_probe_schema_sensitivity']['mean'],
        'schema_sensitivity_ci95':m['hidden_probe_schema_sensitivity'].get('ci95'),
        'majority_baseline_accuracy':m['majority_baseline_accuracy']['mean'],
        'openvla_action_sensitivity':m['openvla_action_sensitivity']['mean'],
        'hidden_probe_action_gap':m['hidden_probe_action_gap']['mean'],
    })
with (out/'hidden_probe_ablation_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
(out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print((out/'hidden_probe_ablation_summary.csv').read_text())
