#!/usr/bin/env python3
import json, csv, numpy as np
from pathlib import Path
p=Path('outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/predictions.jsonl')
rows=[json.loads(l) for l in p.open() if l.strip()]
deltas=np.asarray([float(r['normalized_action_delta']) for r in rows])
controls=np.asarray([float(r['control_normalized_action_delta']) for r in rows if r.get('control_normalized_action_delta') is not None])
schema=np.mean([bool(r.get('qwen_schema_sensitive')) for r in rows])
thresholds=[]
for q in [90,95,99]: thresholds.append((f'control_p{q}',float(np.percentile(controls,q))))
for t in [1.5,2.0,2.2851185083448717,2.5,3.0]: thresholds.append((f'fixed_{t:.3f}',float(t)))
out=[]
for name,t in thresholds:
 action=float(np.mean(deltas>=t)); gap=schema-action
 out.append({'threshold_name':name,'threshold':t,'vlm_schema_sensitivity':schema,'openvla_action_sensitivity':action,'semantic_action_gap':gap,'low_sensitivity_rate':1-action})
dir=Path('outputs/linguistic_blindness/threshold_robustness_exp2'); dir.mkdir(parents=True,exist_ok=True)
with (dir/'threshold_robustness.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
(dir/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print((dir/'threshold_robustness.csv').read_text())
