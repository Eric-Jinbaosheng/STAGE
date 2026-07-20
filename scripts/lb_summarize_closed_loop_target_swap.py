#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, List


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists(): return []
    with path.open() as f: return list(csv.DictReader(f))

def f(x):
    try: return float(x)
    except Exception: return None

def mean(vals):
    vals=[v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir',default='outputs/linguistic_blindness/supp_closed_loop_target_swap')
    ap.add_argument('dirs',nargs='*',default=[
        'outputs/linguistic_blindness/exp6_schema_projection_5tasks_alpha1_scale2_220',
        'outputs/linguistic_blindness/exp6_schema_projection_until_close_5tasks_d008',
        'outputs/linguistic_blindness/exp6_phase_adapter_task1_ramekin',
        'outputs/linguistic_blindness/exp6_phase_adapter_task3_cookiebox',
        'outputs/linguistic_blindness/exp6_phase_adapter_task5_ramekin',
    ])
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for d in map(Path,args.dirs):
        summary=json.loads((d/'summary.json').read_text()) if (d/'summary.json').exists() else {}
        red=read_csv(d/'redirection_summary.csv')
        if red:
            rows.extend([{**r,'experiment':d.name,'projection_mode':r.get('projection_mode') or summary.get('projection_mode','always_project')} for r in red])
    # Experiment-level summary from paired raw/projection redirection rows.
    exp_rows=[]
    for exp in sorted(set(r['experiment'] for r in rows)):
        rs=[r for r in rows if r['experiment']==exp]
        mode=rs[0].get('projection_mode') if rs else ''
        n=len(rs)
        exp_rows.append({
            'experiment':exp,
            'projection_mode':mode,
            'n':n,
            'raw_final_distance':mean([f(r.get('raw_final_distance_to_target')) for r in rs]),
            'projection_final_distance':mean([f(r.get('projection_final_distance_to_target')) for r in rs]),
            'final_distance_redirection':mean([f(r.get('final_distance_redirection_score')) for r in rs]),
            'raw_min_distance':mean([f(r.get('raw_min_distance_to_target')) for r in rs]),
            'projection_min_distance':mean([f(r.get('projection_min_distance_to_target')) for r in rs]),
            'min_distance_redirection':mean([f(r.get('min_distance_redirection_score')) for r in rs]),
            'release_rate':mean([1.0 if str(r.get('released')).lower()=='true' else 0.0 for r in rs if 'released' in r]),
            'target_lift_rate':mean([1.0 if str(r.get('target_lifted')).lower()=='true' else 0.0 for r in rs if 'target_lifted' in r]),
            'success_rate_raw':mean([1.0 if str(r.get('raw_success')).lower()=='true' else 0.0 for r in rs]),
            'success_rate_projection':mean([1.0 if str(r.get('projection_success')).lower()=='true' else 0.0 for r in rs]),
        })
    keys=list(exp_rows[0].keys()) if exp_rows else []
    with (out/'closed_loop_target_swap_summary.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=keys); w.writeheader(); w.writerows(exp_rows)
    keys2=[]
    for r in rows:
        for k in r:
            if k not in keys2:
                keys2.append(k)
    with (out/'closed_loop_target_swap_cases.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=keys2); w.writeheader(); w.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'experiments':exp_rows,'n_cases':len(rows)},indent=2),encoding='utf-8')
    print(json.dumps({'out_dir':str(out),'n_experiments':len(exp_rows),'n_cases':len(rows)},indent=2))
if __name__=='__main__': main()
