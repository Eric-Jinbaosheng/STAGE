#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

def read_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]
def write_csv(p, rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows: p.write_text(''); return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def mean(xs):
    vals=[float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return None if not vals else sum(vals)/len(vals)
def rate(xs):
    vals=[x for x in xs if x is not None]
    return None if not vals else sum(bool(x) for x in vals)/len(vals)
def summarize(rows,label):
    return {'subset':label,'n':len(rows),'oracle_schema_sensitivity':rate([r.get('qwen_schema_sensitive') for r in rows]),'openvla_action_sensitivity':rate([r.get('openvla_action_sensitive') for r in rows]),'semantic_action_gap':(rate([r.get('qwen_schema_sensitive') for r in rows]) or 0)-(rate([r.get('openvla_action_sensitive') for r in rows]) or 0),'low_sensitivity_rate':rate([r.get('low_action_sensitivity') for r in rows]),'mean_normalized_delta':mean([r.get('normalized_action_delta') for r in rows]),'mean_control_delta':mean([r.get('control_normalized_action_delta') for r in rows]),'mean_target_angle_deg':mean([r.get('target_angle_deg') for r in rows]),'mean_target_distance':mean([r.get('target_distance') for r in rows])}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmark',default='outputs/linguistic_blindness/visual_relation_counterfactual_600/benchmark.jsonl')
    ap.add_argument('--predictions',default='outputs/linguistic_blindness/visual_relation_openvla_action_sensitivity_600/predictions.jsonl')
    ap.add_argument('--out-dir',default='outputs/linguistic_blindness/visual_relation_openvla_action_sensitivity_600/analysis')
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    bench={r['example_id']:r for r in read_jsonl(args.benchmark)}
    rows=[]
    for r in read_jsonl(args.predictions):
        b=bench.get(r['example_id'],{})
        st=b.get('observation_state') or {}
        r.update({'target_distance':b.get('target_distance'),'target_angle_deg':b.get('target_angle_deg'),'visual_relation_type':b.get('visual_relation_type'),'relation_original':st.get('original_relation_to_anchor'),'relation_counterfactual':st.get('counterfactual_relation_to_anchor'),'target_name_in_instruction':str(r.get('counterfactual_target_object','')).lower() in str(r.get('counterfactual_instruction','')).lower(),'anchor_name_in_instruction':str(r.get('original_target_object','')).lower() in str(r.get('counterfactual_instruction','')).lower()})
        rows.append(r)
    summ=[summarize(rows,'overall')]
    for pair in sorted({f"{r.get('original_target_object')} -> {r.get('counterfactual_target_object')}" for r in rows}):
        summ.append(summarize([r for r in rows if f"{r.get('original_target_object')} -> {r.get('counterfactual_target_object')}"==pair],pair))
    for lab,cond in [('angle_ge_30',lambda r:(r.get('target_angle_deg') or 0)>=30),('angle_ge_45',lambda r:(r.get('target_angle_deg') or 0)>=45)]:
        g=[r for r in rows if cond(r)]
        if g: summ.append(summarize(g,lab))
    write_csv(out/'visual_relation_summary.csv',summ)
    write_csv(out/'visual_relation_predictions_augmented.csv',rows)
    md=['# Visual Relation Counterfactual Action Sensitivity','','| Subset | N | Oracle schema sens ↑ | OpenVLA action sens ↑ | SAG ↑ | Low sens ↓ | Mean delta |','|---|---:|---:|---:|---:|---:|---:|']
    for s in summ:
        md.append(f"| {s['subset']} | {s['n']} | {s['oracle_schema_sensitivity']:.3f} | {s['openvla_action_sensitivity']:.3f} | {s['semantic_action_gap']:.3f} | {s['low_sensitivity_rate']:.3f} | {s['mean_normalized_delta']:.3f} |")
    (out/'visual_relation_report.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps({'out_dir':str(out),'overall':summ[0]},indent=2))
if __name__=='__main__': main()
