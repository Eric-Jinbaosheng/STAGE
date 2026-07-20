#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def read_jsonl(p):
 with open(p) as f: return [json.loads(l) for l in f if l.strip()]
def write_csv(p,rows):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 if not rows: p.write_text(''); return
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys: keys.append(k)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def rate(xs):
 vals=[x for x in xs if x is not None]
 return None if not vals else sum(bool(x) for x in vals)/len(vals)
def mean(xs):
 vals=[float(x) for x in xs if x is not None]
 return None if not vals else sum(vals)/len(vals)
def summarize(rows,label):
 return {'subset':label,'n':len(rows),'grounding_accuracy':rate([r['target_correct'] for r in rows]),'schema_sensitivity':rate([r['target_correct'] for r in rows]),'openvla_action_sensitivity':rate([r['openvla_action_sensitive'] for r in rows]),'semantic_action_gap_grounding_conditioned':(rate([r['target_correct'] for r in rows]) or 0)-(rate([r['openvla_action_sensitive'] for r in rows]) or 0),'low_sensitivity_rate':rate([r['low_action_sensitivity'] for r in rows]),'mean_delta':mean([r['normalized_action_delta'] for r in rows]),'mean_target_angle_deg':mean([r.get('target_angle_deg') for r in rows])}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--qwen',default='outputs/linguistic_blindness/qwen25vl7b_visual_relation_schema_600/predictions.jsonl'); ap.add_argument('--action',default='outputs/linguistic_blindness/visual_relation_openvla_action_sensitivity_600/analysis/visual_relation_predictions_augmented.csv'); ap.add_argument('--out-dir',default='outputs/linguistic_blindness/pixel_grounded_relation_analysis')
 args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
 q={r['example_id']:r for r in read_jsonl(args.qwen)}
 import csv
 rows=[]
 with open(args.action,newline='') as f:
  for a in csv.DictReader(f):
   r=q[a['example_id']]
   def b(x): return str(x).lower()=='true'
   rows.append({'example_id':a['example_id'],'target_correct':bool(r['target_correct']),'parse_success':bool(r['parse_success']),'pred_target':(r.get('parsed_schema') or {}).get('target_object'),'gold_target':r.get('gold_target_object'),'openvla_action_sensitive':b(a['openvla_action_sensitive']),'low_action_sensitivity':b(a['low_action_sensitivity']),'normalized_action_delta':float(a['normalized_action_delta']),'target_angle_deg':float(a['target_angle_deg']),'target_pair':f"{a['original_target_object']} -> {a['counterfactual_target_object']}"})
 summ=[summarize(rows,'all'),summarize([r for r in rows if r['target_correct']],'grounding_correct_only'),summarize([r for r in rows if r['target_angle_deg']>=45],'hard_angle_ge_45'),summarize([r for r in rows if r['target_correct'] and r['target_angle_deg']>=45],'grounding_correct_and_angle_ge_45')]
 for pair in sorted({r['target_pair'] for r in rows}): summ.append(summarize([r for r in rows if r['target_pair']==pair],pair))
 write_csv(out/'pixel_grounded_relation_summary.csv',summ); write_csv(out/'pixel_grounded_relation_examples.csv',rows)
 md=['# Pixel-Grounded Relation Analysis','','| Subset | N | Grounding acc ↑ | Action sens ↑ | Gap ↑ | Low sens ↓ |','|---|---:|---:|---:|---:|---:|']
 for s in summ: md.append(f"| {s['subset']} | {s['n']} | {s['grounding_accuracy']:.3f} | {s['openvla_action_sensitivity']:.3f} | {s['semantic_action_gap_grounding_conditioned']:.3f} | {s['low_sensitivity_rate']:.3f} |")
 (out/'pixel_grounded_relation_report.md').write_text('\n'.join(md))
 print(json.dumps({'out_dir':str(out),'overall':summ[0],'grounding_correct':summ[1]},indent=2))
if __name__=='__main__': main()
