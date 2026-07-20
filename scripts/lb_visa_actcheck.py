#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

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
def write_jsonl(p, rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
def vec3(x): return np.asarray(x,dtype=float).reshape(-1)[:3]
def unit(v):
    n=np.linalg.norm(v)
    return None if n<1e-12 else v/n
def cos(a,b):
    ua,ub=unit(a),unit(b)
    if ua is None or ub is None: return None
    return float(np.dot(ua,ub))
def mean(xs):
    vals=[float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return None if not vals else sum(vals)/len(vals)
def rate(xs):
    vals=[x for x in xs if x is not None]
    return None if not vals else sum(bool(x) for x in vals)/len(vals)

def check_row(r, margin):
    ee=vec3(r['ee_pos']); po=vec3(r['original_target_pos']); pc=vec3(r['counterfactual_target_pos'])
    ao=vec3(r['openvla_action_original']); ac=vec3(r['openvla_action_counterfactual'])
    vo=po-ee; vc=pc-ee
    orig_to_orig=cos(ao,vo); orig_to_cf=cos(ao,vc)
    cf_to_cf=cos(ac,vc); cf_to_orig=cos(ac,vo)
    # Directional alignment: action points more toward intended target than the swapped/wrong target.
    orig_aligned = orig_to_orig is not None and orig_to_cf is not None and orig_to_orig > orig_to_cf + margin
    cf_aligned = cf_to_cf is not None and cf_to_orig is not None and cf_to_cf > cf_to_orig + margin
    wrong_target = cf_to_orig is not None and cf_to_cf is not None and cf_to_orig >= cf_to_cf + margin
    ambiguous = cf_to_orig is not None and cf_to_cf is not None and abs(cf_to_cf-cf_to_orig) <= margin
    return {
        'example_id':r['example_id'],
        'target_pair':f"{r.get('original_target_object')} -> {r.get('counterfactual_target_object')}",
        'original_target_object':r.get('original_target_object'),
        'counterfactual_target_object':r.get('counterfactual_target_object'),
        'target_distance':r.get('target_distance'),
        'target_angle_deg':r.get('target_angle_deg'),
        'orig_action_cos_to_orig_target':orig_to_orig,
        'orig_action_cos_to_cf_target':orig_to_cf,
        'cf_action_cos_to_cf_target':cf_to_cf,
        'cf_action_cos_to_orig_target':cf_to_orig,
        'orig_target_aligned_action':orig_aligned,
        'cf_target_aligned_action':cf_aligned,
        'cf_wrong_target_aligned_action':wrong_target,
        'cf_ambiguous_target_alignment':ambiguous,
        'actcheck_blocks_cf_action':wrong_target or ambiguous,
        'semantic_action_consistency_score': None if cf_to_cf is None or cf_to_orig is None else cf_to_cf-cf_to_orig,
        'openvla_action_sensitive':r.get('openvla_action_sensitive'),
        'normalized_action_delta':r.get('normalized_action_delta'),
    }

def summarize(rows,label):
    return {
        'subset':label,
        'n':len(rows),
        'target_aligned_action_rate':rate([r['cf_target_aligned_action'] for r in rows]),
        'wrong_target_action_rate':rate([r['cf_wrong_target_aligned_action'] for r in rows]),
        'ambiguous_alignment_rate':rate([r['cf_ambiguous_target_alignment'] for r in rows]),
        'actcheck_block_rate':rate([r['actcheck_blocks_cf_action'] for r in rows]),
        'valid_original_action_preservation':rate([r['orig_target_aligned_action'] for r in rows]),
        'mean_semantic_action_consistency_score':mean([r['semantic_action_consistency_score'] for r in rows]),
        'openvla_action_sensitivity':rate([r['openvla_action_sensitive'] for r in rows]),
        'mean_normalized_action_delta':mean([r['normalized_action_delta'] for r in rows]),
        'mean_target_distance':mean([r['target_distance'] for r in rows]),
        'mean_target_angle_deg':mean([r['target_angle_deg'] for r in rows]),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--predictions',default='outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/analysis/predictions_with_positions.jsonl')
    ap.add_argument('--out-dir',default='outputs/linguistic_blindness/visa_actcheck_exp2')
    ap.add_argument('--alignment-margin',type=float,default=0.05)
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    src=[r for r in read_jsonl(args.predictions) if r.get('position_extraction_status')=='ok']
    rows=[check_row(r,args.alignment_margin) for r in src]
    summ=[summarize(rows,'overall')]
    for pair in sorted({r['target_pair'] for r in rows}): summ.append(summarize([r for r in rows if r['target_pair']==pair],pair))
    for name,cond in [('spatial_angle_ge_30',lambda r:(r.get('target_angle_deg') or 0)>=30),('spatial_angle_ge_45',lambda r:(r.get('target_angle_deg') or 0)>=45)]:
        g=[r for r in rows if cond(r)]
        if g: summ.append(summarize(g,name))
    write_jsonl(out/'actcheck_predictions.jsonl',rows)
    write_csv(out/'actcheck_summary.csv',summ)
    md=['# VISA-ActCheck Summary','','| Subset | N | Target-aligned ↑ | Wrong-target ↓ | Ambiguous ↓ | Block/flag ↑ | Original preserve ↑ | Consistency score ↑ | OpenVLA sens ↑ |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summ:
        md.append(f"| {s['subset']} | {s['n']} | {s['target_aligned_action_rate']:.3f} | {s['wrong_target_action_rate']:.3f} | {s['ambiguous_alignment_rate']:.3f} | {s['actcheck_block_rate']:.3f} | {s['valid_original_action_preservation']:.3f} | {s['mean_semantic_action_consistency_score']:.3f} | {s['openvla_action_sensitivity']:.3f} |")
    (out/'actcheck_report.md').write_text('\n'.join(md),encoding='utf-8')
    (out/'run_metadata.json').write_text(json.dumps({'predictions':args.predictions,'alignment_margin':args.alignment_margin,'num_examples':len(rows)},indent=2),encoding='utf-8')
    print(json.dumps({'out_dir':str(out),'num_examples':len(rows),'overall':summ[0]},indent=2))
if __name__=='__main__': main()
