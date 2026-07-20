#!/usr/bin/env python
import argparse, csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def mean(xs):
    xs=[float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None

def rate(xs):
    xs=[x for x in xs if x is not None]
    return float(np.mean([bool(x) for x in xs])) if xs else None

def ci(xs, seed=0, n=1000):
    xs=np.asarray([float(x) for x in xs if x is not None], dtype=float)
    if xs.size==0: return (None,None)
    rng=np.random.default_rng(seed)
    boots=[xs[rng.integers(0, xs.size, xs.size)].mean() for _ in range(n)]
    return float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5))

def auc_distance(trace, key):
    vals=[float(x[key]) for x in trace if key in x]
    return float(np.mean(vals)) if vals else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rollouts', default='outputs/linguistic_blindness/short_horizon_actcheck_predict_k20_100/short_horizon_rollouts.jsonl')
    ap.add_argument('--out-dir', default='outputs/linguistic_blindness/actcheck_predicts_short_horizon_k20_100')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows=read_jsonl(args.rollouts)
    cf=[r for r in rows if r.get('method')=='counterfactual_instruction']
    enriched=[]
    for r in cf:
        rr=dict(r)
        tr=r.get('trace') or []
        rr['integrated_dist_counterfactual']=r.get('integrated_dist_counterfactual', auc_distance(tr,'dist_counterfactual'))
        rr['integrated_dist_original']=r.get('integrated_dist_original', auc_distance(tr,'dist_original'))
        rr['integrated_target_preference']=None if rr['integrated_dist_counterfactual'] is None or rr['integrated_dist_original'] is None else rr['integrated_dist_original']-rr['integrated_dist_counterfactual']
        rr['final_target_preference']=None if r.get('final_dist_counterfactual') is None or r.get('final_dist_original') is None else float(r['final_dist_original'])-float(r['final_dist_counterfactual'])
        enriched.append(rr)
    summary=[]
    for label in ['aligned','wrong-target','ambiguous','unknown','overall']:
        g=enriched if label=='overall' else [r for r in enriched if r.get('first_action_actcheck_label')==label]
        if not g: continue
        metrics={
            'group': label,
            'n': len(g),
            'counterfactual_approach_rate': rate([r.get('counterfactual_approach') for r in g]),
            'original_approach_rate': rate([r.get('original_approach') for r in g]),
            'mean_delta_dist_counterfactual': mean([r.get('delta_dist_counterfactual') for r in g]),
            'mean_delta_dist_original': mean([r.get('delta_dist_original') for r in g]),
            'mean_final_dist_counterfactual': mean([r.get('final_dist_counterfactual') for r in g]),
            'mean_final_dist_original': mean([r.get('final_dist_original') for r in g]),
            'mean_integrated_dist_counterfactual': mean([r.get('integrated_dist_counterfactual') for r in g]),
            'mean_integrated_dist_original': mean([r.get('integrated_dist_original') for r in g]),
            'mean_final_target_preference': mean([r.get('final_target_preference') for r in g]),
            'mean_integrated_target_preference': mean([r.get('integrated_target_preference') for r in g]),
            'mean_first_action_cos_to_counterfactual': mean([r.get('first_action_cos_to_counterfactual_target') for r in g]),
            'mean_first_action_cos_to_original': mean([r.get('first_action_cos_to_original_target') for r in g]),
        }
        for k in ['counterfactual_approach_rate','mean_delta_dist_counterfactual','mean_final_target_preference','mean_integrated_target_preference']:
            vals=[]
            if k=='counterfactual_approach_rate': vals=[float(bool(r.get('counterfactual_approach'))) for r in g if r.get('counterfactual_approach') is not None]
            elif k=='mean_delta_dist_counterfactual': vals=[r.get('delta_dist_counterfactual') for r in g]
            elif k=='mean_final_target_preference': vals=[r.get('final_target_preference') for r in g]
            elif k=='mean_integrated_target_preference': vals=[r.get('integrated_target_preference') for r in g]
            lo,hi=ci(vals, seed=17)
            metrics[k+'_ci95_low']=lo; metrics[k+'_ci95_high']=hi
        summary.append(metrics)
    write_csv(out/'actcheck_rollout_group_summary.csv', summary)
    write_csv(out/'actcheck_rollout_examples.csv', [{k:v for k,v in r.items() if k!='trace'} for r in enriched])
    # Stepwise group curves.
    curves=[]
    for label in ['aligned','wrong-target','ambiguous','overall']:
        g=enriched if label=='overall' else [r for r in enriched if r.get('first_action_actcheck_label')==label]
        if not g: continue
        max_step=max(len(r.get('trace') or []) for r in g)
        for s in range(max_step):
            cf_dist=[]; orig_dist=[]
            for r in g:
                tr=r.get('trace') or []
                if len(tr)>s:
                    cf_dist.append(tr[s]['dist_counterfactual']); orig_dist.append(tr[s]['dist_original'])
            curves.append({'group':label,'step':s,'n':len(cf_dist),'mean_dist_counterfactual':mean(cf_dist),'mean_dist_original':mean(orig_dist),'mean_target_preference': None if mean(cf_dist) is None else mean(orig_dist)-mean(cf_dist)})
    write_csv(out/'actcheck_rollout_distance_curves.csv', curves)
    Path(out/'report.md').write_text('\n'.join(['# ActCheck predicts K=20 rollout behavior','', 'This analysis groups counterfactual-instruction rollouts by the first-step ActCheck label computed before rollout execution.', '', *[str(r) for r in summary]]), encoding='utf-8')
    print(out)

if __name__=='__main__': main()
