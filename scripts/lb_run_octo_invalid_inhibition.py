#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
REPO_ROOT=Path(__file__).resolve().parents[1]
for p in [REPO_ROOT/'scripts', REPO_ROOT/'src', REPO_ROOT/'third_party'/'octo']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from lb_run_octo_action_sensitivity import load_octo_model, predict_octo_action, action_std_for_rows, action_stats_dynamic, load_image_for_example
from lb_run_exp2_action_sensitivity import load_index_maps, paraphrase_instruction, threshold_from_controls
from linguistic_blindness.utils.io import read_jsonl, write_jsonl, write_json, write_csv, command_string, utc_timestamp

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmark',default='outputs/linguistic_blindness/exp3_invalid_instruction_clean_600/benchmark.jsonl')
    ap.add_argument('--out-dir',default='outputs/linguistic_blindness/octo_invalid_instruction_inhibition_600')
    ap.add_argument('--octo-checkpoint',default='hf://rail-berkeley/octo-small-1.5')
    ap.add_argument('--index',default='data/processed/unified_index.parquet')
    ap.add_argument('--max-per-perturbation',type=int,default=200)
    ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--image-size',type=int,default=256)
    ap.add_argument('--octo-seed',type=int,default=0)
    ap.add_argument('--threshold-method',choices=['p95','mean_plus_2std','fixed'],default='p95')
    ap.add_argument('--fixed-threshold',type=float,default=1e-6)
    ap.add_argument('--log-every',type=int,default=25)
    return ap.parse_args()

def sample_rows(rows, max_per, seed):
    by=defaultdict(list)
    for r in rows: by[r.get('perturbation_type')].append(r)
    rng=random.Random(seed); out=[]
    for p,g in sorted(by.items()):
        rng.shuffle(g); out.extend(g[:max_per if max_per>0 else len(g)])
    rng.shuffle(out); return out

def agg(rows):
    labels=['overall']+sorted({r['perturbation_type'] for r in rows})
    out=[]
    for lab in labels:
        g=rows if lab=='overall' else [r for r in rows if r['perturbation_type']==lab]
        if not g: continue
        n=len(g)
        out.append({'perturbation_type':lab,'n':n,'octo_action_inhibition_rate':sum(r['octo_action_inhibited'] for r in g)/n,'octo_blind_execution_rate':sum(r['octo_blind_execution'] for r in g)/n,'mean_normalized_delta':sum(r['normalized_action_delta'] for r in g)/n,'median_normalized_delta':float(np.median([r['normalized_action_delta'] for r in g])),'p95_normalized_delta':float(np.percentile([r['normalized_action_delta'] for r in g],95))})
    return out

def main():
    args=parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=sample_rows(read_jsonl(args.benchmark),args.max_per_perturbation,args.seed)
    obs_map,_=load_index_maps(args.index)
    model=load_octo_model(args.octo_checkpoint)
    pred=[]
    for i,ex in enumerate(rows,1):
        img,src=load_image_for_example(ex,obs_map,args.image_size)
        orig=str(ex.get('original_instruction') or '')
        pert=str(ex.get('instruction') or ex.get('counterfactual_instruction') or '')
        a_orig=predict_octo_action(model,img,orig,args.octo_seed)
        a_pert=predict_octo_action(model,img,pert,args.octo_seed)
        a_ctrl=predict_octo_action(model,img,paraphrase_instruction(orig),args.octo_seed)
        pred.append({'example_id':ex.get('example_id'),'perturbation_type':ex.get('perturbation_type'),'original_instruction':orig,'perturbed_instruction':pert,'octo_action_original':a_orig,'octo_action_perturbed':a_pert,'control_action':a_ctrl,'image_source':src})
        if args.log_every and (i%args.log_every==0 or i==len(rows)): print(f'progress {i}/{len(rows)}',flush=True)
    std=action_std_for_rows([{'octo_action_original':r['octo_action_original'],'octo_action_counterfactual':r['octo_action_perturbed'],'control_action':r['control_action']} for r in pred])
    controls=[]
    for r in pred:
        r.update(action_stats_dynamic(r['octo_action_original'],r['octo_action_perturbed'],std))
        c=action_stats_dynamic(r['octo_action_original'],r['control_action'],std)['normalized_action_delta']
        r['control_normalized_action_delta']=c; controls.append(c)
    threshold=threshold_from_controls([{'control_normalized_action_delta':c} for c in controls],args.threshold_method,args.fixed_threshold)
    for r in pred:
        r['action_inhibition_threshold']=threshold
        r['octo_action_inhibited']=bool(r['normalized_action_delta']>=threshold)
        r['octo_blind_execution']=not r['octo_action_inhibited']
    summary=agg(pred)
    write_jsonl(out/'predictions.jsonl',pred); write_csv(out/'octo_invalid_inhibition_summary.csv',summary)
    write_json(out/'summary.json',{'num_examples':len(pred),'threshold':threshold,'rows':summary,'command':command_string(),'timestamp_utc':utc_timestamp()})
    print(json.dumps({'out_dir':str(out),'num_examples':len(pred),'threshold':threshold,'overall':summary[0]},indent=2))
if __name__=='__main__': main()
