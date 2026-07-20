#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
REPO_ROOT=Path(__file__).resolve().parents[1]
for p in [REPO_ROOT/'third_party'/'LIBERO_pkg',REPO_ROOT/'third_party'/'openvla',REPO_ROOT/'scripts',REPO_ROOT/'src']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from lb_run_exp6_schema_action_projection import counterfactual_for_task, site_pos, eef_pos, load_model_and_processor_with_diagnostics
from lb_run_exp6_sim_gate_sanity import inject_local_dataset_statistics
from lb_run_exp2_action_sensitivity import predict_action
from lb_run_libero10_smoke import dummy_action,get_image,get_libero_env,normalize_for_env
from linguistic_blindness.utils.io import write_json,write_jsonl,write_csv,command_string,utc_timestamp

def parse_args():
 ap=argparse.ArgumentParser()
 ap.add_argument('--task-suite-name',default='libero_spatial'); ap.add_argument('--task-ids',default='1,3,4,5,6'); ap.add_argument('--num-trials-per-task',type=int,default=2); ap.add_argument('--horizon',type=int,default=10); ap.add_argument('--num-steps-wait',type=int,default=10); ap.add_argument('--model-path',default='/scratch/bj2410/models/openvla-7b-finetuned-libero-90'); ap.add_argument('--unnorm-key',default='libero_90_no_noops'); ap.add_argument('--out-dir',default='outputs/linguistic_blindness/short_horizon_target_approach_50'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--bf16',action='store_true'); ap.add_argument('--fp16',action='store_true'); ap.add_argument('--attn-implementation',default=''); ap.add_argument('--trust-remote-code',action='store_true'); ap.add_argument('--center-crop',action='store_true')
 return ap.parse_args()

def rollout(env,init_state,instruction,orig_target,cf_target,model,processor,args):
    env.reset(); obs=env.set_init_state(init_state)
    distances=[]; trace=[]; done=False; err=None
    first_action_actcheck={}
    try:
        for t in range(args.horizon+args.num_steps_wait):
            po,_,so=site_pos(env,orig_target); pc,_,sc=site_pos(env,cf_target); ee=eef_pos(obs)
            if t>=args.num_steps_wait and po is not None and pc is not None and ee is not None:
                distances.append({'step':t-args.num_steps_wait,'dist_original':float(np.linalg.norm(ee-po)),'dist_counterfactual':float(np.linalg.norm(ee-pc))})
            if t<args.num_steps_wait: action=dummy_action(); raw=None
            else:
                img=get_image(obs,center_crop=args.center_crop); raw=predict_action(model,processor,img,instruction,args); action=normalize_for_env(raw)
                if not first_action_actcheck:
                    po,_,_=site_pos(env,orig_target); pc,_,_=site_pos(env,cf_target); ee=eef_pos(obs)
                    def cos_to(p):
                        if p is None or ee is None or raw is None: return None
                        v=np.asarray(p)-np.asarray(ee); a=np.asarray(raw[:3],dtype=float)
                        den=np.linalg.norm(v)*np.linalg.norm(a)
                        return None if den<1e-9 else float(np.dot(v,a)/den)
                    ccf=cos_to(pc); corig=cos_to(po)
                    aligned = ccf is not None and corig is not None and ccf >= corig + 0.05
                    wrong = ccf is not None and corig is not None and corig >= ccf + 0.05
                    amb = ccf is not None and corig is not None and not aligned and not wrong
                    first_action_actcheck={
                        'first_action_cos_to_counterfactual_target':ccf,
                        'first_action_cos_to_original_target':corig,
                        'first_action_target_aligned':aligned,
                        'first_action_wrong_target':wrong,
                        'first_action_ambiguous':amb,
                        'first_action_actcheck_label':'aligned' if aligned else ('wrong-target' if wrong else ('ambiguous' if amb else 'unknown')),
                    }
            obs,reward,done,info=env.step(action)
            if done: break
    except Exception as e: err=f'{type(e).__name__}: {e}'
    def first_last(key):
        if not distances: return (None,None,None)
        start=distances[0][key]; end=distances[-1][key]; return start,end,start-end
    so,eo,do=first_last('dist_original'); sc,ec,dc=first_last('dist_counterfactual')
    pref=None if do is None or dc is None else dc-do
    if distances:
        integrated_cf=float(np.mean([d['dist_counterfactual'] for d in distances]))
        integrated_orig=float(np.mean([d['dist_original'] for d in distances]))
    else:
        integrated_cf=None; integrated_orig=None
    return {'instruction':instruction,'done':bool(done),'error':err,'start_dist_original':so,'final_dist_original':eo,'delta_dist_original':do,'start_dist_counterfactual':sc,'final_dist_counterfactual':ec,'delta_dist_counterfactual':dc,'integrated_dist_counterfactual':integrated_cf,'integrated_dist_original':integrated_orig,'target_preference_score':pref,'counterfactual_approach': None if dc is None else dc>0,'original_approach': None if do is None else do>0,'trace':distances,**first_action_actcheck}

def mean(vals):
 vals=[float(v) for v in vals if v is not None and not math.isnan(float(v))]
 return None if not vals else sum(vals)/len(vals)
def rate(vals):
 vals=[v for v in vals if v is not None]
 return None if not vals else sum(bool(v) for v in vals)/len(vals)

def main():
 args=parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
 from libero.libero import benchmark
 model,processor=load_model_and_processor_with_diagnostics(args); inject_local_dataset_statistics(model,args.model_path,args.unnorm_key)
 suite=benchmark.get_benchmark_dict()[args.task_suite_name](); task_ids=[int(x) for x in args.task_ids.split(',') if x.strip()]
 rows=[]; pairs=[]
 for tid in task_ids:
  task=suite.get_task(tid); cf=counterfactual_for_task(tid,task.language)
  if not cf: continue
  env,desc=get_libero_env(task,resolution=256); init_states=suite.get_task_init_states(tid)
  for trial in range(min(args.num_trials_per_task,len(init_states))):
   init=init_states[trial]
   orig=rollout(env,init,cf['original_instruction'],cf['original_target_object'],cf['counterfactual_target_object'],model,processor,args)
   cfr=rollout(env,init,cf['counterfactual_instruction'],cf['original_target_object'],cf['counterfactual_target_object'],model,processor,args)
   base={'task_id':tid,'trial_idx':trial,'task_description':desc,'original_instruction':cf['original_instruction'],'counterfactual_instruction':cf['counterfactual_instruction'],'original_target_object':cf['original_target_object'],'counterfactual_target_object':cf['counterfactual_target_object']}
   rows += [{**base,'method':'original_instruction',**orig},{**base,'method':'counterfactual_instruction',**cfr}]
   pairs.append({**base,'orig_delta_cf_target':orig['delta_dist_counterfactual'],'cf_delta_cf_target':cfr['delta_dist_counterfactual'],'orig_delta_orig_target':orig['delta_dist_original'],'cf_delta_orig_target':cfr['delta_dist_original'],'cf_minus_orig_counterfactual_approach':None if orig['delta_dist_counterfactual'] is None or cfr['delta_dist_counterfactual'] is None else cfr['delta_dist_counterfactual']-orig['delta_dist_counterfactual'],'cf_target_preference_score':cfr['target_preference_score'],'orig_target_preference_score':orig['target_preference_score']})
   print(json.dumps(pairs[-1]),flush=True)
  env.close()
 summary=[]
 for method in ['original_instruction','counterfactual_instruction']:
  g=[r for r in rows if r['method']==method]
  summary.append({'method':method,'n':len(g),'mean_delta_dist_counterfactual':mean([r['delta_dist_counterfactual'] for r in g]),'mean_delta_dist_original':mean([r['delta_dist_original'] for r in g]),'counterfactual_approach_rate':rate([r['counterfactual_approach'] for r in g]),'original_approach_rate':rate([r['original_approach'] for r in g]),'mean_target_preference_score':mean([r['target_preference_score'] for r in g])})
 pair_summary={'n':len(pairs),'mean_cf_minus_orig_counterfactual_approach':mean([p['cf_minus_orig_counterfactual_approach'] for p in pairs]),'mean_cf_target_preference_score':mean([p['cf_target_preference_score'] for p in pairs]),'mean_orig_target_preference_score':mean([p['orig_target_preference_score'] for p in pairs])}
 write_jsonl(out/'short_horizon_rollouts.jsonl',rows); write_jsonl(out/'short_horizon_pairs.jsonl',pairs); write_csv(out/'short_horizon_method_summary.csv',summary); write_json(out/'summary.json',{'timestamp_utc':utc_timestamp(),'command':command_string(),'method_summary':summary,'pair_summary':pair_summary,'horizon':args.horizon})
 print(json.dumps({'out_dir':str(out),'summary':summary,'pair_summary':pair_summary},indent=2),flush=True)
if __name__=='__main__': main()
