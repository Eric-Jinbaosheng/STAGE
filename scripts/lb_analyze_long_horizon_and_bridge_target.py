#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, re, random, sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT/'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/'src'))
from linguistic_blindness.utils.io import read_jsonl, write_json, write_csv, write_jsonl, utc_timestamp, command_string


def mean(xs):
    vals=[float(x) for x in xs if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return None if not vals else sum(vals)/len(vals)

def rate(xs):
    vals=[x for x in xs if x is not None]
    return None if not vals else sum(bool(x) for x in vals)/len(vals)

def quantile(xs,q):
    vals=sorted(float(x) for x in xs if x is not None)
    if not vals: return None
    i=(len(vals)-1)*q; lo=math.floor(i); hi=math.ceil(i)
    if lo==hi: return vals[lo]
    return vals[lo]*(hi-i)+vals[hi]*(i-lo)

def extract_primary_target(instr: str) -> str:
    s = re.sub(r'[^a-zA-Z0-9 ]+', ' ', instr.lower()).strip()
    s = re.sub(r'\s+', ' ', s)
    # Bridge commands often look like: put small spoon from basket to tray.
    for pat in [
        r'^(?:put|place|pick up|move|take|get|grab) (.+?) (?:from|to|in|into|on|onto|out of|off|near|next to) ',
        r'^(?:open|close) (.+)$',
        r'^(?:put|place|pick up|move|take|get|grab) (.+)$',
    ]:
        m=re.search(pat,s)
        if m:
            return m.group(1).strip()
    toks=s.split()
    return ' '.join(toks[:3]) if toks else ''

def analyze_short(short_dir: Path, out: Path):
    pairs=read_jsonl(short_dir/'short_horizon_pairs.jsonl')
    rollouts=read_jsonl(short_dir/'short_horizon_rollouts.jsonl')
    cf=[r for r in rollouts if r.get('method')=='counterfactual_instruction']
    orig=[r for r in rollouts if r.get('method')=='original_instruction']
    def wrong_pref(r):
        # target_preference_score = delta_cf_target - delta_orig_target. <=0 means not more approach toward CF target than original target.
        v=r.get('target_preference_score')
        return None if v is None else float(v) <= 0
    rows=[]
    for method,g in [('original_instruction',orig),('counterfactual_instruction',cf)]:
        rows.append({
            'analysis':'short_horizon_k10',
            'method':method,
            'n':len(g),
            'counterfactual_approach_rate':rate(r.get('counterfactual_approach') for r in g),
            'original_approach_rate':rate(r.get('original_approach') for r in g),
            'wrong_target_preference_rate':rate(wrong_pref(r) for r in g),
            'mean_delta_cf_target':mean(r.get('delta_dist_counterfactual') for r in g),
            'mean_delta_orig_target':mean(r.get('delta_dist_original') for r in g),
            'mean_target_preference_score':mean(r.get('target_preference_score') for r in g),
            'median_target_preference_score':quantile([r.get('target_preference_score') for r in g],0.5),
        })
    pair_rows=[{
        'analysis':'paired_cf_vs_orig_rollout',
        'method':'counterfactual_minus_original',
        'n':len(pairs),
        'mean_cf_minus_orig_counterfactual_approach':mean(p.get('cf_minus_orig_counterfactual_approach') for p in pairs),
        'cf_improves_counterfactual_approach_rate':rate((p.get('cf_minus_orig_counterfactual_approach') or 0)>0 for p in pairs),
        'cf_wrong_target_preference_rate':rate((p.get('cf_target_preference_score') is not None and p.get('cf_target_preference_score') <= 0) for p in pairs),
        'orig_wrong_target_preference_rate':rate((p.get('orig_target_preference_score') is not None and p.get('orig_target_preference_score') <= 0) for p in pairs),
    }]
    write_csv(out/'long_horizon_k10_summary.csv', rows+pair_rows)
    write_json(out/'long_horizon_k10_summary.json', {'rows':rows,'paired':pair_rows,'source':str(short_dir),'timestamp_utc':utc_timestamp()})
    return rows+pair_rows

def build_bridge_target_change(bridge_dir: Path, out: Path, max_examples: int, seed: int):
    normals=read_jsonl(bridge_dir/'normal_benchmark.jsonl')
    by_instr=[]
    seen=set()
    for r in normals:
        instr=str(r.get('instruction') or r.get('original_instruction') or '')
        tgt=extract_primary_target(instr)
        if instr and tgt and instr not in seen:
            by_instr.append({'instruction':instr,'target':tgt})
            seen.add(instr)
    rng=random.Random(seed)
    rows=[]
    for r in normals:
        orig=str(r.get('instruction') or r.get('original_instruction') or '')
        orig_t=extract_primary_target(orig)
        cands=[c for c in by_instr if c['instruction'] != orig and c['target'] and c['target'] != orig_t]
        if not cands: continue
        cf=rng.choice(cands)
        row={**r}
        row['example_id']=str(r.get('example_id')).replace('_normal','_bridge_target_change')
        row['eval_split']='bridge_target_change'
        row['perturbation_type']='target_change'
        row['original_instruction']=orig
        row['instruction']=cf['instruction']
        row['counterfactual_instruction']=cf['instruction']
        row['original_target_object']=orig_t
        row['counterfactual_target_object']=cf['target']
        row['target_change_diagnostic']='instruction_target_changed_from_another_bridge_task_same_shard'
        row['gold_schema']={
            'target_object': cf['target'],
            'target_exists': None,
            'phase': 'UNK',
            'allowed_actions': ['MOVE_TO','GRASP','PICK','PLACE','TRANSFER'],
            'blocked_actions': [],
            'next_action': 'MOVE_TO',
            'reason': 'Instruction-level target/task changed; Bridge shard lacks object metadata, so target existence is not asserted.',
            'confidence': None,
        }
        rows.append(row)
    rng.shuffle(rows)
    if max_examples>0:
        rows=rows[:max_examples]
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out/'benchmark.jsonl', rows)
    write_json(out/'benchmark_stats.json', {
        'dataset':'bridge_v2',
        'diagnostic':'target_change_from_distinct_bridge_instruction',
        'source_normal_examples':len(normals),
        'unique_instructions':len(by_instr),
        'examples':len(rows),
        'caveat':'Counterfactual instruction is drawn from another Bridge task in the same public shard. This tests real-robot action sensitivity to target/task language changes, but does not assert object presence because local Bridge TFDS lacks object metadata.',
        'timestamp_utc':utc_timestamp(),
        'command':command_string(),
        'target_pair_counts': Counter(f"{r['original_target_object']} -> {r['counterfactual_target_object']}" for r in rows),
    })
    write_csv(out/'target_pairs.csv', [{'target_pair':k,'n':v} for k,v in Counter(f"{r['original_target_object']} -> {r['counterfactual_target_object']}" for r in rows).most_common()])
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--short-dir', default='outputs/linguistic_blindness/short_horizon_target_approach_50')
    ap.add_argument('--bridge-dir', default='outputs/linguistic_blindness/bridge_v2_invalid_456')
    ap.add_argument('--out-dir', default='outputs/linguistic_blindness/long_horizon_and_bridge_target_supplement')
    ap.add_argument('--bridge-target-out', default='outputs/linguistic_blindness/bridge_v2_target_change_165')
    ap.add_argument('--max-bridge-target-examples', type=int, default=165)
    ap.add_argument('--seed', type=int, default=23)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows=analyze_short(Path(args.short_dir), out)
    bridge_rows=build_bridge_target_change(Path(args.bridge_dir), Path(args.bridge_target_out), args.max_bridge_target_examples, args.seed)
    write_json(out/'summary.json', {'long_horizon_rows':rows,'bridge_target_benchmark':args.bridge_target_out,'bridge_target_examples':len(bridge_rows),'timestamp_utc':utc_timestamp(),'command':command_string()})
    print(json.dumps({'out_dir':str(out),'bridge_target_out':args.bridge_target_out,'bridge_target_examples':len(bridge_rows),'long_horizon_rows':rows}, indent=2))

if __name__=='__main__': main()
