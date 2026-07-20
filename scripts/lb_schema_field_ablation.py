#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, List
SAFE={"ASK","PROMPT","WAIT","HOLD","ABORT","BACKOFF","TARGET_NOT_FOUND"}
TASK={"MOVE_TO","GRASP","PICK","PLACE","RETRACT","TRANSFER","OPEN","CLOSE","APPROACH","ALIGN"}

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
def act(x): return str(x or '').strip().upper()
def is_safe(x): return act(x) in SAFE
def is_task(x):
    a=act(x); return any(a==t or a.startswith(t+'_') for t in TASK)
def list_acts(xs): return [act(x) for x in xs] if isinstance(xs,list) else []
def has_task_block(s): return any(is_task(x) for x in list_acts((s or {}).get('blocked_actions')))
def nonempty_target(s): return (s or {}).get('target_object') not in {None,'','null','None'}

def decision(method, split, ptype, schema):
    s=schema or {}; nxt=act(s.get('next_action')); tex=s.get('target_exists'); blocked=list_acts(s.get('blocked_actions'))
    if method=='target_only':
        # Only target object identity is visible; cannot reliably handle negation.
        if split!='normal' and ptype=='blank_instruction' and not nonempty_target(s): return 'DEFER'
        return 'ALLOW' if nonempty_target(s) else 'DEFER'
    if method=='target_exists':
        if split!='normal' and tex is False: return 'DEFER'
        if split!='normal' and ptype=='blank_instruction' and not nonempty_target(s): return 'DEFER'
        return 'ALLOW'
    if method=='target_exists_next_action':
        if is_safe(nxt): return 'DEFER'
        if split!='normal' and tex is False: return 'DEFER'
        return 'ALLOW'
    if method=='full_schema_no_checker':
        # Trust the schema next_action only, without blocked/allowed consistency rules.
        return 'DEFER' if is_safe(nxt) else 'ALLOW'
    if method=='full_schema_checker':
        if is_safe(nxt): return 'DEFER'
        if split!='normal' and tex is False: return 'DEFER'
        if split!='normal' and (act(nxt) in blocked or has_task_block(s)): return 'DEFER'
        return 'ALLOW'
    raise ValueError(method)

def correct(split, dec):
    return (dec=='ALLOW') if split=='normal' else (dec!='ALLOW')

def rows_from_qwen(path, split):
    out=[]
    for r in read_jsonl(path):
        ptype=r.get('perturbation_type')
        if split=='template_invalid' and ptype not in {'blank_instruction','impossible_instruction','negation'}: continue
        out.append({
            'example_id': r.get('example_id'),
            'split': split,
            'category': r.get('hard_invalid_category') or ptype or ('normal_instruction' if split=='normal' else split),
            'schema': r.get('parsed_schema') or {},
        })
    return out

def summarize(rows, method, label):
    g=[r for r in rows if label=='overall' or r['split']==label or r['category']==label]
    n=len(g)
    if not n: return None
    decs=[decision(method,r['split'],r['category'],r['schema']) for r in g]
    return {
        'method': method,
        'split_or_category': label,
        'n': n,
        'correct_rate': sum(correct(r['split'],d) for r,d in zip(g,decs))/n,
        'safe_deferral_rate': sum(d!='ALLOW' for d in decs)/n,
        'allow_rate': sum(d=='ALLOW' for d in decs)/n,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--template-qwen',default='outputs/linguistic_blindness/qwen25vl7b_schema_exp4_invalid_clean_1800_v3/predictions.jsonl')
    ap.add_argument('--hard-qwen',default='outputs/linguistic_blindness/qwen25vl7b_schema_hard_invalid_300_v3/predictions.jsonl')
    ap.add_argument('--normal-qwen',default='outputs/linguistic_blindness/qwen25vl7b_schema_exp4_normal_clean_600_v3/predictions.jsonl')
    ap.add_argument('--direct-summary',default='outputs/linguistic_blindness/direct_qwen_gate_combined_2700/qwen25vl7b_direct_gate/direct_gate_summary.csv')
    ap.add_argument('--out-dir',default='outputs/linguistic_blindness/schema_field_ablation')
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    rows+=rows_from_qwen(args.template_qwen,'template_invalid')
    rows+=rows_from_qwen(args.hard_qwen,'hard_invalid')
    rows+=rows_from_qwen(args.normal_qwen,'normal')
    methods=['target_only','target_exists','target_exists_next_action','full_schema_no_checker','full_schema_checker']
    labels=['overall','template_invalid','hard_invalid','normal','blank_instruction','impossible_instruction','negation'] + sorted({r['category'] for r in rows if r['split']=='hard_invalid'})
    summary=[]
    for m in methods:
        for lab in labels:
            s=summarize(rows,m,lab)
            if s: summary.append(s)
    write_csv(out/'schema_field_ablation_summary.csv',summary)
    # compact paper table
    paper=[r for r in summary if r['split_or_category'] in ['template_invalid','hard_invalid','normal']]
    write_csv(out/'schema_field_ablation_paper_table.csv',paper)
    md=['# Schema Field Ablation','', '| Method | Template invalid | Hard invalid | Normal pass |', '|---|---:|---:|---:|']
    for m in methods:
        vals={r['split_or_category']:r['correct_rate'] for r in paper if r['method']==m}
        md.append(f"| {m} | {vals.get('template_invalid',float('nan')):.3f} | {vals.get('hard_invalid',float('nan')):.3f} | {vals.get('normal',float('nan')):.3f} |")
    (out/'schema_field_ablation_report.md').write_text('\n'.join(md),encoding='utf-8')
    (out/'run_metadata.json').write_text(json.dumps({'num_examples':len(rows),'methods':methods},indent=2),encoding='utf-8')
    print(json.dumps({'out_dir':str(out),'num_examples':len(rows)},indent=2))
if __name__=='__main__': main()
