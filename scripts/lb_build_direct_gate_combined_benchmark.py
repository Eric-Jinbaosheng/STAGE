#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

def read(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]
def write(p, rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=True,sort_keys=True)+'\n')
parts=[
 ('template_invalid','outputs/linguistic_blindness/exp3_invalid_instruction_clean_600/benchmark.jsonl'),
 ('normal','outputs/linguistic_blindness/exp4_normal_instruction_clean_600/benchmark.jsonl'),
 ('hard_invalid','outputs/linguistic_blindness/hard_invalid_instruction_set_300/benchmark.jsonl'),
]
rows=[]
for split,path in parts:
    for r in read(path):
        rr=dict(r); rr['eval_split']=split; rr['example_id']=f"{split}::{rr.get('example_id')}"; rows.append(rr)
out='outputs/linguistic_blindness/direct_qwen_gate_combined_2700/benchmark.jsonl'
write(out,rows)
stats={'num_examples':len(rows),'splits':{s:sum(1 for r in rows if r['eval_split']==s) for s,_ in parts},'sources':dict(parts)}
Path(out).with_name('benchmark_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
print(json.dumps({'out':out,**stats},indent=2))
