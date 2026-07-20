#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path

def read_jsonl(p):
 with open(p) as f: return [json.loads(l) for l in f if l.strip()]
def write_jsonl(p,rows):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--base',default='outputs/linguistic_blindness/visual_relation_counterfactual_600/benchmark.jsonl'); ap.add_argument('--out-dir',default='outputs/linguistic_blindness/compositional_invalid_set_300'); ap.add_argument('--max-examples',type=int,default=300); ap.add_argument('--seed',type=int,default=21)
 args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
 rows=read_jsonl(args.base); random.Random(args.seed).shuffle(rows); rows=rows[:args.max_examples]
 templates=[
  ('prohibit_then_relation','Do not pick up the {orig}; instead move to the object {rel} the {orig}.','HOLD'),
  ('relation_but_block_anchor','Pick up the object {rel} the {orig}, but do not touch the {cf}.','HOLD'),
  ('unless_not_visible','Move toward the object {rel} the {orig} unless the target is not clearly visible.','ASK'),
  ('ambiguous_or','Pick up the {orig} or the object {rel} the {orig}.','ASK'),
  ('distractor_mention','The user mentioned a red mug, but the actual command is to pick up the object {rel} the {orig}.','MOVE_TO'),
  ('conflict_same_target','Pick up the object {rel} the {orig}, but do not move that object.','HOLD'),
 ]
 outrows=[]
 for i,r in enumerate(rows):
  cat,tpl,next_action=templates[i%len(templates)]
  rel=(r.get('observation_state') or {}).get('counterfactual_relation_to_anchor') or 'near'
  orig=r['original_target_object']; cf=r['counterfactual_target_object']
  instr=tpl.format(orig=orig,cf=cf,rel=rel)
  defer=next_action!='MOVE_TO'
  rr=dict(r)
  rr.update({'example_id':r['example_id'].replace('visual_relation_swap',f'compositional_invalid::{cat}'),'source_example_id':r['example_id'],'instruction':instr,'counterfactual_instruction':instr,'perturbation_type':'compositional_invalid','compositional_category':cat,'eval_split':'compositional_invalid','gold_schema':{'target_object':cf if not defer else None,'target_exists':not defer,'phase':'wait' if defer else 'approach','allowed_actions':['ASK','PROMPT','WAIT','HOLD','ABORT','BACKOFF'] if defer else ['MOVE_TO','APPROACH','GRASP','PICK'],'blocked_actions':['MOVE_TO','GRASP','PICK','PLACE','RETRACT','TRANSFER'] if defer else [],'next_action':next_action,'reason':'compositional invalid/prohibited relation instruction' if defer else 'valid command with distractor mention','confidence':1.0}})
  outrows.append(rr)
 write_jsonl(out/'benchmark.jsonl',outrows)
 (out/'benchmark_statistics.json').write_text(json.dumps({'num_examples':len(outrows),'templates':[x[0] for x in templates],'note':'Programmatic compositional invalid/ambiguous split; not human-written.'},indent=2))
 print(json.dumps({'out_dir':str(out),'num_examples':len(outrows)},indent=2))
if __name__=='__main__': main()
