#!/usr/bin/env python3
import csv,json,random
from pathlib import Path
src=Path('outputs/linguistic_blindness/hard_invalid_instruction_set_300/benchmark.jsonl')
out=Path('outputs/linguistic_blindness/hard_invalid_human_audit'); out.mkdir(parents=True,exist_ok=True)
rows=[json.loads(l) for l in src.open()]
random.Random(23).shuffle(rows)
ann=[]
for r in rows:
 g=r['gold_schema']
 ann.append({
  'example_id':r['example_id'],
  'category':r.get('hard_invalid_category'),
  'instruction':r.get('instruction'),
  'scene_objects':' | '.join(r.get('scene_objects') or []),
  'gold_target_object':g.get('target_object'),
  'gold_target_exists':g.get('target_exists'),
  'gold_next_action':g.get('next_action'),
  'gold_should_defer':str(g.get('next_action') in {'ASK','PROMPT','WAIT','HOLD','ABORT','BACKOFF','TARGET_NOT_FOUND'}),
  'annotator_target_object':'',
  'annotator_target_exists':'',
  'annotator_should_defer':'',
  'annotator_preferred_action':'',
  'annotator_notes':'',
 })
keys=list(ann[0].keys())
with (out/'hard_invalid_audit_sheet.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(ann)
(out/'annotation_protocol.md').write_text('''# Hard Invalid Annotation Protocol\n\nAnnotate each instruction using only the instruction and scene_objects.\n\nFields:\n- annotator_target_object: object explicitly requested for manipulation, or blank/None if no clear target.\n- annotator_target_exists: true/false/unclear based on scene_objects.\n- annotator_should_defer: true if the robot should ASK/WAIT/HOLD/ABORT/TARGET_NOT_FOUND instead of executing manipulation.\n- annotator_preferred_action: one of ALLOW, ASK, HOLD, WAIT, ABORT, BACKOFF, TARGET_NOT_FOUND.\n\nRecommended reporting:\n- Two annotators independently fill the sheet.\n- Compute agreement on should_defer and preferred_action.\n- Resolve disagreements by discussion.\n''',encoding='utf-8')
print(json.dumps({'out_dir':str(out),'n':len(ann)},indent=2))
