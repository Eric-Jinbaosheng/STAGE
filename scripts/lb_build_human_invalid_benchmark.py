#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
SAFE={"ASK","PROMPT","WAIT","HOLD","ABORT","BACKOFF","TARGET_NOT_FOUND"}

def read_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]

def write_jsonl(p, rows):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=True, sort_keys=True)+'\n')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sheet', default='outputs/linguistic_blindness/human_written_invalid_set_packet/human_invalid_annotation_sheet.csv')
    ap.add_argument('--base', default='outputs/linguistic_blindness/exp4_normal_instruction_clean_600/benchmark.jsonl')
    ap.add_argument('--out-dir', default='outputs/linguistic_blindness/human_written_invalid_set')
    ap.add_argument('--require-approved', action='store_true')
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    base={r['example_id']:r for r in read_jsonl(args.base)}
    rows=[]; skipped=[]
    with open(args.sheet, newline='') as f:
        for r in csv.DictReader(f):
            instr=(r.get('human_instruction') or '').strip()
            gold=(r.get('gold_next_action') or r.get('gold_decision') or '').strip().upper()
            if not instr or not gold:
                skipped.append((r.get('human_example_id'),'missing_instruction_or_gold')); continue
            if args.require_approved and (r.get('review_status') or '').strip().lower()!='approved':
                skipped.append((r.get('human_example_id'),'not_approved')); continue
            b=base.get(r.get('source_example_id'))
            if not b:
                skipped.append((r.get('human_example_id'),'missing_base')); continue
            rr=dict(b)
            rr.update({
                'example_id': r.get('human_example_id'),
                'source_example_id': r.get('source_example_id'),
                'instruction': instr,
                'counterfactual_instruction': instr,
                'perturbation_type': 'human_written_invalid',
                'human_invalid_category': r.get('assigned_category'),
                'eval_split': 'human_invalid',
                'human_written': True,
                'annotator_id': r.get('annotator_id'),
                'reviewer_id': r.get('reviewer_id'),
                'review_status': r.get('review_status'),
                'gold_decision': (r.get('gold_decision') or '').strip().upper(),
                'gold_schema': {
                    'target_object': None,
                    'target_exists': False,
                    'phase': 'wait',
                    'allowed_actions': sorted(SAFE),
                    'blocked_actions': ['MOVE_TO','GRASP','PICK','PLACE','RETRACT','TRANSFER'],
                    'next_action': gold if gold in SAFE else 'ASK',
                    'reason': 'human-written invalid/prohibited/ambiguous instruction',
                    'confidence': 1.0,
                },
            })
            rows.append(rr)
    write_jsonl(out/'benchmark.jsonl', rows)
    (out/'build_report.json').write_text(json.dumps({'num_examples':len(rows),'num_skipped':len(skipped),'skipped':skipped[:50]},indent=2),encoding='utf-8')
    print(json.dumps({'out_dir':str(out),'num_examples':len(rows),'num_skipped':len(skipped)},indent=2))
if __name__=='__main__': main()
