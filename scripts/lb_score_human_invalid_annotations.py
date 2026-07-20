#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from collections import defaultdict, Counter

def parse_args():
    p=argparse.ArgumentParser(description='Score human invalid instruction annotations and agreement.')
    p.add_argument('--annotations', required=True, help='CSV with annotation_id, annotator_id, should_execute, reason_type, target_object')
    p.add_argument('--out-dir', default='outputs/linguistic_blindness/human_invalid_annotation_scored')
    return p.parse_args()

def pair_agreement(vals):
    vals=[v for v in vals if v!='']
    if len(vals)<2: return None
    total=agree=0
    for i in range(len(vals)):
        for j in range(i+1,len(vals)):
            total+=1; agree+= int(vals[i]==vals[j])
    return agree/total if total else None

def main():
    args=parse_args(); out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows=list(csv.DictReader(open(args.annotations)))
    by=defaultdict(list)
    for r in rows: by[r['annotation_id']].append(r)
    item_rows=[]
    for aid,rs in sorted(by.items()):
        execs=[r.get('should_execute','').strip().upper() for r in rs]
        reasons=[r.get('reason_type','').strip().lower() for r in rs]
        targets=[r.get('target_object','').strip().lower() for r in rs]
        item_rows.append({'annotation_id':aid,'n_annotators':len(rs),'should_execute_agreement':pair_agreement(execs),'reason_type_agreement':pair_agreement(reasons),'target_object_agreement':pair_agreement(targets),'majority_should_execute':Counter(execs).most_common(1)[0][0] if execs else ''})
    def avg(k):
        vals=[r[k] for r in item_rows if r[k] is not None]
        return sum(vals)/len(vals) if vals else None
    summary={'n_items':len(item_rows),'n_rows':len(rows),'mean_should_execute_pair_agreement':avg('should_execute_agreement'),'mean_reason_type_pair_agreement':avg('reason_type_agreement'),'mean_target_object_pair_agreement':avg('target_object_agreement')}
    with (out/'item_agreement.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(item_rows[0].keys()) if item_rows else ['annotation_id']); w.writeheader(); w.writerows(item_rows)
    (out/'agreement_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
