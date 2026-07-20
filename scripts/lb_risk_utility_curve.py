#!/usr/bin/env python3
from __future__ import annotations
import json,csv
from pathlib import Path

def read_jsonl(p):
 with open(p) as f: return [json.loads(l) for l in f if l.strip()]
def write_csv(p,rows):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
 if not rows: p.write_text(''); return
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys: keys.append(k)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def strip(e):
 for p in ['template_invalid::','hard_invalid::','normal::']:
  if e.startswith(p): return e[len(p):]
 return e

def main():
 out=Path('outputs/linguistic_blindness/risk_utility_direct_visa_hybrid'); out.mkdir(parents=True,exist_ok=True)
 direct=read_jsonl('outputs/linguistic_blindness/direct_qwen_gate_combined_2700/qwen25vl7b_direct_gate/predictions.jsonl')
 overlap=read_jsonl('outputs/linguistic_blindness/direct_qwen_gate_combined_2700/qwen25vl7b_direct_gate/analysis/direct_visa_overlap_examples.jsonl')
 rows=[]
 for split in ['template_invalid','hard_invalid','normal','invalid_all','overall']:
  g=[r for r in overlap if (split=='overall' or r['split']==split or (split=='invalid_all' and r['split']!='normal'))]
  if not g: continue
  for method in ['direct','visa','hybrid']:
   n=len(g)
   if method=='direct': allow=[r['direct_allow'] for r in g]
   elif method=='visa': allow=[r['visa_allow'] for r in g]
   else: allow=[r['hybrid_allow'] for r in g]
   if split=='normal': correct=sum(allow)/n; safe=None
   else: correct=sum(not a for a in allow)/n; safe=correct
   normal_g=[r for r in overlap if r['split']=='normal']
   invalid_g=[r for r in overlap if r['split']!='normal']
   def allow_for(r): return r[f'{method}_allow'] if method!='hybrid' else r['hybrid_allow']
   normal_pass=sum(allow_for(r) for r in normal_g)/len(normal_g)
   invalid_safe=sum(not allow_for(r) for r in invalid_g)/len(invalid_g)
   bal=(normal_pass+invalid_safe)/2
   rows.append({'method':method,'slice':split,'n':n,'correct_rate':correct,'safe_deferral_rate':safe,'allow_rate':sum(allow)/n,'normal_pass_global':normal_pass,'invalid_safe_deferral_global':invalid_safe,'balanced_correct_global':bal})
 write_csv(out/'risk_utility_summary.csv',rows)
 # operating point table
 ops=[]
 for method in ['direct','visa','hybrid']:
  r=next(x for x in rows if x['method']==method and x['slice']=='overall')
  ops.append({'method':method,'normal_pass':r['normal_pass_global'],'invalid_safe_deferral':r['invalid_safe_deferral_global'],'balanced_correct':r['balanced_correct_global']})
 write_csv(out/'risk_utility_operating_points.csv',ops)
 md=['# Risk-Utility Summary','','| Method | Normal pass | Invalid safe deferral | Balanced correct |','|---|---:|---:|---:|']
 for r in ops: md.append(f"| {r['method']} | {r['normal_pass']:.3f} | {r['invalid_safe_deferral']:.3f} | {r['balanced_correct']:.3f} |")
 (out/'risk_utility_report.md').write_text('\n'.join(md))
 print(json.dumps({'out_dir':str(out),'rows':ops},indent=2))
if __name__=='__main__': main()
