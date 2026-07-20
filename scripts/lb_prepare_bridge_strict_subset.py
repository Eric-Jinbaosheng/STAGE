#!/usr/bin/env python
import argparse, csv, json, re, shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

OBJECT_TERMS = [
    'bowl','fork','spoon','knife','cube','block','cloth','pot','pan','can','drawer','fridge','broccoli','potato','pear','brush','toy','carrot','object'
]
BAD_TERMS = ['end effector','sweep into pile','fold the cloth','unfold the cloth','bmbfbbfgjjg','moved the','moves the','picked the piece','out the object']

def norm(s): return str(s or '').strip().lower()
def concrete(s):
    ss=norm(s)
    if not ss or any(b in ss for b in BAD_TERMS): return False
    return any(t in ss for t in OBJECT_TERMS)
def read_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()]
def write_csv(path, rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def load_font(size=18):
    for p in ['/usr/share/fonts/dejavu/DejaVuSans.ttf','/usr/share/fonts/liberation/LiberationSans-Regular.ttf']:
        if Path(p).exists(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='outputs/linguistic_blindness/bridge_v2_target_change_165/benchmark.jsonl')
    ap.add_argument('--predictions', default='outputs/linguistic_blindness/bridge_v2_octo_target_change_165/predictions.jsonl')
    ap.add_argument('--out-dir', default='outputs/linguistic_blindness/bridge_v2_strict_target_audit_subset')
    ap.add_argument('--max-candidates', type=int, default=120)
    args=ap.parse_args()
    out=Path(args.out_dir); imgout=out/'images'; sheetout=out/'contact_sheets'; imgout.mkdir(parents=True, exist_ok=True); sheetout.mkdir(parents=True, exist_ok=True)
    bench={r['example_id']:r for r in read_jsonl(args.benchmark)}
    preds={r['example_id']:r for r in read_jsonl(args.predictions)}
    rows=[]
    for eid,b in bench.items():
        p=preds.get(eid,{})
        ot,ct=norm(b.get('original_target_object')),norm(b.get('counterfactual_target_object'))
        if ot==ct or not concrete(ot) or not concrete(ct): continue
        ip=Path(b.get('image_path',''))
        if not ip.exists(): continue
        rows.append({
            'example_id':eid,
            'image_path':str(ip),
            'original_instruction':b.get('original_instruction'),
            'counterfactual_instruction':b.get('counterfactual_instruction'),
            'original_target_object':ot,
            'counterfactual_target_object':ct,
            'octo_action_sensitive':p.get('octo_action_sensitive'),
            'normalized_action_delta':p.get('normalized_action_delta'),
            'action_cosine':p.get('action_cosine'),
            'needs_visual_human_audit': True,
            'strict_target_visible_label': '',
            'competing_object_visible_label': '',
            'auditor_notes': '',
        })
    rows=rows[:args.max_candidates]
    for i,r in enumerate(rows,1):
        dst=imgout/f'{i:03d}_{r["example_id"]}.png'
        if not dst.exists(): shutil.copyfile(r['image_path'], dst)
        r['audit_image']=str(dst)
    write_csv(out/'bridge_strict_target_audit_sheet.csv', rows)
    # provisional only: concrete-text subset metric, not hand-audited metric.
    vals=[bool(r['octo_action_sensitive']) for r in rows if r.get('octo_action_sensitive') is not None]
    summary=[{'subset':'concrete_text_candidate_requires_visual_audit','n':len(rows),'action_sensitivity':float(np.mean(vals)) if vals else None,'caveat':'Not hand-labeled yet; use audit_sheet labels before calling this strict object-grounded.'}]
    write_csv(out/'bridge_strict_candidate_summary.csv', summary)
    # contact sheets for audit.
    font=load_font(16); W,H=320,300
    for si in range(0,len(rows),20):
        chunk=rows[si:si+20]
        sheet=Image.new('RGB',(5*W,4*H),'white'); draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(chunk):
            im=Image.open(r['audit_image']).convert('RGB'); im.thumbnail((W-20,190))
            x=(j%5)*W+10; y=(j//5)*H+10
            sheet.paste(im,(x,y))
            txt=f"{si+j+1:03d}: {r['original_target_object']} -> {r['counterfactual_target_object']}\nΔ={float(r['normalized_action_delta'] or 0):.2f} sens={r['octo_action_sensitive']}"
            draw.multiline_text((x,y+200),txt,font=font,fill=(0,0,0))
        sheet.save(sheetout/f'bridge_strict_audit_sheet_{si//20+1:02d}.png')
    (out/'README.md').write_text('''# BridgeData V2 strict target audit subset\n\nThis packet prepares concrete-object Bridge target-change candidates for human visual audit.\nDo not report this as a hand-labeled strict subset until `strict_target_visible_label` and `competing_object_visible_label` are filled by a human auditor.\n\nFiles:\n- `bridge_strict_target_audit_sheet.csv`: annotation sheet.\n- `contact_sheets/*.png`: images for visual audit.\n- `bridge_strict_candidate_summary.csv`: provisional concrete-text candidate action sensitivity only.\n''',encoding='utf-8')
    print(out, len(rows))
if __name__=='__main__': main()
