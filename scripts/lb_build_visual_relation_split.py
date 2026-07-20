#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def read_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]
def write_jsonl(p, rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w') as f:
        for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
def relation(pos_a,pos_b):
    dx=pos_a[0]-pos_b[0]; dy=pos_a[1]-pos_b[1]; dz=pos_a[2]-pos_b[2]
    if abs(dz)>0.06 and abs(dz)>max(abs(dx),abs(dy))*0.7:
        return 'above' if dz>0 else 'below'
    if abs(dx)>=abs(dy): return 'right of' if dx>0 else 'left of'
    return 'behind' if dy>0 else 'in front of'
def instr_for(target, anchor, rel):
    if rel in {'above','below'}:
        return f"pick up the object {rel} the {anchor}"
    return f"pick up the object to the {rel} the {anchor}"
def schema(target):
    return {'target_object':target,'target_exists':True,'phase':'approach','human_contact':None,'human_released':None,'robot_contact':None,'robot_grasp_stable':None,'allowed_actions':['MOVE_TO','APPROACH','GRASP','PICK'],'blocked_actions':[],'next_action':'MOVE_TO','reason':'gold visual-relation schema from MuJoCo object positions','confidence':1.0}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--positions',default='outputs/linguistic_blindness/exp2_openvla_qwen_target_swap_clean_600/analysis/predictions_with_positions.jsonl')
    ap.add_argument('--out-dir',default='outputs/linguistic_blindness/visual_relation_counterfactual_600')
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; qcf=[]; qorig=[]
    for r in read_jsonl(args.positions):
        if r.get('position_extraction_status')!='ok': continue
        orig=r['original_target_object']; cf=r['counterfactual_target_object']
        po=r['original_target_pos']; pc=r['counterfactual_target_pos']
        rel_orig=relation(po,pc); rel_cf=relation(pc,po)
        orig_inst=instr_for(orig,cf,rel_orig)
        cf_inst=instr_for(cf,orig,rel_cf)
        eid=r['example_id'].replace('::target_swap','::visual_relation_swap')
        base={
            'example_id':eid,
            'source_example_id':r['example_id'],
            'observation_id':r['observation_id'],
            'obs_ptr':r['image_source'],
            'dataset':'libero',
            'perturbation_type':'target_swap',
            'visual_relation_type':'relative_position',
            'original_instruction':orig_inst,
            'counterfactual_instruction':cf_inst,
            'instruction':cf_inst,
            'original_target_object':orig,
            'counterfactual_target_object':cf,
            'target_object':cf,
            'counterfactual_valid':True,
            'scene_objects':sorted({orig,cf}),
            'observation_state':{'ee_pos':r.get('ee_pos'),'original_target_pos':po,'counterfactual_target_pos':pc,'original_relation_to_anchor':rel_orig,'counterfactual_relation_to_anchor':rel_cf},
            'gold_schema':schema(cf),
            'original_schema':schema(orig),
            'target_distance':r.get('target_distance'),
            'target_angle_deg':r.get('target_angle_deg'),
            'smoke_test':False,
        }
        rows.append(base)
        qcf.append({'example_id':eid,'parsed_schema':schema(cf),'raw_model_output':json.dumps(schema(cf)),'method':'oracle_visual_relation_schema','visual_relation_oracle':True})
        qorig.append({'example_id':eid,'parsed_schema':schema(orig),'raw_model_output':json.dumps(schema(orig)),'method':'oracle_visual_relation_schema','visual_relation_oracle':True})
    write_jsonl(out/'benchmark.jsonl',rows)
    write_jsonl(out/'oracle_schema_counterfactual.jsonl',qcf)
    write_jsonl(out/'oracle_schema_original.jsonl',qorig)
    stats={'num_examples':len(rows),'note':'Visual-relation target swap benchmark with gold targets computed from MuJoCo positions; schema predictions are oracle labels, not VLM outputs.'}
    (out/'benchmark_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps({'out_dir':str(out),**stats},indent=2))
if __name__=='__main__': main()
