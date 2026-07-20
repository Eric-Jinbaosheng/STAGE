#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, sys, platform
from pathlib import Path
from typing import Any, Dict
import torch

REPO_ROOT=Path(__file__).resolve().parents[1]
for p in [REPO_ROOT/'src', REPO_ROOT/'scripts']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from lb_run_openvla_action import load_hdf5_image_pointer
from linguistic_blindness.utils.io import read_jsonl, write_jsonl, write_json, write_csv, command_string, utc_timestamp

SAFE_ACTIONS=['ASK','PROMPT','WAIT','HOLD','ABORT','BACKOFF','TARGET_NOT_FOUND','MOVE_TO','GRASP','PICK','PLACE','RETRACT','TRANSFER']

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmark',default='outputs/linguistic_blindness/visual_relation_counterfactual_600/benchmark.jsonl')
    ap.add_argument('--model-path',default='/scratch/bj2410/models/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--out-dir',default='outputs/linguistic_blindness/qwen25vl7b_visual_relation_schema_600')
    ap.add_argument('--max-examples',type=int,default=0)
    ap.add_argument('--device-map',default='auto')
    ap.add_argument('--bf16',action='store_true')
    ap.add_argument('--trust-remote-code',action='store_true')
    ap.add_argument('--max-new-tokens',type=int,default=160)
    ap.add_argument('--log-every',type=int,default=25)
    ap.add_argument('--save-every',type=int,default=100)
    return ap.parse_args()

def dtype(args): return torch.bfloat16 if args.bf16 else 'auto'

def load_image(ptr):
    if '#' in ptr:
        return load_hdf5_image_pointer(ptr)
    from PIL import Image
    return Image.open(ptr).convert('RGB')

def load_model(args):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc=AutoProcessor.from_pretrained(args.model_path,trust_remote_code=args.trust_remote_code)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path,trust_remote_code=args.trust_remote_code,device_map=args.device_map,torch_dtype=dtype(args),low_cpu_mem_usage=True)
    model.eval(); return model,proc

def prompt(ex):
    return f"""You are a visual relation grounding module for a robot manipulation scene.
Look at the image and the instruction. Return ONLY valid JSON, no markdown.

Instruction: {ex.get('instruction')!r}
Candidate scene objects: {json.dumps(ex.get('scene_objects', []), ensure_ascii=True)}

Task: identify which candidate object the relation-defined instruction refers to. The target object name may NOT appear in the instruction. Use the image relation, not only the text.

Return exactly these keys:
{{
  "anchor_object": string or null,
  "relation": string or null,
  "target_object": string or null,
  "target_exists": boolean,
  "phase": "approach" or "wait" or "UNK",
  "allowed_actions": list[str],
  "blocked_actions": list[str],
  "next_action": one of {json.dumps(SAFE_ACTIONS)},
  "reason": short string,
  "confidence": number between 0 and 1 or null
}}
"""

def parse_json(text):
    raw=text.strip()
    if raw.startswith('```'):
        raw=re.sub(r'^```(?:json)?','',raw).strip(); raw=re.sub(r'```$','',raw).strip()
    try: return json.loads(raw)
    except Exception: pass
    m=re.search(r'\{.*\}',raw,re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return {'_parse_error':True,'raw_text':text}

def normalize_schema(s):
    if not isinstance(s,dict): s={}
    out={
        'anchor_object':s.get('anchor_object'),
        'relation':s.get('relation'),
        'target_object':s.get('target_object'),
        'target_exists':bool(s.get('target_exists')) if s.get('target_exists') is not None else False,
        'phase':s.get('phase') or 'UNK',
        'allowed_actions':s.get('allowed_actions') if isinstance(s.get('allowed_actions'),list) else [],
        'blocked_actions':s.get('blocked_actions') if isinstance(s.get('blocked_actions'),list) else [],
        'next_action':str(s.get('next_action') or 'ASK').strip().upper(),
        'reason':s.get('reason') or '',
        'confidence':s.get('confidence'),
    }
    if s.get('_parse_error'): out['_parse_error']=True; out['raw_text']=s.get('raw_text')
    return out

@torch.no_grad()
def generate(model,proc,ex,args):
    from qwen_vl_utils import process_vision_info
    img=load_image(ex['obs_ptr'])
    messages=[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':prompt(ex)}]}]
    text=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    image_inputs,video_inputs=process_vision_info(messages)
    inputs=proc(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors='pt')
    inputs=inputs.to(next(model.parameters()).device)
    out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
    gen=out[:,inputs.input_ids.shape[1]:]
    return proc.batch_decode(gen,skip_special_tokens=True)[0].strip()

def main():
    args=parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(args.benchmark)
    if args.max_examples>0: rows=rows[:args.max_examples]
    model,proc=load_model(args)
    preds=[]
    for i,ex in enumerate(rows,1):
        raw=generate(model,proc,ex,args)
        schema=normalize_schema(parse_json(raw))
        gold=ex.get('counterfactual_target_object')
        pred=schema.get('target_object')
        correct= str(pred).strip().lower()==str(gold).strip().lower()
        preds.append({'example_id':ex['example_id'],'instruction':ex.get('instruction'),'gold_target_object':gold,'parsed_schema':schema,'raw_model_output':raw,'target_correct':correct,'target_exists_correct':schema.get('target_exists') is True,'parse_success':not schema.get('_parse_error')})
        if args.log_every and (i%args.log_every==0 or i==len(rows)): print(f'progress {i}/{len(rows)}',flush=True)
        if args.save_every and i%args.save_every==0: write_jsonl(out/'predictions.partial.jsonl',preds)
    n=len(preds)
    summary={'n':n,'target_grounding_accuracy':sum(p['target_correct'] for p in preds)/n,'target_exists_accuracy':sum(p['target_exists_correct'] for p in preds)/n,'parse_success_rate':sum(p['parse_success'] for p in preds)/n}
    write_jsonl(out/'predictions.jsonl',preds); write_json(out/'summary.json',summary); write_csv(out/'summary.csv',[summary])
    write_json(out/'run_metadata.json',{'command':command_string(),'timestamp_utc':utc_timestamp(),'model_path':args.model_path,'benchmark':args.benchmark,'python':sys.version,'platform':platform.platform()})
    print(json.dumps({'out_dir':str(out),**summary},indent=2))
if __name__=='__main__': main()
