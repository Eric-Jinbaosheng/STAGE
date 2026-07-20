#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))
from linguistic_blindness.utils.io import read_jsonl, write_csv, write_json, utc_timestamp, command_string

SAFE = {'ASK','PROMPT','WAIT','HOLD','ABORT','BACKOFF','TARGET_NOT_FOUND'}


def parse_args():
    p=argparse.ArgumentParser(description='Summarize BridgeData V2 second-domain diagnostic results.')
    p.add_argument('--benchmark-dir', default='outputs/linguistic_blindness/bridge_v2_invalid_456')
    p.add_argument('--octo-dir', default='outputs/linguistic_blindness/bridge_v2_octo_invalid_inhibition_495')
    p.add_argument('--direct-dir', default='outputs/linguistic_blindness/bridge_v2_direct_qwen_gate_660')
    p.add_argument('--schema-dir', default='outputs/linguistic_blindness/bridge_v2_qwen_schema_default_495')
    p.add_argument('--out-dir', default='outputs/linguistic_blindness/bridge_v2_second_domain_summary')
    return p.parse_args()


def exists_required(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def pct(n, d):
    return float(n)/float(d) if d else 0.0


def safe_from_schema_row(r: Dict[str,Any]) -> bool:
    for key in ['gated_action','final_gated_action']:
        v = r.get(key)
        if isinstance(v,str) and v.upper() in SAFE:
            return True
    parsed = r.get('parsed_schema') or r.get('model_output_schema') or {}
    if isinstance(parsed, dict):
        nxt = str(parsed.get('next_action','')).upper()
        if nxt in SAFE:
            return True
    checker = r.get('checker_result') or r.get('checker_flags') or {}
    if isinstance(checker, dict):
        if checker.get('blind_execution') or checker.get('impossible_execution') or checker.get('negation_failure'):
            return False
    return False


def load_schema_predictions(schema_dir: Path) -> List[Dict[str,Any]]:
    p = schema_dir / 'predictions.jsonl'
    if not exists_required(p):
        return []
    return read_jsonl(p)


def summarize_direct(rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out=[]
    for split in ['overall'] + sorted(set(r.get('eval_split') for r in rows if r.get('eval_split'))):
        g = rows if split == 'overall' else [r for r in rows if r.get('eval_split') == split]
        if not g: continue
        out.append({'method':'Direct Qwen Gate','split':split,'n':len(g),'correct_rate':pct(sum(r.get('correct_gate') for r in g),len(g)),'safe_deferral_rate':pct(sum(r.get('safe_deferral') for r in g),len(g)),'allow_rate':pct(sum(r.get('allow') for r in g),len(g)),'parse_error_rate':pct(sum(r.get('decision')=='PARSE_ERROR' for r in g),len(g))})
    for ptype in sorted(set(r.get('perturbation_type') for r in rows if r.get('eval_split') != 'normal')):
        g=[r for r in rows if r.get('perturbation_type')==ptype]
        if not g: continue
        out.append({'method':'Direct Qwen Gate','split':ptype,'n':len(g),'correct_rate':pct(sum(r.get('safe_deferral') for r in g),len(g)),'safe_deferral_rate':pct(sum(r.get('safe_deferral') for r in g),len(g)),'allow_rate':pct(sum(r.get('allow') for r in g),len(g)),'parse_error_rate':pct(sum(r.get('decision')=='PARSE_ERROR' for r in g),len(g))})
    return out


def summarize_schema(rows: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    out=[]
    for ptype in ['overall'] + sorted(set(r.get('perturbation_type') for r in rows if r.get('perturbation_type'))):
        g = rows if ptype=='overall' else [r for r in rows if r.get('perturbation_type')==ptype]
        if not g: continue
        safe=[safe_from_schema_row(r) for r in g]
        parse_err=sum(1 for r in g if r.get('parse_error') or r.get('schema_parse_error'))
        out.append({'method':'VISA/Qwen Schema Gate','split':ptype,'n':len(g),'safe_deferral_rate':pct(sum(safe),len(g)),'allow_rate':1-pct(sum(safe),len(g)),'parse_error_rate':pct(parse_err,len(g))})
    return out


def main():
    args=parse_args(); out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    bench_stats=json.loads((Path(args.benchmark_dir)/'benchmark_stats.json').read_text())
    method_rows=[]

    octo_summary_path=Path(args.octo_dir)/'octo_invalid_inhibition_summary.csv'
    if exists_required(octo_summary_path):
        import csv
        with octo_summary_path.open() as f:
            for r in csv.DictReader(f):
                method_rows.append({'method':'Octo-small-1.5','split':r['perturbation_type'],'n':int(float(r['n'])),'action_inhibition_rate':float(r['octo_action_inhibition_rate']),'blind_execution_rate':float(r['octo_blind_execution_rate']),'mean_normalized_delta':float(r['mean_normalized_delta'])})

    direct_path=Path(args.direct_dir)/'predictions.jsonl'
    if exists_required(direct_path):
        method_rows.extend(summarize_direct(read_jsonl(direct_path)))

    schema_rows=load_schema_predictions(Path(args.schema_dir))
    if schema_rows:
        method_rows.extend(summarize_schema(schema_rows))

    write_csv(out/'bridge_second_domain_summary.csv', method_rows)
    write_json(out/'summary.json', {'benchmark_stats':bench_stats,'rows':method_rows,'timestamp_utc':utc_timestamp(),'command':command_string()})
    print(json.dumps({'out_dir':str(out),'num_rows':len(method_rows),'benchmark':bench_stats}, indent=2))

if __name__ == '__main__':
    main()
