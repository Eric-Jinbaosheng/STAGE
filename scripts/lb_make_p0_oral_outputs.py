#!/usr/bin/env python
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())

def read_jsonl(path):
    p=Path(path)
    if not p.exists(): return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def read_csv(path):
    p=Path(path)
    if not p.exists(): return []
    return list(csv.DictReader(open(p)))

def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def mean(vals):
    vals=[float(v) for v in vals if v is not None and v!='']
    return float(np.mean(vals)) if vals else None

def ci(vals, seed=0, n=2000):
    vals=np.asarray([float(v) for v in vals], dtype=float)
    if vals.size==0: return (None,None)
    rng=np.random.default_rng(seed)
    boots=[vals[rng.integers(0, vals.size, vals.size)].mean() for _ in range(n)]
    return (float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5)))

def metric(path, key):
    d=read_json(path)
    cur=d
    for p in key.split('.'):
        cur=cur[p]
    return cur

def hidden_row(model, setting, layer_label, path):
    d=read_json(Path(path)/'metrics.json')
    return {
        'model': model,
        'condition': setting,
        'layer': layer_label,
        'n': d['num_examples'],
        'train_examples': d['num_train_examples'],
        'test_examples': d['num_test_examples'],
        'target_accuracy': d['hidden_probe_target_accuracy']['mean'],
        'target_accuracy_ci95': d['hidden_probe_target_accuracy'].get('ci95'),
        'schema_sensitivity': d['hidden_probe_schema_sensitivity']['mean'],
        'schema_sensitivity_ci95': d['hidden_probe_schema_sensitivity'].get('ci95'),
        'action_sensitivity': d['openvla_action_sensitivity']['mean'],
        'semantic_action_gap': d['hidden_probe_action_gap']['mean'],
        'majority_baseline': d['majority_baseline_accuracy']['mean'],
        'random_labels': d.get('random_labels', False),
        'source': str(path),
    }

def rollout_summary(path):
    pairs=read_jsonl(Path(path)/'short_horizon_pairs.jsonl')
    if not pairs: return [], {}
    vals={
        'orig_counterfactual_approach': [1.0 if r['orig_delta_cf_target']>0 else 0.0 for r in pairs],
        'cf_counterfactual_approach': [1.0 if r['cf_delta_cf_target']>0 else 0.0 for r in pairs],
        'orig_original_approach': [1.0 if r['orig_delta_orig_target']>0 else 0.0 for r in pairs],
        'cf_original_approach': [1.0 if r['cf_delta_orig_target']>0 else 0.0 for r in pairs],
        'cf_minus_orig_counterfactual_approach': [r['cf_minus_orig_counterfactual_approach'] for r in pairs],
        'orig_target_preference_score': [r['orig_target_preference_score'] for r in pairs],
        'cf_target_preference_score': [r['cf_target_preference_score'] for r in pairs],
    }
    rows=[]
    for k,v in vals.items():
        lo,hi=ci(v, seed=23)
        rows.append({'metric':k,'n':len(v),'mean':mean(v),'ci95_low':lo,'ci95_high':hi})
    # time curve: mean distance improvement by step and method from rollouts.
    rollouts=read_jsonl(Path(path)/'short_horizon_rollouts.jsonl')
    by=[]
    for method in sorted({r.get('method') for r in rollouts}):
        rs=[r for r in rollouts if r.get('method')==method]
        max_step=max((len(r.get('trace',[])) for r in rs), default=0)
        for step in range(max_step):
            cf=[]; orig=[]
            for r in rs:
                tr=r.get('trace',[])
                if len(tr)>step:
                    cf.append(float(r['start_dist_counterfactual'])-float(tr[step]['dist_counterfactual']))
                    orig.append(float(r['start_dist_original'])-float(tr[step]['dist_original']))
            by.append({'method':method,'step':step,'mean_cf_target_approach':mean(cf),'mean_orig_target_approach':mean(orig),'n':len(cf)})
    return rows, {'curve': by, 'n_pairs': len(pairs)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='outputs/linguistic_blindness/p0_oral_main_outputs')
    ap.add_argument('--rollout-dir', default='outputs/linguistic_blindness/short_horizon_target_approach_k20_100')
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    hidden=[]
    specs=[
        ('OpenVLA base','full','early layer 8','outputs/linguistic_blindness/hprobe_openvla_base_layer8_600'),
        ('OpenVLA base','full','middle layer 16','outputs/linguistic_blindness/hprobe_openvla_base_layer16_600'),
        ('OpenVLA base','full','late/final layer','outputs/linguistic_blindness/exp6_openvla_hidden_state_probe_600'),
        ('OpenVLA base','language-only','late/final layer','outputs/linguistic_blindness/exp6_openvla_hidden_state_probe_600_language_only'),
        ('OpenVLA base','image-only','late/final layer','outputs/linguistic_blindness/exp6_openvla_hidden_state_probe_600_image_only'),
        ('OpenVLA base','mask-target','late/final layer','outputs/linguistic_blindness/exp6_openvla_hidden_state_probe_600_mask_target'),
        ('OpenVLA base','random-label','late/final layer','outputs/linguistic_blindness/hprobe_openvla_base_random_labels_600'),
        ('OpenVLA-LIBERO90','full','early layer 8','outputs/linguistic_blindness/hprobe_openvla_libero90_layer8_600'),
        ('OpenVLA-LIBERO90','full','middle layer 16','outputs/linguistic_blindness/hprobe_openvla_libero90_layer16_600'),
        ('OpenVLA-LIBERO90','full','late/final layer','outputs/linguistic_blindness/hprobe_openvla_libero90_layerlast_600'),
        ('OpenVLA-LIBERO90','language-only','late/final layer','outputs/linguistic_blindness/hprobe_openvla_libero90_language_only_600'),
        ('OpenVLA-LIBERO90','image-only','late/final layer','outputs/linguistic_blindness/hprobe_openvla_libero90_image_only_600'),
        ('OpenVLA-LIBERO90','mask-target','late/final layer','outputs/linguistic_blindness/hprobe_openvla_libero90_mask_target_600'),
        ('OpenVLA-LIBERO90','random-label','late/final layer','outputs/linguistic_blindness/hprobe_openvla_libero90_random_labels_600'),
    ]
    for s in specs:
        if (Path(s[3])/'metrics.json').exists():
            hidden.append(hidden_row(*s))
    write_csv(out/'hidden_state_probe_main_table.csv', hidden)

    # Complete cross-policy/domain matrix with ActCheck and AUC where available.
    matrix=[
        {'policy':'OpenVLA base','domain':'LIBERO','intervention':'target-name swap','n':600,'semantic_signal':1.0,'action_sensitivity':0.06833333333333333,'SAG':0.9316666666666666,'AUC':0.573,'actcheck_flag_rate':'','source':'exp2_openvla_qwen_target_swap_clean_600'},
        {'policy':'OpenVLA-LIBERO90','domain':'LIBERO','intervention':'target-name swap','n':600,'semantic_signal':1.0,'action_sensitivity':0.056667,'SAG':0.943333,'AUC':'','actcheck_flag_rate':'','source':'exp2_second_vla_openvla_libero90_target_swap_600'},
        {'policy':'Octo-small-1.5','domain':'LIBERO','intervention':'target-name swap','n':600,'semantic_signal':1.0,'action_sensitivity':0.426667,'SAG':0.573333,'AUC':'','actcheck_flag_rate':'','source':'exp2_octo_small_target_swap_600'},
        {'policy':'OpenVLA base','domain':'LIBERO','intervention':'pixel relation swap','n':600,'semantic_signal':0.9583333333333334,'action_sensitivity':0.07666666666666666,'SAG':0.8816666666666667,'AUC':0.583,'actcheck_flag_rate':0.455,'source':'visual_relation_openvla + actcheck'},
        {'policy':'Octo-small-1.5','domain':'LIBERO','intervention':'pixel relation swap','n':600,'semantic_signal':0.9583333333333334,'action_sensitivity':0.375,'SAG':0.5833333333333334,'AUC':'','actcheck_flag_rate':0.42833333333333334,'source':'octo_visual_relation_analysis'},
        {'policy':'Octo-small-1.5','domain':'BridgeData V2','intervention':'target/task change','n':165,'semantic_signal':1.0,'action_sensitivity':0.6606060606060606,'SAG':0.33939393939393936,'AUC':0.9289256,'actcheck_flag_rate':'N/A metadata-limited','source':'bridge_v2_target_change_analysis'},
    ]
    write_csv(out/'cross_policy_domain_main_table.csv', matrix)

    if Path(args.rollout_dir).exists():
        roll, meta=rollout_summary(args.rollout_dir)
        write_csv(out/'paired_rollout_k20_100_bootstrap_ci.csv', roll)
        write_csv(out/'paired_rollout_k20_100_target_consistency_curve.csv', meta.get('curve', []))
    patch_path = Path('outputs/linguistic_blindness/openvla_activation_patching_100_layer16/summary.json')
    patch_summary = None
    if patch_path.exists():
        patch_summary = read_json(patch_path)
        write_csv(out/'activation_patching_summary.csv', [patch_summary])
    # Markdown digest.
    lines=['# P0/P1 oral supplement outputs','', '## Hidden-state probe main result']
    for r in hidden:
        lines.append(f"- {r['model']} {r['condition']} {r['layer']}: target acc={float(r['target_accuracy']):.3f}, schema sens={float(r['schema_sensitivity']):.3f}, action sens={float(r['action_sensitivity']):.3f}, gap={float(r['semantic_action_gap']):.3f}")
    lines += ['', '## Cross-policy/domain matrix']
    for r in matrix:
        lines.append(f"- {r['policy']} / {r['domain']} / {r['intervention']}: N={r['n']}, sem={float(r['semantic_signal']):.3f}, action={float(r['action_sensitivity']):.3f}, SAG={float(r['SAG']):.3f}, ActCheck={r['actcheck_flag_rate']}")
    if Path(args.rollout_dir).exists():
        lines += ['', '## Paired K=20 rollout 100+']
        for r in roll:
            lines.append(f"- {r['metric']}: mean={float(r['mean']):.4f}, 95% CI=[{float(r['ci95_low']):.4f}, {float(r['ci95_high']):.4f}], n={r['n']}")
    if patch_summary:
        lines += ['', '## Activation patching causal test']
        lines.append(
            f"- Layer {patch_summary['layer_index']} alpha={patch_summary['alpha']}, N={patch_summary['n']}: "
            f"native action sensitivity={patch_summary['native_action_sensitivity']:.3f}, "
            f"patched action sensitivity={patch_summary['patched_action_sensitivity']:.3f}, "
            f"patch moves toward CF action={patch_summary['patch_moves_toward_cf_action_rate']:.3f} "
            f"(95% CI {patch_summary['patch_moves_toward_cf_ci95']})."
        )
    (out/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    print(out)

if __name__=='__main__':
    main()
