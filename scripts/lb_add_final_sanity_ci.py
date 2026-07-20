#!/usr/bin/env python
import csv, json, math
from pathlib import Path
import numpy as np

OUT = Path('outputs/linguistic_blindness/final_sanity_ci')
OUT.mkdir(parents=True, exist_ok=True)

def read_jsonl(p):
    p=Path(p)
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def write_csv(path, rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def mean(xs):
    xs=[float(x) for x in xs if x is not None and not (isinstance(x,float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else None

def ci(xs, seed=0, n=5000):
    xs=np.asarray([float(x) for x in xs if x is not None], dtype=float)
    if len(xs)==0: return (None,None)
    rng=np.random.default_rng(seed)
    boots=np.empty(n)
    for i in range(n):
        boots[i]=xs[rng.integers(0,len(xs),len(xs))].mean()
    return float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5))

def perm_p(a,b, seed=0, n=10000):
    a=np.asarray([float(x) for x in a if x is not None], dtype=float)
    b=np.asarray([float(x) for x in b if x is not None], dtype=float)
    if len(a)==0 or len(b)==0: return None
    obs=a.mean()-b.mean()
    pooled=np.concatenate([a,b])
    rng=np.random.default_rng(seed)
    cnt=0
    for _ in range(n):
        rng.shuffle(pooled)
        stat=pooled[:len(a)].mean()-pooled[len(a):].mean()
        if abs(stat) >= abs(obs): cnt += 1
    return float((cnt+1)/(n+1))

def summarize_bool(rows, key):
    vals=[1.0 if r.get(key) else 0.0 for r in rows if r.get(key) is not None]
    lo,hi=ci(vals, seed=11)
    return mean(vals), lo, hi, len(vals)

# 1) Pixel-relation hidden probe CI and image-dependent subset.
conds={
    'full':'outputs/linguistic_blindness/hprobe_openvla_pixel_relation_full_600/hidden_state_pair_predictions.jsonl',
    'language_only':'outputs/linguistic_blindness/hprobe_openvla_pixel_relation_language_only_600/hidden_state_pair_predictions.jsonl',
    'shuffled_image':'outputs/linguistic_blindness/hprobe_openvla_pixel_relation_shuffled_image_600/hidden_state_pair_predictions.jsonl',
    'random_label':'outputs/linguistic_blindness/hprobe_openvla_pixel_relation_random_label_600/hidden_state_pair_predictions.jsonl',
}
pairs={k:{r['example_id']:r for r in read_jsonl(v)} for k,v in conds.items()}
common=set.intersection(*[set(x.keys()) for x in pairs.values()]) if pairs else set()
rows=[]
for cond,mp in pairs.items():
    rs=[mp[e] for e in sorted(mp)]
    sens,lo,hi,n=summarize_bool(rs,'hidden_probe_schema_sensitive')
    act,alo,ahi,an=summarize_bool(rs,'openvla_action_sensitive')
    gap_vals=[(1.0 if r.get('hidden_probe_schema_sensitive') else 0.0)-(1.0 if r.get('openvla_action_sensitive') else 0.0) for r in rs if r.get('openvla_action_sensitive') is not None]
    glo,ghi=ci(gap_vals, seed=12)
    rows.append({'condition':cond,'subset':'all_test_pairs','n':n,'schema_sensitivity':sens,'schema_sensitivity_ci95_low':lo,'schema_sensitivity_ci95_high':hi,'action_sensitivity':act,'action_sensitivity_ci95_low':alo,'action_sensitivity_ci95_high':ahi,'gap':mean(gap_vals),'gap_ci95_low':glo,'gap_ci95_high':ghi})
# strict image-dependent subset: full correct, at least one no-image/wrong-image control fails.
img_dep=[]
for eid in sorted(common):
    f=pairs['full'][eid]; lang=pairs['language_only'][eid]; shuf=pairs['shuffled_image'][eid]
    if f.get('hidden_probe_schema_sensitive') and ((not lang.get('hidden_probe_schema_sensitive')) or (not shuf.get('hidden_probe_schema_sensitive'))):
        rr=dict(f)
        rr['language_only_sensitive']=lang.get('hidden_probe_schema_sensitive')
        rr['shuffled_image_sensitive']=shuf.get('hidden_probe_schema_sensitive')
        rr['image_dependency_type']='full_correct_lang_or_shuffled_wrong'
        img_dep.append(rr)
if img_dep:
    sens,lo,hi,n=summarize_bool(img_dep,'hidden_probe_schema_sensitive')
    act,alo,ahi,an=summarize_bool(img_dep,'openvla_action_sensitive')
    gap_vals=[1.0-(1.0 if r.get('openvla_action_sensitive') else 0.0) for r in img_dep]
    glo,ghi=ci(gap_vals, seed=13)
    rows.append({'condition':'full','subset':'image_dependent_full_correct_control_wrong','n':len(img_dep),'schema_sensitivity':sens,'schema_sensitivity_ci95_low':lo,'schema_sensitivity_ci95_high':hi,'action_sensitivity':act,'action_sensitivity_ci95_low':alo,'action_sensitivity_ci95_high':ahi,'gap':mean(gap_vals),'gap_ci95_low':glo,'gap_ci95_high':ghi})
write_csv(OUT/'pixel_relation_hidden_probe_ci_and_image_dependent_subset.csv', rows)
write_csv(OUT/'pixel_relation_image_dependent_examples.csv', img_dep)

# 2) ActCheck predictive table CI and significance.
act_rows=[]
path=Path('outputs/linguistic_blindness/actcheck_predicts_short_horizon_k20_100/actcheck_rollout_examples.csv')
if path.exists():
    act_rows=list(csv.DictReader(open(path)))

def fval(r,k):
    v=r.get(k)
    if v in (None,'','None'): return None
    return float(v)
summary=[]
for group in ['aligned','wrong-target','ambiguous','overall']:
    g=act_rows if group=='overall' else [r for r in act_rows if r.get('first_action_actcheck_label')==group]
    if not g: continue
    for metric in ['delta_dist_counterfactual','final_target_preference','integrated_target_preference','counterfactual_approach']:
        vals=[]
        for r in g:
            if metric=='counterfactual_approach':
                vals.append(1.0 if str(r.get(metric)).lower()=='true' else 0.0)
            else:
                vals.append(fval(r,metric))
        lo,hi=ci(vals, seed=21)
        summary.append({'group':group,'metric':metric,'n':len([v for v in vals if v is not None]),'mean':mean(vals),'ci95_low':lo,'ci95_high':hi})
# Pairwise tests aligned vs ambiguous and aligned vs wrong-target.
tests=[]
for metric in ['delta_dist_counterfactual','final_target_preference','integrated_target_preference','counterfactual_approach']:
    for a,b in [('aligned','ambiguous'),('aligned','wrong-target'),('wrong-target','ambiguous')]:
        ga=[r for r in act_rows if r.get('first_action_actcheck_label')==a]
        gb=[r for r in act_rows if r.get('first_action_actcheck_label')==b]
        def vals(g):
            if metric=='counterfactual_approach': return [1.0 if str(r.get(metric)).lower()=='true' else 0.0 for r in g]
            return [fval(r,metric) for r in g]
        va=[v for v in vals(ga) if v is not None]; vb=[v for v in vals(gb) if v is not None]
        diff=None if not va or not vb else mean(va)-mean(vb)
        # bootstrap CI for difference
        if va and vb:
            rng=np.random.default_rng(22); boots=[]; aa=np.asarray(va); bb=np.asarray(vb)
            for _ in range(5000):
                boots.append(aa[rng.integers(0,len(aa),len(aa))].mean()-bb[rng.integers(0,len(bb),len(bb))].mean())
            dlo,dhi=float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))
        else: dlo=dhi=None
        tests.append({'metric':metric,'group_a':a,'group_b':b,'n_a':len(va),'n_b':len(vb),'mean_a':mean(va),'mean_b':mean(vb),'difference_a_minus_b':diff,'diff_ci95_low':dlo,'diff_ci95_high':dhi,'permutation_p_two_sided':perm_p(va,vb,seed=23)})
write_csv(OUT/'actcheck_predictive_ci_by_group.csv', summary)
write_csv(OUT/'actcheck_predictive_significance_tests.csv', tests)

md=[]
md.append('# Final sanity CI supplements')
md.append('')
md.append('## Pixel-relation hidden probe: bootstrap CI and image-dependent subset')
for r in rows:
    md.append(f"- {r['condition']} / {r['subset']}: n={r['n']}, schema={r['schema_sensitivity']:.3f} [{r['schema_sensitivity_ci95_low']:.3f}, {r['schema_sensitivity_ci95_high']:.3f}], action={r['action_sensitivity']:.3f} [{r['action_sensitivity_ci95_low']:.3f}, {r['action_sensitivity_ci95_high']:.3f}], gap={r['gap']:.3f} [{r['gap_ci95_low']:.3f}, {r['gap_ci95_high']:.3f}]")
md.append('')
md.append('## ActCheck predictive significance')
for t in tests:
    if t['group_a']=='aligned' and t['group_b']=='ambiguous' and t['metric'] in ['final_target_preference','integrated_target_preference','counterfactual_approach']:
        md.append(f"- {t['metric']}: aligned - ambiguous = {t['difference_a_minus_b']:.4f}, 95% CI [{t['diff_ci95_low']:.4f}, {t['diff_ci95_high']:.4f}], permutation p={t['permutation_p_two_sided']:.4f}")
(OUT/'README.md').write_text('\n'.join(md), encoding='utf-8')
print(OUT)
