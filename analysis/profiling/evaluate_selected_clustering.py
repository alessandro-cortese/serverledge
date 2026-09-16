#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    fields=[]; seen=set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def fval(r,*names,default=math.nan):
    for n in names:
        if r.get(n) not in (None,''): return float(r[n])
    return default


def ival(r,*names,default=0):
    for n in names:
        if r.get(n) not in (None,''): return int(float(r[n]))
    return default


def sval(r,*names,default=''):
    for n in names:
        if r.get(n) not in (None,''): return r[n]
    return default


def same(a,b,tol=1e-9): return abs(a-b)<=tol


def km_metrics(r):
    purity=fval(r,'overall_purity','purity')
    majority=fval(r,'majority_ground_truth_share','majority_class_share','baseline_majority_share')
    gain=fval(r,'purity_gain_over_majority_baseline','purity_gain')
    if math.isnan(gain) and not math.isnan(purity) and not math.isnan(majority): gain=purity-majority
    return dict(purity=purity, majority_baseline=majority, purity_gain=gain,
                homogeneity=fval(r,'homogeneity'), ari=fval(r,'adjusted_rand_index','ari'),
                nmi=fval(r,'normalized_mutual_information','nmi'))


def db_metrics(r):
    purity=fval(r,'overall_purity_clustered')
    majority=fval(r,'clustered_majority_ground_truth_share')
    gain=fval(r,'purity_gain_over_clustered_majority_baseline')
    if math.isnan(gain) and not math.isnan(purity) and not math.isnan(majority): gain=purity-majority
    return dict(purity=purity, majority_baseline=majority, purity_gain=gain,
                homogeneity=fval(r,'homogeneity_clustered'), ari=fval(r,'adjusted_rand_index_clustered'),
                nmi=fval(r,'normalized_mutual_information_clustered'))


def plot_metric(rows, algorithm, key, label, out):
    groups=defaultdict(list)
    for r in rows:
        if r['algorithm']==algorithm: groups[r['configuration_id']].append(r)
    fig, ax = plt.subplots(figsize=(10,6))
    for cid, vals in sorted(groups.items()):
        vals=sorted(vals,key=lambda x: float(x['threshold_percent']))
        ax.plot([float(x['threshold_percent']) for x in vals], [float(x[key]) for x in vals], marker='o', label=cid)
    ax.set_xlabel('Ground-truth threshold (%)'); ax.set_ylabel(label); ax.set_title(f'{algorithm}: {label} sensitivity')
    ax.grid(True, alpha=.25); ax.legend(fontsize=7,ncol=2); fig.tight_layout()
    fig.savefig(out.with_suffix('.png'), dpi=180); fig.savefig(out.with_suffix('.svg')); plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True,type=Path)
    ap.add_argument('--selection',required=True,type=Path)
    ap.add_argument('--output-dir',required=True,type=Path)
    a=ap.parse_args()
    root=a.root.resolve(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True); figs=out/'figures'; figs.mkdir(exist_ok=True)
    sel=json.loads(a.selection.read_text(encoding='utf-8'))
    (out/'selected-configurations-frozen.json').write_text(json.dumps(sel,indent=2)+'\n',encoding='utf-8')
    internal=[]; gt=[]

    km_dir=root/sel['kmeans']['source_dir']; km_i=read_csv(km_dir/'kmeans-internal-summary.csv'); km_g=read_csv(km_dir/'kmeans-ground-truth-summary.csv')
    for cfg in sel['kmeans']['configurations']:
        matches=[r for r in km_i if r.get('feature_set')==cfg['feature_set'] and ival(r,'k')==cfg['k']]
        if len(matches)!=1: raise RuntimeError(f'KMeans match error for {cfg}: {len(matches)}')
        r=matches[0]; sizes=[int(x) for x in sval(r,'cluster_size_distribution').split('|') if x]; n=sum(sizes)
        internal.append(dict(configuration_id=cfg['id'],algorithm='kmeans',role=cfg['role'],feature_set=cfg['feature_set'],scaler=sel['kmeans']['scaler'],metric='euclidean',k=cfg['k'],min_samples='',eps_quantile='',eps='',cluster_count=cfg['k'],noise_count=0,coverage=1.0,silhouette=fval(r,'silhouette'),davies_bouldin=fval(r,'davies_bouldin'),calinski_harabasz=fval(r,'calinski_harabasz'),singleton_count=ival(r,'singleton_count'),cluster_size_distribution=sval(r,'cluster_size_distribution'),max_cluster_share=max(sizes)/n if n else math.nan))
        for e in km_g:
            if e.get('feature_set')==cfg['feature_set'] and ival(e,'k')==cfg['k']:
                gt.append(dict(configuration_id=cfg['id'],algorithm='kmeans',role=cfg['role'],feature_set=cfg['feature_set'],scaler=sel['kmeans']['scaler'],metric='euclidean',k=cfg['k'],min_samples='',eps_quantile='',threshold_percent=fval(e,'threshold_percent'),coverage=1.0,**km_metrics(e)))

    db_dir=root/sel['dbscan']['source_dir']; db_i=read_csv(db_dir/'dbscan-internal-summary.csv'); db_g=read_csv(db_dir/'dbscan-ground-truth-summary.csv'); db_n=read_csv(db_dir/'dbscan-noise-functions.csv'); db_c=read_csv(db_dir/'dbscan-cluster-composition.csv')
    source_ids={}
    for cfg in sel['dbscan']['configurations']:
        matches=[r for r in db_i if r.get('feature_set')==cfg['feature_set'] and r.get('metric')==sel['dbscan']['metric'] and ival(r,'min_samples')==cfg['min_samples'] and same(fval(r,'eps_quantile'),cfg['eps_quantile'])]
        if len(matches)!=1: raise RuntimeError(f'DBSCAN match error for {cfg}: {len(matches)}')
        r=matches[0]; sid=r['configuration_id']; source_ids[cfg['id']]=sid; clustered=ival(r,'sample_count')-ival(r,'noise_count')
        internal.append(dict(configuration_id=cfg['id'],algorithm='dbscan',role=cfg['role'],feature_set=cfg['feature_set'],scaler=sel['dbscan']['scaler'],metric=sel['dbscan']['metric'],k='',min_samples=cfg['min_samples'],eps_quantile=cfg['eps_quantile'],eps=fval(r,'eps'),cluster_count=ival(r,'cluster_count'),noise_count=ival(r,'noise_count'),coverage=fval(r,'coverage'),silhouette=fval(r,'silhouette_clustered'),davies_bouldin='',calinski_harabasz='',singleton_count=ival(r,'singleton_count'),cluster_size_distribution=sval(r,'cluster_size_distribution'),max_cluster_share=ival(r,'max_cluster_size')/clustered if clustered else math.nan))
        for e in db_g:
            if e.get('configuration_id')==sid:
                gt.append(dict(configuration_id=cfg['id'],algorithm='dbscan',role=cfg['role'],feature_set=cfg['feature_set'],scaler=sel['dbscan']['scaler'],metric=sel['dbscan']['metric'],k='',min_samples=cfg['min_samples'],eps_quantile=cfg['eps_quantile'],threshold_percent=fval(e,'threshold_percent'),coverage=fval(e,'coverage'),**db_metrics(e)))

    noise=[]
    for fid,sid in source_ids.items():
        for r in db_n:
            if r.get('configuration_id')==sid: noise.append({'configuration_id':fid,'function_name':r['function_name']})
    tau=float(sel['primary_reporting_threshold_percent']); comp=[]
    for fid,sid in source_ids.items():
        for r in db_c:
            if r.get('configuration_id')==sid and same(fval(r,'threshold_percent'),tau): comp.append({'configuration_id':fid,**r})

    internal=sorted(internal,key=lambda r:(r['algorithm'],r['configuration_id'])); gt=sorted(gt,key=lambda r:(r['algorithm'],r['configuration_id'],r['threshold_percent'])); noise=sorted(noise,key=lambda r:(r['configuration_id'],r['function_name']))
    write_csv(out/'selected-internal-summary.csv',internal); write_csv(out/'selected-ground-truth-summary.csv',gt); write_csv(out/'selected-dbscan-noise-functions.csv',noise); write_csv(out/'selected-dbscan-composition-t15.csv',comp)
    for alg in ('kmeans','dbscan'):
        for key,label in [('purity_gain','Purity gain over majority baseline'),('homogeneity','Homogeneity'),('ari','Adjusted Rand Index'),('nmi','Normalized Mutual Information')]: plot_metric(gt,alg,key,label,figs/f'{alg}-{key}-threshold-sensitivity')

    primary=[r for r in gt if same(float(r['threshold_percent']),tau)]
    md=['# Frozen clustering shortlist: ground-truth evaluation','', 'The shortlist in `selected-configurations-frozen.json` was fixed **before** inspecting external ground-truth metrics. Ground truth is used only for post-hoc evaluation.','', '## Frozen internal configurations','', '| ID | Algorithm | Feature set | Scaler | Parameters | Coverage | Silhouette | Cluster sizes | Max cluster share |','|---|---|---|---|---|---:|---:|---|---:|']
    for r in internal:
        params=f'K={r["k"]}' if r['algorithm']=='kmeans' else f'{r["metric"]}, min_samples={r["min_samples"]}, q={float(r["eps_quantile"]):.2f}, eps={float(r["eps"]):.6g}'
        md.append(f'| `{r["configuration_id"]}` | {r["algorithm"]} | `{r["feature_set"]}` | {r["scaler"]} | {params} | {float(r["coverage"]):.4f} | {float(r["silhouette"]):.4f} | {r["cluster_size_distribution"]} | {float(r["max_cluster_share"]):.4f} |')
    md += ['', f'## External evaluation at τ = {tau:g}%', '', '| ID | Coverage | Purity | Majority baseline | Purity gain | Homogeneity | ARI | NMI |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in primary:
        md.append(f'| `{r["configuration_id"]}` | {float(r["coverage"]):.4f} | {float(r["purity"]):.4f} | {float(r["majority_baseline"]):.4f} | {float(r["purity_gain"]):+.4f} | {float(r["homogeneity"]):.4f} | {float(r["ari"]):.4f} | {float(r["nmi"]):.4f} |')
    md += ['', '## DBSCAN noise functions','']
    by=defaultdict(list)
    for r in noise: by[r['configuration_id']].append(r['function_name'])
    for cid in sorted(source_ids): md.append(f'- `{cid}`: {", ".join(by[cid]) if by[cid] else "none"}')
    md += ['', '## Evidence files','', '- `selected-configurations-frozen.json`','- `selected-internal-summary.csv`','- `selected-ground-truth-summary.csv`','- `selected-dbscan-noise-functions.csv`','- `selected-dbscan-composition-t15.csv`','- `figures/*-threshold-sensitivity.{png,svg}`','']
    (out/'GROUND_TRUTH_EVALUATION.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(f'selected_internal={len(internal)} selected_ground_truth_rows={len(gt)} selected_noise_rows={len(noise)} output={out}')

if __name__=='__main__': main()
