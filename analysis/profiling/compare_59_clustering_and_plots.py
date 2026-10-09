#!/usr/bin/env python3
"""Reproducible same-59 Serverledge clustering/donor/language comparison.

Read local final-corpus64 artifacts; never infer results from historical slides.
LOFO fits train-only scaler and KMeans. Figures are FULL-CORPUS descriptive fits
and are not substituted for LOFO performance estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import MinMaxScaler

PAPER5 = ['page_faults_delta', 'utilized_cpus', 'free_memory_mb',
          'cpu_user_delta_ms', 'cpu_kernel_delta_ms']
STATIC2 = ['static_token_count_mean', 'static_function_count']
SEEDS = (11, 23, 37, 41, 53)
PREF_ORDER = ('x86-preferred', 'architecture-independent', 'arm-preferred')
DIR = ('x86-preferred', 'arm-preferred')
LANGS = ('go', 'python', 'nodejs')
CONFIGS = {
    'A_legacy_K5': {'kind': 'legacy', 'k': 5, 'donor': 'nearest', 'lang_weight': 0.0},
    'B_hybrid_K6': {'kind': 'hybrid', 'k': 6, 'donor': 'nearest', 'lang_weight': 0.0},
    'C_final_weighted': {'kind': 'hybrid', 'k': 6, 'donor': 'weighted', 'lang_weight': 0.0},
    'C_lang025': {'kind': 'hybrid', 'k': 6, 'donor': 'weighted', 'lang_weight': 0.25},
    'C_lang100': {'kind': 'hybrid', 'k': 6, 'donor': 'weighted', 'lang_weight': 1.0},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_unique(path, needed):
    if not path.is_file():
        raise SystemExit(f'Missing input: {path}')
    df = pd.read_csv(path)
    missing = set(needed) - set(df.columns)
    if missing:
        raise SystemExit(f'{path}: missing {sorted(missing)}')
    if df['function_name'].duplicated().any():
        raise SystemExit(f'{path}: duplicate function_name')
    return df


def prepare(root):
    paths = {
        'profiles': root / 'resource/x86/function-profiles-median.csv',
        'static': root / 'static-code-metrics.csv',
        'language': root / 'function-languages.csv',
        'pref_2p5': root / 'ground_truth/preferences-2p5.csv',
        'pref_15': root / 'ground_truth/preferences-15.csv',
    }
    prof = load_unique(paths['profiles'], ['function_name', *PAPER5])
    static = load_unique(paths['static'], ['function_name', 'static_source_available', *STATIC2])
    lang = load_unique(paths['language'], ['function_name', 'language'])
    p25 = load_unique(paths['pref_2p5'], ['function_name', 'architecture_preference', 'x86_duration_ms', 'arm_duration_ms'])
    p15 = load_unique(paths['pref_15'], ['function_name', 'architecture_preference'])
    df = (prof[['function_name', *PAPER5]]
          .merge(static[['function_name','static_source_available',*STATIC2]], on='function_name', validate='one_to_one')
          .merge(lang[['function_name','language']], on='function_name', validate='one_to_one')
          .merge(p25[['function_name','architecture_preference','x86_duration_ms','arm_duration_ms']], on='function_name', validate='one_to_one')
          .merge(p15[['function_name','architecture_preference']].rename(columns={'architecture_preference':'preference_15'}), on='function_name', validate='one_to_one'))
    ok = df['static_source_available'].astype(str).str.lower().str.strip().isin({'true', '1', 'yes'})
    for field in [*PAPER5, *STATIC2, 'x86_duration_ms','arm_duration_ms']:
        df[field] = pd.to_numeric(df[field], errors='coerce')
        ok &= np.isfinite(df[field].to_numpy(float))
    df = df.loc[ok].copy().sort_values('function_name').reset_index(drop=True)
    if len(df) != 59:
        raise SystemExit(f'Expected exactly 59 eligible functions, got {len(df)}. Check root and static eligibility.')
    if not (df[['x86_duration_ms', 'arm_duration_ms']] > 0).all().all():
        raise SystemExit('Invalid nonpositive duration')
    if not set(df['language']).issubset(set(LANGS)):
        raise SystemExit(f'Unexpected language values: {sorted(set(df.language))}')
    for field in ('architecture_preference', 'preference_15'):
        if not set(df[field]).issubset(set(PREF_ORDER)):
            raise SystemExit(f'Unexpected preference labels in {field}: {sorted(set(df[field]))}')
    return df, paths


def features(train, target, kind, lang_weight):
    def block(cols):
        scaler = MinMaxScaler()
        a = scaler.fit_transform(train[cols].to_numpy(float))
        b = scaler.transform(target[cols].to_numpy(float))
        return a, b
    x, xt = block(PAPER5)
    if kind == 'hybrid':
        y, yt = block(STATIC2)
        x = np.column_stack((x/math.sqrt(len(PAPER5)), y/math.sqrt(len(STATIC2))))
        xt = np.column_stack((xt/math.sqrt(len(PAPER5)), yt/math.sqrt(len(STATIC2))))
    elif kind != 'legacy':
        raise ValueError(kind)
    if lang_weight > 0:
        def onehot(frame):
            return np.column_stack([(frame['language'].to_numpy() == c).astype(float) for c in LANGS])
        x = np.column_stack((x, onehot(train)*lang_weight/math.sqrt(len(LANGS))))
        xt = np.column_stack((xt, onehot(target)*lang_weight/math.sqrt(len(LANGS))))
    return x, xt


def raw_pref(row):
    return 'x86-preferred' if row['x86_duration_ms'] <= row['arm_duration_ms'] else 'arm-preferred'


def choose(train, target, cluster_labels, target_cluster, x, xt, kind):
    same = np.flatnonzero(cluster_labels == target_cluster)
    if len(same) == 0:
        return None, '', 'no_cluster_member'
    names = train['function_name'].astype(str).to_numpy()
    distances = np.sum(np.abs(x-xt[0]), axis=1)
    prefs = train['architecture_preference'].astype(str).to_numpy()
    if kind == 'nearest':
        candid = same
        predicted = ''
    elif kind == 'weighted':
        candid_dir = same[np.isin(prefs[same], DIR)]
        if not len(candid_dir):
            return None, '', 'no_directional_member'
        weights = 1.0/np.power(distances[candid_dir] + 1e-9, 2)
        scores = [float(weights[prefs[candid_dir] == label].sum()) for label in DIR]
        if abs(scores[0]-scores[1]) <= 1e-15:
            return None, '', 'tied_vote'
        predicted = DIR[int(scores[1] > scores[0])]
        candid = same[prefs[same] == predicted]
    else:
        raise ValueError(kind)
    order = np.lexsort((names[candid], distances[candid]))
    selected_idx = int(candid[order[0]])
    if kind == 'nearest':
        predicted = raw_pref(train.iloc[selected_idx])
    return selected_idx, predicted, 'selected'


def one_outcome(target, donor, pred):
    tx, ta = float(target.x86_duration_ms), float(target.arm_duration_ms)
    dx, da = float(donor.x86_duration_ms), float(donor.arm_duration_ms)
    best = min(tx,ta)
    chosen = tx if pred == 'x86-preferred' else ta
    alternative = ta if pred == 'x86-preferred' else tx
    return dict(
        raw_agreement=float(raw_pref(donor) == raw_pref(target)),
        directional_agreement=(float(pred == target.architecture_preference)
                               if target.architecture_preference in DIR else np.nan),
        is_directional=bool(target.architecture_preference in DIR),
        regret_percent=(chosen/best-1)*100,
        realized_speedup=alternative/chosen,
        multiplicative_mismatch=math.exp(abs(math.log(ta/tx)-math.log(da/dx))),
        same_language=float(donor.language == target.language),
    )


def lofo(df, seeds):
    result = []
    for seed in seeds:
        print(f'LOFO seed {seed}...', flush=True)
        for i in range(len(df)):
            target = df.iloc[[i]]
            train = df.drop(index=i).reset_index(drop=True)
            cache = {}
            for name, cfg in CONFIGS.items():
                geom = (cfg['kind'], cfg['k'], cfg['lang_weight'])
                if geom not in cache:
                    x, xt = features(train, target, cfg['kind'], cfg['lang_weight'])
                    model = KMeans(n_clusters=cfg['k'], n_init=50, random_state=seed)
                    cluster_labels = model.fit_predict(x)
                    target_cluster = int(model.predict(xt)[0])
                    cache[geom] = x, xt, cluster_labels, target_cluster
                x, xt, labels, tc = cache[geom]
                donor_i, pred, status = choose(train, target, labels, tc, x, xt, cfg['donor'])
                base = dict(config=name, seed=seed, target_function=str(target.iloc[0].function_name),
                            target_language=str(target.iloc[0].language), target_preference=str(target.iloc[0].architecture_preference),
                            status=status, donor_function='', prediction=pred)
                if donor_i is not None:
                    donor = train.iloc[donor_i]
                    base['donor_function'] = str(donor.function_name)
                    base.update(one_outcome(target.iloc[0], donor, pred))
                result.append(base)
    data = pd.DataFrame(result)
    expected = 59*len(seeds)*len(CONFIGS)
    if len(data) != expected or data.duplicated(['config','seed','target_function']).any():
        raise AssertionError(f'Invalid LOFO rows: {len(data)} != {expected}')
    if ((data['donor_function'] == data['target_function']) & (data.status == 'selected')).any():
        raise AssertionError('Target leaked into donor')
    return data


def metrics(data):
    result=[]
    for (cfg,seed),g in data.groupby(['config','seed'],sort=False):
        selected = g[g.status == 'selected']
        directed = selected[selected.is_directional == True]
        all_dir = int((g.target_preference.isin(DIR)).sum())
        result.append(dict(config=cfg, seed=int(seed), target_count=len(g), selection_coverage=len(selected)/len(g),
                           directional_coverage=len(directed)/all_dir if all_dir else np.nan,
                           raw_agreement=float(selected.raw_agreement.mean()),
                           directional_agreement=float(directed.directional_agreement.mean()),
                           geomean_speedup=float(np.exp(np.log(directed.realized_speedup).mean())) if len(directed) else np.nan,
                           regret_percent=float(directed.regret_percent.mean()),
                           geomean_mismatch=float(np.exp(np.log(selected.multiplicative_mismatch).mean())) if len(selected) else np.nan,
                           same_language=float(selected.same_language.mean())))
    byseed = pd.DataFrame(result)
    summary = byseed.groupby('config',as_index=False).agg(
        seeds=('seed','nunique'),selection_coverage=('selection_coverage','mean'),
        directional_coverage=('directional_coverage','mean'),raw_agreement=('raw_agreement','mean'),
        directional_agreement=('directional_agreement','mean'),geomean_speedup=('geomean_speedup','mean'),
        regret_percent=('regret_percent','mean'),geomean_mismatch=('geomean_mismatch','mean'),
        same_language=('same_language','mean'))
    comparisons=[]
    for a,b in [('A_legacy_K5','B_hybrid_K6'),('B_hybrid_K6','C_final_weighted'),
                ('A_legacy_K5','C_final_weighted'),('C_final_weighted','C_lang025'),('C_final_weighted','C_lang100')]:
        aa=byseed[byseed.config==a].set_index('seed'); bb=byseed[byseed.config==b].set_index('seed')
        item={'comparison':f'{b}_minus_{a}'}
        for metric in ['selection_coverage','directional_coverage','raw_agreement','directional_agreement','geomean_speedup','regret_percent','same_language']:
            item['delta_'+metric]=float((bb[metric]-aa[metric]).mean())
        comparisons.append(item)
    return byseed,summary,pd.DataFrame(comparisons)


def plot_panel(ax, coords, labels, cats, title):
    for category in cats:
        idx=np.asarray(labels)==category
        if idx.any():
            ax.scatter(coords[idx,0],coords[idx,1], label=f'{category} ({idx.sum()})', s=36, alpha=.82)
    ax.set_title(title,fontsize=11)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(alpha=.15)
    ax.legend(fontsize=7, loc='best', framealpha=.9)


def figure_pair(pca, umap_coords, labs, cats, title, outfile):
    fig, axes=plt.subplots(1,2,figsize=(13.3,5.6),layout='constrained')
    plot_panel(axes[0],pca,labs,cats,'PCA')
    plot_panel(axes[1],umap_coords,labs,cats,'UMAP')
    fig.suptitle(title,fontsize=14)
    fig.savefig(outfile,dpi=190)
    fig.savefig(outfile.with_suffix('.svg'))
    plt.close(fig)


def plot_full(df,out,seed,umap_neighbors):
    try:
        import umap
    except ImportError as exc:
        raise SystemExit('UMAP unavailable: install umap-learn in .venv-analysis') from exc
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in ['A_legacy_K5','C_final_weighted','C_lang100']:
        cfg=CONFIGS[name]
        x,_=features(df,df,cfg['kind'],cfg['lang_weight'])
        lab=KMeans(n_clusters=cfg['k'],n_init=50,random_state=seed).fit_predict(x)
        pca=PCA(n_components=2,random_state=seed).fit_transform(x)
        um=umap.UMAP(n_components=2,n_neighbors=umap_neighbors,min_dist=.1,random_state=seed).fit_transform(x)
        cluster_names=np.asarray([f'cluster {k}' for k in lab])
        cats=[f'cluster {k}' for k in sorted(np.unique(lab))]
        figure_pair(pca,um,cluster_names,cats,f'{name} | 59 funzioni | fit completo descrittivo',out/f'{name}_clusters_pca_umap.png')
        figure_pair(pca,um,df['language'].to_numpy(),LANGS,
                    f'{name} | Linguaggio post-hoc | n=59',
                    out/f'{name}_language_pca_umap.png')
        for threshold, field in [('2p5','architecture_preference'),('15','preference_15')]:
            figure_pair(pca,um,df[field].to_numpy(),PREF_ORDER,
                        f'{name} | Ground truth post-hoc tau={threshold.replace("p", ".")}% | n=59',
                        out/f'{name}_groundtruth_t{threshold}_pca_umap.png')
        sizes=pd.Series(lab).value_counts()
        rows.append(dict(config=name,seed=seed,n=59,k=cfg['k'],silhouette=silhouette_score(x,lab),
                         davies_bouldin=davies_bouldin_score(x,lab),calinski_harabasz=calinski_harabasz_score(x,lab),
                         max_cluster_size=int(sizes.max()),singleton_clusters=int((sizes==1).sum()),
                         explained_variance_pca_2=float(PCA(n_components=2).fit(x).explained_variance_ratio_.sum()),
                         language_nmi=normalized_mutual_info_score(df.language,lab),
                         preference_nmi_t2p5=normalized_mutual_info_score(df.architecture_preference,lab)))
        pd.DataFrame({'function_name':df.function_name,'config':name,'cluster':lab,'language':df.language,
                      'preference_2p5':df.architecture_preference,'preference_15':df.preference_15,
                      'pca1':pca[:,0],'pca2':pca[:,1],'umap1':um[:,0],'umap2':um[:,1]}).to_csv(out/f'{name}_fullfit_coordinates.csv',index=False)
    return pd.DataFrame(rows)


def report(df,summary,deltas,cluster,out,paths,seeds):
    def f(name,col):return float(summary.set_index('config').loc[name,col])
    lines=['# Serverledge: confronto controllato sulle stesse 59 funzioni', '',
           '**Ambito:** solo clustering, donor selection e UCB1 (questo script misura le prime due fasi).',
           '', '## Controlli sperimentali',
           '- Corpus unico: 59 funzioni static-eligible; le altre 5 del corpus64 sono escluse **da entrambi** i lati.',
           f'- Seed LOFO: {", ".join(map(str,seeds))}; KMeans `n_init=50`, scaling train-only.',
           '- Architettura di riferimento x86. I profili ARM non sono feature di clustering.',
           '- Baseline A: PAPER5 MinMax K5 + nearest same-cluster Manhattan, senza filtro direzionale.',
           '- B: PAPER5 + token_count_mean + function_count, MinMax per blocco, K6, stesso donor di A.',
           '- C: stessa rappresentazione di B, weighted vote direzionale inverse-Manhattan 1/d², nearest nello stesso cluster e classe.',
           '- Linguaggio: C + one-hot con peso relativo 0.25 o 1.0, tutto il resto identico.',
           '- Le etichette target sono usate solo a posteriori; le preferenze del catalogo sono disponibili per il voto dei donor.',
           '- Le metriche condizionate ai target direzionali e la coverage vanno lette insieme; le astensioni non sono scartate dai futuri replay TL.',
           '', '## Sintesi donor LOFO (media sui seed)',
           '', '| Caso | Coverage | Raw agreement | Agreement direzionale | Speedup geo | Regret medio |',
           '|---|---:|---:|---:|---:|---:|']
    for _,r in summary.iterrows():
        lines.append(f"| {r['config']} | {r['selection_coverage']:.2%} | {r['raw_agreement']:.2%} | {r['directional_agreement']:.2%} | {r['geomean_speedup']:.4f}x | {r['regret_percent']:.3f}% |")
    lines+=['','## Effetto isolato dei cambiamenti','','| Confronto | Delta raw (pp) | Delta direzionale (pp) | Delta regret (pp) |','|---|---:|---:|---:|']
    for _,r in deltas.iterrows():
        lines.append(f"| {r['comparison']} | {100*r['delta_raw_agreement']:+.2f} | {100*r['delta_directional_agreement']:+.2f} | {r['delta_regret_percent']:+.3f} |")
    lines+=['','**Attenzione:** accordo direzionale maggiore non garantisce minore regret. Confrontare anche coverage, speedup e confronto downstream.',
            '','## Lingua (ablation corretta)','',
            'Le vecchie prove STATIC4+linguaggio con K diverso sono diagnostica preliminare, NON una dimostrazione isolata del beneficio nella pipeline finale.',
            'Qui confrontiamo C contro C_lang025 e C_lang100 a K6 e weighted vote invariati.',
            'Non affermare "il linguaggio non aiuta" senza verificare i delta su donor e replay UCB1.',
            '','## PCA / UMAP e ground truth',
            '- Per A, C e C+linguaggio vengono prodotti clustering, lingua e ground truth tau=2.5%, 15%, ciascuno con PCA e UMAP affiancati.',
            '- **Figure descrittive**: KMeans fit su tutte le 59 funzioni; **misure donor**: LOFO rigoroso. Non equiparare i cluster ID full-fit ai LOFO.',
            '- Le coordinate PCA e UMAP sono calcolate separatamente negli spazi feature A e C; **non sovrapporre geometricamente** le coordinate di metodi diversi.',
            '','## Formula e miglioramento downstream',
            'Il confronto delle formule UCB1 deve usare le stesse mappature donor A/B/C, lo stesso bootstrap target x86 e gli stessi stream CRN.',
            'Tale prova è separata e viene prodotta dallo script `compare_59_ucb1_replay.py`.',
            '','## Provenienza', '']
    for k,p in paths.items():lines.append(f'- `{k}`: `{p}`; SHA256 `{sha(p)}`')
    (out/'CONFRONTO_59_METODOLOGIA_RISULTATI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--seeds',default='11,23,37,41,53')
    ap.add_argument('--plot-seed',type=int,default=42)
    ap.add_argument('--umap-neighbors',type=int,default=10)
    ap.add_argument('--skip-plots',action='store_true')
    args=ap.parse_args()
    seeds=tuple(int(x) for x in args.seeds.split(',') if x.strip())
    root=args.root.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    df,paths=prepare(root)
    df[['function_name','language','architecture_preference','preference_15']].to_csv(out/'eligible_59_functions.csv',index=False)
    data=lofo(df,seeds)
    data.to_csv(out/'lofo_59_per_target_seed.csv',index=False)
    byseed,summary,deltas=metrics(data)
    byseed.to_csv(out/'lofo_59_per_seed.csv',index=False)
    summary.to_csv(out/'lofo_59_summary.csv',index=False)
    deltas.to_csv(out/'lofo_59_comparisons.csv',index=False)
    cluster = pd.DataFrame()
    if not args.skip_plots:
        cluster=plot_full(df,out/'figures',args.plot_seed,args.umap_neighbors)
        cluster.to_csv(out/'fullfit_cluster_metrics.csv',index=False)
    report(df,summary,deltas,cluster,out,paths,seeds)
    (out/'manifest.json').write_text(json.dumps({'eligible':59,'seeds':seeds,'plot_seed':args.plot_seed,
             'plots':not args.skip_plots,'thresholds':[2.5,15.0],
             'inputs':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},
             'source':'compare_59_clustering_and_plots.py','target_leakage':'LOFO exclusion'},indent=2)+'\n')
    print('PASS: 59 targets; LOFO comparison generated')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('REPORT:',out/'CONFRONTO_59_METODOLOGIA_RISULTATI.md')

if __name__=='__main__':
    main()
