#!/usr/bin/env python3
"""Nested LOFO validation for clustering-as-classification after an exploratory sweep."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

LABELS = ["x86-preferred", "architecture-independent", "arm-preferred"]
DIRECTIONAL = {"x86-preferred", "arm-preferred"}
SCALERS = {"minmax": MinMaxScaler, "robust": RobustScaler, "standard": StandardScaler}


@dataclass(frozen=True)
class Candidate:
    name: str
    variant: str
    scaler: str
    k: int
    include_language: bool
    language_weight: float = 0.0


def threshold_slug(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v).replace(".", "p")


def load_data(profiles, preferences_dir, threshold, language_map):
    prof = pd.read_csv(profiles)
    pref = pd.read_csv(preferences_dir / f"preferences-{threshold_slug(threshold)}.csv")
    df = prof.merge(pref[["function_name","architecture_preference"]], on="function_name", validate="one_to_one")
    if language_map:
        lang = pd.read_csv(language_map)[["function_name","language"]]
        if lang["language"].isna().any():
            raise ValueError("language map contains missing values")
        df = df.merge(lang, on="function_name", validate="one_to_one")
    return df.sort_values("function_name").reset_index(drop=True)


def representation(frame, variant):
    pf = frame["page_faults_delta"].astype(float).to_numpy()
    cpu = frame["utilized_cpus"].astype(float).to_numpy()
    free = frame["free_memory_mb"].astype(float).to_numpy()
    user = frame["cpu_user_delta_ms"].astype(float).to_numpy()
    kernel = frame["cpu_kernel_delta_ms"].astype(float).to_numpy()
    total = user + kernel
    if variant == "paper5_raw":
        return np.column_stack([pf,cpu,free,user,kernel])
    if variant == "paper5_log":
        return np.column_stack([np.log1p(np.clip(pf,0,None)),cpu,free,np.log1p(np.clip(user,0,None)),np.log1p(np.clip(kernel,0,None))])
    if variant == "paper5_no_free_log":
        return np.column_stack([np.log1p(np.clip(pf,0,None)),cpu,np.log1p(np.clip(user,0,None)),np.log1p(np.clip(kernel,0,None))])
    if variant == "behavioral4":
        return np.column_stack([np.log1p(np.clip(pf,0,None)),cpu,np.log1p(np.clip(total,0,None)),kernel/np.maximum(total,1e-12)])
    raise ValueError(f"unknown variant {variant}")


def add_language(train_x, target_x, train_lang, target_lang, weight):
    vocab = sorted(train_lang.astype(str).unique())
    idx = {v:i for i,v in enumerate(vocab)}
    a = np.zeros((len(train_lang),len(vocab)), dtype=float)
    b = np.zeros((1,len(vocab)), dtype=float)
    for row,value in enumerate(train_lang.astype(str)):
        a[row,idx[value]] = weight
    if str(target_lang) in idx:
        b[0,idx[str(target_lang)]] = weight
    return np.hstack([train_x,a]), np.hstack([target_x,b])


def majority_label(values):
    counts = Counter(values.tolist())
    maximum = max(counts.values())
    winners = [label for label in LABELS if counts.get(label,0)==maximum]
    return winners[0] if len(winners)==1 else "abstain"


def fit_predict_one(train, target, candidate, n_init, random_state):
    raw_train = representation(train,candidate.variant)
    raw_target = representation(target,candidate.variant)
    scaler = SCALERS[candidate.scaler]()
    x_train = scaler.fit_transform(raw_train)
    x_target = scaler.transform(raw_target)
    if candidate.include_language:
        x_train,x_target = add_language(x_train,x_target,train["language"],str(target.iloc[0]["language"]),candidate.language_weight)
    km = KMeans(n_clusters=candidate.k,n_init=n_init,random_state=random_state)
    clusters = km.fit_predict(x_train)
    target_cluster = int(km.predict(x_target)[0])
    members = np.flatnonzero(clusters==target_cluster)
    return majority_label(train.iloc[members]["architecture_preference"]), target_cluster, len(members)


def metrics(y,p):
    recalls,f1s = [],[]
    for label in LABELS:
        truth = y==label
        pred = p==label
        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))
        rec = tp/(tp+fn) if tp+fn else 0.0
        prec = tp/(tp+fp) if tp+fp else 0.0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        recalls.append(rec); f1s.append(f1)
    directional = np.isin(y,list(DIRECTIONAL))
    return {
        "strict_accuracy": float(np.mean(y==p)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "directional_accuracy": float(np.mean(y[directional]==p[directional])) if np.any(directional) else 0.0,
        "coverage": float(np.mean(p!="abstain"))
    }


def inner_score(outer_train,candidate,n_init,random_state):
    truth,pred = [],[]
    for pos in range(len(outer_train)):
        inner_target = outer_train.iloc[[pos]]
        inner_train = outer_train.drop(index=outer_train.index[pos])
        prediction,_,_ = fit_predict_one(inner_train,inner_target,candidate,n_init,random_state)
        truth.append(str(inner_target.iloc[0]["architecture_preference"]))
        pred.append(prediction)
    return metrics(np.asarray(truth,dtype=object),np.asarray(pred,dtype=object))


def parse_candidates(path, language_available):
    if path:
        rows = json.loads(path.read_text(encoding="utf-8"))
        candidates = [Candidate(**row) for row in rows]
    else:
        candidates = [
            Candidate("baseline-paper5-minmax-k5","paper5_raw","minmax",5,False,0.0),
            Candidate("paper5-log-robust-k7","paper5_log","robust",7,False,0.0),
            Candidate("paper5-no-free-log-robust-k6","paper5_no_free_log","robust",6,False,0.0),
            Candidate("behavioral4-robust-k7","behavioral4","robust",7,False,0.0),
            Candidate("behavioral4-robust-k8","behavioral4","robust",8,False,0.0),
            Candidate("behavioral4-robust-k9","behavioral4","robust",9,False,0.0),
            Candidate("behavioral4-robust-k10","behavioral4","robust",10,False,0.0),
        ]
        if language_available:
            candidates += [
                Candidate("behavioral4-robust-k8-lang025","behavioral4","robust",8,True,0.25),
                Candidate("behavioral4-robust-k9-lang025","behavioral4","robust",9,True,0.25),
                Candidate("behavioral4-robust-k10-lang025","behavioral4","robust",10,True,0.25),
            ]
    return candidates


def selection_key(row):
    return (row["balanced_accuracy"],row["macro_f1"],row["directional_accuracy"],row["strict_accuracy"],row["coverage"])


def run(args):
    out = args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    df = load_data(args.profiles.resolve(),args.preferences_dir.resolve(),args.threshold,args.language_map.resolve() if args.language_map else None)
    candidates = parse_candidates(args.candidates_json.resolve() if args.candidates_json else None,args.language_map is not None)

    outer_rows,inner_rows = [],[]
    for outer_pos in range(len(df)):
        outer_target = df.iloc[[outer_pos]]
        outer_train = df.drop(index=df.index[outer_pos])
        scored = []
        for candidate in candidates:
            score = inner_score(outer_train,candidate,args.n_init,args.random_state)
            scored.append((candidate,score))
            inner_rows.append({"outer_function":str(outer_target.iloc[0]["function_name"]),"candidate_name":candidate.name,**score})
        selected,selected_score = max(scored,key=lambda item: selection_key(item[1]))
        prediction,cluster,cluster_size = fit_predict_one(outer_train,outer_target,selected,args.n_init,args.random_state)
        outer_rows.append({
            "function_name":str(outer_target.iloc[0]["function_name"]),
            "true_label":str(outer_target.iloc[0]["architecture_preference"]),
            "predicted_label":prediction,
            "selected_candidate":selected.name,
            "selected_variant":selected.variant,
            "selected_scaler":selected.scaler,
            "selected_k":selected.k,
            "selected_include_language":selected.include_language,
            "selected_language_weight":selected.language_weight,
            "inner_balanced_accuracy":selected_score["balanced_accuracy"],
            "inner_macro_f1":selected_score["macro_f1"],
            "inner_directional_accuracy":selected_score["directional_accuracy"],
            "outer_target_cluster":cluster,
            "outer_cluster_size":cluster_size,
        })
        print(f"outer={outer_pos+1:02d}/{len(df)} target={outer_rows[-1]['function_name']:<28} selected={selected.name:<38} pred={prediction}")

    odf = pd.DataFrame(outer_rows)
    idf = pd.DataFrame(inner_rows)
    odf.to_csv(out/"nested-lofo-per-target.csv",index=False)
    idf.to_csv(out/"nested-lofo-inner-scores.csv",index=False)

    final = metrics(odf["true_label"].to_numpy(dtype=object),odf["predicted_label"].to_numpy(dtype=object))
    majority = float(df["architecture_preference"].value_counts(normalize=True).max())
    final["majority_baseline"] = majority
    final["strict_gain_over_majority_baseline"] = final["strict_accuracy"]-majority
    final["target_count"] = len(df)
    pd.DataFrame([final]).to_csv(out/"nested-lofo-summary.csv",index=False)

    selected_counts = odf["selected_candidate"].value_counts().rename_axis("candidate_name").reset_index(name="outer_selection_count")
    selected_counts.to_csv(out/"nested-lofo-selected-candidates.csv",index=False)

    matrix = []
    for label in LABELS:
        t = odf[odf["true_label"]==label]
        matrix.append({
            "true_label":label,
            "pred_x86":int(np.sum(t["predicted_label"]=="x86-preferred")),
            "pred_independent":int(np.sum(t["predicted_label"]=="architecture-independent")),
            "pred_arm":int(np.sum(t["predicted_label"]=="arm-preferred")),
            "pred_abstain":int(np.sum(t["predicted_label"]=="abstain"))
        })
    pd.DataFrame(matrix).to_csv(out/"nested-lofo-confusion-matrix.csv",index=False)

    (out/"nested-lofo-manifest.json").write_text(json.dumps({
        "schema_version":1,
        "purpose":"nested LOFO confirmatory validation after exploratory sweep",
        "threshold_percent":args.threshold,
        "candidate_count":len(candidates),
        "candidates":[c.__dict__ for c in candidates],
        "selection_order":["balanced_accuracy","macro_f1","directional_accuracy","strict_accuracy","coverage"],
        "n_init":args.n_init,
        "random_state":args.random_state
    },indent=2)+"\n",encoding="utf-8")

    print("\nNESTED LOFO FINAL:")
    for key,value in final.items():
        print(f"{key:38s} {value}")
    print("\nSELECTED CANDIDATES:")
    print(selected_counts.to_string(index=False))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--profiles",required=True,type=Path)
    p.add_argument("--preferences-dir",required=True,type=Path)
    p.add_argument("--language-map",type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--threshold",type=float,default=15.0)
    p.add_argument("--candidates-json",type=Path)
    p.add_argument("--n-init",type=int,default=50)
    p.add_argument("--random-state",type=int,default=42)
    return p


if __name__=="__main__":
    run(parser().parse_args())
