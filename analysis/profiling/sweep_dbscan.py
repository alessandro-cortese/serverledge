"""Sweep DBSCAN con esplorazione della distanza di vicinato.

KMeans assegna ogni punto a un cluster, anche quando il punto e' remoto da
tutti gli altri: con trenta funzioni e tre gruppi imposti, una funzione isolata
viene comunque attribuita al centroide meno lontano, e ne sposta la posizione.

DBSCAN non lo fa. Definisce i cluster per densita': un punto appartiene a un
cluster se ha almeno min_samples vicini entro distanza eps, altrimenti viene
etichettato come rumore. Le funzioni isolate restano dichiarate tali.

Sui dati della campagna a trenta funzioni la dispersione e' molto disomogenea:
twin-chacha20 dista 0,27 dal suo vicino piu' prossimo, primenumber ne dista
4,59. E' precisamente il caso in cui la densita' dice qualcosa che la
partizione forzata non puo' dire.

Il rovescio della medaglia e' che DBSCAN non permette di fissare il numero di
cluster, e i professori ne vogliono tre: lo sweep serve quindi a capire se
esista un valore di eps che produca tre gruppi densi, non a sostituire KMeans.

Uso:

    python3 sweep_dbscan.py \\
        --profiles .../function-profiles-median.csv \\
        --preferences .../preferenze-15.csv \\
        --output-csv sweep-dbscan.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

FEATURES = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
    "framework_runtime_ms",
]

SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


def purezza(assegnazioni, etichette, escludi_rumore=True):
    """Purezza calcolata sui soli punti assegnati a un cluster.

    Includere il rumore falserebbe la misura: DBSCAN puo' ottenere purezza
    apparente alta scartando i punti difficili.
    """
    coppie = [
        (a, e)
        for a, e in zip(assegnazioni, etichette)
        if not (escludi_rumore and a == -1)
    ]

    if not coppie:
        return 0.0

    totale = 0

    for cluster in set(a for a, _ in coppie):
        membri = [e for a, e in coppie if a == cluster]
        totale += max(membri.count(c) for c in set(membri))

    return totale / len(coppie)


def suggerisci_eps(X, k):
    """Distanze al k-esimo vicino, ordinate.

    E' il metodo classico per scegliere eps: si osserva dove la curva delle
    distanze ordinate presenta un ginocchio, che separa i punti densi da
    quelli isolati. Qui la curva viene campionata per suggerire un intervallo
    di valori da esplorare, invece di sceglierli a caso.
    """
    vicini = NearestNeighbors(n_neighbors=k).fit(X)
    distanze, _ = vicini.kneighbors(X)

    return np.sort(distanze[:, -1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--machine-tag", default="amd64")
    parser.add_argument("--min-samples", default="2,3,4")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    righe = [
        r
        for r in csv.DictReader(Path(args.profiles).open(newline=""))
        if r["machine_tag"].strip() == args.machine_tag
    ]

    etichette_per_nome = {
        r["function_name"].strip(): r["architecture_preference"].strip()
        for r in csv.DictReader(Path(args.preferences).open(newline=""))
    }

    righe = [r for r in righe if r["function_name"].strip() in etichette_per_nome]

    nomi = [r["function_name"].strip() for r in righe]
    verita = [etichette_per_nome[n] for n in nomi]

    grezzi = np.array([[float(r[c]) for c in FEATURES] for r in righe], dtype=float)

    print(f"\nfunzioni: {len(nomi)}   riferimento: {args.machine_tag}")
    print(f"classi: {({c: verita.count(c) for c in sorted(set(verita))})}\n")

    risultati = []

    for nome_scaler, classe in SCALERS.items():
        X = classe().fit_transform(grezzi)

        for min_samples in [int(m) for m in args.min_samples.split(",")]:
            curva = suggerisci_eps(X, min_samples)

            print(f"--- {nome_scaler}, min_samples={min_samples}")
            print(
                f"    distanze al {min_samples}-esimo vicino: "
                f"min={curva[0]:.3f} mediana={np.median(curva):.3f} max={curva[-1]:.3f}"
            )

            # I valori di eps vengono presi lungo la curva delle distanze
            # invece che su una griglia fissa: cosi' l'esplorazione si adatta
            # alla scala prodotta dallo scaler, che cambia di ordini di
            # grandezza fra standard e minmax.
            candidati = sorted(
                set(
                    round(float(np.percentile(curva, p)), 4)
                    for p in (10, 20, 30, 40, 50, 60, 70, 80, 90)
                )
            )

            for eps in candidati:
                if eps <= 0:
                    continue

                modello = DBSCAN(eps=eps, min_samples=min_samples)
                assegnazioni = modello.fit_predict(X)

                rumore = int((assegnazioni == -1).sum())
                n_cluster = len(set(assegnazioni) - {-1})

                if n_cluster == 0:
                    continue

                assegnati = assegnazioni != -1

                # La silhouette richiede almeno due cluster fra i punti non
                # rumorosi.
                if len(set(assegnazioni[assegnati])) >= 2:
                    sil = round(
                        silhouette_score(X[assegnati], assegnazioni[assegnati]), 4
                    )
                else:
                    sil = ""

                risultati.append(
                    {
                        "scaler": nome_scaler,
                        "min_samples": min_samples,
                        "eps": eps,
                        "n_cluster": n_cluster,
                        "rumore": rumore,
                        "coverage": round(1 - rumore / len(nomi), 4),
                        "silhouette_su_assegnati": sil,
                        "purity": round(purezza(assegnazioni, verita), 4),
                        "homogeneity": round(
                            homogeneity_score(verita, assegnazioni), 4
                        ),
                        "v_measure": round(v_measure_score(verita, assegnazioni), 4),
                        "ari": round(adjusted_rand_score(verita, assegnazioni), 4),
                        "nmi": round(
                            normalized_mutual_info_score(verita, assegnazioni), 4
                        ),
                        "isolate": ";".join(
                            n for n, a in zip(nomi, assegnazioni) if a == -1
                        ),
                    }
                )

    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(risultati[0].keys()))
        writer.writeheader()
        writer.writerows(risultati)

    print(f"\nconfigurazioni: {len(risultati)}")
    print(f"output: {args.output_csv}\n")

    # Le configurazioni con tre cluster sono quelle che rispondono al requisito
    # applicativo, e vanno guardate per prime anche se non sono le migliori in
    # assoluto.
    tre = [r for r in risultati if r["n_cluster"] == 3]

    if tre:
        print("--- configurazioni con esattamente tre cluster")
        print(f"{'scaler':10} {'ms':>3} {'eps':>8} {'rumore':>7} {'purity':>7} {'ARI':>8}")

        for r in sorted(tre, key=lambda r: -r["ari"])[:8]:
            print(
                f"{r['scaler']:10} {r['min_samples']:>3} {r['eps']:>8} "
                f"{r['rumore']:>7} {r['purity']:>7} {r['ari']:>8}"
            )
    else:
        print("Nessuna configurazione produce esattamente tre cluster.")

    print("\n--- migliori otto per ARI, qualunque numero di cluster")
    print(
        f"{'scaler':10} {'ms':>3} {'eps':>8} {'ncl':>4} {'rumore':>7} "
        f"{'purity':>7} {'ARI':>8}"
    )

    for r in sorted(risultati, key=lambda r: -r["ari"])[:8]:
        print(
            f"{r['scaler']:10} {r['min_samples']:>3} {r['eps']:>8} "
            f"{r['n_cluster']:>4} {r['rumore']:>7} {r['purity']:>7} {r['ari']:>8}"
        )

    migliore = max(risultati, key=lambda r: r["ari"])

    if migliore["isolate"]:
        print(f"\n--- funzioni isolate nella configurazione con ARI migliore")
        for n in migliore["isolate"].split(";"):
            print(f"    {n}")


if __name__ == "__main__":
    main()
