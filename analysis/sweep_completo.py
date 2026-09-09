"""Sweep completo del clustering lungo quattro dimensioni.

Esplora sistematicamente lo spazio delle configurazioni per rispondere alla
domanda posta dai relatori: esiste una combinazione per cui il clustering sui
profili di risorse separa le funzioni secondo la preferenza architetturale?

Le quattro dimensioni sono:

    soglia di preferenza    determina le etichette, quindi il bersaglio
    vettore di feature      quali colonne definiscono lo spazio
    strategia di scaling    come le colonne vengono normalizzate
    numero di cluster       la granularita' della partizione

La soglia e' la dimensione piu' importante e la meno ovvia: non e' un parametro
del clustering ma della ground truth. Il paper usa 15%, la tesi precedente
2,5%, e con soglie diverse la stessa funzione cambia classe. Variarla significa
chiedersi se il clustering fallisca perche' non coglie la struttura o perche'
la tripartizione stessa e' definita male.

La metrica decisiva e' l'Adjusted Rand Index, che misura la corrispondenza fra
partizioni correggendo per il caso: zero equivale a un raggruppamento casuale,
negativo e' peggio del caso. Silhouette e purezza sono riportate ma non
bastano da sole -- la purezza cresce banalmente con il numero di cluster, e la
silhouette misura la separazione geometrica, non la corrispondenza con le
classi.

Uso:

    python3 sweep_completo.py \\
        --profiles .../function-profiles-median.csv \\
        --preferences 2.5=.../pref-2.5.csv \\
        --preferences 5=.../pref-5.csv \\
        --preferences 15=.../pref-15.csv \\
        --output-csv sweep-completo.csv
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ASSOLUTE = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
    "framework_runtime_ms",
]

CONFIG = ["configured_cpus", "configured_memory_mb"]


def arricchisci(riga):
    """Aggiunge le colonne derivate: rapporti e logaritmi.

    I rapporti normalizzano sul tempo CPU totale, cosi' che due funzioni con
    volumi molto diversi ma composizione simile risultino vicine. I logaritmi
    comprimono i quattro ordini di grandezza che separano page_faults_delta
    dalle altre colonne.
    """
    user = float(riga["cpu_user_delta_ms"])
    kernel = float(riga["cpu_kernel_delta_ms"])
    faults = float(riga["page_faults_delta"])
    framework = float(riga["framework_runtime_ms"])
    util = float(riga["utilized_cpus"])

    cpu = user + kernel

    derivate = {
        "quota_kernel": kernel / cpu if cpu > 0 else 0.0,
        "faults_per_cpu_ms": faults / cpu if cpu > 0 else 0.0,
        "framework_per_cpu_ms": framework / cpu if cpu > 0 else 0.0,
        "cpu_per_utilizzo": cpu / util if util > 0 else 0.0,
    }

    for nome in ASSOLUTE:
        derivate[f"log_{nome}"] = math.log1p(max(0.0, float(riga[nome])))

    return derivate


VARIANTI = {
    "sei_assolute": ASSOLUTE,
    "sei_piu_config": ASSOLUTE + CONFIG,
    "tre_principali": [
        "page_faults_delta",
        "cpu_user_delta_ms",
        "cpu_kernel_delta_ms",
    ],
    "sei_log": [f"log_{n}" for n in ASSOLUTE],
    "rapporti": [
        "quota_kernel",
        "faults_per_cpu_ms",
        "framework_per_cpu_ms",
        "utilized_cpus",
    ],
    "rapporti_piu_scala": [
        "quota_kernel",
        "faults_per_cpu_ms",
        "framework_per_cpu_ms",
        "utilized_cpus",
        "log_cpu_user_delta_ms",
    ],
    "log_tre_piu_quota": [
        "log_page_faults_delta",
        "log_cpu_user_delta_ms",
        "log_cpu_kernel_delta_ms",
        "quota_kernel",
    ],
}

SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


def purezza(assegnazioni, etichette):
    """Purezza sui soli punti assegnati, escludendo il rumore di DBSCAN.

    Includerlo permetterebbe di ottenere purezza alta scartando i punti
    difficili, che non e' un risultato.
    """
    coppie = [(a, e) for a, e in zip(assegnazioni, etichette) if a != -1]

    if not coppie:
        return 0.0

    totale = 0

    for cluster in set(a for a, _ in coppie):
        membri = [e for a, e in coppie if a == cluster]
        totale += max(membri.count(c) for c in set(membri))

    return totale / len(coppie)


def valuta(X, assegnazioni, verita, nomi):
    """Calcola l'insieme completo delle metriche per una configurazione."""
    assegnati = np.array(assegnazioni) != -1
    n_cluster = len(set(assegnazioni) - {-1})

    if n_cluster < 1:
        return None

    if len(set(np.array(assegnazioni)[assegnati])) >= 2:
        sil = round(
            silhouette_score(X[assegnati], np.array(assegnazioni)[assegnati]), 4
        )
    else:
        sil = ""

    return {
        "n_cluster": n_cluster,
        "rumore": int((~assegnati).sum()),
        "coverage": round(assegnati.sum() / len(nomi), 4),
        "silhouette": sil,
        "purity": round(purezza(assegnazioni, verita), 4),
        "homogeneity": round(homogeneity_score(verita, assegnazioni), 4),
        "completeness": round(completeness_score(verita, assegnazioni), 4),
        "v_measure": round(v_measure_score(verita, assegnazioni), 4),
        "ari": round(adjusted_rand_score(verita, assegnazioni), 4),
        "nmi": round(normalized_mutual_info_score(verita, assegnazioni), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True)
    parser.add_argument(
        "--preferences",
        action="append",
        required=True,
        help="soglia=percorso, ripetibile",
    )
    parser.add_argument("--machine-tag", default="amd64")
    parser.add_argument("--clusters", default="2,3,4,5,6")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    righe = [
        r
        for r in csv.DictReader(Path(args.profiles).open(newline=""))
        if r["machine_tag"].strip() == args.machine_tag
    ]

    for r in righe:
        r.update(arricchisci(r))

    nomi = [r["function_name"].strip() for r in righe]

    print(f"\nfunzioni: {len(nomi)}   riferimento: {args.machine_tag}")

    # Ogni soglia produce un insieme di etichette diverso, quindi un bersaglio
    # diverso per lo stesso clustering.
    etichette_per_soglia = {}

    for spec in args.preferences:
        soglia, percorso = spec.split("=", 1)

        mappa = {
            r["function_name"].strip(): r["architecture_preference"].strip()
            for r in csv.DictReader(Path(percorso).open(newline=""))
        }

        verita = [mappa.get(n) for n in nomi]

        if any(v is None for v in verita):
            mancanti = [n for n, v in zip(nomi, verita) if v is None]
            print(f"  soglia {soglia}: {len(mancanti)} funzioni senza etichetta")
            continue

        etichette_per_soglia[soglia] = verita

        distribuzione = {c: verita.count(c) for c in sorted(set(verita))}
        print(f"  soglia {soglia:>5}%: {distribuzione}")

    ks = [int(k) for k in args.clusters.split(",")]

    risultati = []

    for soglia, verita in etichette_per_soglia.items():
        for nome_variante, colonne in VARIANTI.items():
            grezzi = np.array(
                [[float(r[c]) for c in colonne] for r in righe], dtype=float
            )

            for nome_scaler, classe in SCALERS.items():
                X = classe().fit_transform(grezzi)

                for k in ks:
                    if k >= len(nomi):
                        continue

                    modello = KMeans(
                        n_clusters=k, n_init=10, random_state=args.random_state
                    )
                    assegnazioni = modello.fit_predict(X)

                    metriche = valuta(X, assegnazioni, verita, nomi)

                    if metriche:
                        risultati.append(
                            {
                                "soglia": soglia,
                                "algoritmo": "kmeans",
                                "variante": nome_variante,
                                "n_feature": len(colonne),
                                "scaler": nome_scaler,
                                "parametro": f"k={k}",
                                **metriche,
                            }
                        )

                # DBSCAN: eps campionato lungo la curva delle distanze al
                # k-esimo vicino, che si adatta alla scala prodotta dallo
                # scaler invece di usare una griglia fissa.
                for min_samples in (2, 3, 4):
                    vicini = NearestNeighbors(n_neighbors=min_samples).fit(X)
                    distanze, _ = vicini.kneighbors(X)
                    curva = np.sort(distanze[:, -1])

                    for p in (20, 40, 50, 60, 80):
                        eps = round(float(np.percentile(curva, p)), 4)

                        if eps <= 0:
                            continue

                        assegnazioni = DBSCAN(
                            eps=eps, min_samples=min_samples
                        ).fit_predict(X)

                        metriche = valuta(X, assegnazioni, verita, nomi)

                        if metriche and metriche["n_cluster"] >= 1:
                            risultati.append(
                                {
                                    "soglia": soglia,
                                    "algoritmo": "dbscan",
                                    "variante": nome_variante,
                                    "n_feature": len(colonne),
                                    "scaler": nome_scaler,
                                    "parametro": f"eps={eps},ms={min_samples}",
                                    **metriche,
                                }
                            )

    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(risultati[0].keys()))
        writer.writeheader()
        writer.writerows(risultati)

    print(f"\nconfigurazioni valutate: {len(risultati)}")
    print(f"output: {args.output_csv}\n")

    print("--- migliori quindici per Adjusted Rand Index")
    print(
        f"{'soglia':>6} {'algo':7} {'variante':20} {'scaler':9} "
        f"{'parametro':18} {'ncl':>3} {'rum':>4} {'ARI':>8} {'pur':>6}"
    )

    for r in sorted(risultati, key=lambda r: -r["ari"])[:15]:
        print(
            f"{r['soglia']:>6} {r['algoritmo']:7} {r['variante']:20} "
            f"{r['scaler']:9} {r['parametro']:18} {r['n_cluster']:>3} "
            f"{r['rumore']:>4} {r['ari']:>8} {r['purity']:>6}"
        )

    print("\n--- migliore ARI per soglia")
    for soglia in etichette_per_soglia:
        della_soglia = [r for r in risultati if r["soglia"] == soglia]
        migliore = max(della_soglia, key=lambda r: r["ari"])
        print(
            f"  soglia {soglia:>5}%: ARI={migliore['ari']:>8}  "
            f"({migliore['algoritmo']}, {migliore['variante']}, "
            f"{migliore['scaler']}, {migliore['parametro']}, "
            f"rumore={migliore['rumore']})"
        )

    print("\n--- migliore ARI fra le configurazioni con tre cluster e senza rumore")
    tre = [
        r for r in risultati if r["n_cluster"] == 3 and r["rumore"] == 0
    ]

    if tre:
        for r in sorted(tre, key=lambda r: -r["ari"])[:8]:
            print(
                f"  soglia {r['soglia']:>5}% {r['variante']:20} {r['scaler']:9} "
                f"{r['parametro']:12} ARI={r['ari']:>8} purity={r['purity']}"
            )
    else:
        print("  nessuna")

    massimo = max(risultati, key=lambda r: r["ari"])

    print()
    print(f"ARI massimo su {len(risultati)} configurazioni: {massimo['ari']}")
    print(
        "Un valore prossimo a zero indica che il clustering sui profili di "
        "risorse\nnon separa le funzioni secondo la preferenza architetturale."
    )


if __name__ == "__main__":
    main()
