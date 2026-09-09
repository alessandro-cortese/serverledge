"""Confronto sistematico di varianti del vettore di feature per il clustering.

Devo valutare quale combinazione di caratteristiche
produca il clustering migliore. Questo strumento esplora tre dimensioni --
sottoinsieme di feature, strategia di scaling, numero di cluster -- e per
ciascuna configurazione riporta sia le metriche geometriche sia quelle di
coerenza con la ground truth architetturale.

E' deliberatamente separato dalla pipeline di produzione. preprocess.py e
cluster.py hanno FEATURE_NAMES fisso a sei colonne, che attraversa header,
validazione e modello serializzato: renderlo parametrico richiederebbe di
toccare codice gia' validato. Qui la selezione delle colonne avviene a monte,
e i risultati servono a scegliere quale vettore adottare, non a sostituire la
pipeline.

Uso:

    python3 sweep_feature_vectors.py \\
        --x86-profiles data/profiling/raw/campagna-x86-.../function-profiles-median.csv \\
        --preferences data/profiling/derived/architecture-preferences.csv \\
        --output-csv risultati-sweep.csv \\
        --output-plot clustering.png
"""

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import (
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

# Le sei feature del paper, piu' i due parametri di configurazione che oggi
# fungono da filtro anziche' da dimensioni della distanza.
SIX = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
    "framework_runtime_ms",
]

CONFIG = ["configured_cpus", "configured_memory_mb"]

# Le varianti da confrontare.
#
# "sei" e' il riferimento del paper. "sei+config" verifica l'ipotesi emersa al
# ricevimento, cioe' se includere CPU e memoria fra le dimensioni della
# distanza riduca i punti isolati. Le varianti ridotte eliminano una feature
# per volta fra quelle che nel paper hanno importanza minore, per misurarne il
# contributo effettivo.
VARIANTI = {
    "sei": SIX,
    "sei+config": SIX + CONFIG,
    "senza_framework": [f for f in SIX if f != "framework_runtime_ms"],
    "senza_freemem": [f for f in SIX if f != "free_memory_mb"],
    "senza_utilcpu": [f for f in SIX if f != "utilized_cpus"],
    "solo_cpu_e_faults": [
        "page_faults_delta",
        "cpu_user_delta_ms",
        "cpu_kernel_delta_ms",
    ],
}

SCALERS = {
    "none": None,
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


def leggi_profili(path: Path, machine_tag: str):
    """Legge il CSV dei FunctionProfile filtrando su una sola architettura."""
    righe = []

    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["machine_tag"].strip() != machine_tag:
                continue
            righe.append(row)

    if not righe:
        raise SystemExit(f"nessun profilo con machine_tag={machine_tag} in {path}")

    return righe


def leggi_preferenze(path: Path):
    """Legge le etichette a tre classi prodotte da preference.py."""
    etichette = {}

    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            etichette[row["function_name"].strip()] = row[
                "architecture_preference"
            ].strip()

    return etichette


def purezza(assegnazioni, etichette):
    """Frazione di punti che ricade nella classe maggioritaria del proprio cluster.

    Non e' in sklearn perche' e' una riga di codice, ma e' la metrica piu'
    diretta per rispondere alla domanda del ricevimento: i cluster sono
    omogenei rispetto alla preferenza architetturale?
    """
    totale = 0

    for cluster in set(assegnazioni):
        membri = [e for a, e in zip(assegnazioni, etichette) if a == cluster]

        if membri:
            totale += max(membri.count(c) for c in set(membri))

    return totale / len(etichette)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x86-profiles", required=True)
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--machine-tag", default="amd64")
    parser.add_argument("--clusters", default="2,3,4,5")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-plot", default="")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    profili = leggi_profili(Path(args.x86_profiles), args.machine_tag)
    preferenze = leggi_preferenze(Path(args.preferences))

    nomi = [r["function_name"].strip() for r in profili]

    mancanti = [n for n in nomi if n not in preferenze]
    if mancanti:
        print(f"ATTENZIONE: {len(mancanti)} funzioni senza etichetta, escluse:")
        for n in mancanti:
            print(f"  {n}")

    tenuti = [i for i, n in enumerate(nomi) if n in preferenze]
    profili = [profili[i] for i in tenuti]
    nomi = [nomi[i] for i in tenuti]
    verita = [preferenze[n] for n in nomi]

    print(f"\nfunzioni: {len(nomi)}   architettura di riferimento: {args.machine_tag}")

    distribuzione = {c: verita.count(c) for c in sorted(set(verita))}
    print(f"classi: {distribuzione}\n")

    ks = [int(k) for k in args.clusters.split(",")]

    risultati = []

    for nome_variante, colonne in VARIANTI.items():
        grezzi = np.array(
            [[float(r[c]) for c in colonne] for r in profili], dtype=float
        )

        for nome_scaler, classe in SCALERS.items():
            if classe is None:
                scalati = grezzi.copy()
            else:
                scalati = classe().fit_transform(grezzi)

            for k in ks:
                if k >= len(nomi):
                    continue

                modello = KMeans(
                    n_clusters=k,
                    n_init=10,
                    random_state=args.random_state,
                )
                assegnazioni = modello.fit_predict(scalati)

                # La silhouette non e' definita con un solo cluster popolato.
                if len(set(assegnazioni)) < 2:
                    continue

                risultati.append(
                    {
                        "variante": nome_variante,
                        "n_feature": len(colonne),
                        "scaler": nome_scaler,
                        "k": k,
                        "silhouette": round(silhouette_score(scalati, assegnazioni), 4),
                        "inertia": round(modello.inertia_, 4),
                        "purity": round(purezza(assegnazioni, verita), 4),
                        "homogeneity": round(homogeneity_score(verita, assegnazioni), 4),
                        "completeness": round(completeness_score(verita, assegnazioni), 4),
                        "v_measure": round(v_measure_score(verita, assegnazioni), 4),
                        "ari": round(adjusted_rand_score(verita, assegnazioni), 4),
                        "nmi": round(
                            normalized_mutual_info_score(verita, assegnazioni), 4
                        ),
                    }
                )

    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(risultati[0].keys()))
        writer.writeheader()
        writer.writerows(risultati)

    print(f"configurazioni valutate: {len(risultati)}")
    print(f"output: {args.output_csv}\n")

    # Le due classifiche rispondono a domande diverse: la silhouette dice
    # quanto i cluster sono geometricamente separati, la purity quanto
    # corrispondono alla tripartizione architetturale. Una configurazione puo'
    # essere ottima secondo l'una e mediocre secondo l'altra.
    for chiave in ("silhouette", "purity"):
        print(f"--- migliori cinque per {chiave}")
        migliori = sorted(risultati, key=lambda r: -r[chiave])[:5]

        for r in migliori:
            print(
                f"  {r['variante']:20} {r['scaler']:9} k={r['k']}  "
                f"silhouette={r['silhouette']:>7}  purity={r['purity']:>6}  "
                f"v_measure={r['v_measure']:>7}"
            )
        print()

    if args.output_plot:
        disegna(profili, nomi, verita, args)


def disegna(profili, nomi, verita, args):
    """Proiezione bidimensionale colorata per preferenza architetturale.

    E' il grafico richiesto al ricevimento: se i colori si separano in gruppi
    distinti, lo spazio delle feature cattura la preferenza architetturale; se
    restano mescolati, l'approccio va ripensato.

    La proiezione usa le prime due componenti principali dello spazio a sei
    dimensioni: e' una riduzione, quindi due punti vicini nel grafico possono
    essere piu' distanti nello spazio reale. La percentuale di varianza
    spiegata, riportata negli assi, dice quanto la proiezione sia fedele.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grezzi = np.array(
        [[float(r[c]) for c in SIX] for r in profili], dtype=float
    )
    scalati = StandardScaler().fit_transform(grezzi)

    pca = PCA(n_components=2, random_state=args.random_state)
    proiettati = pca.fit_transform(scalati)

    colori = {
        "x86-preferred": "#d62728",
        "arm-preferred": "#1f77b4",
        "architecture-independent": "#7f7f7f",
    }

    fig, ax = plt.subplots(figsize=(11, 8))

    for classe, colore in colori.items():
        indici = [i for i, v in enumerate(verita) if v == classe]

        if not indici:
            continue

        ax.scatter(
            proiettati[indici, 0],
            proiettati[indici, 1],
            c=colore,
            label=f"{classe} ({len(indici)})",
            s=110,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.85,
        )

    for i, nome in enumerate(nomi):
        ax.annotate(
            nome,
            (proiettati[i, 0], proiettati[i, 1]),
            fontsize=7,
            alpha=0.75,
            xytext=(4, 4),
            textcoords="offset points",
        )

    varianza = pca.explained_variance_ratio_

    ax.set_xlabel(f"PC1 ({varianza[0] * 100:.1f}% della varianza)")
    ax.set_ylabel(f"PC2 ({varianza[1] * 100:.1f}% della varianza)")
    ax.set_title(
        "Funzioni nello spazio delle feature, colorate per preferenza architetturale\n"
        f"profilazione su {args.machine_tag}, scaler standard, proiezione PCA"
    )
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=150)

    print(f"grafico: {args.output_plot}")
    print(f"varianza spiegata dalle due componenti: {sum(varianza) * 100:.1f}%")


if __name__ == "__main__":
    main()
