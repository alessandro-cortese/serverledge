"""Sweep esteso: varianti del vettore di feature, incluse quelle derivate.

Estende il confronto oltre le sei feature assolute del paper, aggiungendo due
famiglie di trasformazioni motivate da un'osservazione sperimentale precisa.

Sui dati della campagna a trenta funzioni, il clustering sulle sei feature
assolute produce un Adjusted Rand Index negativo rispetto alla preferenza
architetturale: i cluster corrispondono alla tripartizione peggio di un
raggruppamento casuale. Le funzioni gemelle spiegano perche': coppie con
profilo di risorse simile per costruzione — chacha20 e twin-chacha20 — hanno
preferenze architetturali opposte, perche' la preferenza dipende da COME il
calcolo e' implementato, non da QUANTO consuma.

Le due famiglie aggiunte provano a catturare la composizione anziche' la
quantita':

- RAPPORTI: normalizzano ogni grandezza sul tempo CPU totale, cosi' che due
  funzioni con volumi molto diversi ma composizione simile risultino vicine.
  dna-visualisation ha 7228 campioni e filehandle 16: in valore assoluto sono
  incomparabili, in composizione potrebbero non esserlo.

- LOGARITMI: comprimono i quattro ordini di grandezza che separano
  page_faults_delta dalle altre feature. Senza compressione la distanza
  euclidea e' dominata da una sola dimensione, ed e' anche la ragione per cui
  la proiezione PCA senza scaling produce un allineamento verticale.

Uso:

    python3 sweep_vettori_esteso.py \\
        --profiles .../function-profiles-median.csv \\
        --preferences .../preferenze-15.csv \\
        --output-csv sweep-esteso.csv \\
        --output-plot-dir grafici/
"""

import argparse
import csv
import math
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

COLORI = {
    "x86-preferred": "#d62728",
    "arm-preferred": "#1f77b4",
    "architecture-independent": "#7f7f7f",
}


def derivate(riga):
    """Calcola le feature derivate a partire da una riga di FunctionProfile.

    Il denominatore e' il tempo CPU totale, utente piu' kernel: e' la misura
    piu' diretta di "quanto lavoro" ha fatto la funzione, e normalizzarci sopra
    trasforma le grandezze assolute in proporzioni.
    """
    user = float(riga["cpu_user_delta_ms"])
    kernel = float(riga["cpu_kernel_delta_ms"])
    faults = float(riga["page_faults_delta"])
    framework = float(riga["framework_runtime_ms"])
    freemem = float(riga["free_memory_mb"])
    util = float(riga["utilized_cpus"])

    cpu_totale = user + kernel

    # Una funzione puo' avere tempo CPU misurato nullo se molto breve: in quel
    # caso i rapporti non sono definiti e si usa zero, che e' il valore verso
    # cui tendono.
    if cpu_totale <= 0:
        return {
            "quota_kernel": 0.0,
            "faults_per_cpu_ms": 0.0,
            "framework_per_cpu_ms": 0.0,
            "cpu_per_utilizzo": 0.0,
        }

    return {
        # Quanto del tempo CPU e' speso nel kernel: distingue un carico di
        # calcolo puro da uno dominato da chiamate di sistema.
        "quota_kernel": kernel / cpu_totale,
        # Intensita' di page fault per millisecondo di CPU: distingue un
        # accesso alla memoria regolare da uno irregolare, indipendentemente
        # dalla durata della funzione.
        "faults_per_cpu_ms": faults / cpu_totale,
        # Peso dell'overhead di framework rispetto al lavoro utile.
        "framework_per_cpu_ms": framework / cpu_totale,
        # Tempo CPU per unita' di utilizzazione: proxy della durata effettiva
        # normalizzata sulla quota di core concessa.
        "cpu_per_utilizzo": cpu_totale / util if util > 0 else 0.0,
    }


def log_assolute(riga):
    """Versione logaritmica delle feature assolute.

    log1p invece di log perche' page_faults_delta puo' valere zero — le due
    funzioni sintetiche a bassa attivita' di memoria lo producono — e log(0)
    non e' definito.
    """
    return {
        f"log_{nome}": math.log1p(max(0.0, float(riga[nome])))
        for nome in ASSOLUTE
    }


def costruisci_varianti():
    """Definisce le combinazioni da confrontare.

    Ogni variante e' una lista di nomi di colonna, che possono provenire dal
    CSV originale o essere state derivate.
    """
    tre_principali = [
        "page_faults_delta",
        "cpu_user_delta_ms",
        "cpu_kernel_delta_ms",
    ]

    return {
        # Riferimento: le sei feature del paper, come sono.
        "sei_assolute": ASSOLUTE,
        # Verifica l'ipotesi emersa al ricevimento: includere configurazione
        # di CPU e memoria fra le dimensioni della distanza anziche' usarle
        # come filtro.
        "sei_piu_config": ASSOLUTE + CONFIG,
        # Le tre a cui il paper attribuisce importanza maggiore.
        "tre_principali": tre_principali,
        # Le sei feature compresse in scala logaritmica.
        "sei_log": [f"log_{n}" for n in ASSOLUTE],
        # Sola composizione: nessuna grandezza assoluta.
        "rapporti": [
            "quota_kernel",
            "faults_per_cpu_ms",
            "framework_per_cpu_ms",
            "utilized_cpus",
        ],
        # Composizione piu' una misura di scala in forma logaritmica.
        "rapporti_piu_scala": [
            "quota_kernel",
            "faults_per_cpu_ms",
            "framework_per_cpu_ms",
            "utilized_cpus",
            "log_cpu_user_delta_ms",
        ],
        # Ibrido: le tre principali in scala logaritmica piu' la quota kernel.
        "log_tre_piu_quota": [
            "log_page_faults_delta",
            "log_cpu_user_delta_ms",
            "log_cpu_kernel_delta_ms",
            "quota_kernel",
        ],
    }


SCALERS = {
    "none": None,
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
}


def purezza(assegnazioni, etichette):
    totale = 0

    for cluster in set(assegnazioni):
        membri = [e for a, e in zip(assegnazioni, etichette) if a == cluster]
        if membri:
            totale += max(membri.count(c) for c in set(membri))

    return totale / len(etichette)


def carica(profili_path, preferenze_path, machine_tag):
    righe = []

    with Path(profili_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["machine_tag"].strip() == machine_tag:
                righe.append(row)

    etichette = {}

    with Path(preferenze_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            etichette[row["function_name"].strip()] = row[
                "architecture_preference"
            ].strip()

    dati = []

    for row in righe:
        nome = row["function_name"].strip()

        if nome not in etichette:
            print(f"  {nome}: nessuna etichetta, escluso")
            continue

        completa = dict(row)
        completa.update(derivate(row))
        completa.update(log_assolute(row))

        dati.append((nome, completa, etichette[nome]))

    return dati


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--machine-tag", default="amd64")
    parser.add_argument("--clusters", default="2,3,4,5,6")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-plot-dir", default="")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dati = carica(args.profiles, args.preferences, args.machine_tag)

    nomi = [d[0] for d in dati]
    verita = [d[2] for d in dati]

    distribuzione = {c: verita.count(c) for c in sorted(set(verita))}

    print(f"\nfunzioni: {len(nomi)}   riferimento: {args.machine_tag}")
    print(f"classi: {distribuzione}\n")

    varianti = costruisci_varianti()
    ks = [int(k) for k in args.clusters.split(",")]

    risultati = []
    migliore_per_grafico = {}

    for nome_variante, colonne in varianti.items():
        grezzi = np.array(
            [[float(d[1][c]) for c in colonne] for d in dati], dtype=float
        )

        for nome_scaler, classe in SCALERS.items():
            scalati = grezzi.copy() if classe is None else classe().fit_transform(grezzi)

            for k in ks:
                if k >= len(nomi):
                    continue

                modello = KMeans(
                    n_clusters=k, n_init=10, random_state=args.random_state
                )
                assegnazioni = modello.fit_predict(scalati)

                if len(set(assegnazioni)) < 2:
                    continue

                ari = adjusted_rand_score(verita, assegnazioni)

                riga = {
                    "variante": nome_variante,
                    "n_feature": len(colonne),
                    "scaler": nome_scaler,
                    "k": k,
                    "silhouette": round(silhouette_score(scalati, assegnazioni), 4),
                    "purity": round(purezza(assegnazioni, verita), 4),
                    "homogeneity": round(homogeneity_score(verita, assegnazioni), 4),
                    "completeness": round(completeness_score(verita, assegnazioni), 4),
                    "v_measure": round(v_measure_score(verita, assegnazioni), 4),
                    "ari": round(ari, 4),
                    "nmi": round(
                        normalized_mutual_info_score(verita, assegnazioni), 4
                    ),
                }

                risultati.append(riga)

                chiave = (nome_variante, nome_scaler)
                if chiave not in migliore_per_grafico or ari > migliore_per_grafico[chiave][0]:
                    migliore_per_grafico[chiave] = (ari, scalati, assegnazioni, riga)

    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(risultati[0].keys()))
        writer.writeheader()
        writer.writerows(risultati)

    print(f"configurazioni: {len(risultati)}")
    print(f"output: {args.output_csv}\n")

    # L'ARI e' la metrica decisiva: misura la corrispondenza fra clustering e
    # tripartizione architetturale corretta per il caso. Zero significa
    # equivalente a un raggruppamento casuale, negativo significa peggiore.
    print("--- migliori dieci per Adjusted Rand Index")
    print(f"{'variante':22} {'scaler':9} {'k':>2}  {'ARI':>7} {'purity':>7} {'silh':>7}")

    for r in sorted(risultati, key=lambda r: -r["ari"])[:10]:
        print(
            f"{r['variante']:22} {r['scaler']:9} {r['k']:>2}  "
            f"{r['ari']:>7} {r['purity']:>7} {r['silhouette']:>7}"
        )

    print()
    print("--- confronto per variante, migliore ARI di ciascuna")

    for nome_variante in varianti:
        della_variante = [r for r in risultati if r["variante"] == nome_variante]
        migliore = max(della_variante, key=lambda r: r["ari"])
        print(
            f"{nome_variante:22} ARI={migliore['ari']:>7}  "
            f"({migliore['scaler']}, k={migliore['k']}, purity={migliore['purity']})"
        )

    if args.output_plot_dir:
        disegna(nomi, verita, migliore_per_grafico, args)


def disegna(nomi, verita, migliori, args):
    """Un grafico per variante, usando la configurazione con ARI migliore.

    I punti sono colorati per preferenza architetturale e la forma indica il
    cluster assegnato: se colore e forma coincidessero, il clustering
    predirebbe la preferenza. La proiezione e' PCA a due componenti.

    Lo scaling non e' solo una scelta di preprocessing ma condiziona la
    leggibilita': senza, page_faults_delta domina la varianza e la proiezione
    collassa su un asse, producendo l'allineamento verticale che rende il
    grafico inutile.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cartella = Path(args.output_plot_dir)
    cartella.mkdir(parents=True, exist_ok=True)

    marcatori = ["o", "s", "^", "D", "v", "P", "X", "*"]

    for (variante, scaler), (ari, scalati, assegnazioni, riga) in migliori.items():
        if scaler == "none":
            continue

        pca = PCA(n_components=2, random_state=args.random_state)
        proiettati = pca.fit_transform(scalati)

        fig, ax = plt.subplots(figsize=(11, 8))

        for i, nome in enumerate(nomi):
            ax.scatter(
                proiettati[i, 0],
                proiettati[i, 1],
                c=COLORI[verita[i]],
                marker=marcatori[assegnazioni[i] % len(marcatori)],
                s=130,
                edgecolors="black",
                linewidths=0.6,
                alpha=0.85,
            )
            ax.annotate(
                nome,
                (proiettati[i, 0], proiettati[i, 1]),
                fontsize=7,
                alpha=0.75,
                xytext=(5, 4),
                textcoords="offset points",
            )

        for classe, colore in COLORI.items():
            if classe in verita:
                ax.scatter([], [], c=colore, s=110, edgecolors="black", label=classe)

        varianza = pca.explained_variance_ratio_

        ax.set_xlabel(f"PC1 ({varianza[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({varianza[1] * 100:.1f}%)")
        ax.set_title(
            f"Variante «{variante}» — scaler {scaler}, k={riga['k']}\n"
            f"colore = preferenza architetturale, forma = cluster   "
            f"ARI={ari:.4f}  purity={riga['purity']}"
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.25)

        fig.tight_layout()

        percorso = cartella / f"{variante}_{scaler}_k{riga['k']}.png"
        fig.savefig(percorso, dpi=150)
        plt.close(fig)

    print(f"\ngrafici in: {cartella}")


if __name__ == "__main__":
    main()
