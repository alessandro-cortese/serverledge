#!/usr/bin/env python3

"""
Confronto controllato delle caratteristiche statiche e delle strategie
di scaling sul corpus Serverledge di 59 funzioni.

Principi:
- 59 funzioni static-eligible;
- LOFO: il target viene escluso dal preprocessing, dal K-Means e dal
  catalogo dei donor;
- K-Means K=6;
- n_init=50;
- seed: 11, 23, 37, 41, 53;
- weighted directional vote 1/d^2;
- Manhattan per scegliere il donor nella classe direzionale vincente;
- la ground truth ARM/x86 del target viene utilizzata soltanto
  per la valutazione finale.

Lo script NON modifica alcun dato del corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from threadpoolctl import threadpool_limits


# ============================================================
# Feature dinamiche PAPER5
# ============================================================

DYN = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]


# ============================================================
# Feature statiche candidate
# ============================================================

TOK = "static_token_count_mean"
FUN = "static_function_count"
CCN = "static_ccn_mean"
NLOC = "static_function_nloc_mean"


# Tutti i sottoinsiemi che vogliamo confrontare.
SUBSETS = {
    "paper5": [],

    "token": [
        TOK,
    ],

    "functions": [
        FUN,
    ],

    "token_functions": [
        TOK,
        FUN,
    ],

    "token_functions_ccn": [
        TOK,
        FUN,
        CCN,
    ],

    "token_functions_nloc": [
        TOK,
        FUN,
        NLOC,
    ],

    "static4": [
        TOK,
        FUN,
        CCN,
        NLOC,
    ],
}


# ============================================================
# Strategie di preprocessing
# ============================================================

SCALERS = (
    "minmax_plain",
    "standard_plain",
    "minmax_block_sqrt",
)


# Preferenze architetturali che partecipano al voto.
DIRECTIONS = (
    "x86-preferred",
    "arm-preferred",
)


SEEDS = (11, 23, 37, 41, 53)


# Configurazione che avevamo scelto come pipeline finale.
BASE = "token_functions__minmax_block_sqrt"


# ============================================================
# Utility
# ============================================================

def digest(path: Path) -> str:
    """SHA256 di un file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(root: Path, rel: str, columns: list[str]):
    """
    Carica un CSV e verifica che contenga le colonne necessarie.
    """

    path = root / rel

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: colonne mancanti: {missing}"
        )

    if df.function_name.duplicated().any():
        raise ValueError(
            f"Function name duplicati in {path}"
        )

    return df, path


# ============================================================
# Caricamento corpus
# ============================================================

def load(root: Path):

    profiles, profile_path = source(
        root,
        "resource/x86/function-profiles-median.csv",
        [
            "function_name",
            *DYN,
        ],
    )

    static, static_path = source(
        root,
        "static-code-metrics.csv",
        [
            "function_name",
            "static_source_available",
            TOK,
            FUN,
            CCN,
            NLOC,
        ],
    )

    ground_truth, gt_path = source(
        root,
        "ground_truth/preferences-2p5.csv",
        [
            "function_name",
            "architecture_preference",
            "x86_duration_ms",
            "arm_duration_ms",
        ],
    )

    languages, language_path = source(
        root,
        "function-languages.csv",
        [
            "function_name",
            "language",
        ],
    )


    # --------------------------------------------------------
    # Merge dei dati
    # --------------------------------------------------------

    df = (
        profiles[
            [
                "function_name",
                *DYN,
            ]
        ]
        .merge(
            static[
                [
                    "function_name",
                    "static_source_available",
                    TOK,
                    FUN,
                    CCN,
                    NLOC,
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            ground_truth[
                [
                    "function_name",
                    "architecture_preference",
                    "x86_duration_ms",
                    "arm_duration_ms",
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
        .merge(
            languages[
                [
                    "function_name",
                    "language",
                ]
            ],
            on="function_name",
            validate="one_to_one",
        )
    )


    # --------------------------------------------------------
    # Manteniamo solo le funzioni per cui è disponibile
    # l'analisi statica.
    # --------------------------------------------------------

    mask = (
        df.static_source_available
        .astype(str)
        .str.lower()
        .isin(("true", "1", "yes"))
    )


    numeric_columns = [
        *DYN,
        TOK,
        FUN,
        CCN,
        NLOC,
        "x86_duration_ms",
        "arm_duration_ms",
    ]


    for name in numeric_columns:

        df[name] = pd.to_numeric(
            df[name],
            errors="coerce",
        )

        mask &= np.isfinite(
            df[name].to_numpy(float)
        )


    df = (
        df[mask]
        .sort_values("function_name")
        .reset_index(drop=True)
    )


    # Il corpus finale deve essere esattamente quello da 59.
    if len(df) != 59:
        raise ValueError(
            f"Attese 59 funzioni eleggibili, trovate {len(df)}"
        )


    if not (
            df[
                [
                    "x86_duration_ms",
                    "arm_duration_ms",
                ]
            ] > 0
    ).all().all():

        raise ValueError(
            "Trovata durata <= 0"
        )


    hashes = {}

    for path in [
        profile_path,
        static_path,
        gt_path,
        language_path,
    ]:

        hashes[
            str(path.relative_to(root))
        ] = digest(path)


    return df, hashes


# ============================================================
# Scaling
# ============================================================

def transform(
        train: pd.DataFrame,
        target: pd.DataFrame,
        subset: list[str],
        scheme: str,
):

    """
    Applica il preprocessing.

    Importante:
    lo scaler viene FITTATO SOLTANTO sul training set LOFO.
    Il target viene soltanto trasformato dopo.
    """

    columns = DYN + subset


    # --------------------------------------------------------
    # Strategia 1:
    # StandardScaler classico
    # --------------------------------------------------------

    if scheme == "standard_plain":
        scaler = StandardScaler()


    # --------------------------------------------------------
    # Strategie MinMax:
    # sia plain sia block_sqrt iniziano da MinMax.
    # --------------------------------------------------------

    else:
        scaler = MinMaxScaler()


    x_train = scaler.fit_transform(
        train[columns].to_numpy(dtype=float)
    )

    x_target = scaler.transform(
        target[columns].to_numpy(dtype=float)
    )


    # --------------------------------------------------------
    # Strategia 3:
    # MinMax + bilanciamento per dimensionalità dei blocchi.
    #
    # dinamiche / sqrt(5)
    # statiche  / sqrt(numero_feature_statiche)
    # --------------------------------------------------------

    if (
            scheme == "minmax_block_sqrt"
            and subset
    ):

        scales = np.array(
            [math.sqrt(len(DYN))] * len(DYN)
            +
            [math.sqrt(len(subset))] * len(subset)
        )

        x_train = x_train / scales
        x_target = x_target / scales


    return x_train, x_target


# ============================================================
# Weighted vote e donor selection
# ============================================================

def donor_choice(
        train: pd.DataFrame,
        labels: np.ndarray,
        target_label: int,
        x_train: np.ndarray,
        x_target: np.ndarray,
):

    """
    Selezione donor finale.

    1. prendiamo soltanto le funzioni dello stesso cluster;
    2. consideriamo soltanto quelle con direzione x86/ARM;
    3. weighted vote 1/d^2;
    4. scegliamo la classe vincente;
    5. all'interno della classe vincente selezioniamo
       la funzione con distanza Manhattan minima.
    """


    # Membri dello stesso cluster del target.
    same_cluster = np.flatnonzero(
        labels == target_label
    )


    if not len(same_cluster):

        return (
            -1,
            "",
            "no_members",
        )


    preferences = (
        train
        .architecture_preference
        .to_numpy(str)
    )


    # Escludiamo architecture-independent dal voto.
    eligible = same_cluster[
        np.isin(
            preferences[same_cluster],
            DIRECTIONS,
        )
    ]


    if not len(eligible):

        return (
            -1,
            "",
            "no_directional_member",
        )


    # --------------------------------------------------------
    # Distanza Manhattan
    # --------------------------------------------------------

    distances = np.abs(
        x_train - x_target[0]
    ).sum(axis=1)


    # --------------------------------------------------------
    # Weighted vote:
    #
    # voto = 1 / (d + epsilon)^2
    # --------------------------------------------------------

    votes = {}

    for preference in DIRECTIONS:

        members = eligible[
            preferences[eligible] == preference
            ]

        votes[preference] = float(
            np.sum(
                1.0 /
                (
                        distances[members]
                        + 1e-9
                ) ** 2
            )
        )


    # Parità esatta -> astensione.
    if abs(
            votes[DIRECTIONS[0]]
            -
            votes[DIRECTIONS[1]]
    ) <= 1e-15:

        return (
            -1,
            "",
            "tie",
        )


    predicted = max(
        DIRECTIONS,
        key=lambda direction: votes[direction],
    )


    # --------------------------------------------------------
    # Consideriamo solo i donor appartenenti alla
    # classe architetturale vincente.
    # --------------------------------------------------------

    pool = eligible[
        preferences[eligible] == predicted
        ]


    names = (
        train
        .function_name
        .to_numpy(str)
    )


    # --------------------------------------------------------
    # Donor finale:
    # minima distanza Manhattan.
    #
    # In caso di stessa distanza:
    # ordinamento lessicografico del nome -> determinismo.
    # --------------------------------------------------------

    order = np.lexsort(
        (
            names[pool],
            distances[pool],
        )
    )


    donor_index = int(
        pool[order[0]]
    )


    return (
        donor_index,
        predicted,
        "selected",
    )


# ============================================================
# Esperimento LOFO
# ============================================================

def evaluate(
        df: pd.DataFrame,
        seeds: tuple[int, ...],
        n_init: int,
        verbose: bool = True,
):

    result = []


    # Rendiamo più controllabile il comportamento BLAS.
    with threadpool_limits(limits=1):

        for seed in seeds:

            start = time.monotonic()


            # ------------------------------------------------
            # Ogni funzione diventa una volta il target.
            # ------------------------------------------------

            for i in range(len(df)):

                target = df.iloc[[i]]

                # TARGET ESCLUSO COMPLETAMENTE DAL TRAINING.
                train = (
                    df.drop(index=i)
                    .reset_index(drop=True)
                )

                target_row = target.iloc[0]


                # --------------------------------------------
                # Tutte le combinazioni di feature.
                # --------------------------------------------

                for subset_name, subset in SUBSETS.items():


                    # ----------------------------------------
                    # Tutti gli scaler.
                    # ----------------------------------------

                    for scheme in SCALERS:


                        # Con zero feature statiche,
                        # dividere tutte le cinque dinamiche
                        # per sqrt(5) è un puro fattore comune.
                        #
                        # Non cambia:
                        # - K-Means;
                        # - ranking Manhattan.
                        #
                        # Quindi la configurazione sarebbe
                        # perfettamente ridondante.
                        if (
                                not subset
                                and
                                scheme == "minmax_block_sqrt"
                        ):
                            continue


                        config_name = (
                            f"{subset_name}__{scheme}"
                        )


                        # ------------------------------------
                        # PREPROCESSING TRAIN-ONLY.
                        # ------------------------------------

                        x_train, x_target = transform(
                            train,
                            target,
                            subset,
                            scheme,
                        )


                        # ------------------------------------
                        # K-Means K6.
                        # ------------------------------------

                        kmeans = KMeans(
                            n_clusters=6,
                            n_init=n_init,
                            random_state=seed,
                        )


                        labels = kmeans.fit_predict(
                            x_train
                        )


                        # Il target NON ha partecipato al fit.
                        # Viene assegnato successivamente.
                        target_cluster = int(
                            kmeans.predict(
                                x_target
                            )[0]
                        )


                        # ------------------------------------
                        # Weighted vote + Manhattan donor.
                        # ------------------------------------

                        (
                            donor_index,
                            prediction,
                            status,
                        ) = donor_choice(
                            train,
                            labels,
                            target_cluster,
                            x_train,
                            x_target,
                        )


                        row = {

                            "config":
                                config_name,

                            "subset":
                                subset_name,

                            "scaler":
                                scheme,

                            "seed":
                                seed,

                            "target_function":
                                target_row.function_name,

                            "target_preference":
                                target_row.architecture_preference,

                            "status":
                                status,

                            "donor_function":
                                "",

                            "prediction":
                                prediction,

                            "directional":
                                target_row.architecture_preference
                                in DIRECTIONS,

                            "raw_agreement":
                                np.nan,

                            "directional_agreement":
                                np.nan,

                            "regret_percent":
                                np.nan,

                            "log_speedup":
                                np.nan,

                            "same_language":
                                np.nan,
                        }


                        # ------------------------------------
                        # Valutazione POST-HOC.
                        #
                        # Solo qui utilizziamo la ground truth
                        # del target.
                        # ------------------------------------

                        if donor_index >= 0:

                            donor = train.iloc[
                                donor_index
                            ]


                            tx = float(
                                target_row.x86_duration_ms
                            )

                            ta = float(
                                target_row.arm_duration_ms
                            )


                            # Architettura realmente migliore
                            # senza soglia.
                            best_raw = (
                                "x86-preferred"
                                if tx <= ta
                                else "arm-preferred"
                            )


                            # Preferenza raw del donor.
                            donor_best = (
                                "x86-preferred"
                                if donor.x86_duration_ms
                                   <= donor.arm_duration_ms
                                else "arm-preferred"
                            )


                            # Latenza dell'architettura predetta.
                            chosen = (
                                tx
                                if prediction == "x86-preferred"
                                else ta
                            )


                            alternate = (
                                ta
                                if prediction == "x86-preferred"
                                else tx
                            )


                            row.update(

                                donor_function=
                                donor.function_name,

                                raw_agreement=
                                float(
                                    best_raw
                                    ==
                                    donor_best
                                ),

                                directional_agreement=
                                (
                                    float(
                                        prediction
                                        ==
                                        target_row
                                        .architecture_preference
                                    )
                                    if row["directional"]
                                    else np.nan
                                ),

                                regret_percent=
                                (
                                    (
                                            chosen /
                                            min(tx, ta)
                                            - 1
                                    ) * 100
                                    if row["directional"]
                                    else np.nan
                                ),

                                log_speedup=
                                (
                                    math.log(
                                        alternate /
                                        chosen
                                    )
                                    if row["directional"]
                                    else np.nan
                                ),

                                same_language=
                                float(
                                    target_row.language
                                    ==
                                    donor.language
                                ),
                            )


                        result.append(row)


            if verbose:

                elapsed = (
                        time.monotonic()
                        - start
                )

                print(
                    f"LOFO seed={seed} "
                    f"complete in {elapsed:.1f}s",
                    flush=True,
                )


    return pd.DataFrame(result)


# ============================================================
# Aggregazione risultati
# ============================================================

def summarize(rows: pd.DataFrame):

    stats = []


    for config, group in rows.groupby(
            "config",
            sort=True,
    ):

        selected = group[
            group.status == "selected"
            ]


        directional = selected[
            selected.directional
        ]


        stats.append({

            "config":
                config,

            "selected":
                len(selected),

            "total":
                len(group),

            "coverage":
                len(selected) / len(group),

            "directional_coverage":
                (
                        len(directional)
                        /
                        len(
                            group[
                                group.directional
                            ]
                        )
                ),

            "raw_agreement":
                selected.raw_agreement.mean(),

            "directional_agreement":
                directional
                .directional_agreement
                .mean(),

            "regret_percent":
                directional
                .regret_percent
                .mean(),

            "geomean_speedup":
                float(
                    np.exp(
                        directional
                        .log_speedup
                        .mean()
                    )
                ),

            "same_language":
                selected
                .same_language
                .mean(),
        })


    return (
        pd.DataFrame(stats)
        .sort_values(
            [
                "regret_percent",
                "config",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Bootstrap accoppiato
# ============================================================

def paired(
        rows: pd.DataFrame,
        base: str,
        nboot: int,
        seed: int,
):

    """
    Confronto di ogni configurazione con la pipeline finale.

    Importante:
    i cinque seed NON vengono considerati cinque osservazioni
    indipendenti.

    Prima:
        media dei seed per target.

    Poi:
        bootstrap delle funzioni target.
    """

    reference = (
        rows[
            rows.config == base
            ]
        .set_index(
            [
                "target_function",
                "seed",
            ]
        )
    )


    pairs = []

    rng = np.random.default_rng(seed)


    for config, group in rows.groupby(
            "config",
            sort=True,
    ):

        if config == base:
            continue


        other = group.set_index(
            [
                "target_function",
                "seed",
            ]
        )


        shared = (
            reference.index
            .intersection(other.index)
        )


        a = reference.loc[shared]
        b = other.loc[shared]


        # Supporto comune:
        # entrambe devono aver scelto un donor.
        good = (
                (a.status == "selected")
                &
                (b.status == "selected")
        )


        for field in [

            "raw_agreement",
            "directional_agreement",
            "regret_percent",
            "log_speedup",

        ]:


            valid = (
                    good
                    &
                    a[field].notna()
                    &
                    b[field].notna()
            )


            # ------------------------------------------------
            # Delta per target×seed
            # ------------------------------------------------

            delta = (
                    b.loc[valid, field]
                    -
                    a.loc[valid, field]
            )


            # ------------------------------------------------
            # Prima media dei cinque seed per funzione.
            # ------------------------------------------------

            delta = (
                delta
                .groupby(level=0)
                .mean()
                .to_numpy(float)
            )


            if not len(delta):
                continue


            # ------------------------------------------------
            # Bootstrap delle FUNZIONI.
            # ------------------------------------------------

            bootstrap_means = []


            for start in range(
                    0,
                    nboot,
                    1000,
            ):

                current = min(
                    1000,
                    nboot - start,
                    )


                draws = rng.integers(
                    0,
                    len(delta),
                    size=(
                        current,
                        len(delta),
                    ),
                )


                means = (
                    delta[draws]
                    .mean(axis=1)
                )


                bootstrap_means.extend(
                    means.tolist()
                )


            low, high = np.percentile(
                bootstrap_means,
                [
                    2.5,
                    97.5,
                ],
            )


            pairs.append({

                "comparison":
                    f"{config}_minus_{base}",

                "metric":
                    field,

                "n_targets":
                    len(delta),

                "n_common_target_seeds":
                    int(valid.sum()),

                "delta":
                    float(delta.mean()),

                "ci95_low":
                    float(low),

                "ci95_high":
                    float(high),
            })


    return pd.DataFrame(pairs)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--root",
        required=True,
        type=Path,
    )


    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )


    parser.add_argument(
        "--seeds",
        default="11,23,37,41,53",
    )


    parser.add_argument(
        "--lofo-input",
        type=Path,
        default=None,
        help=(
            "Usa un CSV LOFO già calcolato "
            "invece di rifare K-Means."
        ),
    )


    parser.add_argument(
        "--n-init",
        type=int,
        default=50,
    )


    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=5000,
    )


    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20261009,
    )


    args = parser.parse_args()


    seeds = tuple(
        int(seed)
        for seed
        in args.seeds.split(",")
    )


    df, hashes = load(
        args.root
    )


    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Calcolo reale oppure aggregazione di risultati esistenti.
    # --------------------------------------------------------

    if args.lofo_input:

        rows = pd.read_csv(
            args.lofo_input
        )

    else:

        rows = evaluate(
            df,
            seeds,
            args.n_init,
        )


    # --------------------------------------------------------
    # Numero atteso:
    #
    # 59 target
    # × 5 seed
    # × 20 configurazioni
    # = 5900
    # --------------------------------------------------------

    expected = (
            59
            *
            len(seeds)
            *
            (
                    len(SUBSETS)
                    * len(SCALERS)
                    - 1
            )
    )


    if (
            len(rows) != expected
            or
            rows.duplicated(
                [
                    "config",
                    "seed",
                    "target_function",
                ]
            ).any()
    ):

        raise ValueError(
            "LOFO incompleto: "
            f"attese {expected} righe, "
            f"trovate {len(rows)}"
        )


    summary = summarize(
        rows
    )


    if BASE not in set(
            summary.config
    ):

        raise RuntimeError(
            "Baseline finale assente"
        )


    comparisons = paired(
        rows,
        BASE,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    rows.to_csv(
        args.output_dir
        /
        "lofo_static_scaling_59.csv",
        index=False,
        )


    summary.to_csv(
        args.output_dir
        /
        "summary_static_scaling_59.csv",
        index=False,
        )


    comparisons.to_csv(
        args.output_dir
        /
        "paired_vs_final_static_scaling_59.csv",
        index=False,
        )


    manifest = {

        "input_hashes":
            hashes,

        "seeds":
            seeds,

        "k":
            6,

        "n_init":
            args.n_init,

        "static_subsets":
            SUBSETS,

        "scalers":
            SCALERS,

        "baseline":
            BASE,

        "bootstrap_replicates":
            args.bootstrap_replicates,

        "bootstrap_seed":
            args.bootstrap_seed,

        "note":
            (
                "All 59 target function outcomes are LOFO; "
                "pairwise differences restricted to common "
                "selected support. Exploratory CIs, "
                "no multiplicity adjustment."
            ),
    }


    (
            args.output_dir
            /
            "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # Report Markdown
    # --------------------------------------------------------

    report = [

        "# Ablation 59 target: caratteristiche e scaling, "
        "K6 + weighted donor",

        "",

        (
            "Analisi esplorativa; nessuna configurazione nuova "
            "va dichiarata definitiva senza replay UCB1 e "
            "validazione indipendente."
        ),

        "",

        "## Confronto sintetico",

        "",

        summary.to_markdown(
            index=False,
            floatfmt=".4f",
        ),

        "",

        (
            "## Intervalli accoppiati vs rappresentazione finale "
            "(token + function count + MinMax blocchi /sqrt(p))"
        ),

        "",

        comparisons[
            comparisons.metric.isin(
                [
                    "directional_agreement",
                    "regret_percent",
                ]
            )
        ].to_markdown(
            index=False,
            floatfmt=".4f",
        ),

        "",

        "## Note metodologiche",

        "",

        (
            "- LOFO train-only con 5 seed, "
            "K6, n_init=50."
        ),

        (
            "- La funzione target è esclusa dal catalogo donor "
            "e dal preprocessing."
        ),

        (
            "- La ground truth del target viene usata soltanto "
            "nella valutazione."
        ),

        (
            "- MinMax plain e StandardScaler sono le alternative "
            "standard."
        ),

        (
            "- MinMax a blocchi /sqrt(p) riproduce "
            "la pipeline attuale."
        ),

        (
            "- Il bootstrap usa le funzioni come unità "
            "di ricampionamento dopo la media sui seed."
        ),

        (
            "- Gli intervalli sono esplorativi e non corretti "
            "per confronti multipli."
        ),
    ]


    (
            args.output_dir
            /
            "REPORT_STATIC_SCALING_59.md"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )


    print(
        "PASS:",
        len(df),
        "target,",
        len(summary),
        "configs,",
        len(rows),
        "LOFO rows",
    )


    print()


    print(
        summary[
            [
                "config",
                "coverage",
                "directional_agreement",
                "regret_percent",
                "geomean_speedup",
            ]
        ].to_string(
            index=False
        )
    )


    print()


    print(
        "Output:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()