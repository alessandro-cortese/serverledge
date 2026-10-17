import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from compare_59_clustering_and_plots import (
    prepare,
    PAPER5,
    STATIC2,
    SEEDS,
    DIR,
    LANGS,
)


FEATURES = PAPER5 + STATIC2

EPSILON = 1e-9

# Non possono diventare donor.
EXCLUDED_DONORS = {
    "amd_faster",
    "arm_faster",
}

# Servono soltanto per produrre anche la vista "primary".
EXCLUDED_PRIMARY_TARGETS = {
    "amd_faster",
    "arm_faster",
}


def raw_preference(row):
    """
    Best architecture senza neutral band.
    Usata soltanto come diagnostica aggiuntiva.
    """
    return (
        "x86-preferred"
        if float(row["x86_duration_ms"])
           <= float(row["arm_duration_ms"])
        else "arm-preferred"
    )


def build_spaces(
        train: pd.DataFrame,
        target: pd.DataFrame,
        include_language: bool,
):
    """
    Costruisce DUE rappresentazioni, in ordine temporale:

    1. plain:
       7 feature MinMax.
       Questa rappresentazione verrà usata successivamente
       per la Manhattan donor distance.

    2. cluster:
       plain + block scaling 1/sqrt(p).
       Questa rappresentazione viene usata SOLO da KMeans.

    Se include_language=True, il one-hot del linguaggio viene
    aggiunto SOLO alla rappresentazione di clustering.

    La distanza donor rimane quindi sempre plain Manhattan
    sulle stesse 7 feature numeriche.
    """

    scaler = MinMaxScaler()

    x_plain = scaler.fit_transform(
        train[FEATURES].to_numpy(float)
    )

    target_plain = scaler.transform(
        target[FEATURES].to_numpy(float)
    )

    scales = np.array(
        [math.sqrt(len(PAPER5))] * len(PAPER5)
        + [math.sqrt(len(STATIC2))] * len(STATIC2),
        dtype=float,
        )

    x_cluster = x_plain / scales
    target_cluster = target_plain / scales

    if include_language:

        def one_hot(frame):
            return np.column_stack(
                [
                    (
                            frame["language"].to_numpy()
                            == language
                    ).astype(float)
                    for language in LANGS
                ]
            )

        # Language è un blocco one-hot di 3 coordinate.
        language_scale = math.sqrt(len(LANGS))

        x_language = one_hot(train) / language_scale
        target_language = one_hot(target) / language_scale

        x_cluster = np.column_stack(
            (
                x_cluster,
                x_language,
            )
        )

        target_cluster = np.column_stack(
            (
                target_cluster,
                target_language,
            )
        )

    return (
        x_plain,
        target_plain,
        x_cluster,
        target_cluster,
    )


def choose_donor(
        train: pd.DataFrame,
        x_plain: np.ndarray,
        target_plain: np.ndarray,
        cluster_labels: np.ndarray,
        target_cluster: int,
        strategy: str,
):
    """
    Donor selection DOPO il clustering.

    Candidate:
        - stesso cluster della target;
        - classe direzionale x86/ARM;
        - amd_faster e arm_faster esclusi.

    Distanza:
        Manhattan plain sulle 7 feature MinMax.

    strategy:
        majority
            voto uniforme della classe direzionale.

        weighted
            voto 1/(d + epsilon)^2.

    Dopo la predizione della classe:
        nearest donor all'interno della classe vincente.
    """

    same_cluster = np.flatnonzero(
        cluster_labels == target_cluster
    )

    if len(same_cluster) == 0:
        return None, "", "no_cluster_member", np.nan

    names = (
        train["function_name"]
        .astype(str)
        .to_numpy()
    )

    prefs = (
        train["architecture_preference"]
        .astype(str)
        .to_numpy()
    )

    distances = np.abs(
        x_plain - target_plain[0]
    ).sum(axis=1)

    excluded = np.isin(
        names[same_cluster],
        list(EXCLUDED_DONORS),
    )

    directional = same_cluster[
        np.isin(
            prefs[same_cluster],
            DIR,
        )
        & ~excluded
        ]

    if len(directional) == 0:
        return (
            None,
            "",
            "no_directional_member",
            np.nan,
        )

    if strategy == "majority":

        scores = {
            direction: float(
                np.sum(
                    prefs[directional]
                    == direction
                )
            )
            for direction in DIR
        }

    elif strategy == "weighted":

        scores = {}

        for direction in DIR:

            members = directional[
                prefs[directional]
                == direction
                ]

            if len(members) == 0:
                scores[direction] = 0.0
            else:
                scores[direction] = float(
                    np.sum(
                        1.0
                        / (
                                distances[members]
                                + EPSILON
                        )
                        ** 2
                    )
                )

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    if math.isclose(
            scores[DIR[0]],
            scores[DIR[1]],
            rel_tol=0.0,
            abs_tol=1e-15,
    ):
        return (
            None,
            "",
            "tied_vote",
            np.nan,
        )

    predicted = max(
        DIR,
        key=lambda direction:
        scores[direction],
    )

    pool = directional[
        prefs[directional]
        == predicted
        ]

    if len(pool) == 0:
        return (
            None,
            predicted,
            "no_candidate_in_predicted_class",
            np.nan,
        )

    # Tie-break deterministico:
    # prima distanza, poi nome funzione.
    order = np.lexsort(
        (
            names[pool],
            distances[pool],
        )
    )

    selected_idx = int(
        pool[order[0]]
    )

    return (
        selected_idx,
        predicted,
        "selected",
        float(distances[selected_idx]),
    )


def outcome(
        target,
        donor,
        predicted,
):
    """
    Metriche downstream della scelta architetturale.
    """

    tx = float(
        target["x86_duration_ms"]
    )

    ta = float(
        target["arm_duration_ms"]
    )

    dx = float(
        donor["x86_duration_ms"]
    )

    da = float(
        donor["arm_duration_ms"]
    )

    best = min(tx, ta)

    chosen = (
        tx
        if predicted == "x86-preferred"
        else ta
    )

    alternative = (
        ta
        if predicted == "x86-preferred"
        else tx
    )

    target_pref = str(
        target["architecture_preference"]
    )

    return {
        "is_directional":
            target_pref in DIR,

        "directional_agreement":
            (
                float(
                    predicted
                    == target_pref
                )
                if target_pref in DIR
                else np.nan
            ),

        "raw_agreement":
            float(
                raw_preference(donor)
                == raw_preference(target)
            ),

        "regret_percent":
            (
                    chosen / best - 1.0
            )
            * 100.0,

        "realized_speedup":
            alternative / chosen,

        "multiplicative_mismatch":
            math.exp(
                abs(
                    math.log(ta / tx)
                    - math.log(da / dx)
                )
            ),

        "same_language":
            float(
                donor["language"]
                == target["language"]
            ),
    }


def run_sweep(
        df: pd.DataFrame,
        seeds,
        k_values,
):
    rows = []

    strategies = (
        "majority",
        "weighted",
    )

    language_modes = (
        ("no_language", False),
        ("with_language", True),
    )

    for seed in seeds:

        print(
            f"===== SEED {seed} =====",
            flush=True,
        )

        for target_idx in range(len(df)):

            target = df.iloc[
                [target_idx]
            ].copy()

            train = (
                df.drop(index=target_idx)
                .reset_index(drop=True)
            )

            target_name = str(
                target.iloc[0][
                    "function_name"
                ]
            )

            target_pref = str(
                target.iloc[0][
                    "architecture_preference"
                ]
            )

            for (
                    language_name,
                    include_language,
            ) in language_modes:

                (
                    x_plain,
                    target_plain,
                    x_cluster,
                    target_cluster_features,
                ) = build_spaces(
                    train,
                    target,
                    include_language,
                )

                for k in k_values:

                    model = KMeans(
                        n_clusters=k,
                        n_init=50,
                        random_state=seed,
                    )

                    cluster_labels = (
                        model.fit_predict(
                            x_cluster
                        )
                    )

                    target_cluster = int(
                        model.predict(
                            target_cluster_features
                        )[0]
                    )

                    for strategy in strategies:

                        (
                            donor_idx,
                            predicted,
                            status,
                            donor_distance,
                        ) = choose_donor(
                            train=train,
                            x_plain=x_plain,
                            target_plain=target_plain,
                            cluster_labels=
                            cluster_labels,
                            target_cluster=
                            target_cluster,
                            strategy=strategy,
                        )

                        row = {
                            "seed": seed,
                            "k": k,
                            "language_mode":
                                language_name,
                            "strategy":
                                strategy,

                            "target_function":
                                target_name,

                            "target_language":
                                str(
                                    target.iloc[0][
                                        "language"
                                    ]
                                ),

                            "target_preference":
                                target_pref,

                            "target_cluster":
                                target_cluster,

                            "status":
                                status,

                            "prediction":
                                predicted,

                            "donor_function":
                                "",

                            "donor_distance":
                                donor_distance,

                            "is_directional":
                                target_pref in DIR,

                            "directional_agreement":
                                np.nan,

                            "raw_agreement":
                                np.nan,

                            "regret_percent":
                                np.nan,

                            "realized_speedup":
                                np.nan,

                            "multiplicative_mismatch":
                                np.nan,

                            "same_language":
                                np.nan,
                        }

                        if donor_idx is not None:

                            donor = train.iloc[
                                donor_idx
                            ]

                            row[
                                "donor_function"
                            ] = str(
                                donor[
                                    "function_name"
                                ]
                            )

                            row.update(
                                outcome(
                                    target.iloc[0],
                                    donor,
                                    predicted,
                                )
                            )

                        rows.append(row)

    result = pd.DataFrame(rows)

    if (
            (
                    result[
                        "donor_function"
                    ]
                    == result[
                        "target_function"
                    ]
            )
            & (
                    result["status"]
                    == "selected"
            )
    ).any():
        raise AssertionError(
            "Target leaked into donor"
        )

    if (
            result[
                "donor_function"
            ]
                    .isin(EXCLUDED_DONORS)
                    .any()
    ):
        raise AssertionError(
            "Synthetic donor selected"
        )

    return result


def summarize(
        data: pd.DataFrame,
        scope_name: str,
):
    if scope_name == "all59":

        scoped = data.copy()

    elif scope_name == "primary57":

        scoped = data[
            ~data[
                "target_function"
            ].isin(
                EXCLUDED_PRIMARY_TARGETS
            )
        ].copy()

    else:
        raise ValueError(scope_name)

    rows = []

    grouping = [
        "seed",
        "k",
        "language_mode",
        "strategy",
    ]

    for keys, group in scoped.groupby(
            grouping,
            sort=True,
    ):

        (
            seed,
            k,
            language_mode,
            strategy,
        ) = keys

        selected = group[
            group["status"]
            == "selected"
            ]

        target_directional = group[
            group["is_directional"]
            == True
            ]

        selected_directional = selected[
            selected["is_directional"]
            == True
            ]

        rows.append(
            {
                "scope":
                    scope_name,

                "seed":
                    int(seed),

                "k":
                    int(k),

                "language_mode":
                    language_mode,

                "strategy":
                    strategy,

                "target_count":
                    len(group),

                "directional_target_count":
                    len(
                        target_directional
                    ),

                "selection_coverage":
                    len(selected)
                    / len(group),

                "directional_coverage":
                    (
                        len(
                            selected_directional
                        )
                        / len(
                            target_directional
                        )
                        if len(
                            target_directional
                        )
                        else np.nan
                    ),

                "raw_agreement":
                    float(
                        selected[
                            "raw_agreement"
                        ].mean()
                    )
                    if len(selected)
                    else np.nan,

                "directional_agreement":
                    float(
                        selected_directional[
                            "directional_agreement"
                        ].mean()
                    )
                    if len(
                        selected_directional
                    )
                    else np.nan,

                "geomean_speedup":
                    float(
                        np.exp(
                            np.log(
                                selected_directional[
                                    "realized_speedup"
                                ]
                            ).mean()
                        )
                    )
                    if len(
                        selected_directional
                    )
                    else np.nan,

                "regret_percent":
                    float(
                        selected_directional[
                            "regret_percent"
                        ].mean()
                    )
                    if len(
                        selected_directional
                    )
                    else np.nan,

                "same_language":
                    float(
                        selected[
                            "same_language"
                        ].mean()
                    )
                    if len(selected)
                    else np.nan,
            }
        )

    per_seed = pd.DataFrame(rows)

    summary = (
        per_seed
        .groupby(
            [
                "scope",
                "k",
                "language_mode",
                "strategy",
            ],
            as_index=False,
        )
        .agg(
            seed_count=(
                "seed",
                "nunique",
            ),

            target_count=(
                "target_count",
                "first",
            ),

            directional_target_count=(
                "directional_target_count",
                "first",
            ),

            selection_coverage=(
                "selection_coverage",
                "mean",
            ),

            directional_coverage=(
                "directional_coverage",
                "mean",
            ),

            raw_agreement=(
                "raw_agreement",
                "mean",
            ),

            directional_agreement=(
                "directional_agreement",
                "mean",
            ),

            geomean_speedup=(
                "geomean_speedup",
                "mean",
            ),

            regret_percent=(
                "regret_percent",
                "mean",
            ),

            same_language=(
                "same_language",
                "mean",
            ),
        )
    )

    return per_seed, summary


def self_check(summary):
    """
    K=6, no-language deve riprodurre
    la validazione già esistente.

    Se non passa, NON utilizzare
    lo sweep per le slide.
    """

    expected = {
        "majority": {
            "selection_coverage":
                0.9830508474576272,

            "directional_coverage":
                0.98,

            "directional_agreement":
                0.6693877551020407,

            "geomean_speedup":
                1.1440512931403692,

            "regret_percent":
                7.015915162141079,
        },

        "weighted": {
            "selection_coverage":
                0.9830508474576272,

            "directional_coverage":
                0.98,

            "directional_agreement":
                0.7306122448979593,

            "geomean_speedup":
                1.2029908404238079,

            "regret_percent":
                3.726153324592535,
        },
    }

    for strategy, metrics in expected.items():

        row = summary[
            (summary["scope"] == "all59")
            & (summary["k"] == 6)
            & (
                    summary["language_mode"]
                    == "no_language"
            )
            & (
                    summary["strategy"]
                    == strategy
            )
            ]

        if len(row) != 1:
            raise AssertionError(
                f"Missing self-check row "
                f"for {strategy}"
            )

        row = row.iloc[0]

        for metric, value in metrics.items():

            actual = float(
                row[metric]
            )

            tolerance = (
                1e-5
                if metric
                   == "geomean_speedup"
                else 1e-9
            )

            if not math.isclose(
                    actual,
                    value,
                    rel_tol=0.0,
                    abs_tol=tolerance,
            ):
                raise AssertionError(
                    f"SELF CHECK FAILED: "
                    f"{strategy} {metric}: "
                    f"{actual} != {value}"
                )

    print(
        "implementation_self_check=PASS"
    )


def slide_tables(summary):
    """
    Tabelle già pronte per le slide.
    Usiamo all59 per mantenere continuità
    con le analisi precedenti; produciamo
    comunque anche primary57 nel CSV completo.
    """

    base = summary[
        summary["scope"]
        == "all59"
        ].copy()

    # --------------------------------------
    # Majority vs weighted, NO LANGUAGE
    # --------------------------------------

    strategies = base[
        base["language_mode"]
        == "no_language"
        ]

    majority = strategies[
        strategies["strategy"]
        == "majority"
        ][
        [
            "k",
            "selection_coverage",
            "directional_coverage",
            "directional_agreement",
            "geomean_speedup",
            "regret_percent",
        ]
    ].rename(
        columns={
            "selection_coverage":
                "majority_selection_coverage",

            "directional_coverage":
                "majority_directional_coverage",

            "directional_agreement":
                "majority_directional_agreement",

            "geomean_speedup":
                "majority_geomean_speedup",

            "regret_percent":
                "majority_regret_percent",
        }
    )

    weighted = strategies[
        strategies["strategy"]
        == "weighted"
        ][
        [
            "k",
            "selection_coverage",
            "directional_coverage",
            "directional_agreement",
            "geomean_speedup",
            "regret_percent",
        ]
    ].rename(
        columns={
            "selection_coverage":
                "weighted_selection_coverage",

            "directional_coverage":
                "weighted_directional_coverage",

            "directional_agreement":
                "weighted_directional_agreement",

            "geomean_speedup":
                "weighted_geomean_speedup",

            "regret_percent":
                "weighted_regret_percent",
        }
    )

    strategy_table = (
        majority
        .merge(
            weighted,
            on="k",
            validate="one_to_one",
        )
        .sort_values("k")
        .reset_index(drop=True)
    )

    # --------------------------------------
    # No Language vs Language,
    # weighted vote fixed
    # --------------------------------------

    languages = base[
        base["strategy"]
        == "weighted"
        ]

    no_lang = languages[
        languages["language_mode"]
        == "no_language"
        ][
        [
            "k",
            "selection_coverage",
            "directional_coverage",
            "directional_agreement",
            "geomean_speedup",
            "regret_percent",
            "same_language",
        ]
    ].rename(
        columns={
            "selection_coverage":
                "no_lang_selection_coverage",

            "directional_coverage":
                "no_lang_directional_coverage",

            "directional_agreement":
                "no_lang_directional_agreement",

            "geomean_speedup":
                "no_lang_geomean_speedup",

            "regret_percent":
                "no_lang_regret_percent",

            "same_language":
                "no_lang_same_language",
        }
    )

    with_lang = languages[
        languages["language_mode"]
        == "with_language"
        ][
        [
            "k",
            "selection_coverage",
            "directional_coverage",
            "directional_agreement",
            "geomean_speedup",
            "regret_percent",
            "same_language",
        ]
    ].rename(
        columns={
            "selection_coverage":
                "lang_selection_coverage",

            "directional_coverage":
                "lang_directional_coverage",

            "directional_agreement":
                "lang_directional_agreement",

            "geomean_speedup":
                "lang_geomean_speedup",

            "regret_percent":
                "lang_regret_percent",

            "same_language":
                "lang_same_language",
        }
    )

    language_table = (
        no_lang
        .merge(
            with_lang,
            on="k",
            validate="one_to_one",
        )
        .sort_values("k")
        .reset_index(drop=True)
    )

    return (
        strategy_table,
        language_table,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seeds",
        default="11,23,37,41,53",
    )

    parser.add_argument(
        "--k-min",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--k-max",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    seeds = tuple(
        int(value)
        for value
        in args.seeds.split(",")
        if value.strip()
    )

    k_values = tuple(
        range(
            args.k_min,
            args.k_max + 1,
            )
    )

    root = args.root.resolve()

    out = args.output_dir.resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, paths = prepare(root)

    if len(df) != 59:
        raise AssertionError(
            f"Expected 59 functions, "
            f"got {len(df)}"
        )

    print(
        f"eligible_functions={len(df)}"
    )

    print(
        f"seeds={seeds}"
    )

    print(
        f"k_values={k_values}"
    )

    data = run_sweep(
        df=df,
        seeds=seeds,
        k_values=k_values,
    )

    data.to_csv(
        out
        / "factorial_per_target_seed.csv",
        index=False,
        )

    all59_seed, all59_summary = (
        summarize(
            data,
            "all59",
        )
    )

    primary_seed, primary_summary = (
        summarize(
            data,
            "primary57",
        )
    )

    per_seed = pd.concat(
        [
            all59_seed,
            primary_seed,
        ],
        ignore_index=True,
    )

    summary = pd.concat(
        [
            all59_summary,
            primary_summary,
        ],
        ignore_index=True,
    )

    per_seed.to_csv(
        out
        / "factorial_per_seed.csv",
        index=False,
        )

    summary.to_csv(
        out
        / "factorial_summary.csv",
        index=False,
        )

    self_check(summary)

    (
        strategy_table,
        language_table,
    ) = slide_tables(summary)

    strategy_table.to_csv(
        out
        / "slide_strategy_by_k_no_language.csv",
        index=False,
        )

    language_table.to_csv(
        out
        / "slide_language_by_k_weighted.csv",
        index=False,
        )

    manifest = {
        "eligible_functions": 59,

        "primary_functions": 57,

        "seeds": list(seeds),

        "k_values": list(k_values),

        "n_init": 50,

        "threshold_percent": 2.5,

        "features": FEATURES,

        "clustering": {
            "algorithm": "KMeans",

            "dynamic_scaling":
                "1/sqrt(5)",

            "static_scaling":
                "1/sqrt(2)",

            "language_modes": {
                "no_language":
                    "no language block",

                "with_language":
                    (
                        "one-hot "
                        "go/python/nodejs "
                        "/ sqrt(3)"
                    ),
            },
        },

        "donor_selection": {
            "candidate_restriction":
                (
                    "same cluster + "
                    "directional class"
                ),

            "distance":
                (
                    "plain Manhattan "
                    "on 7 MinMax features"
                ),

            "strategies": {
                "majority":
                    (
                        "uniform directional "
                        "class vote"
                    ),

                "weighted":
                    (
                        "inverse Manhattan "
                        "1/(d+1e-9)^2"
                    ),
            },

            "excluded_donors":
                sorted(
                    EXCLUDED_DONORS
                ),
        },

        "target_leakage":
            (
                "LOFO: target excluded "
                "from scaler fit, "
                "KMeans fit and donor catalog"
            ),

        "inputs": {
            key: str(path)
            for key, path
            in paths.items()
        },
    }

    (
            out
            / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        )

    print()

    print(
        "===== STRATEGY BY K "
        "(NO LANGUAGE) ====="
    )

    print(
        strategy_table.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    print()

    print(
        "===== LANGUAGE BY K "
        "(WEIGHTED) ====="
    )

    print(
        language_table.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    print()

    print("saved:")
    for path in [
        out
        / "factorial_per_target_seed.csv",

        out
        / "factorial_per_seed.csv",

        out
        / "factorial_summary.csv",

        out
        / "slide_strategy_by_k_no_language.csv",

        out
        / "slide_language_by_k_weighted.csv",

        out
        / "manifest.json",
    ]:
        print(path)


if __name__ == "__main__":
    main()