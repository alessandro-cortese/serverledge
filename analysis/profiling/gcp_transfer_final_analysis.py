#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


PAPER5 = [
    "page_faults_delta",
    "utilized_cpus",
    "free_memory_mb",
    "cpu_user_delta_ms",
    "cpu_kernel_delta_ms",
]

SELECTION_SCHEMA_VERSION = 1
QUERY_SCHEMA_VERSION = 1


SENTINEL_FUNCTIONS = {
    "amd_faster",
    "arm_faster",
    "amd-faster",
    "arm-faster",
}


def is_special_function(function_name: str) -> bool:
    name = function_name.strip().lower()
    return name.startswith("twin-") or name in SENTINEL_FUNCTIONS

def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_float(value, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{field} non numerico: {value!r}") from exc

    if not math.isfinite(result):
        raise SystemExit(f"{field} non finito: {value!r}")

    return result


def parse_positive_float(value, field: str) -> float:
    result = parse_float(value, field)

    if result <= 0.0:
        raise SystemExit(f"{field} deve essere > 0")

    return result


def load_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise SystemExit(f"CSV senza header: {path}")

        fields = list(reader.fieldnames)
        rows = list(reader)

    return fields, rows


def require_paper5_columns(fields: list[str], source: Path) -> None:
    missing = [feature for feature in PAPER5 if feature not in fields]

    if missing:
        raise SystemExit(f"{source}: mancano feature PAPER-5: {missing}")


def feature_vector(row: dict) -> list[float]:
    return [parse_float(row[name], name) for name in PAPER5]


def filter_profile(args: argparse.Namespace) -> None:
    source = Path(args.raw)
    eligible: list[tuple[dict, float]] = []

    for raw in source.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue

        sample = json.loads(raw)

        if sample.get("function_name") != args.target:
            continue
        if sample.get("machine_tag") != args.machine_tag:
            continue
        if sample.get("warm_start") is not True:
            continue
        if sample.get("execution_succeeded") is not True:
            continue
        if (sample.get("eligibility") or {}).get("performance_analysis") is not True:
            continue

        profile = sample.get("profile") or {}

        if profile.get("valid") is not True or profile.get("exclusive_container") is not True:
            continue

        duration_ms = parse_float(
            (sample.get("timing") or {}).get("duration_ms", 0.0),
            "duration_ms",
        )

        if duration_ms <= 0.0:
            continue

        eligible.append((sample, duration_ms))

    if len(eligible) < args.samples:
        raise SystemExit(f"campioni eleggibili insufficienti: {len(eligible)}/{args.samples}")

    selected = eligible[-args.samples:]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        "".join(json.dumps(sample, separators=(",", ":")) + "\n" for sample, _ in selected),
        encoding="utf-8",
    )

    rewards = [-math.log(duration_ms) for _, duration_ms in selected]

    print(
        json.dumps(
            {
                "selected_samples": len(selected),
                "target_reference_mean_reward": statistics.fmean(rewards),
            },
            sort_keys=True,
        )
    )


def select_donor(args: argparse.Namespace) -> None:
    historical_path = Path(args.historical_profiles)
    target_path = Path(args.target_profiles)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    historical_fields, historical_rows = load_csv(historical_path)
    target_fields, target_rows = load_csv(target_path)

    require_paper5_columns(historical_fields, historical_path)
    require_paper5_columns(target_fields, target_path)

    # ------------------------------------------------------------------
    # LOFO: il target viene escluso dal corpus usato per fit scaler/KMeans.
    # Solo profili dell'architettura di riferimento e CPU=1.
    # ------------------------------------------------------------------

    donors = []

    for row in historical_rows:
        if row.get("function_name") == args.target:
            continue

        if row.get("machine_tag") != args.reference_tag:
            continue

        configured_cpus = parse_positive_float(row.get("configured_cpus"), "configured_cpus")

        if not math.isclose(configured_cpus, 1.0, rel_tol=0.0, abs_tol=1e-12):
            continue

        donors.append(row)

    if len(donors) < args.k:
        raise SystemExit(
            f"corpus LOFO troppo piccolo per KMeans k={args.k}: {len(donors)} righe"
        )

    # ------------------------------------------------------------------
    # Deve esistere una sola riga aggregata del target corrente.
    # ------------------------------------------------------------------

    target_matches = []

    for row in target_rows:
        if row.get("function_name") != args.target:
            continue

        if row.get("machine_tag") != args.reference_tag:
            continue

        configured_cpus = parse_positive_float(row.get("configured_cpus"), "configured_cpus")
        configured_memory_mb = int(float(row.get("configured_memory_mb", 0)))

        if not math.isclose(configured_cpus, 1.0, rel_tol=0.0, abs_tol=1e-12):
            continue

        if configured_memory_mb != args.configured_memory_mb:
            continue

        target_matches.append(row)

    if len(target_matches) != 1:
        raise SystemExit(f"target FunctionProfile compatibili: {len(target_matches)}")

    target_row = target_matches[0]

    # ------------------------------------------------------------------
    # PAPER-5 + MinMax.
    # framework_runtime_ms può essere presente nei CSV, ma viene ignorata.
    # ------------------------------------------------------------------

    donor_raw_matrix = np.asarray(
        [feature_vector(row) for row in donors],
        dtype=np.float64,
    )

    target_raw_vector = np.asarray(
        [feature_vector(target_row)],
        dtype=np.float64,
    )

    scaler = MinMaxScaler()
    donor_matrix = scaler.fit_transform(donor_raw_matrix)
    target_vector = scaler.transform(target_raw_vector)[0]

    # ------------------------------------------------------------------
    # Clustering congelato:
    # MinMax + KMeans k=5, n_init=50, seed=42.
    # ------------------------------------------------------------------

    kmeans = KMeans(
        n_clusters=args.k,
        n_init=args.n_init,
        random_state=args.random_state,
    )

    labels = kmeans.fit_predict(donor_matrix)

    # Nuovo target -> centroide più vicino con distanza Euclidea.
    centroid_distances = np.linalg.norm(kmeans.cluster_centers_ - target_vector, axis=1)
    target_cluster = int(np.argmin(centroid_distances))

    # ------------------------------------------------------------------
    # Donor candidati:
    # - stesso cluster;
    # - stessa configurazione CPU/memoria;
    # - ranking Manhattan.
    # ------------------------------------------------------------------

    candidates = []

    excluded_special_donors = []

    for row, label, vector in zip(donors, labels.tolist(), donor_matrix.tolist()):
        function_name = row["function_name"].strip()
        configured_cpus = parse_positive_float(row.get("configured_cpus"), "configured_cpus")
        configured_memory_mb = int(float(row.get("configured_memory_mb", 0)))

        if label != target_cluster:
            continue

        if is_special_function(function_name):
            excluded_special_donors.append(function_name)
            continue

        if not math.isclose(configured_cpus, 1.0, rel_tol=0.0, abs_tol=1e-12):
            continue

        if configured_memory_mb != args.configured_memory_mb:
            continue

        distance = float(
            np.abs(np.asarray(vector, dtype=np.float64) - target_vector).sum()
        )

        candidates.append(
            {
                "function_name": row["function_name"],
                "configured_cpus": configured_cpus,
                "configured_memory_mb": configured_memory_mb,
                "cluster_label": int(label),
                "distance": distance,
            }
        )

    candidates.sort(key=lambda item: (item["distance"], item["function_name"]))

    query_id = f"query-{args.run_id}"
    selection_run_id = f"selection-{args.run_id}"

    # ------------------------------------------------------------------
    # Query PAPER-5.
    #
    # Il runtime Go usa soltanto schema_version/query_id/function_name;
    # gli altri campi restano nell'artefatto per tracciabilità scientifica.
    # ------------------------------------------------------------------

    query = {
        "schema_version": QUERY_SCHEMA_VERSION,
        "query_id": query_id,
        "function_name": args.target,
        "configured_cpus": 1.0,
        "configured_memory_mb": args.configured_memory_mb,
        "sample_count": int(float(target_row.get("sample_count", 0))),
        "profile_machine_tag": args.reference_tag,
        "aggregation": target_row.get("aggregation", "median"),
        "scaler": "minmax",
        "feature_names": PAPER5,
        "feature_vector": [float(value) for value in target_vector.tolist()],
        "cluster_label": target_cluster,
    }

    atomic_json(output_dir / "transfer-query.json", query)

    # ------------------------------------------------------------------
    # Ranking completo per audit.
    # ------------------------------------------------------------------

    ranking_path = output_dir / "selection.csv"

    with ranking_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "selection_run_id",
            "query_id",
            "rank",
            "function_name",
            "configured_cpus",
            "configured_memory_mb",
            "cluster_label",
            "distance",
            "within_threshold",
            "selected",
        ]

        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for rank, candidate in enumerate(candidates, start=1):
            within_threshold = candidate["distance"] <= args.max_distance
            selected = rank == 1 and within_threshold

            writer.writerow(
                {
                    "selection_run_id": selection_run_id,
                    "query_id": query_id,
                    "rank": rank,
                    **candidate,
                    "within_threshold": str(within_threshold).lower(),
                    "selected": str(selected).lower(),
                }
            )

    # ------------------------------------------------------------------
    # Decisione.
    # ------------------------------------------------------------------

    if not candidates:
        status = "no-transfer"
        reason = "no_same_cluster_candidates"
        selected_donor = None

    elif candidates[0]["distance"] > args.max_distance:
        status = "no-transfer"
        reason = "distance_threshold_exceeded"
        selected_donor = None

    else:
        status = "selected"
        reason = ""
        selected_donor = {
            "function_name": candidates[0]["function_name"],
            "distance": candidates[0]["distance"],
        }

    # ------------------------------------------------------------------
    # Artefatto runtime-facing.
    #
    # Questi sono i campi che il validatore Go richiede:
    # - schema_version=1
    # - selection_run_id
    # - query schema/query_id/function_name
    # - distance supportata
    # - max_distance > 0
    # - configuration_match_required=true
    # - bandit_prior_materialized=false
    # - bandit_prior=null
    # - selected donor valido se status=selected
    # ------------------------------------------------------------------

    selection = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_run_id": selection_run_id,
        "status": status,
        "reason": reason,
        "query": query,
        "selection_policy": {
            "distance": "manhattan",
            "max_distance": args.max_distance,
            "configuration_match_required": True,
            "require_same_cluster": True,
            "bandit_prior_materialized": False,
        },
        "selected_donor": selected_donor,
        "candidate_count": len(candidates),
        "bandit_prior": None,

        # Campi extra per audit: json.Unmarshal Go li ignora.
        "ranking": candidates,
        "paper5": {
            "feature_names": PAPER5,
            "target_assignment": "euclidean_to_centroid",
            "donor_ranking": "manhattan",
            "scaler": "minmax",
            "kmeans": {
                "k": args.k,
                "n_init": args.n_init,
                "random_state": args.random_state,
            },
            "excluded_special_donors": sorted(set(excluded_special_donors)),
        },
        "sources": {
            "historical_profiles": {
                "path": str(historical_path.resolve()),
                "sha256": sha256_file(historical_path),
            },
            "target_profiles": {
                "path": str(target_path.resolve()),
                "sha256": sha256_file(target_path),
            },
        },
    }

    selection_path = output_dir / "selection.json"
    atomic_json(selection_path, selection)

    # Salviamo anche scaler + centroidi per poter ricostruire esattamente
    # l'assegnazione del target in fase di analisi della tesi.
    model = {
        "schema_version": 1,
        "feature_names": PAPER5,
        "scaler": "minmax",
        "data_min": [float(value) for value in scaler.data_min_.tolist()],
        "data_max": [float(value) for value in scaler.data_max_.tolist()],
        "scale": [float(value) for value in scaler.scale_.tolist()],
        "offset": [float(value) for value in scaler.min_.tolist()],
        "kmeans": {
            "k": args.k,
            "n_init": args.n_init,
            "random_state": args.random_state,
            "cluster_centers": [
                [float(value) for value in center]
                for center in kmeans.cluster_centers_.tolist()
            ],
        },
        "lofo_target": args.target,
        "historical_row_count": len(donors),
    }

    atomic_json(output_dir / "paper5-model.json", model)

    print(
        json.dumps(
            {
                "status": status,
                "cluster_label": target_cluster,
                "selected_donor": selected_donor["function_name"] if selected_donor else "",
                "selected_distance": selected_donor["distance"] if selected_donor else None,
                "candidate_count": len(candidates),
                "selection_json": str(selection_path),
                "query_json": str(output_dir / "transfer-query.json"),
                "model_json": str(output_dir / "paper5-model.json"),
                "historical_donor_rows": len(donors),
            },
            sort_keys=True,
        )
    )


def build_prior(args: argparse.Namespace) -> None:
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))

    if selection.get("status") != "selected":
        raise SystemExit(
            f"selection non trasferibile: "
            f"status={selection.get('status')} reason={selection.get('reason')}"
        )

    prior_config = {
        "min_real_observations_per_arm": args.minimum,
        "ucb1_reference_anchor": {
            "enabled": True,
            "reference_arm": args.reference_arm,
            "target_reference_mean_reward": args.target_reference_mean_reward,
        },
    }

    if args.mode == "coupled":
        prior_config["equivalent_observation_weight"] = args.weight

    elif args.mode == "decoupled":
        prior_config["reward_observation_weight"] = args.reward_weight
        prior_config["exploration_observation_weight"] = args.exploration_weight

    else:
        raise SystemExit("build-prior accetta solo coupled o decoupled")

    atomic_json(
        Path(args.output),
        {
            "target_function_name": args.target,
            "selection_artifact": selection,
            "prior_config": prior_config,
        },
    )


def verify_prior(args: argparse.Namespace) -> None:
    response = json.loads(Path(args.response).read_text(encoding="utf-8"))
    expected_policy = "UCB1Decoupled" if args.mode == "decoupled" else "UCB1"

    if response.get("target_function_name") != args.target:
        raise SystemExit("target inatteso nella transfer response")

    if response.get("selected_donor_function_name") != args.donor:
        raise SystemExit("donor inatteso nella transfer response")

    if response.get("transfer_attempted") is not True or response.get("transfer_applied") is not True:
        raise SystemExit(f"transfer non applicato: {response.get('runtime_reason')}")

    prior = response.get("prior") or {}

    if prior.get("has_prior") is not True:
        raise SystemExit("weak prior assente")

    if prior.get("policy") != expected_policy:
        raise SystemExit(f"policy prior inattesa: {prior.get('policy')}")

    config = prior.get("config") or {}

    if args.mode == "coupled":
        actual_weight = float(config.get("equivalent_observation_weight", -1))

        if not math.isclose(actual_weight, args.weight, abs_tol=1e-12):
            raise SystemExit("equivalent_observation_weight inatteso")

        expected_reward_weight = args.weight
        expected_exploration_weight = args.weight

    else:
        actual_reward_weight = float(config.get("reward_observation_weight", -1))
        actual_exploration_weight = float(
            config.get("exploration_observation_weight", -1)
        )

        if not math.isclose(actual_reward_weight, args.reward_weight, abs_tol=1e-12):
            raise SystemExit("reward_observation_weight inatteso")

        if not math.isclose(
                actual_exploration_weight,
                args.exploration_weight,
                abs_tol=1e-12,
        ):
            raise SystemExit("exploration_observation_weight inatteso")

        expected_reward_weight = args.reward_weight
        expected_exploration_weight = args.exploration_weight

    anchor = config.get("ucb1_reference_anchor") or {}

    if anchor.get("enabled") is not True or anchor.get("reference_arm") != args.reference_arm:
        raise SystemExit("reference anchor errato")

    actual_target_reference = float(anchor.get("target_reference_mean_reward"))

    if not math.isclose(
            actual_target_reference,
            args.target_reference_mean_reward,
            rel_tol=1e-12,
            abs_tol=1e-12,
    ):
        raise SystemExit("target reference mean reward inatteso")

    arms = prior.get("arms") or {}

    for arm in (args.reference_arm, args.other_arm):
        arm_prior = arms.get(arm) or {}

        if arm_prior.get("transferred") is not True:
            raise SystemExit(f"prior non trasferito su {arm}")

        ucb = arm_prior.get("ucb1") or {}

        reward_weight = float(ucb.get("observation_weight", -1))
        exploration_weight = float(
            ucb.get("exploration_observation_weight", -1)
        )

        if not math.isclose(reward_weight, expected_reward_weight, abs_tol=1e-12):
            raise SystemExit(f"reward weight inatteso su {arm}")

        if not math.isclose(
                exploration_weight,
                expected_exploration_weight,
                abs_tol=1e-12,
        ):
            raise SystemExit(f"exploration weight inatteso su {arm}")

    reference_ucb = (arms.get(args.reference_arm) or {}).get("ucb1") or {}
    reference_mean = float(reference_ucb.get("mean_reward"))

    if not math.isclose(
            reference_mean,
            args.target_reference_mean_reward,
            rel_tol=1e-12,
            abs_tol=1e-12,
    ):
        raise SystemExit("reference anchoring errato")

    print(
        f"transfer=PASS policy={expected_policy} "
        f"donor={args.donor} anchor={args.reference_arm}"
    )


def summarize(args: argparse.Namespace) -> None:
    token_re = re.compile(r"([A-Za-z_]+)=([^\s]+)")
    selects: list[dict] = []
    updates: list[dict] = []

    for line in Path(args.lb_log).read_text(
            encoding="utf-8",
            errors="replace",
    ).splitlines():

        if "[MAB]" not in line or f"function={args.target}" not in line:
            continue

        fields = dict(token_re.findall(line))

        if fields.get("event") == "select_arm":
            selects.append(fields)

        elif fields.get("event") == "update_reward":
            updates.append(fields)

    if len(selects) != args.requests or len(updates) != args.requests:
        raise SystemExit(
            f"event count inatteso: select={len(selects)} "
            f"update={len(updates)} expected={args.requests}"
        )

    with Path(args.request_csv).open(newline="", encoding="utf-8") as handle:
        request_rows = list(csv.DictReader(handle))

    if len(request_rows) != args.requests:
        raise SystemExit(
            f"request CSV rows={len(request_rows)} expected={args.requests}"
        )

    rows = []

    for index, (selected, update, request) in enumerate(
            zip(selects, updates, request_rows),
            start=1,
    ):
        selected_arm = selected.get("selected_arm", "")
        execution_arm = update.get("arm", "")

        duration_ms = float(update.get("duration_ms", "nan"))
        reward = float(update.get("reward", "nan"))

        if not math.isfinite(duration_ms) or duration_ms <= 0.0 or not math.isfinite(reward):
            raise SystemExit(f"feedback non valido alla richiesta {index}")

        header_arch = request.get("node_arch", "")

        normalized_header = {
            "x86": "amd64",
            "x86_64": "amd64",
            "aarch64": "arm64",
        }.get(header_arch, header_arch)

        if normalized_header and normalized_header != execution_arm:
            raise SystemExit(
                f"request {index}: header={header_arch} "
                f"({normalized_header}) update_arm={execution_arm}"
            )

        rows.append(
            {
                "request_index": index,
                "selected_arm": selected_arm,
                "execution_arm": execution_arm,
                "fallback": selected_arm != execution_arm,
                "duration_ms": duration_ms,
                "reward": reward,
                "warm_start": update.get("warm_start", ""),
            }
        )

    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "target_function": args.target,
        "mode": args.mode,
        "policy": args.policy,
        "request_count": args.requests,
        "all_warm": all(row["warm_start"] == "true" for row in rows),
        "fallback_count": sum(bool(row["fallback"]) for row in rows),
        "selected_arm_counts": {},
        "execution_arm_counts": {},
        "horizons": {},
    }

    for arm in sorted({row["selected_arm"] for row in rows}):
        summary["selected_arm_counts"][arm] = sum(
            row["selected_arm"] == arm for row in rows
        )

    for arm in sorted({row["execution_arm"] for row in rows}):
        summary["execution_arm_counts"][arm] = sum(
            row["execution_arm"] == arm for row in rows
        )

    for horizon in (5, 10, 20, 50):
        subset = rows[:horizon]
        durations = [row["duration_ms"] for row in subset]

        summary["horizons"][str(horizon)] = {
            "cumulative_duration_ms": sum(durations),
            "mean_duration_ms": statistics.fmean(durations),
            "median_duration_ms": statistics.median(durations),
            "amd64_executions": sum(
                row["execution_arm"] == "amd64" for row in subset
            ),
            "arm64_executions": sum(
                row["execution_arm"] == "arm64" for row in subset
            ),
            "fallbacks": sum(bool(row["fallback"]) for row in subset),
        }

    atomic_json(Path(args.output_json), summary)

    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Helper PAPER-5 per la campagna GCP finale di transfer learning."
    )

    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("filter-profile")
    profile.add_argument("--raw", required=True)
    profile.add_argument("--target", required=True)
    profile.add_argument("--machine-tag", required=True)
    profile.add_argument("--samples", type=int, required=True)
    profile.add_argument("--output", required=True)
    profile.set_defaults(func=filter_profile)

    select = commands.add_parser("select-donor")
    select.add_argument("--historical-profiles", required=True)
    select.add_argument("--target-profiles", required=True)
    select.add_argument("--target", required=True)
    select.add_argument("--reference-tag", required=True)
    select.add_argument("--configured-memory-mb", type=int, required=True)
    select.add_argument("--run-id", required=True)
    select.add_argument("--output-dir", required=True)
    select.add_argument("--k", type=int, default=5)
    select.add_argument("--n-init", type=int, default=50)
    select.add_argument("--random-state", type=int, default=42)
    select.add_argument("--max-distance", type=float, default=1e12)
    select.set_defaults(func=select_donor)

    prior = commands.add_parser("build-prior")
    prior.add_argument("--selection", required=True)
    prior.add_argument("--target", required=True)
    prior.add_argument("--mode", choices=("coupled", "decoupled"), required=True)
    prior.add_argument("--weight", type=float, default=0.25)
    prior.add_argument("--reward-weight", type=float, default=0.25)
    prior.add_argument("--exploration-weight", type=float, default=1.0)
    prior.add_argument("--minimum", type=int, default=10)
    prior.add_argument("--target-reference-mean-reward", type=float, required=True)
    prior.add_argument("--reference-arm", default="amd64")
    prior.add_argument("--output", required=True)
    prior.set_defaults(func=build_prior)

    verify = commands.add_parser("verify-prior")
    verify.add_argument("--response", required=True)
    verify.add_argument("--target", required=True)
    verify.add_argument("--donor", required=True)
    verify.add_argument("--mode", choices=("coupled", "decoupled"), required=True)
    verify.add_argument("--weight", type=float, default=0.25)
    verify.add_argument("--reward-weight", type=float, default=0.25)
    verify.add_argument("--exploration-weight", type=float, default=1.0)
    verify.add_argument("--target-reference-mean-reward", type=float, required=True)
    verify.add_argument("--reference-arm", default="amd64")
    verify.add_argument("--other-arm", default="arm64")
    verify.set_defaults(func=verify_prior)

    summary = commands.add_parser("summarize")
    summary.add_argument("--lb-log", required=True)
    summary.add_argument("--request-csv", required=True)
    summary.add_argument("--target", required=True)
    summary.add_argument("--requests", type=int, required=True)
    summary.add_argument("--mode", required=True)
    summary.add_argument("--policy", required=True)
    summary.add_argument("--output-csv", required=True)
    summary.add_argument("--output-json", required=True)
    summary.set_defaults(func=summarize)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()