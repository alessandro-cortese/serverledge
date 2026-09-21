#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HORIZONS = (5, 10, 20, 50)
MODES = ("no-transfer", "coupled", "decoupled")


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"File non trovato: {path}")

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"CSV non trovato: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_manifest(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}

    result = {}

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()

        if not line or "=" not in line:
            continue

        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()

    return result


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        )


def gain_percent(reference_ms: float, candidate_ms: float) -> float:
    if reference_ms <= 0:
        raise SystemExit(f"Durata reference non valida: {reference_ms}")

    return (reference_ms - candidate_ms) / reference_ms * 100.0


def validate_summary(mode: str, summary: dict, target: str, expected_policy: str) -> None:
    if summary.get("target_function") != target:
        raise SystemExit(
            f"{mode}: target inatteso {summary.get('target_function')!r}"
        )

    if summary.get("mode") != mode:
        raise SystemExit(
            f"{mode}: modalità inattesa {summary.get('mode')!r}"
        )

    if summary.get("policy") != expected_policy:
        raise SystemExit(
            f"{mode}: policy inattesa {summary.get('policy')!r}"
        )

    if int(summary.get("request_count", 0)) != 50:
        raise SystemExit(f"{mode}: request_count deve essere 50")

    if summary.get("all_warm") is not True:
        raise SystemExit(f"{mode}: non tutte le richieste risultano warm")

    if int(summary.get("fallback_count", -1)) != 0:
        raise SystemExit(f"{mode}: fallback_count deve essere 0")

    horizons = summary.get("horizons") or {}

    for horizon in HORIZONS:
        if str(horizon) not in horizons:
            raise SystemExit(f"{mode}: manca H{horizon}")


def validate_transfer_manifest(
        mode: str,
        manifest: dict[str, str],
        replica_id: str,
        selection_sha: str,
        donor: str,
) -> None:
    if not manifest:
        raise SystemExit(f"{mode}: manifest.txt mancante o vuoto")

    if manifest.get("mode") != mode:
        raise SystemExit(
            f"{mode}: manifest mode={manifest.get('mode')!r}"
        )

    if manifest.get("bootstrap_reused") != "true":
        raise SystemExit(f"{mode}: bootstrap_reused non è true")

    if manifest.get("bootstrap_replica_id") != replica_id:
        raise SystemExit(f"{mode}: bootstrap replica differente")

    if manifest.get("bootstrap_selection_sha256") != selection_sha:
        raise SystemExit(f"{mode}: SHA selection differente")

    if manifest.get("block_selected_donor") != donor:
        raise SystemExit(f"{mode}: block donor differente")

    if manifest.get("selected_donor") != donor:
        raise SystemExit(f"{mode}: donor applicato differente")


def find_control_response(run_dir: Path) -> Path | None:
    standard = run_dir / "transfer" / "control-response.json"

    if standard.is_file():
        return standard

    candidates = sorted(run_dir.rglob("*transfer-response.json"))

    if not candidates:
        return None

    if len(candidates) > 1:
        raise SystemExit(
            f"{run_dir}: trovate più transfer-response: {candidates}"
        )

    return candidates[0]


def extract_prior(response_path: Path | None) -> dict | None:
    if response_path is None:
        return None

    response = load_json(response_path)
    prior = response.get("prior") or {}
    arms = prior.get("arms") or {}

    arm_summary = {}

    for arm, arm_data in arms.items():
        arm_data = arm_data or {}
        ucb = arm_data.get("ucb1") or {}

        arm_summary[arm] = {
            "transferred": arm_data.get("transferred"),
            "mean_reward": ucb.get("mean_reward"),
            "observation_weight": ucb.get("observation_weight"),
            "exploration_observation_weight": ucb.get(
                "exploration_observation_weight"
            ),
        }

    return {
        "path": str(response_path),
        "target_function_name": response.get("target_function_name"),
        "selected_donor_function_name": response.get(
            "selected_donor_function_name"
        ),
        "transfer_applied": response.get("transfer_applied"),
        "policy": prior.get("policy"),
        "config": prior.get("config") or {},
        "arms": arm_summary,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit(f"Nessuna riga da scrivere in {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> None:
    bootstrap_dir = Path(args.bootstrap).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    run_dirs = {
        "no-transfer": Path(args.baseline).expanduser().resolve(),
        "coupled": Path(args.coupled).expanduser().resolve(),
        "decoupled": Path(args.decoupled).expanduser().resolve(),
    }

    expected_policies = {
        "no-transfer": "UCB1",
        "coupled": "UCB1",
        "decoupled": "UCB1Decoupled",
    }

    bootstrap = load_json(bootstrap_dir / "bootstrap.json")
    selection = load_json(bootstrap_dir / "selection" / "selection.json")

    replica_id = bootstrap["replica_id"]
    target = bootstrap["target_function"]
    donor = bootstrap["donor_selection"]["selected_donor"]
    selection_sha = bootstrap["donor_selection"]["selection_artifact_sha256"]
    anchor = float(
        bootstrap["target_profile"]["target_reference_mean_reward"]
    )

    if selection["query"]["function_name"] != target:
        raise SystemExit("Il target della selection non coincide con il bootstrap")

    if selection["selected_donor"]["function_name"] != donor:
        raise SystemExit("Il donor della selection non coincide con il bootstrap")

    summaries = {}
    requests = {}
    manifests = {}
    priors = {}

    for mode, run_dir in run_dirs.items():
        summary = load_json(run_dir / "measure" / "summary.json")

        validate_summary(
            mode=mode,
            summary=summary,
            target=target,
            expected_policy=expected_policies[mode],
        )

        request_rows = load_csv(run_dir / "measure" / "mab-requests.csv")

        if len(request_rows) != 50:
            raise SystemExit(
                f"{mode}: mab-requests contiene {len(request_rows)} righe, attese 50"
            )

        summaries[mode] = summary
        requests[mode] = request_rows
        manifests[mode] = load_manifest(run_dir / "manifest.txt")

        if mode == "no-transfer":
            continue

        validate_transfer_manifest(
            mode=mode,
            manifest=manifests[mode],
            replica_id=replica_id,
            selection_sha=selection_sha,
            donor=donor,
        )

        prior = extract_prior(find_control_response(run_dir))

        if prior is None:
            raise SystemExit(f"{mode}: control-response non trovata")

        if prior["transfer_applied"] is not True:
            raise SystemExit(f"{mode}: transfer_applied != true")

        if prior["selected_donor_function_name"] != donor:
            raise SystemExit(f"{mode}: donor nella response differente")

        priors[mode] = prior

    horizon_rows = []

    for horizon in HORIZONS:
        key = str(horizon)

        baseline = summaries["no-transfer"]["horizons"][key]
        coupled = summaries["coupled"]["horizons"][key]
        decoupled = summaries["decoupled"]["horizons"][key]

        baseline_ms = float(baseline["cumulative_duration_ms"])
        coupled_ms = float(coupled["cumulative_duration_ms"])
        decoupled_ms = float(decoupled["cumulative_duration_ms"])

        horizon_rows.append(
            {
                "replica_id": replica_id,
                "target_function": target,
                "selected_donor": donor,
                "horizon": horizon,

                "baseline_cumulative_ms": baseline_ms,
                "coupled_cumulative_ms": coupled_ms,
                "decoupled_cumulative_ms": decoupled_ms,

                "coupled_vs_baseline_gain_pct": gain_percent(
                    baseline_ms, coupled_ms
                ),
                "decoupled_vs_baseline_gain_pct": gain_percent(
                    baseline_ms, decoupled_ms
                ),
                "decoupled_vs_coupled_gain_pct": gain_percent(
                    coupled_ms, decoupled_ms
                ),

                "baseline_amd64": int(baseline["amd64_executions"]),
                "baseline_arm64": int(baseline["arm64_executions"]),
                "coupled_amd64": int(coupled["amd64_executions"]),
                "coupled_arm64": int(coupled["arm64_executions"]),
                "decoupled_amd64": int(decoupled["amd64_executions"]),
                "decoupled_arm64": int(decoupled["arm64_executions"]),
            }
        )

    per_request_rows = []

    cumulative = {
        "no-transfer": 0.0,
        "coupled": 0.0,
        "decoupled": 0.0,
    }

    for index in range(50):
        row = {"request_index": index + 1}

        for mode in MODES:
            source = requests[mode][index]
            duration_ms = float(source["duration_ms"])

            cumulative[mode] += duration_ms
            prefix = mode.replace("-", "_")

            row[f"{prefix}_selected_arm"] = source["selected_arm"]
            row[f"{prefix}_execution_arm"] = source["execution_arm"]
            row[f"{prefix}_duration_ms"] = duration_ms
            row[f"{prefix}_reward"] = float(source["reward"])
            row[f"{prefix}_cumulative_ms"] = cumulative[mode]

        row["coupled_vs_baseline_cumulative_gain_pct"] = gain_percent(
            cumulative["no-transfer"],
            cumulative["coupled"],
        )

        row["decoupled_vs_baseline_cumulative_gain_pct"] = gain_percent(
            cumulative["no-transfer"],
            cumulative["decoupled"],
        )

        row["decoupled_vs_coupled_cumulative_gain_pct"] = gain_percent(
            cumulative["coupled"],
            cumulative["decoupled"],
        )

        per_request_rows.append(row)

    horizon_csv = output_dir / "horizon-comparison.csv"
    request_csv = output_dir / "per-request-comparison.csv"

    write_csv(horizon_csv, horizon_rows)
    write_csv(request_csv, per_request_rows)

    analysis = {
        "schema_version": 1,
        "replica_id": replica_id,
        "target_function": target,
        "selected_donor": donor,
        "target_reference_mean_reward": anchor,
        "selection_artifact_sha256": selection_sha,

        "runs": {
            mode: {
                "path": str(run_dirs[mode]),
                "policy": summaries[mode]["policy"],
                "all_warm": summaries[mode]["all_warm"],
                "fallback_count": summaries[mode]["fallback_count"],
                "execution_arm_counts": summaries[mode]["execution_arm_counts"],
            }
            for mode in MODES
        },

        "priors": priors,

        "horizons": {
            str(row["horizon"]): {
                key: value
                for key, value in row.items()
                if key not in {
                    "replica_id",
                    "target_function",
                    "selected_donor",
                    "horizon",
                }
            }
            for row in horizon_rows
        },

        "artifacts": {
            "horizon_comparison_csv": str(horizon_csv),
            "per_request_comparison_csv": str(request_csv),
        },
    }

    write_json(output_dir / "replica-analysis.json", analysis)

    print(f"replica={replica_id}")
    print(f"target={target}")
    print(f"donor={donor}")
    print()

    print(
        f"{'H':>4} "
        f"{'baseline':>12} "
        f"{'coupled':>12} "
        f"{'decoupled':>12} "
        f"{'C vs B':>10} "
        f"{'D vs B':>10} "
        f"{'D vs C':>10}"
    )

    for row in horizon_rows:
        print(
            f"{row['horizon']:>4} "
            f"{row['baseline_cumulative_ms']:>12.3f} "
            f"{row['coupled_cumulative_ms']:>12.3f} "
            f"{row['decoupled_cumulative_ms']:>12.3f} "
            f"{row['coupled_vs_baseline_gain_pct']:>+9.3f}% "
            f"{row['decoupled_vs_baseline_gain_pct']:>+9.3f}% "
            f"{row['decoupled_vs_coupled_gain_pct']:>+9.3f}%"
        )

    print()
    print(f"PASS: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analizza una replica GCP no-transfer/coupled/decoupled."
    )

    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--coupled", required=True)
    parser.add_argument("--decoupled", required=True)
    parser.add_argument("--output-dir", required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    analyze(args)


if __name__ == "__main__":
    main()