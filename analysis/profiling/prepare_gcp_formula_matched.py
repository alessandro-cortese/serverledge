from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

from analysis.profiling.transfer_materialized_tuned import (
    tuned_prior,
)
from analysis.profiling.transfer_materialized_prior import (
    json_number,
    load_json,
    sha256_file,
    write_json,
)


TARGETS = (
    "billibench-knn",
    "dynamic-html-py",
    "functionbench-gzip",
    "graph-bfs",
)


FAMILIES: dict[str, dict[str, float]] = {
    "B": {
        "w_r": 0.05,
        "w_e": 64.0,
        "c": 0.2,
    },
    "C": {
        "w_r": 0.01,
        "w_e": 64.0,
        "c": 0.4,
    },
}


def close(a: float, b: float) -> bool:
    return math.isclose(
        float(a),
        float(b),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open(
            "r",
            encoding="utf-8",
    ) as handle:
        return json.load(handle)


def save_json(
        path: Path,
        document: dict[str, Any],
) -> None:
    with path.open(
            "w",
            encoding="utf-8",
    ) as handle:
        json.dump(
            document,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
                lambda: handle.read(
                    1024 * 1024
                ),
                b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require_file(
        path: Path,
        label: str,
) -> None:
    if not path.is_file():
        raise ValueError(
            f"missing {label}: {path}"
        )


def verify_original_bootstrap(
        bootstrap: dict[str, Any],
        target: str,
) -> None:
    if (
            bootstrap.get("target_function")
            != target
    ):
        raise ValueError(
            f"{target}: bootstrap target mismatch"
        )

    conditions = (
        bootstrap["experimental_block"]
        ["conditions"]
    )

    expected = {
        "difference": {
            "mode": "difference",
            "c": 0.2,
            "reward_observation_weight":
                0.05,
            "exploration_observation_weight":
                64.0,
        },
        "ratio": {
            "mode": "ratio",
            "c": 0.4,
            "reward_observation_weight":
                0.01,
            "exploration_observation_weight":
                64.0,
        },
    }

    for condition, wanted in (
            expected.items()
    ):
        if condition not in conditions:
            raise ValueError(
                f"{target}: missing original "
                f"condition {condition}"
            )

        actual = conditions[condition]

        if (
                actual.get("mode")
                != wanted["mode"]
        ):
            raise ValueError(
                f"{target}: unexpected "
                f"{condition}.mode"
            )

        for key in (
                "c",
                "reward_observation_weight",
                "exploration_observation_weight",
        ):
            if not close(
                    actual[key],
                    wanted[key],
            ):
                raise ValueError(
                    f"{target}: unexpected "
                    f"{condition}.{key}: "
                    f"{actual[key]} != "
                    f"{wanted[key]}"
                )


def create_bundle(
        source_bundle: Path,
        output_bundle: Path,
        *,
        mode: str,
        target_c: float,
        reward_weight: float,
        exploration_weight: float,
        donor_reward_ref: float | None = None,
        donor_reward_other: float | None = None,
) -> dict[str, Any]:

    source_prior_path = (
            source_bundle
            / "frozen-prior.json"
    )

    source_manifest_path = (
            source_bundle
            / "manifest.json"
    )

    source_sha_path = (
            source_bundle
            / "prior.sha256"
    )

    source_bootstrap_dir = (
            source_bundle
            / "bootstrap"
    )

    for path, label in (
            (
                    source_prior_path,
                    "source frozen prior",
            ),
            (
                    source_manifest_path,
                    "source manifest",
            ),
            (
                    source_sha_path,
                    "source prior SHA",
            ),
            (
                    source_bootstrap_dir
                    / "bootstrap.json",
                    "source bootstrap",
            ),
    ):
        require_file(
            path,
            label,
        )

    if output_bundle.exists():
        raise ValueError(
            "output bundle already exists: "
            f"{output_bundle}"
        )

    source_prior_sha = (
        sha256_file(
            source_prior_path
        )
    )

    recorded_sha = (
        source_sha_path
        .read_text(
            encoding="utf-8"
        )
        .split()[0]
    )

    if (
            recorded_sha
            != source_prior_sha
    ):
        raise ValueError(
            "source prior.sha256 mismatch"
        )

    source_manifest = load_json(
        source_manifest_path
    )

    if (
            source_manifest.get(
                "prior_sha256"
            )
            != source_prior_sha
    ):
        raise ValueError(
            "source manifest SHA mismatch"
        )

    source_prior = load_json(
        source_prior_path
    )

    if (
            source_manifest.get(
                "donor_function"
            )
            != source_prior.get(
        "donor_function_name"
    )
    ):
        raise ValueError(
            "source donor mismatch"
        )

    prior = tuned_prior(
        source_prior,
        mode=mode,
        reward_weight=reward_weight,
        exploration_weight=(
            exploration_weight
        ),
        donor_reward_ref=(
            donor_reward_ref
        ),
        donor_reward_other=(
            donor_reward_other
        ),
    )

    output_bundle.mkdir(
        parents=True,
        exist_ok=False,
    )

    frozen_path = (
            output_bundle
            / "frozen-prior.json"
    )

    write_json(
        frozen_path,
        prior,
        compact=True,
    )

    prior_sha = sha256_file(
        frozen_path
    )

    (
            output_bundle
            / "prior.sha256"
    ).write_text(
        f"{prior_sha}  "
        "frozen-prior.json\n",
        encoding="utf-8",
    )

    shutil.copy2(
        source_prior_path,
        output_bundle
        / "source-coupled-frozen-prior.json",
        )

    (
            output_bundle
            / "source-coupled-prior.sha256"
    ).write_text(
        f"{source_prior_sha}  "
        "source-coupled-frozen-prior.json\n",
        encoding="utf-8",
    )

    shutil.copytree(
        source_bootstrap_dir,
        output_bundle
        / "bootstrap",
        )

    donor_readiness = (
            source_bundle
            / "donor-readiness.json"
    )

    if donor_readiness.is_file():
        shutil.copy2(
            donor_readiness,
            output_bundle
            / "donor-readiness.json",
            )

        shutil.copy2(
            donor_readiness,
            output_bundle
            / (
                "source-coupled-"
                "donor-readiness.json"
            ),
            )

    write_json(
        output_bundle
        / "materialized-request.json",
        {
            "target_function_name":
                source_manifest[
                    "target_function"
                ],
            "prior": prior,
        },
        )

    manifest = {
        "schema_version": 1,
        "status": "materialized",
        "policy": "UCB1Decoupled",
        "mode": mode,
        "target_function":
            source_manifest[
                "target_function"
            ],
        "donor_function":
            source_manifest[
                "donor_function"
            ],
        "reward_observation_weight":
            float(reward_weight),
        "exploration_observation_weight":
            json_number(
                exploration_weight
            ),
        "source_c":
            source_manifest.get(
                "source_c"
            ),
        "target_c":
            float(target_c),
        "source_coupled_prior_sha256":
            source_prior_sha,
        "prior_sha256":
            prior_sha,
        "bootstrap_sha256":
            sha256_file(
                output_bundle
                / "bootstrap"
                / "bootstrap.json"
            ),
        "donor_observations":
            source_manifest.get(
                "donor_observations",
                {},
            ),
        "donor_reward_ref":
            donor_reward_ref,
        "donor_reward_other":
            donor_reward_other,
        "donor_reward_provenance":
            (
                "caller_supplied"
                if donor_reward_ref
                   is not None
                else
                "source_difference_only"
            ),
        "target_real_feedback_before_export":
            source_manifest.get(
                "target_real_feedback_before_export"
            ),
    }

    write_json(
        output_bundle
        / "manifest.json",
        manifest,
        )

    return manifest


def verify_generated_bundle(
        bundle: Path,
        *,
        target: str,
        mode: str,
        w_r: float,
        w_e: float,
        c: float,
) -> None:

    manifest_path = (
            bundle
            / "manifest.json"
    )

    prior_path = (
            bundle
            / "frozen-prior.json"
    )

    sha_path = (
            bundle
            / "prior.sha256"
    )

    request_path = (
            bundle
            / "materialized-request.json"
    )

    for path, label in (
            (
                    manifest_path,
                    "generated manifest",
            ),
            (
                    prior_path,
                    "generated prior",
            ),
            (
                    sha_path,
                    "generated SHA",
            ),
            (
                    request_path,
                    "generated request",
            ),
    ):
        require_file(
            path,
            label,
        )

    manifest = read_json(
        manifest_path
    )

    prior = read_json(
        prior_path
    )

    if (
            manifest["target_function"]
            != target
    ):
        raise ValueError(
            f"{target}: target mismatch"
        )

    if manifest["mode"] != mode:
        raise ValueError(
            f"{target}: formula mismatch"
        )

    if not close(
            manifest["target_c"],
            c,
    ):
        raise ValueError(
            f"{target}: c mismatch"
        )

    if not close(
            manifest[
                "reward_observation_weight"
            ],
            w_r,
    ):
        raise ValueError(
            f"{target}: w_R mismatch"
        )

    if not close(
            manifest[
                "exploration_observation_weight"
            ],
            w_e,
    ):
        raise ValueError(
            f"{target}: w_E mismatch"
        )

    actual_sha = file_sha256(
        prior_path
    )

    recorded_sha = (
        sha_path
        .read_text(
            encoding="utf-8"
        )
        .split()[0]
    )

    if (
            actual_sha != recorded_sha
            or actual_sha
            != manifest["prior_sha256"]
    ):
        raise ValueError(
            f"{target}: prior SHA mismatch"
        )

    config = prior["config"]

    if (
            config[
                "ucb1_reference_anchor"
            ]["mode"]
            != mode
    ):
        raise ValueError(
            f"{target}: prior mode mismatch"
        )

    if not close(
            config[
                "reward_observation_weight"
            ],
            w_r,
    ):
        raise ValueError(
            f"{target}: prior w_R mismatch"
        )

    if not close(
            config[
                "exploration_observation_weight"
            ],
            w_e,
    ):
        raise ValueError(
            f"{target}: prior w_E mismatch"
        )


def prepare_replica(
        source_root: Path,
        family_root: Path,
        *,
        family: str,
        target: str,
) -> None:

    params = FAMILIES[family]

    replica_name = (
        f"{target}-r01"
    )

    source_replica = (
            source_root
            / "replicas"
            / replica_name
    )

    output_replica = (
            family_root
            / "replicas"
            / replica_name
    )

    if not source_replica.is_dir():
        raise ValueError(
            "missing source replica: "
            f"{source_replica}"
        )

    shutil.copytree(
        source_replica,
        output_replica,
    )

    bootstrap_path = (
            output_replica
            / "bootstrap.json"
    )

    bootstrap = read_json(
        bootstrap_path
    )

    verify_original_bootstrap(
        bootstrap,
        target,
    )

    conditions = (
        bootstrap[
            "experimental_block"
        ]["conditions"]
    )

    original_conditions = (
        copy.deepcopy(
            conditions
        )
    )

    # Same parameters for both formulae.
    conditions["difference"] = {
        "prior": True,
        "mode": "difference",
        "c": params["c"],
        "reward_observation_weight":
            params["w_r"],
        "exploration_observation_weight":
            params["w_e"],
    }

    conditions["ratio"] = {
        "prior": True,
        "mode": "ratio",
        "c": params["c"],
        "reward_observation_weight":
            params["w_r"],
        "exploration_observation_weight":
            params["w_e"],
    }

    # Both No-TL aliases use the same c
    # inside each family.
    conditions[
        "notl-difference"
    ] = {
        "prior": False,
        "c": params["c"],
    }

    conditions[
        "notl-ratio"
    ] = {
        "prior": False,
        "c": params["c"],
    }

    bootstrap[
        "formula_matched_experiment"
    ] = {
        "schema_version": 1,
        "family": family,
        "w_r": params["w_r"],
        "w_e": params["w_e"],
        "c": params["c"],
        "original_final_conditions":
            original_conditions,
    }

    save_json(
        bootstrap_path,
        bootstrap,
    )

    evidence_dir = (
            output_replica
            / "evidence"
    )

    original_difference = (
            evidence_dir
            / "difference"
    )

    original_ratio = (
            evidence_dir
            / "ratio"
    )

    preserved_difference = (
            evidence_dir
            / "difference-final-original"
    )

    preserved_ratio = (
            evidence_dir
            / "ratio-final-original"
    )

    ratio_manifest = read_json(
        original_ratio
        / "manifest.json"
    )

    donor_reward_ref = (
        ratio_manifest.get(
            "donor_reward_ref"
        )
    )

    donor_reward_other = (
        ratio_manifest.get(
            "donor_reward_other"
        )
    )

    if (
            donor_reward_ref is None
            or donor_reward_other is None
    ):
        raise ValueError(
            f"{target}: ratio manifest "
            "does not contain raw donor "
            "reward means"
        )

    original_difference.rename(
        preserved_difference
    )

    original_ratio.rename(
        preserved_ratio
    )

    source_bundle = (
            evidence_dir
            / "source"
    )

    create_bundle(
        source_bundle,
        evidence_dir
        / "difference",
        mode="difference",
        target_c=params["c"],
        reward_weight=params["w_r"],
        exploration_weight=params["w_e"],
        )

    create_bundle(
        source_bundle,
        evidence_dir
        / "ratio",
        mode="ratio",
        target_c=params["c"],
        reward_weight=params["w_r"],
        exploration_weight=params["w_e"],
        donor_reward_ref=float(
            donor_reward_ref
        ),
        donor_reward_other=float(
            donor_reward_other
        ),
        )

    verify_generated_bundle(
        evidence_dir
        / "difference",
        target=target,
        mode="difference",
        w_r=params["w_r"],
        w_e=params["w_e"],
        c=params["c"],
        )

    verify_generated_bundle(
        evidence_dir
        / "ratio",
        target=target,
        mode="ratio",
        w_r=params["w_r"],
        w_e=params["w_e"],
        c=params["c"],
        )

    # Very useful consistency check:
    # B/Difference must reproduce the
    # original frozen Difference prior.
    if family == "B":
        generated = file_sha256(
            evidence_dir
            / "difference"
            / "frozen-prior.json"
        )

        original = file_sha256(
            preserved_difference
            / "frozen-prior.json"
        )

        if generated != original:
            raise ValueError(
                f"{target}: B/Difference "
                "does not reproduce the "
                "original frozen prior"
            )

    # C/Ratio must reproduce the
    # original frozen Ratio prior.
    if family == "C":
        generated = file_sha256(
            evidence_dir
            / "ratio"
            / "frozen-prior.json"
        )

        original = file_sha256(
            preserved_ratio
            / "frozen-prior.json"
        )

        if generated != original:
            raise ValueError(
                f"{target}: C/Ratio "
                "does not reproduce the "
                "original frozen prior"
            )

    print(
        f"PASS family={family} "
        f"target={target} "
        f"w_R={params['w_r']} "
        f"w_E={params['w_e']} "
        f"c={params['c']}"
    )


def prepare_family(
        source_root: Path,
        output_root: Path,
        family: str,
) -> None:

    family_root = (
            output_root
            / family
    )

    (
            family_root
            / "replicas"
    ).mkdir(
        parents=True,
        exist_ok=False,
    )

    for target in TARGETS:
        prepare_replica(
            source_root,
            family_root,
            family=family,
            target=target,
        )

    save_json(
        family_root
        / "family-manifest.json",
        {
            "schema_version": 1,
            "experiment":
                "gcp-ucb1-formula-matched",
            "family": family,
            "parameters":
                FAMILIES[family],
            "targets":
                list(TARGETS),
            "conditions": [
                "notl-difference",
                "difference",
                "notl-ratio",
                "ratio",
            ],
        },
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare isolated GCP "
            "Difference-vs-Ratio experiments "
            "at identical parameter values."
        )
    )

    root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=(
                root
                / "data"
                / "profiling"
                / "gcp-ucb1-final"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
                root
                / "data"
                / "profiling"
                / "gcp-ucb1-formula-matched"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild only the dedicated "
            "formula-matched output root."
        ),
    )

    args = parser.parse_args()

    source_root = (
        args.source_root.resolve()
    )

    output_root = (
        args.output_root.resolve()
    )

    if (
            source_root
            == output_root
    ):
        raise SystemExit(
            "FATAL: source and output "
            "roots must differ"
        )

    if not source_root.is_dir():
        raise SystemExit(
            "FATAL: source root not found: "
            f"{source_root}"
        )

    if output_root.exists():
        if not args.force:
            raise SystemExit(
                "FATAL: output root already "
                f"exists: {output_root}\n"
                "Use --force only to rebuild "
                "this dedicated experiment."
            )

        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    for family in (
            "B",
            "C",
    ):
        prepare_family(
            source_root,
            output_root,
            family,
        )

    save_json(
        output_root
        / "campaign-manifest.json",
        {
            "schema_version": 1,
            "experiment":
                "gcp-ucb1-formula-matched",
            "source_root":
                str(source_root),
            "families":
                FAMILIES,
            "targets":
                list(TARGETS),
            "original_artifacts_modified":
                False,
        },
        )

    print()
    print(
        "GCP FORMULA-MATCHED "
        "PREPARATION: PASS"
    )

    print(
        f"source={source_root}"
    )

    print(
        f"output={output_root}"
    )

    print(
        "original_artifacts_modified=false"
    )


if __name__ == "__main__":
    main()