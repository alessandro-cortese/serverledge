"""
Utilities for Serverledge materialized weak priors.

This module derives a UCB1Decoupled materialized prior from an already
materialized coupled UCB1 prior.

The transformation preserves the reward-side knowledge transferred from
the donor and changes only the exploration pseudo-count contribution.

Coupled:
    reward weight      = 0.25
    exploration weight = 0.25

Decoupled:
    reward weight      = 0.25
    exploration weight = 1.00

The module works only on local JSON artifacts and does not contact GCP.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


DEFAULT_REWARD_WEIGHT = 0.25
DEFAULT_EXPLORATION_WEIGHT = 1.0


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict[str, Any], *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        if compact:
            json.dump(data, file, sort_keys=True, separators=(",", ":"))
        else:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def floats_equal(left: float, right: float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def derive_decoupled_prior(
        coupled_prior: dict[str, Any],
        *,
        reward_weight: float = DEFAULT_REWARD_WEIGHT,
        exploration_weight: float = DEFAULT_EXPLORATION_WEIGHT,
) -> dict[str, Any]:
    """
    Convert a materialized coupled UCB1 prior into UCB1Decoupled.

    The reward-side state remains unchanged:
        - observation_weight
        - mean_reward
        - reward_sum
        - attenuation_scale

    Only the exploration pseudo-count weight changes.
    """

    if reward_weight <= 0:
        raise ValueError("reward_weight must be > 0")

    if exploration_weight <= 0:
        raise ValueError("exploration_weight must be > 0")

    if coupled_prior.get("schema_version") != 1:
        raise ValueError("unsupported prior schema_version")

    if coupled_prior.get("policy") != "UCB1":
        raise ValueError("source materialized prior must use UCB1")

    if coupled_prior.get("has_prior") is not True:
        raise ValueError("source materialized prior has no prior")

    source_config = coupled_prior.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("source prior config is missing")

    coupled_weight = source_config.get("equivalent_observation_weight")
    if coupled_weight is None:
        raise ValueError("source coupled prior has no equivalent_observation_weight")

    if not floats_equal(coupled_weight, reward_weight):
        raise ValueError(
            f"unexpected coupled weight: {coupled_weight}, expected {reward_weight}"
        )

    source_arms = coupled_prior.get("arms")
    if not isinstance(source_arms, dict) or not source_arms:
        raise ValueError("source materialized prior has no arms")

    decoupled_prior = copy.deepcopy(coupled_prior)
    decoupled_prior["policy"] = "UCB1Decoupled"

    decoupled_config = copy.deepcopy(source_config)
    decoupled_config.pop("equivalent_observation_weight", None)
    decoupled_config["reward_observation_weight"] = float(reward_weight)
    decoupled_config["exploration_observation_weight"] = float(
        exploration_weight
    )

    decoupled_prior["config"] = decoupled_config

    transferred_arms = 0

    for arm_name, arm_prior in decoupled_prior["arms"].items():
        if arm_prior.get("transferred") is not True:
            continue

        transferred_arms += 1

        ucb1 = arm_prior.get("ucb1")
        if not isinstance(ucb1, dict):
            raise ValueError(f"transferred arm {arm_name!r} has no ucb1 state")

        observation_weight = ucb1.get("observation_weight")
        if observation_weight is None:
            raise ValueError(f"arm {arm_name!r} has no observation_weight")

        if not floats_equal(observation_weight, reward_weight):
            raise ValueError(
                f"arm {arm_name!r} has unexpected observation_weight "
                f"{observation_weight}"
            )

        mean_reward = float(ucb1["mean_reward"])
        reward_sum = float(ucb1["reward_sum"])

        expected_reward_sum = reward_weight * mean_reward

        if not floats_equal(reward_sum, expected_reward_sum):
            raise ValueError(
                f"arm {arm_name!r} has inconsistent reward_sum: "
                f"{reward_sum} != {expected_reward_sum}"
            )

        # Reward-side prior remains unchanged.
        ucb1["observation_weight"] = float(reward_weight)
        ucb1["mean_reward"] = mean_reward
        ucb1["reward_sum"] = reward_sum

        # Only the exploration contribution changes.
        ucb1["exploration_observation_weight"] = float(exploration_weight)

        # Runtime audit fields.
        arm_prior["applied_equivalent_observation_weight"] = float(
            reward_weight
        )
        arm_prior["applied_exploration_observation_weight"] = float(
            exploration_weight
        )

    if transferred_arms == 0:
        raise ValueError("source prior contains no transferred arms")

    return decoupled_prior


def verify_reward_prior_preserved(
        coupled_prior: dict[str, Any],
        decoupled_prior: dict[str, Any],
        *,
        reward_weight: float = DEFAULT_REWARD_WEIGHT,
        exploration_weight: float = DEFAULT_EXPLORATION_WEIGHT,
) -> None:
    """
    Verify that coupled and decoupled priors contain exactly the same
    reward-side knowledge.
    """

    if coupled_prior.get("policy") != "UCB1":
        raise ValueError("coupled prior policy is not UCB1")

    if decoupled_prior.get("policy") != "UCB1Decoupled":
        raise ValueError("decoupled prior policy is not UCB1Decoupled")

    metadata_fields = (
        "donor_function_name",
        "source_real_observation_count",
        "source_excluded_synthetic_observation_count",
        "arm_count",
        "transferred_arm_count",
        "skipped_arm_count",
    )

    for field in metadata_fields:
        if coupled_prior.get(field) != decoupled_prior.get(field):
            raise ValueError(f"metadata changed during conversion: {field}")

    coupled_anchor = coupled_prior["config"]["ucb1_reference_anchor"]
    decoupled_anchor = decoupled_prior["config"]["ucb1_reference_anchor"]

    if coupled_anchor != decoupled_anchor:
        raise ValueError("reference anchor changed during conversion")

    for arm_name, coupled_arm in coupled_prior["arms"].items():
        decoupled_arm = decoupled_prior["arms"][arm_name]

        if coupled_arm.get("transferred") != decoupled_arm.get("transferred"):
            raise ValueError(f"transferred flag changed for arm {arm_name}")

        if not coupled_arm.get("transferred"):
            continue

        coupled_ucb1 = coupled_arm["ucb1"]
        decoupled_ucb1 = decoupled_arm["ucb1"]

        reward_fields = (
            "observation_weight",
            "reward_sum",
            "mean_reward",
        )

        for field in reward_fields:
            if not floats_equal(coupled_ucb1[field], decoupled_ucb1[field]):
                raise ValueError(
                    f"reward-side field {field} changed for arm {arm_name}"
                )

        if not floats_equal(
                coupled_arm["attenuation_scale"],
                decoupled_arm["attenuation_scale"],
        ):
            raise ValueError(f"attenuation_scale changed for arm {arm_name}")

        if not floats_equal(
                decoupled_ucb1["observation_weight"],
                reward_weight,
        ):
            raise ValueError(f"invalid reward weight for arm {arm_name}")

        if not floats_equal(
                decoupled_ucb1["exploration_observation_weight"],
                exploration_weight,
        ):
            raise ValueError(f"invalid exploration weight for arm {arm_name}")


def derive_bundle(
        source_bundle: Path,
        output_bundle: Path,
        *,
        reward_weight: float = DEFAULT_REWARD_WEIGHT,
        exploration_weight: float = DEFAULT_EXPLORATION_WEIGHT,
) -> dict[str, Any]:
    """
    Create a complete decoupled materialized bundle from a coupled bundle.
    """

    source_bundle = source_bundle.resolve()
    output_bundle = output_bundle.resolve()

    source_prior_path = source_bundle / "frozen-prior.json"
    source_manifest_path = source_bundle / "manifest.json"
    source_sha_path = source_bundle / "prior.sha256"
    source_bootstrap_path = source_bundle / "bootstrap" / "bootstrap.json"

    required_files = (
        source_prior_path,
        source_manifest_path,
        source_sha_path,
        source_bootstrap_path,
    )

    for path in required_files:
        if not path.is_file():
            raise ValueError(f"required source artifact missing: {path}")

    if output_bundle.exists() and any(output_bundle.iterdir()):
        raise ValueError(f"output bundle is not empty: {output_bundle}")

    coupled_prior = load_json(source_prior_path)
    source_manifest = load_json(source_manifest_path)

    recorded_sha = source_sha_path.read_text(encoding="utf-8").split()[0]
    actual_sha = sha256_file(source_prior_path)

    if recorded_sha != actual_sha:
        raise ValueError(
            f"source prior SHA mismatch: {recorded_sha} != {actual_sha}"
        )

    if source_manifest.get("prior_sha256") != actual_sha:
        raise ValueError("source manifest prior SHA does not match frozen prior")

    decoupled_prior = derive_decoupled_prior(
        coupled_prior,
        reward_weight=reward_weight,
        exploration_weight=exploration_weight,
    )

    verify_reward_prior_preserved(
        coupled_prior,
        decoupled_prior,
        reward_weight=reward_weight,
        exploration_weight=exploration_weight,
    )

    output_bundle.mkdir(parents=True, exist_ok=True)

    decoupled_prior_path = output_bundle / "frozen-prior.json"

    # Compact + sorted representation gives us a deterministic SHA.
    write_json(
        decoupled_prior_path,
        decoupled_prior,
        compact=True,
    )

    decoupled_sha = sha256_file(decoupled_prior_path)

    (output_bundle / "prior.sha256").write_text(
        f"{decoupled_sha}  frozen-prior.json\n",
        encoding="utf-8",
    )

    shutil.copy2(
        source_prior_path,
        output_bundle / "source-coupled-frozen-prior.json",
        )

    (output_bundle / "source-coupled-prior.sha256").write_text(
        f"{actual_sha}  source-coupled-frozen-prior.json\n",
        encoding="utf-8",
    )

    bootstrap_dir = output_bundle / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        source_bootstrap_path,
        bootstrap_dir / "bootstrap.json",
        )

    donor_readiness_path = source_bundle / "donor-readiness.json"

    if donor_readiness_path.is_file():
        # Keep the canonical materialized-bundle contract.
        shutil.copy2(
            donor_readiness_path,
            output_bundle / "donor-readiness.json",
            )

        # Keep an explicit provenance copy as well.
        shutil.copy2(
            donor_readiness_path,
            output_bundle / "source-coupled-donor-readiness.json",
            )

    materialized_request = {
        "target_function_name": source_manifest["target_function"],
        "prior": decoupled_prior,
    }

    write_json(
        output_bundle / "materialized-request.json",
        materialized_request,
        )

    manifest = {
        "schema_version": 1,
        "status": "materialized",
        "target_function": source_manifest["target_function"],
        "donor_function": source_manifest["donor_function"],
        "policy": "UCB1Decoupled",
        "reward_observation_weight": float(reward_weight),
        "exploration_observation_weight": float(exploration_weight),
        "source_c": source_manifest.get("source_c"),
        "source_coupled_prior_sha256": actual_sha,
        "prior_sha256": decoupled_sha,
        "bootstrap_sha256": sha256_file(
            bootstrap_dir / "bootstrap.json"
        ),
        "donor_observations": source_manifest.get("donor_observations", {}),
        "target_real_feedback_before_export": source_manifest.get(
            "target_real_feedback_before_export"
        ),
    }

    write_json(
        output_bundle / "manifest.json",
        manifest,
        )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a UCB1Decoupled materialized prior "
            "from a coupled Serverledge prior."
        )
    )

    parser.add_argument(
        "--source-bundle",
        type=Path,
        required=True,
        help="Existing coupled materialized bundle",
    )

    parser.add_argument(
        "--output-bundle",
        type=Path,
        required=True,
        help="Destination decoupled materialized bundle",
    )

    parser.add_argument(
        "--reward-weight",
        type=float,
        default=DEFAULT_REWARD_WEIGHT,
    )

    parser.add_argument(
        "--exploration-weight",
        type=float,
        default=DEFAULT_EXPLORATION_WEIGHT,
    )

    args = parser.parse_args()

    try:
        manifest = derive_bundle(
            args.source_bundle,
            args.output_bundle,
            reward_weight=args.reward_weight,
            exploration_weight=args.exploration_weight,
        )
    except ValueError as error:
        raise SystemExit(f"FATAL: {error}") from error

    print("PASS — decoupled materialized bundle derived")
    print(f"target={manifest['target_function']}")
    print(f"donor={manifest['donor_function']}")
    print(
        "source_coupled_prior_sha256="
        f"{manifest['source_coupled_prior_sha256']}"
    )
    print(f"decoupled_prior_sha256={manifest['prior_sha256']}")
    print(
        "reward_observation_weight="
        f"{manifest['reward_observation_weight']}"
    )
    print(
        "exploration_observation_weight="
        f"{manifest['exploration_observation_weight']}"
    )
    print("reward prior preserved: PASS")


if __name__ == "__main__":
    main()