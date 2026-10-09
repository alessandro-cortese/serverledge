#!/usr/bin/env python3
"""Create frozen tuned UCB1Decoupled prior bundles for Serverledge.

This intentionally takes a previously verified coupled/difference bundle as
provenance. Difference can reuse its anchored donor gap. Ratio requires BOTH
raw donor mean rewards, which the coupled prior cannot recover uniquely.

No cloud calls, no mutation of the source bundle.
Run: python3 -m analysis.profiling.transfer_materialized_tuned --help
"""
from __future__ import annotations

import argparse
import copy
import math
import shutil
from pathlib import Path
from typing import Any

from analysis.profiling.transfer_materialized_prior import (
    json_number, load_json, sha256_file, write_json,
)


def close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-10)


def valid_number(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def tuned_prior(
    coupled: dict[str, Any], *, mode: str, reward_weight: float,
    exploration_weight: float, donor_reward_ref: float | None = None,
    donor_reward_other: float | None = None,
) -> dict[str, Any]:
    """Return a fully recalculated materialized prior, not just config metadata."""
    if mode not in ("difference", "ratio"):
        raise ValueError("mode must be difference or ratio")
    reward_weight = valid_number(reward_weight, "reward_weight")
    exploration_weight = valid_number(exploration_weight, "exploration_weight")
    if not 0 < reward_weight <= 1:
        raise ValueError("reward_weight must be in (0, 1] (Go bound)")
    if exploration_weight <= 0:
        raise ValueError("exploration_weight must be > 0")
    if coupled.get("schema_version") != 1 or coupled.get("policy") != "UCB1":
        raise ValueError("source must be a schema-v1 coupled UCB1 prior")
    if coupled.get("has_prior") is not True:
        raise ValueError("source has no prior")

    src_config = coupled["config"]
    src_anchor = src_config.get("ucb1_reference_anchor")
    if not isinstance(src_anchor, dict) or src_anchor.get("enabled") is not True:
        raise ValueError("source requires an enabled target reference anchor")
    if src_anchor.get("mode", "difference") not in ("", "difference"):
        raise ValueError("source must have a difference anchor")

    ref = src_anchor.get("reference_arm")
    arms = coupled.get("arms")
    if not isinstance(arms, dict) or len(arms) != 2 or ref not in arms:
        raise ValueError("source must have exactly two arms including the reference arm")
    other = next(arm for arm in arms if arm != ref)
    if not (arms[ref].get("transferred") and arms[other].get("transferred")):
        raise ValueError("both arms must have transferable donor evidence")

    anchor_mean = valid_number(src_anchor["target_reference_mean_reward"], "target anchor")
    old_ref = valid_number(arms[ref]["ucb1"]["mean_reward"], "old reference mean")
    old_other = valid_number(arms[other]["ucb1"]["mean_reward"], "old other-arm mean")
    if not close(old_ref, anchor_mean):
        raise ValueError("source reference mean does not equal its recorded target anchor")
    source_gap = old_other - old_ref

    both = donor_reward_ref is not None and donor_reward_other is not None
    neither = donor_reward_ref is None and donor_reward_other is None
    if not (both or neither):
        raise ValueError("provide both donor reward means, or neither")
    if mode == "ratio" and not both:
        raise ValueError("ratio needs BOTH donor mean rewards; the source difference prior only stores their gap")
    if both:
        donor_reward_ref = valid_number(donor_reward_ref, "donor reference reward")
        donor_reward_other = valid_number(donor_reward_other, "donor other-arm reward")
        if not close(donor_reward_other - donor_reward_ref, source_gap):
            raise ValueError(
                "supplied donor reward gap disagrees with source coupled prior: "
                f"supplied={donor_reward_other-donor_reward_ref:.12g} source={source_gap:.12g}"
            )

    if mode == "difference":
        prior_other = anchor_mean + source_gap
    else:
        assert donor_reward_ref is not None and donor_reward_other is not None
        if abs(donor_reward_ref) <= 1e-12:
            raise ValueError("ratio undefined: donor reference mean reward ~0")
        prior_other = anchor_mean * (donor_reward_other / donor_reward_ref)
    valid_number(prior_other, "new prior mean")

    result = copy.deepcopy(coupled)
    result["policy"] = "UCB1Decoupled"
    config = result["config"]
    config.pop("equivalent_observation_weight", None)
    config["reward_observation_weight"] = float(reward_weight)
    config["exploration_observation_weight"] = json_number(exploration_weight)
    config["ucb1_reference_anchor"]["mode"] = mode

    for arm, expected_mean in ((ref, anchor_mean), (other, prior_other)):
        entry = result["arms"][arm]
        count = int(entry["source_real_observation_count"])
        if count < max(1, int(config["min_real_observations_per_arm"])):
            raise ValueError(f"insufficient real donor observations on {arm}")
        applied_reward = min(reward_weight, float(count))
        entry["applied_equivalent_observation_weight"] = float(applied_reward)
        entry["applied_exploration_observation_weight"] = json_number(exploration_weight)
        entry["attenuation_scale"] = float(applied_reward / count)
        entry["ucb1"]["observation_weight"] = float(applied_reward)
        entry["ucb1"]["exploration_observation_weight"] = json_number(exploration_weight)
        entry["ucb1"]["mean_reward"] = float(expected_mean)
        entry["ucb1"]["reward_sum"] = float(expected_mean * applied_reward)

    return result


def tuned_bundle(
    source: Path, output: Path, *, mode: str, reward_weight: float,
    exploration_weight: float, donor_reward_ref: float | None = None,
    donor_reward_other: float | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("source and destination must be distinct non-nested folders")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output folder is not empty: {output}")

    prior_path = source / "frozen-prior.json"
    manifest_path = source / "manifest.json"
    hash_path = source / "prior.sha256"
    bootstrap_dir = source / "bootstrap"
    for p in (prior_path, manifest_path, hash_path, bootstrap_dir / "bootstrap.json"):
        if not p.is_file():
            raise ValueError(f"required source artifact missing: {p}")
    original_sha = sha256_file(prior_path)
    if hash_path.read_text(encoding="utf-8").split()[0] != original_sha:
        raise ValueError("source prior.sha256 does not match actual content")
    source_manifest = load_json(manifest_path)
    if source_manifest.get("prior_sha256") != original_sha:
        raise ValueError("source manifest SHA does not match actual content")
    source_prior = load_json(prior_path)
    if source_manifest.get("donor_function") != source_prior.get("donor_function_name"):
        raise ValueError("source manifest donor does not match source prior")

    prior = tuned_prior(
        source_prior, mode=mode, reward_weight=reward_weight,
        exploration_weight=exploration_weight,
        donor_reward_ref=donor_reward_ref, donor_reward_other=donor_reward_other,
    )
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "frozen-prior.json"
    write_json(frozen_path, prior, compact=True)
    prior_sha = sha256_file(frozen_path)
    (output / "prior.sha256").write_text(
        f"{prior_sha}  frozen-prior.json\n", encoding="utf-8"
    )
    shutil.copy2(prior_path, output / "source-coupled-frozen-prior.json")
    (output / "source-coupled-prior.sha256").write_text(
        f"{original_sha}  source-coupled-frozen-prior.json\n", encoding="utf-8"
    )
    shutil.copytree(bootstrap_dir, output / "bootstrap")
    donor_ready = source / "donor-readiness.json"
    if donor_ready.is_file():
        shutil.copy2(donor_ready, output / "donor-readiness.json")
        shutil.copy2(donor_ready, output / "source-coupled-donor-readiness.json")
    write_json(output / "materialized-request.json", {
        "target_function_name": source_manifest["target_function"], "prior": prior,
    })
    manifest = {
        "schema_version": 1,
        "status": "materialized",
        "policy": "UCB1Decoupled",
        "mode": mode,
        "target_function": source_manifest["target_function"],
        "donor_function": source_manifest["donor_function"],
        "reward_observation_weight": reward_weight,
        "exploration_observation_weight": json_number(exploration_weight),
        "source_c": source_manifest.get("source_c"),
        "target_c": 0.2 if mode == "difference" else 0.4,
        "source_coupled_prior_sha256": original_sha,
        "prior_sha256": prior_sha,
        "bootstrap_sha256": sha256_file(output / "bootstrap" / "bootstrap.json"),
        "donor_observations": source_manifest.get("donor_observations", {}),
        "donor_reward_ref": donor_reward_ref,
        "donor_reward_other": donor_reward_other,
        "donor_reward_provenance": "caller_supplied" if donor_reward_ref is not None else "source_difference_only",
        "target_real_feedback_before_export": source_manifest.get("target_real_feedback_before_export"),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-bundle", type=Path, required=True)
    p.add_argument("--output-bundle", type=Path, required=True)
    p.add_argument("--mode", choices=["difference", "ratio"], required=True)
    p.add_argument("--reward-weight", type=float, required=True)
    p.add_argument("--exploration-weight", type=float, required=True)
    p.add_argument("--donor-reward-ref", type=float)
    p.add_argument("--donor-reward-other", type=float)
    args = p.parse_args()
    try:
        result = tuned_bundle(
            args.source_bundle, args.output_bundle, mode=args.mode,
            reward_weight=args.reward_weight, exploration_weight=args.exploration_weight,
            donor_reward_ref=args.donor_reward_ref, donor_reward_other=args.donor_reward_other,
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.error(str(exc))
    print("PASS — tuned materialized prior generated")
    for k in ("mode", "target_function", "donor_function", "target_c", "reward_observation_weight",
              "exploration_observation_weight", "prior_sha256"):
        print(f"{k}={result[k]}")


if __name__ == "__main__":
    main()
