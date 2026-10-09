#!/usr/bin/env python3
"""Generate a deterministic Python golden replay for the Serverledge Go test.

Run from repo root:
  python3 -m analysis.profiling.generate_ucb1_parity_fixture \
    --output internal/mab/testdata/ucb1_python_parity.json

Uses the ACTUAL frozen Python build_prior and NumericTLPolicy implementations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from analysis.profiling.transfer_prior_formula_comparison import build_prior
from analysis.profiling.transfer_parameter_tuning_cv import NumericTLPolicy

# Independent samples for x86 anchor, donor means and the online feedback.
DONOR_DURATIONS = {
    "x86": [1150, 1190, 1220, 1200, 1180, 1240, 1170, 1210, 1160, 1230, 1250, 1205],
    "arm64": [870, 855, 900, 845, 890, 860, 880, 865, 905, 850, 875, 895],
}
TARGET_ANCHOR = [1050, 1030, 1080, 1065, 1025, 1075, 1010, 1100, 1045, 1090]
TARGET_ONLINE = {
    "x86": [1035, 1100, 1040, 1125, 1070, 1160, 1050, 1080, 1200, 1060,
            1115, 1045, 1090, 1130, 1075, 1020, 1180, 1055, 1105, 1140],
    "arm64": [1255, 1205, 1290, 1190, 1335, 1220, 1275, 1350, 1180, 1265,
              1230, 1310, 1210, 1370, 1245, 1400, 1260, 1295, 1195, 1320],
}
CASES = [
    ("difference", .05, 64., .2),
    ("ratio", .01, 64., .4),
    ("difference", .25, 1., .8),
]
NAMES = ("x86", "arm64")
GO_ARM = {"x86": "amd64", "arm64": "arm64"}


def mean_log_reward(data: list[float]) -> float:
    return sum(-math.log(float(x)) for x in data) / len(data)


def make_fixture() -> dict:
    anchor = mean_log_reward(TARGET_ANCHOR)
    donor = {arm: mean_log_reward(DONOR_DURATIONS[arm]) for arm in NAMES}
    fixture = {"schema_version": 1, "source": "Python build_prior + NumericTLPolicy",
               "cases": []}
    for mode, w_r, w_e, c in CASES:
        prior, _ = build_prior(
            mode=mode, anchor_mean_reward_x86=anchor,
            donor_stats={"mean_reward": donor},
        )
        policy = NumericTLPolicy(
            c=c, w_r=w_r, w_e=w_e,
            prior_mean=[prior["x86"], prior["arm64"]],
        )
        pos = {arm: 0 for arm in NAMES}
        steps = []
        for t in range(len(TARGET_ONLINE["x86"])):
            scores_before = {GO_ARM[arm]: policy.score(idx)
                             for idx, arm in enumerate(NAMES)}
            idx = policy.select_idx()
            arm = NAMES[idx]
            duration = float(TARGET_ONLINE[arm][pos[arm]])
            pos[arm] += 1
            reward = policy.update(idx, duration)
            scores_after = {GO_ARM[name]: policy.score(j)
                            for j, name in enumerate(NAMES)}
            steps.append({
                "t": t + 1, "scores_before": scores_before,
                "selected_arm": GO_ARM[arm], "duration_ms": duration,
                "reward": reward, "scores_after": scores_after,
                "counts_after": {GO_ARM[name]: int(policy.counts[j])
                                 for j, name in enumerate(NAMES)},
            })
        fixture["cases"].append({
            "mode": mode, "w_r": w_r, "w_e": w_e, "c": c,
            "anchor_mean_reward": anchor,
            "donor_mean_reward": {GO_ARM[arm]: donor[arm] for arm in NAMES},
            "prior_mean_reward": {GO_ARM[arm]: prior[arm] for arm in NAMES},
            "donor_n_per_arm": len(DONOR_DURATIONS["x86"]),
            "steps": steps,
        })
    return fixture


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    fixture = make_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"PASS — generated {len(fixture['cases'])} Python parity cases: {args.output}")


if __name__ == "__main__":
    main()
