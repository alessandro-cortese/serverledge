#!/usr/bin/env python3

import csv
import json
import math
from pathlib import Path

import numpy as np

from analysis.profiling.transfer_ucb1_offline import (
    ARMS,
    UCB1Policy,
    stable_seed,
)

from analysis.profiling.transfer_decoupled_ucb1 import (
    DecoupledUCB1Policy,
)


# ============================================================
# CONFIGURAZIONE
# ============================================================

ROOT = Path("data/profiling/final-20260913-analysis-01")

RAW_X86 = ROOT / "raw/x86/all_samples.jsonl"
RAW_ARM = ROOT / "raw/arm64/all_samples.jsonl"

DONOR_CSV = Path(
    "data/profiling/"
    "analysis-transfer-donor-metrics-20260918/"
    "donor-metric-per-target.csv"
)

OUT_DIR = Path(
    "data/profiling/"
    "analysis-transfer-zero-weight-ablation"
)

WEIGHTS = [0.0, 0.25, 0.50, 1.00]

C = 0.8
HORIZON = 10
REPLICATES = 500
SEED = 42
CONVERGENCE_RUN = 5


# ============================================================
# LETTURA DATI
# ============================================================

def normalize_arm(tag):
    if tag in {"amd64", "x86"}:
        return "x86"
    if tag == "arm64":
        return "arm64"
    return None


def load_samples(path):
    result = {}

    with path.open(encoding="utf-8") as f:
        for line in f:

            if not line.strip():
                continue

            sample = json.loads(line)

            if sample.get("warm_start") is not True:
                continue

            if sample.get("execution_succeeded") is not True:
                continue

            eligibility = sample.get("eligibility") or {}
            if eligibility.get("performance_analysis") is not True:
                continue

            profile = sample.get("profile") or {}
            if profile.get("valid") is not True:
                continue

            if profile.get("exclusive_container") is not True:
                continue

            function = sample.get("function_name")
            arm = normalize_arm(sample.get("machine_tag"))

            timing = sample.get("timing") or {}

            try:
                duration = float(timing.get("duration_ms", 0.0))
            except (TypeError, ValueError):
                continue

            if (
                not function
                or arm is None
                or not math.isfinite(duration)
                or duration <= 0
            ):
                continue

            result.setdefault(function, {})
            result[function].setdefault(arm, [])
            result[function][arm].append(duration)

    return result


x86_raw = load_samples(RAW_X86)
arm_raw = load_samples(RAW_ARM)

durations = {}

for function in sorted(set(x86_raw) | set(arm_raw)):

    x = x86_raw.get(function, {}).get("x86", [])
    a = arm_raw.get(function, {}).get("arm64", [])

    if not x or not a:
        continue

    durations[function] = {
        "x86": np.asarray(x, dtype=float),
        "arm64": np.asarray(a, dtype=float),
    }


def mean_reward(values):
    return float(
        np.mean(
            [-math.log(float(v)) for v in values]
        )
    )


performance = {}

for function, arms in durations.items():

    rewards = {
        arm: mean_reward(arms[arm])
        for arm in ARMS
    }

    best_arm = max(
        ARMS,
        key=lambda arm: (
            rewards[arm],
            -ARMS.index(arm),
        ),
    )

    performance[function] = {
        "durations": arms,
        "mean_reward": rewards,
        "best_reward_arm": best_arm,
    }


# ============================================================
# DONOR K-MEANS + MANHATTAN
# ============================================================

with DONOR_CSV.open(
    newline="",
    encoding="utf-8",
) as f:
    donor_rows = list(csv.DictReader(f))


donors = {}

for row in donor_rows:

    if row.get("algorithm", "").lower() != "kmeans":
        continue

    if row.get("ranker", "").lower() != "manhattan":
        continue

    status = row.get("selection_status", "")

    if status and status != "selected":
        continue

    target = row.get("target_function", "")
    donor = row.get("donor_function", "")

    if target and donor:
        donors[target] = donor


targets = sorted(
    set(performance)
    & set(donors)
)

print(f"Target validi: {len(targets)}")

if len(targets) != 53:
    raise RuntimeError(
        f"Attese 53 target, trovate {len(targets)}"
    )


for target in targets:

    for arm in ARMS:
        n = len(
            performance[target]["durations"][arm]
        )

        if n != 12:
            raise RuntimeError(
                f"{target}/{arm}: "
                f"attese 12 misure, trovate {n}"
            )

    if donors[target] not in performance:
        raise RuntimeError(
            f"Donor mancante: "
            f"{target} -> {donors[target]}"
        )


# ============================================================
# PRIOR REFERENCE-ANCHORED
# ============================================================

def build_prior(target, donor):

    target_x86 = float(
        performance[target]["mean_reward"]["x86"]
    )

    donor_x86 = float(
        performance[donor]["mean_reward"]["x86"]
    )

    donor_arm = float(
        performance[donor]["mean_reward"]["arm64"]
    )

    delta = donor_arm - donor_x86

    return {
        "x86": target_x86,
        "arm64": target_x86 + delta,
    }


# ============================================================
# POLICY ESTESA CHE AMMETTE ZERO
# ============================================================

class ZeroSafeDecoupledUCB1:

    def __init__(
        self,
        c,
        reward_weight,
        exploration_weight,
        prior,
    ):

        if reward_weight < 0:
            raise ValueError("w_R deve essere >= 0")

        if exploration_weight < 0:
            raise ValueError("w_E deve essere >= 0")

        self.c = float(c)
        self.wr = float(reward_weight)
        self.we = float(exploration_weight)

        self.counts = {
            arm: 0
            for arm in ARMS
        }

        self.reward_sums = {
            arm: 0.0
            for arm in ARMS
        }

        self.prior_reward_sums = {
            arm: self.wr * float(prior[arm])
            for arm in ARMS
        }

    def effective_mean(self, arm):

        denom = self.counts[arm] + self.wr

        if denom <= 0:
            return None

        return float(
            (
                self.reward_sums[arm]
                + self.prior_reward_sums[arm]
            )
            / denom
        )

    def exploration_count(self, arm):
        return self.counts[arm] + self.we

    def exploration_total(self):
        return sum(
            self.exploration_count(arm)
            for arm in ARMS
        )

    def exploration_bonus(self, arm):

        count = self.exploration_count(arm)

        # Semantica UCB1 standard:
        # braccio non esplorato -> priorità infinita.
        if count <= 0:
            return float("inf")

        total = self.exploration_total()

        if total <= 1:
            return 0.0

        return float(
            self.c
            * math.sqrt(
                math.log(total)
                / count
            )
        )

    def score(self, arm):

        mean = self.effective_mean(arm)

        # Se w_R=0 e non abbiamo feedback reale,
        # non esiste ancora una stima del reward.
        if mean is None:
            return float("inf")

        bonus = self.exploration_bonus(arm)

        if math.isinf(bonus):
            return float("inf")

        return mean + bonus

    def select_arm(self):

        scores = {
            arm: self.score(arm)
            for arm in ARMS
        }

        return max(
            ARMS,
            key=lambda arm: (
                scores[arm],
                -ARMS.index(arm),
            ),
        )

    def update(self, arm, duration):

        reward = -math.log(duration)

        self.counts[arm] += 1
        self.reward_sums[arm] += reward

        return reward


# ============================================================
# SANITY CHECK:
# CON PESI > 0 DEVE COINCIDERE CON POLICY ORIGINALE
# ============================================================

def sanity_check():

    prior = {
        "x86": -6.0,
        "arm64": -5.5,
    }

    updates = [
        ("x86", 800.0),
        ("arm64", 600.0),
        ("x86", 780.0),
        ("arm64", 590.0),
    ]

    for wr in [0.25, 0.50, 1.00]:

        for we in [0.25, 0.50, 1.00]:

            old = DecoupledUCB1Policy(
                c=C,
                reward_prior_weight=wr,
                exploration_prior_weight=we,
                prior_mean_reward=prior,
            )

            new = ZeroSafeDecoupledUCB1(
                c=C,
                reward_weight=wr,
                exploration_weight=we,
                prior=prior,
            )

            for arm in ARMS:

                if not math.isclose(
                    old.score(arm),
                    new.score(arm),
                    abs_tol=1e-12,
                    rel_tol=0.0,
                ):
                    raise AssertionError(
                        f"Score differente: "
                        f"wR={wr}, wE={we}, arm={arm}"
                    )

            if old.select_arm() != new.select_arm():
                raise AssertionError(
                    f"Select differente: "
                    f"wR={wr}, wE={we}"
                )

            for arm, duration in updates:

                old.update(arm, duration)
                new.update(arm, duration)

                for candidate in ARMS:

                    if not math.isclose(
                        old.score(candidate),
                        new.score(candidate),
                        abs_tol=1e-12,
                        rel_tol=0.0,
                    ):
                        raise AssertionError(
                            f"Post-update score differente: "
                            f"wR={wr}, wE={we}"
                        )

    print(
        "PASS — policy estesa identica alla "
        "DecoupledUCB1 originale per tutti i pesi > 0"
    )


sanity_check()


# ============================================================
# CONVERGENZA
# ============================================================

def first_true_run(values, run_length):

    streak = 0

    for idx, value in enumerate(
        values,
        start=1,
    ):

        if value:
            streak += 1

            if streak >= run_length:
                return idx

        else:
            streak = 0

    return None


def baseline_mean(policy, arm):

    count = policy.counts[arm]

    if count <= 0:
        return None

    return float(
        policy.reward_sums[arm]
        / count
    )


# ============================================================
# SIMULAZIONE BASELINE
# ============================================================

def simulate_baseline(
    target_stats,
    sequences,
):

    policy = UCB1Policy(
        c=C,
        prior_weight=0.0,
        prior_mean_reward=None,
    )

    positions = {
        arm: 0
        for arm in ARMS
    }

    chosen = []
    exploitation_correct = []

    cumulative_latency = 0.0
    reward_regret = 0.0

    best_arm = target_stats["best_reward_arm"]

    best_mean_reward = float(
        target_stats["mean_reward"][best_arm]
    )

    for _ in range(HORIZON):

        arm = policy.select_arm()

        duration = float(
            sequences[arm][positions[arm]]
        )

        positions[arm] += 1

        policy.update(arm, duration)

        chosen.append(arm)

        cumulative_latency += duration

        reward_regret += (
            best_mean_reward
            - float(
                target_stats["mean_reward"][arm]
            )
        )

        means = {
            candidate: baseline_mean(
                policy,
                candidate,
            )
            for candidate in ARMS
        }

        if all(
            means[a] is not None
            for a in ARMS
        ):

            exploitation_arm = max(
                ARMS,
                key=lambda candidate: (
                    means[candidate],
                    -ARMS.index(candidate),
                ),
            )

            exploitation_correct.append(
                exploitation_arm == best_arm
            )

        else:
            exploitation_correct.append(False)

    conv = first_true_run(
        exploitation_correct,
        CONVERGENCE_RUN,
    )

    wrong = sum(
        arm != best_arm
        for arm in chosen
    )

    return {
        "latency": cumulative_latency,
        "wrong": wrong,
        "convergence": conv,
        "first_optimal": (
            chosen[0] == best_arm
        ),
        "regret": reward_regret,
    }


# ============================================================
# SIMULAZIONE TL ZERO-SAFE
# ============================================================

def simulate_transfer(
    target_stats,
    prior,
    sequences,
    wr,
    we,
):

    policy = ZeroSafeDecoupledUCB1(
        c=C,
        reward_weight=wr,
        exploration_weight=we,
        prior=prior,
    )

    positions = {
        arm: 0
        for arm in ARMS
    }

    chosen = []
    exploitation_correct = []

    cumulative_latency = 0.0
    reward_regret = 0.0

    best_arm = target_stats["best_reward_arm"]

    best_mean_reward = float(
        target_stats["mean_reward"][best_arm]
    )

    for _ in range(HORIZON):

        arm = policy.select_arm()

        duration = float(
            sequences[arm][positions[arm]]
        )

        positions[arm] += 1

        policy.update(arm, duration)

        chosen.append(arm)

        cumulative_latency += duration

        reward_regret += (
            best_mean_reward
            - float(
                target_stats["mean_reward"][arm]
            )
        )

        means = {
            candidate: policy.effective_mean(
                candidate
            )
            for candidate in ARMS
        }

        if all(
            means[a] is not None
            for a in ARMS
        ):

            exploitation_arm = max(
                ARMS,
                key=lambda candidate: (
                    means[candidate],
                    -ARMS.index(candidate),
                ),
            )

            exploitation_correct.append(
                exploitation_arm == best_arm
            )

        else:
            exploitation_correct.append(False)

    conv = first_true_run(
        exploitation_correct,
        CONVERGENCE_RUN,
    )

    wrong = sum(
        arm != best_arm
        for arm in chosen
    )

    return {
        "latency": cumulative_latency,
        "wrong": wrong,
        "convergence": conv,
        "first_optimal": (
            chosen[0] == best_arm
        ),
        "regret": reward_regret,
    }


# ============================================================
# ESPERIMENTO
# ============================================================

per_target = []

configurations = [
    (wr, we)
    for wr in WEIGHTS
    for we in WEIGHTS
]


for idx, target in enumerate(
    targets,
    start=1,
):

    donor = donors[target]

    print(
        f"[{idx:02d}/53] "
        f"{target:<25} donor={donor}"
    )

    target_stats = performance[target]

    prior = build_prior(
        target,
        donor,
    )

    rng = np.random.default_rng(
        stable_seed(
            SEED,
            target,
        )
    )

    sequences = [
        {
            arm: rng.choice(
                target_stats["durations"][arm],
                size=HORIZON,
                replace=True,
            )
            for arm in ARMS
        }
        for _ in range(REPLICATES)
    ]

    baseline_runs = [
        simulate_baseline(
            target_stats,
            seq,
        )
        for seq in sequences
    ]

    for wr, we in configurations:

        # (0,0) DEVE essere esattamente baseline.
        if wr == 0.0 and we == 0.0:

            transfer_runs = baseline_runs

        else:

            transfer_runs = [
                simulate_transfer(
                    target_stats,
                    prior,
                    seq,
                    wr,
                    we,
                )
                for seq in sequences
            ]

        gains = []
        wrong_saved = []
        convergence = []
        first_optimal = []
        regrets = []

        for baseline, transfer in zip(
            baseline_runs,
            transfer_runs,
        ):

            gains.append(
                100.0
                * (
                    baseline["latency"]
                    - transfer["latency"]
                )
                / baseline["latency"]
            )

            wrong_saved.append(
                baseline["wrong"]
                - transfer["wrong"]
            )

            convergence.append(
                1.0
                if transfer["convergence"]
                is not None
                else 0.0
            )

            first_optimal.append(
                1.0
                if transfer["first_optimal"]
                else 0.0
            )

            regrets.append(
                transfer["regret"]
            )

        per_target.append({
            "target_function": target,
            "donor_function": donor,
            "reward_prior_weight": wr,
            "exploration_prior_weight": we,
            "mean_latency_gain_pct": float(
                np.mean(gains)
            ),
            "mean_wrong_choices_saved": float(
                np.mean(wrong_saved)
            ),
            "convergence_probability": float(
                np.mean(convergence)
            ),
            "first_arm_optimal_probability": float(
                np.mean(first_optimal)
            ),
            "mean_reward_pseudo_regret": float(
                np.mean(regrets)
            ),
        })


# ============================================================
# MACRO-AVERAGE SULLE 53 TARGET
# ============================================================

summary = []

for wr, we in configurations:

    rows = [
        row
        for row in per_target
        if (
            row["reward_prior_weight"] == wr
            and row["exploration_prior_weight"] == we
        )
    ]

    if len(rows) != 53:
        raise RuntimeError(
            f"wR={wr}, wE={we}: "
            f"attese 53 righe, trovate {len(rows)}"
        )

    summary.append({
        "w_R": wr,
        "w_E": we,
        "macro_mean_latency_gain_pct": float(
            np.mean([
                r["mean_latency_gain_pct"]
                for r in rows
            ])
        ),
        "macro_mean_wrong_choices_saved": float(
            np.mean([
                r["mean_wrong_choices_saved"]
                for r in rows
            ])
        ),
        "macro_convergence_probability": float(
            np.mean([
                r["convergence_probability"]
                for r in rows
            ])
        ),
        "macro_first_arm_optimal_probability": float(
            np.mean([
                r["first_arm_optimal_probability"]
                for r in rows
            ])
        ),
        "macro_mean_reward_pseudo_regret": float(
            np.mean([
                r["mean_reward_pseudo_regret"]
                for r in rows
            ])
        ),
    })


# ============================================================
# EXPORT
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def write_rows(path, rows):

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


write_rows(
    OUT_DIR
    / "zero-weight-ablation-per-target.csv",
    per_target,
)

write_rows(
    OUT_DIR
    / "zero-weight-ablation-summary.csv",
    summary,
)


# ============================================================
# OUTPUT CONSOLE
# ============================================================

print()
print("=" * 100)
print("ZERO-WEIGHT ABLATION @ H=10")
print("=" * 100)

print(
    f"{'wR':>5} "
    f"{'wE':>5} "
    f"{'latency gain %':>16} "
    f"{'wrong saved':>14} "
    f"{'conv %':>10} "
    f"{'first optimal %':>16} "
    f"{'pseudo-regret':>14}"
)

print("-" * 100)

for row in summary:

    print(
        f"{row['w_R']:5.2f} "
        f"{row['w_E']:5.2f} "
        f"{row['macro_mean_latency_gain_pct']:+16.6f} "
        f"{row['macro_mean_wrong_choices_saved']:+14.6f} "
        f"{100*row['macro_convergence_probability']:10.4f} "
        f"{100*row['macro_first_arm_optimal_probability']:16.4f} "
        f"{row['macro_mean_reward_pseudo_regret']:14.6f}"
    )


# ============================================================
# VERIFICA CON I 9 RISULTATI PRECEDENTI
# ============================================================

expected = {
    (0.25, 0.25): 0.1283,
    (0.25, 0.50): 1.2919,
    (0.25, 1.00): 2.0391,

    (0.50, 0.25): 0.0504,
    (0.50, 0.50): 1.2422,
    (0.50, 1.00): 1.8602,

    (1.00, 0.25): -0.2019,
    (1.00, 0.50): 0.5850,
    (1.00, 1.00): 1.7452,
}


print()
print("=" * 100)
print("CHECK CONTRO LA GRIGLIA PRECEDENTE")
print("=" * 100)

max_delta = 0.0

for row in summary:

    key = (
        row["w_R"],
        row["w_E"],
    )

    if key not in expected:
        continue

    new = row[
        "macro_mean_latency_gain_pct"
    ]

    old = expected[key]

    delta = new - old

    max_delta = max(
        max_delta,
        abs(delta),
    )

    print(
        f"wR={key[0]:.2f} "
        f"wE={key[1]:.2f} | "
        f"new={new:+.4f}% | "
        f"old={old:+.4f}% | "
        f"delta={delta:+.4f} pp"
    )


print()

if max_delta <= 0.02:

    print(
        "PASS — i 9 casi positivi riproducono "
        "la griglia precedente entro 0.02 punti percentuali"
    )

else:

    print(
        "WARNING — differenza rispetto alla griglia precedente: "
        f"max delta={max_delta:.4f} pp"
    )


print()
print(
    "Output:",
    OUT_DIR / "zero-weight-ablation-summary.csv"
)
