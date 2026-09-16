import math
import unittest

from analysis.profiling.transfer_decoupled_ucb1 import (
    DecoupledUCB1Policy,
    build_reference_anchored_prior,
    first_completed_true_run,
)
from analysis.profiling.transfer_ucb1_offline import (
    ARMS,
    UCB1Policy,
)


class TransferDecoupledUCB1Test(unittest.TestCase):

    def setUp(self):
        self.prior = {
            "x86": -4.0,
            "arm64": -3.5,
        }

    def test_reference_anchored_prior_preserves_arch_effect(self):
        target = {
            "mean_reward": {
                "x86": -6.0,
                "arm64": -5.8,
            }
        }
        donor = {
            "mean_reward": {
                "x86": -4.0,
                "arm64": -3.5,
            }
        }

        prior = build_reference_anchored_prior(
            target,
            donor,
        )

        self.assertAlmostEqual(
            prior["x86"],
            -6.0,
        )
        self.assertAlmostEqual(
            prior["arm64"],
            -5.5,
        )
        self.assertAlmostEqual(
            prior["arm64"] - prior["x86"],
            donor["mean_reward"]["arm64"]
            - donor["mean_reward"]["x86"],
        )

    def test_diagonal_scores_match_current_policy_initially(self):
        for weight in (0.25, 0.5, 1.0):
            current = UCB1Policy(
                c=0.8,
                prior_weight=weight,
                prior_mean_reward=self.prior,
            )
            decoupled = DecoupledUCB1Policy(
                c=0.8,
                reward_prior_weight=weight,
                exploration_prior_weight=weight,
                prior_mean_reward=self.prior,
            )

            for arm in ARMS:
                self.assertAlmostEqual(
                    current.score(arm),
                    decoupled.score(arm),
                    places=12,
                )

            self.assertEqual(
                current.select_arm(),
                decoupled.select_arm(),
            )

    def test_diagonal_matches_current_policy_after_updates(self):
        current = UCB1Policy(
            c=0.8,
            prior_weight=0.5,
            prior_mean_reward=self.prior,
        )
        decoupled = DecoupledUCB1Policy(
            c=0.8,
            reward_prior_weight=0.5,
            exploration_prior_weight=0.5,
            prior_mean_reward=self.prior,
        )

        durations = [
            ("arm64", 30.0),
            ("x86", 40.0),
            ("arm64", 25.0),
            ("x86", 35.0),
        ]

        for arm, duration in durations:
            self.assertEqual(
                current.select_arm(),
                decoupled.select_arm(),
            )

            current.update(
                arm,
                duration,
            )
            decoupled.update(
                arm,
                duration,
            )

            for candidate in ARMS:
                self.assertAlmostEqual(
                    current.score(candidate),
                    decoupled.score(candidate),
                    places=12,
                )

    def test_reward_weight_changes_mean_not_exploration_count(self):
        low = DecoupledUCB1Policy(
            c=0.8,
            reward_prior_weight=0.25,
            exploration_prior_weight=0.5,
            prior_mean_reward=self.prior,
        )
        high = DecoupledUCB1Policy(
            c=0.8,
            reward_prior_weight=1.0,
            exploration_prior_weight=0.5,
            prior_mean_reward=self.prior,
        )

        self.assertEqual(
            low.exploration_count("x86"),
            high.exploration_count("x86"),
        )
        self.assertNotEqual(
            low.exploitation_count("x86"),
            high.exploitation_count("x86"),
        )

    def test_exploration_weight_changes_bonus_not_effective_mean(self):
        low = DecoupledUCB1Policy(
            c=0.8,
            reward_prior_weight=0.5,
            exploration_prior_weight=0.25,
            prior_mean_reward=self.prior,
        )
        high = DecoupledUCB1Policy(
            c=0.8,
            reward_prior_weight=0.5,
            exploration_prior_weight=1.0,
            prior_mean_reward=self.prior,
        )

        # After one real observation total exploration count exceeds one
        # for both policies, making the bonus comparison meaningful.
        low.update("arm64", 30.0)
        high.update("arm64", 30.0)

        self.assertAlmostEqual(
            low.effective_mean("arm64"),
            high.effective_mean("arm64"),
            places=12,
        )
        self.assertNotAlmostEqual(
            low.exploration_bonus("arm64"),
            high.exploration_bonus("arm64"),
            places=12,
        )

    def test_first_completed_true_run(self):
        self.assertEqual(
            first_completed_true_run(
                [False, True, True, True],
                3,
            ),
            4,
        )
        self.assertIsNone(
            first_completed_true_run(
                [True, False, True, True],
                3,
            )
        )


if __name__ == "__main__":
    unittest.main()
