import math
import unittest

from analysis.profiling.transfer_decoupled_ucb1 import (
    DecoupledUCB1Policy,
    build_reference_anchored_prior,
    first_completed_true_run,
)
from analysis.profiling.transfer_materialized_prior import (
    derive_decoupled_prior,
    verify_reward_prior_preserved,
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

            current.update(arm, duration)
            decoupled.update(arm, duration)

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

    def test_c_zero_coupled_and_decoupled_match(self):
        """
        With the same reward prior and c=0, the different exploration
        pseudo-count weights must not affect the score or arm selection.
        """

        coupled = UCB1Policy(
            c=0.0,
            prior_weight=0.25,
            prior_mean_reward=self.prior,
        )

        decoupled = DecoupledUCB1Policy(
            c=0.0,
            reward_prior_weight=0.25,
            exploration_prior_weight=1.0,
            prior_mean_reward=self.prior,
        )

        updates = [
            ("x86", 50.0),
            ("arm64", 30.0),
            ("arm64", 28.0),
            ("x86", 45.0),
            ("arm64", 27.0),
        ]

        for arm, duration in updates:
            for candidate in ARMS:
                self.assertAlmostEqual(
                    coupled.score(candidate),
                    decoupled.score(candidate),
                    places=12,
                )

            self.assertEqual(
                coupled.select_arm(),
                decoupled.select_arm(),
            )

            coupled.update(arm, duration)
            decoupled.update(arm, duration)

        for candidate in ARMS:
            self.assertAlmostEqual(
                coupled.score(candidate),
                decoupled.score(candidate),
                places=12,
            )

    def test_c_positive_coupled_and_decoupled_can_differ(self):
        """
        With c>0, changing only the exploration pseudo-count must change
        the exploration bonus while preserving the exploitation mean.
        """

        coupled = DecoupledUCB1Policy(
            c=0.8,
            reward_prior_weight=0.25,
            exploration_prior_weight=0.25,
            prior_mean_reward=self.prior,
        )

        decoupled = DecoupledUCB1Policy(
            c=0.8,
            reward_prior_weight=0.25,
            exploration_prior_weight=1.0,
            prior_mean_reward=self.prior,
        )

        coupled.update("x86", 50.0)
        decoupled.update("x86", 50.0)

        for arm in ARMS:
            self.assertAlmostEqual(
                coupled.effective_mean(arm),
                decoupled.effective_mean(arm),
                places=12,
            )

        bonuses_differ = any(
            not math.isclose(
                coupled.exploration_bonus(arm),
                decoupled.exploration_bonus(arm),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for arm in ARMS
        )

        self.assertTrue(bonuses_differ)

    def test_materialized_decoupled_prior_preserves_reward_side(self):
        coupled = {
            "schema_version": 1,
            "donor_function_name": "compression",
            "policy": "UCB1",
            "config": {
                "equivalent_observation_weight": 0.25,
                "min_real_observations_per_arm": 10,
                "ucb1_reference_anchor": {
                    "enabled": True,
                    "reference_arm": "amd64",
                    "target_reference_mean_reward": -7.36,
                },
            },
            "has_prior": True,
            "source_real_observation_count": 30,
            "source_excluded_synthetic_observation_count": 0,
            "arm_count": 2,
            "transferred_arm_count": 2,
            "skipped_arm_count": 0,
            "arms": {
                "amd64": {
                    "source_real_observation_count": 16,
                    "source_excluded_synthetic_observation_count": 0,
                    "transferred": True,
                    "applied_equivalent_observation_weight": 0.25,
                    "applied_exploration_observation_weight": 0.25,
                    "attenuation_scale": 0.015625,
                    "ucb1": {
                        "observation_weight": 0.25,
                        "exploration_observation_weight": 0.25,
                        "reward_sum": -1.84,
                        "mean_reward": -7.36,
                    },
                },
                "arm64": {
                    "source_real_observation_count": 14,
                    "source_excluded_synthetic_observation_count": 0,
                    "transferred": True,
                    "applied_equivalent_observation_weight": 0.25,
                    "applied_exploration_observation_weight": 0.25,
                    "attenuation_scale": 0.017857142857142856,
                    "ucb1": {
                        "observation_weight": 0.25,
                        "exploration_observation_weight": 0.25,
                        "reward_sum": -1.85,
                        "mean_reward": -7.4,
                    },
                },
            },
        }

        decoupled = derive_decoupled_prior(
            coupled,
            reward_weight=0.25,
            exploration_weight=1.0,
        )

        verify_reward_prior_preserved(
            coupled,
            decoupled,
            reward_weight=0.25,
            exploration_weight=1.0,
        )

        self.assertEqual(
            decoupled["policy"],
            "UCB1Decoupled",
        )

        self.assertEqual(
            decoupled["config"]["reward_observation_weight"],
            0.25,
        )

        self.assertEqual(
            decoupled["config"]["exploration_observation_weight"],
            1.0,
        )

        for arm in ("amd64", "arm64"):
            coupled_ucb1 = coupled["arms"][arm]["ucb1"]
            decoupled_ucb1 = decoupled["arms"][arm]["ucb1"]

            self.assertEqual(
                coupled_ucb1["reward_sum"],
                decoupled_ucb1["reward_sum"],
            )

            self.assertEqual(
                coupled_ucb1["mean_reward"],
                decoupled_ucb1["mean_reward"],
            )

            self.assertEqual(
                decoupled_ucb1["observation_weight"],
                0.25,
            )

            self.assertEqual(
                decoupled_ucb1["exploration_observation_weight"],
                1.0,
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