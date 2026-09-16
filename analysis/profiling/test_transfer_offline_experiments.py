import unittest

import numpy as np

from analysis.profiling.transfer_donor_lofo import (
    dbscan_eps,
    threshold_slug,
)
from analysis.profiling.transfer_ucb1_offline import (
    UCB1Policy,
    simulate_policy,
)


class TransferOfflineExperimentTest(unittest.TestCase):

    def test_threshold_slug(self):
        self.assertEqual(threshold_slug(2.5), "2p5")
        self.assertEqual(threshold_slug(15.0), "15")

    def test_dbscan_eps_is_finite_positive(self):
        x = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [1.0, 1.0],
                [1.1, 1.0],
                [1.0, 1.1],
            ],
            dtype=float,
        )
        eps = dbscan_eps(
            x,
            min_samples=3,
            eps_quantile=0.8,
            metric="euclidean",
        )
        self.assertTrue(np.isfinite(eps))
        self.assertGreater(eps, 0.0)

    def test_no_transfer_forces_unseen_arms(self):
        policy = UCB1Policy(c=0.8)
        self.assertEqual(policy.select_arm(), "x86")
        policy.update("x86", 10.0)
        self.assertEqual(policy.select_arm(), "arm64")

    def test_prior_selects_better_donor_arm_before_real_feedback(self):
        policy = UCB1Policy(
            c=0.8,
            prior_weight=0.5,
            prior_mean_reward={
                "x86": -2.0,
                "arm64": -1.0,
            },
        )
        self.assertEqual(policy.effective_total(), 1.0)
        self.assertEqual(policy.select_arm(), "arm64")
        self.assertAlmostEqual(policy.score("x86"), -2.0)
        self.assertAlmostEqual(policy.score("arm64"), -1.0)

    def test_prior_weight_one_is_half_after_first_observation(self):
        policy = UCB1Policy(
            c=0.0,
            prior_weight=1.0,
            prior_mean_reward={
                "x86": -2.0,
                "arm64": -3.0,
            },
        )
        reward = policy.update("x86", np.exp(1.0))
        expected = (-2.0 + reward) / 2.0
        effective = (
            policy.reward_sums["x86"]
            + policy.prior_reward_sums["x86"]
        ) / policy.effective_count("x86")
        self.assertAlmostEqual(effective, expected)

    def test_simulation_returns_expected_horizon(self):
        target = {
            "mean_reward": {"x86": -1.0, "arm64": -2.0},
            "mean_duration": {"x86": 3.0, "arm64": 7.0},
            "best_reward_arm": "x86",
            "best_latency_arm": "x86",
        }
        sequences = {
            "x86": np.asarray([3.0] * 5),
            "arm64": np.asarray([7.0] * 5),
        }
        result = simulate_policy(
            target,
            sequences,
            horizon=5,
            c=0.8,
            prior_weight=0.0,
            donor_stats=None,
        )
        self.assertEqual(len(result["chosen_arms"]), 5)
        self.assertEqual(len(result["cumulative_latency"]), 5)


if __name__ == "__main__":
    unittest.main()
