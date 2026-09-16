import unittest

import numpy as np

from analysis.profiling.transfer_prior_diagnostics import (
    prior_mean_for_strategy,
)
from analysis.profiling.transfer_ucb1_offline import (
    UCB1Policy,
)


class TransferPriorDiagnosticsTest(unittest.TestCase):

    def setUp(self):
        self.target_stats = {
            "mean_reward": {
                "x86": -1.0,
                "arm64": -2.0,
            },
            "best_reward_arm": "x86",
        }
        self.performance = {
            "donor": {
                "mean_reward": {
                    "x86": -1.5,
                    "arm64": -2.5,
                }
            }
        }

    def test_oracle_prior_equals_target_means(self):
        prior, label, applied = prior_mean_for_strategy(
            "oracle",
            self.target_stats,
            None,
            self.performance,
        )
        self.assertTrue(applied)
        self.assertEqual(label, "TARGET_ORACLE")
        self.assertEqual(prior, self.target_stats["mean_reward"])

    def test_wrong_prior_swaps_target_means(self):
        prior, label, applied = prior_mean_for_strategy(
            "wrong",
            self.target_stats,
            None,
            self.performance,
        )
        self.assertTrue(applied)
        self.assertEqual(label, "TARGET_REVERSED")
        self.assertEqual(prior["x86"], -2.0)
        self.assertEqual(prior["arm64"], -1.0)

    def test_selected_dbscan_abstention_becomes_no_transfer(self):
        prior, label, applied = prior_mean_for_strategy(
            "selected-dbscan",
            self.target_stats,
            {
                "selection_status": "no-transfer",
                "donor_function": "",
            },
            self.performance,
        )
        self.assertFalse(applied)
        self.assertEqual(label, "NO_TRANSFER")
        self.assertIsNone(prior)

    def test_selected_kmeans_uses_donor_means(self):
        prior, label, applied = prior_mean_for_strategy(
            "selected-kmeans",
            self.target_stats,
            {
                "selection_status": "selected",
                "donor_function": "donor",
            },
            self.performance,
        )
        self.assertTrue(applied)
        self.assertEqual(label, "donor")
        self.assertEqual(
            prior,
            self.performance["donor"]["mean_reward"],
        )

    def test_oracle_prior_selects_true_best_arm_before_feedback(self):
        prior, _, _ = prior_mean_for_strategy(
            "oracle",
            self.target_stats,
            None,
            self.performance,
        )
        policy = UCB1Policy(
            c=0.8,
            prior_weight=0.25,
            prior_mean_reward=prior,
        )
        self.assertEqual(policy.select_arm(), "x86")

    def test_wrong_prior_selects_wrong_arm_before_feedback(self):
        prior, _, _ = prior_mean_for_strategy(
            "wrong",
            self.target_stats,
            None,
            self.performance,
        )
        policy = UCB1Policy(
            c=0.8,
            prior_weight=0.25,
            prior_mean_reward=prior,
        )
        self.assertEqual(policy.select_arm(), "arm64")


if __name__ == "__main__":
    unittest.main()
