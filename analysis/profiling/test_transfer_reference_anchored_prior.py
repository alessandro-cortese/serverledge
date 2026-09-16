import unittest

from analysis.profiling.transfer_reference_anchored_prior import (
    build_prior_mean_reward,
    first_completed_true_run,
)


class TransferReferenceAnchoredPriorTest(unittest.TestCase):

    def setUp(self):
        self.target = {
            "mean_reward": {
                "x86": -6.0,
                "arm64": -5.7,
            }
        }
        self.donor = {
            "mean_reward": {
                "x86": -4.0,
                "arm64": -3.5,
            }
        }

    def test_raw_prior_uses_donor_absolute_rewards(self):
        prior = build_prior_mean_reward(
            "raw",
            self.target,
            self.donor,
        )

        self.assertEqual(
            prior["x86"],
            -4.0,
        )
        self.assertEqual(
            prior["arm64"],
            -3.5,
        )

    def test_reference_anchored_prior_uses_target_x86(self):
        prior = build_prior_mean_reward(
            "reference-anchored",
            self.target,
            self.donor,
        )

        self.assertEqual(
            prior["x86"],
            -6.0,
        )

    def test_reference_anchored_prior_preserves_donor_arch_effect(self):
        prior = build_prior_mean_reward(
            "reference-anchored",
            self.target,
            self.donor,
        )

        donor_delta = (
            self.donor["mean_reward"]["arm64"]
            - self.donor["mean_reward"]["x86"]
        )
        prior_delta = (
            prior["arm64"]
            - prior["x86"]
        )

        self.assertAlmostEqual(
            prior_delta,
            donor_delta,
        )
        self.assertAlmostEqual(
            prior["arm64"],
            -5.5,
        )

    def test_first_completed_true_run(self):
        self.assertEqual(
            first_completed_true_run(
                [False, True, True, True],
                3,
            ),
            4,
        )

    def test_first_completed_true_run_none(self):
        self.assertIsNone(
            first_completed_true_run(
                [True, False, True, True],
                3,
            )
        )


if __name__ == "__main__":
    unittest.main()
