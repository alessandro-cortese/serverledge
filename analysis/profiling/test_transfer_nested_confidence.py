import unittest

from analysis.profiling.transfer_nested_confidence import (
    calibrate_gate,
    quantile_label,
    safe_ratio,
)


class TransferNestedConfidenceTest(unittest.TestCase):

    def test_safe_ratio_regular(self):
        self.assertAlmostEqual(
            safe_ratio(0.5, 0.25),
            2.0,
        )

    def test_safe_ratio_zero_reference(self):
        self.assertEqual(
            safe_ratio(0.0, 0.0),
            0.0,
        )
        self.assertEqual(
            safe_ratio(1.0, 0.0),
            float("inf"),
        )

    def test_quantile_label(self):
        self.assertEqual(
            quantile_label(0.25),
            "q25",
        )
        self.assertEqual(
            quantile_label(1.0),
            "q100",
        )

    def test_calibration_can_choose_abstain_all(self):
        records = [
            {
                "selection_status": "selected",
                "confidence_score": 0.1,
                "best_arm_correct": False,
            },
            {
                "selection_status": "selected",
                "confidence_score": 0.2,
                "best_arm_correct": False,
            },
            {
                "selection_status": "selected",
                "confidence_score": 0.3,
                "best_arm_correct": True,
            },
            {
                "selection_status": "selected",
                "confidence_score": 0.4,
                "best_arm_correct": False,
            },
        ]

        best, _ = calibrate_gate(
            records,
            quantiles=[0.5, 1.0],
        )

        self.assertEqual(
            best["gate_label"],
            "abstain-all",
        )
        self.assertEqual(
            best["utility"],
            0.0,
        )

    def test_calibration_prefers_positive_net_correctness(self):
        records = [
            {
                "selection_status": "selected",
                "confidence_score": 0.1,
                "best_arm_correct": True,
            },
            {
                "selection_status": "selected",
                "confidence_score": 0.2,
                "best_arm_correct": True,
            },
            {
                "selection_status": "selected",
                "confidence_score": 0.3,
                "best_arm_correct": False,
            },
            {
                "selection_status": "selected",
                "confidence_score": 0.4,
                "best_arm_correct": False,
            },
        ]

        best, candidates = calibrate_gate(
            records,
            quantiles=[0.5, 1.0],
        )

        self.assertEqual(
            best["gate_label"],
            "q50",
        )
        self.assertGreater(
            best["utility"],
            0.0,
        )

        selected = [
            row
            for row in candidates
            if row["selected_gate"]
        ]
        self.assertEqual(
            len(selected),
            1,
        )


if __name__ == "__main__":
    unittest.main()
