import unittest

from analysis.profiling.transfer_donor_confidence import (
    build_selected_donor_rows,
    compute_gate_thresholds,
    retained_targets,
)


class TransferDonorConfidenceTest(unittest.TestCase):

    def test_only_selected_donors_enter_distance_analysis(self):
        rows = [
            {
                "algorithm": "kmeans",
                "target_function": "a",
                "donor_function": "d1",
                "selection_status": "selected",
                "donor_distance": "0.1",
                "best_reward_arm_agreement": "True",
                "abs_architecture_delta_error_percent": "1.0",
                "abs_reward_gap_error": "0.1",
                "target_ground_truth_label": "x86-preferred",
                "donor_ground_truth_label": "x86-preferred",
            },
            {
                "algorithm": "dbscan",
                "target_function": "b",
                "donor_function": "",
                "selection_status": "no-transfer",
                "donor_distance": "",
                "best_reward_arm_agreement": "",
                "abs_architecture_delta_error_percent": "",
                "abs_reward_gap_error": "",
                "target_ground_truth_label": "arm-preferred",
                "donor_ground_truth_label": "",
            },
        ]

        selected = build_selected_donor_rows(rows)

        self.assertIn("kmeans", selected)
        self.assertNotIn("dbscan", selected)
        self.assertEqual(len(selected["kmeans"]), 1)

    def test_full_quantile_retains_all_selected_targets(self):
        selected = [
            {
                "target_function": "a",
                "donor_distance": 0.1,
            },
            {
                "target_function": "b",
                "donor_distance": 0.2,
            },
            {
                "target_function": "c",
                "donor_distance": 0.3,
            },
        ]

        gates = compute_gate_thresholds(selected)
        full = [g for g in gates if g["gate_quantile"] == 1.0][0]

        retained = retained_targets(
            selected,
            full["distance_threshold"],
        )

        self.assertEqual(retained, {"a", "b", "c"})
        self.assertEqual(full["retained_selected_count"], 3)

    def test_half_quantile_rejects_larger_distances(self):
        selected = [
            {
                "target_function": "a",
                "donor_distance": 0.1,
            },
            {
                "target_function": "b",
                "donor_distance": 0.2,
            },
            {
                "target_function": "c",
                "donor_distance": 0.3,
            },
            {
                "target_function": "d",
                "donor_distance": 0.4,
            },
        ]

        gates = compute_gate_thresholds(selected)
        half = [g for g in gates if g["gate_quantile"] == 0.5][0]

        retained = retained_targets(
            selected,
            half["distance_threshold"],
        )

        self.assertEqual(retained, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
