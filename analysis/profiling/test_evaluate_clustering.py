import csv
import tempfile
import unittest
from pathlib import Path

from analysis.profiling import cluster, evaluate_clustering, preference


class EvaluateClusteringTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def assignments(self, rows, aggregation="mean", algorithm="kmeans"):
        path = self.root / "assignments.csv"

        with path.open("w", newline="", encoding="utf-8") as handle:

            writer = csv.writer(handle)

            writer.writerow(cluster.ASSIGNMENT_HEADER)

            for name, label, machine_tag in rows:

                writer.writerow(
                    [
                        1,
                        1,
                        "cluster-run",
                        1,
                        1,
                        1,
                        "experiment",
                        "fit",
                        aggregation,
                        "standard",
                        algorithm,
                        1,
                        name,
                        machine_tag,
                        1,
                        128,
                        10,
                        label,
                        str(label == -1).lower(),
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                )

        return path

    def preferences(self, rows, aggregation="mean", threshold=2.5):
        path = self.root / "preferences.csv"

        with path.open("w", newline="", encoding="utf-8") as handle:

            writer = csv.writer(handle)

            writer.writerow(preference.ARCHITECTURE_PREFERENCE_HEADER)

            for name, label in rows:

                delta = {
                    preference.PREFERENCE_X86: 10.0,
                    preference.PREFERENCE_ARM: -10.0,
                    preference.PREFERENCE_INDEPENDENT: 0.0,
                }[label]

                writer.writerow(
                    [
                        1,
                        1,
                        "preference-run",
                        "performance-run",
                        "a" * 64,
                        name,
                        1,
                        128,
                        aggregation,
                        "duration_ms",
                        threshold,
                        "x86",
                        "arm64",
                        10,
                        10,
                        100,
                        100 + delta,
                        delta,
                        label,
                    ]
                )

        return path

    def evaluate(self, a_path, p_path, machine_tag="x86"):
        assignments, a_meta = evaluate_clustering.load_assignments(a_path, machine_tag)

        preferences, p_meta = evaluate_clustering.load_preferences(p_path)

        matched, join = evaluate_clustering.join_assignments_preferences(assignments, preferences)

        evaluation, summaries = evaluate_clustering.build_evaluation(
            "evaluation-run", a_path, p_path, a_meta, p_meta, matched, join
        )

        return (evaluation, summaries, matched, join)

    def test_perfect_three_group_mapping_has_perfect_scores(self):
        assignment_rows = []

        preference_rows = []

        groups = [
            (0, preference.PREFERENCE_X86, "x"),
            (1, preference.PREFERENCE_INDEPENDENT, "n"),
            (2, preference.PREFERENCE_ARM, "a"),
        ]

        for cluster_label, pref, prefix in groups:

            for index in range(3):
                name = f"{prefix}-{index}"

                assignment_rows.append((name, cluster_label, "x86"))

                preference_rows.append((name, pref))

        evaluation, summaries, _, _ = self.evaluate(
            self.assignments(assignment_rows), self.preferences(preference_rows)
        )

        external = evaluation["external_metrics_clustered_only"]

        self.assertEqual(len(summaries), 3)

        self.assertEqual(evaluation["cluster_summary"]["overall_purity"], 1.0)

        self.assertTrue(external["defined"])

        for field in (
            "homogeneity",
            "completeness",
            "v_measure",
            "adjusted_rand_index",
            "normalized_mutual_information",
        ):
            self.assertAlmostEqual(external[field], 1.0)

    def test_cluster_label_permutation_does_not_change_scores(self):
        assignments = self.assignments(
            [("x1", 7, "x86"), ("x2", 7, "x86"), ("a1", 3, "x86"), ("a2", 3, "x86")]
        )

        preferences = self.preferences(
            [
                ("x1", preference.PREFERENCE_X86),
                ("x2", preference.PREFERENCE_X86),
                ("a1", preference.PREFERENCE_ARM),
                ("a2", preference.PREFERENCE_ARM),
            ]
        )

        evaluation, _, _, _ = self.evaluate(assignments, preferences)

        external = evaluation["external_metrics_clustered_only"]

        self.assertAlmostEqual(external["adjusted_rand_index"], 1.0)

        self.assertAlmostEqual(external["normalized_mutual_information"], 1.0)

    def test_mixed_clusters_reduce_quality(self):
        assignments = self.assignments(
            [
                ("f1", 0, "x86"),
                ("f2", 0, "x86"),
                ("f3", 0, "x86"),
                ("f4", 1, "x86"),
                ("f5", 1, "x86"),
                ("f6", 1, "x86"),
            ]
        )

        preferences = self.preferences(
            [
                ("f1", preference.PREFERENCE_X86),
                ("f2", preference.PREFERENCE_X86),
                ("f3", preference.PREFERENCE_ARM),
                ("f4", preference.PREFERENCE_ARM),
                ("f5", preference.PREFERENCE_ARM),
                ("f6", preference.PREFERENCE_X86),
            ]
        )

        evaluation, _, _, _ = self.evaluate(assignments, preferences)

        self.assertLess(evaluation["cluster_summary"]["overall_purity"], 1.0)

        self.assertLess(evaluation["external_metrics_clustered_only"]["homogeneity"], 1.0)

    def test_dbscan_noise_is_excluded_and_reduces_coverage(self):
        assignments = self.assignments(
            [
                ("x1", 0, "x86"),
                ("x2", 0, "x86"),
                ("a1", 1, "x86"),
                ("a2", 1, "x86"),
                ("n1", -1, "x86"),
            ],
            algorithm="dbscan",
        )

        preferences = self.preferences(
            [
                ("x1", preference.PREFERENCE_X86),
                ("x2", preference.PREFERENCE_X86),
                ("a1", preference.PREFERENCE_ARM),
                ("a2", preference.PREFERENCE_ARM),
                ("n1", preference.PREFERENCE_INDEPENDENT),
            ]
        )

        evaluation, summaries, _, _ = self.evaluate(assignments, preferences)

        self.assertEqual(len(summaries), 2)

        self.assertEqual(evaluation["cluster_summary"]["noise_count"], 1)

        self.assertAlmostEqual(evaluation["cluster_summary"]["coverage"], 4 / 5)

        self.assertAlmostEqual(
            evaluation["external_metrics_clustered_only"]["adjusted_rand_index"], 1.0
        )

    def test_profile_machine_tag_filter_prevents_double_counting(self):
        assignments = self.assignments(
            [("f1", 0, "x86"), ("f1", 9, "arm64"), ("f2", 1, "x86"), ("f2", 9, "arm64")]
        )

        preferences = self.preferences(
            [("f1", preference.PREFERENCE_X86), ("f2", preference.PREFERENCE_ARM)]
        )

        evaluation, _, matched, _ = self.evaluate(assignments, preferences, "x86")

        self.assertEqual(len(matched), 2)

        self.assertEqual(evaluation["clustering"]["selected_assignment_count"], 2)

        self.assertEqual(evaluation["clustering"]["source_assignment_count"], 4)

    def test_missing_preference_is_reported(self):
        assignments = self.assignments([("f1", 0, "x86"), ("f2", 1, "x86"), ("f3", 1, "x86")])

        preferences = self.preferences(
            [("f1", preference.PREFERENCE_X86), ("f2", preference.PREFERENCE_ARM)]
        )

        _, _, _, join = self.evaluate(assignments, preferences)

        self.assertEqual(join["matched_count"], 2)

        self.assertEqual(join["unmatched_assignment_count"], 1)

        self.assertAlmostEqual(join["assignment_match_coverage"], 2 / 3)

    def test_ambiguous_cluster_majority_is_not_given_semantics(self):
        matched = [
            {
                "cluster_label": 0,
                "is_noise": False,
                "architecture_preference": preference.PREFERENCE_X86,
            },
            {
                "cluster_label": 0,
                "is_noise": False,
                "architecture_preference": preference.PREFERENCE_ARM,
            },
        ]

        summaries, overall = evaluate_clustering.cluster_preference_summaries(matched)

        self.assertEqual(summaries[0]["majority_preference"], "ambiguous")

        self.assertEqual(summaries[0]["cluster_purity"], 0.5)

        self.assertEqual(overall["overall_purity"], 0.5)

    def test_external_metrics_undefined_for_trivial_case(self):
        matched = [
            {
                "cluster_label": 0,
                "is_noise": False,
                "architecture_preference": preference.PREFERENCE_X86,
            },
            {
                "cluster_label": 0,
                "is_noise": False,
                "architecture_preference": preference.PREFERENCE_X86,
            },
        ]

        metrics = evaluate_clustering.external_metrics_if_informative(matched)

        self.assertFalse(metrics["defined"])

        self.assertIsNone(metrics["adjusted_rand_index"])

    def test_aggregation_mismatch_is_rejected(self):
        assignments_path = self.assignments([("f1", 0, "x86")], aggregation="mean")

        preferences_path = self.preferences(
            [("f1", preference.PREFERENCE_X86)], aggregation="median"
        )

        assignments, a_meta = evaluate_clustering.load_assignments(assignments_path, "x86")

        preferences, p_meta = evaluate_clustering.load_preferences(preferences_path)

        matched, join = evaluate_clustering.join_assignments_preferences(assignments, preferences)

        with self.assertRaisesRegex(ValueError, "aggregation does not match"):
            evaluate_clustering.build_evaluation(
                "run", assignments_path, preferences_path, a_meta, p_meta, matched, join
            )

    def test_profile_machine_tag_must_be_ground_truth_architecture(self):
        assignments_path = self.assignments([("f1", 0, "profiling-test")])

        preferences_path = self.preferences([("f1", preference.PREFERENCE_X86)])

        assignments, a_meta = evaluate_clustering.load_assignments(
            assignments_path, "profiling-test"
        )

        preferences, p_meta = evaluate_clustering.load_preferences(preferences_path)

        matched, join = evaluate_clustering.join_assignments_preferences(assignments, preferences)

        with self.assertRaisesRegex(ValueError, "profile machine tag must match"):
            evaluate_clustering.build_evaluation(
                "run", assignments_path, preferences_path, a_meta, p_meta, matched, join
            )


if __name__ == "__main__":
    unittest.main()
