import csv
import tempfile
import unittest
from pathlib import Path

from analysis.profiling import clustering_study, preference, preprocess


class ClusteringStudyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def create_profiles(self):
        path = self.root / "function-profiles-median.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(preprocess.SOURCE_HEADER)

            groups = [
                ("x", 0.0),
                ("n", 10.0),
                ("a", 20.0),
            ]
            for prefix, base in groups:
                for index in range(3):
                    # Slightly different dimensions avoid all-constant columns
                    # while retaining three well-separated groups.
                    values = [
                        base + index * 0.10,
                        base + index * 0.11 + 0.2,
                        base + index * 0.12 + 0.4,
                        base + index * 0.13 + 0.6,
                        base + index * 0.14 + 0.8,
                        base + index * 0.15 + 1.0,
                    ]
                    writer.writerow(
                        [
                            preprocess.SOURCE_CSV_SCHEMA_VERSION,
                            "study-test",
                            "median",
                            preprocess.FUNCTION_PROFILE_SCHEMA_VERSION,
                            f"{prefix}-{index}",
                            "amd64",
                            1,
                            1024,
                            12,
                            *values,
                        ]
                    )
        return path

    def create_preferences(self, threshold):
        path = self.root / f"preferences-{threshold}.csv"
        items = []
        for prefix, arm_duration in (("x", 130.0), ("n", 100.0), ("a", 70.0)):
            delta = arm_duration - 100.0
            label = preference.classify_delta(delta, threshold)
            for index in range(3):
                items.append(
                    {
                        "function_name": f"{prefix}-{index}",
                        "configured_cpus": 1,
                        "configured_memory_mb": 1024,
                        "aggregation": "median",
                        "performance_metric": "duration_ms",
                        "threshold_percent": threshold,
                        "x86_machine_tag": "amd64",
                        "arm_machine_tag": "arm64",
                        "x86_sample_count": 12,
                        "arm_sample_count": 12,
                        "x86_duration_ms": 100.0,
                        "arm_duration_ms": arm_duration,
                        "arm_vs_x86_delta_percent": delta,
                        "architecture_preference": label,
                    }
                )
        preference.write_architecture_preferences(
            path,
            f"pref-{threshold}",
            "performance-test",
            "0" * 64,
            items,
        )
        return path

    def test_feature_sets_are_paper6_lofo_plus_combined_reduced_candidates(self):
        self.assertEqual(clustering_study.FEATURE_SETS["paper6"], preprocess.FEATURE_NAMES)
        self.assertEqual(len(clustering_study.LOFO_FEATURE_SETS), 6)
        self.assertEqual(len(clustering_study.COMBINED_REDUCED_FEATURE_SETS), 2)
        self.assertEqual(len(clustering_study.FEATURE_SETS), 9)

        for feature in preprocess.FEATURE_NAMES:
            key = f"paper6_no_{feature}"
            self.assertIn(key, clustering_study.FEATURE_SETS)
            self.assertNotIn(feature, clustering_study.FEATURE_SETS[key])
            self.assertEqual(len(clustering_study.FEATURE_SETS[key]), 5)

        self.assertEqual(
            clustering_study.FEATURE_SETS[
                "paper4_no_framework_runtime_ms_no_page_faults_delta"
            ],
            [
                "utilized_cpus",
                "free_memory_mb",
                "cpu_user_delta_ms",
                "cpu_kernel_delta_ms",
            ],
        )
        self.assertEqual(
            clustering_study.FEATURE_SETS[
                "paper4_no_framework_runtime_ms_no_cpu_kernel_delta_ms"
            ],
            [
                "page_faults_delta",
                "utilized_cpus",
                "free_memory_mb",
                "cpu_user_delta_ms",
            ],
        )

    def test_threshold_sweep_reuses_each_cluster_configuration(self):
        output = self.root / "study"
        manifest = clustering_study.run_study(
            self.create_profiles(),
            [self.create_preferences(2.5), self.create_preferences(15.0)],
            output,
            "synthetic-study",
            "standard",
            [3],
            3,
            15.0,
            10,
            42,
            False,
        )

        self.assertEqual(manifest["function_count"], 9)
        self.assertEqual(manifest["ground_truth_thresholds_percent"], [2.5, 15.0])

        with (output / "kmeans-internal-summary.csv").open(newline="", encoding="utf-8") as handle:
            internal_rows = list(csv.DictReader(handle))
        with (output / "kmeans-ground-truth-summary.csv").open(newline="", encoding="utf-8") as handle:
            external_rows = list(csv.DictReader(handle))

        # 9 feature sets x 1 K are fitted once.
        self.assertEqual(len(internal_rows), 9)
        # The same 9 assignments are then evaluated at both thresholds.
        self.assertEqual(len(external_rows), 18)

        paper6 = [row for row in external_rows if row["feature_set"] == "paper6"]
        self.assertEqual({float(row["threshold_percent"]) for row in paper6}, {2.5, 15.0})
        self.assertTrue(all(float(row["overall_purity"]) == 1.0 for row in paper6))

        self.assertTrue((output / "ablation-deltas.csv").is_file())
        self.assertTrue((output / "feature-correlation-spearman.csv").is_file())
        self.assertTrue((output / "pca-coordinates-primary.csv").is_file())
        self.assertTrue((output / "feature-summary.csv").is_file())
        self.assertTrue((output / "feature-outlier-zscores.csv").is_file())
        self.assertTrue((output / "CLUSTERING_ANALYSIS.md").is_file())
        self.assertTrue((output / "clustering-study-manifest.json").is_file())

        with (output / "kmeans-ground-truth-summary.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            row = next(csv.DictReader(handle))
        self.assertIn("majority_ground_truth_share", row)
        self.assertIn("purity_gain_over_majority_baseline", row)


if __name__ == "__main__":
    unittest.main()
