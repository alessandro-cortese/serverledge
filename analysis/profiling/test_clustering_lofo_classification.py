import csv
import tempfile
import unittest
from pathlib import Path

from analysis.profiling import preference, preprocess
from analysis.profiling import clustering_lofo_classification as study


class ClusteringLofoClassificationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def create_profiles(self) -> Path:
        path = self.root / "function-profiles-median.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(preprocess.SOURCE_HEADER)
            # 3 compact, separated groups x 4 members = 12 targets.
            groups = [
                ("x", 0.0),
                ("n", 10.0),
                ("a", 20.0),
            ]
            for prefix, base in groups:
                for index in range(4):
                    features = [
                        base + index * 0.03 + 0.1,
                        base + index * 0.04 + 0.2,
                        base + index * 0.05 + 0.3,
                        base + index * 0.06 + 0.4,
                        base + index * 0.07 + 0.5,
                        base + index * 0.08 + 0.6,
                    ]
                    writer.writerow(
                        [
                            preprocess.SOURCE_CSV_SCHEMA_VERSION,
                            "lofo-test",
                            "median",
                            preprocess.FUNCTION_PROFILE_SCHEMA_VERSION,
                            f"{prefix}-{index}",
                            "amd64",
                            1,
                            1024,
                            12,
                            *features,
                        ]
                    )
        return path

    def create_preferences(self) -> Path:
        directory = self.root / "ground_truth"
        directory.mkdir()
        path = directory / "preferences-15.csv"
        items = []
        for prefix, arm_duration in (("x", 130.0), ("n", 100.0), ("a", 70.0)):
            delta = (arm_duration - 100.0) / 100.0 * 100.0
            label = preference.classify_delta(delta, 15.0)
            for index in range(4):
                items.append(
                    {
                        "function_name": f"{prefix}-{index}",
                        "configured_cpus": 1,
                        "configured_memory_mb": 1024,
                        "aggregation": "median",
                        "performance_metric": "duration_ms",
                        "threshold_percent": 15.0,
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
            "pref-test",
            "performance-test",
            "0" * 64,
            items,
        )
        return directory

    def create_languages(self) -> Path:
        path = self.root / "function-languages.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["function_name", "language"])
            writer.writeheader()
            for prefix in ("x", "n", "a"):
                for index in range(4):
                    writer.writerow(
                        {
                            "function_name": f"{prefix}-{index}",
                            "language": "go" if index % 2 == 0 else "python",
                        }
                    )
        return path

    def test_majority_label_marks_tie_ambiguous(self):
        label, counts, purity = study.majority_label(
            [preference.PREFERENCE_X86, preference.PREFERENCE_ARM]
        )
        self.assertEqual(label, study.AMBIGUOUS_LABEL)
        self.assertEqual(counts[preference.PREFERENCE_X86], 1)
        self.assertEqual(counts[preference.PREFERENCE_ARM], 1)
        self.assertEqual(purity, 0.5)

    def test_analysis_runs_all_four_feature_sets_with_language(self):
        output = self.root / "out"
        manifest = study.run_analysis(
            self.create_profiles(),
            self.create_preferences(),
            output,
            threshold=15.0,
            language_map_path=self.create_languages(),
            donor_ranker="manhattan",
            k=3,
            n_init=20,
            random_state=42,
            dbscan_min_samples=2,
            dbscan_eps_quantile=0.80,
            dbscan_metric="cosine",
        )
        self.assertEqual(manifest["target_count"], 12)
        self.assertEqual(len(manifest["feature_sets"]), 4)

        for filename in (
            "lofo-per-target.csv",
            "lofo-classification-summary.csv",
            "lofo-per-class-metrics.csv",
            "lofo-confusion-matrix.csv",
            "full-fit-cluster-summary.csv",
            "full-fit-clustering-summary.csv",
            "clustering-lofo-classification-manifest.json",
        ):
            self.assertTrue((output / filename).is_file(), filename)

        with (output / "lofo-classification-summary.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8)  # 4 feature sets x 2 clusterers
        kmeans_paper5 = next(
            row
            for row in rows
            if row["algorithm"] == "kmeans" and row["feature_set"] == "paper5"
        )
        self.assertGreater(float(kmeans_paper5["prediction_coverage"]), 0.0)
        self.assertGreater(float(kmeans_paper5["strict_accuracy_all_targets"]), 0.0)

        with (output / "lofo-per-target.csv").open(newline="", encoding="utf-8") as handle:
            per_target = list(csv.DictReader(handle))
        predicted = [row for row in per_target if row["prediction_status"] == "predicted"]
        self.assertTrue(predicted)
        # By construction the donor is selected only from the predicted majority class.
        self.assertTrue(
            all(
                row["donor_ground_truth_label"] == row["cluster_majority_label"]
                for row in predicted
            )
        )

    def test_language_feature_sets_are_skipped_without_mapping(self):
        output = self.root / "out-no-language"
        manifest = study.run_analysis(
            self.create_profiles(),
            self.create_preferences(),
            output,
            threshold=15.0,
            language_map_path=None,
            donor_ranker="manhattan",
            k=3,
            n_init=20,
            random_state=42,
            dbscan_min_samples=2,
            dbscan_eps_quantile=0.80,
            dbscan_metric="cosine",
        )
        self.assertEqual(set(manifest["feature_sets"]), {"paper5", "paper5_no_free_memory"})


if __name__ == "__main__":
    unittest.main()
